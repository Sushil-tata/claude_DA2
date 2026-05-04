"""
Legal Queue Manager
====================
Manages the legal action queue for collections.

Design:
  - Works alongside OfferGenerator — accounts where LEGAL path wins NPV
    are routed here for queue management, not to the TDR agent screen.
  - Queue prioritisation: NPV-weighted, adjusted for asset confidence
    and account age (older accounts = lower recovery probability).
  - Case assignment: round-robin with workload cap, configurable.
  - Status lifecycle: QUEUED → UNDER_REVIEW → FILED → ACTIVE → CLOSED
  - Escalation rules: auto-escalate stale cases.
  - Export-ready: Delta-compatible DataFrame output.

Fields used (confirmed available from data_contract.py Q1-D):
  REQUIRED: account_id, balance, stage, days_past_due
  STANDARD: last_contact_date, last_contact_outcome, ptp_made, ptp_kept
  BUREAU:   bureau_secured_loan_flag, bureau_secured_outstanding,
            bureau_total_outstanding, bureau_delinquent_other
  DERIVED:  legal_npv (from NPVEngine), willingness_score (from DataContract)

No field is assumed — engine degrades gracefully with lower priority score
when bureau data is unavailable.
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .data_contract import DataContract, _is_null
from .npv_engine import NPVEngine, NPVConfig

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# ENUMS & CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

class CaseStatus(str, Enum):
    QUEUED        = "QUEUED"         # Scored and waiting assignment
    UNDER_REVIEW  = "UNDER_REVIEW"   # Assigned to legal officer
    FILED         = "FILED"          # Court filing submitted
    ACTIVE        = "ACTIVE"         # Active litigation
    SETTLED       = "SETTLED"        # Settled pre-judgment
    CLOSED_WON    = "CLOSED_WON"     # Judgment obtained
    CLOSED_LOST   = "CLOSED_LOST"    # Judgment against / case dropped
    ON_HOLD       = "ON_HOLD"        # Awaiting information or BOT instruction

class LegalAction(str, Enum):
    LETTER_BEFORE_ACTION = "LETTER_BEFORE_ACTION"   # Pre-litigation demand letter
    SMALL_CLAIMS         = "SMALL_CLAIMS"            # Fast-track court < threshold
    CIVIL_LITIGATION     = "CIVIL_LITIGATION"        # Standard civil court
    ENFORCEMENT          = "ENFORCEMENT"             # Post-judgment enforcement
    BANKRUPTCY_WATCH     = "BANKRUPTCY_WATCH"        # Monitor bankruptcy proceedings
    WRITE_OFF_RECOMMEND  = "WRITE_OFF_RECOMMEND"     # No viable legal path — write off

class EscalationReason(str, Enum):
    STALE_QUEUED       = "STALE_QUEUED"        # No action after N days in QUEUED
    STALE_UNDER_REVIEW = "STALE_UNDER_REVIEW"  # No update after N days in review
    HIGH_VALUE         = "HIGH_VALUE"          # Balance > threshold, auto-escalate
    SECURED_ASSET      = "SECURED_ASSET"       # Secured asset detected — priority
    PTP_BROKEN         = "PTP_BROKEN"          # Customer broke promise again


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LegalConfig:
    """
    All parameters configurable — load from YAML in production.
    """

    # ── Queue thresholds ──────────────────────────────────────────────────────
    min_balance_for_legal: float = 10_000.0         # Below this → write-off path
    small_claims_threshold: float = 100_000.0        # Fast-track limit (Thai court)
    priority_balance_threshold: float = 500_000.0    # Auto-escalate above this

    # ── Eligibility rules ─────────────────────────────────────────────────────
    min_dpd_for_legal: int = 90                      # Must be 90+ DPD for queue
    stages_eligible: List[str] = field(default_factory=lambda: ["NPL", "CHARGEOFF"])

    # ── Scoring weights (sum = 1.0) ───────────────────────────────────────────
    # Weights for priority score: higher score = more urgent to file
    weight_legal_npv: float      = 0.40    # NPV of legal path (normalised)
    weight_balance: float        = 0.25    # Raw outstanding (normalised)
    weight_asset_flag: float     = 0.20    # Has secured asset (binary × weight)
    weight_dpd_age: float        = 0.10    # Higher DPD = higher urgency (normalised)
    weight_willingness: float    = 0.05    # Lower willingness → less likely to settle voluntarily

    # ── Workload caps ─────────────────────────────────────────────────────────
    max_cases_per_officer: int = 50        # Max active cases per legal officer
    team_officers: List[str] = field(default_factory=lambda: [
        "officer_01", "officer_02", "officer_03"
    ])

    # ── Escalation timers (days) ──────────────────────────────────────────────
    stale_queued_days: int       = 7       # Escalate if QUEUED > 7d with no action
    stale_review_days: int       = 14     # Escalate if UNDER_REVIEW > 14d with no update
    high_priority_review_sla: int = 3     # High-value cases: review SLA 3 days

    # ── Action routing by balance + asset ─────────────────────────────────────
    # (has_asset, balance_band) → recommended first action
    # balance_band: 0=below small_claims, 1=small_claims to priority, 2=above priority
    action_routing: Dict = field(default_factory=lambda: {
        (False, 0): LegalAction.SMALL_CLAIMS,
        (False, 1): LegalAction.CIVIL_LITIGATION,
        (False, 2): LegalAction.CIVIL_LITIGATION,
        (True,  0): LegalAction.SMALL_CLAIMS,
        (True,  1): LegalAction.CIVIL_LITIGATION,
        (True,  2): LegalAction.ENFORCEMENT,     # Secured asset → go straight to enforcement
    })


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT DATACLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LegalCase:
    """One account's legal case record."""
    # Identity
    case_id: str
    account_id: str
    created_at: str

    # Account snapshot
    balance: float
    stage: str
    days_past_due: int
    has_secured_asset: bool
    secured_outstanding: float

    # Scores
    priority_score: float           # 0-100, higher = more urgent
    legal_npv: float
    willingness_score: float
    data_confidence: str            # HIGH | MEDIUM | LOW

    # Routing
    recommended_action: LegalAction
    balance_band: int               # 0=small_claims, 1=civil, 2=priority
    eligible: bool
    ineligibility_reason: Optional[str]

    # Workflow
    status: CaseStatus
    assigned_to: Optional[str]
    assigned_at: Optional[str]
    last_updated: str

    # Escalation
    escalation_flags: List[str]
    escalation_due_date: Optional[str]

    # Notes
    notes: List[str]


@dataclass
class QueueSummary:
    """Batch queue run summary."""
    run_id: str
    run_at: str
    total_accounts: int
    eligible_count: int
    ineligible_count: int
    queued_new: int
    escalated: int
    by_action: Dict[str, int]
    by_stage: Dict[str, int]
    total_balance_queued: float
    avg_priority_score: float


# ─────────────────────────────────────────────────────────────────────────────
# LEGAL QUEUE MANAGER
# ─────────────────────────────────────────────────────────────────────────────

class LegalQueueManager:
    """
    Scores, queues, and manages accounts recommended for legal action.

    Usage:
        mgr = LegalQueueManager()
        cases_df, summary = mgr.process_batch(accounts_df)
        # cases_df is Delta-ready with one row per account
    """

    def __init__(
        self,
        config: Optional[LegalConfig] = None,
        npv_config: Optional[NPVConfig] = None,
    ):
        self.cfg       = config    or LegalConfig()
        self.npv_engine = NPVEngine(npv_config or NPVConfig())
        self.contract   = DataContract()

    # ── PUBLIC ────────────────────────────────────────────────────────────────

    def process_batch(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, QueueSummary]:
        """
        Score and queue a batch of accounts.

        Returns:
            cases_df  : DataFrame with one LegalCase per row (Delta-ready)
            summary   : QueueSummary aggregate stats
        """
        run_id = str(uuid.uuid4())[:8]
        cases  = []

        for _, row in df.iterrows():
            case = self._process_one(row)
            cases.append(case)

        cases_df = self._to_dataframe(cases)
        summary  = self._summarise(cases, run_id)

        logger.info(
            "Legal queue run %s: %d eligible / %d total, "
            "total balance %.0f",
            run_id, summary.eligible_count,
            summary.total_accounts, summary.total_balance_queued,
        )

        return cases_df, summary

    def process_one(self, account: pd.Series) -> LegalCase:
        """Score and create a LegalCase for a single account."""
        return self._process_one(account)

    def assign_cases(
        self,
        cases_df: pd.DataFrame,
        current_load: Optional[Dict[str, int]] = None,
    ) -> pd.DataFrame:
        """
        Round-robin assign queued cases to officers respecting max_cases_per_officer.

        Args:
            cases_df     : DataFrame from process_batch (or subset)
            current_load : {officer_id: current_case_count} — if None, assume 0

        Returns:
            cases_df with assigned_to and assigned_at filled for QUEUED rows.
        """
        load = {o: 0 for o in self.cfg.team_officers}
        if current_load:
            load.update(current_load)

        df = cases_df.copy()
        queued_mask = (df["status"] == CaseStatus.QUEUED.value) & df["eligible"]

        # Sort by priority descending — assign highest priority first
        queued_idx = df[queued_mask].sort_values("priority_score", ascending=False).index

        officer_cycle = list(self.cfg.team_officers)
        officer_pos   = 0

        for idx in queued_idx:
            # Find next officer with capacity
            assigned = False
            for _ in range(len(officer_cycle)):
                officer = officer_cycle[officer_pos % len(officer_cycle)]
                officer_pos += 1
                if load[officer] < self.cfg.max_cases_per_officer:
                    df.at[idx, "assigned_to"]  = officer
                    df.at[idx, "assigned_at"]  = datetime.now().isoformat()
                    df.at[idx, "status"]        = CaseStatus.UNDER_REVIEW.value
                    df.at[idx, "last_updated"]  = datetime.now().isoformat()
                    load[officer] += 1
                    assigned = True
                    break

            if not assigned:
                logger.warning("No officer capacity for case %s", df.at[idx, "case_id"])

        return df

    def check_escalations(self, cases_df: pd.DataFrame) -> pd.DataFrame:
        """
        Scan existing queue for cases that breach SLA or escalation rules.
        Sets escalation_flags and escalation_due_date.

        Call this daily as part of the queue maintenance job.
        """
        df   = cases_df.copy()
        now  = datetime.now()

        for idx, row in df.iterrows():
            flags = []

            # Parse timestamps
            created  = _parse_dt(row.get("created_at"))
            updated  = _parse_dt(row.get("last_updated"))
            status   = row.get("status", "")

            # Stale QUEUED
            if status == CaseStatus.QUEUED.value and created:
                age_days = (now - created).days
                if age_days >= self.cfg.stale_queued_days:
                    flags.append(EscalationReason.STALE_QUEUED.value)

            # Stale UNDER_REVIEW
            if status == CaseStatus.UNDER_REVIEW.value and updated:
                stale_days = (now - updated).days
                threshold  = (
                    self.cfg.high_priority_review_sla
                    if float(row.get("priority_score", 0)) >= 75
                    else self.cfg.stale_review_days
                )
                if stale_days >= threshold:
                    flags.append(EscalationReason.STALE_UNDER_REVIEW.value)

            # High-value auto-escalate
            bal = float(row.get("balance", 0) or 0)
            if bal >= self.cfg.priority_balance_threshold:
                if status in (CaseStatus.QUEUED.value, CaseStatus.UNDER_REVIEW.value):
                    flags.append(EscalationReason.HIGH_VALUE.value)

            # Secured asset
            if row.get("has_secured_asset") and status == CaseStatus.QUEUED.value:
                flags.append(EscalationReason.SECURED_ASSET.value)

            if flags:
                existing = row.get("escalation_flags") or ""
                all_flags = list(set(
                    existing.split("|") + flags
                )) if existing else flags
                df.at[idx, "escalation_flags"] = "|".join(f for f in all_flags if f)
                df.at[idx, "escalation_due_date"] = (
                    now + timedelta(days=1)
                ).date().isoformat()

        return df

    def queue_report(self, cases_df: pd.DataFrame) -> str:
        """
        Plain-text report of current queue state.
        Suitable for daily management email or dashboard widget.
        """
        df = cases_df[cases_df["eligible"] == True].copy()

        lines = [
            "=" * 60,
            "  LEGAL QUEUE REPORT",
            f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "=" * 60,
            "",
        ]

        # By status
        status_counts = df["status"].value_counts().to_dict()
        lines.append("STATUS BREAKDOWN:")
        for s, n in status_counts.items():
            lines.append(f"  {s:<20} {n:>5}")

        # By action type
        lines.append("")
        lines.append("RECOMMENDED ACTION:")
        action_counts = df["recommended_action"].value_counts().to_dict()
        for a, n in action_counts.items():
            lines.append(f"  {a:<30} {n:>5}")

        # Escalations
        esc_df = df[df["escalation_flags"].notna() & (df["escalation_flags"] != "")]
        lines.append("")
        lines.append(f"ESCALATIONS PENDING: {len(esc_df)}")
        if not esc_df.empty:
            for _, r in esc_df.head(10).iterrows():
                lines.append(
                    f"  {r['account_id']:<15} {r['status']:<20} "
                    f"bal={float(r['balance']):>10,.0f}  "
                    f"flags={r['escalation_flags']}"
                )

        # Top 10 by priority
        lines.append("")
        lines.append("TOP 10 BY PRIORITY SCORE:")
        top = df[df["status"].isin([
            CaseStatus.QUEUED.value, CaseStatus.UNDER_REVIEW.value
        ])].nlargest(10, "priority_score")
        for _, r in top.iterrows():
            lines.append(
                f"  {r['account_id']:<15} score={float(r['priority_score']):>5.1f}  "
                f"bal={float(r['balance']):>10,.0f}  "
                f"action={r['recommended_action']:<25} "
                f"officer={r.get('assigned_to') or 'UNASSIGNED'}"
            )

        lines.append("")
        lines.append(f"Total balance in queue: {df['balance'].sum():,.0f}")
        lines.append("=" * 60)

        return "\n".join(lines)

    # ── PRIVATE ───────────────────────────────────────────────────────────────

    def _process_one(self, account: pd.Series) -> LegalCase:
        c   = self.cfg
        aid = str(account.get("account_id", "unknown"))
        now = datetime.now().isoformat()

        # Enrich with derived fields
        try:
            enriched = self.contract.enrich(account)
            quality  = self.contract.validate(enriched)
        except ValueError as e:
            # Missing required fields — create minimal ineligible case
            return LegalCase(
                case_id=str(uuid.uuid4())[:8],
                account_id=aid,
                created_at=now,
                balance=float(account.get("balance", 0) or 0),
                stage=str(account.get("stage", "UNKNOWN")),
                days_past_due=int(account.get("days_past_due", 0) or 0),
                has_secured_asset=False,
                secured_outstanding=0.0,
                priority_score=0.0,
                legal_npv=0.0,
                willingness_score=0.0,
                data_confidence="INVALID",
                recommended_action=LegalAction.WRITE_OFF_RECOMMEND,
                balance_band=0,
                eligible=False,
                ineligibility_reason=str(e),
                status=CaseStatus.ON_HOLD,
                assigned_to=None,
                assigned_at=None,
                last_updated=now,
                escalation_flags=[],
                escalation_due_date=None,
                notes=[f"Validation error: {e}"],
            )

        balance = float(enriched.get("balance", 0) or 0)
        stage   = str(enriched.get("stage", "NPL"))
        dpd     = int(enriched.get("days_past_due", 0) or 0)

        # ── Eligibility check ─────────────────────────────────────────────────
        eligible, reason = self._check_eligibility(balance, stage, dpd)

        # ── Asset detection ───────────────────────────────────────────────────
        has_asset = bool(
            not _is_null(enriched.get("bureau_secured_loan_flag"))
            and enriched.get("bureau_secured_loan_flag")
        )
        secured_out = float(enriched.get("bureau_secured_outstanding", 0) or 0)

        # ── NPV of legal path ─────────────────────────────────────────────────
        try:
            path = self.npv_engine.compare_paths(enriched)
            legal_npv = path.legal_npv
        except Exception:
            legal_npv = 0.0

        # ── Willingness score (from DataContract enrich) ──────────────────────
        willingness = float(enriched.get("willingness_score", 0.0) or 0.0)

        # ── Priority score 0-100 ──────────────────────────────────────────────
        priority = self._compute_priority(
            balance, legal_npv, has_asset, dpd, willingness, eligible
        )

        # ── Balance band ──────────────────────────────────────────────────────
        balance_band = self._balance_band(balance)

        # ── Recommended action ────────────────────────────────────────────────
        if not eligible:
            action = LegalAction.WRITE_OFF_RECOMMEND
        elif balance < c.min_balance_for_legal:
            action = LegalAction.WRITE_OFF_RECOMMEND
        else:
            action = c.action_routing.get(
                (has_asset, balance_band),
                LegalAction.CIVIL_LITIGATION,
            )

        # ── Notes ─────────────────────────────────────────────────────────────
        notes = list(quality.warnings)
        if not eligible:
            notes.append(f"Ineligible: {reason}")
        if has_asset:
            notes.append(f"Secured asset detected — outstanding {secured_out:,.0f}")
        if quality.confidence_level == "LOW":
            notes.append("LOW data confidence — priority score less reliable")

        return LegalCase(
            case_id=str(uuid.uuid4())[:8],
            account_id=aid,
            created_at=now,
            balance=round(balance, 2),
            stage=stage,
            days_past_due=dpd,
            has_secured_asset=has_asset,
            secured_outstanding=round(secured_out, 2),
            priority_score=round(priority, 1),
            legal_npv=round(legal_npv, 2),
            willingness_score=round(willingness, 4),
            data_confidence=quality.confidence_level,
            recommended_action=action,
            balance_band=balance_band,
            eligible=eligible,
            ineligibility_reason=reason,
            status=CaseStatus.QUEUED if eligible else CaseStatus.ON_HOLD,
            assigned_to=None,
            assigned_at=None,
            last_updated=now,
            escalation_flags=[],
            escalation_due_date=None,
            notes=notes,
        )

    def _check_eligibility(
        self, balance: float, stage: str, dpd: int
    ) -> Tuple[bool, Optional[str]]:
        c = self.cfg
        if stage not in c.stages_eligible:
            return False, f"Stage {stage} not in eligible stages {c.stages_eligible}"
        if dpd < c.min_dpd_for_legal:
            return False, f"DPD {dpd} < minimum {c.min_dpd_for_legal}"
        if balance < c.min_balance_for_legal:
            return False, f"Balance {balance:,.0f} < minimum {c.min_balance_for_legal:,.0f}"
        return True, None

    def _balance_band(self, balance: float) -> int:
        if balance < self.cfg.small_claims_threshold:
            return 0
        if balance < self.cfg.priority_balance_threshold:
            return 1
        return 2

    def _compute_priority(
        self,
        balance: float,
        legal_npv: float,
        has_asset: bool,
        dpd: int,
        willingness: float,
        eligible: bool,
    ) -> float:
        if not eligible:
            return 0.0

        c = self.cfg

        # Normalise balance (0-1) using log scale to handle large range
        bal_norm = min(np.log1p(balance) / np.log1p(10_000_000), 1.0)

        # Normalise legal NPV
        npv_norm = min(max(legal_npv, 0) / max(balance, 1), 1.0)

        # Asset flag (0 or 1)
        asset_flag = 1.0 if has_asset else 0.0

        # DPD normalised — 90d = 0, 1000d+ = 1
        dpd_norm = min(max(dpd - 90, 0) / 910, 1.0)

        # Lower willingness → more likely to need legal (invert)
        recalcitrance = 1.0 - willingness

        score = (
            c.weight_legal_npv   * npv_norm
          + c.weight_balance     * bal_norm
          + c.weight_asset_flag  * asset_flag
          + c.weight_dpd_age     * dpd_norm
          + c.weight_willingness * recalcitrance
        )

        return round(score * 100, 1)

    def _to_dataframe(self, cases: List[LegalCase]) -> pd.DataFrame:
        rows = []
        for lc in cases:
            rows.append({
                "case_id":               lc.case_id,
                "account_id":            lc.account_id,
                "created_at":            lc.created_at,
                "balance":               lc.balance,
                "stage":                 lc.stage,
                "days_past_due":         lc.days_past_due,
                "has_secured_asset":     lc.has_secured_asset,
                "secured_outstanding":   lc.secured_outstanding,
                "priority_score":        lc.priority_score,
                "legal_npv":             lc.legal_npv,
                "willingness_score":     lc.willingness_score,
                "data_confidence":       lc.data_confidence,
                "recommended_action":    lc.recommended_action.value if isinstance(lc.recommended_action, LegalAction) else lc.recommended_action,
                "balance_band":          lc.balance_band,
                "eligible":              lc.eligible,
                "ineligibility_reason":  lc.ineligibility_reason or "",
                "status":                lc.status.value if isinstance(lc.status, CaseStatus) else lc.status,
                "assigned_to":           lc.assigned_to or "",
                "assigned_at":           lc.assigned_at or "",
                "last_updated":          lc.last_updated,
                "escalation_flags":      "|".join(lc.escalation_flags),
                "escalation_due_date":   lc.escalation_due_date or "",
                "notes":                 " | ".join(lc.notes),
            })
        return pd.DataFrame(rows)

    def _summarise(self, cases: List[LegalCase], run_id: str) -> QueueSummary:
        eligible  = [c for c in cases if c.eligible]
        new_queued = [c for c in eligible if c.status == CaseStatus.QUEUED]

        action_counts: Dict[str, int] = {}
        for c in eligible:
            a = c.recommended_action.value if isinstance(c.recommended_action, LegalAction) else c.recommended_action
            action_counts[a] = action_counts.get(a, 0) + 1

        stage_counts: Dict[str, int] = {}
        for c in eligible:
            stage_counts[c.stage] = stage_counts.get(c.stage, 0) + 1

        return QueueSummary(
            run_id=run_id,
            run_at=datetime.now().isoformat(),
            total_accounts=len(cases),
            eligible_count=len(eligible),
            ineligible_count=len(cases) - len(eligible),
            queued_new=len(new_queued),
            escalated=0,     # Filled by check_escalations() call
            by_action=action_counts,
            by_stage=stage_counts,
            total_balance_queued=round(sum(c.balance for c in eligible), 2),
            avg_priority_score=round(
                np.mean([c.priority_score for c in eligible]) if eligible else 0.0, 1
            ),
        )


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _parse_dt(val) -> Optional[datetime]:
    if not val or (isinstance(val, float) and np.isnan(val)):
        return None
    try:
        return datetime.fromisoformat(str(val))
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SMOKE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    accounts = pd.DataFrame([
        {
            # High-value NPL with secured asset — expect CIVIL_LITIGATION, high priority
            "account_id": "ACC001", "balance": 800_000, "stage": "NPL",
            "days_past_due": 180, "bureau_secured_loan_flag": True,
            "bureau_secured_outstanding": 2_000_000,
            "bureau_monthly_instalment": 25_000, "bureau_total_outstanding": 1_200_000,
            "last_payment_amount": 5_000, "payment_count_12m": 3,
            "total_paid_12m": 15_000, "ptp_made": 4, "ptp_kept": 1,
            "outbound_calls_made": 20, "calls_connected": 8,
        },
        {
            # Small CHARGEOFF, no bureau — expect SMALL_CLAIMS, lower priority
            "account_id": "ACC002", "balance": 45_000, "stage": "CHARGEOFF",
            "days_past_due": 360, "bureau_secured_loan_flag": False,
            "last_payment_amount": 500, "payment_count_12m": 0,
            "total_paid_12m": 500, "ptp_made": 2, "ptp_kept": 0,
            "outbound_calls_made": 15, "calls_connected": 1,
        },
        {
            # SM stage — ineligible (stage not in eligible list)
            "account_id": "ACC003", "balance": 200_000, "stage": "SM",
            "days_past_due": 45,
        },
        {
            # Very large NPL — expect ENFORCEMENT (has asset), top priority
            "account_id": "ACC004", "balance": 2_500_000, "stage": "NPL",
            "days_past_due": 270, "bureau_secured_loan_flag": True,
            "bureau_secured_outstanding": 5_000_000,
            "bureau_monthly_instalment": 80_000,
            "last_payment_amount": 10_000, "payment_count_12m": 1,
            "total_paid_12m": 10_000, "ptp_made": 1, "ptp_kept": 0,
            "outbound_calls_made": 30, "calls_connected": 2,
        },
    ])

    mgr = LegalQueueManager()
    cases_df, summary = mgr.process_batch(accounts)

    print("\n── CASES ──")
    print(cases_df[["account_id", "balance", "stage", "eligible",
                     "priority_score", "recommended_action", "status",
                     "data_confidence", "notes"]].to_string(index=False))

    # Assign
    cases_df = mgr.assign_cases(cases_df)
    print("\n── AFTER ASSIGNMENT ──")
    print(cases_df[["account_id", "priority_score", "status", "assigned_to"]].to_string(index=False))

    # Escalation check
    cases_df = mgr.check_escalations(cases_df)

    # Report
    print("\n" + mgr.queue_report(cases_df))

    print(f"\nSummary: {summary.eligible_count}/{summary.total_accounts} eligible, "
          f"total balance {summary.total_balance_queued:,.0f}, "
          f"avg priority {summary.avg_priority_score}")
