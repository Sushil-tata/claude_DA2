"""
Offer Generator + Path Ranker (v2)
====================================
Generates single best TDR offer using paydown-curve NPV (no take-up rate).

Key design:
  - No take-up probability — NPV grounded in empirical paydown curves
  - Balance decomposition: waivers applied cheapest-first (charges → interest → principal)
  - Offer ranking by incremental_npv (gain over Curve A baseline)
  - Affordability check: instalment ≤ ATP from AffordabilityEngine
  - within_policy flag for approval workflow
  - Confidence level from data completeness
  - Audit log row per recommendation
"""

import logging
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime

import numpy as np
import pandas as pd

from .affordability_engine import AffordabilityEngine, AffordabilityConfig, AffordabilityProfile
from .npv_engine import NPVEngine, NPVConfig, LoanNPV, PathValuation
from .data_contract import DataContract, DataQuality
from .paydown_curves import PaydownCurveEngine, BalanceDecomposition, compute_waivers

logger = logging.getLogger(__name__)


@dataclass
class PolicyConfig:
    """
    Concession guardrails by stage (Q6 - policy table).
    Loaded from config YAML — not hardcoded.
    """
    max_haircut_by_stage: Dict = field(default_factory=lambda: {
        "SM": 0.20, "NPL": 0.40, "CHARGEOFF": 0.65
    })
    max_fee_waiver_pct: Dict = field(default_factory=lambda: {
        "SM": 0.50, "NPL": 0.80, "CHARGEOFF": 1.00
    })
    policy_rate_by_stage: Dict = field(default_factory=lambda: {
        "SM": 0.15, "NPL": 0.10, "CHARGEOFF": 0.00
    })
    approval_haircut_threshold: Dict = field(default_factory=lambda: {
        "SM": 0.10, "NPL": 0.20, "CHARGEOFF": 0.40
    })
    min_settlement_amount: float = 1000.0
    offer_tenors: List[int] = field(default_factory=lambda: [12, 24, 36, 48, 60])
    settlement_haircuts: List[float] = field(default_factory=lambda:
        [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
    )


@dataclass
class OfferRecommendation:
    """Single best offer recommendation for agent screen."""
    # Identity
    recommendation_id: str
    account_id: str
    generated_at: str

    # Account context
    outstanding: float
    stage: str
    persona: str
    months_at_180plus: int
    confidence_level: str
    completeness_pct: float

    # Affordability
    estimated_monthly_income: float
    income_tier_used: int
    payment_capacity: float
    is_over_indebted: bool

    # Recommended path
    recommended_path: str
    path_rationale: str

    # Best offer (if TDR) — paydown-curve NPV, no take-up probability
    offer_type: str
    monthly_instalment: float
    tenor_months: int
    haircut_pct: float
    total_recovery: float
    loan_npv: float            # PV(Curve_B) − concession
    incremental_npv: float     # gain over natural recovery (Curve_A baseline)
    hold_npv: float            # PV(Curve_A) — what we get without any offer

    # Balance decomposition
    principal_waived: float
    interest_waived: float
    charges_waived: float
    concession_cost: float

    # Policy
    within_policy: bool
    needs_approval: bool
    approval_reason: str

    # Comparators
    debt_sale_floor: float
    legal_viable: bool

    # Agent talking points
    talking_points: List[str]

    # Audit
    agent_id: Optional[str]
    outcome: Optional[str]       # filled later: accepted/rejected/countered
    outcome_timestamp: Optional[str]
    data_warnings: List[str]


@dataclass
class AuditLogRow:
    """One row per recommendation — for feedback loop."""
    recommendation_id: str
    account_id: str
    generated_at: str
    stage: str
    persona: str
    months_at_180plus: int
    outstanding: float
    recommended_path: str
    offer_type: str
    monthly_instalment: float
    tenor_months: int
    haircut_pct: float
    loan_npv: float
    incremental_npv: float
    hold_npv: float
    concession_cost: float
    within_policy: bool
    confidence_level: str
    completeness_pct: float
    income_tier_used: int
    agent_id: Optional[str]
    outcome: Optional[str]
    outcome_timestamp: Optional[str]


class OfferGenerator:
    """
    Generates single best TDR offer recommendation.
    Uses paydown-curve NPV (no take-up rate) and cheapest-first balance waivers.
    Degrades gracefully on missing data.
    """

    def __init__(
        self,
        policy_config:        Optional[PolicyConfig]        = None,
        affordability_config: Optional[AffordabilityConfig] = None,
        npv_config:           Optional[NPVConfig]           = None,
        curve_engine:         Optional[PaydownCurveEngine]  = None,
    ):
        self.policy       = policy_config        or PolicyConfig()
        self.aff_engine   = AffordabilityEngine(affordability_config)
        self.curve_engine = curve_engine or PaydownCurveEngine()
        self.npv_engine   = NPVEngine(npv_config, self.curve_engine)
        self.contract     = DataContract()

    def generate(
        self,
        account: pd.Series,
        stage: Optional[str] = None,
        persona: Optional[str] = None,
        months_at_180plus: Optional[int] = None,
        agent_id: Optional[str] = None,
    ) -> OfferRecommendation:
        """
        Generate single best offer recommendation for one account.

        Args:
            account: pd.Series with account fields (see DataContract)
            stage: SM | NPL | CHARGEOFF (overrides account["stage"])
            persona: recovery persona from RecoveryScorecard
                     (SELECTIVE_DEFAULTER, LIFE_EVENT, etc.)
                     Falls back to account["persona"] then "UNKNOWN"
            months_at_180plus: how long account has been at 180+ DPD;
                               falls back to account["months_at_180plus"] then 0
            agent_id: agent identifier for audit log
        """

        # 1. Validate + enrich
        quality  = self.contract.validate(account)
        account  = self.contract.enrich(account)

        outstanding        = float(account.get("balance", 0))
        account_id         = str(account.get("account_id", "unknown"))
        stage              = (stage or str(account.get("stage", "NPL"))).upper()
        persona            = persona or str(account.get("persona", "UNKNOWN"))
        months_at_180plus  = int(months_at_180plus or account.get("months_at_180plus", 0) or 0)
        rec_id             = str(uuid.uuid4())[:12]
        now                = datetime.now().isoformat(timespec="seconds")

        # 2. Affordability
        aff = self.aff_engine.assess(account)

        # 3. Balance decomposition (cheapest-first waiver ordering)
        decomp = self._extract_balance_decomp(account, outstanding)

        # 4. Generate offer candidates (uses paydown-curve NPV, no take-up rate)
        candidates = self._generate_candidates(
            outstanding, stage, decomp, aff, persona, months_at_180plus
        )

        # 5. Rank by incremental_npv (gain over Curve A baseline)
        affordable = [c for c in candidates if c.is_affordable]
        ranked     = affordable or candidates
        best_offer = max(ranked, key=lambda x: x.incremental_npv) if ranked else None

        # 6. Path comparison (includes HOLD = Curve A NPV)
        bureau_asset = float(account.get("bureau_secured_outstanding", 0) or 0)
        has_asset    = bool(account.get("bureau_secured_loan_flag", False))

        path_val = self.npv_engine.compare_paths(
            account_id=account_id,
            outstanding=outstanding,
            stage=stage,
            tdr_offers=candidates,
            persona=persona,
            months_at_180plus=months_at_180plus,
            bureau_asset_value=bureau_asset,
            has_secured_asset=has_asset,
        )

        # 7. Policy check
        within_policy, needs_approval, approval_reason = self._check_policy(
            best_offer, stage
        )

        # 8. Talking points
        talking_points = self._build_talking_points(
            account, aff, best_offer, stage, path_val, quality
        )

        return OfferRecommendation(
            recommendation_id=rec_id,
            account_id=account_id,
            generated_at=now,
            outstanding=outstanding,
            stage=stage,
            persona=persona,
            months_at_180plus=months_at_180plus,
            confidence_level=quality.confidence_level,
            completeness_pct=quality.completeness_pct,
            estimated_monthly_income=aff.estimated_monthly_income,
            income_tier_used=aff.income_tier_used,
            payment_capacity=aff.payment_capacity,
            is_over_indebted=aff.is_over_indebted,
            recommended_path=path_val.recommended_path,
            path_rationale=path_val.path_rationale,
            offer_type=best_offer.offer_type if best_offer else "NONE",
            monthly_instalment=best_offer.monthly_instalment if best_offer else 0.0,
            tenor_months=best_offer.tenor_months if best_offer else 0,
            haircut_pct=best_offer.haircut_pct if best_offer else 0.0,
            total_recovery=round(
                (best_offer.monthly_instalment or 0) * (best_offer.tenor_months or 0), 2
            ) if best_offer else 0.0,
            loan_npv=best_offer.loan_npv if best_offer else 0.0,
            incremental_npv=best_offer.incremental_npv if best_offer else 0.0,
            hold_npv=path_val.hold_npv,
            principal_waived=best_offer.principal_waived if best_offer else 0.0,
            interest_waived=best_offer.interest_waived if best_offer else 0.0,
            charges_waived=best_offer.charges_waived if best_offer else 0.0,
            concession_cost=best_offer.concession_cost if best_offer else 0.0,
            within_policy=within_policy,
            needs_approval=needs_approval,
            approval_reason=approval_reason,
            debt_sale_floor=path_val.debt_sale_npv,
            legal_viable=path_val.legal_viable,
            talking_points=talking_points,
            agent_id=agent_id,
            outcome=None,
            outcome_timestamp=None,
            data_warnings=quality.warnings,
        )

    def generate_batch(
        self, df: pd.DataFrame, stage_col: str = "stage", agent_id: Optional[str] = None
    ) -> List[OfferRecommendation]:
        results = []
        for _, row in df.iterrows():
            try:
                results.append(self.generate(row, stage=str(row.get(stage_col, "NPL")), agent_id=agent_id))
            except Exception as e:
                logger.warning("Failed for %s: %s", row.get("account_id"), e)
        return results

    def to_dataframe(self, recommendations: List[OfferRecommendation]) -> pd.DataFrame:
        return pd.DataFrame([{
            "recommendation_id":  r.recommendation_id,
            "account_id":         r.account_id,
            "generated_at":       r.generated_at,
            "stage":              r.stage,
            "persona":            r.persona,
            "months_at_180plus":  r.months_at_180plus,
            "outstanding":        r.outstanding,
            "confidence_level":   r.confidence_level,
            "completeness_pct":   r.completeness_pct,
            "recommended_path":   r.recommended_path,
            "offer_type":         r.offer_type,
            "monthly_instalment": r.monthly_instalment,
            "tenor_months":       r.tenor_months,
            "haircut_pct":        r.haircut_pct,
            "total_recovery":     r.total_recovery,
            "loan_npv":           r.loan_npv,
            "incremental_npv":    r.incremental_npv,
            "hold_npv":           r.hold_npv,
            "principal_waived":   r.principal_waived,
            "interest_waived":    r.interest_waived,
            "charges_waived":     r.charges_waived,
            "concession_cost":    r.concession_cost,
            "within_policy":      r.within_policy,
            "needs_approval":     r.needs_approval,
            "payment_capacity":   r.payment_capacity,
            "income_tier_used":   r.income_tier_used,
            "debt_sale_floor":    r.debt_sale_floor,
            "legal_viable":       r.legal_viable,
            "talking_points":     " | ".join(r.talking_points),
            "agent_id":           r.agent_id,
            "outcome":            r.outcome,
            "data_warnings":      "; ".join(r.data_warnings),
        } for r in recommendations])

    def to_audit_log(self, recommendations: List[OfferRecommendation]) -> pd.DataFrame:
        return pd.DataFrame([{
            "recommendation_id":  r.recommendation_id,
            "account_id":         r.account_id,
            "generated_at":       r.generated_at,
            "stage":              r.stage,
            "persona":            r.persona,
            "months_at_180plus":  r.months_at_180plus,
            "outstanding":        r.outstanding,
            "recommended_path":   r.recommended_path,
            "offer_type":         r.offer_type,
            "monthly_instalment": r.monthly_instalment,
            "tenor_months":       r.tenor_months,
            "haircut_pct":        r.haircut_pct,
            "loan_npv":           r.loan_npv,
            "incremental_npv":    r.incremental_npv,
            "hold_npv":           r.hold_npv,
            "concession_cost":    r.concession_cost,
            "within_policy":      r.within_policy,
            "confidence_level":   r.confidence_level,
            "completeness_pct":   r.completeness_pct,
            "income_tier_used":   r.income_tier_used,
            "agent_id":           r.agent_id,
            "outcome":            r.outcome,
            "outcome_timestamp":  r.outcome_timestamp,
        } for r in recommendations])

    def format_agent_screen(self, rec: OfferRecommendation) -> str:
        """Plain text for agent screen display."""
        conf_icon = {"HIGH": "✅", "MEDIUM": "⚠️", "LOW": "❌"}.get(rec.confidence_level, "?")
        policy_icon = "✅ WITHIN POLICY" if rec.within_policy else "⚠️ NEEDS APPROVAL"

        lines = [
            "=" * 55,
            f"  {rec.account_id}  |  {rec.stage}  |  Balance: {rec.outstanding:,.0f}",
            f"  Data confidence: {conf_icon} {rec.confidence_level} ({rec.completeness_pct:.0f}% complete)",
            "=" * 55,
            "",
            "  AFFORDABILITY",
            f"  Income estimate : {rec.estimated_monthly_income:>10,.0f}  (Tier {rec.income_tier_used})",
            f"  Obligations     : {rec.estimated_monthly_income - rec.payment_capacity / 0.6:>10,.0f}",
            f"  Payment capacity: {rec.payment_capacity:>10,.0f} /mo",
            f"  {'⚠️ OVER-INDEBTED' if rec.is_over_indebted else '✅ Serviceable'}",
            "",
            f"  RECOMMENDED PATH: {rec.recommended_path}",
        ]

        if rec.recommended_path == "TDR" and rec.offer_type != "NONE":
            lines += [
                f"  Offer type      : {rec.offer_type}",
                f"  Monthly payment : {rec.monthly_instalment:>10,.0f} /mo",
                f"  Tenor           : {rec.tenor_months} months",
                f"  Haircut         : {rec.haircut_pct:.0%}",
                f"  Charges waived  : {rec.charges_waived:>10,.0f}",
                f"  Interest waived : {rec.interest_waived:>10,.0f}",
                f"  Principal waived: {rec.principal_waived:>10,.0f}",
                f"  Concession cost : {rec.concession_cost:>10,.0f}",
                f"  Loan NPV        : {rec.loan_npv:>10,.0f}",
                f"  Incremental NPV : {rec.incremental_npv:>10,.0f}  (vs natural recovery)",
                f"  Natural recovery: {rec.hold_npv:>10,.0f}  (Curve A baseline)",
                f"  {policy_icon}",
            ]
            if rec.needs_approval:
                lines.append(f"  Reason: {rec.approval_reason}")

        lines += [
            "",
            f"  Debt sale floor : {rec.debt_sale_floor:>10,.0f}",
            f"  Legal path      : {'Viable' if rec.legal_viable else 'Not recommended'}",
            "",
            "  TALKING POINTS",
        ]
        for tp in rec.talking_points:
            lines.append(f"  • {tp}")

        if rec.data_warnings:
            lines += ["", "  DATA GAPS"]
            for w in rec.data_warnings:
                lines.append(f"  ⚠ {w}")

        lines.append("=" * 55)
        return "\n".join(lines)

    # ── PRIVATE ───────────────────────────────────────────────────────────────

    def _extract_balance_decomp(
        self, account: pd.Series, outstanding: float
    ) -> BalanceDecomposition:
        """
        Extract principal / interest / charges from account fields.
        Falls back to ratio estimates if decomposition not available.
        """
        principal  = float(account.get("principal_outstanding", 0) or 0)
        interest   = float(account.get("accrued_interest",      0) or 0)
        charges    = float(account.get("penalty_charges",        0) or 0)

        if principal + interest + charges > 0:
            # Normalise to match outstanding (rounding)
            total    = principal + interest + charges
            scale    = outstanding / max(total, 1)
            principal = principal * scale
            interest  = interest  * scale
            charges   = charges   * scale
        else:
            # Fall back: assume 70% principal / 20% interest / 10% charges
            principal = outstanding * 0.70
            interest  = outstanding * 0.20
            charges   = outstanding * 0.10

        return BalanceDecomposition(
            principal=round(principal, 2),
            accrued_interest=round(interest, 2),
            penalty_charges=round(charges, 2),
            total=round(outstanding, 2),
        )

    def _generate_candidates(
        self,
        outstanding: float,
        stage: str,
        decomp: BalanceDecomposition,
        aff: AffordabilityProfile,
        persona: str,
        months_at_180plus: int,
    ) -> List[LoanNPV]:
        """
        Generate offer candidates using paydown-curve NPV.
        No take-up rate — rank by incremental_npv (gain over Curve A baseline).
        Waivers applied cheapest-first: charges → interest → principal.
        """
        policy     = self.policy
        orig_rate  = policy.policy_rate_by_stage.get(stage, 0.10)
        max_hcut   = policy.max_haircut_by_stage.get(stage, 0.40)
        candidates = []

        for offer_type, haircut, rate, tenor in self._offer_grid(stage, orig_rate, max_hcut):
            # Waiver amount = haircut × outstanding (cheapest-first decomposition)
            target_waiver = outstanding * haircut
            waived_decomp, concession_cost = compute_waivers(
                decomp,
                target_waiver_amount=target_waiver,
                processing_cost=self.npv_engine.config.processing_cost_per_tdr,
            )

            params = dict(
                offer_id=str(uuid.uuid4())[:8],
                offer_type=offer_type,
                outstanding=outstanding,
                haircut_pct=haircut,
                interest_rate_pa=rate,
                tenor_months=tenor,
                principal_waived=waived_decomp.principal_waived,
                interest_waived=waived_decomp.interest_waived,
                charges_waived=waived_decomp.charges_waived,
            )

            ln = self.npv_engine.compute_loan_npv(
                offer_params=params,
                persona=persona,
                months_at_180plus=months_at_180plus,
                payment_capacity=aff.payment_capacity,
                concession_cost=concession_cost,
            )
            candidates.append(ln)

        return candidates

    def _offer_grid(self, stage, orig_rate, max_hcut):
        """Generate (offer_type, haircut, rate, tenor) combinations."""
        policy = self.policy
        grid   = []

        # Full settlement
        for h in [h for h in policy.settlement_haircuts if h <= max_hcut * 1.5]:
            grid.append(("FULL_SETTLEMENT", h, 0.0, 1))

        # Reschedule rate reduce
        for rate_cut in [0.25, 0.50, 0.75, 1.00]:
            new_rate = max(orig_rate * (1 - rate_cut), 0.0)
            for tenor in policy.offer_tenors:
                grid.append(("RESCHEDULE_RATE_REDUCE", 0.0, new_rate, tenor))

        # Reschedule same rate
        for tenor in policy.offer_tenors:
            grid.append(("RESCHEDULE_SAME_RATE", 0.0, orig_rate, tenor))

        # Full restructure (rate cut + partial haircut)
        for h in [0.10, 0.20]:
            if h <= max_hcut:
                for tenor in [36, 48, 60]:
                    grid.append(("FULL_RESTRUCTURE", h, orig_rate * 0.50, tenor))

        return grid

    def _check_policy(self, offer: Optional[LoanNPV], stage: str):
        if offer is None:
            return True, False, ""

        threshold = self.policy.approval_haircut_threshold.get(stage, 0.20)
        if offer.haircut_pct > threshold:
            return (
                False, True,
                f"Haircut {offer.haircut_pct:.0%} exceeds auto-approve limit "
                f"{threshold:.0%} for {stage}"
            )
        return True, False, ""

    def _build_talking_points(
        self, account, aff, offer, stage, path_val, quality
    ) -> List[str]:
        points = []

        # Payment behaviour
        p12 = int(account.get("payment_count_12m", 0) or 0)
        if p12 > 0:
            points.append(f"Made {p12} payments in last 12 months — shows willingness")
        else:
            points.append("No payments in last 12 months — address willingness first")

        # Recency
        days_pay = account.get("days_since_last_payment")
        if days_pay and not pd.isna(days_pay) and int(days_pay) < 90:
            points.append(f"Last payment {int(days_pay)} days ago — recently active")

        # PTP history
        ptp_rate = float(account.get("ptp_kept_rate", 0) or 0)
        if ptp_rate > 0.6:
            points.append(f"Keeps {ptp_rate:.0%} of promises — reliable customer")
        elif ptp_rate > 0 and ptp_rate <= 0.6:
            points.append(f"Kept {ptp_rate:.0%} of promises — reinforce commitment")

        # Offer attractiveness
        if offer:
            if offer.incremental_npv > 0:
                points.append(
                    f"Offer generates {offer.incremental_npv:,.0f} incremental recovery "
                    f"over natural paydown (Curve A baseline={offer.pv_baseline:,.0f})"
                )
            else:
                points.append(
                    f"Offer NPV ({offer.loan_npv:,.0f}) below natural recovery baseline "
                    f"({offer.pv_baseline:,.0f}) — consider debt sale"
                )
            if offer.is_affordable:
                points.append(
                    f"Instalment {offer.monthly_instalment:,.0f} is within "
                    f"estimated capacity {aff.payment_capacity:,.0f}/mo"
                )
            else:
                points.append(
                    f"Instalment {offer.monthly_instalment:,.0f} may strain capacity "
                    f"({aff.payment_capacity:,.0f}/mo) — explore longer tenor"
                )

        # Escalation guidance
        max_hcut = self.policy.max_haircut_by_stage.get(stage, 0.40)
        if offer and offer.haircut_pct < max_hcut:
            remaining = max_hcut - offer.haircut_pct
            points.append(
                f"If rejected: policy allows up to {max_hcut:.0%} haircut "
                f"({remaining:.0%} room remaining) — escalate to supervisor"
            )

        # Legal flag
        if path_val.legal_viable:
            points.append(
                "Secured asset on bureau — legal path viable if TDR fails"
            )

        # Low confidence warning
        if quality.confidence_level == "LOW":
            points.append(
                "Low data confidence — verify income and contact details with customer"
            )

        return points[:6]  # cap at 6 for screen readability
