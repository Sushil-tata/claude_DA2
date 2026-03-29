"""
Suppression Overlay
====================
Applied AFTER NBA scoring as a separate, auditable layer.
Never modifies scores — only blocks or flags actions.

Design:
  - Model produces best action with score
  - Overlay checks each rule independently
  - Each suppression is logged with reason (audit trail)
  - Falls back to next-best action if primary is suppressed
  - If all actions suppressed → NO_CONTACT with reason

Suppression tiers (in order of application):
  TIER 1 — Account-level hold (STOP_FROM_DT / STOP_TO_DT)
            Hard block — no contact of any kind
  TIER 2 — Legal hold
            Hard block — no direct contact while litigation active
  TIER 3 — Thai BOT operational rules
            Frequency cap: max 1 call/day, 3 calls/week
            Quiet hours: 08:00–20:00 Bangkok time only
  TIER 4 — Marketing consent flags (DO_NOT_MKTG_*)
            Channel-specific soft block — pending compliance confirmation
            whether PDPA marketing flags extend to collections contact.
            Configurable: apply_mktg_flags_to_collections (default: False)

Open question noted in schema_mapper.py:
  No collections-specific DNC flag found in source schema.
  If one is introduced later, add as TIER 1.5 here.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import pandas as pd

from ..tdr.data_contract import _is_null

logger = logging.getLogger(__name__)

BKK = ZoneInfo("Asia/Bangkok")


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SuppressionConfig:
    """All suppression rules — load from YAML in production."""

    # ── Thai BOT frequency limits ─────────────────────────────────────────────
    max_voice_calls_per_day: int  = 1
    max_voice_calls_per_week: int = 3
    quiet_hour_start: int         = 8    # 08:00 Bangkok time
    quiet_hour_end: int           = 20   # 20:00 Bangkok time

    # ── Marketing flag handling ───────────────────────────────────────────────
    # IMPORTANT: Set False until compliance team confirms PDPA marketing flags
    # also apply to debt collection contact.
    apply_mktg_flags_to_collections: bool = False

    # ── Legal hold ───────────────────────────────────────────────────────────
    # Statuses in LegalQueueManager that block direct customer contact
    legal_no_contact_statuses: List[str] = field(default_factory=lambda: [
        "FILED", "ACTIVE"
    ])

    # ── Channel → tier mapping ────────────────────────────────────────────────
    # Which channels are subject to BOT frequency limits
    frequency_capped_channels: List[str] = field(default_factory=lambda: [
        "AGENT_CALL", "VOICE_IVR"
    ])
    quiet_hour_channels: List[str] = field(default_factory=lambda: [
        "AGENT_CALL", "VOICE_IVR"
    ])

    # ── Marketing flag → channel mapping ─────────────────────────────────────
    mktg_flag_channel_map: Dict[str, List[str]] = field(default_factory=lambda: {
        "mktg_suppress_phone": ["AGENT_CALL", "VOICE_IVR"],
        "mktg_suppress_sms":   ["SMS"],
        "mktg_suppress_email": ["EMAIL"],
        "mktg_suppress_line":  ["LINE"],
    })


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SuppressionResult:
    """Outcome of suppression check for one action recommendation."""
    account_id: str
    original_action: str
    final_action: str           # Same as original if not suppressed
    is_suppressed: bool
    suppression_tier: Optional[int]   # 1/2/3/4 — which tier blocked it
    suppression_reason: str
    fallback_used: bool
    fallback_action: Optional[str]
    checked_at: str


# ─────────────────────────────────────────────────────────────────────────────
# SUPPRESSION OVERLAY
# ─────────────────────────────────────────────────────────────────────────────

class SuppressionOverlay:
    """
    Applies suppression rules to NBA recommendations after scoring.

    Usage (single account):
        overlay = SuppressionOverlay()
        result = overlay.apply(
            account=row,
            recommended_action="AGENT_CALL",
            fallback_actions=["SMS", "EMAIL"],
            contact_history_today=2,
            contact_history_week=3,
            legal_status=None,
        )

    Usage (batch):
        results_df = overlay.apply_batch(nba_df, accounts_df, contact_summary_df)
    """

    def __init__(self, config: Optional[SuppressionConfig] = None):
        self.cfg = config or SuppressionConfig()

    # ── PUBLIC ────────────────────────────────────────────────────────────────

    def apply(
        self,
        account: pd.Series,
        recommended_action: str,
        fallback_actions: Optional[List[str]] = None,
        contact_history_today: int = 0,
        contact_history_week: int = 0,
        legal_status: Optional[str] = None,
        check_time: Optional[datetime] = None,
    ) -> SuppressionResult:
        """
        Check all suppression tiers for a single recommended action.
        Returns SuppressionResult with final_action (suppressed or original).
        """
        aid = str(account.get("account_id", "unknown"))
        now = check_time or datetime.now(tz=BKK)

        # Run tiers in order — first hit wins
        suppressed, tier, reason = self._check_all_tiers(
            account, recommended_action,
            contact_history_today, contact_history_week,
            legal_status, now,
        )

        fallback_used   = False
        fallback_action = None
        final_action    = recommended_action

        if suppressed:
            # Try fallback actions in order
            for fb in (fallback_actions or []):
                fb_suppressed, _, _ = self._check_all_tiers(
                    account, fb,
                    contact_history_today, contact_history_week,
                    legal_status, now,
                )
                if not fb_suppressed:
                    fallback_used   = True
                    fallback_action = fb
                    final_action    = fb
                    break
            else:
                # All actions suppressed
                final_action = "NO_CONTACT"

        return SuppressionResult(
            account_id=aid,
            original_action=recommended_action,
            final_action=final_action,
            is_suppressed=suppressed,
            suppression_tier=tier if suppressed else None,
            suppression_reason=reason if suppressed else "",
            fallback_used=fallback_used,
            fallback_action=fallback_action,
            checked_at=now.isoformat(),
        )

    def apply_batch(
        self,
        nba_df: pd.DataFrame,
        accounts_df: pd.DataFrame,
        contact_summary_df: Optional[pd.DataFrame] = None,
        legal_cases_df: Optional[pd.DataFrame] = None,
        check_time: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Apply suppression to a batch NBA output DataFrame.

        Expected columns in nba_df:
            account_id, recommended_action, fallback_action_1, fallback_action_2

        Returns nba_df with added columns:
            final_action, is_suppressed, suppression_reason, suppression_tier
        """
        # Build lookup dicts for performance
        account_map  = accounts_df.set_index("account_id").to_dict("index") if "account_id" in accounts_df.columns else {}
        contact_map  = self._build_contact_map(contact_summary_df)
        legal_map    = self._build_legal_map(legal_cases_df)

        results = []
        for _, row in nba_df.iterrows():
            aid = str(row.get("account_id", "unknown"))

            account = pd.Series(account_map.get(aid, {}))
            account["account_id"] = aid

            fallbacks = [
                a for a in [
                    row.get("fallback_action_1"),
                    row.get("fallback_action_2"),
                ] if a and not _is_null(a)
            ]

            today_calls, week_calls = contact_map.get(aid, (0, 0))
            legal_status = legal_map.get(aid)

            result = self.apply(
                account=account,
                recommended_action=str(row.get("recommended_action", "NO_CONTACT")),
                fallback_actions=fallbacks,
                contact_history_today=today_calls,
                contact_history_week=week_calls,
                legal_status=legal_status,
                check_time=check_time,
            )
            results.append({
                "account_id":        aid,
                "original_action":   result.original_action,
                "final_action":      result.final_action,
                "is_suppressed":     result.is_suppressed,
                "suppression_tier":  result.suppression_tier,
                "suppression_reason":result.suppression_reason,
                "fallback_used":     result.fallback_used,
            })

        overlay_df = pd.DataFrame(results)
        return nba_df.merge(overlay_df, on="account_id", how="left")

    # ── PRIVATE — TIER CHECKS ─────────────────────────────────────────────────

    def _check_all_tiers(
        self,
        account: pd.Series,
        action: str,
        today_calls: int,
        week_calls: int,
        legal_status: Optional[str],
        now: datetime,
    ) -> Tuple[bool, Optional[int], str]:
        """
        Returns (is_suppressed, tier, reason).
        Checks tiers 1-4 in order; first hit returns immediately.
        """

        # TIER 1: Account-level hold
        suppressed, reason = self._check_account_hold(account, now)
        if suppressed:
            return True, 1, reason

        # TIER 2: Legal hold
        suppressed, reason = self._check_legal_hold(legal_status, action)
        if suppressed:
            return True, 2, reason

        # TIER 3: BOT operational rules (voice channels only)
        suppressed, reason = self._check_bot_rules(action, today_calls, week_calls, now)
        if suppressed:
            return True, 3, reason

        # TIER 4: Marketing consent flags (configurable)
        if self.cfg.apply_mktg_flags_to_collections:
            suppressed, reason = self._check_mktg_flags(account, action)
            if suppressed:
                return True, 4, reason

        return False, None, ""

    def _check_account_hold(
        self, account: pd.Series, now: datetime
    ) -> Tuple[bool, str]:
        """TIER 1: STOP_FROM_DT / STOP_TO_DT account hold."""
        stop_from = account.get("stop_from_dt")
        stop_to   = account.get("stop_to_dt")
        reason    = account.get("stop_reason_code", "")

        if _is_null(stop_from):
            return False, ""

        try:
            from_dt = pd.to_datetime(stop_from)
            now_naive = now.replace(tzinfo=None) if now.tzinfo else now

            if from_dt <= now_naive:
                if _is_null(stop_to):
                    return True, f"Account on hold from {from_dt.date()} (open-ended) reason={reason}"
                to_dt = pd.to_datetime(stop_to)
                if now_naive <= to_dt:
                    return True, f"Account on hold {from_dt.date()} to {to_dt.date()} reason={reason}"
        except Exception:
            pass

        return False, ""

    def _check_legal_hold(
        self, legal_status: Optional[str], action: str
    ) -> Tuple[bool, str]:
        """TIER 2: Block direct contact when account is in active litigation."""
        if not legal_status:
            return False, ""

        # Only voice channels are blocked during litigation
        # Written/digital contact (SMS, EMAIL, LINE) may still be permitted
        # depending on court order — leave that to legal team
        if (legal_status in self.cfg.legal_no_contact_statuses
                and action in self.cfg.frequency_capped_channels):
            return True, f"Account in active litigation (status={legal_status}) — no direct voice contact"

        return False, ""

    def _check_bot_rules(
        self,
        action: str,
        today_calls: int,
        week_calls: int,
        now: datetime,
    ) -> Tuple[bool, str]:
        """TIER 3: Thai BOT debt collection operational rules."""
        c = self.cfg

        # Quiet hours — voice only
        if action in c.quiet_hour_channels:
            bkk_hour = now.astimezone(BKK).hour if now.tzinfo else now.hour
            if not (c.quiet_hour_start <= bkk_hour < c.quiet_hour_end):
                return True, (
                    f"Outside permitted contact hours "
                    f"({c.quiet_hour_start:02d}:00-{c.quiet_hour_end:02d}:00 BKK) "
                    f"— current hour {bkk_hour:02d}:00"
                )

        # Frequency cap — voice only
        if action in c.frequency_capped_channels:
            if today_calls >= c.max_voice_calls_per_day:
                return True, (
                    f"Daily call limit reached: "
                    f"{today_calls}/{c.max_voice_calls_per_day} calls today"
                )
            if week_calls >= c.max_voice_calls_per_week:
                return True, (
                    f"Weekly call limit reached: "
                    f"{week_calls}/{c.max_voice_calls_per_week} calls this week"
                )

        return False, ""

    def _check_mktg_flags(
        self, account: pd.Series, action: str
    ) -> Tuple[bool, str]:
        """
        TIER 4: Marketing consent flags.
        Only applied when apply_mktg_flags_to_collections=True.
        Default: False — pending compliance team confirmation.
        """
        for flag_field, blocked_channels in self.cfg.mktg_flag_channel_map.items():
            flag_val = account.get(flag_field)
            if not _is_null(flag_val) and int(flag_val) == 1:
                if action in blocked_channels:
                    return True, (
                        f"Marketing suppression flag {flag_field}=1 "
                        f"— channel {action} blocked (pending compliance review)"
                    )
        return False, ""

    # ── PRIVATE — BATCH HELPERS ───────────────────────────────────────────────

    def _build_contact_map(
        self, contact_summary_df: Optional[pd.DataFrame]
    ) -> Dict[str, Tuple[int, int]]:
        """Build {account_id: (calls_today, calls_week)} from contact summary."""
        if contact_summary_df is None or contact_summary_df.empty:
            return {}
        result = {}
        for _, row in contact_summary_df.iterrows():
            aid = str(row.get("account_id", ""))
            result[aid] = (
                int(row.get("voice_calls_today", 0) or 0),
                int(row.get("voice_calls_week",  0) or 0),
            )
        return result

    def _build_legal_map(
        self, legal_cases_df: Optional[pd.DataFrame]
    ) -> Dict[str, Optional[str]]:
        """Build {account_id: legal_status} from legal queue output."""
        if legal_cases_df is None or legal_cases_df.empty:
            return {}
        result = {}
        for _, row in legal_cases_df.iterrows():
            aid = str(row.get("account_id", ""))
            result[aid] = row.get("status")
        return result


# ─────────────────────────────────────────────────────────────────────────────
# SMOKE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    overlay = SuppressionOverlay()

    accounts = [
        # Normal account, business hours — no suppression
        (pd.Series({"account_id": "A001"}),
         "AGENT_CALL", ["SMS"], 0, 1, None,
         datetime(2026, 2, 10, 10, 0, tzinfo=BKK)),

        # Account on hold — TIER 1
        (pd.Series({"account_id": "A002",
                    "stop_from_dt": "2026-01-01", "stop_to_dt": "2026-12-31",
                    "stop_reason_code": "LEGAL"}),
         "AGENT_CALL", ["SMS"], 0, 0, None,
         datetime(2026, 2, 10, 10, 0, tzinfo=BKK)),

        # Active litigation — TIER 2 (voice blocked, SMS fallback used)
        (pd.Series({"account_id": "A003"}),
         "AGENT_CALL", ["SMS", "EMAIL"], 0, 0, "ACTIVE",
         datetime(2026, 2, 10, 10, 0, tzinfo=BKK)),

        # Outside quiet hours (22:00) — TIER 3
        (pd.Series({"account_id": "A004"}),
         "AGENT_CALL", ["SMS"], 0, 0, None,
         datetime(2026, 2, 10, 22, 0, tzinfo=BKK)),

        # Daily call limit hit — TIER 3
        (pd.Series({"account_id": "A005"}),
         "AGENT_CALL", ["SMS"], 1, 2, None,
         datetime(2026, 2, 10, 10, 0, tzinfo=BKK)),

        # Weekly call limit hit — TIER 3
        (pd.Series({"account_id": "A006"}),
         "AGENT_CALL", ["SMS"], 0, 3, None,
         datetime(2026, 2, 10, 10, 0, tzinfo=BKK)),

        # Marketing flag set — BUT apply_mktg_flags=False (default)
        # So NOT suppressed
        (pd.Series({"account_id": "A007", "mktg_suppress_sms": 1}),
         "SMS", [], 0, 0, None,
         datetime(2026, 2, 10, 10, 0, tzinfo=BKK)),
    ]

    print(f"{'ACCT':<6} {'ORIGINAL':<15} {'FINAL':<15} {'SUPP':<6} {'TIER':<5} {'REASON'}")
    print("-" * 90)
    for account, action, fallbacks, today, week, legal, ts in accounts:
        r = overlay.apply(account, action, fallbacks, today, week, legal, ts)
        print(
            f"{r.account_id:<6} {r.original_action:<15} {r.final_action:<15} "
            f"{str(r.is_suppressed):<6} {str(r.suppression_tier):<5} "
            f"{r.suppression_reason[:55]}"
        )
