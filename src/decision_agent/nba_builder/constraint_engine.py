"""
Constraint Engine - Enforces all business constraints on account-action pairs.

Hard constraints (block action):
  - Compliance: DNC, cease_and_desist, bankruptcy, quiet hours
  - Operability: channel not available (no mobile, no email)
  - Fatigue: too many contacts recently, cooldown not met
  - Eligibility: balance/DPD thresholds not met

Soft constraints (reduce score, don't block):
  - Cost cap: action would exceed monthly budget
  - ROI: expected value below minimum threshold
  - Sequencing: digital-first preference
"""

from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Dict, List, Optional, Tuple, Any
import pytz


@dataclass
class AccountContext:
    """All data needed to evaluate constraints for one account."""
    account_id: str

    # Delinquency state
    days_past_due: int        = 0
    bucket: int               = 0
    balance: float            = 0.0

    # Contact flags
    mobile_present: bool      = False
    email_present: bool       = False
    line_optin: bool          = False
    sms_optin: bool           = False

    # Compliance flags
    dnc: bool                 = False
    cease_and_desist: bool    = False
    active_dispute: bool      = False
    bankruptcy_flag: bool     = False
    deceased_flag: bool       = False
    fraud_flag: bool          = False

    # Contact history
    total_contacts_7d: int    = 0
    total_contacts_30d: int   = 0
    days_since_last_sms: int  = 99
    days_since_last_call: int = 99
    days_since_last_email: int = 99
    days_since_last_line: int  = 99
    fatigue_score: float      = 0.0
    contact_cost_this_month: float = 0.0

    # Offer history
    active_settlement: bool   = False
    active_payment_plan: bool = False
    active_ptp: bool          = False
    days_since_last_settlement_offer: int = 99
    settlement_offered_last_30d: bool     = False


@dataclass
class ActionEvaluation:
    """Result of evaluating one action for one account."""
    action: str
    eligible: bool
    blocked_reasons: List[str] = field(default_factory=list)
    warnings: List[str]        = field(default_factory=list)
    score_multiplier: float    = 1.0   # 0.0 = fully blocked, 1.0 = no penalty


class ConstraintEngine:
    """
    Evaluates hard and soft constraints for every account-action pair.
    """

    def __init__(self, config: Dict):
        """
        Args:
            config: Full NBA config from Questionnaire.build_config()
        """
        self.config      = config
        self.actions     = config.get("actions", {})
        self.compliance  = config.get("compliance", {})
        self.fatigue     = config.get("fatigue", {})
        self.eligibility = config.get("eligibility", {})
        self.cost_cfg    = config.get("cost_constraints", {})
        self.capacity    = config.get("capacity", {})

        # Pre-build cooldown map: {action -> min_days}
        self.cooldown_days: Dict[str, int] = self.fatigue.get("cooldown_days", {
            "SMS": 2, "EMAIL": 1, "LINE": 1, "VOICE_IVR": 3, "AGENT_CALL": 7,
        })

        # Channel -> days_since_last_X field mapping
        self._last_contact_field = {
            "SMS":        "days_since_last_sms",
            "AGENT_CALL": "days_since_last_call",
            "VOICE_IVR":  "days_since_last_call",
            "EMAIL":      "days_since_last_email",
            "LINE":       "days_since_last_line",
        }

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────────────────────────────────────

    def evaluate_all_actions(
        self,
        account: AccountContext,
        current_datetime: Optional[datetime] = None,
    ) -> Dict[str, ActionEvaluation]:
        """
        Evaluate ALL actions for one account.

        Returns:
            Dict mapping action_name -> ActionEvaluation
        """
        if current_datetime is None:
            current_datetime = datetime.now()

        results = {}
        for action_name, action_cfg in self.actions.items():
            if not action_cfg.get("enabled", True):
                results[action_name] = ActionEvaluation(
                    action=action_name, eligible=False,
                    blocked_reasons=["Action disabled in config"]
                )
                continue
            results[action_name] = self._evaluate_action(
                account, action_name, action_cfg, current_datetime
            )
        return results

    def get_feasible_actions(
        self,
        account: AccountContext,
        current_datetime: Optional[datetime] = None,
    ) -> List[str]:
        """Return list of actions that pass all hard constraints."""
        evals = self.evaluate_all_actions(account, current_datetime)
        return [a for a, e in evals.items() if e.eligible]

    def explain_decision(
        self,
        account: AccountContext,
        recommended_action: str,
        current_datetime: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Explain why an action was recommended and why others were blocked.
        Returns human-readable explanation dict.
        """
        evals = self.evaluate_all_actions(account, current_datetime)
        feasible = [a for a, e in evals.items() if e.eligible]
        blocked  = {a: e.blocked_reasons for a, e in evals.items() if not e.eligible}

        return {
            "account_id":          account.account_id,
            "recommended_action":  recommended_action,
            "feasible_actions":    feasible,
            "blocked_actions":     blocked,
            "compliance_flags": {
                "dnc":             account.dnc,
                "cease_desist":    account.cease_and_desist,
                "active_dispute":  account.active_dispute,
                "bankruptcy":      account.bankruptcy_flag,
            },
            "fatigue_status": {
                "contacts_this_week":  account.total_contacts_7d,
                "max_allowed":         self.compliance.get("max_contacts_per_week", 3),
                "fatigue_score":       account.fatigue_score,
            },
        }

    # ──────────────────────────────────────────────────────────────────────────
    # PRIVATE: EVALUATORS
    # ──────────────────────────────────────────────────────────────────────────

    def _evaluate_action(
        self,
        account: AccountContext,
        action: str,
        action_cfg: Dict,
        current_dt: datetime,
    ) -> ActionEvaluation:
        blocked = []
        warnings = []
        multiplier = 1.0

        # ── 1. Global compliance (block everything) ────────────────────────
        if account.deceased_flag:
            return ActionEvaluation(action, False, ["Account holder deceased"])
        if account.fraud_flag and action != "NO_ACTION":
            return ActionEvaluation(action, False, ["Fraud flag active"])
        if account.bankruptcy_flag and self.compliance.get("exclude_bankruptcy", True):
            return ActionEvaluation(action, False, ["Bankruptcy filed"])

        # ── 2. Contact actions (not NO_ACTION) ────────────────────────────
        if action == "NO_ACTION":
            return ActionEvaluation(action, True)

        # Cease and desist / DNC
        if account.cease_and_desist:
            blocked.append("Cease and desist order active")
        if account.dnc and action not in ("SETTLEMENT_OFFER", "PAYMENT_PLAN"):
            blocked.append("Do Not Call registry")

        # Active dispute
        if account.active_dispute:
            blocked.append("Active dispute on account")

        # ── 3. Fatigue / contact frequency ────────────────────────────────
        max_weekly = self.compliance.get("max_contacts_per_week", 3)
        if account.total_contacts_7d >= max_weekly:
            blocked.append(f"Contact limit reached: {account.total_contacts_7d}/{max_weekly} this week")

        # Fatigue score threshold
        if account.fatigue_score >= 0.85:
            blocked.append(f"Fatigue score too high: {account.fatigue_score:.2f}")
        elif account.fatigue_score >= 0.70:
            warnings.append(f"High fatigue score: {account.fatigue_score:.2f}")
            multiplier *= 0.7

        # ── 4. Channel-specific cooldown ──────────────────────────────────
        channel = action_cfg.get("channel", "")
        cooldown = self.cooldown_days.get(action, self.cooldown_days.get(channel, 0))
        field_name = self._last_contact_field.get(action, self._last_contact_field.get(channel))
        if field_name:
            days_since = getattr(account, field_name, 99)
            if days_since < cooldown:
                blocked.append(
                    f"Cooldown not met: {days_since}d since last {action} "
                    f"(need {cooldown}d)"
                )

        # ── 5. Operability checks ──────────────────────────────────────────
        if channel == "sms":
            if not account.mobile_present:
                blocked.append("No mobile number on file")
            if not account.sms_optin:
                blocked.append("Customer not opted in to SMS")
        if channel == "line":
            if not account.line_optin:
                blocked.append("Customer not opted in to LINE")
        if channel == "email":
            if not account.email_present:
                blocked.append("No email on file")
        if channel in ("call", "voice_ivr"):
            if not account.mobile_present:
                blocked.append("No phone number on file")

        # ── 6. Eligibility thresholds ──────────────────────────────────────
        elig = self.eligibility.get(action, {})
        if elig:
            if "min_balance" in elig and account.balance < elig["min_balance"]:
                blocked.append(
                    f"Balance ${account.balance:,.0f} below minimum ${elig['min_balance']:,.0f}"
                )
            if "max_balance" in elig and account.balance > elig["max_balance"]:
                blocked.append(
                    f"Balance ${account.balance:,.0f} above maximum ${elig['max_balance']:,.0f}"
                )
            if "min_days_past_due" in elig and account.days_past_due < elig["min_days_past_due"]:
                blocked.append(
                    f"DPD {account.days_past_due} below minimum {elig['min_days_past_due']}"
                )
            if "max_days_past_due" in elig and account.days_past_due > elig["max_days_past_due"]:
                blocked.append(
                    f"DPD {account.days_past_due} above maximum {elig['max_days_past_due']}"
                )
            if "min_bucket" in elig and account.bucket < elig["min_bucket"]:
                blocked.append(
                    f"Bucket {account.bucket} below minimum bucket {elig['min_bucket']}"
                )
            if "max_bucket" in elig and account.bucket > elig["max_bucket"]:
                blocked.append(
                    f"Bucket {account.bucket} above maximum bucket {elig['max_bucket']}"
                )

        # ── 7. Offer-specific constraints ──────────────────────────────────
        if action in ("SETTLEMENT_OFFER", "SETTLEMENT_ONE_TIME", "SETTLEMENT_PAYMENT_PLAN"):
            if account.active_settlement:
                blocked.append("Active settlement already in place")
            cooldown_days = self.eligibility.get(action, {}).get("cooldown_days", 30)
            if account.settlement_offered_last_30d:
                blocked.append(f"Settlement already offered within last {cooldown_days} days")

        if action == "PAYMENT_PLAN":
            if account.active_payment_plan:
                blocked.append("Active payment plan already in place")

        if action == "AGENT_CALL":
            if account.active_ptp:
                warnings.append("Promise to pay active – call may not be needed")
                multiplier *= 0.5

        # ── 8. Cost cap ────────────────────────────────────────────────────
        action_cost = action_cfg.get("cost", 0.0)
        max_monthly = self.cost_cfg.get("max_cost_per_account_per_month", 15.0)
        if account.contact_cost_this_month + action_cost > max_monthly:
            blocked.append(
                f"Monthly cost cap would be exceeded: "
                f"${account.contact_cost_this_month:.2f} + ${action_cost:.2f} > ${max_monthly:.2f}"
            )

        # ── 9. Quiet hours (for voice/call channels) ───────────────────────
        if channel in ("call", "voice_ivr"):
            if self._is_quiet_hours(current_dt):
                blocked.append("Quiet hours: calls not allowed at this time")

        eligible = len(blocked) == 0
        return ActionEvaluation(action, eligible, blocked, warnings, multiplier)

    def _is_quiet_hours(self, dt: datetime) -> bool:
        """Return True if current time is outside allowed contact window."""
        tz_str = self.compliance.get("timezone", "UTC")
        try:
            tz = pytz.timezone(tz_str)
            local_dt = dt.astimezone(tz)
        except Exception:
            local_dt = dt

        start_str = self.compliance.get("quiet_hours_start", "08:00")
        end_str   = self.compliance.get("quiet_hours_end",   "21:00")

        def parse_t(s: str) -> time:
            h, m = map(int, s.split(":"))
            return time(h, m)

        start = parse_t(start_str)
        end   = parse_t(end_str)
        current = local_dt.time()

        # Quiet if before start OR after end
        return current < start or current > end
