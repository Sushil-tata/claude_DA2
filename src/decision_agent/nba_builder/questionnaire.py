"""
Questionnaire Engine - Interactive business constraint collection.

Guides the user through defining:
- Available actions (channels, offers)
- Costs per action
- Capacity limits
- Fatigue / cooldown rules
- Compliance requirements
- Business objectives
- Persona thresholds
"""

from typing import Dict, List, Any, Optional


# ─────────────────────────────────────────────────────────────────────────────
# DEFAULT VALUES (shown as suggestions, user can override)
# ─────────────────────────────────────────────────────────────────────────────
DEFAULTS = {
    "actions": {
        "NO_ACTION":         {"enabled": True,  "cost": 0.00,  "channel": "none"},
        "SMS":               {"enabled": True,  "cost": 0.25,  "channel": "sms"},
        "EMAIL":             {"enabled": True,  "cost": 0.05,  "channel": "email"},
        "LINE":              {"enabled": False, "cost": 0.15,  "channel": "line"},
        "VOICE_IVR":         {"enabled": True,  "cost": 1.00,  "channel": "voice_ivr"},
        "AGENT_CALL":        {"enabled": True,  "cost": 5.00,  "channel": "call"},
        "SETTLEMENT_OFFER":  {"enabled": True,  "cost": 0.30,  "channel": "sms"},
        "PAYMENT_PLAN":      {"enabled": True,  "cost": 0.30,  "channel": "sms"},
        "OA_REFERRAL":       {"enabled": False, "cost": 10.00, "channel": "oa"},
        "LEGAL_ACTION":      {"enabled": False, "cost": 50.00, "channel": "legal"},
    },
    "capacity": {
        "AGENT_CALL":   {"daily_limit": 5000,  "method": "top_expected_value"},
        "VOICE_IVR":    {"daily_limit": 10000, "method": "top_expected_value"},
        "OA_REFERRAL":  {"daily_limit": 1000,  "method": "worst_performers_high_balance"},
    },
    "fatigue": {
        "max_contacts_per_week":  3,
        "max_contacts_per_month": 8,
        "cooldown_days": {
            "SMS":         2,
            "EMAIL":       1,
            "LINE":        1,
            "VOICE_IVR":   3,
            "AGENT_CALL":  7,
        },
        "fatigue_weights": {
            "SMS":        0.3,
            "EMAIL":      0.1,
            "LINE":       0.2,
            "VOICE_IVR":  0.5,
            "AGENT_CALL": 1.0,
        },
        "decay_half_life_days": 14,
    },
    "compliance": {
        "respect_dnc":             True,
        "respect_cease_desist":    True,
        "quiet_hours_start":       "08:00",
        "quiet_hours_end":         "21:00",
        "timezone":                "UTC",
        "max_contacts_per_week":   3,
        "exclude_bankruptcy":      True,
        "exclude_deceased":        True,
        "exclude_fraud":           True,
    },
    "cost_constraints": {
        "max_cost_per_account_per_month": 15.00,
        "max_cost_per_dollar_collected":  0.35,
        "min_roi_threshold":              1.0,
    },
    "eligibility": {
        "SETTLEMENT_OFFER": {
            "min_days_past_due":  60,
            "min_balance":        2000,
            "min_bucket":         2,
            "cooldown_days":      30,
        },
        "OA_REFERRAL": {
            "min_days_past_due":  120,
            "min_balance":        3000,
            "min_bucket":         3,
        },
        "AGENT_CALL": {
            "min_balance":        1000,
            "max_bucket":         3,
        },
    },
    "business_objective": "maximize_net_recovery",
    "settlement": {
        "discount_tiers": [
            {"persona": "high_value_unresponsive", "max_discount_pct": 20},
            {"persona": "medium_value_willing",    "max_discount_pct": 30},
            {"persona": "low_value_chronic",       "max_discount_pct": 50},
        ],
        "min_payment_plan_months": 3,
        "max_payment_plan_months": 24,
    },
    "training": {
        "outcome_horizon_days":   7,
        "min_training_rows":      1000,
        "test_size_pct":          20,
        "validation_size_pct":    10,
        "use_uplift_model":       True,
        "use_tweedie_for_amount": True,
    },
}


class Questionnaire:
    """
    Interactive business constraint collection engine.
    Presents questions, accepts answers, builds config.
    """

    def __init__(self):
        self.config: Dict = {}
        self.answered: Dict[str, Any] = {}
        self._build_questions()

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────────────────────────────────────

    def get_all_questions(self) -> List[Dict]:
        """Return ordered list of all configuration questions."""
        return self.questions

    def get_pending_questions(self) -> List[Dict]:
        """Return questions not yet answered."""
        return [q for q in self.questions if q["key"] not in self.answered]

    def answer(self, key: str, value: Any) -> Optional[str]:
        """
        Record answer to a question.
        Returns follow-up question text if the answer triggers one, else None.
        """
        self.answered[key] = value
        return self._check_followup(key, value)

    def accept_defaults(self, section: Optional[str] = None):
        """
        Accept all defaults for a section or all sections.
        Useful when user says 'use defaults' for a section.
        """
        defaults_map = {
            "actions":          ("actions_enabled",       DEFAULTS["actions"]),
            "capacity":         ("capacity_limits",       DEFAULTS["capacity"]),
            "fatigue":          ("fatigue_rules",         DEFAULTS["fatigue"]),
            "compliance":       ("compliance_rules",      DEFAULTS["compliance"]),
            "cost_constraints": ("cost_constraints",      DEFAULTS["cost_constraints"]),
            "eligibility":      ("eligibility_rules",     DEFAULTS["eligibility"]),
            "settlement":       ("settlement_config",     DEFAULTS["settlement"]),
            "training":         ("training_config",       DEFAULTS["training"]),
        }
        if section and section in defaults_map:
            key, val = defaults_map[section]
            self.answered[key] = val
        else:
            for _, (key, val) in defaults_map.items():
                if key not in self.answered:
                    self.answered[key] = val

    def build_config(self) -> Dict:
        """
        Build final NBA configuration from all answers.
        Uses defaults for any unanswered questions.
        """
        self.accept_defaults()  # fill any gaps

        cfg = {
            "actions":           self._build_action_config(),
            "capacity":          self.answered.get("capacity_limits", DEFAULTS["capacity"]),
            "fatigue":           self.answered.get("fatigue_rules", DEFAULTS["fatigue"]),
            "compliance":        self.answered.get("compliance_rules", DEFAULTS["compliance"]),
            "cost_constraints":  self.answered.get("cost_constraints", DEFAULTS["cost_constraints"]),
            "eligibility":       self.answered.get("eligibility_rules", DEFAULTS["eligibility"]),
            "settlement":        self.answered.get("settlement_config", DEFAULTS["settlement"]),
            "business_objective": self.answered.get("business_objective", DEFAULTS["business_objective"]),
            "training":          self.answered.get("training_config", DEFAULTS["training"]),
            "personas":          self.answered.get("persona_definitions", self._default_personas()),
        }
        self.config = cfg
        return cfg

    def summary(self) -> str:
        """Return human-readable config summary."""
        cfg = self.build_config()
        enabled_actions = [k for k, v in cfg["actions"].items() if v.get("enabled")]
        lines = [
            "=" * 60,
            "NBA CONFIGURATION SUMMARY",
            "=" * 60,
            f"Actions enabled ({len(enabled_actions)}): {', '.join(enabled_actions)}",
            f"Business objective: {cfg['business_objective']}",
            "",
            "CHANNEL COSTS:",
        ]
        for action, meta in cfg["actions"].items():
            if meta.get("enabled"):
                lines.append(f"  {action}: ${meta['cost']:.2f}/contact")

        lines += ["", "CAPACITY LIMITS:"]
        for action, cap in cfg["capacity"].items():
            lines.append(f"  {action}: {cap['daily_limit']:,}/day")

        lines += ["", "FATIGUE RULES:"]
        fatigue = cfg["fatigue"]
        lines.append(f"  Max contacts/week: {fatigue['max_contacts_per_week']}")
        lines.append(f"  Max contacts/month: {fatigue['max_contacts_per_month']}")

        lines += ["", "COMPLIANCE:"]
        comp = cfg["compliance"]
        lines.append(f"  Respect DNC: {comp['respect_dnc']}")
        lines.append(f"  Quiet hours: {comp['quiet_hours_start']} - {comp['quiet_hours_end']} ({comp['timezone']})")
        lines.append(f"  Max contacts/week: {comp['max_contacts_per_week']}")

        lines += ["", "=" * 60]
        return "\n".join(lines)

    # ──────────────────────────────────────────────────────────────────────────
    # PRIVATE: QUESTION DEFINITIONS
    # ──────────────────────────────────────────────────────────────────────────

    def _build_questions(self):
        self.questions = [
            # ── Section 1: Actions ───────────────────────────────────────────
            {
                "section":  "actions",
                "key":      "market",
                "question": "What market/country is this deployment for?",
                "hint":     "e.g. Thailand, US, Philippines, Singapore",
                "type":     "text",
                "default":  None,
                "required": True,
            },
            {
                "section":  "actions",
                "key":      "channels_available",
                "question": "Which contact channels does your team use? (select all that apply)",
                "options":  ["SMS", "EMAIL", "LINE", "VOICE_IVR", "AGENT_CALL", "WHATSAPP", "PUSH_NOTIFICATION"],
                "type":     "multi_select",
                "default":  ["SMS", "EMAIL", "AGENT_CALL"],
                "required": True,
            },
            {
                "section":  "actions",
                "key":      "offers_available",
                "question": "Which offer types can your team make? (select all that apply)",
                "options":  ["SETTLEMENT_ONE_TIME", "SETTLEMENT_PAYMENT_PLAN",
                             "DEBT_RESTRUCTURE", "OA_REFERRAL", "LEGAL_ACTION"],
                "type":     "multi_select",
                "default":  ["SETTLEMENT_ONE_TIME", "SETTLEMENT_PAYMENT_PLAN"],
                "required": True,
            },

            # ── Section 2: Costs ─────────────────────────────────────────────
            {
                "section":  "costs",
                "key":      "cost_sms",
                "question": "Cost per SMS message (USD)?",
                "type":     "number",
                "default":  0.25,
                "required": False,
            },
            {
                "section":  "costs",
                "key":      "cost_email",
                "question": "Cost per email sent (USD)?",
                "type":     "number",
                "default":  0.05,
                "required": False,
            },
            {
                "section":  "costs",
                "key":      "cost_agent_call",
                "question": "Cost per outbound agent call (USD)? Include agent time + system cost.",
                "type":     "number",
                "default":  5.00,
                "required": False,
            },
            {
                "section":  "costs",
                "key":      "cost_voice_ivr",
                "question": "Cost per automated IVR call (USD)?",
                "type":     "number",
                "default":  1.00,
                "required": False,
            },

            # ── Section 3: Capacity ──────────────────────────────────────────
            {
                "section":  "capacity",
                "key":      "agent_call_capacity",
                "question": "How many outbound calls can your agents make per day?",
                "hint":     "Total call center capacity (all agents combined)",
                "type":     "integer",
                "default":  5000,
                "required": False,
            },
            {
                "section":  "capacity",
                "key":      "oa_referral_capacity",
                "question": "How many accounts can you refer to the collection agency per day?",
                "type":     "integer",
                "default":  1000,
                "required": False,
            },

            # ── Section 4: Compliance ────────────────────────────────────────
            {
                "section":  "compliance",
                "key":      "has_dnc_registry",
                "question": "Does your market have a Do Not Call (DNC) registry?",
                "type":     "yes_no",
                "default":  True,
                "required": True,
            },
            {
                "section":  "compliance",
                "key":      "quiet_hours_start",
                "question": "Earliest time allowed to contact customers (HH:MM, 24h)?",
                "type":     "time",
                "default":  "08:00",
                "required": True,
            },
            {
                "section":  "compliance",
                "key":      "quiet_hours_end",
                "question": "Latest time allowed to contact customers (HH:MM, 24h)?",
                "type":     "time",
                "default":  "21:00",
                "required": True,
            },
            {
                "section":  "compliance",
                "key":      "timezone",
                "question": "Customer timezone for contact hours (e.g. Asia/Bangkok, America/New_York)?",
                "type":     "text",
                "default":  "UTC",
                "required": True,
            },
            {
                "section":  "compliance",
                "key":      "max_contacts_per_week",
                "question": "Maximum contacts per customer per week (across all channels)?",
                "hint":     "Recommended: 3. Regulatory max in most markets: 7.",
                "type":     "integer",
                "default":  3,
                "required": True,
            },

            # ── Section 5: Fatigue ───────────────────────────────────────────
            {
                "section":  "fatigue",
                "key":      "sms_cooldown_days",
                "question": "Minimum days between SMS messages to same customer?",
                "type":     "integer",
                "default":  2,
                "required": False,
            },
            {
                "section":  "fatigue",
                "key":      "call_cooldown_days",
                "question": "Minimum days between agent calls to same customer?",
                "type":     "integer",
                "default":  7,
                "required": False,
            },

            # ── Section 6: Business Objective ────────────────────────────────
            {
                "section":  "objective",
                "key":      "business_objective",
                "question": "Primary business objective?",
                "options": [
                    "maximize_net_recovery",
                    "maximize_recovery_rate",
                    "minimize_cost_per_collected",
                    "balance_recovery_and_experience",
                ],
                "type":     "single_select",
                "default":  "maximize_net_recovery",
                "required": True,
            },
            {
                "section":  "objective",
                "key":      "min_balance_for_call",
                "question": "Minimum account balance to justify an agent call (USD)?",
                "hint":     "Accounts below this threshold get digital-only actions.",
                "type":     "number",
                "default":  1000.00,
                "required": False,
            },

            # ── Section 7: Settlement ────────────────────────────────────────
            {
                "section":  "settlement",
                "key":      "settlement_enabled",
                "question": "Do you offer settlement discounts to customers?",
                "type":     "yes_no",
                "default":  True,
                "required": True,
            },
            {
                "section":  "settlement",
                "key":      "max_discount_pct",
                "question": "Maximum settlement discount allowed (% of balance)?",
                "hint":     "e.g. 30 means customer can settle at 70 cents on the dollar",
                "type":     "integer",
                "default":  30,
                "required": False,
            },
            {
                "section":  "settlement",
                "key":      "min_dpd_for_settlement",
                "question": "Minimum days past due before offering settlement?",
                "hint":     "Typically 60-90 days. Too early reduces collections.",
                "type":     "integer",
                "default":  60,
                "required": False,
            },
            {
                "section":  "settlement",
                "key":      "payment_plan_months",
                "question": "Maximum number of months for a payment plan?",
                "type":     "integer",
                "default":  24,
                "required": False,
            },

            # ── Section 8: Delinquency Definition ───────────────────────────
            {
                "section":  "delinquency",
                "key":      "delay_definition",
                "question": "How is 'delay' (delinquency severity) defined in your business?",
                "options": [
                    "days_past_due_from_system",
                    "business_date_minus_due_date",
                    "billing_cycles_missed",
                    "custom",
                ],
                "type":     "single_select",
                "default":  "business_date_minus_due_date",
                "required": True,
            },
            {
                "section":  "delinquency",
                "key":      "bucket_definition",
                "question": "How are delinquency buckets defined?",
                "options": [
                    "standard_30_day_buckets",
                    "custom",
                ],
                "hint":     "Standard: B0=current, B1=1-30 DPD, B2=31-60, B3=61-90, B4=91+",
                "type":     "single_select",
                "default":  "standard_30_day_buckets",
                "required": True,
            },
        ]

    # ──────────────────────────────────────────────────────────────────────────
    # PRIVATE: CONFIG BUILDERS
    # ──────────────────────────────────────────────────────────────────────────

    def _build_action_config(self) -> Dict:
        channels = self.answered.get("channels_available", ["SMS", "EMAIL", "AGENT_CALL"])
        offers   = self.answered.get("offers_available", ["SETTLEMENT_ONE_TIME", "SETTLEMENT_PAYMENT_PLAN"])
        all_enabled = set(channels) | set(offers) | {"NO_ACTION"}

        config = {}
        for action, defaults in DEFAULTS["actions"].items():
            enabled = action in all_enabled or action == "NO_ACTION"
            cost_key = f"cost_{action.lower()}"
            cost = self.answered.get(cost_key, defaults["cost"])
            config[action] = {
                "enabled":  enabled,
                "cost":     cost,
                "channel":  defaults["channel"],
            }

        # Override capacity-limited actions
        if self.answered.get("agent_call_capacity"):
            config["AGENT_CALL"]["daily_capacity"] = self.answered["agent_call_capacity"]
        if self.answered.get("oa_referral_capacity"):
            config["OA_REFERRAL"]["daily_capacity"] = self.answered["oa_referral_capacity"]

        return config

    def _check_followup(self, key: str, value: Any) -> Optional[str]:
        """Return follow-up question text if answer triggers one."""
        if key == "has_dnc_registry" and value is True:
            return "Make sure the 'dnc' column is mapped in your schema (Do Not Call flag)."
        if key == "settlement_enabled" and value is False:
            self.answered["offers_available"] = [
                o for o in self.answered.get("offers_available", [])
                if "SETTLEMENT" not in o
            ]
            return "Settlement offers removed from action set."
        if key == "market":
            market_tz = {
                "thailand":     "Asia/Bangkok",
                "thailand":     "Asia/Bangkok",
                "us":           "America/New_York",
                "usa":          "America/New_York",
                "philippines":  "Asia/Manila",
                "singapore":    "Asia/Singapore",
                "uk":           "Europe/London",
                "indonesia":    "Asia/Jakarta",
                "vietnam":      "Asia/Ho_Chi_Minh",
                "india":        "Asia/Kolkata",
            }
            suggested_tz = market_tz.get(value.lower().strip())
            if suggested_tz:
                return f"Based on market '{value}', suggested timezone: {suggested_tz}. Confirm with 'timezone' question."
        return None

    def _default_personas(self) -> List[Dict]:
        return [
            {
                "name": "high_value_cooperative",
                "rules": {"min_balance": 5000, "max_days_past_due": 60, "min_response_rate": 0.3},
                "strategy": "premium_digital_first",
            },
            {
                "name": "high_value_unresponsive",
                "rules": {"min_balance": 5000, "min_days_past_due": 30, "max_response_rate": 0.1},
                "strategy": "escalated_call",
            },
            {
                "name": "medium_value_willing",
                "rules": {"min_balance": 1000, "max_balance": 5000, "max_days_past_due": 90},
                "strategy": "settlement_offer",
            },
            {
                "name": "medium_value_struggling",
                "rules": {"min_balance": 1000, "max_balance": 5000, "min_days_past_due": 60},
                "strategy": "payment_plan",
            },
            {
                "name": "low_value_chronic",
                "rules": {"max_balance": 1000, "min_days_past_due": 90},
                "strategy": "digital_only_or_oa",
            },
            {
                "name": "promise_keeper",
                "rules": {"active_ptp": True},
                "strategy": "ptp_follow_up",
            },
        ]
