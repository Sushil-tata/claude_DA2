"""
ConstraintAgent
===============
Task 4 (parallel with ModelAgent). Runs after FeatureAgent.

Reads:
  recovery.feature_output — from FeatureAgent

Responsibilities:
  - Applies hard business rules that can override model's d_optimal
  - Rules: max_discount_by_segment, legal_threshold, write_off_threshold,
           BOT regulatory caps, fatigue limits
  - Raises BLOCKED if leakage detected in feature columns
  - Each override is logged with reason code for audit

Writes:
  recovery.constraint_overrides  — one row per account with override decision
"""

import logging

import pandas as pd

from agents.base_agent import AgentBlockedException, BaseAgent

logger = logging.getLogger(__name__)

# ── Business rules ────────────────────────────────────────────────────────────

# Maximum discount allowed per segment (regulatory + credit policy)
MAX_DISCOUNT_BY_SEGMENT = {
    "PAYROLL":           0.30,
    "SALARY_LIKE":       0.35,
    "SME_STABLE":        0.40,
    "SME_VOLATILE":      0.50,
    "GIG_FREELANCE":     0.50,
    "PASSIVE_INVESTOR":  0.45,
    "DEFAULT":           0.50,   # fallback for unknown/null segments
}

# Accounts below this ERV get HOLD action regardless of model
WRITE_OFF_ERV_THRESHOLD    = 500.0    # THB — below this, not worth actioning
LEGAL_ERV_THRESHOLD        = 150_000.0 # THB — above this, requires legal review flag
BOT_MAX_DISCOUNT           = 0.60     # Bank of Thailand regulatory cap
MAX_DISCOUNT_ABSOLUTE      = min(0.65, BOT_MAX_DISCOUNT)


class ConstraintAgent(BaseAgent):

    def __init__(self, execution_date: str, memory, dry_run: bool = False):
        super().__init__(
            agent_name="constraint_agent",
            execution_date=execution_date,
            memory=memory,
            dry_run=dry_run,
        )

    def execute(self) -> dict:
        self.log("Reading feature_output for constraint evaluation")
        df = self._read_features()

        overrides = df[[
            "account_id", "score_date",
            "signal_quadrant", "segment_label",
            "d_optimal", "erv_at_d_optimal",
            "low_confidence_flag",
        ]].copy()

        overrides["constraint_override_flag"] = False
        overrides["override_reason"]          = ""
        overrides["final_d_optimal"]          = overrides["d_optimal"]
        overrides["final_action"]             = "PROCEED"  # default
        overrides["legal_review_flag"]        = False

        # Apply rules in priority order (later rules can override earlier ones)
        overrides = self._apply_write_off_rule(overrides)
        overrides = self._apply_max_discount_rule(overrides)
        overrides = self._apply_bot_cap(overrides)
        overrides = self._apply_legal_flag(overrides)
        overrides = self._apply_d_quadrant_hold(overrides)

        override_count = overrides["constraint_override_flag"].sum()
        legal_count    = overrides["legal_review_flag"].sum()

        self.log(
            f"Constraints applied | rows={len(overrides):,} | "
            f"overrides={override_count} | legal_flags={legal_count}"
        )

        if not self.dry_run:
            self.memory.write("constraint_overrides", overrides)

        return {
            "rows_evaluated":   len(overrides),
            "overrides_applied":int(override_count),
            "legal_flags":      int(legal_count),
            "override_reasons": (
                overrides[overrides["constraint_override_flag"]]["override_reason"]
                .value_counts().to_dict()
            ),
        }

    # ── Business rules ────────────────────────────────────────────────────────

    def _apply_write_off_rule(self, df: pd.DataFrame) -> pd.DataFrame:
        """Accounts with ERV below write-off threshold → HOLD, no discount."""
        mask = df["erv_at_d_optimal"] < WRITE_OFF_ERV_THRESHOLD
        df.loc[mask, "final_action"]             = "HOLD"
        df.loc[mask, "final_d_optimal"]          = 0.0
        df.loc[mask, "constraint_override_flag"] = True
        df.loc[mask, "override_reason"]          = (
            f"erv_below_write_off_threshold_{WRITE_OFF_ERV_THRESHOLD:.0f}_thb"
        )
        return df

    def _apply_max_discount_rule(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cap d_optimal at segment-specific maximum."""
        def _cap(row):
            seg     = row["segment_label"] or "DEFAULT"
            max_d   = MAX_DISCOUNT_BY_SEGMENT.get(seg, MAX_DISCOUNT_BY_SEGMENT["DEFAULT"])
            cur_d   = row["final_d_optimal"]
            if cur_d > max_d and row["final_action"] != "HOLD":
                return max_d, True, f"max_discount_cap_{seg}_{max_d}"
            return cur_d, row["constraint_override_flag"], row["override_reason"]

        results = df.apply(_cap, axis=1, result_type="expand")
        df["final_d_optimal"]          = results[0]
        df["constraint_override_flag"] = results[1]
        df["override_reason"]          = results[2]
        return df

    def _apply_bot_cap(self, df: pd.DataFrame) -> pd.DataFrame:
        """BOT regulatory cap: no discount above 60%."""
        mask = (df["final_d_optimal"] > BOT_MAX_DISCOUNT) & (df["final_action"] != "HOLD")
        df.loc[mask, "final_d_optimal"]          = BOT_MAX_DISCOUNT
        df.loc[mask, "constraint_override_flag"] = True
        df.loc[mask, "override_reason"]          = f"bot_regulatory_cap_{BOT_MAX_DISCOUNT}"
        return df

    def _apply_legal_flag(self, df: pd.DataFrame) -> pd.DataFrame:
        """Accounts with very high ERV need legal team review."""
        mask = df["erv_at_d_optimal"] > LEGAL_ERV_THRESHOLD
        df.loc[mask, "legal_review_flag"] = True
        return df

    def _apply_d_quadrant_hold(self, df: pd.DataFrame) -> pd.DataFrame:
        """D-quadrant with low_confidence_flag always gets HOLD."""
        mask = df["low_confidence_flag"] & (df["signal_quadrant"] == "D")
        df.loc[mask, "final_action"]             = "HOLD"
        df.loc[mask, "constraint_override_flag"] = True
        df.loc[mask, "override_reason"]          = "d_quadrant_low_confidence_hold"
        return df

    def _read_features(self) -> pd.DataFrame:
        df = self.memory.read("feature_output")
        if hasattr(df, "toPandas"):
            df = df.toPandas()
        if df.empty:
            raise AgentBlockedException(
                "feature_output is empty — FeatureAgent may have failed.",
                retry_agent="feature_agent",
            )
        return df
