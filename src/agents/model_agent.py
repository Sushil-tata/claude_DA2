"""
ModelAgent
==========
Task 3 (parallel with ConstraintAgent). Runs after FeatureAgent.

Reads:
  recovery.feature_output — from FeatureAgent

Responsibilities:
  - Selects d_optimal and erv_at_d_optimal per account (already computed
    by recovery-engine-v2 — this agent does NOT re-score)
  - Identifies the optimal discount level from the elasticity grid
  - Adds recommended_channel based on quadrant + recovery_tier
  - Passes clean model selections to DecisionAgent

Writes:
  recovery.model_agent_output
"""

import logging

import pandas as pd

from agents.base_agent import AgentBlockedException, BaseAgent

logger = logging.getLogger(__name__)

# Channel recommendation rules by quadrant
CHANNEL_MAP = {
    "A": "DIGITAL_PLUS_CALL",    # High propensity + High ERV → digital nudge + agent call
    "B": "DIGITAL_ONLY",         # High propensity + Low ERV  → self-serve digital
    "C": "AGENT_CALL",           # Low propensity  + High ERV → agent escalation
    "D": "HOLD",                 # Low propensity  + Low ERV  → no contact
}

TIMING_MAP = {
    "TIER_1_HIGH_RECOVERY": "WITHIN_24H",
    "TIER_2_MEDIUM_RECOVERY": "WITHIN_72H",
    "TIER_3_LOW_RECOVERY":  "WITHIN_7D",
}


class ModelAgent(BaseAgent):

    def __init__(self, execution_date: str, memory, dry_run: bool = False):
        super().__init__(
            agent_name="model_agent",
            execution_date=execution_date,
            memory=memory,
            dry_run=dry_run,
        )

    def execute(self) -> dict:
        self.log("Reading feature_output")
        df = self._read_features()

        # ── Select model outputs (no re-scoring) ─────────────────────────────
        output = df[[
            "account_id", "score_date",
            "signal_quadrant", "segment_label",
            "propensity_180d", "low_confidence_flag",
            "d_optimal", "erv_at_d_optimal",
            "erv_band", "recovery_tier", "contact_priority",
            "model_version", "experiment_id",
        ]].copy()

        # ── Add channel + timing recommendations ──────────────────────────────
        output["recommended_channel"] = (
            output["signal_quadrant"].map(CHANNEL_MAP).fillna("HOLD")
        )
        output["contact_timing"] = (
            output["recovery_tier"].map(TIMING_MAP).fillna("WITHIN_7D")
        )

        # ── Override channel for D-quadrant — never contact ───────────────────
        output.loc[output["low_confidence_flag"], "recommended_channel"] = "HOLD"
        output.loc[output["low_confidence_flag"], "contact_timing"]      = "NO_CONTACT"

        # ── Flag accounts that need constraint check ───────────────────────────
        output["needs_constraint_check"] = (
            (output["d_optimal"] > 0.45) |
            (output["erv_at_d_optimal"] > 50_000) |
            (output["signal_quadrant"] == "C")
        )

        if not self.dry_run:
            self.memory.write("model_agent_output", output)

        channel_dist = output["recommended_channel"].value_counts().to_dict()
        self.log(f"Model selections ready | rows={len(output):,} | channels={channel_dist}")

        return {
            "rows_written":              len(output),
            "channel_distribution":      channel_dist,
            "needs_constraint_check":    int(output["needs_constraint_check"].sum()),
        }

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
