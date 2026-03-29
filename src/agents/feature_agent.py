"""
FeatureAgent
============
Task 2 in the agentic NBA pipeline. Runs after DataQualityAgent.

Reads:
  recovery.model_scores — validated scores from recovery-engine-v2

Responsibilities:
  - Selects and validates the feature columns needed downstream
  - Computes derived features: recovery_tier, erv_band, contact_priority
  - Ensures point-in-time safety (no future-dated features)
  - Writes a clean feature_output table for ModelAgent + ConstraintAgent

Writes:
  recovery.feature_output
"""

import logging

import pandas as pd

from agents.base_agent import AgentBlockedException, BaseAgent

logger = logging.getLogger(__name__)

# Columns passed downstream — explicit allowlist
FEATURE_COLUMNS = [
    "account_id", "score_date",
    "signal_quadrant", "segment_label",
    "propensity_30d", "propensity_90d", "propensity_180d",
    "low_confidence_flag",
    "d_optimal", "erv_at_d_optimal",
    "erv_d20", "erv_d30", "erv_d40", "erv_d50", "erv_d60",
    "p_accept_d20", "p_accept_d30", "p_accept_d40", "p_accept_d50", "p_accept_d60",
    "model_version", "experiment_id",
]

# ERV bands for ConstraintAgent rules (THB)
ERV_BANDS = [
    (50_000, "PREMIUM"),
    (30_000, "HIGH"),
    (10_000, "MEDIUM"),
    (0,      "LOW"),
]


class FeatureAgent(BaseAgent):

    def __init__(self, execution_date: str, memory, dry_run: bool = False):
        super().__init__(
            agent_name="feature_agent",
            execution_date=execution_date,
            memory=memory,
            dry_run=dry_run,
        )

    def execute(self) -> dict:
        self.log("Reading validated model_scores")
        df = self._read_scores()

        # ── Select feature columns (drop anything not in allowlist) ───────────
        available = [c for c in FEATURE_COLUMNS if c in df.columns]
        missing   = [c for c in FEATURE_COLUMNS if c not in df.columns]
        if missing:
            self.log(f"Optional columns not present: {missing}", level="warning")

        df = df[available].copy()

        # ── Derive computed features ──────────────────────────────────────────
        df["erv_band"]         = df["erv_at_d_optimal"].apply(self._erv_band)
        df["recovery_tier"]    = df.apply(self._recovery_tier, axis=1)
        df["contact_priority"] = df.apply(self._contact_priority, axis=1)

        # ── Point-in-time safety check ────────────────────────────────────────
        if "score_date" in df.columns:
            future_rows = df[df["score_date"] > self.execution_date]
            if len(future_rows) > 0:
                raise AgentBlockedException(
                    f"PIT violation: {len(future_rows)} rows have "
                    f"score_date > execution_date ({self.execution_date}). "
                    "Possible data leakage.",
                    retry_agent="data_quality_agent",
                )

        if not self.dry_run:
            self.memory.write("feature_output", df)

        self.log(f"Feature output ready | rows={len(df):,} | derived_cols=3")
        return {
            "rows_written":      len(df),
            "feature_columns":   len(df.columns),
            "erv_band_dist":     df["erv_band"].value_counts().to_dict(),
            "recovery_tier_dist":df["recovery_tier"].value_counts().to_dict(),
        }

    # ── Derived feature logic ─────────────────────────────────────────────────

    @staticmethod
    def _erv_band(erv: float) -> str:
        for threshold, label in ERV_BANDS:
            if erv >= threshold:
                return label
        return "LOW"

    @staticmethod
    def _recovery_tier(row) -> str:
        """Combine quadrant + propensity into a 3-tier recovery classification."""
        q   = row.get("signal_quadrant", "D")
        p180 = row.get("propensity_180d", 0.0)
        if q in ("A", "B") and p180 >= 0.40:
            return "TIER_1_HIGH_RECOVERY"
        if q in ("A", "B", "C") and p180 >= 0.20:
            return "TIER_2_MEDIUM_RECOVERY"
        return "TIER_3_LOW_RECOVERY"

    @staticmethod
    def _contact_priority(row) -> int:
        """1 = highest priority for outbound contact scheduling."""
        q    = row.get("signal_quadrant", "D")
        erv  = row.get("erv_at_d_optimal", 0.0) or 0.0
        p180 = row.get("propensity_180d", 0.0) or 0.0
        if q == "A" and erv >= 30_000:
            return 1
        if q in ("A", "C") and erv >= 10_000:
            return 2
        if q == "B":
            return 3
        return 4  # D-quadrant or low ERV

    def _read_scores(self) -> pd.DataFrame:
        df = self.memory.read("model_scores")
        if hasattr(df, "toPandas"):
            df = df.toPandas()
        if df.empty:
            raise AgentBlockedException(
                "model_scores is empty — DataQualityAgent may have been skipped.",
                retry_agent="data_quality_agent",
            )
        return df
