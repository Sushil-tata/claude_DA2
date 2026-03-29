"""
FeatureAgent
============
Task 2 in the agentic NBA pipeline. Runs after DataQualityAgent.

Reads:
  recovery.model_scores — validated scores from recovery-engine-v2

Responsibilities:
  - Selects and validates the feature columns needed downstream
  - Computes derived features: recovery_tier, erv_band, contact_priority
  - Derives SIGNAL_SEGMENT (A/B/C/D) based on data availability
  - Derives BEHAVIOURAL_PERSONA (Cooperative/Stressed/Sporadic/Disconnected)
  - Keeps willingness_score + capacity_score as model FEATURES (not segments)
  - Ensures point-in-time safety (no future-dated features)
  - Writes a clean feature_output table for ModelAgent + ConstraintAgent

Writes:
  recovery.feature_output

──────────────────────────────────────────────────────────────────────────────
CONFIGURATION INSTRUCTIONS
──────────────────────────────────────────────────────────────────────────────
BUREAU_SIGNAL_LAG_DAYS (default: 60)
    Maximum number of days since the last NCB/TUEF bureau pull for the
    bureau signal to be considered "available" for a given account.

    Why this matters:
    - SIGNAL_SEGMENT is derived from CardX signal availability + Bureau
      signal availability. If the bureau pull is older than this threshold,
      the account is treated as having NO bureau signal (Segment C or D).
    - Default of 60 days (2 months) reflects the standard NCB refresh
      cycle at CardX. Adjust if your bureau refresh cadence changes.

    How to override at runtime (Databricks task parameter):
        --bureau-signal-lag-days 90

    How to override in code:
        FeatureAgent(..., bureau_signal_lag_days=90)

    Impact of increasing this value:
    - More accounts classified as Segment A or C (bureau signal available)
    - Lower proportion of Segment D (no signal) accounts
    - Use a higher value only if bureau data is reliably refreshed more often

    Impact of decreasing this value:
    - More accounts fall to Segment D (no signal) → high-discount treatment
    - Use a lower value if bureau data quality is degrading at month 2

SIGNAL_SEGMENT definitions:
    A = CardX internal signal available  AND  Bureau signal available
    B = CardX internal signal available  AND  Bureau signal NOT available
    C = CardX internal signal NOT available  AND  Bureau signal available
    D = No CardX signal  AND  No Bureau signal  (lowest information)

    CardX signal is considered available if:
    - propensity_30d is not null (model was able to score the account)
    - score_date is not null

    Bureau signal is considered available if:
    - bureau_pull_date is present AND within BUREAU_SIGNAL_LAG_DAYS of score_date
    - OR ncb_tradeline_count > 0 (at least one tradeline returned from NCB)

BEHAVIOURAL_PERSONA definitions (derived from willingness + capacity scores):
    Cooperative   = willingness >= 0.5 AND capacity >= 0.5
    Stressed      = willingness >= 0.5 AND capacity <  0.5
    Sporadic      = willingness <  0.5 AND capacity >= 0.5
    Disconnected  = willingness <  0.5 AND capacity <  0.5

    willingness_score and capacity_score are passed as FEATURES to downstream
    models (propensity, amount, elasticity). They are NOT used as segmentation
    rules to pre-assign actions.
──────────────────────────────────────────────────────────────────────────────
"""

import logging
from datetime import datetime, timedelta

import pandas as pd

from agents.base_agent import AgentBlockedException, BaseAgent

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

# See CONFIGURATION INSTRUCTIONS above before changing this value.
DEFAULT_BUREAU_SIGNAL_LAG_DAYS = 60

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
    # Bureau signal fields (optional — used for SIGNAL_SEGMENT derivation)
    "bureau_pull_date", "ncb_tradeline_count",
    # Willingness + capacity (optional — used for BEHAVIOURAL_PERSONA derivation)
    "willingness_score", "capacity_score",
]

# ERV bands for ConstraintAgent rules (THB)
ERV_BANDS = [
    (50_000, "PREMIUM"),
    (30_000, "HIGH"),
    (10_000, "MEDIUM"),
    (0,      "LOW"),
]

# Willingness / capacity thresholds for BEHAVIOURAL_PERSONA derivation
WILLINGNESS_THRESHOLD = 0.5
CAPACITY_THRESHOLD    = 0.5


class FeatureAgent(BaseAgent):

    def __init__(
        self,
        execution_date: str,
        memory,
        dry_run: bool = False,
        bureau_signal_lag_days: int = DEFAULT_BUREAU_SIGNAL_LAG_DAYS,
    ):
        """
        Args:
            execution_date:         Scoring date (YYYY-MM-DD).
            memory:                 AgentMemory instance.
            dry_run:                If True, skip all Delta writes.
            bureau_signal_lag_days: Maximum days since last bureau pull for
                                    bureau signal to count as available.
                                    Default: 60 (2 months). See module
                                    docstring for full guidance.
        """
        super().__init__(
            agent_name="feature_agent",
            execution_date=execution_date,
            memory=memory,
            dry_run=dry_run,
        )
        self.bureau_signal_lag_days = bureau_signal_lag_days
        self.log(
            f"bureau_signal_lag_days={self.bureau_signal_lag_days} "
            f"(bureau signal valid if pull date within {self.bureau_signal_lag_days}d of score_date)"
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

        # ── Derive SIGNAL_SEGMENT ─────────────────────────────────────────────
        df["signal_segment"] = df.apply(
            lambda r: self._signal_segment(r, self.bureau_signal_lag_days), axis=1
        )

        # ── Derive BEHAVIOURAL_PERSONA ────────────────────────────────────────
        df["behavioural_persona"] = df.apply(self._behavioural_persona, axis=1)

        # ── Composite segment key (SIGNAL_SEGMENT + BEHAVIOURAL_PERSONA) ──────
        df["final_segment"] = df["signal_segment"] + "_" + df["behavioural_persona"]

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

        signal_dist  = df["signal_segment"].value_counts().to_dict()
        persona_dist = df["behavioural_persona"].value_counts().to_dict()
        self.log(
            f"Feature output ready | rows={len(df):,} | "
            f"signal_segments={signal_dist} | personas={persona_dist}"
        )
        return {
            "rows_written":         len(df),
            "feature_columns":      len(df.columns),
            "erv_band_dist":        df["erv_band"].value_counts().to_dict(),
            "recovery_tier_dist":   df["recovery_tier"].value_counts().to_dict(),
            "signal_segment_dist":  signal_dist,
            "behavioural_persona_dist": persona_dist,
        }

    # ── SIGNAL_SEGMENT derivation ─────────────────────────────────────────────

    @staticmethod
    def _has_cardx_signal(row) -> bool:
        """
        CardX internal signal is available when the propensity model was
        able to score the account (propensity_30d is not null).
        """
        return pd.notna(row.get("propensity_30d")) and pd.notna(row.get("score_date"))

    @staticmethod
    def _has_bureau_signal(row, lag_days: int) -> bool:
        """
        Bureau (NCB/TUEF) signal is available when:
          - bureau_pull_date is present AND within lag_days of score_date, OR
          - ncb_tradeline_count > 0 (at least one tradeline returned)

        lag_days is configurable via bureau_signal_lag_days (default 60).
        See module docstring for full guidance.
        """
        tradeline_count = row.get("ncb_tradeline_count", 0) or 0
        if tradeline_count > 0:
            return True

        bureau_pull = row.get("bureau_pull_date")
        score_date  = row.get("score_date")
        if pd.isna(bureau_pull) or pd.isna(score_date):
            return False

        try:
            pull_dt  = pd.Timestamp(bureau_pull)
            score_dt = pd.Timestamp(score_date)
            return (score_dt - pull_dt).days <= lag_days
        except Exception:
            return False

    @staticmethod
    def _signal_segment(row, lag_days: int) -> str:
        """
        Derives top-level signal segment:
          A = CardX signal + Bureau signal
          B = CardX signal only
          C = Bureau signal only
          D = No signal (lowest information — apply high-discount or low-cost treatment)
        """
        has_cardx  = FeatureAgent._has_cardx_signal(row)
        has_bureau = FeatureAgent._has_bureau_signal(row, lag_days)

        if has_cardx and has_bureau:
            return "A"
        if has_cardx and not has_bureau:
            return "B"
        if not has_cardx and has_bureau:
            return "C"
        return "D"

    # ── BEHAVIOURAL_PERSONA derivation ────────────────────────────────────────

    @staticmethod
    def _behavioural_persona(row) -> str:
        """
        Derives behavioural persona from willingness and capacity scores.

        IMPORTANT: willingness_score and capacity_score are used here ONLY
        to derive a stable behavioural label. They must also be passed as
        raw numeric FEATURES to the propensity, amount, and elasticity models.
        Do NOT use this persona as a decisioning rule — it is a feature/label.

          Cooperative  = willingness >= 0.5 AND capacity >= 0.5
          Stressed     = willingness >= 0.5 AND capacity <  0.5
          Sporadic     = willingness <  0.5 AND capacity >= 0.5
          Disconnected = willingness <  0.5 AND capacity <  0.5
        """
        w = row.get("willingness_score", 0.0) or 0.0
        c = row.get("capacity_score",    0.0) or 0.0

        if w >= WILLINGNESS_THRESHOLD and c >= CAPACITY_THRESHOLD:
            return "Cooperative"
        if w >= WILLINGNESS_THRESHOLD and c < CAPACITY_THRESHOLD:
            return "Stressed"
        if w < WILLINGNESS_THRESHOLD and c >= CAPACITY_THRESHOLD:
            return "Sporadic"
        return "Disconnected"

    # ── Existing derived feature logic ────────────────────────────────────────

    @staticmethod
    def _erv_band(erv: float) -> str:
        for threshold, label in ERV_BANDS:
            if erv >= threshold:
                return label
        return "LOW"

    @staticmethod
    def _recovery_tier(row) -> str:
        """Combine signal_segment + propensity_30d into a 3-tier classification.
        Uses P_1M (propensity_30d) as primary signal per design spec."""
        seg  = row.get("signal_segment", "D")
        p30  = row.get("propensity_30d", 0.0) or 0.0
        if seg in ("A", "B") and p30 >= 0.40:
            return "TIER_1_HIGH_RECOVERY"
        if seg in ("A", "B", "C") and p30 >= 0.20:
            return "TIER_2_MEDIUM_RECOVERY"
        return "TIER_3_LOW_RECOVERY"

    @staticmethod
    def _contact_priority(row) -> int:
        """1 = highest priority for outbound contact scheduling."""
        seg  = row.get("signal_segment", "D")
        erv  = row.get("erv_at_d_optimal", 0.0) or 0.0
        p30  = row.get("propensity_30d", 0.0) or 0.0
        if seg == "A" and erv >= 30_000:
            return 1
        if seg in ("A", "C") and erv >= 10_000:
            return 2
        if seg == "B":
            return 3
        return 4  # Segment D or low ERV

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
