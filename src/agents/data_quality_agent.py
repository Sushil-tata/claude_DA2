"""
DataQualityAgent
================
Task 1 in the agentic NBA pipeline. Runs before all other agents.

Reads:
  recovery.model_scores        — written by recovery-engine-v2
  recovery.data_quality_metrics — historical metrics for drift comparison

Checks:
  1. Null rate > 5% on any segmentation feature → BLOCKED
  2. Signal quadrant distribution shifts > 10% week-over-week → BLOCKED
  3. Total row count sanity check (< 50% of prior week → BLOCKED)
  4. Low-confidence (D-quadrant) rate > 60% → WARNING (not blocking)

Writes:
  recovery.data_quality_metrics — today's quality snapshot for future drift checks
"""

import logging
from typing import Optional

import pandas as pd

from agents.base_agent import AgentBlockedException, BaseAgent

logger = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────────────────────
NULL_RATE_THRESHOLD         = 0.05   # 5% — hard block
QUADRANT_DRIFT_THRESHOLD    = 0.10   # 10% absolute shift — hard block
MIN_ROW_COUNT_PCT_OF_PRIOR  = 0.50   # < 50% of prior week → block
D_QUADRANT_WARNING_THRESHOLD= 0.60   # > 60% D-quadrant → warning only

# Features where nulls are critical (segmentation inputs)
CRITICAL_FEATURES = [
    "signal_quadrant", "propensity_30d", "propensity_90d",
    "propensity_180d", "d_optimal", "erv_at_d_optimal",
    "low_confidence_flag",
]


class DataQualityAgent(BaseAgent):

    def __init__(self, execution_date: str, memory, dry_run: bool = False):
        super().__init__(
            agent_name="data_quality_agent",
            execution_date=execution_date,
            memory=memory,
            dry_run=dry_run,
        )

    def execute(self) -> dict:
        self.log(f"Reading model_scores for {self.execution_date}")
        df = self._read_scores()

        issues   = []
        warnings = []

        # ── Check 1: Null rates on critical features ──────────────────────────
        for col in CRITICAL_FEATURES:
            if col not in df.columns:
                issues.append(f"Critical column '{col}' is missing entirely")
                continue
            null_rate = df[col].isnull().mean()
            if null_rate > NULL_RATE_THRESHOLD:
                issues.append(
                    f"Column '{col}' null rate {null_rate:.1%} exceeds "
                    f"threshold {NULL_RATE_THRESHOLD:.0%}"
                )

        # ── Check 2: Quadrant distribution drift ──────────────────────────────
        current_dist = (
            df["signal_quadrant"].value_counts(normalize=True)
            .reindex(["A", "B", "C", "D"], fill_value=0.0)
            .to_dict()
        )
        prior_dist = self._get_prior_quadrant_distribution()

        if prior_dist:
            for q, current_pct in current_dist.items():
                prior_pct = prior_dist.get(q, 0.0)
                drift = abs(current_pct - prior_pct)
                if drift > QUADRANT_DRIFT_THRESHOLD:
                    issues.append(
                        f"Quadrant {q} distribution drifted {drift:.1%} "
                        f"(was {prior_pct:.1%}, now {current_pct:.1%})"
                    )

        # ── Check 3: Row count sanity ─────────────────────────────────────────
        prior_count = self._get_prior_row_count()
        if prior_count and len(df) < prior_count * MIN_ROW_COUNT_PCT_OF_PRIOR:
            issues.append(
                f"Row count {len(df):,} is less than "
                f"{MIN_ROW_COUNT_PCT_OF_PRIOR:.0%} of prior week's "
                f"{prior_count:,} rows"
            )

        # ── Check 4: D-quadrant rate warning (non-blocking) ───────────────────
        d_rate = current_dist.get("D", 0.0)
        if d_rate > D_QUADRANT_WARNING_THRESHOLD:
            warnings.append(
                f"D-quadrant accounts = {d_rate:.1%} — unusually high. "
                f"Consider reviewing segmentation model."
            )

        for w in warnings:
            self.log(w, level="warning")

        # ── Write quality metrics to Delta ────────────────────────────────────
        if not self.dry_run:
            self._write_quality_metrics(df, current_dist, issues, warnings)

        # ── BLOCK if any hard issues found ────────────────────────────────────
        if issues:
            raise AgentBlockedException(
                f"Data quality checks failed ({len(issues)} issues): "
                + "; ".join(issues),
                retry_agent="data_quality_agent",
            )

        self.log(
            f"All quality checks passed | rows={len(df):,} | "
            f"quadrants={current_dist} | warnings={len(warnings)}"
        )
        return {
            "rows_checked":          len(df),
            "quadrant_distribution": current_dist,
            "issues_found":          0,
            "warnings":              warnings,
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _read_scores(self) -> pd.DataFrame:
        df = self.memory.read("model_scores")
        if hasattr(df, "toPandas"):
            df = df.toPandas()
        if len(df) == 0:
            raise AgentBlockedException(
                f"No model scores found for execution_date={self.execution_date}. "
                "recovery-engine-v2 may not have run yet.",
                retry_agent=None,
            )
        return df

    def _get_prior_quadrant_distribution(self) -> Optional[dict]:
        """Return quadrant distribution from 7 days ago, or None if unavailable."""
        try:
            from datetime import datetime, timedelta
            prior_date = (
                datetime.strptime(self.execution_date, "%Y-%m-%d")
                - timedelta(days=7)
            ).strftime("%Y-%m-%d")

            prior_memory = self.memory.__class__(
                execution_date=prior_date,
                spark=self.memory.spark,
                local_mode=self.memory.local_mode,
            )
            prior_df = prior_memory.read("data_quality_metrics")
            if hasattr(prior_df, "toPandas"):
                prior_df = prior_df.toPandas()
            if prior_df.empty:
                return None
            row = prior_df.iloc[-1]
            return {
                "A": float(row.get("quadrant_A_pct", 0)),
                "B": float(row.get("quadrant_B_pct", 0)),
                "C": float(row.get("quadrant_C_pct", 0)),
                "D": float(row.get("quadrant_D_pct", 0)),
            }
        except Exception as e:
            self.log(f"Could not load prior distribution: {e}", level="warning")
            return None

    def _get_prior_row_count(self) -> Optional[int]:
        try:
            from datetime import datetime, timedelta
            prior_date = (
                datetime.strptime(self.execution_date, "%Y-%m-%d")
                - timedelta(days=7)
            ).strftime("%Y-%m-%d")
            prior_memory = self.memory.__class__(
                execution_date=prior_date,
                spark=self.memory.spark,
                local_mode=self.memory.local_mode,
            )
            prior_df = prior_memory.read("data_quality_metrics")
            if hasattr(prior_df, "toPandas"):
                prior_df = prior_df.toPandas()
            if prior_df.empty:
                return None
            return int(prior_df.iloc[-1].get("total_accounts", 0))
        except Exception:
            return None

    def _write_quality_metrics(
        self, df: pd.DataFrame, dist: dict, issues: list, warnings: list
    ) -> None:
        metrics = pd.DataFrame([{
            "score_date":              self.execution_date,
            "total_accounts":          len(df),
            "quadrant_A_pct":          round(dist.get("A", 0), 4),
            "quadrant_B_pct":          round(dist.get("B", 0), 4),
            "quadrant_C_pct":          round(dist.get("C", 0), 4),
            "quadrant_D_pct":          round(dist.get("D", 0), 4),
            "propensity_30d_null_pct": round(df["propensity_30d"].isnull().mean(), 4),
            "low_confidence_pct":      round(df["low_confidence_flag"].mean(), 4),
            "issues_count":            len(issues),
            "warnings_count":          len(warnings),
            "issues_detail":           "; ".join(issues) if issues else "",
        }])
        self.memory.write("data_quality_metrics", metrics)
