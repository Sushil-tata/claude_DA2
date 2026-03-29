"""
Stability Monitor - Prediction Stability vs Champion

Ensures Challenger predictions are stable relative to Champion.
Prevents "prediction cliffs" where model behavior changes dramatically.

Why This Matters:
- Downstream systems expect consistent score interpretation
- Large prediction shifts cause operational disruption
- Stability = trust for business stakeholders

Quality Gate:
- Challenger MUST maintain high correlation with Champion (> 0.85)
- Mean prediction shift must be < 5%
- No individual segment should have correlation < 0.75

Integration Point:
- Called by champion_challenger.py during comparison
- Also used in production monitoring (compare current vs baseline)
"""

import logging
from typing import Dict, Tuple
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F
import numpy as np

logger = logging.getLogger(__name__)


class StabilityMonitor:
    """
    Monitor prediction stability between two models or time periods.

    Used for:
    1. Champion/Challenger validation (pre-deployment)
    2. Production monitoring (detect model drift)
    """

    def __init__(self, config: Dict[str, float]):
        """
        Initialize monitor with stability thresholds.

        Args:
            config: Stability configuration:
                - min_correlation: Minimum Pearson correlation (default 0.85)
                - min_rank_correlation: Minimum Spearman rank correlation (default 0.80)
                - max_mean_shift_pct: Maximum mean prediction shift % (default 0.05)
                - max_segment_divergence: Max correlation drop in any segment (default 0.75)
        """
        self.config = config
        self.min_correlation = config.get("min_correlation", 0.85)
        self.min_rank_correlation = config.get("min_rank_correlation", 0.80)
        self.max_mean_shift_pct = config.get("max_mean_shift_pct", 0.05)
        self.max_segment_divergence = config.get("max_segment_divergence", 0.75)

    def evaluate_stability(
        self,
        df: DataFrame,
        model_a_col: str,
        model_b_col: str,
        segment_cols: list = None,
        label_col: str = None
    ) -> Tuple[bool, Dict]:
        """
        Evaluate prediction stability between two models.

        Args:
            df: DataFrame with predictions from both models
            model_a_col: Column name for Model A predictions (e.g., Champion)
            model_b_col: Column name for Model B predictions (e.g., Challenger)
            segment_cols: Optional list of segment columns to analyze
            label_col: Optional true label column for ground truth comparison

        Returns:
            Tuple of (passed, results)
            - passed: True if stability checks pass
            - results: Detailed stability metrics
        """
        logger.info(f"Evaluating stability between {model_a_col} and {model_b_col}...")

        results = {
            "model_a": model_a_col,
            "model_b": model_b_col,
            "stability_checks": {}
        }

        # 1. Overall Correlation
        overall_passed, overall_results = self._check_overall_correlation(
            df, model_a_col, model_b_col
        )
        results["stability_checks"]["overall_correlation"] = overall_results

        # 2. Mean Prediction Shift
        shift_passed, shift_results = self._check_mean_shift(
            df, model_a_col, model_b_col
        )
        results["stability_checks"]["mean_shift"] = shift_results

        # 3. Distribution Similarity
        dist_passed, dist_results = self._check_distribution_similarity(
            df, model_a_col, model_b_col
        )
        results["stability_checks"]["distribution_similarity"] = dist_results

        # 4. Segment-Level Stability (if segments provided)
        segment_passed = True
        if segment_cols:
            segment_passed, segment_results = self._check_segment_stability(
                df, model_a_col, model_b_col, segment_cols
            )
            results["stability_checks"]["segment_stability"] = segment_results

        # 5. Rank Order Stability
        rank_passed, rank_results = self._check_rank_stability(
            df, model_a_col, model_b_col
        )
        results["stability_checks"]["rank_stability"] = rank_results

        # Overall pass/fail
        all_passed = (
            overall_passed and
            shift_passed and
            dist_passed and
            segment_passed and
            rank_passed
        )

        results["passed"] = all_passed
        results["stability_score"] = self._compute_overall_stability_score(results)

        if not all_passed:
            logger.warning(f"Stability evaluation FAILED. Issues detected.")
        else:
            logger.info(f"Stability evaluation PASSED. Score: {results['stability_score']:.3f}")

        return all_passed, results

    def _check_overall_correlation(
        self,
        df: DataFrame,
        col_a: str,
        col_b: str
    ) -> Tuple[bool, Dict]:
        """Check Pearson correlation between predictions."""
        correlation = df.stat.corr(col_a, col_b)

        results = {
            "pearson_correlation": float(correlation),
            "threshold": self.min_correlation,
            "passed": correlation >= self.min_correlation
        }

        logger.info(f"  Overall correlation: {correlation:.3f} (threshold: {self.min_correlation})")

        return results["passed"], results

    def _check_mean_shift(
        self,
        df: DataFrame,
        col_a: str,
        col_b: str
    ) -> Tuple[bool, Dict]:
        """Check if mean prediction shifted significantly."""
        stats = df.agg(
            F.mean(col_a).alias("mean_a"),
            F.mean(col_b).alias("mean_b")
        ).collect()[0]

        mean_a = stats["mean_a"]
        mean_b = stats["mean_b"]

        # Compute shift percentage
        shift_pct = (mean_b - mean_a) / mean_a if mean_a != 0 else 0
        abs_shift_pct = abs(shift_pct)

        results = {
            "mean_a": float(mean_a),
            "mean_b": float(mean_b),
            "shift_pct": float(shift_pct),
            "abs_shift_pct": float(abs_shift_pct),
            "threshold": self.max_mean_shift_pct,
            "passed": abs_shift_pct <= self.max_mean_shift_pct
        }

        logger.info(f"  Mean shift: {shift_pct:.2%} (threshold: {self.max_mean_shift_pct:.2%})")

        return results["passed"], results

    def _check_distribution_similarity(
        self,
        df: DataFrame,
        col_a: str,
        col_b: str
    ) -> Tuple[bool, Dict]:
        """
        Check if prediction distributions are similar.

        Uses percentile comparison (P10, P25, P50, P75, P90).
        """
        # Compute percentiles for both models
        percentiles = [0.10, 0.25, 0.50, 0.75, 0.90]
        percentile_names = ["p10", "p25", "p50", "p75", "p90"]

        stats = df.agg(
            *[F.expr(f"percentile({col_a}, {p})").alias(f"a_{name}")
              for p, name in zip(percentiles, percentile_names)],
            *[F.expr(f"percentile({col_b}, {p})").alias(f"b_{name}")
              for p, name in zip(percentiles, percentile_names)]
        ).collect()[0]

        results = {
            "model_a_percentiles": {},
            "model_b_percentiles": {},
            "max_percentile_diff_pct": 0.0
        }

        max_diff = 0.0
        for name in percentile_names:
            val_a = stats[f"a_{name}"]
            val_b = stats[f"b_{name}"]

            results["model_a_percentiles"][name] = float(val_a)
            results["model_b_percentiles"][name] = float(val_b)

            # Compute percentage difference
            diff_pct = abs(val_b - val_a) / val_a if val_a != 0 else 0
            max_diff = max(max_diff, diff_pct)

        results["max_percentile_diff_pct"] = float(max_diff)
        # Allow up to 10% difference in any percentile
        results["passed"] = max_diff <= 0.10

        logger.info(f"  Distribution similarity: max diff = {max_diff:.2%}")

        return results["passed"], results

    def _check_segment_stability(
        self,
        df: DataFrame,
        col_a: str,
        col_b: str,
        segment_cols: list
    ) -> Tuple[bool, Dict]:
        """
        Check stability within each segment.

        Ensures no segment has severe correlation drop.
        """
        results = {}
        all_passed = True

        for segment in segment_cols:
            if segment not in df.columns:
                logger.warning(f"Segment column '{segment}' not found. Skipping.")
                continue

            # Compute correlation by segment
            from decision_agent.utils.spark_guards import safe_to_pandas

            # Sample for correlation computation (expensive on full data)
            segment_sample = safe_to_pandas(
                df.select(segment, col_a, col_b).sample(fraction=0.1),
                max_rows=10000
            )

            segment_correlations = {}
            min_segment_corr = 1.0

            for seg_value in segment_sample[segment].unique():
                seg_data = segment_sample[segment_sample[segment] == seg_value]

                if len(seg_data) < 30:  # Need minimum samples
                    continue

                corr = seg_data[[col_a, col_b]].corr().iloc[0, 1]
                segment_correlations[str(seg_value)] = float(corr)
                min_segment_corr = min(min_segment_corr, corr)

            results[segment] = {
                "segment_correlations": segment_correlations,
                "min_correlation": float(min_segment_corr),
                "threshold": self.max_segment_divergence,
                "passed": min_segment_corr >= self.max_segment_divergence
            }

            if not results[segment]["passed"]:
                all_passed = False
                logger.warning(
                    f"  Segment '{segment}' stability issue: min_corr={min_segment_corr:.3f}"
                )

        return all_passed, results

    def _check_rank_stability(
        self,
        df: DataFrame,
        col_a: str,
        col_b: str
    ) -> Tuple[bool, Dict]:
        """
        Check rank order stability (Spearman correlation).

        Important for ranking-based decisions (top-K customers).
        """
        # Add rank columns
        window_spec_a = Window.orderBy(col_a)
        window_spec_b = Window.orderBy(col_b)

        df_with_ranks = df.withColumn("rank_a", F.row_number().over(window_spec_a)) \
                          .withColumn("rank_b", F.row_number().over(window_spec_b))

        # Compute rank correlation (Pearson on ranks = Spearman)
        rank_correlation = df_with_ranks.stat.corr("rank_a", "rank_b")

        results = {
            "rank_correlation": float(rank_correlation),
            "threshold": self.min_rank_correlation,
            "passed": rank_correlation >= self.min_rank_correlation
        }

        logger.info(f"  Rank correlation: {rank_correlation:.3f} (threshold: {self.min_rank_correlation})")

        return results["passed"], results

    def _compute_overall_stability_score(self, results: Dict) -> float:
        """
        Compute overall stability score [0, 1].

        Weighted average of individual checks.
        """
        checks = results["stability_checks"]

        # Extract scores (convert pass/fail to 1/0, or use actual values)
        corr_score = checks["overall_correlation"]["pearson_correlation"]

        shift_score = 1.0 - min(
            checks["mean_shift"]["abs_shift_pct"] / self.max_mean_shift_pct,
            1.0
        )

        rank_score = checks["rank_stability"]["rank_correlation"]

        # Weighted average
        stability_score = (
            0.4 * corr_score +
            0.3 * shift_score +
            0.3 * rank_score
        )

        return float(min(max(stability_score, 0.0), 1.0))


def evaluate_stability(
    df: DataFrame,
    model_a_col: str,
    model_b_col: str,
    config: Dict = None
) -> Tuple[bool, Dict]:
    """
    Convenience function to evaluate stability.

    Args:
        df: DataFrame with predictions
        model_a_col: First model prediction column
        model_b_col: Second model prediction column
        config: Stability configuration

    Returns:
        Tuple of (passed, results)
    """
    if config is None:
        config = {}

    monitor = StabilityMonitor(config)
    return monitor.evaluate_stability(df, model_a_col, model_b_col)
