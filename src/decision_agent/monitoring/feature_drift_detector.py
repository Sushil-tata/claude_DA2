"""
Feature Drift Detector - PSI (Population Stability Index)

Detects feature distribution drift over time using PSI metric.

PSI Formula:
PSI = Σ (actual_pct - expected_pct) * ln(actual_pct / expected_pct)

PSI Thresholds:
- PSI < 0.1: No significant drift
- 0.1 ≤ PSI < 0.25: Moderate drift (investigate)
- PSI ≥ 0.25: Significant drift (retrain model)

Why This Matters:
- Feature drift → model drift
- Detect data quality issues early
- Trigger retraining when needed

Integration Point:
- Called by monitoring/model_monitor.py (daily job)
- Compares current features vs training baseline
- Alerts if PSI exceeds threshold
"""

import logging
from typing import Dict, List, Tuple
import numpy as np
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

logger = logging.getLogger(__name__)


class FeatureDriftDetector:
    """
    Detect feature distribution drift using PSI.

    PSI is the standard metric for monitoring feature stability in production.
    """

    def __init__(self, config: Dict):
        """
        Initialize detector.

        Args:
            config: Configuration:
                - psi_threshold: Alert threshold (default 0.25)
                - num_bins: Number of bins for PSI calculation (default 10)
                - min_samples: Minimum samples required (default 100)
        """
        self.config = config
        self.psi_threshold = config.get("psi_threshold", 0.25)
        self.num_bins = config.get("num_bins", 10)
        self.min_samples = config.get("min_samples", 100)

    def detect_drift(
        self,
        current_df: DataFrame,
        baseline_df: DataFrame,
        feature_cols: List[str]
    ) -> Tuple[bool, Dict]:
        """
        Detect drift across all features.

        Args:
            current_df: Current production data
            baseline_df: Baseline (training) data
            feature_cols: List of feature columns to monitor

        Returns:
            Tuple of (alert, results)
            - alert: True if any feature exceeds threshold
            - results: Dict with PSI scores per feature
        """
        logger.info(f"Detecting feature drift for {len(feature_cols)} features...")

        results = {
            "feature_psi_scores": {},
            "features_above_threshold": [],
            "max_psi": 0.0,
            "alert": False
        }

        for feature_col in feature_cols:
            if feature_col not in current_df.columns or feature_col not in baseline_df.columns:
                logger.warning(f"Feature '{feature_col}' not found. Skipping.")
                continue

            # Compute PSI for this feature
            psi_score = self._compute_psi_spark(
                current_df, baseline_df, feature_col
            )

            results["feature_psi_scores"][feature_col] = float(psi_score)

            if psi_score > results["max_psi"]:
                results["max_psi"] = float(psi_score)

            if psi_score >= self.psi_threshold:
                results["features_above_threshold"].append({
                    "feature": feature_col,
                    "psi": float(psi_score),
                    "threshold": self.psi_threshold
                })

        # Set alert if any feature exceeds threshold
        results["alert"] = len(results["features_above_threshold"]) > 0

        if results["alert"]:
            logger.warning(
                f"DRIFT ALERT: {len(results['features_above_threshold'])} features above threshold"
            )
            for item in results["features_above_threshold"]:
                logger.warning(f"  {item['feature']}: PSI={item['psi']:.3f}")
        else:
            logger.info(f"No drift detected. Max PSI: {results['max_psi']:.3f}")

        return results["alert"], results

    def _compute_psi_spark(
        self,
        current_df: DataFrame,
        baseline_df: DataFrame,
        feature_col: str
    ) -> float:
        """
        Compute PSI for a single feature using Spark.

        PSI = Σ (actual% - expected%) * ln(actual% / expected%)
        """
        # Get percentiles from baseline to define bins
        percentiles = [i / self.num_bins for i in range(self.num_bins + 1)]

        baseline_percentiles = baseline_df.stat.approxQuantile(
            feature_col, percentiles, 0.01
        )

        # Handle edge case: all same value
        if len(set(baseline_percentiles)) == 1:
            logger.warning(f"Feature '{feature_col}' has constant value. PSI = 0")
            return 0.0

        # Create bins using baseline percentiles
        # Add -inf and +inf to capture all values
        bins = [-float('inf')] + baseline_percentiles[1:-1] + [float('inf')]

        # Count samples in each bin for baseline
        baseline_counts = self._bin_counts(baseline_df, feature_col, bins)
        baseline_total = sum(baseline_counts)

        if baseline_total < self.min_samples:
            logger.warning(f"Insufficient baseline samples for '{feature_col}'. PSI = 0")
            return 0.0

        baseline_pcts = [count / baseline_total for count in baseline_counts]

        # Count samples in each bin for current
        current_counts = self._bin_counts(current_df, feature_col, bins)
        current_total = sum(current_counts)

        if current_total < self.min_samples:
            logger.warning(f"Insufficient current samples for '{feature_col}'. PSI = 0")
            return 0.0

        current_pcts = [count / current_total for count in current_counts]

        # Compute PSI
        psi = 0.0
        for current_pct, baseline_pct in zip(current_pcts, baseline_pcts):
            # Handle zero percentages (add small epsilon)
            if current_pct == 0:
                current_pct = 0.0001
            if baseline_pct == 0:
                baseline_pct = 0.0001

            psi += (current_pct - baseline_pct) * np.log(current_pct / baseline_pct)

        return float(psi)

    def _bin_counts(
        self,
        df: DataFrame,
        feature_col: str,
        bins: List[float]
    ) -> List[int]:
        """
        Count samples in each bin.

        Args:
            df: DataFrame
            feature_col: Feature column name
            bins: List of bin edges (including -inf and +inf)

        Returns:
            List of counts per bin
        """
        counts = []

        for i in range(len(bins) - 1):
            lower = bins[i]
            upper = bins[i + 1]

            count = df.filter(
                (F.col(feature_col) > lower) & (F.col(feature_col) <= upper)
            ).count()

            counts.append(count)

        return counts


def compute_psi(
    current_df: DataFrame,
    baseline_df: DataFrame,
    feature_col: str,
    num_bins: int = 10
) -> float:
    """
    Convenience function to compute PSI for one feature.

    Args:
        current_df: Current data
        baseline_df: Baseline data
        feature_col: Feature column name
        num_bins: Number of bins for PSI

    Returns:
        PSI score
    """
    detector = FeatureDriftDetector({"num_bins": num_bins})
    return detector._compute_psi_spark(current_df, baseline_df, feature_col)
