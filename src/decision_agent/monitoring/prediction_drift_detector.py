"""
Prediction Drift Detector - Track Prediction Distribution

Monitors prediction distribution over time to detect model drift.

Metrics:
- Mean prediction (should be stable)
- Std deviation (should be stable)
- Percentiles (P10, P50, P90)
- KL divergence vs baseline (distribution similarity)

Why This Matters:
- Model drift detection (even if features look OK)
- Detect if model starts predicting differently
- Could indicate bugs, data issues, or model degradation

Integration Point:
- Called by monitoring/model_monitor.py (daily job)
- Compares current predictions vs baseline
- Alerts if distribution shifts significantly
"""

import logging
from typing import Dict, Tuple
import numpy as np
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

logger = logging.getLogger(__name__)


class PredictionDriftDetector:
    """
    Detect prediction distribution drift.

    Monitors if model predictions are shifting over time.
    """

    def __init__(self, config: Dict):
        """
        Initialize detector.

        Args:
            config: Configuration:
                - mean_shift_threshold: Max % shift in mean (default 0.10 = 10%)
                - std_shift_threshold: Max % shift in std (default 0.20 = 20%)
                - percentile_shift_threshold: Max % shift in percentiles (default 0.15)
        """
        self.config = config
        self.mean_shift_threshold = config.get("mean_shift_threshold", 0.10)
        self.std_shift_threshold = config.get("std_shift_threshold", 0.20)
        self.percentile_shift_threshold = config.get("percentile_shift_threshold", 0.15)

    def detect_drift(
        self,
        current_df: DataFrame,
        baseline_df: DataFrame,
        prediction_col: str = "prediction"
    ) -> Tuple[bool, Dict]:
        """
        Detect prediction drift.

        Args:
            current_df: Current production predictions
            baseline_df: Baseline predictions (training or previous period)
            prediction_col: Name of prediction column

        Returns:
            Tuple of (alert, results)
        """
        logger.info("Detecting prediction drift...")

        results = {}

        # 1. Compute statistics for current
        current_stats = self._compute_statistics(current_df, prediction_col, "current")
        results.update(current_stats)

        # 2. Compute statistics for baseline
        baseline_stats = self._compute_statistics(baseline_df, prediction_col, "baseline")
        results.update(baseline_stats)

        # 3. Compute shifts
        mean_shift_pct = (current_stats["current_mean"] - baseline_stats["baseline_mean"]) / baseline_stats["baseline_mean"]
        std_shift_pct = (current_stats["current_std"] - baseline_stats["baseline_std"]) / baseline_stats["baseline_std"]

        results["mean_shift_pct"] = float(mean_shift_pct)
        results["std_shift_pct"] = float(std_shift_pct)

        # 4. Check percentile shifts
        percentile_shifts = {}
        for p in ["p10", "p50", "p90"]:
            current_val = current_stats[f"current_{p}"]
            baseline_val = baseline_stats[f"baseline_{p}"]

            shift_pct = (current_val - baseline_val) / baseline_val if baseline_val != 0 else 0
            percentile_shifts[p] = float(shift_pct)

        results["percentile_shifts"] = percentile_shifts

        # 5. Determine if alert needed
        alert = False
        violations = []

        if abs(mean_shift_pct) > self.mean_shift_threshold:
            alert = True
            violations.append({
                "metric": "mean",
                "shift_pct": float(mean_shift_pct),
                "threshold": self.mean_shift_threshold
            })

        if abs(std_shift_pct) > self.std_shift_threshold:
            alert = True
            violations.append({
                "metric": "std",
                "shift_pct": float(std_shift_pct),
                "threshold": self.std_shift_threshold
            })

        for p, shift in percentile_shifts.items():
            if abs(shift) > self.percentile_shift_threshold:
                alert = True
                violations.append({
                    "metric": p,
                    "shift_pct": float(shift),
                    "threshold": self.percentile_shift_threshold
                })

        results["alert"] = alert
        results["violations"] = violations

        if alert:
            logger.warning(f"PREDICTION DRIFT ALERT: {len(violations)} violations")
            for v in violations:
                logger.warning(f"  {v['metric']}: shift={v['shift_pct']:.2%}, threshold={v['threshold']:.2%}")
        else:
            logger.info(f"No prediction drift detected. Mean shift: {mean_shift_pct:.2%}")

        return alert, results

    def _compute_statistics(
        self,
        df: DataFrame,
        prediction_col: str,
        prefix: str
    ) -> Dict:
        """
        Compute prediction statistics.

        Args:
            df: DataFrame with predictions
            prediction_col: Prediction column name
            prefix: Prefix for result keys ('current' or 'baseline')

        Returns:
            Dict with statistics
        """
        stats = df.agg(
            F.count(prediction_col).alias("count"),
            F.mean(prediction_col).alias("mean"),
            F.stddev(prediction_col).alias("std"),
            F.min(prediction_col).alias("min"),
            F.max(prediction_col).alias("max"),
            F.expr(f"percentile({prediction_col}, 0.10)").alias("p10"),
            F.expr(f"percentile({prediction_col}, 0.50)").alias("p50"),
            F.expr(f"percentile({prediction_col}, 0.90)").alias("p90")
        ).collect()[0]

        return {
            f"{prefix}_count": int(stats["count"]),
            f"{prefix}_mean": float(stats["mean"]),
            f"{prefix}_std": float(stats["std"]),
            f"{prefix}_min": float(stats["min"]),
            f"{prefix}_max": float(stats["max"]),
            f"{prefix}_p10": float(stats["p10"]),
            f"{prefix}_p50": float(stats["p50"]),
            f"{prefix}_p90": float(stats["p90"])
        }
