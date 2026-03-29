"""
Deposit Stability Calculator - Income Volatility Signals

Computes robust statistics to measure deposit amount stability:
- CV (Coefficient of Variation): std / mean
- MAD (Median Absolute Deviation): Robust alternative to std
- Trimmed Mean: Mean after removing top/bottom 10% outliers

Why This Matters:
- Stable deposits (low CV) → salaried employee → predictable income
- Volatile deposits (high CV) → gig worker / commission → unpredictable income
- Useful for risk assessment in underwriting

Integration Point:
- Called by features/income_features.py after deposit_periodicity
- Outputs: deposit_cv, deposit_mad, deposit_trimmed_mean, deposit_stability_score
"""

import logging
from typing import Dict
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

logger = logging.getLogger(__name__)


class DepositStabilityCalculator:
    """
    Calculate deposit stability metrics using robust statistics.

    Robust statistics are critical for financial data which often has outliers
    (bonuses, tax refunds, one-time payments).
    """

    def __init__(
        self,
        min_deposit_amount: float = 100.0,
        lookback_days: int = 180,
        min_deposits_required: int = 3,
        trim_fraction: float = 0.1
    ):
        """
        Initialize calculator.

        Args:
            min_deposit_amount: Minimum deposit to consider
            lookback_days: Days of history to analyze
            min_deposits_required: Minimum deposits needed
            trim_fraction: Fraction to trim from each tail for trimmed mean (e.g., 0.1 = 10%)
        """
        self.min_deposit_amount = min_deposit_amount
        self.lookback_days = lookback_days
        self.min_deposits_required = min_deposits_required
        self.trim_fraction = trim_fraction

    def compute_stability_metrics(
        self,
        df: DataFrame,
        entity_key: str = "customer_id",
        timestamp_col: str = "transaction_timestamp",
        amount_col: str = "transaction_amount"
    ) -> DataFrame:
        """
        Compute deposit stability metrics for each customer.

        Args:
            df: Spark DataFrame with transaction data
            entity_key: Customer identifier column
            timestamp_col: Transaction timestamp column
            amount_col: Transaction amount column

        Returns:
            DataFrame with one row per customer containing:
            - deposit_mean: Mean deposit amount
            - deposit_std: Standard deviation of deposits
            - deposit_cv: Coefficient of variation (std / mean)
            - deposit_mad: Median Absolute Deviation
            - deposit_median: Median deposit amount
            - deposit_trimmed_mean: Mean after trimming outliers
            - deposit_min: Minimum deposit
            - deposit_max: Maximum deposit
            - deposit_p25: 25th percentile
            - deposit_p75: 75th percentile
            - deposit_count: Number of deposits
            - deposit_stability_score: Overall stability score [0, 1]
        """
        logger.info("Computing deposit stability metrics...")

        # Filter to deposits (positive amounts above threshold)
        deposits = df.filter(
            (F.col(amount_col) >= self.min_deposit_amount)
        ).select(
            entity_key,
            timestamp_col,
            amount_col
        )

        # Aggregate basic statistics per customer
        deposit_stats = deposits.groupBy(entity_key).agg(
            F.count("*").alias("deposit_count"),
            F.mean(amount_col).alias("deposit_mean"),
            F.stddev(amount_col).alias("deposit_std"),
            F.expr(f"percentile({amount_col}, 0.5)").alias("deposit_median"),
            F.min(amount_col).alias("deposit_min"),
            F.max(amount_col).alias("deposit_max"),
            F.expr(f"percentile({amount_col}, 0.25)").alias("deposit_p25"),
            F.expr(f"percentile({amount_col}, 0.75)").alias("deposit_p75")
        )

        # Compute derived metrics
        deposit_stats = deposit_stats.withColumn(
            "deposit_cv",
            F.when(
                F.col("deposit_mean") > 0,
                F.col("deposit_std") / F.col("deposit_mean")
            ).otherwise(F.lit(None))
        )

        # Compute MAD (Median Absolute Deviation) using window function
        # MAD is more robust to outliers than standard deviation
        deposit_stats = self._compute_mad(deposits, deposit_stats, entity_key, amount_col)

        # Compute trimmed mean (remove top/bottom 10% outliers)
        deposit_stats = self._compute_trimmed_mean(deposits, deposit_stats, entity_key, amount_col)

        # Compute overall stability score
        deposit_stats = deposit_stats.withColumn(
            "deposit_stability_score",
            self._compute_stability_score_udf()(
                F.col("deposit_cv"),
                F.col("deposit_count"),
                F.col("deposit_mean"),
                F.col("deposit_median")
            )
        )

        # Fill nulls for customers with insufficient data
        deposit_stats = deposit_stats.fillna({
            "deposit_mean": 0.0,
            "deposit_std": 0.0,
            "deposit_cv": 0.0,
            "deposit_mad": 0.0,
            "deposit_median": 0.0,
            "deposit_trimmed_mean": 0.0,
            "deposit_min": 0.0,
            "deposit_max": 0.0,
            "deposit_p25": 0.0,
            "deposit_p75": 0.0,
            "deposit_count": 0,
            "deposit_stability_score": 0.0
        })

        logger.info(f"Deposit stability computation complete. Customers analyzed: {deposit_stats.count()}")

        return deposit_stats

    def _compute_mad(
        self,
        deposits: DataFrame,
        stats: DataFrame,
        entity_key: str,
        amount_col: str
    ) -> DataFrame:
        """
        Compute Median Absolute Deviation (MAD).

        MAD = median(|x - median(x)|)

        This requires two passes:
        1. Compute median per customer
        2. Compute median of absolute deviations
        """
        # Join deposits with their customer median
        deposits_with_median = deposits.join(
            stats.select(entity_key, "deposit_median"),
            entity_key,
            "left"
        ).withColumn(
            "abs_deviation",
            F.abs(F.col(amount_col) - F.col("deposit_median"))
        )

        # Compute median of absolute deviations
        mad_values = deposits_with_median.groupBy(entity_key).agg(
            F.expr("percentile(abs_deviation, 0.5)").alias("deposit_mad")
        )

        # Join back to stats
        return stats.join(mad_values, entity_key, "left")

    def _compute_trimmed_mean(
        self,
        deposits: DataFrame,
        stats: DataFrame,
        entity_key: str,
        amount_col: str
    ) -> DataFrame:
        """
        Compute trimmed mean (remove top/bottom trim_fraction outliers).

        For trim_fraction = 0.1:
        - Remove bottom 10% and top 10% of deposits
        - Compute mean of remaining 80%
        """
        # Add percentile rank to each deposit
        window_spec = Window.partitionBy(entity_key).orderBy(amount_col)

        deposits_with_rank = deposits.withColumn(
            "percentile_rank",
            F.percent_rank().over(window_spec)
        )

        # Filter to middle (1 - 2*trim_fraction) fraction
        lower_bound = self.trim_fraction
        upper_bound = 1.0 - self.trim_fraction

        trimmed_deposits = deposits_with_rank.filter(
            (F.col("percentile_rank") >= lower_bound) &
            (F.col("percentile_rank") <= upper_bound)
        )

        # Compute trimmed mean
        trimmed_mean_values = trimmed_deposits.groupBy(entity_key).agg(
            F.mean(amount_col).alias("deposit_trimmed_mean")
        )

        # Join back to stats
        return stats.join(trimmed_mean_values, entity_key, "left")

    @staticmethod
    def _compute_stability_score_udf():
        """
        UDF to compute overall deposit stability score.

        Score combines:
        - Low CV (good)
        - Sufficient sample size (good)
        - Mean close to median (good - means symmetric distribution)

        Returns: Score in [0, 1] where 1 = very stable
        """
        def compute_score(cv: float, count: int, mean: float, median: float) -> float:
            if count is None or count < 3:
                return 0.0

            # CV score: Lower is better (more stable)
            # CV < 0.2 = very stable (score 1.0)
            # CV > 1.0 = very volatile (score 0.0)
            if cv is None or cv < 0:
                cv_score = 0.0
            elif cv < 0.2:
                cv_score = 1.0
            elif cv > 1.0:
                cv_score = 0.0
            else:
                cv_score = 1.0 - (cv - 0.2) / 0.8

            # Sample size score
            sample_score = min(count / 10.0, 1.0)  # Saturates at 10 deposits

            # Symmetry score: mean should be close to median for stable deposits
            if mean is None or median is None or median == 0:
                symmetry_score = 0.5
            else:
                symmetry_ratio = mean / median
                # Ratio close to 1.0 is good
                symmetry_score = max(0, 1.0 - abs(symmetry_ratio - 1.0))

            # Combined score (weighted average)
            stability_score = (0.5 * cv_score + 0.3 * sample_score + 0.2 * symmetry_score)

            return float(min(max(stability_score, 0.0), 1.0))

        return F.udf(compute_score, DoubleType())


def compute_deposit_stability_metrics(
    df: DataFrame,
    entity_key: str = "customer_id",
    timestamp_col: str = "transaction_timestamp",
    amount_col: str = "transaction_amount",
    min_deposit_amount: float = 100.0
) -> DataFrame:
    """
    Convenience function to compute deposit stability metrics.

    Args:
        df: Spark DataFrame with transaction data
        entity_key: Customer identifier column
        timestamp_col: Transaction timestamp column
        amount_col: Transaction amount column
        min_deposit_amount: Minimum deposit to consider

    Returns:
        DataFrame with stability metrics per customer
    """
    calculator = DepositStabilityCalculator(
        min_deposit_amount=min_deposit_amount
    )

    return calculator.compute_stability_metrics(
        df, entity_key, timestamp_col, amount_col
    )
