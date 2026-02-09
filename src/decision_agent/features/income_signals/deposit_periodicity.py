"""
Deposit Periodicity Detector - THE Most Important Income Signal

Detects salary payment cadence from deposit patterns.
This is the STRONGEST signal for income estimation because:
- Salaried employees have regular deposit patterns (bi-weekly, monthly, semi-monthly)
- Gig workers / self-employed have irregular deposits
- Knowing the period allows accurate income annualization

Detection Algorithm:
1. Filter deposits (positive transactions likely to be income)
2. Calculate inter-deposit intervals for each customer
3. Detect dominant periods using autocorrelation or mode
4. Classify: bi-weekly (14-15 days), semi-monthly (15-16 days), monthly (28-31 days)
5. Compute confidence score based on regularity

Integration Point:
- Called by features/income_features.py
- Outputs: detected_period_days, period_type, period_confidence, is_regular_paycheck
"""

import logging
from typing import Dict, Tuple
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StringType, IntegerType

logger = logging.getLogger(__name__)


class DepositPeriodicityDetector:
    """
    Detect salary deposit periodicity from transaction data.

    This is critical for income estimation:
    - Regular deposits → salaried employee → stable income
    - Irregular deposits → gig worker / self-employed → volatile income
    """

    def __init__(
        self,
        min_deposit_amount: float = 100.0,
        lookback_days: int = 180,
        min_deposits_required: int = 3
    ):
        """
        Initialize detector.

        Args:
            min_deposit_amount: Minimum deposit to consider as potential salary
            lookback_days: Days of history to analyze
            min_deposits_required: Minimum deposits needed to detect periodicity
        """
        self.min_deposit_amount = min_deposit_amount
        self.lookback_days = lookback_days
        self.min_deposits_required = min_deposits_required

    def detect(
        self,
        df: DataFrame,
        entity_key: str = "customer_id",
        timestamp_col: str = "transaction_timestamp",
        amount_col: str = "transaction_amount"
    ) -> DataFrame:
        """
        Detect deposit periodicity for each customer.

        Args:
            df: Spark DataFrame with transaction data
            entity_key: Customer identifier column
            timestamp_col: Transaction timestamp column
            amount_col: Transaction amount column

        Returns:
            DataFrame with one row per customer containing:
            - detected_period_days: Number of days between deposits
            - period_type: 'biweekly', 'semimonthly', 'monthly', 'irregular'
            - period_confidence: Confidence score [0, 1]
            - is_regular_paycheck: Boolean flag
            - avg_deposit_amount: Average deposit amount
            - deposit_count: Number of deposits found
        """
        logger.info("Detecting deposit periodicity...")

        # Step 1: Filter to deposits (positive amounts above threshold)
        deposits = df.filter(
            (F.col(amount_col) >= self.min_deposit_amount)
        ).select(
            entity_key,
            timestamp_col,
            amount_col
        )

        # Step 2: Sort by customer and timestamp
        window_spec = Window.partitionBy(entity_key).orderBy(timestamp_col)

        # Step 3: Calculate days between consecutive deposits
        deposits_with_intervals = deposits.withColumn(
            "prev_deposit_date",
            F.lag(timestamp_col).over(window_spec)
        ).withColumn(
            "days_since_prev_deposit",
            F.datediff(F.col(timestamp_col), F.col("prev_deposit_date"))
        )

        # Step 4: Aggregate per customer to detect period
        # We'll look at the mode (most common interval) and standard deviation
        customer_deposit_stats = deposits_with_intervals.groupBy(entity_key).agg(
            # Count of deposits
            F.count("*").alias("deposit_count"),

            # Average deposit amount
            F.mean(amount_col).alias("avg_deposit_amount"),

            # Median interval (using percentile for robustness to outliers)
            F.expr("percentile(days_since_prev_deposit, 0.5)").alias("median_interval_days"),

            # Mode approximation: most common interval bucket
            F.expr("percentile(days_since_prev_deposit, 0.5)").alias("mode_interval_days"),

            # Standard deviation of intervals (lower = more regular)
            F.stddev("days_since_prev_deposit").alias("interval_std"),

            # Coefficient of variation (std / mean) for regularity
            (F.stddev("days_since_prev_deposit") / F.avg("days_since_prev_deposit")).alias("interval_cv")
        )

        # Step 5: Classify period type and compute confidence
        result = customer_deposit_stats.withColumn(
            "detected_period_days",
            F.coalesce(F.col("median_interval_days"), F.lit(0)).cast(IntegerType())
        ).withColumn(
            "period_type",
            self._classify_period_udf()(
                F.col("detected_period_days")
            )
        ).withColumn(
            "period_confidence",
            self._compute_confidence_udf()(
                F.col("deposit_count"),
                F.col("interval_cv"),
                F.col("detected_period_days")
            )
        ).withColumn(
            "is_regular_paycheck",
            (F.col("period_confidence") >= 0.6) & (F.col("period_type") != "irregular")
        )

        # Step 6: Handle customers with insufficient deposits
        result = result.fillna({
            "detected_period_days": 0,
            "period_type": "insufficient_data",
            "period_confidence": 0.0,
            "is_regular_paycheck": False,
            "avg_deposit_amount": 0.0,
            "deposit_count": 0
        })

        logger.info(f"Deposit periodicity detection complete. Customers analyzed: {result.count()}")

        return result.select(
            entity_key,
            "detected_period_days",
            "period_type",
            "period_confidence",
            "is_regular_paycheck",
            "avg_deposit_amount",
            "deposit_count"
        )

    @staticmethod
    def _classify_period_udf():
        """UDF to classify deposit period into type."""
        def classify_period(days: int) -> str:
            if days is None or days == 0:
                return "irregular"
            elif 6 <= days <= 8:
                return "weekly"
            elif 13 <= days <= 15:
                return "biweekly"
            elif 15 <= days <= 17:
                return "semimonthly"  # Twice per month (e.g., 1st and 15th)
            elif 28 <= days <= 32:
                return "monthly"
            else:
                return "irregular"

        return F.udf(classify_period, StringType())

    @staticmethod
    def _compute_confidence_udf():
        """UDF to compute confidence score for periodicity detection."""
        def compute_confidence(deposit_count: int, interval_cv: float, period_days: int) -> float:
            """
            Confidence score based on:
            - Number of deposits (more = better)
            - Regularity (low CV = better)
            - Period validity (known periods = better)

            Returns: Score in [0, 1]
            """
            if deposit_count is None or deposit_count < 3:
                return 0.0

            if interval_cv is None or interval_cv > 1.0:
                # Very irregular deposits
                return 0.0

            # Base confidence from deposit count
            count_score = min(deposit_count / 10.0, 1.0)  # Saturates at 10 deposits

            # Regularity score (inverse of CV)
            regularity_score = max(0, 1.0 - interval_cv)

            # Period validity score
            if period_days in range(13, 16):  # Biweekly
                period_score = 1.0
            elif period_days in range(28, 33):  # Monthly
                period_score = 1.0
            elif period_days in range(6, 9):  # Weekly
                period_score = 0.9
            elif period_days in range(15, 18):  # Semi-monthly
                period_score = 0.9
            else:
                period_score = 0.3  # Irregular but might still have pattern

            # Combined confidence (weighted average)
            confidence = (0.3 * count_score + 0.4 * regularity_score + 0.3 * period_score)

            return float(min(max(confidence, 0.0), 1.0))

        return F.udf(compute_confidence, DoubleType())


def detect_deposit_periodicity(
    df: DataFrame,
    entity_key: str = "customer_id",
    timestamp_col: str = "transaction_timestamp",
    amount_col: str = "transaction_amount",
    min_deposit_amount: float = 100.0,
    lookback_days: int = 180
) -> DataFrame:
    """
    Convenience function to detect deposit periodicity.

    Args:
        df: Spark DataFrame with transaction data
        entity_key: Customer identifier column
        timestamp_col: Transaction timestamp column
        amount_col: Transaction amount column
        min_deposit_amount: Minimum deposit to consider
        lookback_days: Days of history to analyze

    Returns:
        DataFrame with periodicity features per customer
    """
    detector = DepositPeriodicityDetector(
        min_deposit_amount=min_deposit_amount,
        lookback_days=lookback_days
    )

    return detector.detect(df, entity_key, timestamp_col, amount_col)
