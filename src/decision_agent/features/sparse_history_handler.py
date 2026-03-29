"""
Sparse History Handler - Data Sufficiency Gating

Handles customers with insufficient transaction history.
This is CRITICAL for production because:
- New customers have < 30 days of data
- Current pipeline will fail or produce garbage features
- Regulatory risk: Disparate impact on new-to-bank customers

Strategy:
1. Compute data_sufficiency_score [0, 1] based on:
   - Transaction count
   - Days of history
   - Deposit count
   - Data completeness
2. Gate feature computation based on score
3. Route low-score customers to fallback model (Champion)

Integration Point:
- Wraps all feature modules in income_features.py
- Integrates with inference/model_router.py for model selection
- Outputs: data_sufficiency_score, data_quality_flag, recommended_model
"""

import logging
from typing import Dict, Tuple
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StringType

logger = logging.getLogger(__name__)


class SparseHistoryHandler:
    """
    Assess data sufficiency and handle sparse history cases.

    This is a GATE that prevents feature engineering on insufficient data.
    """

    def __init__(self, config: Dict[str, float]):
        """
        Initialize handler with thresholds.

        Args:
            config: Configuration with thresholds:
                - min_days_history: Minimum days of history (default 30)
                - ideal_days_history: Ideal days for full features (default 90)
                - min_transaction_count: Minimum transactions (default 10)
                - ideal_transaction_count: Ideal transactions (default 50)
                - min_deposit_count: Minimum deposits (default 2)
                - ideal_deposit_count: Ideal deposits (default 6)
                - sufficiency_threshold: Threshold for "sufficient" (default 0.5)
        """
        self.min_days_history = config.get("min_days_history", 30)
        self.ideal_days_history = config.get("ideal_days_history", 90)
        self.min_transaction_count = config.get("min_transaction_count", 10)
        self.ideal_transaction_count = config.get("ideal_transaction_count", 50)
        self.min_deposit_count = config.get("min_deposit_count", 2)
        self.ideal_deposit_count = config.get("ideal_deposit_count", 6)
        self.sufficiency_threshold = config.get("sufficiency_threshold", 0.5)

    def assess_data_sufficiency(
        self,
        df: DataFrame,
        entity_key: str = "customer_id",
        timestamp_col: str = "transaction_timestamp",
        amount_col: str = "transaction_amount"
    ) -> DataFrame:
        """
        Assess data sufficiency for each customer.

        Args:
            df: Spark DataFrame with transaction data
            entity_key: Customer identifier column
            timestamp_col: Transaction timestamp column
            amount_col: Transaction amount column

        Returns:
            DataFrame with one row per customer containing:
            - days_history: Days of transaction history
            - transaction_count: Number of transactions
            - deposit_count: Number of deposits (amount > 0)
            - expense_count: Number of expenses (amount < 0)
            - data_sufficiency_score: Score in [0, 1]
            - data_quality_flag: 'sufficient', 'marginal', 'insufficient'
            - recommended_model: 'challenger', 'champion', 'manual_review'
            - feature_computation_allowed: Boolean (can we compute features?)
        """
        logger.info("Assessing data sufficiency for customers...")

        # Calculate temporal coverage
        temporal_stats = df.groupBy(entity_key).agg(
            F.min(timestamp_col).alias("first_transaction_date"),
            F.max(timestamp_col).alias("last_transaction_date"),
            F.datediff(F.max(timestamp_col), F.min(timestamp_col)).alias("days_history")
        )

        # Calculate transaction counts
        transaction_counts = df.groupBy(entity_key).agg(
            F.count("*").alias("transaction_count"),
            F.sum(F.when(F.col(amount_col) > 0, 1).otherwise(0)).alias("deposit_count"),
            F.sum(F.when(F.col(amount_col) < 0, 1).otherwise(0)).alias("expense_count")
        )

        # Join stats
        customer_stats = temporal_stats.join(transaction_counts, entity_key, "inner")

        # Compute data sufficiency score
        customer_stats = customer_stats.withColumn(
            "data_sufficiency_score",
            self._compute_sufficiency_score_udf()(
                F.col("days_history"),
                F.col("transaction_count"),
                F.col("deposit_count"),
                F.lit(self.min_days_history),
                F.lit(self.ideal_days_history),
                F.lit(self.min_transaction_count),
                F.lit(self.ideal_transaction_count),
                F.lit(self.min_deposit_count),
                F.lit(self.ideal_deposit_count)
            )
        )

        # Classify data quality
        customer_stats = customer_stats.withColumn(
            "data_quality_flag",
            F.when(F.col("data_sufficiency_score") >= 0.7, "sufficient")
            .when(F.col("data_sufficiency_score") >= self.sufficiency_threshold, "marginal")
            .otherwise("insufficient")
        )

        # Recommend model routing
        customer_stats = customer_stats.withColumn(
            "recommended_model",
            F.when(F.col("data_quality_flag") == "sufficient", "challenger")
            .when(F.col("data_quality_flag") == "marginal", "champion")
            .otherwise("champion")  # Use Champion for insufficient data (safer)
        )

        # Can we compute full features?
        customer_stats = customer_stats.withColumn(
            "feature_computation_allowed",
            (F.col("days_history") >= self.min_days_history) &
            (F.col("transaction_count") >= self.min_transaction_count) &
            (F.col("deposit_count") >= self.min_deposit_count)
        )

        logger.info("Data sufficiency assessment complete.")

        # Log distribution of data quality
        quality_dist = customer_stats.groupBy("data_quality_flag").count().collect()
        for row in quality_dist:
            logger.info(f"  {row['data_quality_flag']}: {row['count']} customers")

        return customer_stats.select(
            entity_key,
            "days_history",
            "transaction_count",
            "deposit_count",
            "expense_count",
            "data_sufficiency_score",
            "data_quality_flag",
            "recommended_model",
            "feature_computation_allowed"
        )

    def filter_sufficient_customers(
        self,
        df: DataFrame,
        sufficiency_df: DataFrame,
        entity_key: str = "customer_id"
    ) -> Tuple[DataFrame, DataFrame]:
        """
        Split customers into sufficient and insufficient groups.

        Args:
            df: Original transaction DataFrame
            sufficiency_df: Output of assess_data_sufficiency()
            entity_key: Customer identifier

        Returns:
            Tuple of (sufficient_df, insufficient_df)
            - sufficient_df: Customers with sufficient data for feature engineering
            - insufficient_df: Customers to route to Champion or manual review
        """
        # Get customers with sufficient data
        sufficient_customers = sufficiency_df.filter(
            F.col("feature_computation_allowed") == True
        ).select(entity_key)

        # Get customers with insufficient data
        insufficient_customers = sufficiency_df.filter(
            F.col("feature_computation_allowed") == False
        ).select(entity_key)

        # Split original DataFrame
        sufficient_df = df.join(sufficient_customers, entity_key, "inner")
        insufficient_df = df.join(insufficient_customers, entity_key, "inner")

        logger.info(f"Split customers: {sufficient_customers.count()} sufficient, "
                   f"{insufficient_customers.count()} insufficient")

        return sufficient_df, insufficient_df

    def create_fallback_features(
        self,
        insufficient_df: DataFrame,
        entity_key: str = "customer_id",
        amount_col: str = "transaction_amount"
    ) -> DataFrame:
        """
        Create minimal fallback features for insufficient data cases.

        These are simple aggregates that don't require long history:
        - Total transaction count
        - Average transaction amount
        - Balance (if available)

        Args:
            insufficient_df: DataFrame with insufficient history customers
            entity_key: Customer identifier
            amount_col: Transaction amount column

        Returns:
            DataFrame with fallback features per customer
        """
        logger.info("Creating fallback features for insufficient data customers...")

        fallback_features = insufficient_df.groupBy(entity_key).agg(
            F.count("*").alias("total_transaction_count"),
            F.mean(amount_col).alias("avg_transaction_amount"),
            F.sum(F.when(F.col(amount_col) > 0, F.col(amount_col)).otherwise(0)).alias("total_deposits"),
            F.sum(F.when(F.col(amount_col) < 0, F.abs(F.col(amount_col))).otherwise(0)).alias("total_expenses"),
            F.mean(F.when(F.col(amount_col) > 0, F.col(amount_col))).alias("avg_deposit_amount")
        ).withColumn(
            "fallback_income_estimate",
            # Very crude estimate: avg deposit * 2 (assume bi-weekly) * 26 (weeks per year)
            F.col("avg_deposit_amount") * 2 * 26
        )

        logger.info(f"Fallback features created for {fallback_features.count()} customers")

        return fallback_features

    @staticmethod
    def _compute_sufficiency_score_udf():
        """
        UDF to compute data sufficiency score.

        Score is based on:
        - Days of history (linear scale from min to ideal)
        - Transaction count (linear scale from min to ideal)
        - Deposit count (linear scale from min to ideal)

        Returns: Score in [0, 1] where 1 = ideal data sufficiency
        """
        def compute_score(
            days_history: int,
            transaction_count: int,
            deposit_count: int,
            min_days: int,
            ideal_days: int,
            min_txn: int,
            ideal_txn: int,
            min_dep: int,
            ideal_dep: int
        ) -> float:
            """Compute sufficiency score."""
            if days_history is None or transaction_count is None or deposit_count is None:
                return 0.0

            # Days score (linear interpolation between min and ideal)
            if days_history < min_days:
                days_score = 0.0
            elif days_history >= ideal_days:
                days_score = 1.0
            else:
                days_score = (days_history - min_days) / (ideal_days - min_days)

            # Transaction count score
            if transaction_count < min_txn:
                txn_score = 0.0
            elif transaction_count >= ideal_txn:
                txn_score = 1.0
            else:
                txn_score = (transaction_count - min_txn) / (ideal_txn - min_txn)

            # Deposit count score
            if deposit_count < min_dep:
                dep_score = 0.0
            elif deposit_count >= ideal_dep:
                dep_score = 1.0
            else:
                dep_score = (deposit_count - min_dep) / (ideal_dep - min_dep)

            # Combined score (weighted average)
            # Days and deposits are more important than total transaction count
            sufficiency_score = (0.4 * days_score + 0.3 * dep_score + 0.3 * txn_score)

            return float(min(max(sufficiency_score, 0.0), 1.0))

        return F.udf(compute_score, DoubleType())


def assess_data_sufficiency(
    df: DataFrame,
    entity_key: str = "customer_id",
    timestamp_col: str = "transaction_timestamp",
    amount_col: str = "transaction_amount",
    config: Dict[str, float] = None
) -> DataFrame:
    """
    Convenience function to assess data sufficiency.

    Args:
        df: Spark DataFrame with transaction data
        entity_key: Customer identifier
        timestamp_col: Transaction timestamp column
        amount_col: Transaction amount column
        config: Configuration dict with thresholds

    Returns:
        DataFrame with sufficiency metrics per customer
    """
    if config is None:
        config = {}

    handler = SparseHistoryHandler(config)

    return handler.assess_data_sufficiency(df, entity_key, timestamp_col, amount_col)
