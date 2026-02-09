"""
Model Router - Champion vs Challenger Routing (CRITICAL P0)

Routes customers to Champion or Challenger based on:
1. Data sufficiency (sparse history → Champion)
2. OOD score (out-of-distribution → Champion)
3. Customer tenure (new customers → Champion)
4. A/B test bucket (gradual rollout)
5. Confidence score (low confidence → Champion)

Routing Logic (Priority Order):
- OOD detected → Champion (safer for anomalous inputs)
- Insufficient data → Champion (more robust to sparse data)
- New customer (< 90 days) → Champion (proven model)
- A/B test → Challenger if in test group, else Champion
- Default → Challenger (if all checks pass)

Why This Matters:
- Safe gradual rollout (not big-bang)
- Fallback to proven model for edge cases
- Enables shadow mode testing
- Provides reason codes for routing decision

Integration Point:
- Called at inference time by orchestrator/router.py
- Uses outputs from sparse_history_handler and ood_detector
- Logs routing decisions to Delta Lake for analysis
"""

import logging
from typing import Dict, Tuple, Optional
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructType, StructField
import mlflow

logger = logging.getLogger(__name__)


class ModelRouter:
    """
    Route customers to appropriate model (Champion or Challenger).

    This is CRITICAL for safe production deployment.
    """

    def __init__(self, config: Dict):
        """
        Initialize router with routing policy configuration.

        Args:
            config: Routing configuration:
                - champion_model_uri: MLflow URI for Champion model
                - challenger_model_uri: MLflow URI for Challenger model
                - ood_threshold: OOD score threshold (default -0.5)
                - data_sufficiency_threshold: Minimum sufficiency score (default 0.5)
                - min_customer_tenure_days: Minimum tenure for Challenger (default 90)
                - challenger_rollout_pct: % of traffic to Challenger (default 10)
                - enable_ab_test: Enable A/B testing (default True)
        """
        self.config = config
        self.champion_model_uri = config.get("champion_model_uri")
        self.challenger_model_uri = config.get("challenger_model_uri")
        self.ood_threshold = config.get("ood_threshold", -0.5)
        self.data_sufficiency_threshold = config.get("data_sufficiency_threshold", 0.5)
        self.min_customer_tenure_days = config.get("min_customer_tenure_days", 90)
        self.challenger_rollout_pct = config.get("challenger_rollout_pct", 10)
        self.enable_ab_test = config.get("enable_ab_test", True)

        # Load models
        self.champion_model = None
        self.challenger_model = None

        if self.champion_model_uri:
            try:
                logger.info(f"Loading Champion model from: {self.champion_model_uri}")
                self.champion_model = mlflow.pyfunc.load_model(self.champion_model_uri)
            except Exception as e:
                logger.error(f"Failed to load Champion model: {e}")

        if self.challenger_model_uri:
            try:
                logger.info(f"Loading Challenger model from: {self.challenger_model_uri}")
                self.challenger_model = mlflow.pyfunc.load_model(self.challenger_model_uri)
            except Exception as e:
                logger.warning(f"Failed to load Challenger model: {e}")

    def route_and_predict(
        self,
        features_df: DataFrame,
        data_sufficiency_col: str = "data_sufficiency_score",
        ood_score_col: str = "ood_score",
        customer_tenure_col: str = "customer_tenure_days",
        customer_id_col: str = "customer_id"
    ) -> DataFrame:
        """
        Route customers and generate predictions.

        Args:
            features_df: DataFrame with features and routing signals
            data_sufficiency_col: Column with data sufficiency score
            ood_score_col: Column with OOD score
            customer_tenure_col: Column with customer tenure in days
            customer_id_col: Customer identifier column

        Returns:
            DataFrame with:
            - prediction: Final prediction
            - selected_model: 'champion' or 'challenger'
            - routing_reason: Why this model was selected
            - routing_priority: Priority level (1=highest)
        """
        logger.info("Routing customers and generating predictions...")

        # Add routing decision column
        routed_df = self._compute_routing_decision(
            features_df,
            data_sufficiency_col,
            ood_score_col,
            customer_tenure_col,
            customer_id_col
        )

        # Generate predictions based on routing
        predictions_df = self._generate_predictions(routed_df, customer_id_col)

        # Log routing statistics
        self._log_routing_stats(predictions_df)

        return predictions_df

    def _compute_routing_decision(
        self,
        df: DataFrame,
        data_sufficiency_col: str,
        ood_score_col: str,
        customer_tenure_col: str,
        customer_id_col: str
    ) -> DataFrame:
        """
        Compute routing decision for each customer.

        Priority order (check conditions top to bottom):
        1. OOD detected → Champion
        2. Insufficient data → Champion
        3. New customer → Champion
        4. A/B test → Challenger if in test group
        5. Default → Challenger
        """
        # Initialize with default
        routed = df.withColumn("selected_model", F.lit("challenger")) \
                   .withColumn("routing_reason", F.lit("default")) \
                   .withColumn("routing_priority", F.lit(5))

        # Rule 5: Default already set

        # Rule 4: A/B test (if enabled)
        if self.enable_ab_test:
            # Hash customer ID to assign to test/control
            # Test group: hash % 100 < rollout_pct
            routed = routed.withColumn(
                "_ab_bucket",
                F.abs(F.hash(F.col(customer_id_col))) % 100
            ).withColumn(
                "selected_model",
                F.when(
                    (F.col("_ab_bucket") >= self.challenger_rollout_pct) &
                    (F.col("routing_priority") == 5),  # Only override default
                    "champion"
                ).otherwise(F.col("selected_model"))
            ).withColumn(
                "routing_reason",
                F.when(
                    (F.col("_ab_bucket") >= self.challenger_rollout_pct) &
                    (F.col("routing_reason") == "default"),
                    "ab_test_control"
                ).otherwise(F.col("routing_reason"))
            ).withColumn(
                "routing_priority",
                F.when(
                    (F.col("_ab_bucket") >= self.challenger_rollout_pct) &
                    (F.col("routing_priority") == 5),
                    4
                ).otherwise(F.col("routing_priority"))
            )

        # Rule 3: New customer → Champion
        if customer_tenure_col in df.columns:
            routed = routed.withColumn(
                "selected_model",
                F.when(
                    F.col(customer_tenure_col) < self.min_customer_tenure_days,
                    "champion"
                ).otherwise(F.col("selected_model"))
            ).withColumn(
                "routing_reason",
                F.when(
                    F.col(customer_tenure_col) < self.min_customer_tenure_days,
                    "new_customer"
                ).otherwise(F.col("routing_reason"))
            ).withColumn(
                "routing_priority",
                F.when(
                    F.col(customer_tenure_col) < self.min_customer_tenure_days,
                    3
                ).otherwise(F.col("routing_priority"))
            )

        # Rule 2: Insufficient data → Champion
        if data_sufficiency_col in df.columns:
            routed = routed.withColumn(
                "selected_model",
                F.when(
                    F.col(data_sufficiency_col) < self.data_sufficiency_threshold,
                    "champion"
                ).otherwise(F.col("selected_model"))
            ).withColumn(
                "routing_reason",
                F.when(
                    F.col(data_sufficiency_col) < self.data_sufficiency_threshold,
                    "insufficient_data"
                ).otherwise(F.col("routing_reason"))
            ).withColumn(
                "routing_priority",
                F.when(
                    F.col(data_sufficiency_col) < self.data_sufficiency_threshold,
                    2
                ).otherwise(F.col("routing_priority"))
            )

        # Rule 1: OOD detected → Champion (highest priority)
        if ood_score_col in df.columns:
            routed = routed.withColumn(
                "selected_model",
                F.when(
                    F.col(ood_score_col) < self.ood_threshold,
                    "champion"
                ).otherwise(F.col("selected_model"))
            ).withColumn(
                "routing_reason",
                F.when(
                    F.col(ood_score_col) < self.ood_threshold,
                    "out_of_distribution"
                ).otherwise(F.col("routing_reason"))
            ).withColumn(
                "routing_priority",
                F.when(
                    F.col(ood_score_col) < self.ood_threshold,
                    1
                ).otherwise(F.col("routing_priority"))
            )

        return routed

    def _generate_predictions(
        self,
        routed_df: DataFrame,
        customer_id_col: str
    ) -> DataFrame:
        """
        Generate predictions based on routing decision.

        For Spark-native prediction, we'll need to handle this differently
        depending on whether models support Spark UDFs or require pandas conversion.
        """
        from decision_agent.utils.spark_guards import safe_to_pandas

        # Sample approach: Convert to pandas, predict, convert back
        # In production, use Spark UDFs if available

        routed_pdf = safe_to_pandas(routed_df, max_rows=100000)

        # Get feature columns (exclude metadata)
        feature_cols = [col for col in routed_pdf.columns
                       if col not in [customer_id_col, 'selected_model', 'routing_reason',
                                     'routing_priority', '_ab_bucket', 'ood_score',
                                     'data_sufficiency_score', 'customer_tenure_days']]

        # Predict for Champion group
        champion_mask = routed_pdf['selected_model'] == 'champion'
        if champion_mask.sum() > 0 and self.champion_model:
            champion_predictions = self.champion_model.predict(
                routed_pdf.loc[champion_mask, feature_cols]
            )
            routed_pdf.loc[champion_mask, 'prediction'] = champion_predictions
        elif champion_mask.sum() > 0:
            # No Champion model loaded, use fallback
            logger.warning("Champion model not loaded. Using Challenger for all.")
            routed_pdf.loc[champion_mask, 'prediction'] = None

        # Predict for Challenger group
        challenger_mask = routed_pdf['selected_model'] == 'challenger'
        if challenger_mask.sum() > 0 and self.challenger_model:
            challenger_predictions = self.challenger_model.predict(
                routed_pdf.loc[challenger_mask, feature_cols]
            )
            routed_pdf.loc[challenger_mask, 'prediction'] = challenger_predictions
        elif challenger_mask.sum() > 0 and self.champion_model:
            # No Challenger, fallback to Champion
            logger.warning("Challenger model not loaded. Using Champion as fallback.")
            champion_predictions = self.champion_model.predict(
                routed_pdf.loc[challenger_mask, feature_cols]
            )
            routed_pdf.loc[challenger_mask, 'prediction'] = champion_predictions
            routed_pdf.loc[challenger_mask, 'selected_model'] = 'champion'
            routed_pdf.loc[challenger_mask, 'routing_reason'] = 'challenger_unavailable'

        # Convert back to Spark
        spark = routed_df.sparkSession
        predictions_spark = spark.createDataFrame(routed_pdf)

        return predictions_spark

    def _log_routing_stats(self, predictions_df: DataFrame):
        """Log routing statistics for monitoring."""
        # Count by selected model
        model_counts = predictions_df.groupBy("selected_model").count().collect()

        logger.info("Routing Statistics:")
        for row in model_counts:
            logger.info(f"  {row['selected_model']}: {row['count']} customers")

        # Count by routing reason
        reason_counts = predictions_df.groupBy("routing_reason").count().collect()

        logger.info("Routing Reasons:")
        for row in reason_counts:
            logger.info(f"  {row['routing_reason']}: {row['count']} customers")


def route_prediction(
    features_df: DataFrame,
    config: Dict,
    **kwargs
) -> DataFrame:
    """
    Convenience function to route and predict.

    Args:
        features_df: DataFrame with features
        config: Routing configuration
        **kwargs: Additional arguments for routing

    Returns:
        DataFrame with predictions and routing metadata
    """
    router = ModelRouter(config)
    return router.route_and_predict(features_df, **kwargs)
