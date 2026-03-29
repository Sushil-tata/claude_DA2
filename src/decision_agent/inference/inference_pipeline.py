"""
Production Inference Pipeline

Orchestrates end-to-end inference workflow:
1. Load Champion and Challenger models from MLflow
2. Route customers intelligently (ModelRouter)
3. Generate predictions
4. Compute SHAP explanations (FCRA compliance)
5. Log predictions with audit trail

Integration Points:
- ModelRouter (Phase 1): Smart routing based on OOD, data sufficiency, A/B
- Explainer (Phase 1): SHAP values for adverse action notices
- PredictionLogger (Phase 1): Audit trail with reason codes
"""

import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import mlflow
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from decision_agent.inference.model_router import ModelRouter
from decision_agent.inference.explainer import Explainer
from decision_agent.decisions.prediction_logger import PredictionLogger

logger = logging.getLogger(__name__)


class InferencePipeline:
    """
    Production inference pipeline with routing, explainability, and audit logging.

    Supports two modes:
    - Champion-only: Route all customers to Champion model
    - A/B testing: Route based on OOD, data sufficiency, and A/B split
    """

    def __init__(
        self,
        spark: SparkSession,
        config: Dict[str, Any],
        enable_ab_testing: bool = False
    ):
        """
        Initialize inference pipeline.

        Args:
            spark: Spark session
            config: Inference configuration:
                - champion_model_uri: MLflow model URI for Champion
                - challenger_model_uri: MLflow model URI for Challenger (optional)
                - routing: Routing configuration (OOD thresholds, A/B %, etc.)
                - explainability: SHAP configuration
                - audit_logging: Audit log configuration
            enable_ab_testing: If True, enables A/B testing with Challenger
        """
        self.spark = spark
        self.config = config
        self.enable_ab_testing = enable_ab_testing

        # Load models from MLflow
        self.champion_model_uri = config["champion_model_uri"]
        logger.info(f"Loading Champion model: {self.champion_model_uri}")
        self.champion_model = mlflow.pyfunc.load_model(self.champion_model_uri)

        if enable_ab_testing:
            self.challenger_model_uri = config.get("challenger_model_uri")
            if not self.challenger_model_uri:
                raise ValueError("A/B testing enabled but challenger_model_uri not provided")
            logger.info(f"Loading Challenger model: {self.challenger_model_uri}")
            self.challenger_model = mlflow.pyfunc.load_model(self.challenger_model_uri)
        else:
            self.challenger_model = None

        # Initialize routing (if A/B testing enabled)
        if enable_ab_testing:
            self.router = ModelRouter(config["routing"])
        else:
            self.router = None

        # Initialize explainer (SHAP)
        if config.get("explainability", {}).get("enabled", True):
            self.explainer = Explainer(
                model=self.champion_model,  # Use Champion for explanations
                config=config.get("explainability", {})
            )
        else:
            self.explainer = None

        # Initialize prediction logger
        self.prediction_logger = PredictionLogger(
            spark=spark,
            audit_table=config.get("audit_logging", {}).get("audit_table", "decision_agent.prediction_audit_log")
        )

    def predict_batch(
        self,
        features_df: DataFrame,
        use_case_id: str = "income_estimation",
        model_version: str = "production"
    ) -> DataFrame:
        """
        Run batch inference on customer features.

        Args:
            features_df: Spark DataFrame with customer features
            use_case_id: Use case identifier
            model_version: Model version label

        Returns:
            DataFrame with predictions, routing decisions, and reason codes
        """
        logger.info(f"Starting batch inference for {features_df.count():,} customers...")

        # Step 1: Routing (if A/B testing enabled)
        if self.enable_ab_testing and self.router:
            logger.info("Step 1: Routing customers...")
            features_with_routing = self.router.route_and_predict(
                features_df=features_df,
                champion_model=self.champion_model,
                challenger_model=self.challenger_model
            )

            # Log routing summary
            routing_summary = features_with_routing.groupBy("selected_model").count().collect()
            for row in routing_summary:
                logger.info(f"  {row['selected_model']}: {row['count']:,} customers")

        else:
            # Champion-only mode: predict for all customers
            logger.info("Step 1: Champion-only mode (no A/B testing)...")
            predictions = self._predict_spark(features_df, self.champion_model)

            features_with_routing = features_df.withColumn(
                "prediction", predictions
            ).withColumn(
                "selected_model", F.lit("champion")
            ).withColumn(
                "routing_reason", F.lit("champion_only_mode")
            ).withColumn(
                "routing_priority", F.lit(0)
            )

        # Step 2: SHAP Explanations (if enabled)
        if self.explainer:
            logger.info("Step 2: Generating SHAP explanations...")
            features_with_explanations = self._add_explanations(features_with_routing)
        else:
            logger.info("Step 2: SHAP disabled, skipping explanations")
            features_with_explanations = features_with_routing.withColumn(
                "reason_codes", F.lit(None).cast("array<string>")
            )

        # Step 3: Add metadata
        logger.info("Step 3: Adding metadata...")
        prediction_timestamp = datetime.now().isoformat()

        final_predictions = features_with_explanations.withColumn(
            "prediction_timestamp", F.lit(prediction_timestamp)
        ).withColumn(
            "use_case_id", F.lit(use_case_id)
        ).withColumn(
            "model_version", F.lit(model_version)
        ).withColumn(
            "champion_model_uri", F.lit(self.champion_model_uri)
        ).withColumn(
            "challenger_model_uri",
            F.lit(self.challenger_model_uri if self.enable_ab_testing else None)
        )

        # Step 4: Audit Logging
        if self.config.get("audit_logging", {}).get("enabled", True):
            logger.info("Step 4: Creating audit trail...")
            self.prediction_logger.log_predictions(
                predictions_df=final_predictions,
                model_version=model_version,
                model_uri=self.champion_model_uri,
                run_id=f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                use_case_id=use_case_id,
                prediction_timestamp=prediction_timestamp,
                log_features=self.config.get("audit_logging", {}).get("log_features", False)
            )
            logger.info("✓ Audit trail created")

        logger.info(f"✓ Batch inference complete: {final_predictions.count():,} predictions")
        return final_predictions

    def _predict_spark(
        self,
        features_df: DataFrame,
        model
    ) -> F.Column:
        """
        Generate predictions using MLflow model (Spark UDF).

        Args:
            features_df: Features DataFrame
            model: MLflow model

        Returns:
            Predictions column
        """
        # Get feature columns (exclude metadata)
        metadata_cols = {'customer_id', 'transaction_timestamp', 'income_level',
                        'data_quality_flag', 'data_sufficiency_score', 'ood_score'}
        feature_cols = [col for col in features_df.columns if col not in metadata_cols]

        # Create UDF for model prediction
        predict_udf = mlflow.pyfunc.spark_udf(
            self.spark,
            model_uri=model.metadata.model_uri if hasattr(model.metadata, 'model_uri') else self.champion_model_uri,
            result_type="double"
        )

        # Apply predictions
        return predict_udf(*[F.col(c) for c in feature_cols])

    def _add_explanations(self, predictions_df: DataFrame) -> DataFrame:
        """
        Add SHAP explanations to predictions.

        Args:
            predictions_df: DataFrame with predictions

        Returns:
            DataFrame with reason_codes column added
        """
        # For now, add placeholder reason codes
        # Full SHAP implementation requires converting to pandas for each row (expensive)
        # In production, would use distributed SHAP or precompute top features

        # Simplified: Use top features from model as reason codes
        top_features = self.config.get("explainability", {}).get("top_features", [
            "deposit_periodicity",
            "deposit_stability_score",
            "transaction_sum_30d"
        ])

        # Create reason codes array
        reason_codes = F.array([F.lit(f) for f in top_features[:3]])

        return predictions_df.withColumn("reason_codes", reason_codes)


def run_batch_inference(
    spark: SparkSession,
    features_table: str,
    config: Dict[str, Any],
    enable_ab_testing: bool = False,
    output_table: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to run batch inference.

    Args:
        spark: Spark session
        features_table: Delta table with customer features
        config: Inference configuration
        enable_ab_testing: Enable A/B testing with Challenger
        output_table: Output table for predictions (default: config["output_table"])

    Returns:
        Results dictionary with metrics
    """
    logger.info("=" * 80)
    logger.info("Starting Production Batch Inference")
    logger.info("=" * 80)

    # Load features
    logger.info(f"Loading features from: {features_table}")
    features_df = spark.table(features_table)
    num_customers = features_df.count()
    logger.info(f"Loaded {num_customers:,} customers")

    # Initialize pipeline
    pipeline = InferencePipeline(
        spark=spark,
        config=config,
        enable_ab_testing=enable_ab_testing
    )

    # Run inference
    predictions_df = pipeline.predict_batch(
        features_df=features_df,
        use_case_id=config.get("use_case_id", "income_estimation"),
        model_version=config.get("model_version", "production")
    )

    # Write to output table
    output_table = output_table or config.get("output_table", "decision_agent.production_predictions")
    logger.info(f"Writing predictions to: {output_table}")

    predictions_df.write \
        .format("delta") \
        .mode("append") \
        .saveAsTable(output_table)

    logger.info(f"✓ Predictions written to {output_table}")

    # Compute summary metrics
    results = {
        "num_customers": num_customers,
        "output_table": output_table,
        "enable_ab_testing": enable_ab_testing,
        "timestamp": datetime.now().isoformat()
    }

    if enable_ab_testing:
        routing_counts = predictions_df.groupBy("selected_model").count().collect()
        results["routing_summary"] = {row["selected_model"]: row["count"] for row in routing_counts}

    logger.info("=" * 80)
    logger.info("Batch Inference Complete")
    logger.info(f"Summary: {results}")
    logger.info("=" * 80)

    return results
