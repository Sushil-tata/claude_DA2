"""
Shadow Deployment - Safe Production Testing

Runs Challenger model in shadow mode:
- Generates predictions but does NOT affect production decisions
- Logs predictions to separate Delta table for offline analysis
- Allows comparison with Champion without risk

Why This Matters:
- Safe way to test new model in production
- Collect real-world data before full deployment
- Detect issues before they impact customers

Shadow Mode Workflow:
1. Production request comes in
2. Champion makes production decision (as normal)
3. Challenger ALSO makes prediction (shadow)
4. Shadow prediction logged to separate table
5. Daily job compares Champion vs Challenger performance
6. If Challenger performs well → increase rollout %

Integration Point:
- Runs in parallel to production inference
- Logs to: decision_agent.shadow_predictions
- Monitored by: monitoring/model_monitor.py
"""

import logging
from typing import Dict, Optional
from datetime import datetime
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
import mlflow

logger = logging.getLogger(__name__)


class ShadowDeployment:
    """
    Run model in shadow mode (predictions logged but not used).

    This is the SAFEST way to test a new model in production.
    """

    def __init__(self, config: Dict):
        """
        Initialize shadow deployment.

        Args:
            config: Shadow deployment configuration:
                - shadow_model_uri: MLflow URI for shadow model
                - shadow_table_name: Delta table for shadow predictions (default: shadow_predictions)
                - shadow_enabled: Enable shadow mode (default: True)
                - sampling_rate: Fraction of traffic to shadow (default: 1.0 = 100%)
        """
        self.config = config
        self.shadow_model_uri = config.get("shadow_model_uri")
        self.shadow_table_name = config.get("shadow_table_name", "decision_agent.shadow_predictions")
        self.shadow_enabled = config.get("shadow_enabled", True)
        self.sampling_rate = config.get("sampling_rate", 1.0)

        self.shadow_model = None

        if self.shadow_enabled and self.shadow_model_uri:
            try:
                logger.info(f"Loading shadow model from: {self.shadow_model_uri}")
                self.shadow_model = mlflow.pyfunc.load_model(self.shadow_model_uri)
            except Exception as e:
                logger.error(f"Failed to load shadow model: {e}")
                self.shadow_enabled = False

    def predict_and_log(
        self,
        features_df: DataFrame,
        production_predictions_col: str = "production_prediction",
        customer_id_col: str = "customer_id",
        run_id: str = None
    ) -> DataFrame:
        """
        Generate shadow predictions and log to Delta table.

        Args:
            features_df: DataFrame with features and production predictions
            production_predictions_col: Column with Champion production predictions
            customer_id_col: Customer identifier column
            run_id: Optional run ID for tracking

        Returns:
            Original DataFrame (unchanged - shadow mode doesn't affect production)
        """
        if not self.shadow_enabled:
            logger.debug("Shadow mode disabled. Skipping shadow predictions.")
            return features_df

        if self.shadow_model is None:
            logger.warning("Shadow model not loaded. Cannot generate shadow predictions.")
            return features_df

        logger.info("Generating shadow predictions...")

        try:
            # Sample if sampling_rate < 1.0
            if self.sampling_rate < 1.0:
                shadow_df = features_df.sample(fraction=self.sampling_rate)
            else:
                shadow_df = features_df

            # Generate shadow predictions
            shadow_predictions_df = self._generate_shadow_predictions(
                shadow_df, customer_id_col
            )

            # Add metadata
            shadow_predictions_df = shadow_predictions_df.withColumn(
                "shadow_timestamp", F.current_timestamp()
            ).withColumn(
                "shadow_model_uri", F.lit(self.shadow_model_uri)
            ).withColumn(
                "run_id", F.lit(run_id) if run_id else F.lit(None)
            )

            # Log to Delta table
            self._log_to_delta(shadow_predictions_df)

        except Exception as e:
            logger.error(f"Shadow prediction failed: {e}")
            # Don't fail production on shadow errors

        # Return original DataFrame unchanged
        return features_df

    def _generate_shadow_predictions(
        self,
        df: DataFrame,
        customer_id_col: str
    ) -> DataFrame:
        """Generate predictions with shadow model."""
        from decision_agent.utils.spark_guards import safe_to_pandas

        # Convert to pandas for prediction
        pdf = safe_to_pandas(df, max_rows=100000)

        # Get feature columns
        feature_cols = [col for col in pdf.columns
                       if col not in [customer_id_col, 'production_prediction',
                                     'selected_model', 'routing_reason']]

        # Predict
        shadow_predictions = self.shadow_model.predict(pdf[feature_cols])

        pdf['shadow_prediction'] = shadow_predictions

        # Convert back to Spark
        spark = df.sparkSession
        shadow_spark = spark.createDataFrame(pdf)

        return shadow_spark

    def _log_to_delta(self, shadow_predictions_df: DataFrame):
        """
        Log shadow predictions to Delta table.

        Table schema:
        - customer_id
        - production_prediction (from Champion)
        - shadow_prediction (from Challenger)
        - features (JSON or individual columns)
        - shadow_timestamp
        - shadow_model_uri
        - run_id
        """
        logger.info(f"Logging shadow predictions to: {self.shadow_table_name}")

        try:
            # Write to Delta table (append mode)
            shadow_predictions_df.write \
                .format("delta") \
                .mode("append") \
                .saveAsTable(self.shadow_table_name)

            logger.info(f"Logged {shadow_predictions_df.count()} shadow predictions")

        except Exception as e:
            logger.error(f"Failed to log shadow predictions: {e}")

    def analyze_shadow_performance(
        self,
        spark,
        start_date: str = None,
        end_date: str = None,
        label_col: str = "actual_income"
    ) -> Dict:
        """
        Analyze shadow model performance vs production model.

        This is run offline (daily job) to evaluate shadow model.

        Args:
            spark: Spark session
            start_date: Start date for analysis (YYYY-MM-DD)
            end_date: End date for analysis (YYYY-MM-DD)
            label_col: Column with actual labels (if available with lag)

        Returns:
            Dict with comparison metrics
        """
        logger.info("Analyzing shadow model performance...")

        # Read shadow predictions
        shadow_df = spark.table(self.shadow_table_name)

        # Filter by date range if provided
        if start_date:
            shadow_df = shadow_df.filter(F.col("shadow_timestamp") >= start_date)
        if end_date:
            shadow_df = shadow_df.filter(F.col("shadow_timestamp") <= end_date)

        # If labels available, compute metrics
        if label_col in shadow_df.columns:
            # MAE for production model
            production_mae = shadow_df.agg(
                F.mean(F.abs(F.col(label_col) - F.col("production_prediction")))
            ).collect()[0][0]

            # MAE for shadow model
            shadow_mae = shadow_df.agg(
                F.mean(F.abs(F.col(label_col) - F.col("shadow_prediction")))
            ).collect()[0][0]

            # Compute improvement
            improvement_pct = (production_mae - shadow_mae) / production_mae if production_mae > 0 else 0

            results = {
                "production_mae": float(production_mae),
                "shadow_mae": float(shadow_mae),
                "improvement_pct": float(improvement_pct),
                "sample_size": shadow_df.count()
            }

        else:
            # No labels yet - just compare distributions
            production_stats = shadow_df.agg(
                F.mean("production_prediction").alias("prod_mean"),
                F.stddev("production_prediction").alias("prod_std")
            ).collect()[0]

            shadow_stats = shadow_df.agg(
                F.mean("shadow_prediction").alias("shadow_mean"),
                F.stddev("shadow_prediction").alias("shadow_std")
            ).collect()[0]

            # Correlation
            correlation = shadow_df.stat.corr("production_prediction", "shadow_prediction")

            results = {
                "production_mean": float(production_stats["prod_mean"]),
                "production_std": float(production_stats["prod_std"]),
                "shadow_mean": float(shadow_stats["shadow_mean"]),
                "shadow_std": float(shadow_stats["shadow_std"]),
                "correlation": float(correlation),
                "sample_size": shadow_df.count()
            }

        logger.info(f"Shadow analysis results: {results}")

        return results
