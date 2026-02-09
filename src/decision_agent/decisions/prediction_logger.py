"""
Prediction Logger - Audit Trail for Compliance

Logs every prediction with full context for regulatory compliance and debugging.

Why This Matters:
- FCRA requires audit trail for credit decisions
- Debugging production issues (why did model predict X?)
- Model monitoring and drift detection
- Regulatory audits (prove model behavior)

Logged Information:
- Prediction value
- Input features (JSON or columns)
- Model version/URI
- Timestamp
- Reason codes (from explainer)
- Routing decision (Champion vs Challenger)
- Customer identifier

Integration Point:
- Called by decisions/output_writer.py after prediction
- Extends output_writer to add audit logging
- Writes to: decision_agent.prediction_audit_log (separate from decisions table)
"""

import logging
import json
from typing import Dict, List, Optional
from datetime import datetime
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StringType

logger = logging.getLogger(__name__)


class PredictionLogger:
    """
    Log predictions with full context for audit trail.

    Separate from decision output table to avoid bloating production table.
    """

    def __init__(self, config: Dict):
        """
        Initialize logger.

        Args:
            config: Logging configuration:
                - audit_table_name: Delta table for audit log (default: prediction_audit_log)
                - log_features: Whether to log full features (default: True)
                - log_shap_values: Whether to log SHAP values (default: False, expensive)
                - retention_days: Days to retain audit logs (default: 365)
        """
        self.config = config
        self.audit_table_name = config.get("audit_table_name", "decision_agent.prediction_audit_log")
        self.log_features = config.get("log_features", True)
        self.log_shap_values = config.get("log_shap_values", False)
        self.retention_days = config.get("retention_days", 365)

    def log_predictions(
        self,
        predictions_df: DataFrame,
        customer_id_col: str = "customer_id",
        prediction_col: str = "prediction",
        model_version: str = None,
        model_uri: str = None,
        run_id: str = None,
        selected_model_col: str = "selected_model",
        routing_reason_col: str = "routing_reason",
        reason_codes_col: str = "reason_codes"
    ) -> DataFrame:
        """
        Log predictions to audit table.

        Args:
            predictions_df: DataFrame with predictions and metadata
            customer_id_col: Customer identifier column
            prediction_col: Prediction column
            model_version: Model version string
            model_uri: MLflow model URI
            run_id: MLflow run ID
            selected_model_col: Column indicating which model was used
            routing_reason_col: Column with routing reason
            reason_codes_col: Column with reason codes (from explainer)

        Returns:
            Original DataFrame (unchanged)
        """
        logger.info(f"Logging predictions to audit table: {self.audit_table_name}")

        try:
            # Prepare audit records
            audit_df = self._prepare_audit_records(
                predictions_df,
                customer_id_col,
                prediction_col,
                model_version,
                model_uri,
                run_id,
                selected_model_col,
                routing_reason_col,
                reason_codes_col
            )

            # Write to Delta table
            self._write_to_audit_table(audit_df)

            logger.info(f"Logged {audit_df.count()} predictions to audit table")

        except Exception as e:
            logger.error(f"Failed to log predictions: {e}")
            # Don't fail production on logging errors

        return predictions_df

    def _prepare_audit_records(
        self,
        df: DataFrame,
        customer_id_col: str,
        prediction_col: str,
        model_version: str,
        model_uri: str,
        run_id: str,
        selected_model_col: str,
        routing_reason_col: str,
        reason_codes_col: str
    ) -> DataFrame:
        """Prepare audit records with metadata."""
        # Add timestamp
        audit_df = df.withColumn("audit_timestamp", F.current_timestamp())

        # Add model metadata
        audit_df = audit_df.withColumn("model_version", F.lit(model_version)) \
                           .withColumn("model_uri", F.lit(model_uri)) \
                           .withColumn("run_id", F.lit(run_id))

        # If log_features enabled, convert features to JSON
        if self.log_features:
            # Get feature columns (exclude metadata)
            feature_cols = [col for col in df.columns
                           if col not in [customer_id_col, prediction_col, selected_model_col,
                                         routing_reason_col, reason_codes_col, 'ood_score',
                                         'data_sufficiency_score']]

            # Create JSON string of features
            audit_df = audit_df.withColumn(
                "features_json",
                F.to_json(F.struct([F.col(c).alias(c) for c in feature_cols]))
            )
        else:
            audit_df = audit_df.withColumn("features_json", F.lit(None))

        # Select columns for audit table
        audit_columns = [
            customer_id_col,
            prediction_col,
            "audit_timestamp",
            "model_version",
            "model_uri",
            "run_id"
        ]

        # Add optional columns if they exist
        if selected_model_col in df.columns:
            audit_columns.append(selected_model_col)
        if routing_reason_col in df.columns:
            audit_columns.append(routing_reason_col)
        if reason_codes_col in df.columns:
            audit_columns.append(reason_codes_col)

        audit_columns.append("features_json")

        return audit_df.select(*audit_columns)

    def _write_to_audit_table(self, audit_df: DataFrame):
        """Write to Delta audit table."""
        # Write in append mode
        audit_df.write \
            .format("delta") \
            .mode("append") \
            .option("mergeSchema", "true") \
            .saveAsTable(self.audit_table_name)

    def cleanup_old_logs(self, spark):
        """
        Clean up old audit logs beyond retention period.

        Run this periodically (e.g., weekly) to manage storage.

        Args:
            spark: Spark session
        """
        logger.info(f"Cleaning up audit logs older than {self.retention_days} days...")

        try:
            cutoff_date = F.date_sub(F.current_date(), self.retention_days)

            # Delete old records
            spark.sql(f"""
                DELETE FROM {self.audit_table_name}
                WHERE audit_timestamp < '{cutoff_date}'
            """)

            logger.info("Old audit logs cleaned up successfully")

        except Exception as e:
            logger.error(f"Failed to cleanup old logs: {e}")

    def query_audit_trail(
        self,
        spark,
        customer_id: str = None,
        start_date: str = None,
        end_date: str = None,
        model_version: str = None
    ) -> DataFrame:
        """
        Query audit trail for analysis/debugging.

        Args:
            spark: Spark session
            customer_id: Optional customer ID to filter
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            model_version: Optional model version to filter

        Returns:
            DataFrame with audit records
        """
        audit_df = spark.table(self.audit_table_name)

        # Apply filters
        if customer_id:
            audit_df = audit_df.filter(F.col("customer_id") == customer_id)

        if start_date:
            audit_df = audit_df.filter(F.col("audit_timestamp") >= start_date)

        if end_date:
            audit_df = audit_df.filter(F.col("audit_timestamp") <= end_date)

        if model_version:
            audit_df = audit_df.filter(F.col("model_version") == model_version)

        return audit_df


def log_predictions(
    predictions_df: DataFrame,
    config: Dict,
    **kwargs
) -> DataFrame:
    """
    Convenience function to log predictions.

    Args:
        predictions_df: DataFrame with predictions
        config: Logging configuration
        **kwargs: Additional arguments for logging

    Returns:
        Original DataFrame (unchanged)
    """
    logger_instance = PredictionLogger(config)
    return logger_instance.log_predictions(predictions_df, **kwargs)
