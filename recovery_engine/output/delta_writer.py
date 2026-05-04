"""
DeltaLakeWriter
===============
Writes recovery-engine-v2 model scores to Delta Lake table: recovery.model_scores

Responsibilities:
  - Validate output against model_output_contract.py before writing
  - Enforce PII policy (raises PIIViolationError on any PII column)
  - Stamp score_date, model_version, experiment_id on every row
  - Write to Delta in overwrite-partition mode (idempotent daily runs)
  - Log row counts and schema to recovery.data_quality_metrics
"""

import logging
from datetime import date
from typing import Optional

import pandas as pd

from contracts.model_output_contract import (
    SchemaValidationError,
    validate_pii,
    validate_schema,
)

logger = logging.getLogger(__name__)

DELTA_TABLE = "recovery.model_scores"
METRICS_TABLE = "recovery.data_quality_metrics"


class DeltaLakeWriter:
    """
    Validates and writes model scores to Delta Lake.

    Args:
        spark:          Active SparkSession (injected by Databricks task)
        model_version:  MLflow model version string e.g. "recovery-engine-v2.1.3"
        experiment_id:  MLflow experiment ID for this run
        score_date:     Date of scoring run (defaults to today)
    """

    def __init__(
        self,
        spark,
        model_version: str,
        experiment_id: str,
        score_date: Optional[date] = None,
    ):
        self.spark          = spark
        self.model_version  = model_version
        self.experiment_id  = experiment_id
        self.score_date     = str(score_date or date.today())

    def write(self, df: pd.DataFrame) -> dict:
        """
        Validate and write a scored DataFrame to recovery.model_scores.

        Args:
            df: pandas DataFrame with model scores (one row per account)

        Returns:
            dict with write summary: rows_written, quadrant_distribution, warnings

        Raises:
            PIIViolationError:     if any PII column is present
            SchemaValidationError: if required columns are missing or invalid
        """
        # 1. Stamp metadata columns
        df = df.copy()
        df["score_date"]     = self.score_date
        df["model_version"]  = self.model_version
        df["experiment_id"]  = self.experiment_id

        # 2. Validate (PII check is inside validate_schema)
        warnings = validate_schema(df)
        if warnings:
            for w in warnings:
                logger.warning(f"[DeltaLakeWriter] Schema warning: {w}")

        # 3. Convert to Spark and write
        import pyspark.sql.functions as F
        spark_df = self.spark.createDataFrame(df)

        (
            spark_df.write
            .format("delta")
            .mode("overwrite")
            .option("replaceWhere", f"score_date = '{self.score_date}'")
            .saveAsTable(DELTA_TABLE)
        )

        # 4. Write data quality metrics
        self._write_quality_metrics(df)

        summary = {
            "rows_written":           len(df),
            "score_date":             self.score_date,
            "model_version":          self.model_version,
            "quadrant_distribution":  df["signal_quadrant"].value_counts().to_dict(),
            "null_rates":             df.isnull().mean().round(4).to_dict(),
            "schema_warnings":        warnings,
        }
        logger.info(f"[DeltaLakeWriter] Written {len(df)} rows → {DELTA_TABLE} | {summary['quadrant_distribution']}")
        return summary

    def _write_quality_metrics(self, df: pd.DataFrame) -> None:
        """Log feature null rates and quadrant distribution for DataQualityAgent."""
        metrics = pd.DataFrame([{
            "score_date":             self.score_date,
            "model_version":          self.model_version,
            "total_accounts":         len(df),
            "quadrant_A_pct":         round((df["signal_quadrant"] == "A").mean(), 4),
            "quadrant_B_pct":         round((df["signal_quadrant"] == "B").mean(), 4),
            "quadrant_C_pct":         round((df["signal_quadrant"] == "C").mean(), 4),
            "quadrant_D_pct":         round((df["signal_quadrant"] == "D").mean(), 4),
            "propensity_30d_null_pct": round(df["propensity_30d"].isnull().mean(), 4),
            "low_confidence_pct":      round(df["low_confidence_flag"].mean(), 4),
        }])

        spark_metrics = self.spark.createDataFrame(metrics)
        (
            spark_metrics.write
            .format("delta")
            .mode("append")
            .saveAsTable(METRICS_TABLE)
        )
