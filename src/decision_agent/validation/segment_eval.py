"""
Segment-based model evaluation (Spark-native).
"""
import logging
import numpy as np
from typing import Dict, List, Any, Union

logger = logging.getLogger(__name__)


class SegmentEvaluator:
    """
    Evaluate model performance across customer segments using Spark aggregations.

    Ensures model works well across different customer groups.
    Returns aggregated Spark DataFrames (no toPandas() unless explicitly limited).
    """

    def __init__(self, segment_dimensions: List[Dict[str, Any]]):
        """
        Initialize segment evaluator.

        Args:
            segment_dimensions: List of segment definitions, e.g.:
                [{"dimension": "income_quartile", "values": ["Q1", "Q2", "Q3", "Q4"]}]
        """
        self.segment_dimensions = segment_dimensions

    def evaluate(
        self,
        features_df,
        predictions: Union[np.ndarray, str],
        target_col: str
    ) -> Dict[str, Any]:
        """
        Evaluate model performance across segments.

        Args:
            features_df: Spark or pandas DataFrame with features and target
            predictions: Either numpy array of predictions, or column name in features_df
            target_col: Name of target column

        Returns:
            Dictionary of segment metrics (aggregated Spark DataFrames or dicts)
        """
        logger.info("Evaluating model performance across segments...")

        # Check if Spark DataFrame
        try:
            from pyspark.sql import DataFrame as SparkDataFrame
            is_spark = isinstance(features_df, SparkDataFrame)
        except ImportError:
            is_spark = False

        if is_spark:
            return self._evaluate_spark(features_df, predictions, target_col)
        else:
            return self._evaluate_pandas(features_df, predictions, target_col)

    def _evaluate_spark(self, df, predictions, target_col):
        """
        Spark-native segment evaluation using aggregations.

        No toPandas() - all computations use Spark SQL.
        """
        from pyspark.sql import functions as F
        from pyspark.sql.window import Window

        # Add predictions column if it's a numpy array
        if isinstance(predictions, np.ndarray):
            # Convert predictions to list and add as column
            # For large datasets, use createDataFrame and join instead
            from pyspark.sql.types import DoubleType, LongType, StructType, StructField

            # Create DataFrame with row numbers and predictions
            predictions_data = [(i, float(pred)) for i, pred in enumerate(predictions)]
            predictions_schema = StructType([
                StructField("_row_num", LongType(), False),
                StructField("_prediction", DoubleType(), False)
            ])

            spark = df.sparkSession
            predictions_df = spark.createDataFrame(predictions_data, predictions_schema)

            # Add row numbers to original DataFrame
            window_spec = Window.orderBy(F.monotonically_increasing_id())
            df_with_rownum = df.withColumn("_row_num", F.row_number().over(window_spec) - 1)

            # Join predictions
            df = df_with_rownum.join(predictions_df, "_row_num").drop("_row_num")
        else:
            # Predictions is already a column name
            df = df.withColumnRenamed(predictions, "_prediction")

        results = {}

        for segment_def in self.segment_dimensions:
            dimension = segment_def["dimension"]

            # Create segment column if it doesn't exist
            if dimension not in df.columns:
                df = self._create_segment_column_spark(df, dimension, target_col)

            # Compute metrics by segment using Spark aggregations
            segment_metrics_df = df.groupBy(dimension).agg(
                F.count("*").alias("n_samples"),
                F.mean(F.abs(F.col(target_col) - F.col("_prediction"))).alias("mae"),
                F.sqrt(F.mean(F.pow(F.col(target_col) - F.col("_prediction"), 2))).alias("rmse"),
                F.mean(target_col).alias("mean_true"),
                F.mean("_prediction").alias("mean_pred"),
                # R² calculation: 1 - (SS_res / SS_tot)
                (1 - (
                    F.sum(F.pow(F.col(target_col) - F.col("_prediction"), 2)) /
                    F.sum(F.pow(F.col(target_col) - F.mean(target_col).over(Window.partitionBy()), 2))
                )).alias("r2")
            ).filter(F.col("n_samples") >= 10)  # Skip small segments

            # Convert to dictionary for logging (small result, safe to collect)
            segment_metrics = {}
            for row in segment_metrics_df.collect():
                segment_value = row[dimension]
                segment_metrics[str(segment_value)] = {
                    "n_samples": int(row["n_samples"]),
                    "mae": float(row["mae"]),
                    "rmse": float(row["rmse"]),
                    "r2": float(row["r2"]) if row["r2"] is not None else 0.0,
                    "mean_true": float(row["mean_true"]),
                    "mean_pred": float(row["mean_pred"])
                }

                logger.info(
                    f"  {dimension}={segment_value}: "
                    f"MAE={segment_metrics[str(segment_value)]['mae']:.2f}, "
                    f"R2={segment_metrics[str(segment_value)]['r2']:.3f}, "
                    f"n={segment_metrics[str(segment_value)]['n_samples']}"
                )

            results[dimension] = segment_metrics

        logger.info("✓ Segment evaluation completed (Spark-native)")
        return results

    def _evaluate_pandas(self, df, predictions, target_col):
        """Pandas evaluation (fallback for non-Spark DataFrames)"""
        import pandas as pd

        # Add predictions
        pdf = df.copy()
        if isinstance(predictions, np.ndarray):
            pdf["_prediction"] = predictions
        else:
            pdf = pdf.rename(columns={predictions: "_prediction"})

        results = {}

        for segment_def in self.segment_dimensions:
            dimension = segment_def["dimension"]

            # Create segment column if it doesn't exist
            if dimension not in pdf.columns:
                pdf = self._create_segment_column_pandas(pdf, dimension, target_col)

            # Compute metrics for each segment
            segment_metrics = {}

            for segment_value in pdf[dimension].unique():
                mask = pdf[dimension] == segment_value
                n_samples = mask.sum()

                if n_samples < 10:
                    continue

                y_true = pdf.loc[mask, target_col].values
                y_pred = pdf.loc[mask, "_prediction"].values

                metrics = self._compute_metrics_pandas(y_true, y_pred)
                metrics["n_samples"] = int(n_samples)

                segment_metrics[str(segment_value)] = metrics

                logger.info(
                    f"  {dimension}={segment_value}: "
                    f"MAE={metrics['mae']:.2f}, "
                    f"R2={metrics['r2']:.3f}, "
                    f"n={metrics['n_samples']}"
                )

            results[dimension] = segment_metrics

        logger.info("Segment evaluation completed (pandas)")
        return results

    def _create_segment_column_spark(self, df, dimension: str, target_col: str):
        """Create segment column using Spark (e.g., quartiles)"""
        from pyspark.sql import functions as F

        if dimension == "income_quartile":
            # Use ntile to create quartiles
            from pyspark.sql.window import Window

            window_spec = Window.orderBy(target_col)
            df = df.withColumn(
                "income_quartile",
                F.concat(F.lit("Q"), F.ntile(4).over(window_spec).cast("string"))
            )
            logger.info(f"Created {dimension} column with quartiles (Spark)")

        return df

    def _create_segment_column_pandas(self, pdf, dimension: str, target_col: str):
        """Create segment column using pandas"""
        import pandas as pd

        if dimension == "income_quartile":
            pdf["income_quartile"] = pd.qcut(
                pdf[target_col],
                q=4,
                labels=["Q1", "Q2", "Q3", "Q4"],
                duplicates="drop"
            )
            logger.info(f"Created {dimension} column with quartiles (pandas)")

        return pdf

    def _compute_metrics_pandas(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute evaluation metrics for pandas"""
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        return {
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "r2": float(r2_score(y_true, y_pred)),
            "mean_true": float(np.mean(y_true)),
            "mean_pred": float(np.mean(y_pred))
        }
