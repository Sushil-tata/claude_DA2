"""
Calibration evaluation for regression and classification models (Spark-native).
"""
import logging
import numpy as np
from typing import Dict, Any, Union

logger = logging.getLogger(__name__)


class CalibrationEvaluator:
    """
    Evaluate model calibration using Spark aggregations.

    For regression: compare predicted vs actual by decile.
    For classification: calibration curves for probabilities.
    Returns aggregated Spark DataFrames (no toPandas() unless explicitly limited).
    """

    def __init__(self, calibration_config: Dict[str, Any]):
        """
        Initialize calibration evaluator.

        Args:
            calibration_config: Calibration configuration from YAML
        """
        self.config = calibration_config
        self.method = calibration_config.get("method", "isotonic")
        self.n_bins = calibration_config.get("min_samples_per_bin", 10)

    def evaluate(
        self,
        features_df,
        predictions: Union[np.ndarray, str],
        target_col: str = None
    ) -> Dict[str, Any]:
        """
        Evaluate calibration.

        Args:
            features_df: Spark or pandas DataFrame with features and target
            predictions: Either numpy array or column name with predictions
            target_col: Name of target column (if available)

        Returns:
            Calibration metrics dictionary with aggregated results
        """
        logger.info("Evaluating model calibration...")

        if target_col is None:
            logger.warning("No target column available for calibration evaluation")
            return {"calibration_evaluated": False}

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
        Spark-native calibration evaluation using aggregations.

        No toPandas() - all computations use Spark SQL.
        """
        from pyspark.sql import functions as F
        from pyspark.sql.window import Window

        # Add predictions column if numpy array
        if isinstance(predictions, np.ndarray):
            from pyspark.sql.types import DoubleType, LongType, StructType, StructField

            predictions_data = [(i, float(pred)) for i, pred in enumerate(predictions)]
            predictions_schema = StructType([
                StructField("_row_num", LongType(), False),
                StructField("_prediction", DoubleType(), False)
            ])

            spark = df.sparkSession
            predictions_df = spark.createDataFrame(predictions_data, predictions_schema)

            window_spec = Window.orderBy(F.monotonically_increasing_id())
            df_with_rownum = df.withColumn("_row_num", F.row_number().over(window_spec) - 1)
            df = df_with_rownum.join(predictions_df, "_row_num").drop("_row_num")
        else:
            df = df.withColumnRenamed(predictions, "_prediction")

        # Check if classification or regression
        is_classification = self._is_classification_spark(df, target_col)

        if is_classification:
            return self._evaluate_classification_calibration_spark(df, target_col)
        else:
            return self._evaluate_regression_calibration_spark(df, target_col)

    def _evaluate_regression_calibration_spark(self, df, target_col):
        """
        Evaluate calibration for regression using Spark aggregations.

        Compares predicted vs actual values by decile.
        """
        from pyspark.sql import functions as F

        logger.info("Evaluating regression calibration by decile (Spark)...")

        # Create deciles based on predictions
        from pyspark.sql.window import Window

        window_spec = Window.orderBy("_prediction")
        df_with_decile = df.withColumn(
            "decile",
            F.ntile(10).over(window_spec)
        )

        # Aggregate by decile
        calibration_df = df_with_decile.groupBy("decile").agg(
            F.count("*").alias("n_samples"),
            F.mean("_prediction").alias("mean_predicted"),
            F.mean(target_col).alias("mean_actual"),
            F.abs(F.mean("_prediction") - F.mean(target_col)).alias("calibration_error")
        ).orderBy("decile")

        # Collect results (small - only 10 deciles)
        calibration_data = []
        for row in calibration_df.collect():
            calibration_data.append({
                "bin": int(row["decile"]),
                "n_samples": int(row["n_samples"]),
                "mean_predicted": float(row["mean_predicted"]),
                "mean_actual": float(row["mean_actual"]),
                "calibration_error": float(row["calibration_error"])
            })

        # Overall calibration error
        overall_error = sum(d["calibration_error"] for d in calibration_data) / len(calibration_data)

        logger.info(f"Calibration error: {overall_error:.2f}")

        return {
            "calibration_type": "regression",
            "n_bins": len(calibration_data),
            "bins": calibration_data,
            "overall_calibration_error": float(overall_error)
        }

    def _evaluate_classification_calibration_spark(self, df, target_col):
        """
        Evaluate calibration for classification using Spark aggregations.
        """
        from pyspark.sql import functions as F
        from pyspark.sql.window import Window

        logger.info("Evaluating classification calibration (Spark)...")

        # Create bins based on predicted probabilities
        window_spec = Window.orderBy("_prediction")
        df_with_bin = df.withColumn(
            "prob_bin",
            F.ntile(10).over(window_spec)
        )

        # Aggregate by bin
        calibration_df = df_with_bin.groupBy("prob_bin").agg(
            F.count("*").alias("n_samples"),
            F.mean("_prediction").alias("prob_pred"),
            F.mean(target_col).alias("prob_true")
        ).orderBy("prob_bin")

        # Collect results (small - only 10 bins)
        prob_true_list = []
        prob_pred_list = []

        for row in calibration_df.collect():
            if row["n_samples"] >= 5:  # Minimum samples per bin
                prob_true_list.append(float(row["prob_true"]))
                prob_pred_list.append(float(row["prob_pred"]))

        # Expected Calibration Error (ECE)
        if len(prob_true_list) > 0:
            ece = sum(abs(pt - pp) for pt, pp in zip(prob_true_list, prob_pred_list)) / len(prob_true_list)
        else:
            ece = 0.0

        logger.info(f"Expected Calibration Error (ECE): {ece:.4f}")

        return {
            "calibration_type": "classification",
            "prob_true": prob_true_list,
            "prob_pred": prob_pred_list,
            "ece": float(ece),
            "n_bins": len(prob_true_list)
        }

    def _evaluate_pandas(self, df, predictions, target_col):
        """Pandas implementation (fallback)"""
        import pandas as pd

        # Add predictions
        pdf = df.copy()
        if isinstance(predictions, np.ndarray):
            pdf["_prediction"] = predictions
        else:
            pdf = pdf.rename(columns={predictions: "_prediction"})

        # Check if classification or regression
        is_classification = self._is_classification_pandas(pdf, target_col)

        if is_classification:
            return self._evaluate_classification_calibration_pandas(pdf, target_col)
        else:
            return self._evaluate_regression_calibration_pandas(pdf, target_col)

    def _evaluate_regression_calibration_pandas(self, pdf, target_col):
        """Pandas regression calibration"""
        logger.info("Evaluating regression calibration by decile (pandas)...")

        # Create deciles
        pdf["decile"] = pd.qcut(pdf["_prediction"], q=10, labels=False, duplicates="drop")

        calibration_data = []
        for decile in sorted(pdf["decile"].unique()):
            mask = pdf["decile"] == decile
            n_samples = mask.sum()

            if n_samples < 5:
                continue

            mean_pred = pdf.loc[mask, "_prediction"].mean()
            mean_true = pdf.loc[mask, target_col].mean()

            calibration_data.append({
                "bin": int(decile),
                "n_samples": int(n_samples),
                "mean_predicted": float(mean_pred),
                "mean_actual": float(mean_true),
                "calibration_error": float(abs(mean_pred - mean_true))
            })

        overall_error = sum(d["calibration_error"] for d in calibration_data) / len(calibration_data) if calibration_data else 0.0

        return {
            "calibration_type": "regression",
            "n_bins": len(calibration_data),
            "bins": calibration_data,
            "overall_calibration_error": float(overall_error)
        }

    def _evaluate_classification_calibration_pandas(self, pdf, target_col):
        """Pandas classification calibration"""
        from sklearn.calibration import calibration_curve

        y_true = pdf[target_col].values
        y_pred_proba = pdf["_prediction"].values

        prob_true, prob_pred = calibration_curve(
            y_true, y_pred_proba, n_bins=10, strategy='quantile'
        )

        ece = np.mean(np.abs(prob_true - prob_pred))

        return {
            "calibration_type": "classification",
            "prob_true": prob_true.tolist(),
            "prob_pred": prob_pred.tolist(),
            "ece": float(ece),
            "n_bins": len(prob_true)
        }

    def _is_classification_spark(self, df, target_col):
        """Check if task is classification (few unique values) using Spark"""
        from pyspark.sql import functions as F

        unique_count = df.select(F.countDistinct(target_col)).collect()[0][0]
        return unique_count < 20

    def _is_classification_pandas(self, pdf, target_col):
        """Check if task is classification (pandas)"""
        return pdf[target_col].nunique() < 20
