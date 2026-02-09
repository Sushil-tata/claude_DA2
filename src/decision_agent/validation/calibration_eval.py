"""
Calibration evaluation for regression and classification models.
"""
import logging
import numpy as np
from typing import Dict, Any

logger = logging.getLogger(__name__)


class CalibrationEvaluator:
    """
    Evaluate model calibration.

    For regression: compare predicted vs actual by decile.
    For classification: calibration curves for probabilities.
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

    def evaluate(self, features_df, predictions: np.ndarray, target_col: str = None) -> Dict[str, Any]:
        """
        Evaluate calibration.

        Args:
            features_df: DataFrame with features and target
            predictions: Model predictions
            target_col: Name of target column (if available)

        Returns:
            Calibration metrics dictionary
        """
        logger.info("Evaluating model calibration...")

        # Convert to pandas if needed
        pdf = self._to_pandas(features_df)

        if target_col is None or target_col not in pdf.columns:
            logger.warning("No target column available for calibration evaluation")
            return {"calibration_evaluated": False}

        y_true = pdf[target_col].values

        # Check if regression or classification
        is_classification = self._is_classification(y_true)

        if is_classification:
            return self._evaluate_classification_calibration(y_true, predictions)
        else:
            return self._evaluate_regression_calibration(y_true, predictions)

    def _evaluate_regression_calibration(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate calibration for regression.

        Compares predicted vs actual values by decile.
        """
        logger.info("Evaluating regression calibration by decile...")

        # Create deciles based on predictions
        n_bins = 10
        bin_edges = np.percentile(y_pred, np.linspace(0, 100, n_bins + 1))

        # Ensure unique bin edges
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < 2:
            logger.warning("Not enough unique values for binning")
            return {"calibration_evaluated": False}

        # Assign samples to bins
        bin_indices = np.digitize(y_pred, bin_edges[1:-1])

        calibration_data = []

        for bin_idx in range(len(bin_edges) - 1):
            mask = bin_indices == bin_idx
            n_samples = mask.sum()

            if n_samples < 5:
                continue

            mean_pred = y_pred[mask].mean()
            mean_true = y_true[mask].mean()

            calibration_data.append({
                "bin": int(bin_idx),
                "n_samples": int(n_samples),
                "mean_predicted": float(mean_pred),
                "mean_actual": float(mean_true),
                "calibration_error": float(abs(mean_pred - mean_true))
            })

        # Overall calibration error
        overall_error = np.mean([d["calibration_error"] for d in calibration_data])

        logger.info(f"Calibration error: {overall_error:.2f}")

        return {
            "calibration_type": "regression",
            "n_bins": len(calibration_data),
            "bins": calibration_data,
            "overall_calibration_error": float(overall_error)
        }

    def _evaluate_classification_calibration(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate calibration for classification using calibration curves.
        """
        logger.info("Evaluating classification calibration...")

        try:
            from sklearn.calibration import calibration_curve

            # Compute calibration curve
            prob_true, prob_pred = calibration_curve(
                y_true,
                y_pred_proba,
                n_bins=10,
                strategy='quantile'
            )

            # Expected Calibration Error (ECE)
            ece = np.mean(np.abs(prob_true - prob_pred))

            logger.info(f"Expected Calibration Error (ECE): {ece:.4f}")

            return {
                "calibration_type": "classification",
                "prob_true": prob_true.tolist(),
                "prob_pred": prob_pred.tolist(),
                "ece": float(ece),
                "n_bins": len(prob_true)
            }

        except ImportError:
            logger.warning("scikit-learn not available for calibration curves")
            return {"calibration_evaluated": False}

    def _to_pandas(self, df):
        """Convert Spark DataFrame to pandas if needed"""
        try:
            from pyspark.sql import DataFrame as SparkDataFrame

            if isinstance(df, SparkDataFrame):
                return df.toPandas()
        except ImportError:
            pass

        return df

    def _is_classification(self, y: np.ndarray) -> bool:
        """Check if task is classification (few unique values)"""
        unique_values = len(np.unique(y))
        return unique_values < 20  # Heuristic
