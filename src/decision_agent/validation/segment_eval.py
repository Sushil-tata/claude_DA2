"""
Segment-based model evaluation.
"""
import logging
import numpy as np
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


class SegmentEvaluator:
    """
    Evaluate model performance across customer segments.

    Ensures model works well across different customer groups.
    """

    def __init__(self, segment_dimensions: List[Dict[str, Any]]):
        """
        Initialize segment evaluator.

        Args:
            segment_dimensions: List of segment definitions, e.g.:
                [{"dimension": "income_quartile", "values": ["Q1", "Q2", "Q3", "Q4"]}]
        """
        self.segment_dimensions = segment_dimensions

    def evaluate(self, features_df, predictions: np.ndarray, target_col: str) -> Dict[str, Any]:
        """
        Evaluate model performance across segments.

        Args:
            features_df: DataFrame with features and target
            predictions: Model predictions
            target_col: Name of target column

        Returns:
            Dictionary of segment metrics
        """
        logger.info("Evaluating model performance across segments...")

        # Convert to pandas if needed
        pdf = self._to_pandas(features_df)

        # Add predictions to dataframe
        pdf = pdf.copy()
        pdf["_prediction"] = predictions

        results = {}

        for segment_def in self.segment_dimensions:
            dimension = segment_def["dimension"]

            # Create segment column if it doesn't exist
            if dimension not in pdf.columns:
                pdf = self._create_segment_column(pdf, dimension, target_col)

            # Compute metrics for each segment value
            segment_metrics = {}

            unique_values = pdf[dimension].unique()
            logger.info(f"Evaluating {dimension}: {unique_values}")

            for segment_value in unique_values:
                mask = pdf[dimension] == segment_value
                n_samples = mask.sum()

                if n_samples < 10:  # Skip small segments
                    logger.warning(f"Skipping {dimension}={segment_value}: only {n_samples} samples")
                    continue

                y_true = pdf.loc[mask, target_col].values
                y_pred = pdf.loc[mask, "_prediction"].values

                metrics = self._compute_metrics(y_true, y_pred)
                metrics["n_samples"] = int(n_samples)

                segment_metrics[str(segment_value)] = metrics

                logger.info(
                    f"  {dimension}={segment_value}: "
                    f"MAE={metrics.get('mae', 0):.2f}, "
                    f"R2={metrics.get('r2', 0):.3f}, "
                    f"n={metrics['n_samples']}"
                )

            results[dimension] = segment_metrics

        logger.info("Segment evaluation completed")
        return results

    def _to_pandas(self, df):
        """Convert Spark DataFrame to pandas if needed"""
        try:
            from pyspark.sql import DataFrame as SparkDataFrame

            if isinstance(df, SparkDataFrame):
                return df.toPandas()
        except ImportError:
            pass

        return df

    def _create_segment_column(self, pdf, dimension: str, target_col: str):
        """
        Create segment column based on dimension.

        For income_quartile: creates Q1-Q4 based on target values.
        """
        if dimension == "income_quartile":
            # Create quartiles based on target values
            pdf["income_quartile"] = pd.qcut(
                pdf[target_col],
                q=4,
                labels=["Q1", "Q2", "Q3", "Q4"],
                duplicates="drop"
            )
            logger.info(f"Created {dimension} column with quartiles")

        return pdf

    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute evaluation metrics"""
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        metrics = {
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "r2": float(r2_score(y_true, y_pred)),
            "mean_true": float(np.mean(y_true)),
            "mean_pred": float(np.mean(y_pred))
        }

        return metrics


# Import pandas for segment creation
import pandas as pd
