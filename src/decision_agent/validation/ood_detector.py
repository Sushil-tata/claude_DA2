"""
Out-of-Distribution (OOD) Detector

Detects customers whose features fall outside the training distribution.
Uses Isolation Forest to flag anomalous feature combinations.

Why This Matters:
- Model unreliable on OOD inputs
- Need to route OOD cases to Champion or manual review
- Prevents silent model failures

Integration: Fit during training, score at inference
"""

import logging
from typing import Dict
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from sklearn.ensemble import Isolation Forest
import numpy as np

logger = logging.getLogger(__name__)


class OODDetector:
    """Detect out-of-distribution inputs using Isolation Forest."""

    def __init__(self, contamination: float = 0.05):
        """
        Args:
            contamination: Expected proportion of outliers (default 5%)
        """
        self.contamination = contamination
        self.model = IsolationForest(contamination=contamination, random_state=42)
        self.fitted = False

    def fit(self, train_df: DataFrame, feature_cols: list):
        """
        Fit OOD detector on training data.

        Args:
            train_df: Training DataFrame
            feature_cols: Feature column names
        """
        logger.info("Fitting OOD detector on training data...")

        from decision_agent.utils.spark_guards import safe_to_pandas

        # Sample for fitting (don't need all data)
        train_sample = safe_to_pandas(train_df.sample(fraction=0.1), max_rows=10000)

        X = train_sample[feature_cols]
        self.model.fit(X)
        self.fitted = True

        logger.info("OOD detector fitted.")

    def score(self, df: DataFrame, feature_cols: list) -> DataFrame:
        """
        Score inputs for OOD.

        Args:
            df: DataFrame to score
            feature_cols: Feature column names

        Returns:
            DataFrame with ood_score column (lower = more anomalous)
        """
        if not self.fitted:
            raise ValueError("OOD detector not fitted. Call fit() first.")

        logger.info("Scoring for OOD...")

        from decision_agent.utils.spark_guards import safe_to_pandas

        # Convert to pandas for sklearn prediction
        pdf = safe_to_pandas(df, max_rows=100000)

        X = pdf[feature_cols]
        ood_scores = self.model.score_samples(X)

        pdf['ood_score'] = ood_scores
        pdf['is_ood'] = (ood_scores < np.percentile(ood_scores, self.contamination * 100))

        # Convert back to Spark
        spark = df.sparkSession
        result_df = spark.createDataFrame(pdf)

        logger.info(f"OOD scoring complete. {result_df.filter(F.col('is_ood') == True).count()} flagged as OOD")

        return result_df
