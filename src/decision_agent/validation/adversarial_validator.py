"""
Adversarial Validation - Train/Test Distribution Shift Detector

Detects if training and test sets come from different distributions.
Uses adversarial approach: train classifier to distinguish train vs test.
If classifier AUC > 0.55, distributions are significantly different.

Why This Matters:
- Data leakage detection
- Temporal drift detection
- Feature distribution shift

Integration: Called post-split in training_harness.py
"""

import logging
from typing import Dict, Tuple
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class AdversarialValidator:
    """Detect train/test distribution shift using adversarial validation."""

    def __init__(self, threshold_auc: float = 0.55):
        self.threshold_auc = threshold_auc

    def validate(
        self,
        train_df: DataFrame,
        test_df: DataFrame,
        feature_cols: list
    ) -> Tuple[bool, Dict]:
        """
        Validate train/test similarity.

        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame
            feature_cols: Feature column names

        Returns:
            Tuple of (passed, results)
        """
        logger.info("Running adversarial validation...")

        from decision_agent.utils.spark_guards import safe_to_pandas

        # Sample and convert to pandas
        train_sample = safe_to_pandas(train_df.sample(fraction=0.1), max_rows=10000)
        test_sample = safe_to_pandas(test_df.sample(fraction=0.1), max_rows=10000)

        # Create labels: 0=train, 1=test
        train_sample['is_test'] = 0
        test_sample['is_test'] = 1

        combined = pd.concat([train_sample, test_sample], axis=0)

        # Train classifier
        X = combined[feature_cols]
        y = combined['is_test']

        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        clf.fit(X, y)

        # Compute AUC
        y_pred_proba = clf.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, y_pred_proba)

        passed = auc <= self.threshold_auc

        results = {
            "adversarial_auc": float(auc),
            "threshold": self.threshold_auc,
            "passed": passed,
            "interpretation": "Similar distributions" if passed else "Distributions differ significantly"
        }

        logger.info(f"Adversarial validation: AUC={auc:.3f}, {'PASS' if passed else 'FAIL'}")

        return passed, results
