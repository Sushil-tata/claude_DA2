"""
Leakage Detector
================
Detects data leakage before model training.

Three detection strategies:
  1. Temporal availability — features must exist before the label timestamp
  2. Target correlation — features with suspiciously high correlation to target
     (Pearson for numeric, Cramer's V for categorical)
  3. Adversarial validation — if a RandomForest can distinguish train from test,
     the distributions are too different (AUC > threshold = possible leakage)

Usage:
    detector = LeakageDetector()
    report   = detector.check_all(train_df, test_df, target_col="outcome_pay_any",
                                  feature_cols=feature_cols)
    if report["leakage_detected"]:
        print(report["flagged_features"])
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

logger = logging.getLogger(__name__)


class LeakageDetector:
    """
    Detects data leakage via temporal, correlation, and adversarial checks.
    """

    def __init__(
        self,
        temporal_threshold_days: int = 0,
        correlation_threshold: float = 0.90,
        adversarial_auc_threshold: float = 0.55,
        cramers_v_threshold: float = 0.85,
    ):
        self.temporal_threshold_days    = temporal_threshold_days
        self.correlation_threshold      = correlation_threshold
        self.adversarial_auc_threshold  = adversarial_auc_threshold
        self.cramers_v_threshold        = cramers_v_threshold

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC
    # ──────────────────────────────────────────────────────────────────────────

    def check_all(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_col: str,
        feature_cols: List[str],
        timestamp_col: Optional[str] = None,
        label_timestamp_col: Optional[str] = None,
    ) -> Dict:
        """
        Run all leakage checks. Returns summary report.
        """
        report: Dict = {
            "leakage_detected": False,
            "flagged_features": [],
            "checks": {},
        }

        # 1. Temporal check
        if timestamp_col and label_timestamp_col:
            temporal = self.check_temporal(train_df, timestamp_col, label_timestamp_col)
            report["checks"]["temporal"] = temporal
            if temporal["leakage_rows"] > 0:
                report["leakage_detected"] = True

        # 2. Target correlation check
        correlation = self.check_target_correlation(
            train_df, target_col, feature_cols
        )
        report["checks"]["target_correlation"] = correlation
        if correlation["flagged"]:
            report["leakage_detected"] = True
            report["flagged_features"].extend(correlation["flagged"])

        # 3. Adversarial validation
        adversarial = self.check_adversarial(train_df, test_df, feature_cols)
        report["checks"]["adversarial"] = adversarial
        if adversarial["distributions_differ"]:
            report["leakage_detected"] = True

        logger.info(
            "Leakage check complete. leakage_detected=%s, flagged=%s",
            report["leakage_detected"], report["flagged_features"]
        )
        return report

    def check_temporal(
        self,
        df: pd.DataFrame,
        feature_timestamp_col: str,
        label_timestamp_col: str,
    ) -> Dict:
        """
        Flag rows where feature_timestamp > label_timestamp (future leakage).
        """
        if feature_timestamp_col not in df.columns or label_timestamp_col not in df.columns:
            return {"leakage_rows": 0, "message": "Timestamp columns not found"}

        ft = pd.to_datetime(df[feature_timestamp_col], errors="coerce")
        lt = pd.to_datetime(df[label_timestamp_col],   errors="coerce")
        leakage_mask = ft > lt
        n_leakage    = int(leakage_mask.sum())

        if n_leakage > 0:
            logger.warning(
                "TEMPORAL LEAKAGE: %d rows have features from the future!", n_leakage
            )

        return {
            "leakage_rows": n_leakage,
            "total_rows":   len(df),
            "leakage_pct":  round(n_leakage / max(len(df), 1) * 100, 2),
        }

    def check_target_correlation(
        self,
        df: pd.DataFrame,
        target_col: str,
        feature_cols: List[str],
    ) -> Dict:
        """
        Flag features with suspiciously high correlation to target.
        Uses Pearson for continuous, Cramer's V for categorical.
        """
        if target_col not in df.columns:
            return {"flagged": [], "scores": {}}

        y = df[target_col]
        scores: Dict[str, float] = {}
        flagged: List[str] = []

        for col in feature_cols:
            if col not in df.columns:
                continue
            try:
                if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                    corr = abs(df[col].corr(y))
                    scores[col] = round(float(corr), 4)
                    if corr > self.correlation_threshold:
                        flagged.append(col)
                        logger.warning(
                            "HIGH CORRELATION: %s vs %s = %.3f (threshold=%.2f)",
                            col, target_col, corr, self.correlation_threshold
                        )
                else:
                    v = self._cramers_v(df[col].astype(str), y.astype(str))
                    scores[col] = round(float(v), 4)
                    if v > self.cramers_v_threshold:
                        flagged.append(col)
                        logger.warning(
                            "HIGH CRAMERS_V: %s vs %s = %.3f (threshold=%.2f)",
                            col, target_col, v, self.cramers_v_threshold
                        )
            except Exception as e:
                logger.debug("Could not check %s: %s", col, e)

        return {"flagged": flagged, "scores": scores}

    def check_adversarial(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_cols: List[str],
        n_estimators: int = 50,
    ) -> Dict:
        """
        Adversarial validation: train RandomForest to distinguish train vs test.
        AUC > threshold -> distributions are too different -> possible leakage/shift.
        """
        available = [c for c in feature_cols
                     if c in train_df.columns and c in test_df.columns]
        if not available:
            return {"adversarial_auc": float("nan"), "distributions_differ": False}

        X_train = train_df[available].fillna(0).values.astype(float)
        X_test  = test_df[available].fillna(0).values.astype(float)

        n_train = min(len(X_train), 5000)
        n_test  = min(len(X_test),  5000)
        X_train = X_train[:n_train]
        X_test  = X_test[:n_test]

        X_combined = np.vstack([X_train, X_test])
        y_combined = np.concatenate([
            np.zeros(n_train),
            np.ones(n_test),
        ])

        clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        try:
            auc_scores = cross_val_score(
                clf, X_combined, y_combined, cv=3, scoring="roc_auc"
            )
            mean_auc = float(auc_scores.mean())
        except Exception as e:
            logger.warning("Adversarial validation failed: %s", e)
            mean_auc = float("nan")

        differs = (not np.isnan(mean_auc)) and (mean_auc > self.adversarial_auc_threshold)

        if differs:
            logger.warning(
                "DISTRIBUTION SHIFT: adversarial AUC=%.3f > threshold=%.2f",
                mean_auc, self.adversarial_auc_threshold
            )

        return {
            "adversarial_auc":      round(mean_auc, 4),
            "threshold":            self.adversarial_auc_threshold,
            "distributions_differ": differs,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # PRIVATE
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _cramers_v(x: pd.Series, y: pd.Series) -> float:
        """Cramer's V association for categorical vs categorical/binary."""
        from scipy.stats import chi2_contingency

        contingency = pd.crosstab(x, y)
        chi2, _, _, _ = chi2_contingency(contingency)
        n = contingency.sum().sum()
        min_dim = min(contingency.shape) - 1
        if min_dim <= 0 or n <= 0:
            return 0.0
        return float(np.sqrt(chi2 / (n * min_dim)))
