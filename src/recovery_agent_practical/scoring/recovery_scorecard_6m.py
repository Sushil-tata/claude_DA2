"""
6-Month Recovery Scorecard
===========================

Predicts recovery potential over 180-day forward window from snapshot.

Target Definition:
-----------------
Binary: Will customer pay > threshold THB (default 500) within next 180 days?
Amount: Total THB recovered in next 180 days (for payers only)

Two Implementation Options:
---------------------------
Option A (Default): Two-part model
  - Part 1: GradientBoostingClassifier for P(any payment > threshold)
  - Part 2: GradientBoostingRegressor for E(amount | paid)
  - Expected recovery = P(pay) × E(amount | paid)

Option B: Tweedie regression (single-stage)
  - Handles zero-inflation natively
  - Outputs expected recovery directly
  - Simpler but less interpretable

Score Bands:
-----------
HOT (≥70%): High recovery probability
WARM (40-70%): Moderate recovery probability
COLD (15-40%): Low recovery probability
FROZEN (<15%): Minimal recovery probability
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import TweedieRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RecoveryScore:
    """Recovery score for one account"""
    account_id: str
    score_band: str  # HOT | WARM | COLD | FROZEN

    # Two-part model outputs
    p_recovery: float  # Probability of any payment
    expected_amount: float  # Expected amount if pays
    expected_recovery: float  # P(pay) × E(amount|paid)

    # Model version
    model_version: str
    model_type: str  # TWO_PART | TWEEDIE
    score_date: str


@dataclass
class ModelMetrics:
    """Training/validation metrics"""
    model_type: str

    # Classifier metrics (two-part only)
    auc_roc: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None

    # Regressor metrics
    mae: Optional[float] = None
    rmse: Optional[float] = None

    # Business metrics
    mean_expected_recovery: Optional[float] = None
    pct_hot: Optional[float] = None  # % scored as HOT
    pct_warm: Optional[float] = None
    pct_cold: Optional[float] = None
    pct_frozen: Optional[float] = None


# ─────────────────────────────────────────────────────────────────────────────
# TWO-PART MODEL
# ─────────────────────────────────────────────────────────────────────────────

class TwoPartRecoveryModel:
    """
    Two-part model: Classifier for P(pay) + Regressor for E(amount|paid).

    Handles zero-inflation explicitly by modeling binary outcome first,
    then amount conditional on payment.
    """

    def __init__(
        self,
        payment_threshold: float = 500.0,
        classifier_params: Optional[Dict] = None,
        regressor_params: Optional[Dict] = None,
    ):
        """
        Args:
            payment_threshold: Minimum THB to count as payment
            classifier_params: GradientBoostingClassifier hyperparameters
            regressor_params: GradientBoostingRegressor hyperparameters
        """
        self.payment_threshold = payment_threshold

        # Default hyperparameters
        clf_params = classifier_params or {
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.1,
            "min_samples_leaf": 50,
            "random_state": 42,
        }

        reg_params = regressor_params or {
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.1,
            "min_samples_leaf": 50,
            "random_state": 42,
        }

        self.classifier = GradientBoostingClassifier(**clf_params)
        self.regressor = GradientBoostingRegressor(**reg_params)

        self.feature_names: Optional[List[str]] = None
        self.is_fitted = False

    def fit(
        self,
        X: pd.DataFrame,
        y_recovery_amount: pd.Series,
        sample_weight: Optional[pd.Series] = None,
    ) -> ModelMetrics:
        """
        Train two-part model.

        Args:
            X: Feature matrix
            y_recovery_amount: Total recovery amount in next 180 days
            sample_weight: Optional sample weights

        Returns:
            ModelMetrics with training performance
        """
        self.feature_names = list(X.columns)

        # Part 1: Binary classifier (did customer pay > threshold?)
        y_binary = (y_recovery_amount > self.payment_threshold).astype(int)

        logger.info(f"Training classifier: {y_binary.sum()} payers out of {len(y_binary)} accounts")
        self.classifier.fit(X, y_binary, sample_weight=sample_weight)

        # Part 2: Amount regressor (conditional on payment)
        payer_mask = y_binary == 1
        X_payers = X[payer_mask]
        y_amounts = y_recovery_amount[payer_mask]
        weights_payers = sample_weight[payer_mask] if sample_weight is not None else None

        if len(X_payers) < 10:
            raise ValueError(f"Insufficient payer samples: {len(X_payers)}. Need at least 10.")

        logger.info(f"Training regressor on {len(X_payers)} payers")
        self.regressor.fit(X_payers, y_amounts, sample_weight=weights_payers)

        self.is_fitted = True

        # Compute metrics on training data
        metrics = self._compute_metrics(X, y_binary, y_recovery_amount)
        return metrics

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predict recovery for new accounts.

        Returns DataFrame with columns:
        - p_recovery: P(any payment > threshold)
        - expected_amount_if_pays: E(amount | paid)
        - expected_recovery: P(pay) × E(amount|paid)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if self.feature_names is not None:
            X = X[self.feature_names]

        # Part 1: Predict probability of payment
        p_recovery = self.classifier.predict_proba(X)[:, 1]

        # Part 2: Predict expected amount if pays
        expected_amount_if_pays = self.regressor.predict(X)
        expected_amount_if_pays = np.maximum(0, expected_amount_if_pays)  # Clip negatives

        # Combined: expected recovery
        expected_recovery = p_recovery * expected_amount_if_pays

        return pd.DataFrame({
            "p_recovery": p_recovery,
            "expected_amount_if_pays": expected_amount_if_pays,
            "expected_recovery": expected_recovery,
        })

    def _compute_metrics(
        self,
        X: pd.DataFrame,
        y_binary: pd.Series,
        y_amount: pd.Series,
    ) -> ModelMetrics:
        """Compute performance metrics"""
        # Classifier metrics
        y_pred_proba = self.classifier.predict_proba(X)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)

        auc = roc_auc_score(y_binary, y_pred_proba)
        precision = (y_pred & y_binary).sum() / y_pred.sum() if y_pred.sum() > 0 else 0.0
        recall = (y_pred & y_binary).sum() / y_binary.sum() if y_binary.sum() > 0 else 0.0

        # Regressor metrics (on payers only)
        payer_mask = y_binary == 1
        if payer_mask.sum() > 0:
            X_payers = X[payer_mask]
            y_true_amounts = y_amount[payer_mask]
            y_pred_amounts = self.regressor.predict(X_payers)

            mae = mean_absolute_error(y_true_amounts, y_pred_amounts)
            rmse = np.sqrt(mean_squared_error(y_true_amounts, y_pred_amounts))
        else:
            mae, rmse = None, None

        # Business metrics
        predictions = self.predict(X)
        mean_recovery = predictions["expected_recovery"].mean()

        # Score band distribution
        bands = self._assign_score_bands(predictions["p_recovery"])
        pct_hot = (bands == "HOT").mean() * 100
        pct_warm = (bands == "WARM").mean() * 100
        pct_cold = (bands == "COLD").mean() * 100
        pct_frozen = (bands == "FROZEN").mean() * 100

        return ModelMetrics(
            model_type="TWO_PART",
            auc_roc=auc,
            precision=precision,
            recall=recall,
            mae=mae,
            rmse=rmse,
            mean_expected_recovery=mean_recovery,
            pct_hot=pct_hot,
            pct_warm=pct_warm,
            pct_cold=pct_cold,
            pct_frozen=pct_frozen,
        )

    @staticmethod
    def _assign_score_bands(p_recovery: np.ndarray) -> np.ndarray:
        """Assign score bands based on probability"""
        return np.select(
            [p_recovery >= 0.70, p_recovery >= 0.40, p_recovery >= 0.15],
            ["HOT", "WARM", "COLD"],
            default="FROZEN"
        )


# ─────────────────────────────────────────────────────────────────────────────
# TWEEDIE MODEL (ALTERNATIVE)
# ─────────────────────────────────────────────────────────────────────────────

class TweedieRecoveryModel:
    """
    Single-stage Tweedie regression for recovery prediction.

    Handles zero-inflation natively via compound Poisson-Gamma distribution.
    Simpler than two-part but less interpretable.
    """

    def __init__(
        self,
        payment_threshold: float = 500.0,
        power: float = 1.5,  # Tweedie power parameter (1<p<2 for compound Poisson-Gamma)
        alpha: float = 0.1,  # L2 regularization
        max_iter: int = 100,
    ):
        """
        Args:
            payment_threshold: Minimum THB to count as meaningful recovery
            power: Tweedie variance power (1.5 typical for zero-inflated continuous)
            alpha: L2 penalty
            max_iter: Maximum iterations
        """
        self.payment_threshold = payment_threshold

        self.model = TweedieRegressor(
            power=power,
            alpha=alpha,
            max_iter=max_iter,
        )

        self.feature_names: Optional[List[str]] = None
        self.is_fitted = False

    def fit(
        self,
        X: pd.DataFrame,
        y_recovery_amount: pd.Series,
        sample_weight: Optional[pd.Series] = None,
    ) -> ModelMetrics:
        """Train Tweedie model"""
        self.feature_names = list(X.columns)

        logger.info(f"Training Tweedie model on {len(X)} accounts")
        self.model.fit(X, y_recovery_amount, sample_weight=sample_weight)

        self.is_fitted = True

        # Compute metrics
        y_pred = self.model.predict(X)
        mae = mean_absolute_error(y_recovery_amount, y_pred)
        rmse = np.sqrt(mean_squared_error(y_recovery_amount, y_pred))
        mean_recovery = y_pred.mean()

        # Convert to probability for score banding (rough approximation)
        # P(pay) ≈ min(1, expected_recovery / median_recovery_if_pays)
        payer_amounts = y_recovery_amount[y_recovery_amount > self.payment_threshold]
        median_payer = payer_amounts.median() if len(payer_amounts) > 0 else 1000.0

        p_recovery_approx = np.minimum(1.0, y_pred / median_payer)
        bands = TwoPartRecoveryModel._assign_score_bands(p_recovery_approx)

        return ModelMetrics(
            model_type="TWEEDIE",
            mae=mae,
            rmse=rmse,
            mean_expected_recovery=mean_recovery,
            pct_hot=(bands == "HOT").mean() * 100,
            pct_warm=(bands == "WARM").mean() * 100,
            pct_cold=(bands == "COLD").mean() * 100,
            pct_frozen=(bands == "FROZEN").mean() * 100,
        )

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict expected recovery"""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if self.feature_names is not None:
            X = X[self.feature_names]

        expected_recovery = self.model.predict(X)
        expected_recovery = np.maximum(0, expected_recovery)

        # Approximate probability (for score banding)
        payer_median = 1000.0  # Placeholder - should be fitted
        p_recovery_approx = np.minimum(1.0, expected_recovery / payer_median)

        return pd.DataFrame({
            "p_recovery": p_recovery_approx,
            "expected_amount_if_pays": expected_recovery / np.maximum(0.01, p_recovery_approx),
            "expected_recovery": expected_recovery,
        })


# ─────────────────────────────────────────────────────────────────────────────
# RECOVERY SCORECARD (UNIFIED INTERFACE)
# ─────────────────────────────────────────────────────────────────────────────

class RecoveryScorecard6M:
    """
    6-month recovery scorecard with unified interface.
    Supports both two-part and Tweedie models.
    """

    def __init__(
        self,
        model_type: Literal["TWO_PART", "TWEEDIE"] = "TWO_PART",
        payment_threshold: float = 500.0,
        model_params: Optional[Dict] = None,
    ):
        """
        Args:
            model_type: Model architecture (TWO_PART or TWEEDIE)
            payment_threshold: Minimum THB to count as recovery
            model_params: Model-specific hyperparameters
        """
        self.model_type = model_type
        self.payment_threshold = payment_threshold
        self.model_version = f"{model_type}_v1.0"

        if model_type == "TWO_PART":
            self.model = TwoPartRecoveryModel(
                payment_threshold=payment_threshold,
                **(model_params or {})
            )
        elif model_type == "TWEEDIE":
            self.model = TweedieRecoveryModel(
                payment_threshold=payment_threshold,
                **(model_params or {})
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Use TWO_PART or TWEEDIE.")

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        sample_weight: Optional[pd.Series] = None,
    ) -> Dict[str, ModelMetrics]:
        """
        Train scorecard.

        Args:
            X_train: Training features
            y_train: Training labels (recovery amount in 180d)
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            sample_weight: Sample weights (optional)

        Returns:
            Dict with "train" and "val" metrics
        """
        logger.info(f"Training {self.model_type} model on {len(X_train)} accounts")

        train_metrics = self.model.fit(X_train, y_train, sample_weight)

        metrics = {"train": train_metrics}

        # Validation metrics
        if X_val is not None and y_val is not None:
            val_metrics = self._evaluate(X_val, y_val)
            metrics["val"] = val_metrics

        return metrics

    def score_batch(
        self,
        df: pd.DataFrame,
        score_date: str,
    ) -> pd.DataFrame:
        """
        Score a batch of accounts.

        Args:
            df: DataFrame with features + account_id
            score_date: Score date (YYYY-MM-DD format)

        Returns:
            DataFrame with RecoveryScore columns
        """
        account_ids = df["account_id"].astype(str)

        # Extract features (drop account_id)
        X = df.drop(columns=["account_id"], errors="ignore")

        # Predict
        predictions = self.model.predict(X)

        # Assign score bands
        score_bands = self._assign_bands(predictions["p_recovery"].values)

        # Build output
        scores = pd.DataFrame({
            "account_id": account_ids,
            "score_band": score_bands,
            "p_recovery": predictions["p_recovery"],
            "expected_amount_if_pays": predictions["expected_amount_if_pays"],
            "expected_recovery": predictions["expected_recovery"],
            "model_version": self.model_version,
            "model_type": self.model_type,
            "score_date": score_date,
        })

        return scores

    def _evaluate(self, X: pd.DataFrame, y: pd.Series) -> ModelMetrics:
        """Evaluate on validation set"""
        if self.model_type == "TWO_PART":
            y_binary = (y > self.payment_threshold).astype(int)
            return self.model._compute_metrics(X, y_binary, y)
        else:
            # Tweedie model evaluation
            y_pred = self.model.predict(X)["expected_recovery"]
            mae = mean_absolute_error(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))

            return ModelMetrics(
                model_type="TWEEDIE",
                mae=mae,
                rmse=rmse,
                mean_expected_recovery=y_pred.mean(),
            )

    @staticmethod
    def _assign_bands(p_recovery: np.ndarray) -> np.ndarray:
        """Assign score bands"""
        return np.select(
            [p_recovery >= 0.70, p_recovery >= 0.40, p_recovery >= 0.15],
            ["HOT", "WARM", "COLD"],
            default="FROZEN"
        )
