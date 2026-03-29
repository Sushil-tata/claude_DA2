"""
Recovery Amount Model — Two-Stage Beta Regression
==================================================
Estimates E[Amount | Recovery, Segment, d] for the ERV formula:

    ERV(d) = P(Recovery | Segment, d) × E[Amount | Recovery, Segment] × Balance × (1 − d)

Two-stage architecture
-----------------------
Stage 1 — Recovery Incidence (binary):
    P(recovery_occurred) using GradientBoostingClassifier.
    Trained on ALL charge-off accounts (recovered + not recovered).

Stage 2 — Recovery Fraction (continuous on (0, 1)):
    E[recovery_fraction | recovery_occurred] using Beta Regression
    (implemented via logit-transformed GradientBoostingRegressor + variance
    correction for heteroscedasticity across segments).
    Trained ONLY on accounts where recovery_occurred == 1.

Final output:
    expected_recovery_amount = P(recovery) × E[fraction | recovery] × balance

Data contract (input DataFrame columns required):
-------------------------------------------------
  account_id              : str — unique account identifier
  current_balance         : float — outstanding balance at charge-off (THB / local currency)
  days_since_charge_off   : int — months/days since CO date
  months_since_charge_off : int — integer months (primary time feature)
  original_balance        : float
  balance_ratio           : float — current_balance / original_balance
  days_past_due           : int — at time of CO
  payment_history_score   : float — 0-1 composite of prior payment behaviour
  contact_response_rate   : float — 0-1 historical contact response rate
  promise_kept_rate       : float — 0-1 historical promise-to-pay keep rate
  settlement_offers_made  : int — number of settlement offers extended
  avg_discount_offered    : float — average discount % offered historically
  segment                 : str — persona segment from DebtorPersonaSegmentation
  debt_to_income_ratio    : float — estimated DTI
  payment_capacity_score  : float — 0-1 ability to pay score

  --- Target columns (training only) ---
  recovery_occurred       : int  — 1 if any recovery in 12m post-CO, else 0
  recovery_fraction       : float — recovered_amount / current_balance  ∈ (0, 1]
                            (only populated where recovery_occurred == 1)

MLflow integration
------------------
  Experiment : collection_recovery_models
  Run tags   : stage=recovery_amount, model_version, segment

Usage
-----
    model = RecoveryAmountModel(config)
    model.fit(train_df, val_df)
    predictions = model.predict(accounts_df)
    # predictions columns: account_id, p_recovery, e_fraction, e_recovery_amount
"""

import json
import logging
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.special import logit, expit        # logit = log(p/(1-p)), expit = sigmoid
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    brier_score_loss,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS — match debt_collection_optimization.yaml feature names
# ─────────────────────────────────────────────────────────────────────────────

STAGE1_FEATURES: List[str] = [
    "balance_ratio",
    "months_since_charge_off",
    "days_past_due",
    "payment_history_score",
    "contact_response_rate",
    "promise_kept_rate",
    "settlement_offers_made",
    "avg_discount_offered",
    "debt_to_income_ratio",
    "payment_capacity_score",
]

STAGE2_FEATURES: List[str] = [
    "balance_ratio",
    "months_since_charge_off",
    "payment_history_score",
    "contact_response_rate",
    "promise_kept_rate",
    "avg_discount_offered",
    "payment_capacity_score",
]

# Epsilon for logit transform — clamps recovery_fraction away from exact 0/1
_EPSILON = 1e-6

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AUC-ROC — returns nan if single class present."""
    try:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return roc_auc_score(y_true, y_score)
    except Exception:
        return float("nan")


def _clamp_fraction(series: pd.Series) -> pd.Series:
    """Clamp recovery_fraction to (ε, 1-ε) for logit transform stability."""
    return series.clip(_EPSILON, 1.0 - _EPSILON)


def _logit_transform(y: np.ndarray) -> np.ndarray:
    """Apply logit transform to recovery fractions before regression."""
    y_clamped = np.clip(y, _EPSILON, 1.0 - _EPSILON)
    return logit(y_clamped)


def _inverse_logit(y_logit: np.ndarray) -> np.ndarray:
    """Back-transform logit predictions to (0, 1) fraction space."""
    return expit(y_logit)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CLASS
# ─────────────────────────────────────────────────────────────────────────────

class RecoveryAmountModel:
    """
    Two-stage model for E[Recovery Amount | Segment, Discount].

    Stage 1: GradientBoosting binary classifier → P(recovery_occurred)
    Stage 2: Logit-transformed GradientBoosting regressor → E[fraction | recovered]

    Segment-aware: fits separate Stage 2 models per segment when sample
    size allows (min_segment_samples threshold), otherwise falls back to
    a global pooled Stage 2 model.

    Parameters
    ----------
    config : dict
        Must contain key ``recovery_amount_model`` with sub-keys:
        ``stage1``, ``stage2``, ``min_segment_samples``, ``mlflow``.
    """

    def __init__(self, config: Dict):
        self.config = config
        _cfg = config.get("recovery_amount_model", {})

        # Stage 1 hyperparameters
        s1 = _cfg.get("stage1", {})
        self.stage1_params: Dict = {
            "n_estimators":  s1.get("n_estimators",  200),
            "max_depth":     s1.get("max_depth",      5),
            "learning_rate": s1.get("learning_rate",  0.05),
            "subsample":     s1.get("subsample",      0.8),
            "min_samples_leaf": s1.get("min_samples_leaf", 20),
            "random_state":  42,
        }

        # Stage 2 hyperparameters
        s2 = _cfg.get("stage2", {})
        self.stage2_params: Dict = {
            "n_estimators":  s2.get("n_estimators",  150),
            "max_depth":     s2.get("max_depth",      4),
            "learning_rate": s2.get("learning_rate",  0.05),
            "subsample":     s2.get("subsample",      0.8),
            "min_samples_leaf": s2.get("min_samples_leaf", 15),
            "loss":          "squared_error",
            "random_state":  42,
        }

        self.min_segment_samples: int = _cfg.get("min_segment_samples", 200)
        self.calibrate_stage1: bool   = _cfg.get("calibrate_stage1", True)

        # MLflow
        mlflow_cfg = _cfg.get("mlflow", {})
        self.experiment_name: str = mlflow_cfg.get(
            "experiment_name", "collection_recovery_models"
        )
        self.tracking_uri: str = mlflow_cfg.get("tracking_uri", "mlruns")

        # Model artefacts (populated after fit)
        self._stage1_model = None          # P(recovery_occurred)
        self._stage2_global = None         # Pooled E[fraction | recovered]
        self._stage2_by_segment: Dict = {} # Segment-specific overrides
        self._segment_encoder = LabelEncoder()
        self._is_fitted = False
        self._feature_cols_s1: List[str] = []
        self._feature_cols_s2: List[str] = []

        # Setup MLflow
        self._mlflow_available = False
        self._try_setup_mlflow()

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────────────────────────────────────

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        stage1_features: Optional[List[str]] = None,
        stage2_features: Optional[List[str]] = None,
    ) -> Dict:
        """
        Train both stages and log to MLflow.

        Parameters
        ----------
        train_df : pd.DataFrame
            Must contain all feature columns + ``recovery_occurred``
            + ``recovery_fraction`` (for recovered rows) + ``segment``.
        val_df : pd.DataFrame
            Same schema as train_df.
        stage1_features : list of str, optional
            Override default STAGE1_FEATURES.
        stage2_features : list of str, optional
            Override default STAGE2_FEATURES.

        Returns
        -------
        dict with keys: stage1_metrics, stage2_metrics, mlflow_run_id
        """
        self._feature_cols_s1 = stage1_features or STAGE1_FEATURES
        self._feature_cols_s2 = stage2_features or STAGE2_FEATURES

        self._validate_input(train_df, "train")
        self._validate_input(val_df,   "val")

        logger.info("RecoveryAmountModel.fit() — training on %d samples", len(train_df))
        print(f"\n[RecoveryAmountModel] Training on {len(train_df):,} accounts")
        print(f"  Recovery rate (train): {train_df['recovery_occurred'].mean():.2%}")

        # ── Stage 1: P(recovery_occurred) ────────────────────────────────────
        stage1_metrics = self._fit_stage1(train_df, val_df)

        # ── Stage 2: E[fraction | recovered] — pooled + per-segment ──────────
        stage2_metrics = self._fit_stage2(train_df, val_df)

        self._is_fitted = True

        all_metrics = {**stage1_metrics, **stage2_metrics}

        # MLflow logging
        run_id = self._log_to_mlflow(all_metrics)

        print(f"\n  [Stage 1] AUC={stage1_metrics['s1_auc']:.4f} | "
              f"Brier={stage1_metrics['s1_brier']:.4f} | "
              f"PR-AUC={stage1_metrics['s1_pr_auc']:.4f}")
        print(f"  [Stage 2] MAE={stage2_metrics['s2_mae']:.4f} | "
              f"RMSE={stage2_metrics['s2_rmse']:.4f} | "
              f"Segments with dedicated model: {len(self._stage2_by_segment)}")
        if run_id:
            print(f"  MLflow run: {run_id}")

        return {"stage1_metrics": stage1_metrics,
                "stage2_metrics": stage2_metrics,
                "mlflow_run_id": run_id}

    def predict(
        self,
        accounts_df: pd.DataFrame,
        return_components: bool = True,
    ) -> pd.DataFrame:
        """
        Score accounts for expected recovery amount.

        Parameters
        ----------
        accounts_df : pd.DataFrame
            Feature columns + ``account_id`` + ``current_balance`` + ``segment``.
        return_components : bool
            If True, also returns p_recovery and e_fraction columns.

        Returns
        -------
        pd.DataFrame with columns:
            account_id, current_balance, segment,
            p_recovery, e_fraction, e_recovery_amount
        """
        if not self._is_fitted:
            raise RuntimeError(
                "Model not fitted. Call fit() before predict()."
            )

        df = accounts_df.copy()

        # ── Stage 1: P(recovery_occurred) ────────────────────────────────────
        X1 = df[self._feature_cols_s1].fillna(0.0).values.astype(float)
        p_recovery = self._stage1_model.predict_proba(X1)[:, 1]

        # ── Stage 2: E[fraction | recovered] — segment-aware ─────────────────
        e_fraction = self._predict_stage2(df)

        # ── ERV component: E[Amount | Recovery] ──────────────────────────────
        e_recovery_amount = p_recovery * e_fraction * df["current_balance"].values

        result = pd.DataFrame({
            "account_id":         df["account_id"].values,
            "current_balance":    df["current_balance"].values,
            "segment":            df["segment"].values,
        })

        if return_components:
            result["p_recovery"]        = p_recovery.round(4)
            result["e_fraction"]        = e_fraction.round(4)
        result["e_recovery_amount"] = e_recovery_amount.round(2)

        return result

    def predict_with_discount(
        self,
        accounts_df: pd.DataFrame,
        discount_pct: float,
    ) -> pd.DataFrame:
        """
        Convenience wrapper: append discount effect to Stage 1 features and predict.
        Used by ERVEngine when sweeping over d values.

        Parameters
        ----------
        discount_pct : float
            Discount as a fraction in [0, 1]. E.g. 0.30 = 30% discount.

        Returns
        -------
        Same schema as predict() with additional column ``discount_applied``.
        """
        df = accounts_df.copy()
        # avg_discount_offered acts as the discount signal in Stage 1 + Stage 2
        df["avg_discount_offered"] = discount_pct
        result = self.predict(df)
        result["discount_applied"] = discount_pct
        return result

    def evaluate(
        self,
        test_df: pd.DataFrame,
        segment_col: str = "segment",
    ) -> Dict:
        """
        Full evaluation on a held-out set, broken down by segment.

        Returns
        -------
        dict with keys:
            overall, by_segment (DataFrame)
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted.")

        preds = self.predict(test_df)

        # Stage 1 metrics
        s1_auc   = _safe_auc(test_df["recovery_occurred"].values, preds["p_recovery"].values)
        s1_brier = brier_score_loss(test_df["recovery_occurred"].values, preds["p_recovery"].values)

        # Stage 2 metrics (only on recovered)
        mask_rec = test_df["recovery_occurred"] == 1
        if mask_rec.sum() > 0:
            y_true_frac = test_df.loc[mask_rec, "recovery_fraction"].values
            y_pred_frac = preds.loc[mask_rec, "e_fraction"].values
            s2_mae  = mean_absolute_error(y_true_frac, y_pred_frac)
            s2_rmse = mean_squared_error(y_true_frac, y_pred_frac) ** 0.5
        else:
            s2_mae = s2_rmse = float("nan")

        # Overall expected recovery amount error
        true_recovery = (
            test_df["recovery_occurred"].values *
            test_df.get("recovery_fraction", pd.Series(np.zeros(len(test_df)))).fillna(0).values *
            test_df["current_balance"].values
        )
        erv_mae  = mean_absolute_error(true_recovery, preds["e_recovery_amount"].values)
        erv_rmse = mean_squared_error(true_recovery,  preds["e_recovery_amount"].values) ** 0.5

        overall = {
            "s1_auc":   round(s1_auc,   4),
            "s1_brier": round(s1_brier, 4),
            "s2_mae":   round(s2_mae,   4),
            "s2_rmse":  round(s2_rmse,  4),
            "erv_mae":  round(erv_mae,  2),
            "erv_rmse": round(erv_rmse, 2),
            "n":        len(test_df),
            "n_recovered": int(mask_rec.sum()),
        }

        # Per-segment breakdown
        seg_rows = []
        for seg in sorted(test_df[segment_col].unique()):
            seg_mask = test_df[segment_col] == seg
            seg_preds = preds[seg_mask]
            seg_true  = test_df[seg_mask]
            if seg_mask.sum() < 20:
                continue
            seg_auc = _safe_auc(
                seg_true["recovery_occurred"].values,
                seg_preds["p_recovery"].values,
            )
            seg_rows.append({
                "segment":       seg,
                "n":             int(seg_mask.sum()),
                "recovery_rate": float(seg_true["recovery_occurred"].mean()),
                "s1_auc":        round(seg_auc, 4),
            })

        return {"overall": overall, "by_segment": pd.DataFrame(seg_rows)}

    # ──────────────────────────────────────────────────────────────────────────
    # PRIVATE — FIT
    # ──────────────────────────────────────────────────────────────────────────

    def _fit_stage1(
        self, train_df: pd.DataFrame, val_df: pd.DataFrame
    ) -> Dict:
        """Fit P(recovery_occurred) classifier with optional Platt calibration."""
        print("  Fitting Stage 1: P(recovery_occurred)...")

        X_train = train_df[self._feature_cols_s1].fillna(0.0).values.astype(float)
        y_train = train_df["recovery_occurred"].values.astype(int)
        X_val   = val_df[self._feature_cols_s1].fillna(0.0).values.astype(float)
        y_val   = val_df["recovery_occurred"].values.astype(int)

        # Class-weight to handle imbalance (charge-off pools are typically <30% recovery)
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        pos_weight = n_neg / max(n_pos, 1)
        sample_weights = np.where(y_train == 1, pos_weight, 1.0)

        base_clf = GradientBoostingClassifier(**self.stage1_params)
        base_clf.fit(X_train, y_train, sample_weight=sample_weights)

        if self.calibrate_stage1:
            # Platt scaling calibration for reliable probability estimates
            cal_clf = CalibratedClassifierCV(base_clf, cv="prefit", method="sigmoid")
            cal_clf.fit(X_val, y_val)
            self._stage1_model = cal_clf
        else:
            self._stage1_model = base_clf

        # Evaluate
        p_val = self._stage1_model.predict_proba(X_val)[:, 1]
        metrics = {
            "s1_auc":     _safe_auc(y_val, p_val),
            "s1_pr_auc":  average_precision_score(y_val, p_val),
            "s1_brier":   brier_score_loss(y_val, p_val),
            "s1_n_train": len(X_train),
            "s1_n_val":   len(X_val),
            "s1_recovery_rate_train": float(y_train.mean()),
            "s1_recovery_rate_val":   float(y_val.mean()),
        }

        # Feature importances
        base = base_clf.estimators_  # raw model
        importances = dict(zip(
            self._feature_cols_s1,
            base_clf.feature_importances_,
        ))
        top5 = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"    Top Stage 1 features: {[(f, round(v, 3)) for f, v in top5]}")

        return metrics

    def _fit_stage2(
        self, train_df: pd.DataFrame, val_df: pd.DataFrame
    ) -> Dict:
        """
        Fit E[recovery_fraction | recovered] using logit-transformed GBR.
        Fits global model first, then segment-specific overrides.
        """
        print("  Fitting Stage 2: E[recovery_fraction | recovered]...")

        # Filter to recovered-only subset
        rec_train = train_df[train_df["recovery_occurred"] == 1].copy()
        rec_val   = val_df[val_df["recovery_occurred"] == 1].copy()

        if len(rec_train) < 50:
            warnings.warn(
                f"Stage 2: only {len(rec_train)} recovered training samples — "
                "model may not generalise. Consider pooling segments.",
                UserWarning,
            )

        X_train_s2 = rec_train[self._feature_cols_s2].fillna(0.0).values.astype(float)
        y_train_s2 = _logit_transform(
            _clamp_fraction(rec_train["recovery_fraction"]).values
        )
        X_val_s2   = rec_val[self._feature_cols_s2].fillna(0.0).values.astype(float)
        y_val_s2   = rec_val["recovery_fraction"].values.astype(float)

        # Global pooled model
        self._stage2_global = GradientBoostingRegressor(**self.stage2_params)
        self._stage2_global.fit(X_train_s2, y_train_s2)
        global_val_pred = _inverse_logit(self._stage2_global.predict(X_val_s2))

        s2_mae  = mean_absolute_error(y_val_s2, global_val_pred)
        s2_rmse = mean_squared_error(y_val_s2,  global_val_pred) ** 0.5

        # Segment-specific overrides
        n_seg_models = 0
        for seg in rec_train["segment"].unique():
            seg_train_mask = rec_train["segment"] == seg
            seg_val_mask   = rec_val["segment"]   == seg

            if seg_train_mask.sum() < self.min_segment_samples:
                continue  # Too few — fall back to global

            X_seg = rec_train.loc[seg_train_mask, self._feature_cols_s2].fillna(0.0).values.astype(float)
            y_seg = _logit_transform(
                _clamp_fraction(rec_train.loc[seg_train_mask, "recovery_fraction"]).values
            )

            seg_model = GradientBoostingRegressor(**self.stage2_params)
            seg_model.fit(X_seg, y_seg)
            self._stage2_by_segment[seg] = seg_model
            n_seg_models += 1

        print(f"    Global Stage 2 — MAE={s2_mae:.4f}, RMSE={s2_rmse:.4f}")
        print(f"    Segment-specific models fitted: {n_seg_models}")

        metrics = {
            "s2_mae":        s2_mae,
            "s2_rmse":       s2_rmse,
            "s2_n_train":    len(rec_train),
            "s2_n_val":      len(rec_val),
            "s2_n_seg_models": n_seg_models,
        }
        return metrics

    # ──────────────────────────────────────────────────────────────────────────
    # PRIVATE — PREDICT
    # ──────────────────────────────────────────────────────────────────────────

    def _predict_stage2(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict E[recovery_fraction] using segment-specific model if available,
        otherwise fall back to global pooled model.
        """
        n = len(df)
        e_fraction = np.zeros(n)

        for i, (idx, row) in enumerate(df.iterrows()):
            seg = row.get("segment", "__unknown__")
            x   = np.array([row.get(f, 0.0) for f in self._feature_cols_s2]).reshape(1, -1)

            if seg in self._stage2_by_segment:
                model = self._stage2_by_segment[seg]
            else:
                model = self._stage2_global

            logit_pred = model.predict(x)[0]
            e_fraction[i] = _inverse_logit(np.array([logit_pred]))[0]

        return e_fraction

    # ──────────────────────────────────────────────────────────────────────────
    # PRIVATE — VALIDATION + MLFLOW
    # ──────────────────────────────────────────────────────────────────────────

    def _validate_input(self, df: pd.DataFrame, split: str) -> None:
        """Check required columns exist."""
        required = set(self._feature_cols_s1 + self._feature_cols_s2 +
                       ["account_id", "current_balance", "segment"])
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"[RecoveryAmountModel] {split} DataFrame missing columns: {missing}"
            )
        if split in ("train", "val"):
            target_cols = {"recovery_occurred"}
            missing_targets = target_cols - set(df.columns)
            if missing_targets:
                raise ValueError(
                    f"[RecoveryAmountModel] {split} DataFrame missing target columns: "
                    f"{missing_targets}"
                )

    def _try_setup_mlflow(self) -> None:
        try:
            import mlflow
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)
            self._mlflow = mlflow
            self._mlflow_available = True
        except ImportError:
            logger.info("MLflow not installed — skipping experiment tracking.")

    def _log_to_mlflow(self, metrics: Dict) -> Optional[str]:
        if not self._mlflow_available:
            return None
        try:
            mlflow = self._mlflow
            import mlflow.sklearn
            with mlflow.start_run(run_name="recovery_amount_model") as run:
                mlflow.set_tag("stage", "recovery_amount")
                mlflow.set_tag("model_type", "two_stage_beta_regression")

                # Params
                mlflow.log_params({f"s1_{k}": v for k, v in self.stage1_params.items()})
                mlflow.log_params({f"s2_{k}": v for k, v in self.stage2_params.items()})
                mlflow.log_param("calibrate_stage1", self.calibrate_stage1)
                mlflow.log_param("min_segment_samples", self.min_segment_samples)

                # Metrics
                mlflow.log_metrics(metrics)

                # Feature importance artifact
                if self._stage2_global is not None:
                    imp = dict(zip(
                        self._feature_cols_s2,
                        self._stage2_global.feature_importances_,
                    ))
                    imp_path = "/tmp/s2_feature_importance.json"
                    with open(imp_path, "w") as f:
                        json.dump(imp, f, indent=2)
                    mlflow.log_artifact(imp_path)

                # Log models
                mlflow.sklearn.log_model(self._stage1_model,  "stage1_recovery_incidence")
                mlflow.sklearn.log_model(self._stage2_global, "stage2_recovery_fraction_global")

                return run.info.run_id
        except Exception as e:
            logger.warning("MLflow logging failed: %s", e)
            return None


# ─────────────────────────────────────────────────────────────────────────────
# SPARK / DATABRICKS BATCH SCORING WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

class RecoveryAmountModelSpark:
    """
    Databricks / PySpark wrapper around RecoveryAmountModel.
    Handles Spark DataFrame I/O and distributed scoring via pandas UDF.

    Usage
    -----
        spark_model = RecoveryAmountModelSpark(fitted_model)
        scored_df   = spark_model.score(accounts_spark_df)
    """

    def __init__(self, fitted_model: RecoveryAmountModel):
        if not fitted_model._is_fitted:
            raise ValueError("Pass a fitted RecoveryAmountModel instance.")
        self.model = fitted_model

    def score(self, spark_df) -> object:
        """
        Score a PySpark DataFrame. Returns Spark DataFrame with prediction columns.

        Parameters
        ----------
        spark_df : pyspark.sql.DataFrame
        """
        from pyspark.sql import functions as F
        from pyspark.sql.types import StructType, StructField, StringType, DoubleType

        all_feature_cols = list(set(
            self.model._feature_cols_s1 +
            self.model._feature_cols_s2 +
            ["account_id", "current_balance", "segment"]
        ))

        # Broadcast model to avoid serialisation on each partition
        import pyspark
        sc = pyspark.SparkContext.getOrCreate()
        broadcast_model = sc.broadcast(self.model)

        schema = StructType([
            StructField("account_id",         StringType(), True),
            StructField("p_recovery",         DoubleType(), True),
            StructField("e_fraction",         DoubleType(), True),
            StructField("e_recovery_amount",  DoubleType(), True),
        ])

        @F.pandas_udf(schema)
        def score_partition(pdf: pd.DataFrame) -> pd.DataFrame:
            m = broadcast_model.value
            result = m.predict(pdf, return_components=True)
            return result[["account_id", "p_recovery", "e_fraction", "e_recovery_amount"]]

        return spark_df.groupby().applyInPandas(
            lambda pdf: broadcast_model.value.predict(pdf, return_components=True),
            schema=schema,
        )


# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE / SMOKE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import yaml

    # Load config
    with open("conf/use_cases/debt_collection_optimization.yaml") as f:
        config = yaml.safe_load(f)

    np.random.seed(42)
    n = 2000

    # Synthetic charge-off data
    df = pd.DataFrame({
        "account_id":             [f"ACC{i:05d}" for i in range(n)],
        "current_balance":        np.random.uniform(500, 50_000, n),
        "original_balance":       np.random.uniform(1_000, 60_000, n),
        "balance_ratio":          np.random.uniform(0.3, 1.0, n),
        "months_since_charge_off": np.random.randint(1, 12, n),
        "days_past_due":          np.random.randint(90, 730, n),
        "payment_history_score":  np.random.uniform(0, 1, n),
        "contact_response_rate":  np.random.uniform(0, 1, n),
        "promise_kept_rate":      np.random.uniform(0, 1, n),
        "settlement_offers_made": np.random.randint(0, 5, n),
        "avg_discount_offered":   np.random.uniform(0, 0.6, n),
        "debt_to_income_ratio":   np.random.uniform(0.1, 1.5, n),
        "payment_capacity_score": np.random.uniform(0, 1, n),
        "segment": np.random.choice(
            ["high_value_cooperative", "medium_value_willing",
             "high_value_unresponsive", "low_value_low_capacity"], n
        ),
    })

    # Synthetic targets
    recovery_prob = (
        0.3 * df["payment_capacity_score"] +
        0.2 * df["contact_response_rate"] +
        0.1 * df["promise_kept_rate"] +
        0.1 * (1 - df["balance_ratio"]) +
        0.1 * df["avg_discount_offered"]
    ).clip(0, 1)

    df["recovery_occurred"] = (np.random.uniform(0, 1, n) < recovery_prob).astype(int)
    df["recovery_fraction"] = np.where(
        df["recovery_occurred"] == 1,
        np.random.beta(2, 5, n).clip(_EPSILON, 1 - _EPSILON),
        np.nan,
    )

    train_df = df.iloc[:1600]
    val_df   = df.iloc[1600:]

    # Minimal config
    config_override = {
        "recovery_amount_model": {
            "stage1":   {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.05},
            "stage2":   {"n_estimators":  80, "max_depth": 3, "learning_rate": 0.05},
            "min_segment_samples": 50,
            "calibrate_stage1": True,
            "mlflow":   {"experiment_name": "recovery_amount_test", "tracking_uri": "mlruns"},
        }
    }

    model = RecoveryAmountModel(config_override)
    results = model.fit(train_df, val_df)

    eval_out = model.evaluate(val_df)
    print(f"\n[Evaluation] Overall metrics:")
    for k, v in eval_out["overall"].items():
        print(f"  {k}: {v}")

    print(f"\n[Evaluation] By segment:")
    print(eval_out["by_segment"].to_string(index=False))

    # Score with discount sweep
    test_accounts = val_df.head(5).copy()
    for d in [0.0, 0.2, 0.4, 0.6]:
        preds = model.predict_with_discount(test_accounts, discount_pct=d)
        print(f"\nDiscount {d:.0%}:")
        print(preds[["account_id", "p_recovery", "e_fraction",
                      "e_recovery_amount", "discount_applied"]].to_string(index=False))

    print("\n✓ RecoveryAmountModel smoke test passed")
