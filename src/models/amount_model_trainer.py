"""
AmountModelTrainer
==================
Trains E(recovery_amount | recovery occurred) on ALL recovered accounts.

Design principle:
  Train on the full recovered population — not just SETTLE segment.
  Conditioning on SIGNAL_SEGMENT + discount + behavioural features
  gives unbiased estimates across the action space.

──────────────────────────────────────────────────────────────────────────────
CONFIGURATION INSTRUCTIONS
──────────────────────────────────────────────────────────────────────────────
TARGET
    recovery_amount: total THB recovered within the outcome window.
    Log-transformed before training (log1p), exponentiated at inference.
    Reason: recovery amounts are right-skewed (long tail of high recoveries).

TRAINING DATA REQUIREMENT
    Must include ALL recovered accounts regardless of which action was taken.
    Do NOT filter to SETTLE segment only — this biases estimates downward
    for PLAN and AGENCY segments.

    Required columns:
        account_id          — for deduplication
        signal_segment      — A/B/C/D (from FeatureAgent)
        behavioural_persona — Cooperative/Stressed/Sporadic/Disconnected
        discount_offered    — actual discount fraction applied (0.0–0.60)
        recovery_amount     — THB recovered (target, must be > 0)
        propensity_30d      — P_1M score at observation date
        propensity_180d     — P_6M score at observation date
        willingness_score   — engagement proxy
        capacity_score      — ability-to-pay proxy
        erv_at_d_optimal    — outstanding balance proxy
        low_confidence_flag — model uncertainty

    Optional enrichment columns (include if available):
        dpd_at_observation  — DPD bucket at scoring date
        months_delinquent   — duration in delinquency
        ncb_tradeline_count — bureau depth signal

OUTCOME WINDOW
    Default: 180 days (P_6M aligned).
    For tactical amount estimation, also train a 30-day version
    aligned with P_1M. Use 30d amount model for DIGITAL_NUDGE/AGENT_CALL,
    180d amount model for AGENCY/LEGAL.

MODEL_PATH
    models/amount_model/amount_model_{window}d.pkl
    Set AMOUNT_MODEL_DIR env variable to override.

    In Databricks:
        AMOUNT_MODEL_DIR = /dbfs/FileStore/models/amount_model/
──────────────────────────────────────────────────────────────────────────────
"""

import logging
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    logging.warning("LightGBM not available. Install with: pip install lightgbm")

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

AMOUNT_FEATURES = [
    "signal_segment",
    "behavioural_persona",
    "discount_offered",
    "propensity_30d",
    "propensity_180d",
    "willingness_score",
    "capacity_score",
    "erv_at_d_optimal",
    "low_confidence_flag",
    # Optional — included if present
    "dpd_at_observation",
    "months_delinquent",
    "ncb_tradeline_count",
]

DEFAULT_MODEL_DIR = Path(
    os.environ.get("AMOUNT_MODEL_DIR", "models/amount_model")
)

DEFAULT_LGB_PARAMS = {
    "objective":        "regression",
    "metric":           "rmse",
    "num_leaves":       63,
    "learning_rate":    0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq":     5,
    "min_child_samples":20,
    "n_estimators":     500,
    "early_stopping_rounds": 50,
    "verbose":          -1,
}


class AmountModelTrainer:

    def __init__(
        self,
        outcome_window_days: int = 180,
        model_dir: Path = DEFAULT_MODEL_DIR,
        lgb_params: dict = None,
    ):
        """
        Args:
            outcome_window_days: 30 or 180. Train separate models per window.
                                 Use 30d for DIGITAL_NUDGE/AGENT_CALL ERV.
                                 Use 180d for AGENCY/LEGAL ERV.
            model_dir:           Save path. Set AMOUNT_MODEL_DIR to override.
            lgb_params:          Override default LightGBM params.
        """
        if not LGB_AVAILABLE:
            raise ImportError("LightGBM required. pip install lightgbm")

        self.outcome_window_days = outcome_window_days
        self.model_dir           = Path(model_dir)
        self.lgb_params          = {**DEFAULT_LGB_PARAMS, **(lgb_params or {})}
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def fit(self, df: pd.DataFrame) -> dict:
        """
        Trains E(log1p(recovery_amount)) on all recovered accounts.

        Args:
            df: All recovered accounts with required columns.
                See CONFIGURATION INSTRUCTIONS for column spec.
                Must NOT be pre-filtered to any segment or action type.

        Returns:
            dict with RMSE, feature importance, training set size.
        """
        # ── Filter to recovered accounts only ────────────────────────────────
        df = df[df["recovery_amount"] > 0].copy()
        logger.info(f"Training amount model on {len(df):,} recovered accounts")

        # ── Log-transform target ──────────────────────────────────────────────
        df["log_recovery_amount"] = np.log1p(df["recovery_amount"])

        X, cat_features = self._prepare_features(df)
        y = df["log_recovery_amount"].values

        # ── Train/validation split (time-based if score_date available) ───────
        if "score_date" in df.columns:
            df_sorted  = df.sort_values("score_date")
            split_idx  = int(len(df_sorted) * 0.8)
            train_idx  = df_sorted.index[:split_idx]
            val_idx    = df_sorted.index[split_idx:]
            X_train, X_val = X.loc[train_idx], X.loc[val_idx]
            y_train, y_val = y[df_sorted.index.get_indexer(train_idx)], \
                             y[df_sorted.index.get_indexer(val_idx)]
        else:
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

        model = lgb.LGBMRegressor(**self.lgb_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            categorical_feature=cat_features,
        )

        # ── Evaluate ──────────────────────────────────────────────────────────
        val_pred   = model.predict(X_val)
        rmse_log   = float(np.sqrt(np.mean((y_val - val_pred) ** 2)))
        # RMSE in original THB space
        rmse_thb   = float(np.sqrt(np.mean(
            (np.expm1(y_val) - np.expm1(val_pred)) ** 2
        )))

        feat_imp = dict(zip(X.columns, model.feature_importances_))

        self._save(model, X.columns.tolist(), cat_features)

        logger.info(
            f"Amount model trained | window={self.outcome_window_days}d | "
            f"n_train={len(X_train):,} | rmse_log={rmse_log:.4f} | "
            f"rmse_thb={rmse_thb:,.0f}"
        )
        return {
            "n_train":          len(X_train),
            "n_val":            len(X_val),
            "rmse_log":         round(rmse_log, 4),
            "rmse_thb":         round(rmse_thb, 2),
            "top_features":     sorted(
                feat_imp.items(), key=lambda x: x[1], reverse=True
            )[:10],
        }

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Returns predicted recovery amount in THB (exponentiated from log space).

        Args:
            df: Accounts to score with AMOUNT_FEATURES columns.

        Returns:
            pd.Series of predicted THB amounts (same index as df).
        """
        model, feature_cols, cat_features = self._load()
        X, _ = self._prepare_features(df, feature_cols=feature_cols)
        log_pred = model.predict(X)
        return pd.Series(np.expm1(log_pred), index=df.index, name="predicted_amount")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _prepare_features(
        self, df: pd.DataFrame, feature_cols: list = None
    ):
        available = [
            c for c in (feature_cols or AMOUNT_FEATURES) if c in df.columns
        ]
        X = df[available].copy()

        cat_features = []
        for col in ["signal_segment", "behavioural_persona"]:
            if col in X.columns:
                X[col] = X[col].astype("category")
                cat_features.append(col)

        if "low_confidence_flag" in X.columns:
            X["low_confidence_flag"] = X["low_confidence_flag"].astype(float)

        # Median impute numerics
        for col in X.select_dtypes(include=[np.number]).columns:
            X[col] = X[col].fillna(X[col].median())

        return X, cat_features

    def _model_path(self) -> Path:
        return self.model_dir / f"amount_model_{self.outcome_window_days}d.pkl"

    def _save(self, model, feature_cols: list, cat_features: list) -> None:
        path = self._model_path()
        joblib.dump({
            "model":        model,
            "feature_cols": feature_cols,
            "cat_features": cat_features,
            "window_days":  self.outcome_window_days,
        }, path)
        logger.info(f"Saved amount model → {path}")

    def _load(self):
        obj = joblib.load(self._model_path())
        return obj["model"], obj["feature_cols"], obj["cat_features"]
