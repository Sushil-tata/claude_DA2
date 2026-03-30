"""
CapacityModelTrainer
====================
Trains P(customer can make a payment | financial state).

Replaces the rule-based weighted sum for capacity_score.

Label: made_any_payment_30d = 1 if any payment received within 30 days
       of observation date (not specific to this account — any payment
       across any facility signals liquidity exists).

Design principle:
  capacity_score is a FEATURE — it describes financial ability to pay.
  It does NOT prescribe action. Use bureau + card data ONLY.
  No deposit/CASA/salary inflow signals — CardX does not have these.

──────────────────────────────────────────────────────────────────────────────
CONFIGURATION INSTRUCTIONS
──────────────────────────────────────────────────────────────────────────────
LABEL DEFINITION (made_any_payment_30d)
    Positive (1) = any payment received on this account within 30 days
                   of observation date (any amount, including minimum).

    Negative (0) = no payment received in 30 days.

    Why 30 days (not 180): capacity is a SHORT-TERM signal — does the
    customer have money available right now? 180d conflates capacity with
    willingness and structural recovery.

    ⚠ Do NOT use this label if account is in HOLD or no-contact period —
      absence of payment may reflect lack of contact, not lack of capacity.
      Filter to accounts with active_contact_period = 1.

DATA SOURCE
    Bureau (NCB/TUEF) + internal card data ONLY.
    No deposit/CASA/transaction signals — CardX is a card issuer only.

    Required features (CAPACITY_FEATURES):
        ncb_other_accounts_current    — count of other accounts with DPD=0
        ncb_total_revolving_util      — total utilisation across all revolving facilities
        ncb_active_tradelines         — count of tradelines still in good standing
        ncb_enquiry_count_3m          — bureau enquiries last 3M (financial stress proxy)
        card_payment_pct_minimum_3m   — avg payment as % of minimum due last 3M
        card_months_since_last_payment— months since any payment on this card
        card_balance_to_limit_ratio   — utilisation on this specific card
        dpd_current                   — current DPD bucket (0/1-30/31-60/61-90/90+)

    Optional (include if available):
        ncb_installment_dpd_max_12m   — worst DPD on installment products last 12M
        ncb_secured_product_count     — secured products (home/car) as stability proxy
        card_credit_limit             — absolute limit (proxy for credit quality tier)

OUTCOME WINDOW
    30 days. Use separate model for 90d if needed for AGENCY/LEGAL decisions.
    Override: CapacityModelTrainer(outcome_window_days=90)

MODEL_PATH
    models/capacity_model/capacity_model_{window}d.pkl
    Override via CAPACITY_MODEL_DIR env variable.

    In Databricks: CAPACITY_MODEL_DIR = /dbfs/FileStore/models/capacity_model/
──────────────────────────────────────────────────────────────────────────────
"""

import logging
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

CAPACITY_FEATURES = [
    "ncb_other_accounts_current",
    "ncb_total_revolving_util",
    "ncb_active_tradelines",
    "ncb_enquiry_count_3m",
    "card_payment_pct_minimum_3m",
    "card_months_since_last_payment",
    "card_balance_to_limit_ratio",
    "dpd_current",
    # Optional — included if present
    "ncb_installment_dpd_max_12m",
    "ncb_secured_product_count",
    "card_credit_limit",
    "signal_segment",
]

DEFAULT_MODEL_DIR = Path(
    os.environ.get("CAPACITY_MODEL_DIR", "models/capacity_model")
)

DEFAULT_LGB_PARAMS = {
    "objective":             "binary",
    "metric":                "auc",
    "num_leaves":            31,
    "learning_rate":         0.05,
    "feature_fraction":      0.8,
    "bagging_fraction":      0.8,
    "bagging_freq":          5,
    "min_child_samples":     20,
    "n_estimators":          300,
    "early_stopping_rounds": 30,
    "verbose":               -1,
}


class CapacityModelTrainer:

    def __init__(
        self,
        outcome_window_days: int = 30,
        model_dir: Path = DEFAULT_MODEL_DIR,
        lgb_params: dict = None,
    ):
        """
        Args:
            outcome_window_days: Payment observation window. Default 30.
                                 See CONFIGURATION INSTRUCTIONS.
            model_dir:           Save/load path. Set CAPACITY_MODEL_DIR to override.
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
        Trains P(made_any_payment_30d) on historical account records.

        Args:
            df: Account observation records. Must contain:
                - made_any_payment_30d (0/1) — label
                - CAPACITY_FEATURES columns (missing → median imputed)
                - score_date for time-based train/val split
                - active_contact_period = 1 filter should be applied before passing

        Returns:
            dict: auc, n_train, n_val, feature_importance (top 10),
                  payment_rate (base rate of payment in training data)
        """
        logger.info(
            f"Training capacity model | window={self.outcome_window_days}d | "
            f"n={len(df):,} | payment_rate={df['made_any_payment_30d'].mean():.1%}"
        )

        X, cat_features = self._prepare_features(df)
        y = df["made_any_payment_30d"].astype(int).values

        if "score_date" in df.columns:
            df_s      = df.sort_values("score_date")
            split_idx = int(len(df_s) * 0.8)
            train_idx = df_s.index[:split_idx]
            val_idx   = df_s.index[split_idx:]
            X_train, X_val = X.loc[train_idx], X.loc[val_idx]
            y_train = y[df_s.index.get_indexer(train_idx)]
            y_val   = y[df_s.index.get_indexer(val_idx)]
        else:
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

        model = lgb.LGBMClassifier(**self.lgb_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            categorical_feature=cat_features,
        )

        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(y_val, model.predict_proba(X_val)[:, 1]))

        feat_imp = dict(zip(X.columns, model.feature_importances_))
        self._save(model, X.columns.tolist(), cat_features)

        logger.info(f"Capacity model | AUC={auc:.4f} | n_train={len(X_train):,}")
        return {
            "auc":          round(auc, 4),
            "n_train":      len(X_train),
            "n_val":        len(X_val),
            "payment_rate": round(float(df["made_any_payment_30d"].mean()), 4),
            "top_features": sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:10],
        }

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Returns capacity_score ∈ [0, 1] per account.
        Replaces rule-based weighted sum in FeatureAgent.
        """
        model, feature_cols, cat_features = self._load()
        X, _ = self._prepare_features(df, feature_cols=feature_cols)
        scores = model.predict_proba(X)[:, 1]
        return pd.Series(scores, index=df.index, name="capacity_score")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _prepare_features(self, df: pd.DataFrame, feature_cols: list = None):
        available = [c for c in (feature_cols or CAPACITY_FEATURES) if c in df.columns]
        X = df[available].copy()

        cat_features = []
        if "signal_segment" in X.columns:
            X["signal_segment"] = X["signal_segment"].astype("category")
            cat_features.append("signal_segment")
        if "dpd_current" in X.columns:
            X["dpd_current"] = X["dpd_current"].astype("category")
            cat_features.append("dpd_current")

        for col in X.select_dtypes(include=[np.number]).columns:
            X[col] = X[col].fillna(X[col].median())

        return X, cat_features

    def _model_path(self) -> Path:
        return self.model_dir / f"capacity_model_{self.outcome_window_days}d.pkl"

    def _save(self, model, feature_cols: list, cat_features: list) -> None:
        path = self._model_path()
        joblib.dump({"model": model, "feature_cols": feature_cols,
                     "cat_features": cat_features,
                     "window_days": self.outcome_window_days}, path)
        logger.info(f"Saved capacity model → {path}")

    def _load(self):
        obj = joblib.load(self._model_path())
        return obj["model"], obj["feature_cols"], obj["cat_features"]
