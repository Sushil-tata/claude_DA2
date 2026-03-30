"""
WillingnessModelTrainer
=======================
Trains P(customer will engage | contact attempted).

Replaces the rule-based weighted sum for willingness_score.

Label: contact_responded = 1 if customer picked up / replied / partial payment
       within the observation window after a contact attempt.

Design principle:
  willingness_score is a FEATURE passed to propensity, amount, and elasticity
  models. It describes engagement behaviour. It does NOT prescribe action.

──────────────────────────────────────────────────────────────────────────────
CONFIGURATION INSTRUCTIONS
──────────────────────────────────────────────────────────────────────────────
LABEL DEFINITION (contact_responded)
    Positive (1) = any of the following within 14 days of last contact attempt:
        - Call picked up (contact_picked_up = 1)
        - Promise to pay recorded (ttp_recorded = 1)
        - Partial payment made (partial_payment_made = 1)
        - Digital nudge opened AND clicked (digital_opened AND digital_clicked)

    Negative (0) = contact attempted but none of the above occurred.

    ⚠ Do NOT include accounts where no contact was attempted — these are
      not labelled examples. Filter to attempted_contact = 1 only.

    ⚠ Do NOT use payment_made_180d as the label — that is the propensity
      model's target. willingness is about engagement, not recovery outcome.

OBSERVATION WINDOW
    Default: 14 days after last contact attempt.
    Why 14 days: long enough to capture delayed responses, short enough
    to reflect current engagement state rather than eventual recovery.
    Override: WillingnessModelTrainer(response_window_days=7)

WILLINGNESS_FEATURES
    Source: collections CRM + contact logs.
    Bureau/card data excluded — willingness is a behavioural signal,
    not a capacity signal.

    Required:
        contact_attempts_30d    — count of outbound attempts last 30 days
        contact_success_rate_3m — % of attempts that reached customer last 3M
        days_since_last_response— days since last meaningful engagement
        broken_promise_count_3m — broken PTPs last 3 months
        partial_payment_count_3m— count of partial payments last 3 months
        digital_open_rate_3m    — % of digital nudges opened last 3M
        avg_response_lag_days   — average days to respond when contact made

    Optional (include if available):
        preferred_contact_hour  — customer's historical peak response hour
        preferred_channel       — SMS / call / email
        escalation_flag         — account in legal escalation

MODEL_PATH
    models/willingness_model/willingness_model.pkl
    Override via WILLINGNESS_MODEL_DIR env variable.

    In Databricks: WILLINGNESS_MODEL_DIR = /dbfs/FileStore/models/willingness_model/
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

WILLINGNESS_FEATURES = [
    "contact_attempts_30d",
    "contact_success_rate_3m",
    "days_since_last_response",
    "broken_promise_count_3m",
    "partial_payment_count_3m",
    "digital_open_rate_3m",
    "avg_response_lag_days",
    # Optional — included if present
    "preferred_contact_hour",
    "escalation_flag",
    "signal_segment",       # segment context
    "propensity_30d",       # short-term recovery signal adds engagement context
]

DEFAULT_MODEL_DIR = Path(
    os.environ.get("WILLINGNESS_MODEL_DIR", "models/willingness_model")
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


class WillingnessModelTrainer:

    def __init__(
        self,
        response_window_days: int = 14,
        model_dir: Path = DEFAULT_MODEL_DIR,
        lgb_params: dict = None,
    ):
        """
        Args:
            response_window_days: Days after contact to observe response.
                                  Default 14. See CONFIGURATION INSTRUCTIONS.
            model_dir:            Save/load path. Set WILLINGNESS_MODEL_DIR to override.
            lgb_params:           Override default LightGBM params.
        """
        if not LGB_AVAILABLE:
            raise ImportError("LightGBM required. pip install lightgbm")

        self.response_window_days = response_window_days
        self.model_dir            = Path(model_dir)
        self.lgb_params           = {**DEFAULT_LGB_PARAMS, **(lgb_params or {})}
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def fit(self, df: pd.DataFrame) -> dict:
        """
        Trains P(contact_responded) on historical contact attempt records.

        Args:
            df: Contact attempt records. Must contain:
                - contact_responded (0/1) — label, see CONFIGURATION INSTRUCTIONS
                - attempted_contact = 1 filter should be applied BEFORE passing
                - WILLINGNESS_FEATURES columns (missing → median imputed)
                - score_date for time-based train/val split

        Returns:
            dict: auc, n_train, n_val, feature_importance (top 10),
                  response_rate (base rate of contact_responded in training data)
        """
        df = df[df.get("attempted_contact", pd.Series(1, index=df.index)) == 1].copy()
        logger.info(
            f"Training willingness model | n={len(df):,} | "
            f"response_rate={df['contact_responded'].mean():.1%}"
        )

        X, cat_features = self._prepare_features(df)
        y = df["contact_responded"].astype(int).values

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

        logger.info(f"Willingness model | AUC={auc:.4f} | n_train={len(X_train):,}")
        return {
            "auc":           round(auc, 4),
            "n_train":       len(X_train),
            "n_val":         len(X_val),
            "response_rate": round(float(df["contact_responded"].mean()), 4),
            "top_features":  sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:10],
        }

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Returns willingness_score ∈ [0, 1] per account.
        Replaces rule-based weighted sum in FeatureAgent.
        """
        model, feature_cols, cat_features = self._load()
        X, _ = self._prepare_features(df, feature_cols=feature_cols)
        scores = model.predict_proba(X)[:, 1]
        return pd.Series(scores, index=df.index, name="willingness_score")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _prepare_features(self, df: pd.DataFrame, feature_cols: list = None):
        available = [c for c in (feature_cols or WILLINGNESS_FEATURES) if c in df.columns]
        X = df[available].copy()

        cat_features = []
        if "signal_segment" in X.columns:
            X["signal_segment"] = X["signal_segment"].astype("category")
            cat_features.append("signal_segment")
        if "escalation_flag" in X.columns:
            X["escalation_flag"] = X["escalation_flag"].astype(float)

        for col in X.select_dtypes(include=[np.number]).columns:
            X[col] = X[col].fillna(X[col].median())

        return X, cat_features

    def _model_path(self) -> Path:
        return self.model_dir / "willingness_model.pkl"

    def _save(self, model, feature_cols: list, cat_features: list) -> None:
        path = self._model_path()
        joblib.dump({"model": model, "feature_cols": feature_cols,
                     "cat_features": cat_features}, path)
        logger.info(f"Saved willingness model → {path}")

    def _load(self):
        obj = joblib.load(self._model_path())
        return obj["model"], obj["feature_cols"], obj["cat_features"]
