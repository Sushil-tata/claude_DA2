"""
ElasticityModelTrainer
======================
Trains P(accept_offer | discount, signal_segment, features).

Design principle:
  Elasticity is a separate model — it is NOT derived from ERV.
  Using ERV as an input to the elasticity model creates circular logic:
    ERV depends on P(accept), which depends on ERV → undefined.

  Keep the models separate:
    ElasticityModel:  P(accept | d, segment, features)
    AmountModel:      E(amount | recovery, d, segment, features)
    PropensityModel:  P(recovery in T days | features, segment)
    ERV:              combine all three at inference time

──────────────────────────────────────────────────────────────────────────────
CONFIGURATION INSTRUCTIONS
──────────────────────────────────────────────────────────────────────────────
TARGET
    accepted: 1 if the customer accepted the settlement offer, else 0.
    Source: collections CRM — offer made + offer accepted flag.

    Positive class = accepted the discount offer AND made payment.
    Negative class = offer made but rejected or no response.

    Do NOT include accounts where no offer was made (no counterfactual).

TRAINING DATA REQUIREMENT
    Must include offer_made records across all discount levels.
    Ideally covers the full discount range (0%–60%) to avoid extrapolation.

    If historical data only covers 20%–40% discounts, the model will
    extrapolate for 0% and 60% — flag this in your MLflow experiment
    notes and treat those predictions with lower confidence.

    Required columns:
        signal_segment      — A/B/C/D
        behavioural_persona — from cluster model
        discount_offered    — fraction (0.0–0.60)
        accepted            — 1/0 (target)
        propensity_30d      — P_1M at offer date
        willingness_score   — engagement at offer date
        capacity_score      — capacity at offer date
        erv_at_d_optimal    — account size proxy

    ⚠ DO NOT include recovery_amount or ERV as features.
      This would create circular dependency with ERV computation.

SEGMENT-SPECIFIC CALIBRATION
    The model is trained on all segments jointly with signal_segment
    as a categorical feature. This gives segment-specific discount curves
    through the interaction terms learned by LightGBM.

    For Segment D (no signal): predictions will be less reliable.
    Apply a discount floor (e.g. minimum 30%) for Segment D accounts
    regardless of model output — set in ConstraintAgent, not here.

MODEL_PATH
    models/elasticity_model/elasticity_model.pkl
    Set ELASTICITY_MODEL_DIR env variable to override.
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
    logging.warning("LightGBM not available. pip install lightgbm")

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

ELASTICITY_FEATURES = [
    "signal_segment",
    "behavioural_persona",
    "discount_offered",
    "propensity_30d",
    "willingness_score",
    "capacity_score",
    "erv_at_d_optimal",
    # ⚠ DO NOT add recovery_amount or erv — circular dependency
]

DEFAULT_MODEL_DIR = Path(
    os.environ.get("ELASTICITY_MODEL_DIR", "models/elasticity_model")
)

DEFAULT_LGB_PARAMS = {
    "objective":         "binary",
    "metric":            "auc",
    "num_leaves":        31,
    "learning_rate":     0.05,
    "feature_fraction":  0.8,
    "bagging_fraction":  0.8,
    "bagging_freq":      5,
    "min_child_samples": 20,
    "n_estimators":      300,
    "early_stopping_rounds": 30,
    "verbose":           -1,
}

# Discount grid used at inference (must match ModelAgent DISCOUNT_GRID)
INFERENCE_DISCOUNT_GRID = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60]


class ElasticityModelTrainer:

    def __init__(
        self,
        model_dir: Path = DEFAULT_MODEL_DIR,
        lgb_params: dict = None,
    ):
        """
        Args:
            model_dir:   Save/load path. Set ELASTICITY_MODEL_DIR to override.
            lgb_params:  Override default LightGBM params.
        """
        if not LGB_AVAILABLE:
            raise ImportError("LightGBM required. pip install lightgbm")

        self.model_dir  = Path(model_dir)
        self.lgb_params = {**DEFAULT_LGB_PARAMS, **(lgb_params or {})}
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def fit(self, df: pd.DataFrame) -> dict:
        """
        Trains P(accept | discount, segment, features) on historical offer data.

        Args:
            df: Historical offer records. See CONFIGURATION INSTRUCTIONS.
                Must contain 'accepted' (0/1) and 'discount_offered' columns.
                Must NOT be pre-filtered to accepted offers only.

        Returns:
            dict with AUC, feature importance, training set size per segment.
        """
        df = df.copy()
        logger.info(
            f"Training elasticity model | n={len(df):,} | "
            f"accept_rate={df['accepted'].mean():.1%}"
        )

        X, cat_features = self._prepare_features(df)
        y = df["accepted"].astype(int).values

        if "score_date" in df.columns:
            df_sorted = df.sort_values("score_date")
            split_idx = int(len(df_sorted) * 0.8)
            train_idx = df_sorted.index[:split_idx]
            val_idx   = df_sorted.index[split_idx:]
            X_train, X_val = X.loc[train_idx], X.loc[val_idx]
            y_train = y[df_sorted.index.get_indexer(train_idx)]
            y_val   = y[df_sorted.index.get_indexer(val_idx)]
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
        val_prob = model.predict_proba(X_val)[:, 1]
        auc      = float(roc_auc_score(y_val, val_prob))

        feat_imp = dict(zip(X.columns, model.feature_importances_))
        self._save(model, X.columns.tolist(), cat_features)

        logger.info(
            f"Elasticity model trained | AUC={auc:.4f} | "
            f"n_train={len(X_train):,}"
        )
        return {
            "n_train":      len(X_train),
            "n_val":        len(X_val),
            "auc":          round(auc, 4),
            "accept_rate":  round(float(df["accepted"].mean()), 4),
            "top_features": sorted(
                feat_imp.items(), key=lambda x: x[1], reverse=True
            )[:10],
        }

    def predict_accept_curve(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns P(accept) for each account across the full discount grid.
        Used by ModelAgent to replace hardcoded elasticity alphas.

        Args:
            df: Accounts to score (without discount_offered — added internally).

        Returns:
            DataFrame with columns [account_id, d_0.0, d_0.1, ..., d_0.6]
            Each cell = P(accept | that discount level).
        """
        model, feature_cols, cat_features = self._load()
        rows = []

        for d in INFERENCE_DISCOUNT_GRID:
            df_d = df.copy()
            df_d["discount_offered"] = d
            X, _ = self._prepare_features(df_d, feature_cols=feature_cols)
            p    = model.predict_proba(X)[:, 1]
            rows.append(pd.Series(p, index=df.index, name=f"p_accept_d{int(d*100)}"))

        result = pd.concat(rows, axis=1)
        if "account_id" in df.columns:
            result.insert(0, "account_id", df["account_id"].values)
        return result

    # ── Internal ──────────────────────────────────────────────────────────────

    def _prepare_features(
        self, df: pd.DataFrame, feature_cols: list = None
    ):
        available = [
            c for c in (feature_cols or ELASTICITY_FEATURES) if c in df.columns
        ]
        X = df[available].copy()

        cat_features = []
        for col in ["signal_segment", "behavioural_persona"]:
            if col in X.columns:
                X[col] = X[col].astype("category")
                cat_features.append(col)

        if "low_confidence_flag" in X.columns:
            X["low_confidence_flag"] = X["low_confidence_flag"].astype(float)

        for col in X.select_dtypes(include=[np.number]).columns:
            X[col] = X[col].fillna(X[col].median())

        return X, cat_features

    def _model_path(self) -> Path:
        return self.model_dir / "elasticity_model.pkl"

    def _save(self, model, feature_cols: list, cat_features: list) -> None:
        path = self._model_path()
        joblib.dump({
            "model":        model,
            "feature_cols": feature_cols,
            "cat_features": cat_features,
        }, path)
        logger.info(f"Saved elasticity model → {path}")

    def _load(self):
        obj = joblib.load(self._model_path())
        return obj["model"], obj["feature_cols"], obj["cat_features"]
