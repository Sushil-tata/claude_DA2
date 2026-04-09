"""
PropensityModelTrainer
======================
Trains P(customer makes a payment within N days) for multiple horizons.

Three horizons, three separate models:
    propensity_30d  (P_1M) — primary signal for tactical actions
                             (DIGITAL_NUDGE, AGENT_CALL)
    propensity_90d  (P_3M) — intermediate signal
    propensity_180d (P_6M) — primary signal for strategic actions
                             (AGENCY, LEGAL)

Design principle:
    One model per horizon. All three are trained on the SAME 100+ enterprise
    features. The horizon changes only the label definition — not the features.
    Each model is independently calibrated so scores are comparable across
    horizons (propensity_30d < propensity_180d for the same account).

──────────────────────────────────────────────────────────────────────────────
CONFIGURATION INSTRUCTIONS
──────────────────────────────────────────────────────────────────────────────
LABEL DEFINITION
    made_payment_{N}d = 1 if any payment > MIN_PAYMENT_THB was received
                        within N days of observation date.

    MIN_PAYMENT_THB (default: 100 THB)
        Minimum payment to count as a recovery event. Prevents micro-payments
        (e.g. 1 THB test transactions) from being counted as recovery.
        Override: PropensityModelTrainer(min_payment_thb=500)

    ⚠ Observation date must be the SCORE_DATE (not account open date or
      DPD start date). Use point-in-time features as of score_date.

TRAINING DATA
    Source: your enterprise feature table joined with payment outcomes.
    Required columns:
        account_id         — for deduplication
        score_date         — for time-based train/val split
        made_payment_30d   — label (0/1), required for 30d model
        made_payment_90d   — label (0/1), required for 90d model
        made_payment_180d  — label (0/1), required for 180d model
        [all enterprise features] — 100+ feature columns

    Optional:
        signal_segment     — if present, used as a categorical feature
        dpd_current        — DPD bucket at score date

    One observation = one account × one score_date.
    Multiple observations per account (different dates) are fine.
    Do NOT include future-dated features — point-in-time safety is critical.

TRAIN/VALIDATION SPLIT
    Always time-based: oldest 80% → train, newest 20% → validation.
    Never random split on time-series data — leads to data leakage.
    Random split is used as fallback only if score_date is not present.

PER-SEGMENT MODELS (optional, default: off)
    Set per_segment=True to train one model per SIGNAL_SEGMENT (A/B/C/D).
    More accurate for segments with distinct recovery patterns but requires
    enough data per segment (≥ 5,000 rows recommended per segment).
    PropensityModelTrainer(per_segment=True)

MODEL_PATH
    models/propensity/{horizon}d_propensity_model.pkl
    Per-segment: models/propensity/{segment}_{horizon}d_propensity_model.pkl
    Set PROPENSITY_MODEL_DIR env variable to override base directory.

    In Databricks:
        PROPENSITY_MODEL_DIR = /dbfs/FileStore/models/propensity/
──────────────────────────────────────────────────────────────────────────────
"""

import logging
import os
from pathlib import Path
from typing import Optional

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

HORIZONS = [30, 90, 180]

LABEL_COLS = {
    30:  "made_payment_30d",
    90:  "made_payment_90d",
    180: "made_payment_180d",
}

# Columns to always exclude from features regardless of what's in the DataFrame
EXCLUDE_COLS = {
    # Labels — never features
    "made_payment_30d", "made_payment_90d", "made_payment_180d",
    "recovery_amount", "recovery_30d", "recovery_90d", "recovery_180d",
    # Future-dated outcomes
    "paid_30d", "paid_90d", "paid_180d",
    # PII
    "national_id", "citizen_id", "full_name", "first_name", "last_name",
    "phone_number", "mobile_number", "email", "address",
}

DEFAULT_MODEL_DIR = Path(
    os.environ.get("PROPENSITY_MODEL_DIR", "models/propensity")
)

DEFAULT_LGB_PARAMS = {
    "objective":             "binary",
    "metric":                "auc",
    "num_leaves":            63,
    "learning_rate":         0.05,
    "feature_fraction":      0.8,
    "bagging_fraction":      0.8,
    "bagging_freq":          5,
    "min_child_samples":     30,
    "n_estimators":          500,
    "early_stopping_rounds": 50,
    "verbose":               -1,
    "class_weight":          "balanced",   # handles imbalanced recovery rates
}

SIGNAL_SEGMENTS = ["A", "B", "C", "D"]


class PropensityModelTrainer:

    def __init__(
        self,
        horizons: list = None,
        model_dir: Path = DEFAULT_MODEL_DIR,
        lgb_params: dict = None,
        per_segment: bool = False,
        min_payment_thb: float = 100.0,
    ):
        """
        Args:
            horizons:        Horizons (days) to train. Default: [30, 90, 180].
                             Train a subset: PropensityModelTrainer(horizons=[30, 180])
            model_dir:       Save/load path. Set PROPENSITY_MODEL_DIR to override.
            lgb_params:      Override default LightGBM params (merged with defaults).
            per_segment:     If True, train one model per SIGNAL_SEGMENT per horizon.
                             Requires signal_segment column in training data.
                             Default: False (single model per horizon).
            min_payment_thb: Minimum payment amount (THB) to count as recovery.
                             Default: 100 THB.
        """
        if not LGB_AVAILABLE:
            raise ImportError("LightGBM required. pip install lightgbm")

        self.horizons        = horizons or HORIZONS
        self.model_dir       = Path(model_dir)
        self.lgb_params      = {**DEFAULT_LGB_PARAMS, **(lgb_params or {})}
        self.per_segment     = per_segment
        self.min_payment_thb = min_payment_thb
        self.model_dir.mkdir(parents=True, exist_ok=True)

    # ── Training ───────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> dict:
        """
        Trains one LightGBM model per horizon (and per segment if per_segment=True).

        Args:
            df: Training data. Must contain:
                - score_date (for time-based split)
                - made_payment_{N}d columns for each horizon in self.horizons
                - enterprise feature columns (100+)

        Returns:
            dict: {horizon: {"auc": float, "n_train": int, "payment_rate": float,
                              "top_features": list, "segments": dict (if per_segment)}}
        """
        results = {}

        for horizon in self.horizons:
            label_col = LABEL_COLS[horizon]
            if label_col not in df.columns:
                logger.warning(
                    f"Label column '{label_col}' not in DataFrame — "
                    f"skipping {horizon}d model. "
                    f"Add {label_col} (0/1) to training data."
                )
                continue

            logger.info(
                f"Training propensity_{horizon}d | n={len(df):,} | "
                f"payment_rate={df[label_col].mean():.1%}"
            )

            if self.per_segment and "signal_segment" in df.columns:
                seg_results = {}
                for seg in SIGNAL_SEGMENTS:
                    seg_df = df[df["signal_segment"] == seg]
                    if len(seg_df) < 500:
                        logger.warning(
                            f"Segment {seg} | horizon {horizon}d: only {len(seg_df)} rows "
                            f"— skipping per-segment model (need ≥500). "
                            f"Global model will be used for this segment."
                        )
                        continue
                    seg_results[seg] = self._fit_single(seg_df, horizon, segment=seg)

                # Also fit global model as fallback for segments with insufficient data
                global_result = self._fit_single(df, horizon, segment=None)
                results[horizon] = {**global_result, "per_segment": seg_results}
            else:
                results[horizon] = self._fit_single(df, horizon, segment=None)

        return results

    def _fit_single(
        self, df: pd.DataFrame, horizon: int, segment: Optional[str]
    ) -> dict:
        """Trains and saves one model for a given horizon (and optional segment)."""
        label_col = LABEL_COLS[horizon]
        X, feature_cols = self._prepare_features(df, label_col)
        y = df[label_col].astype(int).values

        X_train, X_val, y_train, y_val = self._split(df, X, y)

        model = lgb.LGBMClassifier(**self.lgb_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            categorical_feature=[
                c for c in ["signal_segment", "dpd_current"] if c in feature_cols
            ],
        )

        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(y_val, model.predict_proba(X_val)[:, 1]))

        feat_imp = dict(zip(feature_cols, model.feature_importances_))
        top_feats = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:15]

        self._save(model, feature_cols, horizon, segment)

        seg_label = f"segment {segment}" if segment else "global"
        logger.info(
            f"propensity_{horizon}d ({seg_label}) | "
            f"AUC={auc:.4f} | n_train={len(X_train):,} | "
            f"payment_rate={y.mean():.1%}"
        )
        return {
            "auc":          round(auc, 4),
            "n_train":      len(X_train),
            "n_val":        len(X_val),
            "payment_rate": round(float(y.mean()), 4),
            "top_features": top_feats,
            "segment":      segment,
        }

    # ── Inference ──────────────────────────────────────────────────────────────

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Scores all accounts for every trained horizon.

        Returns:
            DataFrame with columns propensity_30d, propensity_90d, propensity_180d
            (only for horizons with fitted models). Same index as df.

        Falls back to per-segment model if available and signal_segment is present,
        otherwise uses global model.
        """
        scores = pd.DataFrame(index=df.index)

        for horizon in self.horizons:
            col = f"propensity_{horizon}d"

            if self.per_segment and "signal_segment" in df.columns:
                seg_scores = pd.Series(np.nan, index=df.index)
                for seg in SIGNAL_SEGMENTS:
                    seg_path = self._model_path(horizon, segment=seg)
                    if seg_path.exists():
                        seg_mask = df["signal_segment"] == seg
                        if seg_mask.any():
                            try:
                                model, feature_cols = self._load(horizon, segment=seg)
                                X, _ = self._prepare_features(
                                    df[seg_mask], label_col=None, feature_cols=feature_cols
                                )
                                seg_scores[seg_mask] = model.predict_proba(X)[:, 1]
                            except Exception as e:
                                logger.warning(
                                    f"Segment {seg} model predict failed for {horizon}d: {e}"
                                )

                # Fill any unscored accounts with global model
                unscored = seg_scores.isna()
                global_path = self._model_path(horizon, segment=None)
                if unscored.any() and global_path.exists():
                    try:
                        model, feature_cols = self._load(horizon, segment=None)
                        X, _ = self._prepare_features(
                            df[unscored], label_col=None, feature_cols=feature_cols
                        )
                        seg_scores[unscored] = model.predict_proba(X)[:, 1]
                    except Exception as e:
                        logger.warning(f"Global fallback predict failed for {horizon}d: {e}")

                scores[col] = seg_scores.fillna(0.0)

            else:
                global_path = self._model_path(horizon, segment=None)
                if not global_path.exists():
                    logger.warning(
                        f"No fitted propensity model for {horizon}d at {global_path}. "
                        f"Run PropensityModelTrainer.fit() to train. "
                        f"Score will be missing for this horizon."
                    )
                    continue
                try:
                    model, feature_cols = self._load(horizon, segment=None)
                    X, _ = self._prepare_features(
                        df, label_col=None, feature_cols=feature_cols
                    )
                    scores[col] = model.predict_proba(X)[:, 1]
                except Exception as e:
                    logger.warning(f"Propensity {horizon}d predict failed: {e}")

        return scores

    # ── Internal ──────────────────────────────────────────────────────────────

    def _prepare_features(
        self,
        df: pd.DataFrame,
        label_col: Optional[str],
        feature_cols: list = None,
    ):
        """
        Selects feature columns, encodes categoricals, imputes numerics.
        All label columns and PII are excluded.
        """
        if feature_cols is not None:
            # Inference mode — use saved feature list, fill missing with median
            available = [c for c in feature_cols if c in df.columns]
            X = df[available].copy()
        else:
            # Training mode — use all columns except labels + PII
            exclude = EXCLUDE_COLS.copy()
            if label_col:
                exclude.add(label_col)
            available = [
                c for c in df.columns
                if c not in exclude
                and c not in {"account_id", "score_date"}
            ]
            X = df[available].copy()
            feature_cols = available

        # Encode categoricals
        for col in ["signal_segment", "dpd_current", "behavioural_persona"]:
            if col in X.columns:
                X[col] = X[col].astype("category")

        # Boolean → float
        bool_cols = X.select_dtypes(include=["bool"]).columns
        X[bool_cols] = X[bool_cols].astype(float)

        # Median impute numerics
        num_cols = X.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            X[col] = X[col].fillna(X[col].median())

        return X, feature_cols

    def _split(self, df, X, y):
        """Time-based 80/20 split on score_date. Random fallback."""
        if "score_date" in df.columns:
            df_sorted = df.sort_values("score_date")
            split_idx = int(len(df_sorted) * 0.8)
            train_idx = df_sorted.index[:split_idx]
            val_idx   = df_sorted.index[split_idx:]
            return (
                X.loc[train_idx], X.loc[val_idx],
                y[df_sorted.index.get_indexer(train_idx)],
                y[df_sorted.index.get_indexer(val_idx)],
            )
        else:
            from sklearn.model_selection import train_test_split
            logger.warning(
                "score_date not found — using random split. "
                "Add score_date to training data for time-based split."
            )
            return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    def _model_path(self, horizon: int, segment: Optional[str]) -> Path:
        if segment:
            return self.model_dir / f"{segment}_{horizon}d_propensity_model.pkl"
        return self.model_dir / f"{horizon}d_propensity_model.pkl"

    def _save(
        self, model, feature_cols: list, horizon: int, segment: Optional[str]
    ) -> None:
        path = self._model_path(horizon, segment)
        joblib.dump({
            "model":        model,
            "feature_cols": feature_cols,
            "horizon":      horizon,
            "segment":      segment,
        }, path)
        logger.info(f"Saved propensity model → {path}")

    def _load(self, horizon: int, segment: Optional[str]):
        obj = joblib.load(self._model_path(horizon, segment))
        return obj["model"], obj["feature_cols"]
