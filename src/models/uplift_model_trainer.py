"""
UpliftModelTrainer
==================
T-learner uplift model for the CardX/SCB Thailand debt collections recovery system.
Implements per-segment T-learner using two LightGBM models (treated + control).

CONFIGURATION INSTRUCTIONS
---------------------------

DESIGN: T-LEARNER
  T-learner = two separate outcome models:
    - T_model: trained on TREATMENT group records — predicts P(paid_180d | treated)
    - C_model: trained on CONTROL group records  — predicts P(paid_180d | control)
    - Uplift: τ(x) = T_model.predict(x) - C_model.predict(x)

  Positive τ means treating this account is expected to improve P(recovery).
  Negative τ means treatment HURTS recovery (e.g., contacting a strategic defaulter
  prompts them to dispute the debt).

WHY NOT INCLUDE ERV AS A FEATURE:
  ERV (Expected Recovery Value) = P(pay) × E(amount|paid). It is computed from
  propensity_30d and outstanding balance — both of which are already in UPLIFT_FEATURES.
  Including ERV would create a circular dependency: the uplift model would be learning
  to predict features that are themselves derived from the propensity model input.
  ERV is used downstream for treatment decision logic, not as an input to the
  uplift model.

UPLIFT_FEATURES:
  Features used to train both T_model and C_model. Deliberately excludes:
    - account_id (identifier, not predictive)
    - treatment indicators (holdout_group, holdout_flag — target leakage)
    - outcome variables (paid_180d, recovery_amount — label leakage)
    - ERV (circular dependency — see above)

  discount_offered and channel are treatment variables included intentionally.
  Including them allows the model to learn how discount level and channel
  shift P(pay), which is exactly the uplift signal we want to capture.
  Note: discount_offered is a treatment variable, not just a covariate.
  The T-learner correctly handles this because each sub-model only sees one
  side of the treatment assignment.

MODEL_PATH:
  models/uplift_models/{segment}_treated.pkl
  models/uplift_models/{segment}_control.pkl
  Override via UPLIFT_MODEL_DIR environment variable.

SEGMENT_MIN_SAMPLES (default 100):
  Minimum samples required in treated AND control group to train a segment model.
  If either group is below this threshold the segment is skipped with a WARNING.
  Increase if you observe high variance in held-out AUC across retraining runs.
  Decrease only if portfolio is very small (< 500 accounts per segment total).

WHY PER-SEGMENT MODELS:
  Segments A–D represent materially different behavioural cohorts:
    Segment A: High-propensity accounts — large discount sensitivity
    Segment B: Moderate-propensity — channel sensitivity dominates
    Segment C: Low-propensity — minimal treatment response
    Segment D: Very low propensity — mostly strategic/dormant
  Pooling all segments into one model would dilute these distinct response
  patterns. A pooled model would be dominated by the majority segment and
  give poor uplift estimates for minority segments.

TRANSITION TO X-LEARNER:
  T-learner is appropriate for Phase 2 because treated:control ratio is ~19:1
  (5% holdout = ~95% treatment). T-learner handles this reasonably well when
  both groups have enough samples.

  Transition to X-learner when:
    - You have ~12+ months of labelled data (enough control observations)
    - Treated:control ratio exceeds 10:1 consistently
  X-learner imputes counterfactual outcomes using cross-group predictions,
  which handles severe imbalance better. T-learner is simpler and sufficient
  for Phase 2.
"""

import logging
import os
import warnings
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Feature list ──────────────────────────────────────────────────────────────
# DO NOT include: account_id, treatment indicators (holdout_group/holdout_flag),
# outcome variables (paid_180d, recovery_amount), ERV (circular dependency).
# discount_offered and channel are treatment variables — included intentionally
# so models learn discount/channel response curves per segment.
UPLIFT_FEATURES = [
    "propensity_30d",
    "propensity_180d",
    "willingness_score",
    "capacity_score",
    "erv_at_d_optimal",
    "low_confidence_flag",
    "behavioural_persona",    # categorical — label-encoded before fit
    "discount_offered",       # treatment variable — discount IS a treatment
]

# ── Default LightGBM parameters ───────────────────────────────────────────────
DEFAULT_LGB_PARAMS = {
    "objective":             "binary",
    "metric":                "auc",
    "num_leaves":            31,
    "learning_rate":         0.05,
    "n_estimators":          300,
    "early_stopping_rounds": 30,
    "verbose":               -1,
}

# ── Constants ─────────────────────────────────────────────────────────────────
SEGMENT_MIN_SAMPLES: int = 100
UPLIFT_MODEL_DIR: str = os.environ.get("UPLIFT_MODEL_DIR", "models/uplift_models")


class UpliftModelTrainer:
    """
    T-learner uplift model trainer for debt collections recovery.

    Trains two LightGBM classifiers per SIGNAL_SEGMENT:
      - T_model: outcome model on TREATMENT group
      - C_model: outcome model on CONTROL  group

    Uplift score: τ(x) = T_model.predict_proba(x)[:,1] - C_model.predict_proba(x)[:,1]
    """

    def __init__(
        self,
        model_dir: Optional[str] = None,
        lgb_params: Optional[dict] = None,
        segment_min_samples: int = SEGMENT_MIN_SAMPLES,
    ):
        """
        Args:
            model_dir:            Directory to save/load .pkl model files.
                                  Defaults to UPLIFT_MODEL_DIR env var or
                                  "models/uplift_models".
            lgb_params:           LightGBM params dict. Merged with defaults.
            segment_min_samples:  Min samples per treated/control group to train.
        """
        self.model_dir = Path(model_dir or UPLIFT_MODEL_DIR)
        self.lgb_params = {**DEFAULT_LGB_PARAMS, **(lgb_params or {})}
        self.segment_min_samples = segment_min_samples
        self._models: dict = {}  # {segment: {"treated": model, "control": model}}

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> dict:
        """
        Train T-learner uplift models per SIGNAL_SEGMENT.

        Args:
            df: Historical records with columns:
                  account_id, signal_segment, holdout_group ("TREATMENT"/"CONTROL"),
                  paid_180d (0/1 outcome label), plus all UPLIFT_FEATURES columns.

        Returns:
            dict: {segment: {auc_treated, auc_control, mean_uplift,
                              pct_positive_uplift, n_treated, n_control}}
        """
        self._validate_fit_input(df)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        results = {}
        segments = df["signal_segment"].dropna().unique()
        logger.info(f"UpliftModelTrainer.fit | segments={list(segments)} | total_rows={len(df):,}")

        for segment in segments:
            seg_df = df[df["signal_segment"] == segment].copy()
            treated_df = seg_df[seg_df["holdout_group"] == "TREATMENT"].copy()
            control_df = seg_df[seg_df["holdout_group"] == "CONTROL"].copy()

            n_treated = len(treated_df)
            n_control = len(control_df)

            if n_treated < self.segment_min_samples or n_control < self.segment_min_samples:
                logger.warning(
                    f"Segment {segment}: insufficient samples "
                    f"(treated={n_treated}, control={n_control}, "
                    f"min={self.segment_min_samples}) — skipping."
                )
                continue

            logger.info(
                f"Segment {segment}: training T-learner | "
                f"n_treated={n_treated:,} | n_control={n_control:,}"
            )

            X_treated, y_treated = self._prepare_features(treated_df)
            X_control, y_control = self._prepare_features(control_df)

            t_model, auc_treated = self._fit_lgb(X_treated, y_treated, f"{segment}_treated")
            c_model, auc_control = self._fit_lgb(X_control, y_control, f"{segment}_control")

            # Save models
            self._save_model(t_model, segment, "treated")
            self._save_model(c_model, segment, "control")
            self._models[segment] = {"treated": t_model, "control": c_model}

            # Compute uplift on full segment for diagnostics
            X_all, _ = self._prepare_features(seg_df)
            tau = self._compute_tau(t_model, c_model, X_all)
            mean_uplift       = float(np.mean(tau))
            pct_positive      = float(np.mean(tau > 0)) * 100.0

            logger.info(
                f"Segment {segment} | auc_treated={auc_treated:.4f} | "
                f"auc_control={auc_control:.4f} | mean_uplift={mean_uplift:.4f} | "
                f"pct_positive_uplift={pct_positive:.1f}%"
            )

            results[segment] = {
                "auc_treated":        round(auc_treated, 4),
                "auc_control":        round(auc_control, 4),
                "mean_uplift":        round(mean_uplift, 4),
                "pct_positive_uplift": round(pct_positive, 2),
                "n_treated":          n_treated,
                "n_control":          n_control,
            }

        logger.info(f"UpliftModelTrainer.fit complete | trained_segments={list(results.keys())}")
        return results

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict_uplift(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute uplift scores τ(x) for all accounts.

        For each account loads its segment's T_model and C_model, then:
            τ(x) = T_model.predict_proba(x)[:,1] - C_model.predict_proba(x)[:,1]

        Accounts in segments without fitted models receive τ=0.0 (with warning).

        Args:
            df: DataFrame with signal_segment column and all UPLIFT_FEATURES columns.

        Returns:
            pd.Series of τ values, same index as df.
        """
        tau_values = pd.Series(index=df.index, dtype=float)
        tau_values[:] = 0.0

        segments = df["signal_segment"].dropna().unique()

        for segment in segments:
            seg_mask = df["signal_segment"] == segment
            seg_df = df[seg_mask].copy()

            models = self._load_segment_models(segment)
            if models is None:
                logger.warning(
                    f"predict_uplift: no fitted model for segment={segment} "
                    f"({seg_mask.sum():,} accounts) — setting tau=0.0"
                )
                continue

            t_model = models["treated"]
            c_model = models["control"]

            X, _ = self._prepare_features(seg_df)
            tau = self._compute_tau(t_model, c_model, X)
            tau_values.loc[seg_mask] = tau

        return tau_values

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _prepare_features(self, df: pd.DataFrame):
        """
        Extract and encode UPLIFT_FEATURES from df.

        Handles:
          - Missing columns — filled with 0.0 with a warning
          - Categorical features (behavioural_persona) — label-encoded
          - Returns (X, y) where y is paid_180d if present, else None
        """
        X = df.copy()

        # Label-encode categorical features
        if "behavioural_persona" in X.columns:
            X["behavioural_persona"] = X["behavioural_persona"].astype("category").cat.codes

        # Select only UPLIFT_FEATURES that exist in df
        available_features = [f for f in UPLIFT_FEATURES if f in X.columns]
        missing = [f for f in UPLIFT_FEATURES if f not in X.columns]
        if missing:
            logger.warning(f"_prepare_features: missing columns {missing} — filling with 0.0")
            for col in missing:
                X[col] = 0.0

        X_out = X[UPLIFT_FEATURES].fillna(0.0)

        y_out = df["paid_180d"] if "paid_180d" in df.columns else None
        return X_out, y_out

    def _fit_lgb(self, X: pd.DataFrame, y: pd.Series, model_tag: str):
        """Fit a single LightGBM classifier. Returns (model, auc)."""
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("lightgbm is required for UpliftModelTrainer. Install with: pip install lightgbm")

        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score

        params = dict(self.lgb_params)
        early_stopping = params.pop("early_stopping_rounds", 30)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        model = lgb.LGBMClassifier(**params)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(early_stopping, verbose=False),
                           lgb.log_evaluation(period=-1)],
            )

        y_pred_proba = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred_proba)

        logger.info(f"LGB fit [{model_tag}] | auc={auc:.4f} | best_iteration={model.best_iteration_}")
        return model, auc

    def _compute_tau(self, t_model, c_model, X: pd.DataFrame) -> np.ndarray:
        """Compute τ(x) = T_model.predict_proba(x)[:,1] - C_model.predict_proba(x)[:,1]"""
        p_treated = t_model.predict_proba(X)[:, 1]
        p_control = c_model.predict_proba(X)[:, 1]
        return p_treated - p_control

    def _save_model(self, model, segment: str, group: str) -> None:
        """Save model to {model_dir}/{segment}_{group}.pkl"""
        path = self.model_dir / f"{segment}_{group}.pkl"
        joblib.dump(model, path)
        logger.info(f"Saved uplift model → {path}")

    def _load_segment_models(self, segment: str) -> Optional[dict]:
        """
        Load treated and control models for a segment.

        Returns dict {"treated": model, "control": model} or None if not found.
        Uses in-memory cache first, then disk.
        """
        if segment in self._models:
            return self._models[segment]

        treated_path = self.model_dir / f"{segment}_treated.pkl"
        control_path = self.model_dir / f"{segment}_control.pkl"

        if not treated_path.exists() or not control_path.exists():
            return None

        try:
            t_model = joblib.load(treated_path)
            c_model = joblib.load(control_path)
            self._models[segment] = {"treated": t_model, "control": c_model}
            logger.info(f"Loaded uplift models for segment={segment}")
            return self._models[segment]
        except Exception as e:
            logger.error(f"Failed to load models for segment={segment}: {e}")
            return None

    def _validate_fit_input(self, df: pd.DataFrame) -> None:
        """Validate required columns exist in training dataframe."""
        required = {"account_id", "signal_segment", "holdout_group", "paid_180d"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"UpliftModelTrainer.fit: missing required columns: {sorted(missing)}"
            )
        valid_groups = {"TREATMENT", "CONTROL"}
        actual_groups = set(df["holdout_group"].dropna().unique())
        if not actual_groups.issubset(valid_groups):
            raise ValueError(
                f"holdout_group must be 'TREATMENT' or 'CONTROL'. "
                f"Found: {actual_groups - valid_groups}"
            )
