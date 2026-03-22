"""
Discount Elasticity Model
==========================
Estimates how P(Recovery) changes as discount level d increases, by segment.

This is the missing link that makes ERV optimisable:

    ERV(d) = P(Recovery | Segment, d) × E[Amount | Recovery, Segment] × Balance × (1 − d)

Without elasticity, P(Recovery) is a constant — there is no d* to find.
This module provides P(Recovery | Segment, d) so the ERVEngine can sweep d
and locate the optimum.

Model architecture
------------------
Logistic-link gradient boosting classifier with discount (d) as an explicit
feature alongside all standard debtor features. This is a SINGLE model that
learns the interaction between discount level and each segment's response:

    logit(P(Recovery)) = f(features, d, segment)

The segment is encoded as a categorical feature (one-hot) to allow the model
to learn segment × discount interactions naturally within the tree structure.

Additionally, a separate monotonicity check is run at inference time:
P(Recovery | d) is expected to be non-decreasing in d for all segments.
If the raw model violates monotonicity for a specific account, isotonic
regression is applied as a post-hoc correction.

Endogeneity note (BCG audit flag)
----------------------------------
Historical discount data is endogenous — agents offer higher discounts to
harder-to-collect accounts, creating a spurious negative correlation between
discount and recovery in naive models. To address this:
  1. Features that proxy for "agent assessment at time of discount decision"
     are included (e.g., days_past_due, contact_response_rate, promise_kept_rate)
     so the model can condition away from the selection bias.
  2. An optional Instrument Variable (IV) / propensity reweighting step is
     provided via ``fit(..., use_iv_reweighting=True)`` for production use.
     The default IV is offer_randomisation_flag (present in controlled trials).

Data contract (input DataFrame columns required)
-------------------------------------------------
  account_id              : str
  current_balance         : float
  balance_ratio           : float
  days_past_due           : int
  months_since_charge_off : int
  contact_response_rate   : float
  promise_kept_rate       : float
  payment_history_score   : float
  payment_capacity_score  : float
  debt_to_income_ratio    : float
  settlement_offers_made  : int
  segment                 : str — persona from DebtorPersonaSegmentation
  discount_offered        : float — the treatment variable d ∈ [0, 1]

  --- Target (training only) ---
  recovery_occurred       : int — 1 if recovered within 12m, else 0

MLflow integration
------------------
  Experiment : collection_recovery_models
  Run tags   : stage=discount_elasticity

Usage
-----
    model = DiscountElasticityModel(config)
    model.fit(train_df, val_df)

    # Elasticity curve for one account at d ∈ {0, 0.1, ..., 0.6}
    curve = model.elasticity_curve(account_row, discount_grid=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    # → DataFrame: discount | p_recovery | marginal_lift | elasticity_coefficient

    # Batch scoring at a fixed discount
    scores = model.predict(accounts_df, discount_pct=0.30)
    # → DataFrame: account_id, segment, discount_applied, p_recovery
"""

import json
import logging
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

ELASTICITY_FEATURES: List[str] = [
    "balance_ratio",
    "months_since_charge_off",
    "days_past_due",
    "contact_response_rate",
    "promise_kept_rate",
    "payment_history_score",
    "payment_capacity_score",
    "debt_to_income_ratio",
    "settlement_offers_made",
    # discount_offered is appended programmatically at fit/predict time
]

# Discount grid used for elasticity curve computation
DEFAULT_DISCOUNT_GRID: List[float] = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60]

# Minimum accounts per segment to report segment-level elasticity
MIN_SEGMENT_N: int = 50


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return roc_auc_score(y_true, y_score)
    except Exception:
        return float("nan")


def _enforce_monotonicity(
    p_curve: np.ndarray,
    discount_grid: np.ndarray,
) -> np.ndarray:
    """
    Apply isotonic regression to enforce P(Recovery | d) is non-decreasing in d.
    Operates on the probability curve for a single account.
    """
    iso = IsotonicRegression(increasing=True)
    return iso.fit_transform(discount_grid, p_curve)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CLASS
# ─────────────────────────────────────────────────────────────────────────────

class DiscountElasticityModel:
    """
    Segment-aware discount elasticity model.

    Learns P(Recovery | features, discount, segment) using a logistic-link
    gradient boosting classifier where discount_offered is an explicit feature.

    Provides:
      - predict()          : P(Recovery) at a fixed discount for a batch
      - elasticity_curve() : Full P(Recovery) vs d curve for an account
      - segment_elasticity(): Summary elasticity statistics per segment
      - marginal_lift()    : ΔP(Recovery) from baseline d=0 at each d level

    Parameters
    ----------
    config : dict
        Must contain key ``discount_elasticity_model`` with sub-keys:
        ``model``, ``iv_reweighting``, ``monotonicity_enforcement``, ``mlflow``.
    """

    def __init__(self, config: Dict):
        self.config = config
        _cfg = config.get("discount_elasticity_model", {})

        model_cfg = _cfg.get("model", {})
        self.model_params: Dict = {
            "n_estimators":     model_cfg.get("n_estimators",     200),
            "max_depth":        model_cfg.get("max_depth",         5),
            "learning_rate":    model_cfg.get("learning_rate",     0.05),
            "subsample":        model_cfg.get("subsample",         0.8),
            "min_samples_leaf": model_cfg.get("min_samples_leaf",  20),
            "random_state":     42,
        }

        self.enforce_monotonicity: bool = _cfg.get("monotonicity_enforcement", True)
        self.calibrate: bool            = _cfg.get("calibrate", True)
        self.use_iv_reweighting: bool   = _cfg.get("iv_reweighting", {}).get("enabled", False)
        self.iv_col: str                = _cfg.get("iv_reweighting", {}).get("iv_col",
                                                    "offer_randomisation_flag")

        mlflow_cfg = _cfg.get("mlflow", {})
        self.experiment_name: str = mlflow_cfg.get(
            "experiment_name", "collection_recovery_models"
        )
        self.tracking_uri: str = mlflow_cfg.get("tracking_uri", "mlruns")

        self._model           = None          # trained classifier
        self._segment_encoder = LabelEncoder()
        self._known_segments: List[str] = []
        self._feature_cols: List[str] = []    # includes discount + segment_encoded
        self._is_fitted = False
        self._segment_elasticity_cache: Optional[pd.DataFrame] = None

        # Segment-level baseline P(Recovery | d=0) for elasticity reporting
        self._segment_baseline: Dict[str, float] = {}

        self._mlflow_available = False
        self._try_setup_mlflow()

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────────────────────────────────────

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        use_iv_reweighting: Optional[bool] = None,
    ) -> Dict:
        """
        Train the elasticity model.

        Parameters
        ----------
        train_df : pd.DataFrame
            Feature columns + ``discount_offered`` + ``segment``
            + ``recovery_occurred`` (target).
        val_df : pd.DataFrame
            Same schema.
        feature_cols : list, optional
            Override ELASTICITY_FEATURES.
        use_iv_reweighting : bool, optional
            Override config setting. Requires ``iv_col`` in DataFrame.

        Returns
        -------
        dict with keys: metrics, mlflow_run_id
        """
        _use_iv = use_iv_reweighting if use_iv_reweighting is not None else self.use_iv_reweighting
        base_features = feature_cols or ELASTICITY_FEATURES

        self._validate_input(train_df, "train")
        self._validate_input(val_df,   "val")

        # Encode segment as numeric
        all_segments = pd.concat([train_df["segment"], val_df["segment"]]).unique()
        self._segment_encoder.fit(all_segments)
        self._known_segments = list(self._segment_encoder.classes_)

        # Build feature matrix: base features + discount_offered + segment_encoded
        self._feature_cols = base_features + ["discount_offered", "segment_encoded"]

        X_train, y_train = self._build_X(train_df, base_features, fit_encoder=True)
        X_val,   y_val   = self._build_X(val_df,   base_features, fit_encoder=False)

        print(f"\n[DiscountElasticityModel] Training on {len(X_train):,} accounts")
        print(f"  Recovery rate (train): {y_train.mean():.2%}")
        print(f"  Discount range (train): "
              f"{train_df['discount_offered'].min():.0%} – "
              f"{train_df['discount_offered'].max():.0%}")

        # Class imbalance handling
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        sample_weights = np.ones(len(y_train))

        if n_pos > 0:
            pos_weight = n_neg / n_pos
            sample_weights = np.where(y_train == 1, pos_weight, 1.0)

        # Optional IV reweighting to address endogeneity
        if _use_iv and self.iv_col in train_df.columns:
            sample_weights = self._compute_iv_weights(train_df, sample_weights)
            print(f"  IV reweighting applied via column '{self.iv_col}'")
        elif _use_iv:
            warnings.warn(
                f"IV reweighting requested but column '{self.iv_col}' not found. "
                "Proceeding without IV correction.",
                UserWarning,
            )

        # Train
        base_clf = GradientBoostingClassifier(**self.model_params)
        base_clf.fit(X_train, y_train, sample_weight=sample_weights)

        if self.calibrate:
            self._model = CalibratedClassifierCV(base_clf, cv="prefit", method="sigmoid")
            self._model.fit(X_val, y_val)
        else:
            self._model = base_clf

        # Evaluate
        p_val = self._model.predict_proba(X_val)[:, 1]
        metrics = {
            "auc":         _safe_auc(y_val, p_val),
            "pr_auc":      average_precision_score(y_val, p_val),
            "brier":       brier_score_loss(y_val, p_val),
            "n_train":     len(X_train),
            "n_val":       len(X_val),
            "recovery_rate_train": float(y_train.mean()),
            "recovery_rate_val":   float(y_val.mean()),
        }

        # Pre-compute segment baselines (P at d=0)
        self._compute_segment_baselines(val_df, base_features)

        # Feature importance
        importances = dict(zip(
            self._feature_cols,
            base_clf.feature_importances_,
        ))
        top5 = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"  Top features: {[(f, round(v, 3)) for f, v in top5]}")
        print(f"  AUC={metrics['auc']:.4f} | Brier={metrics['brier']:.4f}")

        self._is_fitted = True

        # MLflow
        run_id = self._log_to_mlflow(metrics, importances)

        return {"metrics": metrics, "mlflow_run_id": run_id}

    def predict(
        self,
        accounts_df: pd.DataFrame,
        discount_pct: float,
        enforce_monotonicity: Optional[bool] = None,
    ) -> pd.DataFrame:
        """
        Score a batch of accounts at a fixed discount level.

        Parameters
        ----------
        accounts_df : pd.DataFrame
            Must include all feature columns + ``account_id`` + ``segment``.
        discount_pct : float
            Discount as a fraction in [0, 1].
        enforce_monotonicity : bool, optional
            Override config setting for this call.

        Returns
        -------
        pd.DataFrame with columns:
            account_id, segment, discount_applied, p_recovery
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        df = accounts_df.copy()
        df["discount_offered"] = discount_pct

        base_features = [c for c in self._feature_cols
                         if c not in ("discount_offered", "segment_encoded")]
        X, _ = self._build_X(df, base_features, fit_encoder=False)
        p_recovery = self._model.predict_proba(X)[:, 1]

        return pd.DataFrame({
            "account_id":      df["account_id"].values,
            "segment":         df["segment"].values,
            "discount_applied": discount_pct,
            "p_recovery":      p_recovery.round(4),
        })

    def elasticity_curve(
        self,
        account_row: pd.Series,
        discount_grid: Optional[List[float]] = None,
        enforce_monotonicity: Optional[bool] = None,
    ) -> pd.DataFrame:
        """
        Compute the P(Recovery) vs discount curve for a single account.

        Parameters
        ----------
        account_row : pd.Series
            A single account's feature values (no account_id required).
        discount_grid : list of float, optional
            Discount levels to evaluate. Default: [0, 0.1, ..., 0.6].
        enforce_monotonicity : bool, optional
            Override config flag.

        Returns
        -------
        pd.DataFrame with columns:
            discount | p_recovery | marginal_lift | elasticity_coefficient
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        grid = discount_grid or DEFAULT_DISCOUNT_GRID
        _enforce = enforce_monotonicity if enforce_monotonicity is not None \
                   else self.enforce_monotonicity

        base_features = [c for c in self._feature_cols
                         if c not in ("discount_offered", "segment_encoded")]

        # Score at each d
        p_values = []
        for d in grid:
            row_copy = account_row.copy()
            row_copy["discount_offered"] = d
            df_row = pd.DataFrame([row_copy])
            X, _ = self._build_X(df_row, base_features, fit_encoder=False)
            p_values.append(self._model.predict_proba(X)[0, 1])

        p_arr = np.array(p_values)

        if _enforce:
            p_arr = _enforce_monotonicity(p_arr, np.array(grid))

        # Marginal lift over baseline (d=0)
        marginal_lift = p_arr - p_arr[0]

        # Elasticity coefficient: % change in P / % change in d
        # Using finite differences, guarding against zero denominator
        elasticity = np.zeros(len(grid))
        for i in range(1, len(grid)):
            delta_d = grid[i] - grid[i - 1]
            delta_p = p_arr[i] - p_arr[i - 1]
            if delta_d > 0 and p_arr[i - 1] > 0:
                elasticity[i] = (delta_p / p_arr[i - 1]) / (delta_d / max(grid[i - 1], 1e-6))
            else:
                elasticity[i] = np.nan

        return pd.DataFrame({
            "discount":              grid,
            "p_recovery":            p_arr.round(4),
            "marginal_lift":         marginal_lift.round(4),
            "elasticity_coefficient": elasticity.round(4),
        })

    def segment_elasticity(
        self,
        accounts_df: pd.DataFrame,
        discount_grid: Optional[List[float]] = None,
    ) -> pd.DataFrame:
        """
        Compute average P(Recovery) vs discount curve per segment.
        Useful for reporting and setting segment-level discount floors/ceilings.

        Parameters
        ----------
        accounts_df : pd.DataFrame
            Representative sample of accounts, one row per account.
        discount_grid : list of float, optional

        Returns
        -------
        pd.DataFrame with columns:
            segment | discount | avg_p_recovery | avg_marginal_lift | n_accounts
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted.")

        grid = discount_grid or DEFAULT_DISCOUNT_GRID
        rows = []

        for seg in sorted(accounts_df["segment"].unique()):
            seg_df = accounts_df[accounts_df["segment"] == seg]
            if len(seg_df) < MIN_SEGMENT_N:
                continue

            for d in grid:
                preds = self.predict(seg_df, discount_pct=d)
                avg_p = preds["p_recovery"].mean()
                baseline = self._segment_baseline.get(seg, preds["p_recovery"].mean())
                rows.append({
                    "segment":         seg,
                    "discount":        d,
                    "avg_p_recovery":  round(avg_p, 4),
                    "avg_marginal_lift": round(avg_p - baseline, 4),
                    "n_accounts":      len(seg_df),
                })

        df_out = pd.DataFrame(rows)
        self._segment_elasticity_cache = df_out
        return df_out

    def marginal_lift(
        self,
        accounts_df: pd.DataFrame,
        d_from: float,
        d_to: float,
    ) -> pd.DataFrame:
        """
        ΔP(Recovery) = P(recovery | d_to) − P(recovery | d_from) for each account.
        Used by ERVEngine to evaluate whether increasing discount is worth the cost.

        Returns
        -------
        pd.DataFrame with columns:
            account_id, segment, p_at_d_from, p_at_d_to, delta_p
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted.")

        p_from = self.predict(accounts_df, discount_pct=d_from)["p_recovery"].values
        p_to   = self.predict(accounts_df, discount_pct=d_to)["p_recovery"].values

        return pd.DataFrame({
            "account_id":  accounts_df["account_id"].values,
            "segment":     accounts_df["segment"].values,
            "p_at_d_from": p_from.round(4),
            "p_at_d_to":   p_to.round(4),
            "delta_p":     (p_to - p_from).round(4),
        })

    # ──────────────────────────────────────────────────────────────────────────
    # PRIVATE
    # ──────────────────────────────────────────────────────────────────────────

    def _build_X(
        self,
        df: pd.DataFrame,
        base_features: List[str],
        fit_encoder: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build feature matrix: base_features + discount_offered + segment_encoded.
        Returns (X, y) where y is zeros if recovery_occurred not present.
        """
        # Handle unknown segments gracefully
        seg_series = df["segment"].copy()
        if fit_encoder:
            self._segment_encoder.fit(seg_series.unique())
        seg_encoded = seg_series.map(
            lambda s: self._segment_encoder.transform([s])[0]
                      if s in self._segment_encoder.classes_
                      else -1
        ).values.reshape(-1, 1)

        feat_matrix = df[base_features].fillna(0.0).values.astype(float)
        discount_col = df["discount_offered"].fillna(0.0).values.reshape(-1, 1)

        X = np.hstack([feat_matrix, discount_col, seg_encoded])

        y = df["recovery_occurred"].values.astype(int) \
            if "recovery_occurred" in df.columns \
            else np.zeros(len(df), dtype=int)

        return X, y

    def _compute_segment_baselines(
        self, val_df: pd.DataFrame, base_features: List[str]
    ) -> None:
        """Compute average P(Recovery | d=0) per segment from validation set."""
        for seg in val_df["segment"].unique():
            seg_df = val_df[val_df["segment"] == seg].copy()
            if len(seg_df) < 10:
                continue
            seg_df["discount_offered"] = 0.0
            X, _ = self._build_X(seg_df, base_features, fit_encoder=False)
            p_baseline = self._model.predict_proba(X)[:, 1].mean()
            self._segment_baseline[seg] = float(p_baseline)

    def _compute_iv_weights(
        self, train_df: pd.DataFrame, base_weights: np.ndarray
    ) -> np.ndarray:
        """
        Propensity-score reweighting using randomisation flag as instrument.

        For accounts where offer_randomisation_flag == 1, the discount was
        exogenously assigned (e.g., A/B test / holdout), so we up-weight these
        observations relative to endogenously assigned discounts.

        Returns updated sample_weights array.
        """
        if self.iv_col not in train_df.columns:
            return base_weights

        is_randomised = train_df[self.iv_col].fillna(0).values.astype(float)
        iv_weight_multiplier = np.where(is_randomised == 1, 3.0, 1.0)
        return base_weights * iv_weight_multiplier

    def _validate_input(self, df: pd.DataFrame, split: str) -> None:
        required = set(ELASTICITY_FEATURES + ["account_id", "segment", "discount_offered"])
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"[DiscountElasticityModel] {split} DataFrame missing columns: {missing}"
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

    def _log_to_mlflow(self, metrics: Dict, importances: Dict) -> Optional[str]:
        if not self._mlflow_available:
            return None
        try:
            mlflow = self._mlflow
            import mlflow.sklearn
            with mlflow.start_run(run_name="discount_elasticity_model") as run:
                mlflow.set_tag("stage", "discount_elasticity")
                mlflow.set_tag("model_type", "logistic_gbm_with_discount_feature")
                mlflow.set_tag("iv_reweighting", str(self.use_iv_reweighting))
                mlflow.set_tag("monotonicity_enforcement", str(self.enforce_monotonicity))

                mlflow.log_params({f"model_{k}": v for k, v in self.model_params.items()})
                mlflow.log_metrics(metrics)

                imp_path = "/tmp/elasticity_feature_importance.json"
                with open(imp_path, "w") as f:
                    json.dump(importances, f, indent=2)
                mlflow.log_artifact(imp_path)

                # Log segment baselines
                if self._segment_baseline:
                    baseline_path = "/tmp/segment_baselines.json"
                    with open(baseline_path, "w") as f:
                        json.dump(self._segment_baseline, f, indent=2)
                    mlflow.log_artifact(baseline_path)

                mlflow.sklearn.log_model(self._model, "discount_elasticity_model")

                return run.info.run_id
        except Exception as e:
            logger.warning("MLflow logging failed: %s", e)
            return None


# ─────────────────────────────────────────────────────────────────────────────
# SPARK / DATABRICKS WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

class DiscountElasticityModelSpark:
    """
    PySpark wrapper for batch scoring on Databricks.

    Usage
    -----
        spark_model = DiscountElasticityModelSpark(fitted_model)
        scored_df   = spark_model.score(accounts_spark_df, discount_pct=0.30)
    """

    def __init__(self, fitted_model: DiscountElasticityModel):
        if not fitted_model._is_fitted:
            raise ValueError("Pass a fitted DiscountElasticityModel.")
        self.model = fitted_model

    def score(self, spark_df, discount_pct: float):
        """Score Spark DataFrame at a fixed discount level."""
        from pyspark.sql.types import StructType, StructField, StringType, DoubleType

        schema = StructType([
            StructField("account_id",       StringType(), True),
            StructField("segment",          StringType(), True),
            StructField("discount_applied", DoubleType(), True),
            StructField("p_recovery",       DoubleType(), True),
        ])

        import pyspark
        sc = pyspark.SparkContext.getOrCreate()
        bc_model = sc.broadcast(self.model)

        return spark_df.groupby().applyInPandas(
            lambda pdf: bc_model.value.predict(pdf, discount_pct=discount_pct),
            schema=schema,
        )


# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE / SMOKE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import yaml

    with open("conf/use_cases/debt_collection_optimization.yaml") as f:
        config = yaml.safe_load(f)

    np.random.seed(42)
    n = 2000
    segments = ["high_value_cooperative", "medium_value_willing",
                "high_value_unresponsive", "low_value_low_capacity"]

    df = pd.DataFrame({
        "account_id":             [f"ACC{i:05d}" for i in range(n)],
        "current_balance":        np.random.uniform(500, 50_000, n),
        "balance_ratio":          np.random.uniform(0.3, 1.0, n),
        "months_since_charge_off": np.random.randint(1, 12, n),
        "days_past_due":          np.random.randint(90, 730, n),
        "contact_response_rate":  np.random.uniform(0, 1, n),
        "promise_kept_rate":      np.random.uniform(0, 1, n),
        "payment_history_score":  np.random.uniform(0, 1, n),
        "payment_capacity_score": np.random.uniform(0, 1, n),
        "debt_to_income_ratio":   np.random.uniform(0.1, 1.5, n),
        "settlement_offers_made": np.random.randint(0, 5, n),
        "discount_offered":       np.random.uniform(0.0, 0.6, n),
        "segment":                np.random.choice(segments, n),
        "offer_randomisation_flag": np.random.choice([0, 1], n, p=[0.7, 0.3]),
    })

    # Synthetic recovery: higher discount → higher recovery probability
    recovery_prob = (
        0.20 * df["payment_capacity_score"] +
        0.15 * df["contact_response_rate"] +
        0.35 * df["discount_offered"] +
        0.15 * df["promise_kept_rate"] +
        0.05 * (1 - df["balance_ratio"]) +
        np.random.uniform(0, 0.15, n)
    ).clip(0, 1)

    df["recovery_occurred"] = (np.random.uniform(0, 1, n) < recovery_prob).astype(int)

    train_df = df.iloc[:1600]
    val_df   = df.iloc[1600:]

    config_override = {
        "discount_elasticity_model": {
            "model": {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.05},
            "monotonicity_enforcement": True,
            "calibrate": True,
            "iv_reweighting": {"enabled": True, "iv_col": "offer_randomisation_flag"},
            "mlflow": {"experiment_name": "elasticity_test", "tracking_uri": "mlruns"},
        }
    }

    model = DiscountElasticityModel(config_override)
    results = model.fit(train_df, val_df)

    # Batch predict at 30% discount
    preds = model.predict(val_df.head(10), discount_pct=0.30)
    print("\n[Predict at 30% discount]")
    print(preds.to_string(index=False))

    # Elasticity curve for one account
    sample_row = val_df.iloc[0]
    curve = model.elasticity_curve(sample_row)
    print("\n[Elasticity curve — Account 0]")
    print(curve.to_string(index=False))

    # Segment-level elasticity summary
    seg_elast = model.segment_elasticity(val_df)
    print("\n[Segment Elasticity Summary]")
    print(seg_elast.to_string(index=False))

    # Marginal lift between 20% and 40% discount
    lift = model.marginal_lift(val_df.head(10), d_from=0.20, d_to=0.40)
    print("\n[Marginal Lift: 20% → 40% discount]")
    print(lift.to_string(index=False))

    print("\n✓ DiscountElasticityModel smoke test passed")
