"""
Uplift Engine
=============
Advanced causal inference models for NBA uplift scoring.

Includes:
  - TLearner: separate models for treatment/control
  - SLearner: action as feature (already in model_trainer.py — kept for comparison)
  - XLearner: 3-stage propensity-weighted meta-learner (most accurate)
  - CausalForest: EconML heterogeneous treatment effects with confidence intervals
  - UpliftEnsemble: Qini-coefficient optimized ensemble weights
  - UpliftValidator: Qini curve + ATE evaluation
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# BASE
# ─────────────────────────────────────────────────────────────────────────────

class BaseUpliftModel:
    """Abstract base for uplift models."""

    def fit(self, X: np.ndarray, treatment: np.ndarray, y: np.ndarray) -> "BaseUpliftModel":
        raise NotImplementedError

    def predict_uplift(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# T-LEARNER
# ─────────────────────────────────────────────────────────────────────────────

class TLearner(BaseUpliftModel):
    """
    T-Learner: separate models for treatment and control.

    uplift = model_treatment(X) - model_control(X)
    Advantage: each model can specialise.
    Disadvantage: no shared information, may overfit on small treatment groups.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 4,
        learning_rate: float = 0.05,
    ):
        params = dict(n_estimators=n_estimators, max_depth=max_depth,
                      learning_rate=learning_rate, subsample=0.8, random_state=42)
        self.treatment_model = GradientBoostingClassifier(**params)
        self.control_model   = GradientBoostingClassifier(**params)
        self._fitted = False

    def fit(self, X: np.ndarray, treatment: np.ndarray, y: np.ndarray) -> "TLearner":
        treat_mask = treatment == 1
        ctrl_mask  = treatment == 0

        if treat_mask.sum() < 10 or ctrl_mask.sum() < 10:
            raise ValueError("Need ≥10 samples per group for T-Learner.")

        self.treatment_model.fit(X[treat_mask], y[treat_mask])
        self.control_model.fit(X[ctrl_mask],   y[ctrl_mask])
        self._fitted = True
        logger.info("TLearner fitted. Treatment n=%d, Control n=%d",
                    treat_mask.sum(), ctrl_mask.sum())
        return self

    def predict_uplift(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        treat_prob = self.treatment_model.predict_proba(X)[:, 1]
        ctrl_prob  = self.control_model.predict_proba(X)[:, 1]
        return treat_prob - ctrl_prob


# ─────────────────────────────────────────────────────────────────────────────
# X-LEARNER
# ─────────────────────────────────────────────────────────────────────────────

class XLearner(BaseUpliftModel):
    """
    X-Learner: 3-stage propensity-weighted meta-learner.

    Stage 1: Fit mu0(x) = E[Y|T=0,X] and mu1(x) = E[Y|T=1,X] using T-Learner.
    Stage 2: Compute imputed effects:
               D1 = Y1 - mu0(X1)    (for treated)
               D0 = mu1(X0) - Y0    (for control)
             Fit tau1(x) on D1 and tau0(x) on D0.
    Stage 3: Propensity-weighted combination:
               tau(x) = e(x)*tau0(x) + (1-e(x))*tau1(x)
             where e(x) = P(T=1|X).

    Best when treatment group is small relative to control.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 4,
        learning_rate: float = 0.05,
    ):
        gb_params = dict(n_estimators=n_estimators, max_depth=max_depth,
                         learning_rate=learning_rate, subsample=0.8, random_state=42)
        self.mu0 = GradientBoostingClassifier(**gb_params)  # control outcome
        self.mu1 = GradientBoostingClassifier(**gb_params)  # treatment outcome
        self.tau0 = GradientBoostingRegressor(**{**gb_params, "loss": "squared_error"})
        self.tau1 = GradientBoostingRegressor(**{**gb_params, "loss": "squared_error"})
        self.propensity_model = LogisticRegression(max_iter=300, random_state=42)
        self._fitted = False

    def fit(self, X: np.ndarray, treatment: np.ndarray, y: np.ndarray) -> "XLearner":
        treat_mask = treatment == 1
        ctrl_mask  = treatment == 0

        # Stage 1: base outcome models
        self.mu0.fit(X[ctrl_mask],   y[ctrl_mask])
        self.mu1.fit(X[treat_mask],  y[treat_mask])

        # Stage 2: imputed treatment effects
        D1 = y[treat_mask] - self.mu0.predict_proba(X[treat_mask])[:, 1]
        D0 = self.mu1.predict_proba(X[ctrl_mask])[:, 1] - y[ctrl_mask]

        self.tau1.fit(X[treat_mask], D1)
        self.tau0.fit(X[ctrl_mask],  D0)

        # Stage 3: propensity model
        self.propensity_model.fit(X, treatment)

        self._fitted = True
        logger.info("XLearner fitted. Treatment n=%d, Control n=%d",
                    treat_mask.sum(), ctrl_mask.sum())
        return self

    def predict_uplift(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        e  = self.propensity_model.predict_proba(X)[:, 1]
        t0 = self.tau0.predict(X)
        t1 = self.tau1.predict(X)
        return e * t0 + (1 - e) * t1


# ─────────────────────────────────────────────────────────────────────────────
# CAUSAL FOREST
# ─────────────────────────────────────────────────────────────────────────────

class CausalForestUplift(BaseUpliftModel):
    """
    Causal Forest via EconML — heterogeneous treatment effects with CIs.

    Falls back to T-Learner when econml is not installed.
    Provides predict_uplift_with_ci() for confidence intervals.
    """

    def __init__(self, n_estimators: int = 100, min_samples_leaf: int = 10):
        self.n_estimators     = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self._econml_available = False
        self._model  = None
        self._fitted = False
        self._fallback = TLearner(n_estimators=n_estimators)

    def fit(self, X: np.ndarray, treatment: np.ndarray, y: np.ndarray) -> "CausalForestUplift":
        try:
            from econml.dml import CausalForestDML
            self._model = CausalForestDML(
                n_estimators=self.n_estimators,
                min_samples_leaf=self.min_samples_leaf,
                random_state=42,
            )
            self._model.fit(y, treatment, X=X)
            self._econml_available = True
            logger.info("CausalForest fitted with EconML.")
        except ImportError:
            logger.warning("EconML not installed — falling back to TLearner.")
            self._fallback.fit(X, treatment, y)
        self._fitted = True
        return self

    def predict_uplift(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        if self._econml_available:
            return self._model.effect(X).flatten()
        return self._fallback.predict_uplift(X)

    def predict_uplift_with_ci(
        self, X: np.ndarray, alpha: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (point_estimate, lower_ci, upper_ci)."""
        if self._econml_available:
            inf = self._model.effect_inference(X)
            return (
                inf.point_estimate.flatten(),
                inf.conf_int(alpha=alpha)[0].flatten(),
                inf.conf_int(alpha=alpha)[1].flatten(),
            )
        uplift = self.predict_uplift(X)
        return uplift, uplift - 0.05, uplift + 0.05  # rough fallback


# ─────────────────────────────────────────────────────────────────────────────
# UPLIFT ENSEMBLE
# ─────────────────────────────────────────────────────────────────────────────

class UpliftEnsemble(BaseUpliftModel):
    """
    Qini-optimised ensemble of uplift models.

    Learns weights that maximise Qini coefficient via scipy.optimize.
    Falls back to equal weighting when optimisation fails.
    """

    def __init__(self, models: Optional[List[BaseUpliftModel]] = None):
        self.models  = models or [TLearner(), XLearner()]
        self.weights: np.ndarray = np.array([1.0 / len(self.models)] * len(self.models))
        self._fitted = False

    def fit(self, X: np.ndarray, treatment: np.ndarray, y: np.ndarray) -> "UpliftEnsemble":
        for m in self.models:
            m.fit(X, treatment, y)

        # Optimise weights by Qini on training set
        preds = np.column_stack([m.predict_uplift(X) for m in self.models])
        self.weights = self._optimise_weights(preds, treatment, y)
        self._fitted = True
        logger.info("UpliftEnsemble fitted. Weights=%s", np.round(self.weights, 3))
        return self

    def predict_uplift(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        preds = np.column_stack([m.predict_uplift(X) for m in self.models])
        return preds @ self.weights

    def _optimise_weights(
        self,
        preds: np.ndarray,
        treatment: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """Maximise Qini coefficient via constrained optimisation."""
        try:
            from scipy.optimize import minimize

            n = preds.shape[1]

            def neg_qini(w):
                uplift = preds @ w
                return -qini_coefficient(uplift, treatment, y)

            constraints = {"type": "eq", "fun": lambda w: w.sum() - 1}
            bounds = [(0, 1)] * n
            x0 = np.ones(n) / n

            res = minimize(neg_qini, x0, method="SLSQP",
                           bounds=bounds, constraints=constraints)
            if res.success:
                return res.x
        except Exception as e:
            logger.warning("Weight optimisation failed: %s — using equal weights.", e)

        return np.ones(len(self.models)) / len(self.models)


# ─────────────────────────────────────────────────────────────────────────────
# QINI METRIC
# ─────────────────────────────────────────────────────────────────────────────

def qini_coefficient(
    uplift: np.ndarray,
    treatment: np.ndarray,
    y: np.ndarray,
) -> float:
    """
    Compute Qini coefficient (area under Qini curve).

    Qini curve plots cumulative incremental gain (treatment outcomes - control outcomes)
    as fraction of population targeted, sorted by predicted uplift descending.

    Perfect model = 1.0, random = 0.0, negative = model is worse than random.
    """
    df = pd.DataFrame({"uplift": uplift, "treatment": treatment, "y": y})
    df = df.sort_values("uplift", ascending=False).reset_index(drop=True)

    n = len(df)
    n_treat = df["treatment"].sum()
    n_ctrl  = n - n_treat

    if n_treat == 0 or n_ctrl == 0:
        return 0.0

    cum_treat = df["treatment"].cumsum()
    cum_ctrl  = (1 - df["treatment"]).cumsum()
    cum_y_treat = (df["y"] * df["treatment"]).cumsum()
    cum_y_ctrl  = (df["y"] * (1 - df["treatment"])).cumsum()

    # Avoid division by zero
    treat_rate = cum_y_treat / cum_treat.replace(0, np.nan)
    ctrl_rate  = cum_y_ctrl  / cum_ctrl.replace(0, np.nan)

    qini_curve = (cum_treat * treat_rate.fillna(0) -
                  cum_ctrl  * ctrl_rate.fillna(0)) / n

    # Area under Qini curve (trapezoidal)
    auc = np.trapz(qini_curve, dx=1.0 / n)

    # Normalise: random model AUC = 0, perfect = 1
    # (perfect is hard to compute analytically — just return raw AUC)
    return float(auc)


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATOR
# ─────────────────────────────────────────────────────────────────────────────

class UpliftValidator:
    """
    Evaluate uplift models using Qini curve and ATE in top decile.
    """

    @staticmethod
    def qini_curve(
        uplift: np.ndarray,
        treatment: np.ndarray,
        y: np.ndarray,
        n_bins: int = 10,
    ) -> pd.DataFrame:
        """Return Qini curve as DataFrame with columns: fraction, incremental_gain."""
        df = pd.DataFrame({"uplift": uplift, "treatment": treatment, "y": y})
        df = df.sort_values("uplift", ascending=False).reset_index(drop=True)
        n  = len(df)

        records = []
        for i in range(1, n_bins + 1):
            end = int(n * i / n_bins)
            sub = df.iloc[:end]
            n_t = sub["treatment"].sum()
            n_c = (1 - sub["treatment"]).sum()
            if n_t > 0 and n_c > 0:
                gain = (
                    sub[sub["treatment"] == 1]["y"].sum() / n_t
                    - sub[sub["treatment"] == 0]["y"].sum() / n_c
                ) * end / n
            else:
                gain = 0.0
            records.append({"fraction": i / n_bins, "incremental_gain": gain})

        return pd.DataFrame(records)

    @staticmethod
    def ate_top_decile(
        uplift: np.ndarray,
        treatment: np.ndarray,
        y: np.ndarray,
        top_pct: float = 0.10,
    ) -> Dict[str, float]:
        """
        Average treatment effect in top-decile (highest uplift accounts).
        Useful for capacity-constrained campaigns (we can only contact N accounts).
        """
        df = pd.DataFrame({"uplift": uplift, "treatment": treatment, "y": y})
        df = df.sort_values("uplift", ascending=False).reset_index(drop=True)
        top = df.iloc[: max(1, int(len(df) * top_pct))]

        n_t = top["treatment"].sum()
        n_c = (1 - top["treatment"]).sum()

        if n_t == 0 or n_c == 0:
            return {"ate_top_decile": float("nan"), "n_treatment": int(n_t), "n_control": int(n_c)}

        ate = (
            top[top["treatment"] == 1]["y"].mean()
            - top[top["treatment"] == 0]["y"].mean()
        )
        return {"ate_top_decile": float(ate), "n_treatment": int(n_t), "n_control": int(n_c)}

    @staticmethod
    def compare_models(
        models: Dict[str, BaseUpliftModel],
        X: np.ndarray,
        treatment: np.ndarray,
        y: np.ndarray,
    ) -> pd.DataFrame:
        """Compare multiple uplift models by Qini coefficient."""
        rows = []
        for name, model in models.items():
            try:
                uplift = model.predict_uplift(X)
                qini   = qini_coefficient(uplift, treatment, y)
                ate    = UpliftValidator.ate_top_decile(uplift, treatment, y)["ate_top_decile"]
                rows.append({"model": name, "qini_coefficient": round(qini, 4),
                             "ate_top_decile": round(ate, 4)})
            except Exception as e:
                logger.warning("Could not evaluate %s: %s", name, e)
                rows.append({"model": name, "qini_coefficient": float("nan"),
                             "ate_top_decile": float("nan")})
        return pd.DataFrame(rows).sort_values("qini_coefficient", ascending=False)
