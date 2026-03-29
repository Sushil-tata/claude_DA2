"""
Uplift Modeling (Treatment Effect Estimation)

Estimate the causal effect of treatments/interventions on customer behavior.

Unlike traditional ML (predicts outcome), uplift models predict:
- How much will treatment CHANGE the outcome?
- Which customers benefit most from treatment?

Use Cases:
- Marketing campaign targeting (who to contact?)
- Retention interventions (who needs outreach?)
- Pricing optimization (who to discount?)
- A/B test analysis

Modeling Approaches:
1. T-Learner: Train separate models for treatment and control
2. S-Learner: Single model with treatment as feature
3. X-Learner: More sophisticated meta-learner
4. Causal Forest: Tree-based approach

Usage:
    model = TLearner(
        base_model=GradientBoostingRegressor()
    )

    uplift = model.fit_predict(
        X_train, y_train, treatment_train,
        X_test
    )
"""

import logging
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class TLearner:
    """
    T-Learner for uplift modeling.

    Trains two separate models:
    - Model for treatment group
    - Model for control group

    Uplift = E[Y|X, T=1] - E[Y|X, T=0]
    """

    def __init__(self, base_model=None):
        """
        Initialize T-Learner.

        Args:
            base_model: Base model to use (cloned for treatment and control)
        """
        self.base_model = base_model or GradientBoostingRegressor()
        self.treatment_model = None
        self.control_model = None

        logger.info(f"T-Learner initialized with {type(self.base_model).__name__}")

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        treatment: pd.Series
    ):
        """
        Fit T-Learner.

        Args:
            X: Features
            y: Outcomes
            treatment: Treatment indicator (1 = treated, 0 = control)
        """
        from sklearn.base import clone

        # Split into treatment and control groups
        treatment_mask = treatment == 1
        control_mask = treatment == 0

        X_treatment = X[treatment_mask]
        y_treatment = y[treatment_mask]

        X_control = X[control_mask]
        y_control = y[control_mask]

        logger.info(f"Fitting T-Learner:")
        logger.info(f"  Treatment group: {len(X_treatment):,}")
        logger.info(f"  Control group: {len(X_control):,}")

        # Train treatment model
        self.treatment_model = clone(self.base_model)
        self.treatment_model.fit(X_treatment, y_treatment)

        # Train control model
        self.control_model = clone(self.base_model)
        self.control_model.fit(X_control, y_control)

        logger.info("✓ T-Learner fitted")

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict uplift (treatment effect).

        Args:
            X: Features

        Returns:
            Uplift scores (higher = more benefit from treatment)
        """
        if self.treatment_model is None or self.control_model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Predict outcomes under treatment and control
        y_treatment_pred = self.treatment_model.predict(X)
        y_control_pred = self.control_model.predict(X)

        # Uplift = difference
        uplift = y_treatment_pred - y_control_pred

        return uplift


class SLearner:
    """
    S-Learner for uplift modeling.

    Trains single model with treatment as a feature.

    Uplift = f(X, T=1) - f(X, T=0)
    """

    def __init__(self, base_model=None):
        """
        Initialize S-Learner.

        Args:
            base_model: Base model to use
        """
        self.base_model = base_model or GradientBoostingRegressor()
        self.model = None

        logger.info(f"S-Learner initialized with {type(self.base_model).__name__}")

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        treatment: pd.Series
    ):
        """
        Fit S-Learner.

        Args:
            X: Features
            y: Outcomes
            treatment: Treatment indicator
        """
        # Add treatment as feature
        X_with_treatment = X.copy()
        X_with_treatment["treatment"] = treatment

        logger.info(f"Fitting S-Learner:")
        logger.info(f"  Total samples: {len(X):,}")

        # Train single model
        self.model = self.base_model
        self.model.fit(X_with_treatment, y)

        logger.info("✓ S-Learner fitted")

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict uplift.

        Args:
            X: Features

        Returns:
            Uplift scores
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Predict with treatment = 1
        X_treatment = X.copy()
        X_treatment["treatment"] = 1
        y_treatment_pred = self.model.predict(X_treatment)

        # Predict with treatment = 0
        X_control = X.copy()
        X_control["treatment"] = 0
        y_control_pred = self.model.predict(X_control)

        # Uplift = difference
        uplift = y_treatment_pred - y_control_pred

        return uplift


class XLearner:
    """
    X-Learner for uplift modeling.

    More sophisticated meta-learner that:
    1. Trains models for treatment and control (like T-Learner)
    2. Computes pseudo-outcomes (imputed treatment effects)
    3. Trains models to predict these pseudo-outcomes
    4. Combines predictions with propensity weights
    """

    def __init__(self, base_model=None):
        """
        Initialize X-Learner.

        Args:
            base_model: Base model to use
        """
        self.base_model = base_model or GradientBoostingRegressor()
        self.treatment_model = None
        self.control_model = None
        self.tau_treatment_model = None
        self.tau_control_model = None
        self.propensity_model = None

        logger.info(f"X-Learner initialized with {type(self.base_model).__name__}")

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        treatment: pd.Series
    ):
        """
        Fit X-Learner.

        Args:
            X: Features
            y: Outcomes
            treatment: Treatment indicator
        """
        from sklearn.base import clone
        from sklearn.linear_model import LogisticRegression

        # Split into treatment and control
        treatment_mask = treatment == 1
        control_mask = treatment == 0

        X_treatment = X[treatment_mask]
        y_treatment = y[treatment_mask]

        X_control = X[control_mask]
        y_control = y[control_mask]

        logger.info(f"Fitting X-Learner:")
        logger.info(f"  Treatment: {len(X_treatment):,}, Control: {len(X_control):,}")

        # Stage 1: Train base models
        self.treatment_model = clone(self.base_model)
        self.treatment_model.fit(X_treatment, y_treatment)

        self.control_model = clone(self.base_model)
        self.control_model.fit(X_control, y_control)

        # Stage 2: Compute pseudo-outcomes
        # For treatment group: D_treatment = Y_treatment - E[Y|X, T=0]
        y_control_imputed_for_treatment = self.control_model.predict(X_treatment)
        D_treatment = y_treatment - y_control_imputed_for_treatment

        # For control group: D_control = E[Y|X, T=1] - Y_control
        y_treatment_imputed_for_control = self.treatment_model.predict(X_control)
        D_control = y_treatment_imputed_for_control - y_control

        # Stage 3: Train models for pseudo-outcomes
        self.tau_treatment_model = clone(self.base_model)
        self.tau_treatment_model.fit(X_treatment, D_treatment)

        self.tau_control_model = clone(self.base_model)
        self.tau_control_model.fit(X_control, D_control)

        # Stage 4: Estimate propensity score
        self.propensity_model = LogisticRegression()
        self.propensity_model.fit(X, treatment)

        logger.info("✓ X-Learner fitted")

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict uplift using propensity-weighted combination.

        Args:
            X: Features

        Returns:
            Uplift scores
        """
        if self.tau_treatment_model is None or self.tau_control_model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Get predictions from both tau models
        tau_treatment = self.tau_treatment_model.predict(X)
        tau_control = self.tau_control_model.predict(X)

        # Get propensity scores
        propensity = self.propensity_model.predict_proba(X)[:, 1]

        # Weighted combination
        uplift = propensity * tau_control + (1 - propensity) * tau_treatment

        return uplift


def compute_uplift_curve(
    y_true: np.ndarray,
    treatment: np.ndarray,
    uplift_scores: np.ndarray,
    n_bins: int = 10
) -> Dict[str, np.ndarray]:
    """
    Compute uplift curve (cumulative gain).

    Uplift curve shows cumulative treatment effect when targeting
    customers by descending uplift score.

    Args:
        y_true: Actual outcomes
        treatment: Treatment indicators
        uplift_scores: Predicted uplift scores
        n_bins: Number of bins for curve

    Returns:
        Dict with uplift curve data
    """
    # Sort by uplift score (descending)
    sorted_indices = np.argsort(-uplift_scores)

    y_sorted = y_true[sorted_indices]
    treatment_sorted = treatment[sorted_indices]

    n = len(y_true)
    bin_size = n // n_bins

    cumulative_gain = []
    percentiles = []

    for i in range(n_bins):
        # Top i% of customers by uplift score
        end_idx = (i + 1) * bin_size

        y_bin = y_sorted[:end_idx]
        treatment_bin = treatment_sorted[:end_idx]

        # Treatment effect in this bin
        treatment_mask = treatment_bin == 1
        control_mask = treatment_bin == 0

        if treatment_mask.sum() > 0 and control_mask.sum() > 0:
            treatment_effect = y_bin[treatment_mask].mean() - y_bin[control_mask].mean()
            cumulative_gain.append(treatment_effect * end_idx)
        else:
            cumulative_gain.append(0)

        percentiles.append((i + 1) * 100 / n_bins)

    return {
        "percentiles": np.array(percentiles),
        "cumulative_gain": np.array(cumulative_gain)
    }


def plot_uplift_curve(
    uplift_curve: Dict[str, np.ndarray],
    title: str = "Uplift Curve"
):
    """
    Plot uplift curve.

    Args:
        uplift_curve: Output from compute_uplift_curve()
        title: Plot title
    """
    plt.figure(figsize=(10, 6))

    plt.plot(
        uplift_curve["percentiles"],
        uplift_curve["cumulative_gain"],
        marker='o',
        label="Uplift Model"
    )

    # Random targeting baseline
    plt.axhline(y=0, color='r', linestyle='--', label="Random Targeting")

    plt.xlabel("% of Population Targeted")
    plt.ylabel("Cumulative Gain")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    return plt


def qini_coefficient(
    y_true: np.ndarray,
    treatment: np.ndarray,
    uplift_scores: np.ndarray
) -> float:
    """
    Compute Qini coefficient (uplift model performance metric).

    Qini coefficient measures area between uplift curve and random curve.
    Higher is better (range: -inf to 1).

    Args:
        y_true: Actual outcomes
        treatment: Treatment indicators
        uplift_scores: Predicted uplift scores

    Returns:
        Qini coefficient
    """
    curve = compute_uplift_curve(y_true, treatment, uplift_scores, n_bins=100)

    # Area under uplift curve
    auc_uplift = np.trapz(curve["cumulative_gain"], curve["percentiles"])

    # Area under random curve (zero)
    auc_random = 0

    # Qini = (AUC_uplift - AUC_random) / AUC_optimal
    # For simplification, normalize by maximum possible area
    qini = auc_uplift / (curve["percentiles"].max() * curve["cumulative_gain"].max())

    return qini


# Example usage
if __name__ == "__main__":
    from sklearn.datasets import make_regression

    # Generate synthetic data with treatment effect
    np.random.seed(42)

    X, y_base = make_regression(n_samples=2000, n_features=10, noise=10)
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])

    # Random treatment assignment
    treatment = np.random.binomial(1, 0.5, size=2000)

    # Add treatment effect (heterogeneous)
    treatment_effect = X["feature_0"] * 5 + X["feature_1"] * 3
    y = y_base + treatment * treatment_effect

    # Split data
    X_train, X_test, y_train, y_test, treatment_train, treatment_test = train_test_split(
        X, y, treatment, test_size=0.3, random_state=42
    )

    # T-Learner
    t_learner = TLearner()
    t_learner.fit(X_train, y_train, treatment_train)
    uplift_t = t_learner.predict_uplift(X_test)

    print("T-Learner:")
    print(f"  Mean uplift: {uplift_t.mean():.2f}")
    print(f"  Std uplift: {uplift_t.std():.2f}")

    # S-Learner
    s_learner = SLearner()
    s_learner.fit(X_train, y_train, treatment_train)
    uplift_s = s_learner.predict_uplift(X_test)

    print("\nS-Learner:")
    print(f"  Mean uplift: {uplift_s.mean():.2f}")
    print(f"  Std uplift: {uplift_s.std():.2f}")

    # X-Learner
    x_learner = XLearner()
    x_learner.fit(X_train, y_train, treatment_train)
    uplift_x = x_learner.predict_uplift(X_test)

    print("\nX-Learner:")
    print(f"  Mean uplift: {uplift_x.mean():.2f}")
    print(f"  Std uplift: {uplift_x.std():.2f}")

    # Qini coefficient
    qini_t = qini_coefficient(y_test, treatment_test, uplift_t)
    print(f"\nQini coefficient (T-Learner): {qini_t:.4f}")

    # Uplift curve
    curve = compute_uplift_curve(y_test, treatment_test, uplift_t)
    print(f"\nUplift curve computed ({len(curve['percentiles'])} bins)")

    print("\nUplift modeling complete!")
