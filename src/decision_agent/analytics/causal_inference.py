"""
Causal Inference Methods

Estimate causal effects using rigorous statistical methods:
1. Propensity Score Matching (PSM)
2. Difference-in-Differences (DiD)
3. Synthetic Control
4. Regression Discontinuity Design (RDD)
5. Instrumental Variables (IV)

Use Cases:
- A/B test analysis with confounders
- Policy impact evaluation
- Marketing campaign effectiveness
- Feature launch impact

Causal Question Examples:
- "Did the new pricing policy increase revenue?"
- "What's the effect of retention emails on churn?"
- "Did the UI redesign improve engagement?"

Usage:
    # Propensity Score Matching
    psm = PropensityScoreMatcher()
    ate = psm.estimate_ate(X, treatment, outcome)

    # Difference-in-Differences
    did = DifferenceInDifferences()
    effect = did.estimate(panel_data, treatment_time, treatment_group, outcome)
"""

import logging
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import NearestNeighbors
from scipy import stats

logger = logging.getLogger(__name__)


class PropensityScoreMatcher:
    """
    Propensity Score Matching for causal inference.

    Matches treated and control units based on propensity scores
    (probability of receiving treatment given covariates).

    Steps:
    1. Estimate propensity scores (prob of treatment given X)
    2. Match treated to control units with similar scores
    3. Compute treatment effect on matched sample
    """

    def __init__(
        self,
        matching_method: str = "nearest",
        caliper: Optional[float] = 0.1,
        n_neighbors: int = 1
    ):
        """
        Initialize PSM.

        Args:
            matching_method: "nearest", "caliper", or "stratification"
            caliper: Maximum allowed difference in propensity scores
            n_neighbors: Number of control matches per treated unit
        """
        self.matching_method = matching_method
        self.caliper = caliper
        self.n_neighbors = n_neighbors
        self.propensity_model = LogisticRegression()

        logger.info(f"PropensityScoreMatcher initialized:")
        logger.info(f"  Method: {matching_method}")
        logger.info(f"  Caliper: {caliper}")

    def fit_propensity_model(
        self,
        X: pd.DataFrame,
        treatment: pd.Series
    ):
        """
        Fit propensity score model.

        Args:
            X: Covariates
            treatment: Treatment indicator
        """
        logger.info("Fitting propensity model...")
        self.propensity_model.fit(X, treatment)

        # Get propensity scores
        propensity_scores = self.propensity_model.predict_proba(X)[:, 1]

        logger.info(f"✓ Propensity model fitted")
        logger.info(f"  Mean propensity: {propensity_scores.mean():.3f}")
        logger.info(f"  Std propensity: {propensity_scores.std():.3f}")

        return propensity_scores

    def match(
        self,
        X: pd.DataFrame,
        treatment: pd.Series,
        propensity_scores: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Match treated and control units.

        Args:
            X: Covariates
            treatment: Treatment indicator
            propensity_scores: Pre-computed propensity scores (optional)

        Returns:
            Tuple of (matched_treated_indices, matched_control_indices)
        """
        if propensity_scores is None:
            propensity_scores = self.fit_propensity_model(X, treatment)

        treatment_mask = treatment == 1
        control_mask = treatment == 0

        treated_indices = np.where(treatment_mask)[0]
        control_indices = np.where(control_mask)[0]

        treated_propensity = propensity_scores[treatment_mask]
        control_propensity = propensity_scores[control_mask]

        logger.info(f"Matching {len(treated_indices)} treated to {len(control_indices)} control...")

        if self.matching_method == "nearest":
            matched_treated, matched_control = self._nearest_neighbor_matching(
                treated_indices, control_indices,
                treated_propensity, control_propensity
            )

        elif self.matching_method == "caliper":
            matched_treated, matched_control = self._caliper_matching(
                treated_indices, control_indices,
                treated_propensity, control_propensity
            )

        else:
            raise ValueError(f"Unknown matching method: {self.matching_method}")

        logger.info(f"✓ Matched {len(matched_treated)} pairs")

        return matched_treated, matched_control

    def _nearest_neighbor_matching(
        self,
        treated_indices: np.ndarray,
        control_indices: np.ndarray,
        treated_propensity: np.ndarray,
        control_propensity: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Nearest neighbor matching"""
        # Fit KNN on control propensity scores
        knn = NearestNeighbors(n_neighbors=self.n_neighbors)
        knn.fit(control_propensity.reshape(-1, 1))

        # Find nearest control for each treated
        distances, indices = knn.kneighbors(treated_propensity.reshape(-1, 1))

        # Apply caliper if specified
        if self.caliper:
            valid_matches = distances[:, 0] <= self.caliper
            matched_treated = treated_indices[valid_matches]
            matched_control = control_indices[indices[valid_matches, 0]]
        else:
            matched_treated = treated_indices
            matched_control = control_indices[indices[:, 0]]

        return matched_treated, matched_control

    def _caliper_matching(
        self,
        treated_indices: np.ndarray,
        control_indices: np.ndarray,
        treated_propensity: np.ndarray,
        control_propensity: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Caliper matching (maximum allowed distance)"""
        matched_treated = []
        matched_control = []

        for i, treated_ps in enumerate(treated_propensity):
            # Find controls within caliper
            distances = np.abs(control_propensity - treated_ps)
            within_caliper = distances <= self.caliper

            if within_caliper.any():
                # Select closest control within caliper
                closest_idx = np.argmin(distances)
                matched_treated.append(treated_indices[i])
                matched_control.append(control_indices[closest_idx])

        return np.array(matched_treated), np.array(matched_control)

    def estimate_ate(
        self,
        X: pd.DataFrame,
        treatment: pd.Series,
        outcome: pd.Series
    ) -> Dict[str, float]:
        """
        Estimate Average Treatment Effect (ATE).

        ATE = E[Y(1) - Y(0)]

        Args:
            X: Covariates
            treatment: Treatment indicator
            outcome: Outcome variable

        Returns:
            Dict with ATE and statistics
        """
        logger.info("Estimating Average Treatment Effect...")

        # Compute propensity scores
        propensity_scores = self.fit_propensity_model(X, treatment)

        # Match
        matched_treated, matched_control = self.match(X, treatment, propensity_scores)

        # Compute treatment effect on matched sample
        outcome_treated = outcome.iloc[matched_treated]
        outcome_control = outcome.iloc[matched_control]

        ate = (outcome_treated - outcome_control).mean()
        ate_std = (outcome_treated - outcome_control).std()

        # T-test
        t_stat, p_value = stats.ttest_rel(outcome_treated, outcome_control)

        result = {
            "ate": ate,
            "ate_std": ate_std,
            "t_statistic": t_stat,
            "p_value": p_value,
            "n_matched_pairs": len(matched_treated),
            "significant": p_value < 0.05
        }

        logger.info(f"✓ ATE estimated:")
        logger.info(f"  ATE: {ate:.3f} (std: {ate_std:.3f})")
        logger.info(f"  p-value: {p_value:.4f}")
        logger.info(f"  Significant: {result['significant']}")

        return result


class DifferenceInDifferences:
    """
    Difference-in-Differences (DiD) estimator.

    Estimates causal effect by comparing change in outcome
    between treatment and control groups before/after intervention.

    DiD = [E[Y_post|Treated] - E[Y_pre|Treated]] -
          [E[Y_post|Control] - E[Y_pre|Control]]

    Assumes parallel trends: without treatment, both groups
    would have followed same trend.
    """

    def estimate(
        self,
        data: pd.DataFrame,
        outcome_col: str,
        treatment_col: str,
        time_col: str,
        treatment_time: Any
    ) -> Dict[str, Any]:
        """
        Estimate DiD treatment effect.

        Args:
            data: Panel data with outcomes over time
            outcome_col: Outcome variable column
            treatment_col: Treatment group indicator (1 = treated, 0 = control)
            time_col: Time period column
            treatment_time: Time when treatment begins

        Returns:
            DiD estimate and statistics
        """
        logger.info("Estimating Difference-in-Differences effect...")

        # Create post-treatment indicator
        data = data.copy()
        data["post"] = (data[time_col] >= treatment_time).astype(int)

        # Create interaction term
        data["treatment_x_post"] = data[treatment_col] * data["post"]

        # DiD regression: Y = β0 + β1*Treatment + β2*Post + β3*Treatment*Post + ε
        # β3 is the DiD estimator
        from statsmodels.formula.api import ols

        formula = f"{outcome_col} ~ {treatment_col} + post + treatment_x_post"
        model = ols(formula, data=data).fit()

        did_effect = model.params["treatment_x_post"]
        did_se = model.bse["treatment_x_post"]
        p_value = model.pvalues["treatment_x_post"]

        result = {
            "did_effect": did_effect,
            "std_error": did_se,
            "t_statistic": model.tvalues["treatment_x_post"],
            "p_value": p_value,
            "significant": p_value < 0.05,
            "r_squared": model.rsquared,
            "full_model": model
        }

        logger.info(f"✓ DiD estimated:")
        logger.info(f"  Effect: {did_effect:.3f} (SE: {did_se:.3f})")
        logger.info(f"  p-value: {p_value:.4f}")
        logger.info(f"  Significant: {result['significant']}")

        return result

    def plot_parallel_trends(
        self,
        data: pd.DataFrame,
        outcome_col: str,
        treatment_col: str,
        time_col: str,
        treatment_time: Any
    ):
        """
        Plot parallel trends assumption.

        Args:
            data: Panel data
            outcome_col: Outcome column
            treatment_col: Treatment group column
            time_col: Time column
            treatment_time: Treatment start time
        """
        import matplotlib.pyplot as plt

        # Aggregate by time and treatment group
        trends = data.groupby([time_col, treatment_col])[outcome_col].mean().reset_index()

        plt.figure(figsize=(10, 6))

        # Plot treatment group
        treatment_data = trends[trends[treatment_col] == 1]
        plt.plot(
            treatment_data[time_col],
            treatment_data[outcome_col],
            marker='o',
            label="Treatment Group"
        )

        # Plot control group
        control_data = trends[trends[treatment_col] == 0]
        plt.plot(
            control_data[time_col],
            control_data[outcome_col],
            marker='s',
            label="Control Group"
        )

        # Mark treatment time
        plt.axvline(x=treatment_time, color='r', linestyle='--', label="Treatment Start")

        plt.xlabel("Time")
        plt.ylabel(outcome_col)
        plt.title("Parallel Trends Check")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        return plt


class SyntheticControl:
    """
    Synthetic Control Method.

    Creates a "synthetic" control unit as a weighted combination
    of other units that closely matches the treated unit's
    pre-treatment trajectory.

    Use case: Evaluate impact when only one unit is treated
    (e.g., policy change in one state/country).
    """

    def __init__(self):
        """Initialize Synthetic Control"""
        self.weights = None
        logger.info("SyntheticControl initialized")

    def fit(
        self,
        pre_treatment_outcomes: pd.DataFrame,
        treated_unit_id: str
    ) -> np.ndarray:
        """
        Fit synthetic control weights.

        Args:
            pre_treatment_outcomes: DataFrame with columns = units, index = time
            treated_unit_id: ID of treated unit (column name)

        Returns:
            Weights for donor pool units
        """
        from scipy.optimize import minimize

        # Extract treated unit outcomes
        y_treated = pre_treatment_outcomes[treated_unit_id].values

        # Extract donor pool (all other units)
        donor_units = [col for col in pre_treatment_outcomes.columns if col != treated_unit_id]
        Y_donors = pre_treatment_outcomes[donor_units].values

        # Optimization: minimize ||Y_treated - Y_donors * w||^2
        # subject to: w >= 0, sum(w) = 1
        def objective(w):
            return np.sum((y_treated - Y_donors @ w) ** 2)

        # Constraints
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1}  # weights sum to 1
        ]
        bounds = [(0, 1) for _ in donor_units]  # weights >= 0

        # Initial guess
        w0 = np.ones(len(donor_units)) / len(donor_units)

        # Optimize
        result = minimize(objective, w0, bounds=bounds, constraints=constraints)

        self.weights = result.x
        self.donor_units = donor_units

        logger.info(f"✓ Synthetic control fitted")
        logger.info(f"  Donor units: {len(donor_units)}")
        logger.info(f"  RMSE: {np.sqrt(objective(self.weights)):.3f}")

        return self.weights

    def predict(
        self,
        outcomes: pd.DataFrame
    ) -> pd.Series:
        """
        Predict synthetic control outcomes.

        Args:
            outcomes: DataFrame with donor unit outcomes

        Returns:
            Synthetic control predictions
        """
        if self.weights is None:
            raise ValueError("Weights not fitted. Call fit() first.")

        Y_donors = outcomes[self.donor_units].values
        synthetic = Y_donors @ self.weights

        return pd.Series(synthetic, index=outcomes.index)

    def estimate_effect(
        self,
        pre_treatment_outcomes: pd.DataFrame,
        post_treatment_outcomes: pd.DataFrame,
        treated_unit_id: str
    ) -> Dict[str, Any]:
        """
        Estimate treatment effect using synthetic control.

        Args:
            pre_treatment_outcomes: Pre-treatment data
            post_treatment_outcomes: Post-treatment data
            treated_unit_id: Treated unit ID

        Returns:
            Treatment effect estimates
        """
        logger.info("Estimating Synthetic Control effect...")

        # Fit on pre-treatment period
        self.fit(pre_treatment_outcomes, treated_unit_id)

        # Predict synthetic control for post-treatment
        synthetic_post = self.predict(post_treatment_outcomes)

        # Actual treated outcomes
        actual_post = post_treatment_outcomes[treated_unit_id]

        # Treatment effect = Actual - Synthetic
        treatment_effects = actual_post - synthetic_post

        result = {
            "mean_effect": treatment_effects.mean(),
            "total_effect": treatment_effects.sum(),
            "effects_by_period": treatment_effects,
            "weights": dict(zip(self.donor_units, self.weights))
        }

        logger.info(f"✓ Effect estimated:")
        logger.info(f"  Mean effect: {result['mean_effect']:.3f}")
        logger.info(f"  Total effect: {result['total_effect']:.3f}")

        return result


# Example usage
if __name__ == "__main__":
    from sklearn.datasets import make_regression

    # Example 1: Propensity Score Matching
    print("=" * 80)
    print("Example 1: Propensity Score Matching")
    print("=" * 80)

    np.random.seed(42)

    # Generate data
    X, _ = make_regression(n_samples=1000, n_features=5, noise=10)
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(5)])

    # Treatment depends on covariates (confounded)
    treatment_propensity = 1 / (1 + np.exp(-(X["feature_0"] + 0.5 * X["feature_1"])))
    treatment = (np.random.random(1000) < treatment_propensity).astype(int)

    # Outcome depends on treatment + covariates
    outcome = 50 + 10 * treatment + 5 * X["feature_0"] + 3 * X["feature_1"] + np.random.normal(0, 5, 1000)

    # PSM
    psm = PropensityScoreMatcher(matching_method="nearest", caliper=0.1)
    ate_result = psm.estimate_ate(X, treatment, pd.Series(outcome))

    print(f"\nTrue ATE: ~10")
    print(f"Estimated ATE: {ate_result['ate']:.2f}")
    print(f"p-value: {ate_result['p_value']:.4f}")

    # Example 2: Difference-in-Differences
    print("\n" + "=" * 80)
    print("Example 2: Difference-in-Differences")
    print("=" * 80)

    # Generate panel data
    times = np.arange(0, 10)
    panel_data = []

    for group in [0, 1]:  # 0 = control, 1 = treatment
        for time in times:
            # Parallel trends before treatment
            base_outcome = 50 + 2 * time

            # Treatment effect after time=5 for treatment group
            if group == 1 and time >= 5:
                treatment_effect = 15
            else:
                treatment_effect = 0

            outcome = base_outcome + treatment_effect + np.random.normal(0, 2)

            panel_data.append({
                "time": time,
                "group": group,
                "outcome": outcome
            })

    panel_df = pd.DataFrame(panel_data)

    # DiD
    did = DifferenceInDifferences()
    did_result = did.estimate(
        panel_df,
        outcome_col="outcome",
        treatment_col="group",
        time_col="time",
        treatment_time=5
    )

    print(f"\nTrue DiD effect: ~15")
    print(f"Estimated DiD effect: {did_result['did_effect']:.2f}")
    print(f"p-value: {did_result['p_value']:.4f}")

    print("\nCausal inference examples complete!")
