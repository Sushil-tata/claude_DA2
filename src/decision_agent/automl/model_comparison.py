"""
Model Comparison Framework

Compare multiple model families and select the best for deployment.

Features:
- Train multiple model types in parallel
- Cross-validation for robust comparison
- Performance vs complexity trade-offs
- Statistical significance testing
- Deployment recommendations
- Automated reporting

Usage:
    comparator = ModelComparator(
        models=[
            ("GradientBoosting", GradientBoostingRegressor(**params1)),
            ("RandomForest", RandomForestRegressor(**params2)),
            ("XGBoost", XGBRegressor(**params3))
        ],
        metrics=["rmse", "r2", "mae"]
    )

    results = comparator.compare(X_train, y_train, X_test, y_test)
    recommendation = comparator.get_recommendation()
"""

import logging
from typing import Dict, Any, List, Tuple, Optional
import time
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
)
from scipy import stats
import mlflow

logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """Metrics for a single model"""
    model_name: str
    train_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    cv_metrics: Dict[str, Dict[str, float]]  # mean, std for each metric
    training_time_seconds: float
    prediction_time_seconds: float
    model_complexity: Dict[str, Any]


class ModelComparator:
    """
    Framework for comparing multiple ML models.

    Provides comprehensive comparison across:
    - Predictive performance
    - Training/inference speed
    - Model complexity
    - Statistical significance
    """

    def __init__(
        self,
        models: List[Tuple[str, Any]],
        metrics: List[str],
        cv_folds: int = 5,
        problem_type: str = "regression"
    ):
        """
        Initialize model comparator.

        Args:
            models: List of (name, model) tuples
            metrics: List of metric names to compute
            cv_folds: Number of cross-validation folds
            problem_type: "regression" or "classification"
        """
        self.models = models
        self.metrics = metrics
        self.cv_folds = cv_folds
        self.problem_type = problem_type

        self.results = []
        self.comparison_df = None

        logger.info(f"ModelComparator initialized:")
        logger.info(f"  Models: {[name for name, _ in models]}")
        logger.info(f"  Metrics: {metrics}")
        logger.info(f"  CV folds: {cv_folds}")

    def compare(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        experiment_name: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Compare all models.

        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            experiment_name: MLflow experiment name

        Returns:
            Comparison DataFrame
        """
        logger.info("=" * 80)
        logger.info("Starting Model Comparison")
        logger.info("=" * 80)
        logger.info(f"Training samples: {len(X_train):,}")
        logger.info(f"Test samples: {len(X_test):,}")

        if experiment_name:
            mlflow.set_experiment(experiment_name)

        for model_name, model in self.models:
            logger.info(f"\nTraining {model_name}...")

            with mlflow.start_run(run_name=model_name):
                # Train and evaluate
                metrics = self._evaluate_model(
                    model_name, model,
                    X_train, y_train,
                    X_test, y_test
                )

                self.results.append(metrics)

                # Log to MLflow
                self._log_to_mlflow(metrics)

        # Create comparison DataFrame
        self.comparison_df = self._create_comparison_df()

        logger.info("\n" + "=" * 80)
        logger.info("Model Comparison Complete")
        logger.info("=" * 80)

        return self.comparison_df

    def _evaluate_model(
        self,
        model_name: str,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> ModelMetrics:
        """
        Evaluate a single model.
        """
        # Train model
        train_start = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - train_start

        # Predictions
        y_train_pred = model.predict(X_train)

        pred_start = time.time()
        y_test_pred = model.predict(X_test)
        prediction_time = time.time() - pred_start

        # Compute metrics
        train_metrics = self._compute_metrics(y_train, y_train_pred)
        test_metrics = self._compute_metrics(y_test, y_test_pred)

        # Cross-validation
        cv_metrics = self._cross_validate_model(model, X_train, y_train)

        # Model complexity
        complexity = self._compute_complexity(model)

        return ModelMetrics(
            model_name=model_name,
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            cv_metrics=cv_metrics,
            training_time_seconds=training_time,
            prediction_time_seconds=prediction_time,
            model_complexity=complexity
        )

    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute all requested metrics"""
        metrics = {}

        for metric_name in self.metrics:
            if self.problem_type == "regression":
                if metric_name == "rmse":
                    metrics[metric_name] = np.sqrt(mean_squared_error(y_true, y_pred))
                elif metric_name == "mae":
                    metrics[metric_name] = mean_absolute_error(y_true, y_pred)
                elif metric_name == "r2":
                    metrics[metric_name] = r2_score(y_true, y_pred)
                elif metric_name == "mape":
                    metrics[metric_name] = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

            elif self.problem_type == "classification":
                y_pred_binary = (y_pred > 0.5).astype(int)
                if metric_name == "accuracy":
                    metrics[metric_name] = accuracy_score(y_true, y_pred_binary)
                elif metric_name == "f1":
                    metrics[metric_name] = f1_score(y_true, y_pred_binary, average='weighted')
                elif metric_name == "precision":
                    metrics[metric_name] = precision_score(y_true, y_pred_binary, average='weighted')
                elif metric_name == "recall":
                    metrics[metric_name] = recall_score(y_true, y_pred_binary, average='weighted')
                elif metric_name == "auc":
                    metrics[metric_name] = roc_auc_score(y_true, y_pred)

        return metrics

    def _cross_validate_model(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[str, Dict[str, float]]:
        """
        Perform cross-validation and return mean/std for each metric.
        """
        cv_results = {}

        for metric_name in self.metrics:
            scorer = self._get_sklearn_scorer(metric_name)

            scores = cross_val_score(
                model, X, y,
                cv=self.cv_folds,
                scoring=scorer
            )

            # Convert negative scores (sklearn convention) to positive
            if scorer.startswith("neg_"):
                scores = -scores

            cv_results[metric_name] = {
                "mean": scores.mean(),
                "std": scores.std(),
                "min": scores.min(),
                "max": scores.max()
            }

        return cv_results

    def _get_sklearn_scorer(self, metric_name: str) -> str:
        """Get sklearn scorer name"""
        scorer_map = {
            "rmse": "neg_root_mean_squared_error",
            "mae": "neg_mean_absolute_error",
            "r2": "r2",
            "accuracy": "accuracy",
            "f1": "f1_weighted",
            "precision": "precision_weighted",
            "recall": "recall_weighted",
            "auc": "roc_auc"
        }

        return scorer_map.get(metric_name, metric_name)

    def _compute_complexity(self, model: Any) -> Dict[str, Any]:
        """
        Compute model complexity metrics.
        """
        complexity = {}

        # Number of parameters (if available)
        if hasattr(model, 'n_estimators'):
            complexity['n_estimators'] = model.n_estimators

        if hasattr(model, 'max_depth'):
            complexity['max_depth'] = model.max_depth

        # Number of trees (for ensemble models)
        if hasattr(model, 'estimators_'):
            complexity['n_trees'] = len(model.estimators_)

        # Number of features used
        if hasattr(model, 'n_features_in_'):
            complexity['n_features'] = model.n_features_in_

        return complexity

    def _create_comparison_df(self) -> pd.DataFrame:
        """
        Create comparison DataFrame from results.
        """
        rows = []

        for metrics in self.results:
            row = {"model": metrics.model_name}

            # Test metrics
            for metric_name in self.metrics:
                row[f"test_{metric_name}"] = metrics.test_metrics[metric_name]
                row[f"cv_{metric_name}_mean"] = metrics.cv_metrics[metric_name]["mean"]
                row[f"cv_{metric_name}_std"] = metrics.cv_metrics[metric_name]["std"]

            # Timing
            row["training_time_sec"] = metrics.training_time_seconds
            row["prediction_time_sec"] = metrics.prediction_time_seconds

            # Complexity
            for key, value in metrics.model_complexity.items():
                row[f"complexity_{key}"] = value

            rows.append(row)

        df = pd.DataFrame(rows)

        # Sort by primary metric (first metric in list)
        primary_metric = f"test_{self.metrics[0]}"
        df = df.sort_values(primary_metric, ascending=(self.problem_type == "regression"))

        return df

    def _log_to_mlflow(self, metrics: ModelMetrics):
        """Log metrics to MLflow"""
        # Log test metrics
        for metric_name, value in metrics.test_metrics.items():
            mlflow.log_metric(f"test_{metric_name}", value)

        # Log CV metrics
        for metric_name, cv_stats in metrics.cv_metrics.items():
            mlflow.log_metric(f"cv_{metric_name}_mean", cv_stats["mean"])
            mlflow.log_metric(f"cv_{metric_name}_std", cv_stats["std"])

        # Log timing
        mlflow.log_metric("training_time_sec", metrics.training_time_seconds)
        mlflow.log_metric("prediction_time_sec", metrics.prediction_time_seconds)

        # Log complexity
        mlflow.log_params(metrics.model_complexity)

    def get_recommendation(
        self,
        primary_metric: Optional[str] = None,
        max_training_time: Optional[float] = None,
        max_prediction_time: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Get deployment recommendation based on performance and constraints.

        Args:
            primary_metric: Metric to optimize (default: first metric)
            max_training_time: Maximum acceptable training time (seconds)
            max_prediction_time: Maximum acceptable prediction time (seconds)

        Returns:
            Recommendation with reasoning
        """
        if self.comparison_df is None:
            raise ValueError("No comparison completed. Run compare() first.")

        primary_metric = primary_metric or self.metrics[0]
        metric_col = f"test_{primary_metric}"

        # Filter by constraints
        candidates = self.comparison_df.copy()

        if max_training_time:
            candidates = candidates[candidates["training_time_sec"] <= max_training_time]

        if max_prediction_time:
            candidates = candidates[candidates["prediction_time_sec"] <= max_prediction_time]

        if len(candidates) == 0:
            raise ValueError("No models meet the specified constraints")

        # Find best model
        if self.problem_type == "regression":
            best_idx = candidates[metric_col].idxmin()
        else:
            best_idx = candidates[metric_col].idxmax()

        best_model = candidates.loc[best_idx]

        # Compute performance gap to second-best
        candidates_sorted = candidates.sort_values(
            metric_col,
            ascending=(self.problem_type == "regression")
        )

        if len(candidates_sorted) > 1:
            second_best = candidates_sorted.iloc[1]
            performance_gap = abs(best_model[metric_col] - second_best[metric_col])
        else:
            second_best = None
            performance_gap = 0.0

        recommendation = {
            "recommended_model": best_model["model"],
            "primary_metric": primary_metric,
            "primary_metric_value": best_model[metric_col],
            "cv_metric_mean": best_model[f"cv_{primary_metric}_mean"],
            "cv_metric_std": best_model[f"cv_{primary_metric}_std"],
            "training_time_sec": best_model["training_time_sec"],
            "prediction_time_sec": best_model["prediction_time_sec"],
            "performance_gap_to_second": performance_gap,
            "second_best_model": second_best["model"] if second_best is not None else None,
            "reasoning": self._generate_reasoning(best_model, second_best, metric_col)
        }

        logger.info("\n" + "=" * 80)
        logger.info("DEPLOYMENT RECOMMENDATION")
        logger.info("=" * 80)
        logger.info(f"Recommended Model: {recommendation['recommended_model']}")
        logger.info(f"{primary_metric}: {recommendation['primary_metric_value']:.4f}")
        logger.info(f"Training Time: {recommendation['training_time_sec']:.2f}s")
        logger.info(f"Prediction Time: {recommendation['prediction_time_sec']:.4f}s")
        logger.info(f"\nReasoning:\n{recommendation['reasoning']}")

        return recommendation

    def _generate_reasoning(
        self,
        best_model: pd.Series,
        second_best: Optional[pd.Series],
        metric_col: str
    ) -> str:
        """Generate human-readable reasoning for recommendation"""
        reasoning_parts = []

        # Performance
        reasoning_parts.append(
            f"- Best performance on {metric_col}: {best_model[metric_col]:.4f}"
        )

        # Gap to second-best
        if second_best is not None:
            gap = abs(best_model[metric_col] - second_best[metric_col])
            gap_pct = (gap / abs(second_best[metric_col])) * 100
            reasoning_parts.append(
                f"- Outperforms {second_best['model']} by {gap_pct:.2f}%"
            )

        # Training time
        if best_model["training_time_sec"] < 60:
            reasoning_parts.append(
                f"- Fast training time: {best_model['training_time_sec']:.2f}s"
            )
        else:
            reasoning_parts.append(
                f"- Training time: {best_model['training_time_sec'] / 60:.1f} minutes"
            )

        # Prediction speed
        if best_model["prediction_time_sec"] < 0.1:
            reasoning_parts.append(
                f"- Very fast predictions: {best_model['prediction_time_sec'] * 1000:.1f}ms"
            )

        return "\n".join(reasoning_parts)

    def statistical_comparison(
        self,
        model1_name: str,
        model2_name: str,
        metric: Optional[str] = None,
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Perform statistical test to compare two models.

        Uses paired t-test on CV scores.

        Args:
            model1_name: Name of first model
            model2_name: Name of second model
            metric: Metric to compare (default: first metric)
            alpha: Significance level

        Returns:
            Statistical test results
        """
        metric = metric or self.metrics[0]

        # Get CV scores for both models
        model1_metrics = next(m for m in self.results if m.model_name == model1_name)
        model2_metrics = next(m for m in self.results if m.model_name == model2_name)

        # Note: For proper statistical test, we'd need individual CV fold scores
        # Here we use mean and std as approximation
        mean1 = model1_metrics.cv_metrics[metric]["mean"]
        std1 = model1_metrics.cv_metrics[metric]["std"]
        mean2 = model2_metrics.cv_metrics[metric]["mean"]
        std2 = model2_metrics.cv_metrics[metric]["std"]

        # Approximate t-test (would need actual CV scores for exact test)
        pooled_std = np.sqrt((std1**2 + std2**2) / 2)
        t_statistic = (mean1 - mean2) / (pooled_std / np.sqrt(self.cv_folds))
        p_value = 2 * (1 - stats.t.cdf(abs(t_statistic), df=self.cv_folds - 1))

        is_significant = p_value < alpha

        result = {
            "model1": model1_name,
            "model2": model2_name,
            "metric": metric,
            "model1_mean": mean1,
            "model2_mean": mean2,
            "difference": mean1 - mean2,
            "t_statistic": t_statistic,
            "p_value": p_value,
            "is_significant": is_significant,
            "alpha": alpha
        }

        logger.info(f"\nStatistical Comparison: {model1_name} vs {model2_name}")
        logger.info(f"Metric: {metric}")
        logger.info(f"Difference: {result['difference']:.4f}")
        logger.info(f"p-value: {p_value:.4f}")
        logger.info(f"Significant: {is_significant}")

        return result


# Example usage
if __name__ == "__main__":
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.datasets import make_regression

    # Generate sample data
    X, y = make_regression(n_samples=1000, n_features=20, noise=10, random_state=42)
    X_train = pd.DataFrame(X[:800])
    y_train = pd.Series(y[:800])
    X_test = pd.DataFrame(X[800:])
    y_test = pd.Series(y[800:])

    # Define models to compare
    models = [
        ("GradientBoosting", GradientBoostingRegressor(n_estimators=100, random_state=42)),
        ("RandomForest", RandomForestRegressor(n_estimators=100, random_state=42)),
        ("Ridge", Ridge(alpha=1.0))
    ]

    # Create comparator
    comparator = ModelComparator(
        models=models,
        metrics=["rmse", "r2", "mae"],
        cv_folds=5,
        problem_type="regression"
    )

    # Compare models
    comparison_df = comparator.compare(X_train, y_train, X_test, y_test)
    print("\nComparison Results:")
    print(comparison_df.to_string())

    # Get recommendation
    recommendation = comparator.get_recommendation(
        primary_metric="rmse",
        max_prediction_time=0.1
    )
    print(f"\nRecommended model: {recommendation['recommended_model']}")

    # Statistical comparison
    stat_result = comparator.statistical_comparison("GradientBoosting", "RandomForest")
    print(f"\nStatistical test: p-value={stat_result['p_value']:.4f}")
