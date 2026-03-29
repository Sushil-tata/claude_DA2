"""
Databricks AutoML Wrapper

Provides a simplified interface to Databricks AutoML for automated:
- Feature engineering
- Model selection
- Hyperparameter tuning
- Experiment tracking

Features:
- One-line AutoML training
- Automatic best model selection
- Integration with MLflow
- Feature importance analysis
- Model comparison with manual models

Usage:
    automl = DatabricksAutoMLWrapper(
        target_col="income",
        problem_type="regression"
    )

    best_model = automl.train(
        training_df,
        timeout_minutes=30,
        max_trials=10
    )
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime

from pyspark.sql import DataFrame
import mlflow
from databricks import automl

logger = logging.getLogger(__name__)


@dataclass
class AutoMLConfig:
    """Configuration for AutoML training"""
    target_col: str
    problem_type: str  # "regression", "classification", "forecasting"
    timeout_minutes: int = 30
    max_trials: int = 10
    metric: Optional[str] = None  # Auto-selected if None
    exclude_frameworks: Optional[List[str]] = None
    exclude_cols: Optional[List[str]] = None
    pos_label: Optional[str] = None  # For binary classification
    time_col: Optional[str] = None  # For forecasting


class DatabricksAutoMLWrapper:
    """
    Wrapper for Databricks AutoML with simplified interface.

    Provides automated model training, selection, and comparison.
    """

    def __init__(
        self,
        target_col: str,
        problem_type: str,
        experiment_path: Optional[str] = None
    ):
        """
        Initialize AutoML wrapper.

        Args:
            target_col: Name of target column
            problem_type: "regression", "classification", or "forecasting"
            experiment_path: Optional MLflow experiment path
        """
        self.target_col = target_col
        self.problem_type = problem_type
        self.experiment_path = experiment_path or f"/Users/automl/{problem_type}_{target_col}"

        self.summary = None
        self.best_trial = None

        logger.info(f"AutoML initialized:")
        logger.info(f"  Target: {target_col}")
        logger.info(f"  Problem type: {problem_type}")
        logger.info(f"  Experiment: {self.experiment_path}")

    def train(
        self,
        dataset: DataFrame,
        timeout_minutes: int = 30,
        max_trials: int = 10,
        metric: Optional[str] = None,
        exclude_frameworks: Optional[List[str]] = None,
        exclude_cols: Optional[List[str]] = None,
        **kwargs
    ) -> Any:
        """
        Train models using Databricks AutoML.

        Args:
            dataset: Training dataset (Spark DataFrame)
            timeout_minutes: Maximum training time
            max_trials: Maximum number of trials
            metric: Optimization metric (auto-selected if None)
            exclude_frameworks: Frameworks to exclude (e.g., ["xgboost"])
            exclude_cols: Columns to exclude from features
            **kwargs: Additional AutoML parameters

        Returns:
            Best model from AutoML trials
        """
        logger.info("=" * 80)
        logger.info("Starting Databricks AutoML")
        logger.info("=" * 80)
        logger.info(f"Dataset size: {dataset.count():,} rows")
        logger.info(f"Features: {len(dataset.columns) - 1}")
        logger.info(f"Timeout: {timeout_minutes} minutes")
        logger.info(f"Max trials: {max_trials}")

        try:
            # Run AutoML based on problem type
            if self.problem_type == "regression":
                self.summary = automl.regress(
                    dataset=dataset,
                    target_col=self.target_col,
                    timeout_minutes=timeout_minutes,
                    max_trials=max_trials,
                    primary_metric=metric or "r2",
                    exclude_frameworks=exclude_frameworks,
                    exclude_cols=exclude_cols,
                    experiment_dir=self.experiment_path,
                    **kwargs
                )

            elif self.problem_type == "classification":
                self.summary = automl.classify(
                    dataset=dataset,
                    target_col=self.target_col,
                    timeout_minutes=timeout_minutes,
                    max_trials=max_trials,
                    primary_metric=metric or "f1",
                    exclude_frameworks=exclude_frameworks,
                    exclude_cols=exclude_cols,
                    experiment_dir=self.experiment_path,
                    **kwargs
                )

            elif self.problem_type == "forecasting":
                if "time_col" not in kwargs:
                    raise ValueError("time_col required for forecasting")

                self.summary = automl.forecast(
                    dataset=dataset,
                    target_col=self.target_col,
                    time_col=kwargs["time_col"],
                    timeout_minutes=timeout_minutes,
                    max_trials=max_trials,
                    primary_metric=metric or "smape",
                    exclude_frameworks=exclude_frameworks,
                    exclude_cols=exclude_cols,
                    experiment_dir=self.experiment_path,
                    **kwargs
                )

            else:
                raise ValueError(f"Unknown problem type: {self.problem_type}")

            # Extract best trial
            self.best_trial = self.summary.best_trial

            logger.info("=" * 80)
            logger.info("AutoML Training Complete")
            logger.info("=" * 80)
            logger.info(f"Total trials: {len(self.summary.trials)}")
            logger.info(f"Best trial ID: {self.best_trial.mlflow_run_id}")
            logger.info(f"Best metric: {self.best_trial.metrics[self.summary.primary_metric]:.4f}")
            logger.info(f"Model URI: {self.best_trial.model_path}")

            return self.best_trial

        except Exception as e:
            logger.error(f"AutoML training failed: {e}")
            raise

    def get_best_model(self) -> Any:
        """
        Get the best model from AutoML trials.

        Returns:
            Best model (MLflow pyfunc model)
        """
        if not self.best_trial:
            raise ValueError("No AutoML trials completed. Run train() first.")

        return mlflow.pyfunc.load_model(self.best_trial.model_path)

    def get_feature_importance(self, top_n: int = 20) -> Dict[str, float]:
        """
        Get feature importance from best model.

        Args:
            top_n: Number of top features to return

        Returns:
            Dict of {feature_name: importance_score}
        """
        if not self.best_trial:
            raise ValueError("No AutoML trials completed. Run train() first.")

        # Load model and get feature importance
        model = self.get_best_model()

        # Try to get feature importance (sklearn models)
        if hasattr(model, 'feature_importances_'):
            feature_names = model.feature_names_in_
            importances = model.feature_importances_

            feature_importance = dict(zip(feature_names, importances))

            # Sort and take top N
            sorted_features = sorted(
                feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:top_n]

            return dict(sorted_features)

        logger.warning("Model does not support feature importance")
        return {}

    def compare_with_baseline(
        self,
        baseline_run_id: str,
        baseline_metric_name: str
    ) -> Dict[str, Any]:
        """
        Compare AutoML best model with a baseline model.

        Args:
            baseline_run_id: MLflow run ID of baseline model
            baseline_metric_name: Metric name to compare

        Returns:
            Comparison results
        """
        if not self.best_trial:
            raise ValueError("No AutoML trials completed. Run train() first.")

        # Get AutoML best metric
        automl_metric = self.best_trial.metrics[self.summary.primary_metric]

        # Get baseline metric
        baseline_run = mlflow.get_run(baseline_run_id)
        baseline_metric = baseline_run.data.metrics.get(baseline_metric_name)

        if baseline_metric is None:
            raise ValueError(f"Metric {baseline_metric_name} not found in baseline run")

        # Compute improvement
        improvement_pct = ((automl_metric - baseline_metric) / abs(baseline_metric)) * 100

        comparison = {
            "automl_metric": automl_metric,
            "baseline_metric": baseline_metric,
            "improvement_pct": improvement_pct,
            "automl_better": automl_metric > baseline_metric,  # Assumes higher is better
            "automl_run_id": self.best_trial.mlflow_run_id,
            "baseline_run_id": baseline_run_id
        }

        logger.info("=" * 80)
        logger.info("AutoML vs Baseline Comparison")
        logger.info("=" * 80)
        logger.info(f"AutoML {self.summary.primary_metric}: {automl_metric:.4f}")
        logger.info(f"Baseline {baseline_metric_name}: {baseline_metric:.4f}")
        logger.info(f"Improvement: {improvement_pct:+.2f}%")
        logger.info(f"Winner: {'AutoML' if comparison['automl_better'] else 'Baseline'}")

        return comparison

    def get_trial_summary(self) -> List[Dict[str, Any]]:
        """
        Get summary of all AutoML trials.

        Returns:
            List of trial summaries
        """
        if not self.summary:
            raise ValueError("No AutoML trials completed. Run train() first.")

        trials = []
        for trial in self.summary.trials:
            trial_info = {
                "trial_id": trial.mlflow_run_id,
                "model_description": trial.model_description,
                "metrics": trial.metrics,
                "model_path": trial.model_path,
                "duration_seconds": trial.duration
            }
            trials.append(trial_info)

        return trials

    def register_best_model(
        self,
        model_name: str,
        stage: str = "Staging"
    ) -> str:
        """
        Register best AutoML model to MLflow Model Registry.

        Args:
            model_name: Name for registered model
            stage: Target stage (None, Staging, Production)

        Returns:
            Model version
        """
        if not self.best_trial:
            raise ValueError("No AutoML trials completed. Run train() first.")

        # Register model
        model_uri = f"runs:/{self.best_trial.mlflow_run_id}/model"

        model_details = mlflow.register_model(
            model_uri=model_uri,
            name=model_name
        )

        # Transition to stage if specified
        if stage:
            from mlflow.tracking import MlflowClient
            client = MlflowClient()

            client.transition_model_version_stage(
                name=model_name,
                version=model_details.version,
                stage=stage
            )

        logger.info(f"Model registered: {model_name} v{model_details.version} ({stage})")

        return model_details.version

    def generate_notebook(self, output_path: str):
        """
        Generate notebook with best trial code.

        Args:
            output_path: Path to save generated notebook
        """
        if not self.summary:
            raise ValueError("No AutoML trials completed. Run train() first.")

        # AutoML automatically generates notebooks
        # This method provides access to the generated notebook path
        notebook_path = self.summary.best_trial.notebook_path

        logger.info(f"Best trial notebook available at: {notebook_path}")
        logger.info(f"To view: Open in Databricks workspace")

        return notebook_path


def compare_automl_with_manual(
    automl_wrapper: DatabricksAutoMLWrapper,
    manual_model_run_ids: List[str],
    metric_name: str
) -> Dict[str, Any]:
    """
    Compare AutoML with multiple manual models.

    Args:
        automl_wrapper: Trained AutoML wrapper
        manual_model_run_ids: List of manual model MLflow run IDs
        metric_name: Metric to compare

    Returns:
        Comparison results
    """
    if not automl_wrapper.best_trial:
        raise ValueError("AutoML not trained")

    automl_metric = automl_wrapper.best_trial.metrics[automl_wrapper.summary.primary_metric]

    comparisons = []
    for run_id in manual_model_run_ids:
        run = mlflow.get_run(run_id)
        manual_metric = run.data.metrics.get(metric_name)

        if manual_metric is None:
            logger.warning(f"Metric {metric_name} not found in run {run_id}")
            continue

        improvement_pct = ((automl_metric - manual_metric) / abs(manual_metric)) * 100

        comparisons.append({
            "manual_run_id": run_id,
            "manual_run_name": run.data.tags.get("mlflow.runName", "Unknown"),
            "manual_metric": manual_metric,
            "automl_metric": automl_metric,
            "improvement_pct": improvement_pct,
            "automl_better": automl_metric > manual_metric
        })

    # Find best overall
    all_metrics = [automl_metric] + [c["manual_metric"] for c in comparisons]
    best_metric = max(all_metrics)

    result = {
        "automl_metric": automl_metric,
        "comparisons": comparisons,
        "best_metric": best_metric,
        "automl_is_best": automl_metric == best_metric,
        "num_better_than_automl": sum(1 for c in comparisons if not c["automl_better"])
    }

    logger.info("=" * 80)
    logger.info("AutoML vs Manual Models Comparison")
    logger.info("=" * 80)
    logger.info(f"AutoML metric: {automl_metric:.4f}")
    logger.info(f"Best manual metric: {max(c['manual_metric'] for c in comparisons):.4f}")
    logger.info(f"AutoML is best: {result['automl_is_best']}")

    return result


# Example usage
if __name__ == "__main__":
    from pyspark.sql import SparkSession

    # Initialize Spark
    spark = SparkSession.builder.appName("AutoMLExample").getOrCreate()

    # Load data
    training_df = spark.table("decision_agent.training_data")

    # Initialize AutoML
    automl = DatabricksAutoMLWrapper(
        target_col="income",
        problem_type="regression"
    )

    # Train with AutoML
    best_trial = automl.train(
        dataset=training_df,
        timeout_minutes=30,
        max_trials=10
    )

    # Get feature importance
    feature_importance = automl.get_feature_importance(top_n=10)
    print(f"Top features: {feature_importance}")

    # Compare with baseline
    comparison = automl.compare_with_baseline(
        baseline_run_id="abc123",
        baseline_metric_name="r2"
    )
    print(f"Improvement: {comparison['improvement_pct']:.2f}%")

    # Register best model
    version = automl.register_best_model(
        model_name="income_estimation_automl",
        stage="Staging"
    )
    print(f"Model registered: v{version}")
