"""
Hyperopt Integration for Hyperparameter Tuning

Advanced hyperparameter optimization using Hyperopt with:
- Bayesian optimization (Tree-structured Parzen Estimator)
- Distributed tuning with Spark Trials
- Early stopping
- Best model selection
- Cross-validation integration

Features:
- Define search spaces for any model
- Distributed parallel search
- MLflow logging integration
- Early stopping for efficiency
- Automatic best hyperparameters selection

Usage:
    search_space = {
        "n_estimators": hp.quniform("n_estimators", 50, 500, 50),
        "max_depth": hp.quniform("max_depth", 3, 15, 1),
        "learning_rate": hp.loguniform("learning_rate", -5, 0)
    }

    tuner = HyperoptTuner(
        model_class=GradientBoostingRegressor,
        search_space=search_space,
        metric="rmse",
        max_evals=100
    )

    best_params = tuner.tune(X_train, y_train)
"""

import logging
from typing import Dict, Any, Callable, Optional, List, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass

from hyperopt import hp, fmin, tpe, Trials, SparkTrials, STATUS_OK, space_eval
from hyperopt.early_stop import no_progress_loss
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import mlflow

logger = logging.getLogger(__name__)


@dataclass
class HyperoptConfig:
    """Configuration for Hyperopt tuning"""
    max_evals: int = 100
    timeout_seconds: Optional[int] = None
    early_stop_rounds: int = 20
    cv_folds: int = 5
    metric: str = "rmse"
    metric_direction: str = "minimize"  # or "maximize"
    parallelism: int = 4


class HyperoptTuner:
    """
    Hyperparameter tuning using Hyperopt with distributed search.

    Supports:
    - Bayesian optimization (TPE algorithm)
    - Distributed search with Spark
    - Cross-validation
    - MLflow logging
    - Early stopping
    """

    def __init__(
        self,
        model_class: Any,
        search_space: Dict[str, Any],
        metric: str = "rmse",
        metric_direction: str = "minimize",
        max_evals: int = 100,
        cv_folds: int = 5,
        early_stop_rounds: int = 20,
        parallelism: int = 4,
        use_spark: bool = False
    ):
        """
        Initialize Hyperopt tuner.

        Args:
            model_class: Model class to tune (e.g., GradientBoostingRegressor)
            search_space: Hyperopt search space
            metric: Optimization metric
            metric_direction: "minimize" or "maximize"
            max_evals: Maximum number of evaluations
            cv_folds: Number of cross-validation folds
            early_stop_rounds: Stop if no improvement for N rounds
            parallelism: Parallel trials (for SparkTrials)
            use_spark: Use SparkTrials for distributed search
        """
        self.model_class = model_class
        self.search_space = search_space
        self.metric = metric
        self.metric_direction = metric_direction
        self.max_evals = max_evals
        self.cv_folds = cv_folds
        self.early_stop_rounds = early_stop_rounds
        self.parallelism = parallelism
        self.use_spark = use_spark

        self.best_params = None
        self.best_score = None
        self.trials = None

        logger.info(f"HyperoptTuner initialized:")
        logger.info(f"  Model: {model_class.__name__}")
        logger.info(f"  Metric: {metric} ({metric_direction})")
        logger.info(f"  Max evaluations: {max_evals}")
        logger.info(f"  CV folds: {cv_folds}")
        logger.info(f"  Distributed: {use_spark}")

    def tune(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        experiment_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run hyperparameter tuning.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Optional validation features (if None, uses CV)
            y_val: Optional validation labels
            experiment_name: MLflow experiment name

        Returns:
            Best hyperparameters
        """
        logger.info("=" * 80)
        logger.info("Starting Hyperparameter Tuning")
        logger.info("=" * 80)
        logger.info(f"Training samples: {len(X_train):,}")
        logger.info(f"Search space size: {len(self.search_space)}")

        # Set MLflow experiment
        if experiment_name:
            mlflow.set_experiment(experiment_name)

        # Define objective function
        def objective(params):
            return self._objective_function(
                params, X_train, y_train, X_val, y_val
            )

        # Initialize Trials
        if self.use_spark:
            self.trials = SparkTrials(parallelism=self.parallelism)
            logger.info(f"Using SparkTrials with parallelism={self.parallelism}")
        else:
            self.trials = Trials()

        # Run optimization
        try:
            best = fmin(
                fn=objective,
                space=self.search_space,
                algo=tpe.suggest,
                max_evals=self.max_evals,
                trials=self.trials,
                early_stop_fn=no_progress_loss(self.early_stop_rounds),
                verbose=True
            )

            # Convert to actual parameter values
            self.best_params = space_eval(self.search_space, best)

            # Get best score
            self.best_score = min(
                trial['result']['loss'] for trial in self.trials.trials
                if trial['result']['status'] == STATUS_OK
            )

            if self.metric_direction == "maximize":
                self.best_score = -self.best_score

            logger.info("=" * 80)
            logger.info("Hyperparameter Tuning Complete")
            logger.info("=" * 80)
            logger.info(f"Best {self.metric}: {self.best_score:.4f}")
            logger.info(f"Best parameters: {self.best_params}")
            logger.info(f"Total trials: {len(self.trials.trials)}")

            return self.best_params

        except Exception as e:
            logger.error(f"Hyperparameter tuning failed: {e}")
            raise

    def _objective_function(
        self,
        params: Dict[str, Any],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame],
        y_val: Optional[pd.Series]
    ) -> Dict[str, Any]:
        """
        Objective function for Hyperopt.

        Args:
            params: Hyperparameters to evaluate
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)

        Returns:
            Loss and status
        """
        # Convert hyperopt params to model params
        model_params = self._convert_params(params)

        with mlflow.start_run(nested=True):
            # Log parameters
            mlflow.log_params(model_params)

            try:
                # Train model
                model = self.model_class(**model_params)

                # Evaluate
                if X_val is not None and y_val is not None:
                    # Use validation set
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_val)
                    score = self._compute_metric(y_val, y_pred)

                else:
                    # Use cross-validation
                    cv_scores = cross_val_score(
                        model, X_train, y_train,
                        cv=self.cv_folds,
                        scoring=self._get_sklearn_scorer()
                    )
                    score = cv_scores.mean()

                # Log metric
                mlflow.log_metric(self.metric, score)

                # Convert to loss (minimization)
                if self.metric_direction == "maximize":
                    loss = -score
                else:
                    loss = score

                return {
                    'loss': loss,
                    'status': STATUS_OK,
                    'score': score
                }

            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                return {
                    'loss': float('inf'),
                    'status': STATUS_OK,
                    'score': float('inf')
                }

    def _convert_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert Hyperopt parameters to model parameters.

        Handles type conversions (e.g., float to int for discrete parameters).
        """
        converted = {}
        for key, value in params.items():
            # Convert to int for discrete parameters
            if key in ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf"]:
                converted[key] = int(value)
            else:
                converted[key] = value

        return converted

    def _compute_metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute metric value"""
        if self.metric == "rmse":
            return np.sqrt(mean_squared_error(y_true, y_pred))
        elif self.metric == "mae":
            return mean_absolute_error(y_true, y_pred)
        elif self.metric == "r2":
            return r2_score(y_true, y_pred)
        elif self.metric == "accuracy":
            return accuracy_score(y_true, y_pred.round())
        elif self.metric == "f1":
            return f1_score(y_true, y_pred.round(), average='weighted')
        elif self.metric == "auc":
            return roc_auc_score(y_true, y_pred)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    def _get_sklearn_scorer(self) -> str:
        """Get sklearn scorer name for cross_val_score"""
        scorer_map = {
            "rmse": "neg_root_mean_squared_error",
            "mae": "neg_mean_absolute_error",
            "r2": "r2",
            "accuracy": "accuracy",
            "f1": "f1_weighted",
            "auc": "roc_auc"
        }

        return scorer_map.get(self.metric, self.metric)

    def get_best_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> Any:
        """
        Train final model with best hyperparameters.

        Args:
            X_train: Training features
            y_train: Training labels

        Returns:
            Trained model with best hyperparameters
        """
        if self.best_params is None:
            raise ValueError("No tuning completed. Run tune() first.")

        model_params = self._convert_params(self.best_params)
        model = self.model_class(**model_params)
        model.fit(X_train, y_train)

        return model

    def plot_optimization_history(self):
        """
        Plot optimization history (loss over trials).

        Returns:
            Plot data for visualization
        """
        if self.trials is None:
            raise ValueError("No trials completed")

        losses = []
        for trial in self.trials.trials:
            if trial['result']['status'] == STATUS_OK:
                losses.append(trial['result']['loss'])

        return {
            "trial_numbers": list(range(1, len(losses) + 1)),
            "losses": losses,
            "best_loss_so_far": np.minimum.accumulate(losses)
        }


def create_search_space_regression(model_type: str) -> Dict[str, Any]:
    """
    Create search space for regression models.

    Args:
        model_type: "gradient_boosting", "random_forest", "xgboost", "lightgbm"

    Returns:
        Hyperopt search space
    """
    if model_type == "gradient_boosting":
        return {
            "n_estimators": hp.quniform("n_estimators", 50, 500, 50),
            "max_depth": hp.quniform("max_depth", 3, 15, 1),
            "learning_rate": hp.loguniform("learning_rate", -5, 0),
            "min_samples_split": hp.quniform("min_samples_split", 2, 20, 2),
            "min_samples_leaf": hp.quniform("min_samples_leaf", 1, 10, 1),
            "subsample": hp.uniform("subsample", 0.5, 1.0)
        }

    elif model_type == "random_forest":
        return {
            "n_estimators": hp.quniform("n_estimators", 50, 500, 50),
            "max_depth": hp.quniform("max_depth", 5, 30, 5),
            "min_samples_split": hp.quniform("min_samples_split", 2, 20, 2),
            "min_samples_leaf": hp.quniform("min_samples_leaf", 1, 10, 1),
            "max_features": hp.choice("max_features", ["sqrt", "log2", None])
        }

    elif model_type == "xgboost":
        return {
            "n_estimators": hp.quniform("n_estimators", 50, 500, 50),
            "max_depth": hp.quniform("max_depth", 3, 15, 1),
            "learning_rate": hp.loguniform("learning_rate", -5, 0),
            "subsample": hp.uniform("subsample", 0.5, 1.0),
            "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1.0),
            "gamma": hp.loguniform("gamma", -5, 2),
            "reg_alpha": hp.loguniform("reg_alpha", -5, 2),
            "reg_lambda": hp.loguniform("reg_lambda", -5, 2)
        }

    elif model_type == "lightgbm":
        return {
            "n_estimators": hp.quniform("n_estimators", 50, 500, 50),
            "max_depth": hp.quniform("max_depth", 3, 15, 1),
            "learning_rate": hp.loguniform("learning_rate", -5, 0),
            "num_leaves": hp.quniform("num_leaves", 10, 200, 10),
            "min_child_samples": hp.quniform("min_child_samples", 5, 50, 5),
            "subsample": hp.uniform("subsample", 0.5, 1.0),
            "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1.0),
            "reg_alpha": hp.loguniform("reg_alpha", -5, 2),
            "reg_lambda": hp.loguniform("reg_lambda", -5, 2)
        }

    else:
        raise ValueError(f"Unknown model type: {model_type}")


# Example usage
if __name__ == "__main__":
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.datasets import make_regression

    # Generate sample data
    X, y = make_regression(n_samples=1000, n_features=20, noise=10, random_state=42)
    X_train = pd.DataFrame(X[:800])
    y_train = pd.Series(y[:800])
    X_val = pd.DataFrame(X[800:])
    y_val = pd.Series(y[800:])

    # Define search space
    search_space = create_search_space_regression("gradient_boosting")

    # Initialize tuner
    tuner = HyperoptTuner(
        model_class=GradientBoostingRegressor,
        search_space=search_space,
        metric="rmse",
        metric_direction="minimize",
        max_evals=50,
        cv_folds=5,
        early_stop_rounds=10
    )

    # Run tuning
    best_params = tuner.tune(X_train, y_train, X_val, y_val)
    print(f"Best parameters: {best_params}")

    # Train final model
    best_model = tuner.get_best_model(X_train, y_train)
    print(f"Best model trained: {best_model}")

    # Plot optimization
    history = tuner.plot_optimization_history()
    print(f"Optimization history: {len(history['losses'])} trials")
