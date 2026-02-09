"""
MLflow training harness for model training and tracking.
"""
import logging
import numpy as np
from typing import Dict, Any, Tuple
from decision_agent.utils.mlflow_utils import setup_mlflow, log_model_metrics, log_model_params, get_mlflow_run_id

logger = logging.getLogger(__name__)


class TrainingHarness:
    """
    Training harness with MLflow integration.

    Handles:
    - Spark -> pandas boundary for small data
    - Model training with scikit-learn
    - MLflow experiment tracking
    - Model evaluation
    """

    def __init__(self, model_config: Dict[str, Any]):
        """
        Initialize training harness.

        Args:
            model_config: Model configuration from YAML
        """
        self.model_config = model_config
        self.algorithm = model_config["algorithm"]
        self.hyperparameters = model_config.get("hyperparameters", {})

        # Setup MLflow
        try:
            self.mlflow = setup_mlflow()
        except Exception as e:
            logger.warning(f"MLflow setup failed: {e}. Continuing without MLflow.")
            self.mlflow = None

    def train(self, features_df, target_col: str) -> Tuple[Any, Dict[str, float]]:
        """
        Train model on features.

        Args:
            features_df: DataFrame with features and target
            target_col: Name of target column

        Returns:
            Tuple of (trained_model, training_metrics)
        """
        logger.info(f"Training {self.algorithm} model...")

        # Convert Spark DataFrame to pandas if needed
        pdf = self._to_pandas(features_df)

        # Prepare feature matrix and target
        X, y = self._prepare_data(pdf, target_col)

        logger.info(f"Training data shape: X={X.shape}, y={y.shape}")

        # Initialize model
        model = self._get_model()

        # Train with MLflow tracking
        if self.mlflow:
            with self.mlflow.start_run():
                # Log parameters
                log_model_params(self.hyperparameters)
                log_model_params({
                    "algorithm": self.algorithm,
                    "n_features": X.shape[1],
                    "n_samples": X.shape[0]
                })

                # Train model
                model.fit(X, y)

                # Evaluate on training set
                train_metrics = self._evaluate_model(model, X, y)

                # Log metrics
                log_model_metrics(train_metrics)

                # Log model
                self.mlflow.sklearn.log_model(model, "model")

                run_id = get_mlflow_run_id()
                logger.info(f"Model logged to MLflow run: {run_id}")

        else:
            # Train without MLflow
            model.fit(X, y)
            train_metrics = self._evaluate_model(model, X, y)

        logger.info(f"Training completed. Metrics: {train_metrics}")
        return model, train_metrics

    def predict(self, model, features_df) -> np.ndarray:
        """
        Make predictions with trained model.

        Args:
            model: Trained model
            features_df: DataFrame with features

        Returns:
            Predictions array
        """
        pdf = self._to_pandas(features_df)
        X, _ = self._prepare_data(pdf, target_col=None, drop_target=False)

        predictions = model.predict(X)
        logger.info(f"Generated {len(predictions)} predictions")

        return predictions

    def evaluate(self, features_df, predictions: np.ndarray, target_col: str) -> Dict[str, float]:
        """
        Evaluate predictions against ground truth.

        Args:
            features_df: DataFrame with features and target
            predictions: Model predictions
            target_col: Name of target column

        Returns:
            Dictionary of evaluation metrics
        """
        pdf = self._to_pandas(features_df)
        y_true = pdf[target_col].values

        metrics = self._compute_metrics(y_true, predictions)
        logger.info(f"Evaluation metrics: {metrics}")

        return metrics

    def _get_model(self):
        """Initialize model based on algorithm"""
        if self.algorithm == "gradient_boosting":
            from sklearn.ensemble import GradientBoostingRegressor
            return GradientBoostingRegressor(**self.hyperparameters)

        elif self.algorithm == "random_forest":
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(**self.hyperparameters)

        elif self.algorithm == "linear_regression":
            from sklearn.linear_model import LinearRegression
            return LinearRegression(**self.hyperparameters)

        elif self.algorithm == "logistic_regression":
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(**self.hyperparameters)

        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

    def _to_pandas(self, df):
        """Convert Spark DataFrame to pandas if needed"""
        try:
            from pyspark.sql import DataFrame as SparkDataFrame

            if isinstance(df, SparkDataFrame):
                logger.info("Converting Spark DataFrame to pandas...")
                pdf = df.toPandas()
                logger.info(f"Converted to pandas: {pdf.shape}")
                return pdf
        except ImportError:
            pass

        return df

    def _prepare_data(self, pdf, target_col: str = None, drop_target: bool = True):
        """
        Prepare feature matrix and target vector.

        Args:
            pdf: Pandas DataFrame
            target_col: Target column name
            drop_target: Whether to drop target from features

        Returns:
            Tuple of (X, y) or (X, None)
        """
        # Columns to exclude from features
        exclude_cols = [
            "customer_id", "transaction_timestamp", "transaction_category",
            "transaction_amount", "account_balance"
        ]

        if target_col:
            exclude_cols.append(target_col)

        # Select feature columns
        feature_cols = [col for col in pdf.columns if col not in exclude_cols]

        logger.info(f"Using {len(feature_cols)} features: {feature_cols[:10]}...")

        X = pdf[feature_cols].fillna(0).values

        if target_col and target_col in pdf.columns:
            y = pdf[target_col].values
        else:
            y = None

        return X, y

    def _evaluate_model(self, model, X, y) -> Dict[str, float]:
        """Evaluate model and return metrics"""
        predictions = model.predict(X)
        return self._compute_metrics(y, predictions)

    def _compute_metrics(self, y_true, y_pred) -> Dict[str, float]:
        """Compute evaluation metrics"""
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        metrics = {
            "mae": mean_absolute_error(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "r2": r2_score(y_true, y_pred)
        }

        # Check if classification (for future)
        if len(np.unique(y_true)) < 20:  # Heuristic for classification
            try:
                from sklearn.metrics import accuracy_score, precision_score, recall_score

                # Binary classification metrics
                if len(np.unique(y_true)) == 2:
                    y_pred_binary = (y_pred > 0.5).astype(int)
                    metrics.update({
                        "accuracy": accuracy_score(y_true, y_pred_binary),
                        "precision": precision_score(y_true, y_pred_binary, zero_division=0),
                        "recall": recall_score(y_true, y_pred_binary, zero_division=0)
                    })
            except Exception as e:
                logger.debug(f"Could not compute classification metrics: {e}")

        return metrics
