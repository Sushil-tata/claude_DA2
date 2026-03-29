"""
MLflow utilities for experiment tracking and model registry.
"""
import logging

logger = logging.getLogger(__name__)


def setup_mlflow(experiment_name: str = "/decision_agent/experiments", tracking_uri: str = None):
    """
    Setup MLflow experiment tracking.

    Args:
        experiment_name: Name of the MLflow experiment
        tracking_uri: MLflow tracking URI (default: local ./mlruns)

    Returns:
        MLflow client
    """
    try:
        import mlflow

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            logger.info(f"MLflow tracking URI set to: {tracking_uri}")
        else:
            logger.info("Using default MLflow tracking URI (local ./mlruns)")

        # Set or create experiment
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
                logger.info(f"Created MLflow experiment: {experiment_name} (ID: {experiment_id})")
            else:
                experiment_id = experiment.experiment_id
                logger.info(f"Using existing MLflow experiment: {experiment_name} (ID: {experiment_id})")

            mlflow.set_experiment(experiment_name)
        except Exception as e:
            logger.warning(f"Could not set MLflow experiment: {e}")
            logger.warning("Continuing without experiment setup")

        return mlflow

    except ImportError:
        logger.error("MLflow not installed. Install with: pip install mlflow")
        raise


def log_model_metrics(metrics: dict, step: int = None):
    """
    Log metrics to MLflow.

    Args:
        metrics: Dictionary of metric name -> value
        step: Optional step number for tracking metrics over time
    """
    try:
        import mlflow

        for metric_name, metric_value in metrics.items():
            if step is not None:
                mlflow.log_metric(metric_name, metric_value, step=step)
            else:
                mlflow.log_metric(metric_name, metric_value)

        logger.info(f"Logged {len(metrics)} metrics to MLflow")

    except ImportError:
        logger.warning("MLflow not available, skipping metric logging")
    except Exception as e:
        logger.error(f"Failed to log metrics: {e}")


def log_model_params(params: dict):
    """
    Log parameters to MLflow.

    Args:
        params: Dictionary of parameter name -> value
    """
    try:
        import mlflow

        # MLflow has character limits on param values
        for param_name, param_value in params.items():
            # Convert to string and truncate if needed
            param_str = str(param_value)
            if len(param_str) > 500:
                param_str = param_str[:500] + "..."

            mlflow.log_param(param_name, param_str)

        logger.info(f"Logged {len(params)} parameters to MLflow")

    except ImportError:
        logger.warning("MLflow not available, skipping parameter logging")
    except Exception as e:
        logger.error(f"Failed to log parameters: {e}")


def get_mlflow_run_id() -> str:
    """
    Get current MLflow run ID.

    Returns:
        Run ID string, or "no_mlflow_run" if not in active run
    """
    try:
        import mlflow

        active_run = mlflow.active_run()
        if active_run:
            return active_run.info.run_id
        else:
            return "no_mlflow_run"

    except ImportError:
        return "no_mlflow_run"
    except Exception:
        return "no_mlflow_run"
