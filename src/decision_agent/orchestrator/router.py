"""
Use case router - maps use_case_id to pipeline functions.
"""

from typing import Dict, Any, Callable
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def income_estimation_pipeline(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Income estimation pipeline.

    Args:
        config: Use case configuration

    Returns:
        Pipeline execution results
    """
    logger.info(f"Running income estimation pipeline (version: {config['version']})")

    results = {
        "use_case_id": config["use_case_id"],
        "status": "pipeline_placeholder",
        "message": "Income estimation pipeline - to be implemented in Phase 3-7"
    }

    # Full implementation will be added in later phases:
    # Phase 2: Data splits + as-of joins
    # Phase 3: Feature engineering
    # Phase 4: Training harness + MLflow
    # Phase 5: Validation
    # Phase 6: Decision output
    # Phase 7: End-to-end integration

    return results


def churn_prediction_pipeline(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Customer churn prediction pipeline (future use case).

    Args:
        config: Use case configuration

    Returns:
        Pipeline execution results
    """
    logger.info(f"Running churn prediction pipeline (version: {config['version']})")

    results = {
        "use_case_id": config["use_case_id"],
        "status": "not_implemented",
        "message": "Churn prediction pipeline - future enhancement"
    }

    return results


# Pipeline registry - maps use_case_id to pipeline function
PIPELINE_REGISTRY: Dict[str, Callable] = {
    "income_estimation": income_estimation_pipeline,
    "churn_prediction": churn_prediction_pipeline,
}


def route_use_case(use_case_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Route to appropriate pipeline based on use_case_id.

    Args:
        use_case_id: Use case identifier
        config: Full configuration dictionary

    Returns:
        Pipeline execution results

    Raises:
        ValueError: If use_case_id not found in registry
    """
    if use_case_id not in PIPELINE_REGISTRY:
        available = ', '.join(PIPELINE_REGISTRY.keys())
        raise ValueError(
            f"Unknown use_case_id: '{use_case_id}'. "
            f"Available use cases: {available}"
        )

    pipeline_fn = PIPELINE_REGISTRY[use_case_id]
    logger.info(f"Routing to pipeline: {use_case_id}")

    return pipeline_fn(config)


def list_available_use_cases() -> Dict[str, str]:
    """
    List all registered use cases.

    Returns:
        Dictionary mapping use_case_id to description
    """
    return {
        "income_estimation": "Estimate income levels from transaction patterns",
        "churn_prediction": "Predict customer churn risk (future)"
    }
