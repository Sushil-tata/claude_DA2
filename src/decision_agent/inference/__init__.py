"""
Inference modules for production model serving.

Components:
- model_router: Route customers to Champion vs Challenger
- shadow_deployment: Run Challenger in shadow mode
- explainer: Generate SHAP values for predictions
"""

from decision_agent.inference.model_router import ModelRouter, route_prediction
from decision_agent.inference.shadow_deployment import ShadowDeployment
from decision_agent.inference.explainer import Explainer, explain_prediction

__all__ = [
    "ModelRouter",
    "route_prediction",
    "ShadowDeployment",
    "Explainer",
    "explain_prediction",
]
