"""
Explainer - Model Interpretability via SHAP

Generates explanations for model predictions using SHAP values.
Required for FCRA compliance (adverse action notices).

Why This Matters:
- FCRA requires explanation for credit decisions
- Stakeholders need to understand model behavior
- Debugging and model validation
- Trust and transparency

SHAP (SHapley Additive exPlanations):
- Game theory-based approach to explain predictions
- Shows contribution of each feature to prediction
- Works with tree-based models (GBM, RF, XGBoost)

Integration Point:
- Called at inference time for each prediction
- Outputs: top_3_features, reason_codes, shap_values
- Logged by decisions/prediction_logger.py
"""

import logging
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Explainer:
    """
    Generate model explanations using SHAP values.

    For production, we explain top-K predictions (not all).
    """

    def __init__(self, model, feature_names: List[str], model_type: str = "tree"):
        """
        Initialize explainer.

        Args:
            model: Trained model (scikit-learn compatible)
            feature_names: List of feature names
            model_type: 'tree' for tree models, 'linear' for linear models
        """
        self.model = model
        self.feature_names = feature_names
        self.model_type = model_type
        self.explainer = None

        self._initialize_explainer()

    def _initialize_explainer(self):
        """Initialize SHAP explainer based on model type."""
        try:
            import shap

            if self.model_type == "tree":
                logger.info("Initializing TreeExplainer for tree-based model...")
                self.explainer = shap.TreeExplainer(self.model)
            elif self.model_type == "linear":
                logger.info("Initializing LinearExplainer for linear model...")
                self.explainer = shap.LinearExplainer(self.model, masker=None)
            else:
                logger.info("Initializing KernelExplainer (model-agnostic, slow)...")
                # For unknown model types, use KernelExplainer (slower)
                # Requires background data - sample from training set
                self.explainer = None  # Need background data

            logger.info("SHAP explainer initialized successfully")

        except ImportError:
            logger.error("SHAP library not installed. Install with: pip install shap")
            self.explainer = None
        except Exception as e:
            logger.error(f"Failed to initialize SHAP explainer: {e}")
            self.explainer = None

    def explain_prediction(
        self,
        features: pd.DataFrame,
        prediction: float,
        top_k: int = 3
    ) -> Dict:
        """
        Explain a single prediction.

        Args:
            features: DataFrame with features for one customer (1 row)
            prediction: Model prediction for this customer
            top_k: Number of top features to return

        Returns:
            Dict with:
            - top_features: List of (feature_name, shap_value) tuples
            - reason_codes: List of feature names (for adverse action)
            - shap_values: Full SHAP value array
            - base_value: Model baseline prediction
        """
        if self.explainer is None:
            logger.warning("SHAP explainer not available. Returning fallback explanation.")
            return self._fallback_explanation(features, top_k)

        try:
            # Compute SHAP values
            shap_values = self.explainer.shap_values(features)

            # Handle different SHAP output formats
            if isinstance(shap_values, list):
                # Multi-class classifier (take first class)
                shap_values = shap_values[0]

            # Flatten if 2D
            if len(shap_values.shape) > 1:
                shap_values = shap_values[0]

            # Get base value (expected value)
            if hasattr(self.explainer, 'expected_value'):
                base_value = self.explainer.expected_value
                if isinstance(base_value, np.ndarray):
                    base_value = base_value[0]
            else:
                base_value = 0.0

            # Get top K features by absolute SHAP value
            abs_shap = np.abs(shap_values)
            top_indices = np.argsort(abs_shap)[-top_k:][::-1]

            top_features = [
                (self.feature_names[i], float(shap_values[i]))
                for i in top_indices
            ]

            reason_codes = [self.feature_names[i] for i in top_indices]

            result = {
                "top_features": top_features,
                "reason_codes": reason_codes,
                "shap_values": shap_values.tolist(),
                "base_value": float(base_value),
                "prediction": float(prediction)
            }

            return result

        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}")
            return self._fallback_explanation(features, top_k)

    def explain_batch(
        self,
        features_df: pd.DataFrame,
        predictions: np.ndarray,
        top_k: int = 3
    ) -> List[Dict]:
        """
        Explain multiple predictions (batch).

        Args:
            features_df: DataFrame with features (N rows)
            predictions: Array of predictions (N values)
            top_k: Number of top features per prediction

        Returns:
            List of explanation dicts (one per prediction)
        """
        if self.explainer is None:
            logger.warning("SHAP explainer not available. Returning fallback explanations.")
            return [self._fallback_explanation(features_df.iloc[[i]], top_k)
                   for i in range(len(features_df))]

        try:
            # Compute SHAP values for batch
            shap_values = self.explainer.shap_values(features_df)

            # Handle different formats
            if isinstance(shap_values, list):
                shap_values = shap_values[0]

            explanations = []

            for i in range(len(features_df)):
                shap_i = shap_values[i] if len(shap_values.shape) > 1 else shap_values

                abs_shap = np.abs(shap_i)
                top_indices = np.argsort(abs_shap)[-top_k:][::-1]

                top_features = [
                    (self.feature_names[idx], float(shap_i[idx]))
                    for idx in top_indices
                ]

                reason_codes = [self.feature_names[idx] for idx in top_indices]

                explanations.append({
                    "top_features": top_features,
                    "reason_codes": reason_codes,
                    "shap_values": shap_i.tolist(),
                    "prediction": float(predictions[i])
                })

            return explanations

        except Exception as e:
            logger.error(f"Batch SHAP explanation failed: {e}")
            return [self._fallback_explanation(features_df.iloc[[i]], top_k)
                   for i in range(len(features_df))]

    def _fallback_explanation(self, features: pd.DataFrame, top_k: int = 3) -> Dict:
        """
        Fallback explanation when SHAP not available.

        Uses feature importance if available, otherwise returns empty.
        """
        logger.debug("Using fallback explanation (no SHAP)")

        # Try to get feature importance from model
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            top_indices = np.argsort(importances)[-top_k:][::-1]

            top_features = [
                (self.feature_names[i], float(importances[i]))
                for i in top_indices
            ]

            reason_codes = [self.feature_names[i] for i in top_indices]

            return {
                "top_features": top_features,
                "reason_codes": reason_codes,
                "explanation_type": "feature_importance",
                "shap_values": None
            }

        # No feature importance available
        return {
            "top_features": [],
            "reason_codes": [],
            "explanation_type": "unavailable",
            "shap_values": None
        }

    def get_global_feature_importance(self) -> Dict[str, float]:
        """
        Get global feature importance across all training data.

        Useful for model documentation and validation.

        Returns:
            Dict mapping feature name to importance score
        """
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            return {
                name: float(imp)
                for name, imp in zip(self.feature_names, importances)
            }

        logger.warning("Model does not have feature_importances_ attribute")
        return {}


def explain_prediction(
    model,
    features: pd.DataFrame,
    prediction: float,
    feature_names: List[str],
    top_k: int = 3,
    model_type: str = "tree"
) -> Dict:
    """
    Convenience function to explain a single prediction.

    Args:
        model: Trained model
        features: Features for one customer (1 row DataFrame)
        prediction: Model prediction
        feature_names: List of feature names
        top_k: Number of top features to return
        model_type: 'tree', 'linear', or 'other'

    Returns:
        Explanation dict
    """
    explainer = Explainer(model, feature_names, model_type)
    return explainer.explain_prediction(features, prediction, top_k)
