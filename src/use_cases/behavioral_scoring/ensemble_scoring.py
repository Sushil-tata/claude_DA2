"""
Multi-Model Ensemble Risk Architecture

Implements ensemble methods for behavioral scoring:
- Segment-wise ensemble (different models per segment)
- Temporal ensemble (combining different time windows)
- Stacking with behavioral meta-features
- Dynamic weighting based on recency
- Model diversity optimization
- Ensemble calibration
- Performance monitoring per segment

Example:
    >>> from src.use_cases.behavioral_scoring.ensemble_scoring import BehavioralEnsembleScorer
    >>> 
    >>> # Create ensemble
    >>> ensemble = BehavioralEnsembleScorer(ensemble_type="stacking")
    >>> ensemble.fit(X_train, y_train, segments=segments_train)
    >>> 
    >>> # Score with ensemble
    >>> scores = ensemble.predict(X_test, segments=segments_test)
    >>> calibrated_scores = ensemble.calibrate_scores(scores, y_test)
"""

import joblib
import numpy as np
import pandas as pd
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import defaultdict

from loguru import logger
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
    StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss
)
from sklearn.preprocessing import StandardScaler

try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    logger.warning("LightGBM not available. Install with: pip install lightgbm")
    LGBM_AVAILABLE = False

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    logger.warning("XGBoost not available. Install with: pip install xgboost")
    XGB_AVAILABLE = False

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    logger.warning("CatBoost not available. Install with: pip install catboost")
    CATBOOST_AVAILABLE = False


class DiversityOptimizer:
    """Optimize model diversity in ensemble."""
    
    def __init__(self):
        """Initialize diversity optimizer."""
        pass
    
    def compute_diversity(
        self,
        predictions: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute diversity metrics for ensemble predictions.
        
        Args:
            predictions: Array of shape (n_models, n_samples) with predictions
            
        Returns:
            Dictionary of diversity metrics
            
        Example:
            >>> predictions = np.array([model1_preds, model2_preds, model3_preds])
            >>> diversity = optimizer.compute_diversity(predictions)
            >>> print(f"Q-statistic: {diversity['q_statistic']:.3f}")
        """
        n_models = predictions.shape[0]
        
        diversity_metrics = {}
        
        # Pairwise disagreement
        disagreements = []
        for i in range(n_models):
            for j in range(i + 1, n_models):
                # Binarize predictions
                pred_i = (predictions[i] > 0.5).astype(int)
                pred_j = (predictions[j] > 0.5).astype(int)
                
                disagreement = np.mean(pred_i != pred_j)
                disagreements.append(disagreement)
        
        diversity_metrics['avg_disagreement'] = np.mean(disagreements) if disagreements else 0.0
        
        # Q-statistic (Yule's Q)
        q_statistics = []
        for i in range(n_models):
            for j in range(i + 1, n_models):
                pred_i = (predictions[i] > 0.5).astype(int)
                pred_j = (predictions[j] > 0.5).astype(int)
                
                n11 = np.sum((pred_i == 1) & (pred_j == 1))
                n00 = np.sum((pred_i == 0) & (pred_j == 0))
                n10 = np.sum((pred_i == 1) & (pred_j == 0))
                n01 = np.sum((pred_i == 0) & (pred_j == 1))
                
                denominator = n11 * n00 + n10 * n01
                if denominator > 0:
                    q = (n11 * n00 - n10 * n01) / denominator
                    q_statistics.append(q)
        
        diversity_metrics['q_statistic'] = np.mean(q_statistics) if q_statistics else 0.0
        
        # Correlation diversity
        correlations = []
        for i in range(n_models):
            for j in range(i + 1, n_models):
                corr = np.corrcoef(predictions[i], predictions[j])[0, 1]
                correlations.append(corr)
        
        diversity_metrics['avg_correlation'] = np.mean(correlations) if correlations else 0.0
        diversity_metrics['diversity_score'] = 1.0 - diversity_metrics['avg_correlation']
        
        logger.debug(f"Diversity metrics: {diversity_metrics}")
        return diversity_metrics
    
    def select_diverse_models(
        self,
        models: List[BaseEstimator],
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_select: int = 5,
        diversity_weight: float = 0.3
    ) -> List[int]:
        """
        Select diverse subset of models.
        
        Args:
            models: List of trained models
            X_val: Validation features
            y_val: Validation labels
            n_select: Number of models to select
            diversity_weight: Weight for diversity vs accuracy (0-1)
            
        Returns:
            Indices of selected models
        """
        n_models = len(models)
        n_select = min(n_select, n_models)
        
        # Get predictions
        predictions = np.array([
            model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba')
            else model.decision_function(X_val)
            for model in models
        ])
        
        # Compute individual accuracies
        accuracies = np.array([
            roc_auc_score(y_val, pred) for pred in predictions
        ])
        
        # Greedy selection
        selected_indices = []
        
        # Start with best model
        best_idx = np.argmax(accuracies)
        selected_indices.append(best_idx)
        
        # Iteratively add most diverse + accurate models
        while len(selected_indices) < n_select:
            best_score = -np.inf
            best_candidate = None
            
            for idx in range(n_models):
                if idx in selected_indices:
                    continue
                
                # Compute diversity with selected models
                candidate_preds = np.vstack([predictions[i] for i in selected_indices] + [predictions[idx]])
                diversity = self.compute_diversity(candidate_preds)
                
                # Combined score
                score = (1 - diversity_weight) * accuracies[idx] + diversity_weight * diversity['diversity_score']
                
                if score > best_score:
                    best_score = score
                    best_candidate = idx
            
            if best_candidate is not None:
                selected_indices.append(best_candidate)
            else:
                break
        
        logger.info(f"Selected {len(selected_indices)} diverse models")
        return selected_indices


class TemporalWeightCalculator:
    """Calculate dynamic weights based on temporal recency."""
    
    def __init__(
        self,
        decay_type: str = "exponential",
        half_life_days: float = 30.0
    ):
        """
        Initialize temporal weight calculator.
        
        Args:
            decay_type: Type of decay ('exponential', 'linear', 'step')
            half_life_days: Half-life for exponential decay in days
            
        Example:
            >>> calculator = TemporalWeightCalculator(decay_type="exponential")
            >>> weights = calculator.calculate_weights(days_ago=np.array([1, 7, 30, 90]))
        """
        self.decay_type = decay_type
        self.half_life_days = half_life_days
        
    def calculate_weights(
        self,
        days_ago: np.ndarray
    ) -> np.ndarray:
        """
        Calculate temporal weights.
        
        Args:
            days_ago: Array of days since observation
            
        Returns:
            Array of weights (0-1)
        """
        if self.decay_type == "exponential":
            # Exponential decay: w = exp(-λt)
            decay_constant = np.log(2) / self.half_life_days
            weights = np.exp(-decay_constant * days_ago)
            
        elif self.decay_type == "linear":
            # Linear decay
            max_days = np.max(days_ago) if len(days_ago) > 0 else 1
            weights = 1.0 - (days_ago / max_days)
            weights = np.clip(weights, 0, 1)
            
        elif self.decay_type == "step":
            # Step decay
            weights = np.ones_like(days_ago, dtype=float)
            weights[days_ago > self.half_life_days] = 0.5
            weights[days_ago > 2 * self.half_life_days] = 0.25
            
        else:
            raise ValueError(f"Unknown decay type: {self.decay_type}")
        
        # Normalize
        weights = weights / (np.sum(weights) + 1e-10)
        
        return weights
    
    def weighted_ensemble(
        self,
        predictions: np.ndarray,
        timestamps: np.ndarray,
        reference_time: Optional[float] = None
    ) -> np.ndarray:
        """
        Combine predictions with temporal weighting.
        
        Args:
            predictions: Array of shape (n_models, n_samples)
            timestamps: Array of timestamps for each model
            reference_time: Reference time (uses max if None)
            
        Returns:
            Weighted ensemble predictions
        """
        if reference_time is None:
            reference_time = np.max(timestamps)
        
        # Calculate days ago
        days_ago = (reference_time - timestamps) / (24 * 3600)  # Assuming timestamps in seconds
        
        # Calculate weights
        weights = self.calculate_weights(days_ago)
        
        # Weighted average
        weighted_preds = np.average(predictions, axis=0, weights=weights)
        
        return weighted_preds


class EnsembleCalibrator:
    """Calibrate ensemble predictions."""
    
    def __init__(self, method: str = "isotonic"):
        """
        Initialize ensemble calibrator.
        
        Args:
            method: Calibration method ('isotonic', 'platt', 'beta')
            
        Example:
            >>> calibrator = EnsembleCalibrator(method="isotonic")
            >>> calibrator.fit(predictions, true_labels)
            >>> calibrated = calibrator.transform(new_predictions)
        """
        self.method = method
        self.calibrator = None
        self.is_fitted = False
        
    def fit(
        self,
        predictions: np.ndarray,
        y_true: np.ndarray
    ):
        """
        Fit calibrator.
        
        Args:
            predictions: Raw predictions (0-1)
            y_true: True labels
        """
        from sklearn.calibration import calibration_curve
        from sklearn.isotonic import IsotonicRegression
        
        if self.method == "isotonic":
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
            self.calibrator.fit(predictions, y_true)
            
        elif self.method == "platt":
            # Platt scaling (logistic regression)
            self.calibrator = LogisticRegression()
            self.calibrator.fit(predictions.reshape(-1, 1), y_true)
            
        elif self.method == "beta":
            # Beta calibration (simplified)
            from scipy.optimize import minimize
            
            def beta_loss(params, pred, y):
                a, b, c = params
                calibrated = 1 / (1 + np.exp(-(a * np.log(pred + 1e-10) + b * np.log(1 - pred + 1e-10) + c)))
                return log_loss(y, calibrated)
            
            result = minimize(
                beta_loss,
                x0=[1.0, 1.0, 0.0],
                args=(predictions, y_true),
                method='L-BFGS-B'
            )
            self.calibrator = result.x
        
        self.is_fitted = True
        logger.info(f"Calibrator fitted with method: {self.method}")
    
    def transform(
        self,
        predictions: np.ndarray
    ) -> np.ndarray:
        """
        Calibrate predictions.
        
        Args:
            predictions: Raw predictions
            
        Returns:
            Calibrated predictions
        """
        if not self.is_fitted:
            raise ValueError("Calibrator not fitted")
        
        if self.method == "isotonic":
            calibrated = self.calibrator.predict(predictions)
            
        elif self.method == "platt":
            calibrated = self.calibrator.predict_proba(predictions.reshape(-1, 1))[:, 1]
            
        elif self.method == "beta":
            a, b, c = self.calibrator
            calibrated = 1 / (1 + np.exp(-(a * np.log(predictions + 1e-10) + b * np.log(1 - predictions + 1e-10) + c)))
        
        return calibrated
    
    def fit_transform(
        self,
        predictions: np.ndarray,
        y_true: np.ndarray
    ) -> np.ndarray:
        """Fit and transform."""
        self.fit(predictions, y_true)
        return self.transform(predictions)


class BehavioralEnsembleScorer:
    """
    Multi-model ensemble scorer for behavioral risk assessment.
    
    Supports multiple ensemble strategies including voting, stacking,
    segment-wise ensembles, and temporal ensembles.
    """
    
    def __init__(
        self,
        ensemble_type: str = "stacking",
        base_models: Optional[List[str]] = None,
        meta_model: Optional[BaseEstimator] = None,
        segment_aware: bool = True,
        temporal_weighting: bool = False,
        calibration_method: Optional[str] = "isotonic"
    ):
        """
        Initialize behavioral ensemble scorer.
        
        Args:
            ensemble_type: Type of ensemble ('voting', 'stacking', 'weighted', 'segment')
            base_models: List of base model types to use
            meta_model: Meta-learner for stacking (uses LogisticRegression if None)
            segment_aware: Whether to train segment-specific models
            temporal_weighting: Whether to use temporal weighting
            calibration_method: Calibration method (None to disable)
            
        Example:
            >>> ensemble = BehavioralEnsembleScorer(
            ...     ensemble_type="stacking",
            ...     base_models=["lgbm", "xgb", "rf"],
            ...     segment_aware=True
            ... )
            >>> ensemble.fit(X_train, y_train, segments=train_segments)
        """
        self.ensemble_type = ensemble_type
        self.base_models = base_models or ["lgbm", "rf", "logistic"]
        self.meta_model = meta_model
        self.segment_aware = segment_aware
        self.temporal_weighting = temporal_weighting
        self.calibration_method = calibration_method
        
        self.models = {}
        self.segment_models = defaultdict(dict)
        self.diversity_optimizer = DiversityOptimizer()
        self.temporal_calculator = TemporalWeightCalculator() if temporal_weighting else None
        self.calibrator = EnsembleCalibrator(calibration_method) if calibration_method else None
        self.scaler = StandardScaler()
        
        self.is_fitted = False
        
    def _create_base_model(self, model_type: str) -> BaseEstimator:
        """Create base model instance."""
        if model_type == "lgbm" and LGBM_AVAILABLE:
            return LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
        elif model_type == "xgb" and XGB_AVAILABLE:
            return XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
        elif model_type == "catboost" and CATBOOST_AVAILABLE:
            return CatBoostClassifier(iterations=100, random_state=42, verbose=0)
        elif model_type == "rf":
            return RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == "gb":
            return GradientBoostingClassifier(n_estimators=100, random_state=42)
        elif model_type == "logistic":
            return LogisticRegression(max_iter=1000, random_state=42)
        else:
            logger.warning(f"Unknown model type: {model_type}, using logistic regression")
            return LogisticRegression(max_iter=1000, random_state=42)
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        segments: Optional[np.ndarray] = None,
        timestamps: Optional[np.ndarray] = None
    ):
        """
        Fit ensemble scorer.
        
        Args:
            X: Features
            y: Labels
            segments: Segment identifiers (optional)
            timestamps: Timestamps for temporal weighting (optional)
            
        Example:
            >>> ensemble.fit(X_train, y_train, segments=train_segments)
        """
        logger.info(f"Training ensemble with type: {self.ensemble_type}")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        if self.segment_aware and segments is not None:
            self._fit_segment_aware(X_scaled, y, segments)
        else:
            self._fit_global(X_scaled, y)
        
        # Mark as fitted before calibration (calibrator needs to call predict)
        self.is_fitted = True
        
        # Fit calibrator if needed
        if self.calibrator is not None:
            train_preds = self.predict(X, segments=segments)
            self.calibrator.fit(train_preds, y)
        
        logger.info("Ensemble training completed")
    
    def _fit_global(self, X: np.ndarray, y: np.ndarray):
        """Fit global ensemble (not segment-aware)."""
        if self.ensemble_type == "voting":
            # Create voting ensemble
            estimators = [
                (f"model_{i}_{model_type}", self._create_base_model(model_type))
                for i, model_type in enumerate(self.base_models)
            ]
            self.models['global'] = VotingClassifier(
                estimators=estimators,
                voting='soft'
            )
            self.models['global'].fit(X, y)
            
        elif self.ensemble_type == "stacking":
            # Create stacking ensemble
            estimators = [
                (f"model_{i}_{model_type}", self._create_base_model(model_type))
                for i, model_type in enumerate(self.base_models)
            ]
            
            final_estimator = self.meta_model or LogisticRegression()
            
            self.models['global'] = StackingClassifier(
                estimators=estimators,
                final_estimator=final_estimator,
                cv=5
            )
            self.models['global'].fit(X, y)
            
        elif self.ensemble_type == "weighted":
            # Train individual models and learn weights
            base_models_list = []
            for model_type in self.base_models:
                model = self._create_base_model(model_type)
                model.fit(X, y)
                base_models_list.append(model)
            
            self.models['global'] = {
                'models': base_models_list,
                'weights': np.ones(len(base_models_list)) / len(base_models_list)
            }
            
        else:
            raise ValueError(f"Unknown ensemble type: {self.ensemble_type}")
    
    def _fit_segment_aware(
        self,
        X: np.ndarray,
        y: np.ndarray,
        segments: np.ndarray
    ):
        """Fit segment-aware ensemble."""
        unique_segments = np.unique(segments)
        logger.info(f"Training on {len(unique_segments)} segments")
        
        for segment_id in unique_segments:
            # Get segment data
            segment_mask = segments == segment_id
            X_seg = X[segment_mask]
            y_seg = y[segment_mask]
            
            if len(y_seg) < 10:
                logger.warning(f"Segment {segment_id} has too few samples ({len(y_seg)}), skipping")
                continue
            
            # Train models for this segment
            segment_models = []
            for model_type in self.base_models:
                model = self._create_base_model(model_type)
                try:
                    model.fit(X_seg, y_seg)
                    segment_models.append(model)
                except Exception as e:
                    logger.warning(f"Failed to train {model_type} for segment {segment_id}: {e}")
            
            if segment_models:
                self.segment_models[segment_id] = {
                    'models': segment_models,
                    'weights': np.ones(len(segment_models)) / len(segment_models)
                }
        
        logger.info(f"Trained models for {len(self.segment_models)} segments")
    
    def predict(
        self,
        X: np.ndarray,
        segments: Optional[np.ndarray] = None,
        return_base_predictions: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict risk scores.
        
        Args:
            X: Features
            segments: Segment identifiers (required if segment_aware=True)
            return_base_predictions: Whether to return base model predictions
            
        Returns:
            Risk scores (and optionally base predictions)
            
        Example:
            >>> scores = ensemble.predict(X_test, segments=test_segments)
            >>> scores, base_preds = ensemble.predict(X_test, return_base_predictions=True)
        """
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted")
        
        X_scaled = self.scaler.transform(X)
        
        if self.segment_aware and segments is not None:
            predictions = self._predict_segment_aware(X_scaled, segments)
        else:
            predictions = self._predict_global(X_scaled)
        
        # Calibrate if needed (check if calibrator is fitted)
        if self.calibrator is not None and self.calibrator.is_fitted:
            predictions = self.calibrator.transform(predictions)
        
        if return_base_predictions:
            base_preds = self._get_base_predictions(X_scaled, segments)
            return predictions, base_preds
        
        return predictions
    
    def _predict_global(self, X: np.ndarray) -> np.ndarray:
        """Predict with global ensemble."""
        if 'global' not in self.models:
            raise ValueError("Global model not fitted")
        
        model = self.models['global']
        
        if isinstance(model, dict):
            # Weighted ensemble
            predictions = []
            for m in model['models']:
                if hasattr(m, 'predict_proba'):
                    pred = m.predict_proba(X)[:, 1]
                else:
                    pred = m.decision_function(X)
                predictions.append(pred)
            
            predictions = np.array(predictions)
            weights = model['weights']
            
            return np.average(predictions, axis=0, weights=weights)
        else:
            # Voting or Stacking
            if hasattr(model, 'predict_proba'):
                return model.predict_proba(X)[:, 1]
            else:
                return model.decision_function(X)
    
    def _predict_segment_aware(
        self,
        X: np.ndarray,
        segments: np.ndarray
    ) -> np.ndarray:
        """Predict with segment-aware ensemble."""
        predictions = np.zeros(len(X))
        
        unique_segments = np.unique(segments)
        
        for segment_id in unique_segments:
            segment_mask = segments == segment_id
            X_seg = X[segment_mask]
            
            if segment_id in self.segment_models:
                # Use segment-specific models
                segment_model = self.segment_models[segment_id]
                segment_preds = []
                
                for model in segment_model['models']:
                    if hasattr(model, 'predict_proba'):
                        pred = model.predict_proba(X_seg)[:, 1]
                    else:
                        pred = model.decision_function(X_seg)
                    segment_preds.append(pred)
                
                segment_preds = np.array(segment_preds)
                weights = segment_model['weights']
                
                predictions[segment_mask] = np.average(segment_preds, axis=0, weights=weights)
            else:
                # Fallback to global model
                logger.warning(f"No model for segment {segment_id}, using global model")
                if 'global' in self.models:
                    predictions[segment_mask] = self._predict_global(X_seg)
                else:
                    # Use first available segment model
                    fallback_model = list(self.segment_models.values())[0]
                    fallback_preds = []
                    for model in fallback_model['models']:
                        if hasattr(model, 'predict_proba'):
                            pred = model.predict_proba(X_seg)[:, 1]
                        else:
                            pred = model.decision_function(X_seg)
                        fallback_preds.append(pred)
                    
                    predictions[segment_mask] = np.mean(fallback_preds, axis=0)
        
        return predictions
    
    def _get_base_predictions(
        self,
        X: np.ndarray,
        segments: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Get base model predictions."""
        base_preds = []
        
        if self.segment_aware and segments is not None:
            # Get predictions from first segment (as example)
            segment_id = list(self.segment_models.keys())[0]
            models = self.segment_models[segment_id]['models']
        elif 'global' in self.models:
            model = self.models['global']
            if isinstance(model, dict):
                models = model['models']
            elif hasattr(model, 'estimators_'):
                models = [est for _, est in model.estimators_]
            else:
                models = [model]
        else:
            return np.array([])
        
        for model in models:
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X)[:, 1]
            else:
                pred = model.decision_function(X)
            base_preds.append(pred)
        
        return np.array(base_preds).T
    
    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        segments: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Evaluate ensemble performance.
        
        Args:
            X: Features
            y: Labels
            segments: Segment identifiers
            
        Returns:
            Dictionary of metrics
        """
        predictions = self.predict(X, segments=segments)
        
        metrics = {
            'auc_roc': roc_auc_score(y, predictions),
            'avg_precision': average_precision_score(y, predictions),
            'brier_score': brier_score_loss(y, predictions)
        }
        
        # Segment-level metrics
        if self.segment_aware and segments is not None:
            unique_segments = np.unique(segments)
            segment_metrics = {}
            
            for segment_id in unique_segments:
                segment_mask = segments == segment_id
                if np.sum(segment_mask) >= 10:
                    seg_auc = roc_auc_score(y[segment_mask], predictions[segment_mask])
                    segment_metrics[f'segment_{segment_id}_auc'] = seg_auc
            
            metrics.update(segment_metrics)
        
        logger.info(f"Evaluation metrics: {metrics}")
        return metrics
    
    def calibrate_scores(
        self,
        predictions: np.ndarray,
        y_true: np.ndarray,
        method: Optional[str] = None
    ) -> np.ndarray:
        """
        Calibrate prediction scores.
        
        Args:
            predictions: Raw predictions
            y_true: True labels
            method: Calibration method (uses default if None)
            
        Returns:
            Calibrated scores
        """
        if method is None:
            method = self.calibration_method or "isotonic"
        
        calibrator = EnsembleCalibrator(method=method)
        calibrated = calibrator.fit_transform(predictions, y_true)
        
        return calibrated
    
    def compute_diversity(self, X: np.ndarray) -> Dict[str, float]:
        """
        Compute ensemble diversity metrics.
        
        Args:
            X: Features
            
        Returns:
            Diversity metrics
        """
        base_preds = self._get_base_predictions(self.scaler.transform(X))
        
        if base_preds.shape[1] > 1:
            diversity = self.diversity_optimizer.compute_diversity(base_preds.T)
        else:
            diversity = {'diversity_score': 0.0}
        
        return diversity
    
    def save(self, path: Union[str, Path]):
        """Save ensemble to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"Ensemble saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'BehavioralEnsembleScorer':
        """Load ensemble from disk."""
        ensemble = joblib.load(path)
        logger.info(f"Ensemble loaded from {path}")
        return ensemble
