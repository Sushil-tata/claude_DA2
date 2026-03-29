"""
Collections NBA Propensity Model

Predicts probability of repayment across multiple time horizons with segment-specific
modeling and feature engineering integration.

Example:
    >>> from src.use_cases.collections_nba.propensity_model import PropensityModel
    >>> model = PropensityModel(config_path='config/model_config.yaml')
    >>> model.fit(X_train, y_train, X_val, y_val)
    >>> predictions = model.predict_multi_horizon(X_test)
"""

import joblib
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from loguru import logger
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.calibration import CalibratedClassifierCV

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    logger.warning("LightGBM not available. Install with: pip install lightgbm")
    LIGHTGBM_AVAILABLE = False


class PropensityModelConfig:
    """Configuration for propensity models."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize propensity model configuration.
        
        Args:
            config_path: Path to model_config.yaml
            
        Example:
            >>> config = PropensityModelConfig("config/model_config.yaml")
            >>> config.horizons
            [7, 14, 30, 60, 90]
        """
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                full_config = yaml.safe_load(f)
                propensity_config = full_config.get('propensity_model', {})
        else:
            propensity_config = {}
        
        self.horizons = propensity_config.get('horizons', [7, 14, 30, 60, 90])
        self.segments = propensity_config.get('segments', ['high_risk', 'medium_risk', 'low_risk'])
        self.calibrate = propensity_config.get('calibrate', True)
        self.threshold_optimization = propensity_config.get('threshold_optimization', 'f1')
        self.model_params = propensity_config.get('model_params', {
            'objective': 'binary',
            'metric': 'auc',
            'boosting': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.7,
            'bagging_freq': 5,
            'verbose': -1,
            'max_depth': 7,
            'min_child_samples': 20,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
        })


class PropensityModel:
    """
    Multi-horizon propensity model for collections NBA.
    
    Predicts probability of payment across multiple time horizons with
    segment-specific models and calibrated probability outputs.
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        segment_col: str = 'risk_segment',
        calibrate: bool = True
    ):
        """
        Initialize propensity model.
        
        Args:
            config_path: Path to model_config.yaml
            segment_col: Column name for customer segment
            calibrate: Whether to calibrate probability outputs
            
        Example:
            >>> model = PropensityModel(
            ...     config_path="config/model_config.yaml",
            ...     segment_col="risk_tier"
            ... )
        """
        self.config = PropensityModelConfig(config_path)
        self.segment_col = segment_col
        self.calibrate = calibrate and self.config.calibrate
        
        # Models: {horizon: {segment: model}}
        self.models: Dict[int, Dict[str, Any]] = {}
        self.thresholds: Dict[int, Dict[str, float]] = {}
        self.feature_names: Optional[List[str]] = None
        self.is_fitted = False
        
        logger.info(f"Initialized PropensityModel with horizons: {self.config.horizons}")
    
    def fit(
        self,
        X: pd.DataFrame,
        y_dict: Dict[int, pd.Series],
        X_val: Optional[pd.DataFrame] = None,
        y_val_dict: Optional[Dict[int, pd.Series]] = None,
        segments: Optional[pd.Series] = None,
        cv_folds: int = 5
    ) -> 'PropensityModel':
        """
        Fit multi-horizon propensity models.
        
        Args:
            X: Training features
            y_dict: Dictionary mapping horizon to target labels {7: y_7d, 14: y_14d, ...}
            X_val: Validation features
            y_val_dict: Validation targets by horizon
            segments: Customer segments for segment-specific modeling
            cv_folds: Number of CV folds for threshold optimization
            
        Returns:
            Self for method chaining
            
        Example:
            >>> y_dict = {7: y_7d, 14: y_14d, 30: y_30d}
            >>> model.fit(X_train, y_dict, X_val, y_val_dict, segments_train)
        """
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is required. Install with: pip install lightgbm")
        
        self.feature_names = list(X.columns)
        
        # Handle segments
        if segments is None:
            segments = pd.Series(['all'] * len(X), index=X.index)
            segment_list = ['all']
        else:
            segment_list = segments.unique().tolist()
        
        logger.info(f"Training models for {len(self.config.horizons)} horizons and {len(segment_list)} segments")
        
        for horizon in self.config.horizons:
            if horizon not in y_dict:
                logger.warning(f"Horizon {horizon}d not in y_dict, skipping")
                continue
            
            self.models[horizon] = {}
            self.thresholds[horizon] = {}
            
            y_train = y_dict[horizon]
            y_val = y_val_dict.get(horizon) if y_val_dict else None
            
            for segment in segment_list:
                logger.info(f"Training {horizon}d propensity model for segment: {segment}")
                
                # Filter data for segment
                if segment == 'all':
                    X_seg = X
                    y_seg = y_train
                    X_val_seg = X_val
                    y_val_seg = y_val
                else:
                    seg_mask = segments == segment
                    X_seg = X[seg_mask]
                    y_seg = y_train[seg_mask]
                    
                    if X_val is not None and segments is not None:
                        val_seg_mask = segments == segment
                        X_val_seg = X_val[val_seg_mask]
                        y_val_seg = y_val[val_seg_mask] if y_val is not None else None
                    else:
                        X_val_seg = None
                        y_val_seg = None
                
                if len(y_seg) < 100:
                    logger.warning(f"Segment {segment} has only {len(y_seg)} samples, skipping")
                    continue
                
                # Train base model
                model = lgb.LGBMClassifier(**self.config.model_params)
                
                eval_set = [(X_val_seg, y_val_seg)] if X_val_seg is not None and y_val_seg is not None else None
                
                model.fit(
                    X_seg, y_seg,
                    eval_set=eval_set,
                    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)] if eval_set else None
                )
                
                # Calibrate if requested
                if self.calibrate:
                    logger.info(f"Calibrating {horizon}d model for segment: {segment}")
                    model = CalibratedClassifierCV(
                        model, method='isotonic', cv=min(3, cv_folds)
                    )
                    model.fit(X_seg, y_seg)
                
                self.models[horizon][segment] = model
                
                # Optimize threshold
                if X_val_seg is not None and y_val_seg is not None:
                    threshold = self._optimize_threshold(
                        model, X_val_seg, y_val_seg, metric=self.config.threshold_optimization
                    )
                else:
                    threshold = 0.5
                
                self.thresholds[horizon][segment] = threshold
                
                # Log metrics
                if X_val_seg is not None and y_val_seg is not None:
                    val_pred = model.predict_proba(X_val_seg)[:, 1]
                    val_auc = roc_auc_score(y_val_seg, val_pred)
                    val_ap = average_precision_score(y_val_seg, val_pred)
                    logger.info(
                        f"Horizon {horizon}d, Segment {segment}: "
                        f"Val AUC={val_auc:.4f}, AP={val_ap:.4f}, Threshold={threshold:.3f}"
                    )
        
        self.is_fitted = True
        logger.info("Propensity model training complete")
        return self
    
    def predict_proba(
        self,
        X: pd.DataFrame,
        horizon: int,
        segments: Optional[pd.Series] = None
    ) -> np.ndarray:
        """
        Predict probability of payment for a specific horizon.
        
        Args:
            X: Features
            horizon: Time horizon in days
            segments: Customer segments
            
        Returns:
            Array of probabilities
            
        Example:
            >>> probs_30d = model.predict_proba(X_test, horizon=30, segments=test_segments)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if horizon not in self.models:
            raise ValueError(f"Horizon {horizon} not available. Available: {list(self.models.keys())}")
        
        if segments is None:
            segments = pd.Series(['all'] * len(X), index=X.index)
        
        predictions = np.zeros(len(X))
        
        for segment in segments.unique():
            seg_mask = segments == segment
            
            if segment not in self.models[horizon]:
                # Fallback to 'all' segment if available
                if 'all' in self.models[horizon]:
                    segment = 'all'
                else:
                    logger.warning(f"Segment {segment} not found, using first available segment")
                    segment = list(self.models[horizon].keys())[0]
            
            model = self.models[horizon][segment]
            X_seg = X[seg_mask]
            
            if len(X_seg) > 0:
                predictions[seg_mask] = model.predict_proba(X_seg)[:, 1]
        
        return predictions
    
    def predict_multi_horizon(
        self,
        X: pd.DataFrame,
        segments: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Predict probabilities across all horizons.
        
        Args:
            X: Features
            segments: Customer segments
            
        Returns:
            DataFrame with columns: prob_7d, prob_14d, prob_30d, etc.
            
        Example:
            >>> predictions = model.predict_multi_horizon(X_test, test_segments)
            >>> predictions.head()
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        result = pd.DataFrame(index=X.index)
        
        for horizon in sorted(self.models.keys()):
            probs = self.predict_proba(X, horizon, segments)
            result[f'prob_{horizon}d'] = probs
        
        return result
    
    def predict(
        self,
        X: pd.DataFrame,
        horizon: int,
        segments: Optional[pd.Series] = None,
        use_optimized_threshold: bool = True
    ) -> np.ndarray:
        """
        Predict binary payment outcome.
        
        Args:
            X: Features
            horizon: Time horizon in days
            segments: Customer segments
            use_optimized_threshold: Whether to use optimized thresholds
            
        Returns:
            Array of binary predictions
            
        Example:
            >>> predictions = model.predict(X_test, horizon=30)
        """
        probs = self.predict_proba(X, horizon, segments)
        
        if use_optimized_threshold and segments is not None:
            predictions = np.zeros(len(X), dtype=int)
            for segment in segments.unique():
                seg_mask = segments == segment
                threshold = self.thresholds[horizon].get(segment, 0.5)
                predictions[seg_mask] = (probs[seg_mask] >= threshold).astype(int)
        else:
            threshold = 0.5
            predictions = (probs >= threshold).astype(int)
        
        return predictions
    
    def _optimize_threshold(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        metric: str = 'f1'
    ) -> float:
        """Optimize classification threshold based on metric."""
        probs = model.predict_proba(X)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y, probs)
        
        if metric == 'f1':
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
            best_idx = np.argmax(f1_scores)
        elif metric == 'precision':
            best_idx = np.argmax(precision)
        elif metric == 'recall':
            best_idx = np.argmax(recall)
        else:
            logger.warning(f"Unknown metric {metric}, using F1")
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
            best_idx = np.argmax(f1_scores)
        
        return thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    
    def get_feature_importance(
        self,
        horizon: int,
        segment: str = 'all',
        importance_type: str = 'gain'
    ) -> pd.DataFrame:
        """
        Get feature importance for a specific horizon and segment.
        
        Args:
            horizon: Time horizon
            segment: Customer segment
            importance_type: 'gain', 'split', or 'shap'
            
        Returns:
            DataFrame with features and importance scores
            
        Example:
            >>> importance = model.get_feature_importance(horizon=30, importance_type='gain')
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if horizon not in self.models or segment not in self.models[horizon]:
            raise ValueError(f"Model not found for horizon {horizon}, segment {segment}")
        
        model = self.models[horizon][segment]
        
        # Handle calibrated models
        if isinstance(model, CalibratedClassifierCV):
            base_model = model.calibrated_classifiers_[0].base_estimator
        else:
            base_model = model
        
        if importance_type == 'shap':
            try:
                import shap
                explainer = shap.TreeExplainer(base_model)
                # Use a sample for SHAP (can be expensive)
                shap_values = explainer.shap_values(X[:1000] if len(X) > 1000 else X)
                importance = np.abs(shap_values).mean(axis=0)
            except Exception as e:
                logger.warning(f"SHAP calculation failed: {e}, falling back to gain")
                importance = base_model.feature_importances_
        else:
            importance = base_model.feature_importances_
        
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return feature_importance
    
    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save model to disk.
        
        Args:
            filepath: Path to save model
            
        Example:
            >>> model.save("models/propensity_model.pkl")
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        save_dict = {
            'models': self.models,
            'thresholds': self.thresholds,
            'feature_names': self.feature_names,
            'config': self.config,
            'segment_col': self.segment_col,
            'calibrate': self.calibrate,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(save_dict, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'PropensityModel':
        """
        Load model from disk.
        
        Args:
            filepath: Path to saved model
            
        Returns:
            Loaded PropensityModel instance
            
        Example:
            >>> model = PropensityModel.load("models/propensity_model.pkl")
        """
        save_dict = joblib.load(filepath)
        
        instance = cls(segment_col=save_dict['segment_col'], calibrate=save_dict['calibrate'])
        instance.models = save_dict['models']
        instance.thresholds = save_dict['thresholds']
        instance.feature_names = save_dict['feature_names']
        instance.config = save_dict['config']
        instance.is_fitted = save_dict['is_fitted']
        
        logger.info(f"Model loaded from {filepath}")
        return instance
