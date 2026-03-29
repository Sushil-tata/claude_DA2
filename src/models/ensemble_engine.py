"""
Ensemble methods for combining multiple models.

This module provides ensemble techniques including weighted averaging, stacking,
blending, segment-wise ensembles, and hybrid rule-ML combinations.

Example:
    >>> from src.models.ensemble_engine import WeightedAverageEnsemble
    >>> ensemble = WeightedAverageEnsemble(models=[model1, model2, model3])
    >>> ensemble.fit(X_val, y_val)
    >>> predictions = ensemble.predict_proba(X_test)
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from loguru import logger
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, mean_squared_error

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    logger.warning("Optuna not available. Install with: pip install optuna")
    OPTUNA_AVAILABLE = False


class BaseEnsemble:
    """Base class for ensemble methods."""
    
    def __init__(self, models: List[Any], task: str = 'classification'):
        """
        Initialize base ensemble.
        
        Args:
            models: List of fitted models with predict/predict_proba methods
            task: 'classification' or 'regression'
        """
        self.models = models
        self.task = task
        self.is_fitted = False
        self.weights: Optional[np.ndarray] = None
    
    def save(self, filepath: Union[str, Path]) -> None:
        """Save ensemble to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        ensemble_data = {
            'models': self.models,
            'task': self.task,
            'is_fitted': self.is_fitted,
            'weights': self.weights
        }
        
        joblib.dump(ensemble_data, filepath)
        logger.info(f"Ensemble saved to {filepath}")
    
    def load(self, filepath: Union[str, Path]) -> 'BaseEnsemble':
        """Load ensemble from disk."""
        ensemble_data = joblib.load(filepath)
        self.models = ensemble_data['models']
        self.task = ensemble_data['task']
        self.is_fitted = ensemble_data['is_fitted']
        self.weights = ensemble_data['weights']
        
        logger.info(f"Ensemble loaded from {filepath}")
        return self


class WeightedAverageEnsemble(BaseEnsemble):
    """
    Weighted average ensemble with optimized weights.
    
    Uses Optuna to find optimal weights that maximize performance on validation set.
    
    Example:
        >>> models = [lgb_model, xgb_model, cat_model]
        >>> ensemble = WeightedAverageEnsemble(models=models)
        >>> ensemble.fit(X_val, y_val, n_trials=100)
        >>> preds = ensemble.predict_proba(X_test)
    """
    
    def __init__(
        self,
        models: List[Any],
        task: str = 'classification',
        metric: str = 'auc'
    ):
        """
        Initialize weighted average ensemble.
        
        Args:
            models: List of fitted models
            task: 'classification' or 'regression'
            metric: Optimization metric ('auc', 'rmse', etc.)
        """
        super().__init__(models, task)
        self.metric = metric
    
    def fit(
        self,
        X_val: Union[pd.DataFrame, np.ndarray],
        y_val: Union[pd.Series, np.ndarray],
        n_trials: int = 100,
        timeout: Optional[int] = None
    ) -> 'WeightedAverageEnsemble':
        """
        Optimize ensemble weights using Optuna.
        
        Args:
            X_val: Validation features
            y_val: Validation target
            n_trials: Number of Optuna trials
            timeout: Optimization timeout in seconds
            
        Returns:
            Self
        """
        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna not available. Using equal weights.")
            self.weights = np.ones(len(self.models)) / len(self.models)
            self.is_fitted = True
            return self
        
        # Get predictions from all models
        predictions = self._get_model_predictions(X_val)
        
        def objective(trial):
            # Sample weights from Dirichlet distribution (ensures they sum to 1)
            weights = np.array([
                trial.suggest_float(f'weight_{i}', 0.0, 1.0)
                for i in range(len(self.models))
            ])
            weights = weights / weights.sum()
            
            # Compute weighted average
            ensemble_pred = np.average(predictions, axis=0, weights=weights)
            
            # Calculate metric
            if self.task == 'classification':
                if self.metric == 'auc':
                    score = roc_auc_score(y_val, ensemble_pred)
                else:
                    score = roc_auc_score(y_val, ensemble_pred)
            else:
                if self.metric == 'rmse':
                    score = -np.sqrt(mean_squared_error(y_val, ensemble_pred))
                else:
                    score = -mean_squared_error(y_val, ensemble_pred)
            
            return score
        
        logger.info("Optimizing ensemble weights with Optuna...")
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
        
        # Extract best weights
        best_params = study.best_params
        self.weights = np.array([best_params[f'weight_{i}'] for i in range(len(self.models))])
        self.weights = self.weights / self.weights.sum()
        
        logger.info(f"Best weights: {self.weights}")
        logger.info(f"Best score: {study.best_value:.6f}")
        
        self.is_fitted = True
        return self
    
    def _get_model_predictions(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """Get predictions from all models."""
        predictions = []
        for model in self.models:
            if self.task == 'classification':
                if hasattr(model, 'predict_proba'):
                    pred = model.predict_proba(X)
                    # Handle binary classification
                    if pred.ndim == 2 and pred.shape[1] == 2:
                        pred = pred[:, 1]
                else:
                    pred = model.predict(X)
            else:
                pred = model.predict(X)
            predictions.append(pred)
        
        return np.array(predictions)
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict class probabilities using weighted average."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
        
        predictions = self._get_model_predictions(X)
        ensemble_pred = np.average(predictions, axis=0, weights=self.weights)
        
        # Return in scikit-learn format for binary classification
        if self.task == 'classification' and ensemble_pred.ndim == 1:
            return np.column_stack([1 - ensemble_pred, ensemble_pred])
        
        return ensemble_pred
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict class labels or regression values."""
        if self.task == 'classification':
            proba = self.predict_proba(X)
            return (proba[:, 1] >= 0.5).astype(int)
        else:
            predictions = self._get_model_predictions(X)
            return np.average(predictions, axis=0, weights=self.weights)


class StackingEnsemble(BaseEnsemble):
    """
    Stacking ensemble with cross-validated base predictions.
    
    Uses cross-validation to generate out-of-fold predictions from base models,
    then trains a meta-learner on these predictions.
    
    Example:
        >>> models = [lgb_model, xgb_model, cat_model]
        >>> ensemble = StackingEnsemble(models=models, meta_learner='logistic')
        >>> ensemble.fit(X_train, y_train, cv=5)
        >>> preds = ensemble.predict_proba(X_test)
    """
    
    def __init__(
        self,
        models: List[Any],
        task: str = 'classification',
        meta_learner: str = 'logistic',
        cv: int = 5,
        use_probas: bool = True
    ):
        """
        Initialize stacking ensemble.
        
        Args:
            models: List of base models (should not be fitted)
            task: 'classification' or 'regression'
            meta_learner: Meta-learner type ('logistic', 'ridge', etc.)
            cv: Number of cross-validation folds
            use_probas: Use probabilities (True) or predictions (False)
        """
        super().__init__(models, task)
        self.meta_learner_name = meta_learner
        self.cv = cv
        self.use_probas = use_probas
        self.meta_model = None
        self.base_models: List[List[Any]] = []
    
    def fit(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        **kwargs
    ) -> 'StackingEnsemble':
        """
        Fit stacking ensemble with cross-validation.
        
        Args:
            X_train: Training features
            y_train: Training target
            **kwargs: Additional arguments for base models
            
        Returns:
            Self
        """
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        if isinstance(y_train, pd.Series):
            y_train = y_train.values
        
        # Initialize cross-validation
        if self.task == 'classification':
            kf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=42)
        else:
            kf = KFold(n_splits=self.cv, shuffle=True, random_state=42)
        
        # Generate out-of-fold predictions
        logger.info("Generating out-of-fold predictions for stacking...")
        oof_predictions = np.zeros((X_train.shape[0], len(self.models)))
        
        self.base_models = [[] for _ in range(len(self.models))]
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
            logger.info(f"Training fold {fold + 1}/{self.cv}")
            
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
            
            for i, model in enumerate(self.models):
                # Clone and fit model
                from copy import deepcopy
                model_clone = deepcopy(model)
                model_clone.fit(X_fold_train, y_fold_train)
                self.base_models[i].append(model_clone)
                
                # Get out-of-fold predictions
                if self.task == 'classification' and self.use_probas:
                    pred = model_clone.predict_proba(X_fold_val)
                    if pred.ndim == 2 and pred.shape[1] == 2:
                        pred = pred[:, 1]
                else:
                    pred = model_clone.predict(X_fold_val)
                
                oof_predictions[val_idx, i] = pred
        
        # Train meta-learner on out-of-fold predictions
        logger.info("Training meta-learner...")
        if self.task == 'classification':
            if self.meta_learner_name == 'logistic':
                self.meta_model = LogisticRegression(max_iter=1000)
            else:
                self.meta_model = LogisticRegression(max_iter=1000)
        else:
            if self.meta_learner_name == 'ridge':
                self.meta_model = Ridge()
            else:
                self.meta_model = Ridge()
        
        self.meta_model.fit(oof_predictions, y_train)
        self.is_fitted = True
        
        logger.info("Stacking ensemble training complete")
        return self
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict class probabilities using stacking."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Get predictions from all base models (averaged across folds)
        meta_features = np.zeros((X.shape[0], len(self.models)))
        
        for i, fold_models in enumerate(self.base_models):
            fold_preds = []
            for model in fold_models:
                if self.task == 'classification' and self.use_probas:
                    pred = model.predict_proba(X)
                    if pred.ndim == 2 and pred.shape[1] == 2:
                        pred = pred[:, 1]
                else:
                    pred = model.predict(X)
                fold_preds.append(pred)
            meta_features[:, i] = np.mean(fold_preds, axis=0)
        
        # Meta-learner prediction
        if self.task == 'classification':
            return self.meta_model.predict_proba(meta_features)
        else:
            return self.meta_model.predict(meta_features)
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict class labels or regression values."""
        if self.task == 'classification':
            proba = self.predict_proba(X)
            return (proba[:, 1] >= 0.5).astype(int)
        else:
            return self.predict_proba(X)


class BlendingEnsemble(BaseEnsemble):
    """
    Blending ensemble using hold-out validation set.
    
    Simpler than stacking - trains base models on training set and meta-learner
    on validation set predictions.
    
    Example:
        >>> models = [lgb_model, xgb_model]
        >>> ensemble = BlendingEnsemble(models=models)
        >>> ensemble.fit(X_train, y_train, X_val, y_val)
        >>> preds = ensemble.predict_proba(X_test)
    """
    
    def __init__(
        self,
        models: List[Any],
        task: str = 'classification',
        meta_learner: str = 'logistic'
    ):
        """Initialize blending ensemble."""
        super().__init__(models, task)
        self.meta_learner_name = meta_learner
        self.meta_model = None
    
    def fit(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_val: Union[pd.DataFrame, np.ndarray],
        y_val: Union[pd.Series, np.ndarray],
        **kwargs
    ) -> 'BlendingEnsemble':
        """
        Fit blending ensemble.
        
        Args:
            X_train: Training features for base models
            y_train: Training target for base models
            X_val: Validation features for meta-learner
            y_val: Validation target for meta-learner
            
        Returns:
            Self
        """
        logger.info("Training base models for blending...")
        
        # Train base models on training set
        for i, model in enumerate(self.models):
            logger.info(f"Training base model {i + 1}/{len(self.models)}")
            model.fit(X_train, y_train, **kwargs)
        
        # Get predictions on validation set
        val_predictions = []
        for model in self.models:
            if self.task == 'classification':
                pred = model.predict_proba(X_val)
                if pred.ndim == 2 and pred.shape[1] == 2:
                    pred = pred[:, 1]
            else:
                pred = model.predict(X_val)
            val_predictions.append(pred)
        
        val_predictions = np.column_stack(val_predictions)
        
        # Train meta-learner
        logger.info("Training meta-learner...")
        if self.task == 'classification':
            self.meta_model = LogisticRegression(max_iter=1000)
        else:
            self.meta_model = Ridge()
        
        self.meta_model.fit(val_predictions, y_val)
        self.is_fitted = True
        
        logger.info("Blending ensemble training complete")
        return self
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict class probabilities using blending."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
        
        # Get predictions from base models
        predictions = []
        for model in self.models:
            if self.task == 'classification':
                pred = model.predict_proba(X)
                if pred.ndim == 2 and pred.shape[1] == 2:
                    pred = pred[:, 1]
            else:
                pred = model.predict(X)
            predictions.append(pred)
        
        predictions = np.column_stack(predictions)
        
        # Meta-learner prediction
        if self.task == 'classification':
            return self.meta_model.predict_proba(predictions)
        else:
            return self.meta_model.predict(predictions)
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict class labels or regression values."""
        if self.task == 'classification':
            proba = self.predict_proba(X)
            return (proba[:, 1] >= 0.5).astype(int)
        else:
            return self.predict_proba(X)


class SegmentWiseEnsemble:
    """
    Segment-wise ensemble using different models for different segments.
    
    Automatically identifies segments or uses provided segment definitions,
    then trains specialized models for each segment.
    
    Example:
        >>> ensemble = SegmentWiseEnsemble(segment_col='customer_segment')
        >>> ensemble.fit(X_train, y_train, models={'A': model_a, 'B': model_b})
        >>> preds = ensemble.predict_proba(X_test)
    """
    
    def __init__(
        self,
        segment_col: Optional[str] = None,
        task: str = 'classification'
    ):
        """
        Initialize segment-wise ensemble.
        
        Args:
            segment_col: Column name for segmentation
            task: 'classification' or 'regression'
        """
        self.segment_col = segment_col
        self.task = task
        self.segment_models: Dict[Any, Any] = {}
        self.is_fitted = False
    
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: Union[pd.Series, np.ndarray],
        models: Dict[Any, Any],
        **kwargs
    ) -> 'SegmentWiseEnsemble':
        """
        Fit segment-specific models.
        
        Args:
            X_train: Training features (must include segment_col)
            y_train: Training target
            models: Dictionary mapping segment values to models
            
        Returns:
            Self
        """
        if self.segment_col not in X_train.columns:
            raise ValueError(f"Segment column '{self.segment_col}' not found in X_train")
        
        segments = X_train[self.segment_col].unique()
        
        logger.info(f"Training models for {len(segments)} segments...")
        
        for segment in segments:
            if segment not in models:
                logger.warning(f"No model provided for segment '{segment}', skipping")
                continue
            
            # Get segment data
            segment_mask = X_train[self.segment_col] == segment
            X_segment = X_train[segment_mask].drop(columns=[self.segment_col])
            y_segment = y_train[segment_mask] if isinstance(y_train, pd.Series) else y_train[segment_mask]
            
            logger.info(f"Training model for segment '{segment}' ({len(X_segment)} samples)")
            
            # Train segment model
            model = models[segment]
            model.fit(X_segment, y_segment, **kwargs)
            self.segment_models[segment] = model
        
        self.is_fitted = True
        logger.info("Segment-wise ensemble training complete")
        return self
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using segment-specific models."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
        
        if self.segment_col not in X.columns:
            raise ValueError(f"Segment column '{self.segment_col}' not found in X")
        
        predictions = np.zeros((len(X), 2)) if self.task == 'classification' else np.zeros(len(X))
        
        for segment, model in self.segment_models.items():
            segment_mask = X[self.segment_col] == segment
            if segment_mask.sum() == 0:
                continue
            
            X_segment = X[segment_mask].drop(columns=[self.segment_col])
            
            if self.task == 'classification':
                predictions[segment_mask] = model.predict_proba(X_segment)
            else:
                predictions[segment_mask] = model.predict(X_segment)
        
        return predictions
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels or regression values."""
        if self.task == 'classification':
            proba = self.predict_proba(X)
            return (proba[:, 1] >= 0.5).astype(int)
        else:
            return self.predict_proba(X)


class HybridRuleMLEnsemble:
    """
    Hybrid ensemble combining rule-based logic with ML predictions.
    
    Uses business rules for high-confidence cases and ML models for uncertain cases.
    
    Example:
        >>> def rule_fn(X):
        ...     # Return (predictions, mask) where mask=True for rule-based cases
        ...     high_risk = X['amount'] > 10000
        ...     return high_risk.astype(int), high_risk
        >>> 
        >>> ensemble = HybridRuleMLEnsemble(ml_model=lgb_model, rule_function=rule_fn)
        >>> ensemble.fit(X_train, y_train)
        >>> preds = ensemble.predict_proba(X_test)
    """
    
    def __init__(
        self,
        ml_model: Any,
        rule_function: Any,
        task: str = 'classification'
    ):
        """
        Initialize hybrid ensemble.
        
        Args:
            ml_model: Machine learning model
            rule_function: Function that returns (predictions, mask)
            task: 'classification' or 'regression'
        """
        self.ml_model = ml_model
        self.rule_function = rule_function
        self.task = task
        self.is_fitted = False
    
    def fit(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        **kwargs
    ) -> 'HybridRuleMLEnsemble':
        """Fit ML model on non-rule-based cases."""
        # Apply rules
        rule_preds, rule_mask = self.rule_function(X_train)
        
        # Train ML model on cases not covered by rules
        if isinstance(X_train, pd.DataFrame):
            X_ml = X_train[~rule_mask]
        else:
            X_ml = X_train[~rule_mask]
        
        y_ml = y_train[~rule_mask] if isinstance(y_train, pd.Series) else y_train[~rule_mask]
        
        logger.info(f"Training ML model on {len(X_ml)} samples (rule-based: {rule_mask.sum()})")
        self.ml_model.fit(X_ml, y_ml, **kwargs)
        
        self.is_fitted = True
        return self
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict using hybrid approach."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
        
        # Apply rules
        rule_preds, rule_mask = self.rule_function(X)
        
        # Initialize predictions
        if self.task == 'classification':
            predictions = np.zeros((len(X), 2))
            predictions[rule_mask, 1] = rule_preds[rule_mask]
            predictions[rule_mask, 0] = 1 - rule_preds[rule_mask]
        else:
            predictions = np.zeros(len(X))
            predictions[rule_mask] = rule_preds[rule_mask]
        
        # ML predictions for non-rule cases
        if (~rule_mask).sum() > 0:
            if isinstance(X, pd.DataFrame):
                X_ml = X[~rule_mask]
            else:
                X_ml = X[~rule_mask]
            
            if self.task == 'classification':
                predictions[~rule_mask] = self.ml_model.predict_proba(X_ml)
            else:
                predictions[~rule_mask] = self.ml_model.predict(X_ml)
        
        return predictions
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict class labels or regression values."""
        if self.task == 'classification':
            proba = self.predict_proba(X)
            return (proba[:, 1] >= 0.5).astype(int)
        else:
            return self.predict_proba(X)


def compare_ensemble_performance(
    ensembles: Dict[str, BaseEnsemble],
    X_test: Union[pd.DataFrame, np.ndarray],
    y_test: Union[pd.Series, np.ndarray],
    metrics: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Compare performance of multiple ensembles.
    
    Args:
        ensembles: Dictionary of ensemble name -> ensemble object
        X_test: Test features
        y_test: Test target
        metrics: List of metrics to compute
        
    Returns:
        DataFrame with performance comparison
    """
    if metrics is None:
        metrics = ['auc', 'accuracy']
    
    results = []
    
    for name, ensemble in ensembles.items():
        logger.info(f"Evaluating {name}...")
        
        result = {'ensemble': name}
        
        # Get predictions
        y_pred_proba = ensemble.predict_proba(X_test)
        if y_pred_proba.ndim == 2:
            y_pred_proba = y_pred_proba[:, 1]
        
        # Compute metrics
        if 'auc' in metrics:
            from sklearn.metrics import roc_auc_score
            result['auc'] = roc_auc_score(y_test, y_pred_proba)
        
        if 'accuracy' in metrics:
            from sklearn.metrics import accuracy_score
            y_pred = (y_pred_proba >= 0.5).astype(int)
            result['accuracy'] = accuracy_score(y_test, y_pred)
        
        results.append(result)
    
    df = pd.DataFrame(results).sort_values('auc', ascending=False)
    return df
