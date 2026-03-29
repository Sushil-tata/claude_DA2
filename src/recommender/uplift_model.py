"""
Causal uplift modeling for treatment effect estimation.

This module provides uplift modeling techniques to estimate heterogeneous
treatment effects, enabling targeted interventions and personalized treatments.

Example:
    >>> from src.recommender.uplift_model import TLearner
    >>> uplift = TLearner(base_model=LGBMClassifier())
    >>> uplift.fit(X, treatment, y)
    >>> cate = uplift.predict_uplift(X_test)
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from loguru import logger
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, mean_squared_error

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    logger.warning("LightGBM not available")
    LIGHTGBM_AVAILABLE = False

try:
    from econml.dml import CausalForestDML
    ECONML_AVAILABLE = True
except ImportError:
    logger.warning("EconML not available. Install with: pip install econml")
    ECONML_AVAILABLE = False


class BaseUpliftModel:
    """Base class for uplift models."""
    
    def __init__(self, task: str = 'classification'):
        """
        Initialize base uplift model.
        
        Args:
            task: 'classification' or 'regression'
        """
        self.task = task
        self.is_fitted = False
        
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        treatment: np.ndarray,
        y: np.ndarray,
        **kwargs
    ) -> 'BaseUpliftModel':
        """Fit uplift model."""
        raise NotImplementedError
        
    def predict_uplift(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """Predict conditional average treatment effect (CATE)."""
        raise NotImplementedError
        
    def save(self, filepath: Union[str, Path]) -> None:
        """Save model to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        logger.info(f"Uplift model saved to {filepath}")
        
    @staticmethod
    def load(filepath: Union[str, Path]) -> 'BaseUpliftModel':
        """Load model from disk."""
        model = joblib.load(filepath)
        logger.info(f"Uplift model loaded from {filepath}")
        return model


class TLearner(BaseUpliftModel):
    """
    T-Learner (Two-Model) approach for uplift modeling.
    
    Trains separate models for treatment and control groups, then estimates
    uplift as the difference in predictions.
    
    Example:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> uplift = TLearner(base_model=RandomForestClassifier())
        >>> uplift.fit(X_train, treatment_train, y_train)
        >>> cate = uplift.predict_uplift(X_test)
    """
    
    def __init__(
        self,
        base_model: Optional[BaseEstimator] = None,
        task: str = 'classification'
    ):
        """
        Initialize T-Learner.
        
        Args:
            base_model: Scikit-learn compatible model (cloned for T/C groups)
            task: 'classification' or 'regression'
        """
        super().__init__(task)
        
        if base_model is None:
            if task == 'classification':
                base_model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                base_model = RandomForestRegressor(n_estimators=100, random_state=42)
                
        self.base_model = base_model
        self.model_treatment = None
        self.model_control = None
        
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        treatment: np.ndarray,
        y: np.ndarray,
        **kwargs
    ) -> 'TLearner':
        """
        Fit T-Learner models.
        
        Args:
            X: Features
            treatment: Binary treatment indicator (1=treatment, 0=control)
            y: Target variable
            **kwargs: Additional arguments for base model fit
            
        Returns:
            Fitted T-Learner
        """
        X = self._convert_to_array(X)
        treatment = np.asarray(treatment)
        y = np.asarray(y)
        
        # Validate inputs
        if len(np.unique(treatment)) != 2:
            raise ValueError("Treatment must be binary (0/1)")
            
        # Split data by treatment
        treated_idx = treatment == 1
        control_idx = treatment == 0
        
        X_treatment = X[treated_idx]
        y_treatment = y[treated_idx]
        X_control = X[control_idx]
        y_control = y[control_idx]
        
        logger.info(
            f"Fitting T-Learner: {treated_idx.sum()} treated, "
            f"{control_idx.sum()} control samples"
        )
        
        # Train separate models
        self.model_treatment = clone(self.base_model)
        self.model_control = clone(self.base_model)
        
        self.model_treatment.fit(X_treatment, y_treatment, **kwargs)
        self.model_control.fit(X_control, y_control, **kwargs)
        
        self.is_fitted = True
        logger.info("T-Learner fitted successfully")
        
        return self
        
    def predict_uplift(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        Predict CATE (uplift) for each sample.
        
        Args:
            X: Features
            
        Returns:
            Array of uplift estimates
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        X = self._convert_to_array(X)
        
        # Predict for both scenarios
        if self.task == 'classification':
            pred_treatment = self.model_treatment.predict_proba(X)[:, 1]
            pred_control = self.model_control.predict_proba(X)[:, 1]
        else:
            pred_treatment = self.model_treatment.predict(X)
            pred_control = self.model_control.predict(X)
            
        # Uplift is difference
        uplift = pred_treatment - pred_control
        
        return uplift
        
    def predict_treatment_response(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict responses under treatment and control.
        
        Returns:
            Tuple of (treatment_response, control_response)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        X = self._convert_to_array(X)
        
        if self.task == 'classification':
            y_t = self.model_treatment.predict_proba(X)[:, 1]
            y_c = self.model_control.predict_proba(X)[:, 1]
        else:
            y_t = self.model_treatment.predict(X)
            y_c = self.model_control.predict(X)
            
        return y_t, y_c
        
    @staticmethod
    def _convert_to_array(X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Convert input to numpy array."""
        if isinstance(X, pd.DataFrame):
            return X.values
        return np.asarray(X)


class SLearner(BaseUpliftModel):
    """
    S-Learner (Single-Model) approach for uplift modeling.
    
    Trains a single model with treatment as a feature, then estimates uplift
    by comparing predictions with treatment on/off.
    
    Example:
        >>> uplift = SLearner(base_model=LGBMClassifier())
        >>> uplift.fit(X_train, treatment_train, y_train)
        >>> cate = uplift.predict_uplift(X_test)
    """
    
    def __init__(
        self,
        base_model: Optional[BaseEstimator] = None,
        task: str = 'classification'
    ):
        """
        Initialize S-Learner.
        
        Args:
            base_model: Scikit-learn compatible model
            task: 'classification' or 'regression'
        """
        super().__init__(task)
        
        if base_model is None:
            if task == 'classification':
                base_model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                base_model = RandomForestRegressor(n_estimators=100, random_state=42)
                
        self.base_model = base_model
        self.model = None
        
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        treatment: np.ndarray,
        y: np.ndarray,
        **kwargs
    ) -> 'SLearner':
        """
        Fit S-Learner model.
        
        Args:
            X: Features
            treatment: Binary treatment indicator
            y: Target variable
            **kwargs: Additional arguments for base model fit
            
        Returns:
            Fitted S-Learner
        """
        X = self._convert_to_array(X)
        treatment = np.asarray(treatment).reshape(-1, 1)
        y = np.asarray(y)
        
        # Concatenate treatment as feature
        X_with_treatment = np.concatenate([X, treatment], axis=1)
        
        logger.info(f"Fitting S-Learner with {X.shape[0]} samples")
        
        self.model = clone(self.base_model)
        self.model.fit(X_with_treatment, y, **kwargs)
        
        self.is_fitted = True
        logger.info("S-Learner fitted successfully")
        
        return self
        
    def predict_uplift(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """Predict CATE (uplift) for each sample."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        X = self._convert_to_array(X)
        n_samples = X.shape[0]
        
        # Predict with treatment = 1
        X_treatment = np.concatenate([X, np.ones((n_samples, 1))], axis=1)
        # Predict with treatment = 0
        X_control = np.concatenate([X, np.zeros((n_samples, 1))], axis=1)
        
        if self.task == 'classification':
            pred_treatment = self.model.predict_proba(X_treatment)[:, 1]
            pred_control = self.model.predict_proba(X_control)[:, 1]
        else:
            pred_treatment = self.model.predict(X_treatment)
            pred_control = self.model.predict(X_control)
            
        uplift = pred_treatment - pred_control
        return uplift
        
    @staticmethod
    def _convert_to_array(X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Convert input to numpy array."""
        if isinstance(X, pd.DataFrame):
            return X.values
        return np.asarray(X)


class XLearner(BaseUpliftModel):
    """
    X-Learner approach with propensity score weighting.
    
    Advanced meta-learner that models treatment effects directly and uses
    propensity scores for efficient estimation.
    
    Example:
        >>> uplift = XLearner(base_model=RandomForestRegressor())
        >>> uplift.fit(X_train, treatment_train, y_train)
        >>> cate = uplift.predict_uplift(X_test)
    """
    
    def __init__(
        self,
        base_model: Optional[BaseEstimator] = None,
        propensity_model: Optional[BaseEstimator] = None,
        task: str = 'classification'
    ):
        """
        Initialize X-Learner.
        
        Args:
            base_model: Model for response estimation
            propensity_model: Model for propensity score (default: LogisticRegression)
            task: 'classification' or 'regression'
        """
        super().__init__(task)
        
        if base_model is None:
            base_model = RandomForestRegressor(n_estimators=100, random_state=42)
            
        if propensity_model is None:
            from sklearn.linear_model import LogisticRegression
            propensity_model = LogisticRegression(random_state=42)
            
        self.base_model = base_model
        self.propensity_model = propensity_model
        
        # Stage 1 models
        self.model_treatment = None
        self.model_control = None
        
        # Stage 2 models (for imputed treatment effects)
        self.model_tau_treatment = None
        self.model_tau_control = None
        
        # Propensity model
        self.propensity = None
        
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        treatment: np.ndarray,
        y: np.ndarray,
        **kwargs
    ) -> 'XLearner':
        """
        Fit X-Learner in three stages.
        
        Args:
            X: Features
            treatment: Binary treatment indicator
            y: Target variable
            
        Returns:
            Fitted X-Learner
        """
        X = self._convert_to_array(X)
        treatment = np.asarray(treatment)
        y = np.asarray(y)
        
        treated_idx = treatment == 1
        control_idx = treatment == 0
        
        X_treatment = X[treated_idx]
        y_treatment = y[treated_idx]
        X_control = X[control_idx]
        y_control = y[control_idx]
        
        logger.info("X-Learner Stage 1: Fitting response models")
        
        # Stage 1: Estimate response functions
        self.model_treatment = clone(self.base_model)
        self.model_control = clone(self.base_model)
        
        self.model_treatment.fit(X_treatment, y_treatment)
        self.model_control.fit(X_control, y_control)
        
        # Stage 2: Impute treatment effects
        logger.info("X-Learner Stage 2: Imputing treatment effects")
        
        # For treated: impute control outcome and calculate difference
        y_control_imputed_t = self.model_control.predict(X_treatment)
        tau_treatment = y_treatment - y_control_imputed_t
        
        # For control: impute treatment outcome and calculate difference
        y_treatment_imputed_c = self.model_treatment.predict(X_control)
        tau_control = y_treatment_imputed_c - y_control
        
        # Fit models on imputed treatment effects
        self.model_tau_treatment = clone(self.base_model)
        self.model_tau_control = clone(self.base_model)
        
        self.model_tau_treatment.fit(X_treatment, tau_treatment)
        self.model_tau_control.fit(X_control, tau_control)
        
        # Stage 3: Estimate propensity scores
        logger.info("X-Learner Stage 3: Estimating propensity scores")
        
        self.propensity = clone(self.propensity_model)
        self.propensity.fit(X, treatment)
        
        self.is_fitted = True
        logger.info("X-Learner fitted successfully")
        
        return self
        
    def predict_uplift(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """Predict CATE using propensity-weighted combination."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        X = self._convert_to_array(X)
        
        # Get propensity scores
        propensity_scores = self.propensity.predict_proba(X)[:, 1]
        
        # Predict treatment effects from both models
        tau_t = self.model_tau_treatment.predict(X)
        tau_c = self.model_tau_control.predict(X)
        
        # Weighted combination using propensity scores
        uplift = propensity_scores * tau_c + (1 - propensity_scores) * tau_t
        
        return uplift
        
    def get_propensity_scores(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """Get estimated propensity scores."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        X = self._convert_to_array(X)
        return self.propensity.predict_proba(X)[:, 1]
        
    @staticmethod
    def _convert_to_array(X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Convert input to numpy array."""
        if isinstance(X, pd.DataFrame):
            return X.values
        return np.asarray(X)


class CausalForestUplift(BaseUpliftModel):
    """
    Causal Forest for uplift modeling using EconML.
    
    Random forest specifically designed for heterogeneous treatment effects.
    
    Example:
        >>> uplift = CausalForestUplift(n_estimators=100)
        >>> uplift.fit(X_train, treatment_train, y_train)
        >>> cate = uplift.predict_uplift(X_test)
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_leaf: int = 5,
        random_state: Optional[int] = None
    ):
        """
        Initialize Causal Forest.
        
        Args:
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            min_samples_leaf: Minimum samples per leaf
            random_state: Random seed
        """
        super().__init__()
        
        if not ECONML_AVAILABLE:
            raise ImportError(
                "EconML required for CausalForest. "
                "Install with: pip install econml"
            )
            
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.model = None
        
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        treatment: np.ndarray,
        y: np.ndarray,
        **kwargs
    ) -> 'CausalForestUplift':
        """
        Fit Causal Forest.
        
        Args:
            X: Features
            treatment: Binary treatment indicator
            y: Target variable
            
        Returns:
            Fitted Causal Forest
        """
        from sklearn.ensemble import GradientBoostingRegressor
        
        X = self._convert_to_array(X)
        treatment = np.asarray(treatment).reshape(-1, 1)
        y = np.asarray(y)
        
        logger.info(f"Fitting Causal Forest with {X.shape[0]} samples")
        
        # Initialize Causal Forest DML
        self.model = CausalForestDML(
            model_y=GradientBoostingRegressor(random_state=self.random_state),
            model_t=GradientBoostingRegressor(random_state=self.random_state),
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state
        )
        
        self.model.fit(y, treatment, X=X)
        
        self.is_fitted = True
        logger.info("Causal Forest fitted successfully")
        
        return self
        
    def predict_uplift(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """Predict CATE."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        X = self._convert_to_array(X)
        uplift = self.model.effect(X)
        
        return uplift.flatten()
        
    def get_confidence_intervals(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        alpha: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get confidence intervals for CATE.
        
        Args:
            X: Features
            alpha: Significance level
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        X = self._convert_to_array(X)
        intervals = self.model.effect_interval(X, alpha=alpha)
        
        return intervals[0].flatten(), intervals[1].flatten()
        
    @staticmethod
    def _convert_to_array(X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Convert input to numpy array."""
        if isinstance(X, pd.DataFrame):
            return X.values
        return np.asarray(X)


class UpliftEnsemble:
    """
    Ensemble of multiple uplift models.
    
    Combines predictions from multiple uplift models for robust estimates.
    
    Example:
        >>> models = [
        ...     TLearner(RandomForestClassifier()),
        ...     SLearner(LGBMClassifier()),
        ...     XLearner(RandomForestRegressor())
        ... ]
        >>> ensemble = UpliftEnsemble(models, weights='auto')
        >>> ensemble.fit(X_train, treatment_train, y_train)
        >>> cate = ensemble.predict_uplift(X_test)
    """
    
    def __init__(
        self,
        models: List[BaseUpliftModel],
        weights: Union[str, List[float]] = 'uniform',
        task: str = 'classification'
    ):
        """
        Initialize uplift ensemble.
        
        Args:
            models: List of uplift models
            weights: 'uniform', 'auto', or list of weights
            task: 'classification' or 'regression'
        """
        self.models = models
        self.weights_type = weights
        self.task = task
        self.weights: Optional[np.ndarray] = None
        self.is_fitted = False
        
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        treatment: np.ndarray,
        y: np.ndarray,
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        treatment_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs
    ) -> 'UpliftEnsemble':
        """
        Fit all models in ensemble.
        
        Args:
            X: Features
            treatment: Binary treatment indicator
            y: Target variable
            X_val: Validation features for weight optimization
            treatment_val: Validation treatment
            y_val: Validation target
            
        Returns:
            Fitted ensemble
        """
        logger.info(f"Fitting {len(self.models)} uplift models")
        
        # Fit each model
        for i, model in enumerate(self.models):
            logger.info(f"Fitting model {i+1}/{len(self.models)}: {type(model).__name__}")
            model.fit(X, treatment, y, **kwargs)
            
        # Set weights
        if self.weights_type == 'uniform':
            self.weights = np.ones(len(self.models)) / len(self.models)
        elif self.weights_type == 'auto':
            if X_val is not None and treatment_val is not None and y_val is not None:
                self._optimize_weights(X_val, treatment_val, y_val)
            else:
                logger.warning("Auto weights requires validation set, using uniform")
                self.weights = np.ones(len(self.models)) / len(self.models)
        else:
            self.weights = np.array(self.weights_type)
            if len(self.weights) != len(self.models):
                raise ValueError("Number of weights must match number of models")
            self.weights = self.weights / self.weights.sum()
            
        self.is_fitted = True
        logger.info(f"Ensemble fitted with weights: {self.weights}")
        
        return self
        
    def predict_uplift(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """Predict weighted average of uplift predictions."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
            
        predictions = np.array([
            model.predict_uplift(X) for model in self.models
        ])
        
        weighted_pred = np.average(predictions, axis=0, weights=self.weights)
        return weighted_pred
        
    def _optimize_weights(
        self,
        X_val: Union[pd.DataFrame, np.ndarray],
        treatment_val: np.ndarray,
        y_val: np.ndarray
    ) -> None:
        """Optimize ensemble weights using validation set."""
        from scipy.optimize import minimize
        
        # Get predictions from all models
        uplift_preds = np.array([
            model.predict_uplift(X_val) for model in self.models
        ])
        
        def objective(w):
            """Objective: negative Qini coefficient."""
            w = w / w.sum()  # Normalize
            ensemble_uplift = np.average(uplift_preds, axis=0, weights=w)
            qini = self._calculate_qini_coefficient(
                ensemble_uplift, treatment_val, y_val
            )
            return -qini
            
        # Optimize
        n_models = len(self.models)
        initial_weights = np.ones(n_models) / n_models
        bounds = [(0, 1) for _ in range(n_models)]
        constraints = {'type': 'eq', 'fun': lambda w: w.sum() - 1}
        
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        self.weights = result.x / result.x.sum()
        logger.info(f"Optimized weights: {self.weights}")
        
    @staticmethod
    def _calculate_qini_coefficient(
        uplift: np.ndarray,
        treatment: np.ndarray,
        y: np.ndarray
    ) -> float:
        """Calculate Qini coefficient."""
        # Sort by uplift (descending)
        sorted_idx = np.argsort(-uplift)
        treatment_sorted = treatment[sorted_idx]
        y_sorted = y[sorted_idx]
        
        # Calculate cumulative treatment and response
        n_samples = len(uplift)
        cum_treatment = np.cumsum(treatment_sorted)
        cum_control = np.arange(1, n_samples + 1) - cum_treatment
        cum_response_t = np.cumsum(treatment_sorted * y_sorted)
        cum_response_c = np.cumsum((1 - treatment_sorted) * y_sorted)
        
        # Qini curve
        qini_curve = (
            cum_response_t / np.maximum(cum_treatment, 1) * cum_treatment -
            cum_response_c / np.maximum(cum_control, 1) * cum_control
        )
        
        # Qini coefficient (area under curve)
        qini_coef = np.trapz(qini_curve) / n_samples
        
        return qini_coef
        
    def save(self, directory: Union[str, Path]) -> None:
        """Save ensemble to directory."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        # Save each model
        for i, model in enumerate(self.models):
            model.save(directory / f"model_{i}.pkl")
            
        # Save ensemble config
        config = {
            'weights': self.weights,
            'weights_type': self.weights_type,
            'task': self.task,
            'is_fitted': self.is_fitted,
            'n_models': len(self.models)
        }
        joblib.dump(config, directory / 'ensemble_config.pkl')
        
        logger.info(f"Ensemble saved to {directory}")
        
    @staticmethod
    def load(directory: Union[str, Path]) -> 'UpliftEnsemble':
        """Load ensemble from directory."""
        directory = Path(directory)
        config = joblib.load(directory / 'ensemble_config.pkl')
        
        # Load models
        models = []
        for i in range(config['n_models']):
            model = BaseUpliftModel.load(directory / f"model_{i}.pkl")
            models.append(model)
            
        # Reconstruct ensemble
        ensemble = UpliftEnsemble(
            models=models,
            weights=config['weights'],
            task=config['task']
        )
        ensemble.is_fitted = config['is_fitted']
        
        logger.info(f"Ensemble loaded from {directory}")
        return ensemble


class UpliftValidator:
    """
    Validator for uplift models with specialized metrics.
    
    Provides Qini curves, uplift curves, and treatment effect validation.
    
    Example:
        >>> validator = UpliftValidator()
        >>> metrics = validator.evaluate(uplift_model, X_test, treatment_test, y_test)
        >>> validator.plot_qini_curve(uplift_preds, treatment_test, y_test)
    """
    
    @staticmethod
    def calculate_qini_curve(
        uplift_pred: np.ndarray,
        treatment: np.ndarray,
        y: np.ndarray,
        n_bins: int = 10
    ) -> Dict[str, np.ndarray]:
        """
        Calculate Qini curve.
        
        Args:
            uplift_pred: Predicted uplift scores
            treatment: Treatment indicator
            y: Actual outcomes
            n_bins: Number of bins for curve
            
        Returns:
            Dictionary with curve data
        """
        # Sort by predicted uplift
        sorted_idx = np.argsort(-uplift_pred)
        treatment_sorted = treatment[sorted_idx]
        y_sorted = y[sorted_idx]
        
        n_samples = len(uplift_pred)
        bin_size = n_samples // n_bins
        
        qini_values = []
        fractions = []
        
        for i in range(1, n_bins + 1):
            end_idx = min(i * bin_size, n_samples)
            
            t_bin = treatment_sorted[:end_idx]
            y_bin = y_sorted[:end_idx]
            
            n_t = t_bin.sum()
            n_c = len(t_bin) - n_t
            
            if n_t > 0 and n_c > 0:
                response_t = (t_bin * y_bin).sum() / n_t
                response_c = ((1 - t_bin) * y_bin).sum() / n_c
                qini = response_t * n_t - response_c * n_c
            else:
                qini = 0.0
                
            qini_values.append(qini)
            fractions.append(end_idx / n_samples)
            
        return {
            'qini_values': np.array(qini_values),
            'fractions': np.array(fractions),
            'auc': np.trapz(qini_values, fractions)
        }
        
    @staticmethod
    def calculate_uplift_curve(
        uplift_pred: np.ndarray,
        treatment: np.ndarray,
        y: np.ndarray,
        n_bins: int = 10
    ) -> Dict[str, np.ndarray]:
        """Calculate uplift curve (incremental gains)."""
        sorted_idx = np.argsort(-uplift_pred)
        treatment_sorted = treatment[sorted_idx]
        y_sorted = y[sorted_idx]
        
        n_samples = len(uplift_pred)
        bin_size = n_samples // n_bins
        
        uplift_values = []
        fractions = []
        
        for i in range(1, n_bins + 1):
            end_idx = min(i * bin_size, n_samples)
            
            t_bin = treatment_sorted[:end_idx]
            y_bin = y_sorted[:end_idx]
            
            n_t = t_bin.sum()
            n_c = len(t_bin) - n_t
            
            if n_t > 0 and n_c > 0:
                response_t = (t_bin * y_bin).sum() / n_t
                response_c = ((1 - t_bin) * y_bin).sum() / n_c
                uplift = response_t - response_c
            else:
                uplift = 0.0
                
            uplift_values.append(uplift)
            fractions.append(end_idx / n_samples)
            
        return {
            'uplift_values': np.array(uplift_values),
            'fractions': np.array(fractions)
        }
        
    def evaluate(
        self,
        model: BaseUpliftModel,
        X: Union[pd.DataFrame, np.ndarray],
        treatment: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation of uplift model.
        
        Returns:
            Dictionary with multiple metrics
        """
        uplift_pred = model.predict_uplift(X)
        
        qini = self.calculate_qini_curve(uplift_pred, treatment, y)
        uplift_curve = self.calculate_uplift_curve(uplift_pred, treatment, y)
        
        # Calculate treatment effect in top decile
        top_decile_idx = np.argsort(-uplift_pred)[:len(uplift_pred) // 10]
        top_treatment = treatment[top_decile_idx]
        top_y = y[top_decile_idx]
        
        if top_treatment.sum() > 0 and (1 - top_treatment).sum() > 0:
            ate_top = (
                (top_treatment * top_y).sum() / top_treatment.sum() -
                ((1 - top_treatment) * top_y).sum() / (1 - top_treatment).sum()
            )
        else:
            ate_top = 0.0
            
        return {
            'qini_auc': qini['auc'],
            'qini_curve': qini,
            'uplift_curve': uplift_curve,
            'ate_top_decile': ate_top,
            'mean_predicted_uplift': uplift_pred.mean(),
            'std_predicted_uplift': uplift_pred.std(),
            'uplift_range': (uplift_pred.min(), uplift_pred.max())
        }
