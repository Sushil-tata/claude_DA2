"""
Tree-based models with unified interface and best practices.

This module provides wrapper classes for tree-based models (LightGBM, XGBoost,
CatBoost, RandomForest) with a consistent API, built-in cross-validation,
feature importance extraction, and hyperparameter management.

Example:
    >>> from src.models.tree_models import LightGBMModel
    >>> model = LightGBMModel(config_path='config/model_config.yaml')
    >>> model.fit(X_train, y_train, X_val, y_val)
    >>> predictions = model.predict_proba(X_test)
    >>> importance = model.get_feature_importance(importance_type='gain')
"""

import joblib
import numpy as np
import pandas as pd
import yaml
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from loguru import logger
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    logger.warning("LightGBM not available. Install with: pip install lightgbm")
    LIGHTGBM_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    logger.warning("XGBoost not available. Install with: pip install xgboost")
    XGBOOST_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    logger.warning("CatBoost not available. Install with: pip install catboost")
    CATBOOST_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    logger.warning("SHAP not available. Install with: pip install shap")
    SHAP_AVAILABLE = False


class BaseTreeModel(ABC):
    """Base class for tree-based models with unified interface."""
    
    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        params: Optional[Dict[str, Any]] = None,
        task: str = 'classification'
    ):
        """
        Initialize base tree model.
        
        Args:
            config_path: Path to model configuration YAML file
            params: Model hyperparameters (overrides config)
            task: 'classification' or 'regression'
        """
        self.config_path = config_path
        self.task = task
        self.model = None
        self.params = self._load_params(params)
        self.feature_names: Optional[List[str]] = None
        self.is_fitted = False
        
    def _load_params(self, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Load parameters from config file or use provided params."""
        if params is not None:
            return params
            
        if self.config_path is not None:
            try:
                with open(self.config_path, 'r') as f:
                    config = yaml.safe_load(f)
                model_name = self._get_model_name()
                return config.get(model_name, {}).get('default', {})
            except Exception as e:
                logger.warning(f"Failed to load config from {self.config_path}: {e}")
                return {}
        return {}
    
    @abstractmethod
    def _get_model_name(self) -> str:
        """Return model name for config lookup."""
        pass
    
    @abstractmethod
    def fit(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None,
        **kwargs
    ) -> 'BaseTreeModel':
        """Fit the model."""
        pass
    
    @abstractmethod
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make predictions."""
        pass
    
    @abstractmethod
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict class probabilities."""
        pass
    
    def cross_validate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        cv: int = 5,
        scoring: str = 'roc_auc'
    ) -> Dict[str, float]:
        """
        Perform cross-validation.
        
        Args:
            X: Features
            y: Target
            cv: Number of folds
            scoring: Scoring metric
            
        Returns:
            Dictionary with mean and std of scores
        """
        if not self.is_fitted:
            logger.warning("Model not fitted. Fitting with provided data first.")
            self.fit(X, y)
        
        scores = cross_val_score(self.model, X, y, cv=cv, scoring=scoring)
        return {
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'scores': scores.tolist()
        }
    
    @abstractmethod
    def get_feature_importance(
        self,
        importance_type: str = 'gain'
    ) -> pd.DataFrame:
        """Get feature importance."""
        pass
    
    def get_shap_values(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        check_additivity: bool = False
    ) -> Optional[np.ndarray]:
        """
        Calculate SHAP values for feature importance.
        
        Args:
            X: Features to explain
            check_additivity: Whether to check SHAP additivity
            
        Returns:
            SHAP values array or None if SHAP unavailable
        """
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available. Install with: pip install shap")
            return None
        
        if not self.is_fitted:
            logger.error("Model must be fitted before calculating SHAP values")
            return None
        
        try:
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X, check_additivity=check_additivity)
            return shap_values
        except Exception as e:
            logger.error(f"Failed to calculate SHAP values: {e}")
            return None
    
    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save model to disk.
        
        Args:
            filepath: Path to save model
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'params': self.params,
            'feature_names': self.feature_names,
            'task': self.task,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: Union[str, Path]) -> 'BaseTreeModel':
        """
        Load model from disk.
        
        Args:
            filepath: Path to load model from
            
        Returns:
            Self
        """
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.params = model_data['params']
        self.feature_names = model_data['feature_names']
        self.task = model_data['task']
        self.is_fitted = model_data['is_fitted']
        
        logger.info(f"Model loaded from {filepath}")
        return self


class LightGBMModel(BaseTreeModel):
    """
    LightGBM model wrapper with best practices.
    
    Example:
        >>> model = LightGBMModel(config_path='config/model_config.yaml')
        >>> model.fit(X_train, y_train, X_val, y_val)
        >>> preds = model.predict_proba(X_test)[:, 1]
    """
    
    def _get_model_name(self) -> str:
        return 'lightgbm'
    
    def fit(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None,
        early_stopping_rounds: int = 50,
        verbose: int = 50,
        **kwargs
    ) -> 'LightGBMModel':
        """
        Fit LightGBM model.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            early_stopping_rounds: Early stopping rounds
            verbose: Verbosity interval
            **kwargs: Additional parameters
            
        Returns:
            Self
        """
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not available. Install with: pip install lightgbm")
        
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
        
        params = {**self.params, **kwargs}
        
        if self.task == 'classification':
            self.model = lgb.LGBMClassifier(**params)
        else:
            params['objective'] = params.get('objective', 'regression')
            params['metric'] = params.get('metric', 'rmse')
            self.model = lgb.LGBMRegressor(**params)
        
        fit_params = {}
        if X_val is not None and y_val is not None:
            fit_params['eval_set'] = [(X_val, y_val)]
            fit_params['eval_metric'] = params.get('metric', 'auc')
            if early_stopping_rounds > 0:
                fit_params['callbacks'] = [
                    lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=verbose > 0),
                    lgb.log_evaluation(period=verbose)
                ]
        
        logger.info("Training LightGBM model...")
        self.model.fit(X_train, y_train, **fit_params)
        self.is_fitted = True
        
        logger.info("LightGBM model training complete")
        return self
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict class labels or regression values."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict class probabilities (classification only)."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        if self.task != 'classification':
            raise ValueError("predict_proba only available for classification")
        return self.model.predict_proba(X)
    
    def get_feature_importance(
        self,
        importance_type: str = 'gain'
    ) -> pd.DataFrame:
        """
        Get feature importance.
        
        Args:
            importance_type: 'gain' or 'split'
            
        Returns:
            DataFrame with features and importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting importance")
        
        importance = self.model.feature_importances_
        feature_names = self.feature_names or [f'feature_{i}' for i in range(len(importance))]
        
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return df


class XGBoostModel(BaseTreeModel):
    """
    XGBoost model wrapper with best practices.
    
    Example:
        >>> model = XGBoostModel(config_path='config/model_config.yaml')
        >>> model.fit(X_train, y_train, X_val, y_val)
        >>> preds = model.predict_proba(X_test)[:, 1]
    """
    
    def _get_model_name(self) -> str:
        return 'xgboost'
    
    def fit(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None,
        early_stopping_rounds: int = 50,
        verbose: int = 50,
        **kwargs
    ) -> 'XGBoostModel':
        """Fit XGBoost model."""
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not available. Install with: pip install xgboost")
        
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
        
        params = {**self.params, **kwargs}
        
        if self.task == 'classification':
            self.model = xgb.XGBClassifier(**params)
        else:
            params['objective'] = params.get('objective', 'reg:squarederror')
            self.model = xgb.XGBRegressor(**params)
        
        fit_params = {}
        if X_val is not None and y_val is not None:
            fit_params['eval_set'] = [(X_val, y_val)]
            if early_stopping_rounds > 0:
                fit_params['early_stopping_rounds'] = early_stopping_rounds
            fit_params['verbose'] = verbose
        
        logger.info("Training XGBoost model...")
        self.model.fit(X_train, y_train, **fit_params)
        self.is_fitted = True
        
        logger.info("XGBoost model training complete")
        return self
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict class labels or regression values."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict class probabilities (classification only)."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        if self.task != 'classification':
            raise ValueError("predict_proba only available for classification")
        return self.model.predict_proba(X)
    
    def get_feature_importance(
        self,
        importance_type: str = 'gain'
    ) -> pd.DataFrame:
        """Get feature importance."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting importance")
        
        importance = self.model.feature_importances_
        feature_names = self.feature_names or [f'feature_{i}' for i in range(len(importance))]
        
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return df


class CatBoostModel(BaseTreeModel):
    """
    CatBoost model wrapper with best practices.
    
    Example:
        >>> model = CatBoostModel(config_path='config/model_config.yaml')
        >>> model.fit(X_train, y_train, X_val, y_val, cat_features=['category_col'])
        >>> preds = model.predict_proba(X_test)[:, 1]
    """
    
    def _get_model_name(self) -> str:
        return 'catboost'
    
    def fit(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None,
        cat_features: Optional[List[str]] = None,
        early_stopping_rounds: int = 50,
        verbose: int = 50,
        **kwargs
    ) -> 'CatBoostModel':
        """
        Fit CatBoost model.
        
        Args:
            cat_features: List of categorical feature names
        """
        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost not available. Install with: pip install catboost")
        
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
        
        params = {**self.params, **kwargs}
        
        if self.task == 'classification':
            self.model = cb.CatBoostClassifier(**params)
        else:
            params['loss_function'] = params.get('loss_function', 'RMSE')
            self.model = cb.CatBoostRegressor(**params)
        
        fit_params = {'cat_features': cat_features} if cat_features else {}
        
        if X_val is not None and y_val is not None:
            fit_params['eval_set'] = [(X_val, y_val)]
            if early_stopping_rounds > 0:
                fit_params['early_stopping_rounds'] = early_stopping_rounds
            fit_params['verbose'] = verbose
        
        logger.info("Training CatBoost model...")
        self.model.fit(X_train, y_train, **fit_params)
        self.is_fitted = True
        
        logger.info("CatBoost model training complete")
        return self
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict class labels or regression values."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict class probabilities (classification only)."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        if self.task != 'classification':
            raise ValueError("predict_proba only available for classification")
        return self.model.predict_proba(X)
    
    def get_feature_importance(
        self,
        importance_type: str = 'gain'
    ) -> pd.DataFrame:
        """Get feature importance."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting importance")
        
        importance = self.model.feature_importances_
        feature_names = self.feature_names or [f'feature_{i}' for i in range(len(importance))]
        
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return df


class RandomForestModel(BaseTreeModel):
    """
    Random Forest model wrapper with best practices.
    
    Example:
        >>> model = RandomForestModel(config_path='config/model_config.yaml')
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict_proba(X_test)[:, 1]
    """
    
    def _get_model_name(self) -> str:
        return 'random_forest'
    
    def fit(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None,
        **kwargs
    ) -> 'RandomForestModel':
        """Fit Random Forest model."""
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
        
        params = {**self.params, **kwargs}
        
        if self.task == 'classification':
            self.model = RandomForestClassifier(**params)
        else:
            self.model = RandomForestRegressor(**params)
        
        logger.info("Training Random Forest model...")
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        logger.info("Random Forest model training complete")
        return self
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict class labels or regression values."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict class probabilities (classification only)."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        if self.task != 'classification':
            raise ValueError("predict_proba only available for classification")
        return self.model.predict_proba(X)
    
    def get_feature_importance(
        self,
        importance_type: str = 'gain'
    ) -> pd.DataFrame:
        """Get feature importance."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting importance")
        
        importance = self.model.feature_importances_
        feature_names = self.feature_names or [f'feature_{i}' for i in range(len(importance))]
        
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return df
