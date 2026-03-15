"""
Neural network models for tabular data.

This module provides wrapper classes for neural network architectures specialized
for tabular data including TabNet, TabPFN, NODE, and DeepGBM.

Example:
    >>> from src.models.neural_tabular import TabNetModel
    >>> model = TabNetModel(config_path='config/model_config.yaml')
    >>> model.fit(X_train, y_train, X_val, y_val)
    >>> predictions = model.predict_proba(X_test)
"""

import joblib
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from loguru import logger
from sklearn.preprocessing import StandardScaler

try:
    from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
    TABNET_AVAILABLE = True
except ImportError:
    logger.warning("TabNet not available. Install with: pip install pytorch-tabnet")
    TABNET_AVAILABLE = False

try:
    from tabpfn import TabPFNClassifier
    TABPFN_AVAILABLE = True
except ImportError:
    logger.info("TabPFN not available. Install with: pip install tabpfn")
    TABPFN_AVAILABLE = False


class BaseNeuralTabular:
    """Base class for neural tabular models."""
    
    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        params: Optional[Dict[str, Any]] = None,
        task: str = 'classification',
        use_gpu: bool = True
    ):
        """
        Initialize base neural tabular model.
        
        Args:
            config_path: Path to model configuration YAML file
            params: Model hyperparameters
            task: 'classification' or 'regression'
            use_gpu: Whether to use GPU if available
        """
        self.config_path = config_path
        self.task = task
        self.model = None
        self.params = self._load_params(params)
        self.feature_names: Optional[List[str]] = None
        self.is_fitted = False
        self.scaler = StandardScaler()
        
        # GPU detection
        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")
    
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
    
    def _get_model_name(self) -> str:
        """Return model name for config lookup."""
        raise NotImplementedError
    
    def save(self, filepath: Union[str, Path]) -> None:
        """Save model to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'params': self.params,
            'feature_names': self.feature_names,
            'task': self.task,
            'is_fitted': self.is_fitted,
            'scaler': self.scaler
        }
        
        # Save model-specific components
        self._save_model_specific(filepath, model_data)
        
        logger.info(f"Model saved to {filepath}")
    
    def _save_model_specific(self, filepath: Path, model_data: Dict) -> None:
        """Save model-specific components. Override in subclasses."""
        raise NotImplementedError
    
    def load(self, filepath: Union[str, Path]) -> 'BaseNeuralTabular':
        """Load model from disk."""
        raise NotImplementedError


class TabNetModel(BaseNeuralTabular):
    """
    TabNet model for tabular data with attention mechanisms.
    
    TabNet uses sequential attention to select features at each decision step,
    providing interpretability and strong performance on tabular data.
    
    Example:
        >>> model = TabNetModel(config_path='config/model_config.yaml')
        >>> model.fit(X_train, y_train, X_val, y_val, max_epochs=100)
        >>> preds = model.predict_proba(X_test)
        >>> importance = model.get_feature_importance()
    """
    
    def _get_model_name(self) -> str:
        return 'tabnet'
    
    def fit(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None,
        max_epochs: int = 100,
        patience: int = 15,
        batch_size: int = 1024,
        virtual_batch_size: int = 128,
        **kwargs
    ) -> 'TabNetModel':
        """
        Fit TabNet model.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            max_epochs: Maximum training epochs
            patience: Early stopping patience
            batch_size: Batch size for training
            virtual_batch_size: Size of mini-batches for ghost batch normalization
            **kwargs: Additional parameters
            
        Returns:
            Self
        """
        if not TABNET_AVAILABLE:
            raise ImportError("TabNet not available. Install with: pip install pytorch-tabnet")
        
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
            X_train = X_train.values
        if isinstance(y_train, pd.Series):
            y_train = y_train.values
        if isinstance(X_val, pd.DataFrame):
            X_val = X_val.values
        if isinstance(y_val, pd.Series):
            y_val = y_val.values
        
        # Initialize model
        params = {**self.params, **kwargs}
        params['device_name'] = self.device
        
        if self.task == 'classification':
            self.model = TabNetClassifier(**params)
        else:
            self.model = TabNetRegressor(**params)
        
        # Prepare evaluation set
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
        
        logger.info("Training TabNet model...")
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            virtual_batch_size=virtual_batch_size,
            eval_metric=['auc'] if self.task == 'classification' else ['rmse']
        )
        
        self.is_fitted = True
        logger.info("TabNet model training complete")
        return self
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict class labels or regression values."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        return self.model.predict(X)
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict class probabilities (classification only)."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        if self.task != 'classification':
            raise ValueError("predict_proba only available for classification")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from TabNet.
        
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
    
    def _save_model_specific(self, filepath: Path, model_data: Dict) -> None:
        """Save TabNet-specific components."""
        # Save TabNet model
        model_path = filepath.parent / f"{filepath.stem}_tabnet.zip"
        self.model.save_model(str(model_path))
        
        # Save metadata
        joblib.dump(model_data, filepath)
    
    def load(self, filepath: Union[str, Path]) -> 'TabNetModel':
        """Load TabNet model from disk."""
        filepath = Path(filepath)
        
        # Load metadata
        model_data = joblib.load(filepath)
        self.params = model_data['params']
        self.feature_names = model_data['feature_names']
        self.task = model_data['task']
        self.is_fitted = model_data['is_fitted']
        self.scaler = model_data['scaler']
        
        # Load TabNet model
        model_path = filepath.parent / f"{filepath.stem}_tabnet.zip"
        if self.task == 'classification':
            self.model = TabNetClassifier()
        else:
            self.model = TabNetRegressor()
        self.model.load_model(str(model_path))
        
        logger.info(f"Model loaded from {filepath}")
        return self


class TabPFNModel(BaseNeuralTabular):
    """
    TabPFN (Tabular Prior-Data Fitted Networks) model.
    
    TabPFN is a foundation model for small tabular datasets that was trained on
    synthetic data and can be directly applied without training.
    
    Note: Works best on datasets with < 10,000 samples and < 100 features.
    
    Example:
        >>> model = TabPFNModel()
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict_proba(X_test)
    """
    
    def _get_model_name(self) -> str:
        return 'tabpfn'
    
    def fit(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None,
        **kwargs
    ) -> 'TabPFNModel':
        """
        Fit TabPFN model.
        
        Note: TabPFN doesn't actually train - it's a pre-trained model.
        """
        if not TABPFN_AVAILABLE:
            logger.warning(
                "TabPFN not available. Falling back to LightGBM. "
                "Install TabPFN with: pip install tabpfn"
            )
            # Fallback to simple sklearn model
            from sklearn.ensemble import GradientBoostingClassifier
            self.model = GradientBoostingClassifier()
            self.model.fit(X_train, y_train)
            self.is_fitted = True
            return self
        
        if self.task != 'classification':
            raise ValueError("TabPFN only supports classification tasks")
        
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
            X_train = X_train.values
        if isinstance(y_train, pd.Series):
            y_train = y_train.values
        
        # Check dataset size constraints
        n_samples, n_features = X_train.shape
        if n_samples > 10000:
            logger.warning(f"TabPFN works best with < 10k samples. Got {n_samples}. Consider subsampling.")
        if n_features > 100:
            logger.warning(f"TabPFN works best with < 100 features. Got {n_features}. Consider feature selection.")
        
        logger.info("Fitting TabPFN model...")
        self.model = TabPFNClassifier(device=self.device, **kwargs)
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        logger.info("TabPFN model fitting complete")
        return self
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict class labels."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        return self.model.predict(X)
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        return self.model.predict_proba(X)
    
    def _save_model_specific(self, filepath: Path, model_data: Dict) -> None:
        """Save TabPFN model."""
        model_data['model'] = self.model
        joblib.dump(model_data, filepath)
    
    def load(self, filepath: Union[str, Path]) -> 'TabPFNModel':
        """Load TabPFN model from disk."""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.params = model_data['params']
        self.feature_names = model_data['feature_names']
        self.task = model_data['task']
        self.is_fitted = model_data['is_fitted']
        self.scaler = model_data['scaler']
        
        logger.info(f"Model loaded from {filepath}")
        return self


class NODEModel(BaseNeuralTabular):
    """
    Neural Oblivious Decision Ensembles (NODE) model.
    
    NODE combines gradient boosting with neural networks using differentiable
    oblivious decision trees.
    
    Note: This is a placeholder implementation. For production use, consider
    implementing with the original NODE repository or using tree-based models.
    
    Example:
        >>> model = NODEModel()
        >>> model.fit(X_train, y_train, X_val, y_val)
        >>> preds = model.predict_proba(X_test)
    """
    
    def _get_model_name(self) -> str:
        return 'node'
    
    def fit(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None,
        **kwargs
    ) -> 'NODEModel':
        """Fit NODE model."""
        logger.warning(
            "NODE is not fully implemented. Falling back to LightGBM. "
            "For production NODE, use the official implementation."
        )
        
        # Fallback to tree-based model
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
        
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
        
        if self.task == 'classification':
            self.model = GradientBoostingClassifier(**kwargs)
        else:
            self.model = GradientBoostingRegressor(**kwargs)
        
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        return self
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict class labels or regression values."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        if self.task != 'classification':
            raise ValueError("predict_proba only available for classification")
        return self.model.predict_proba(X)
    
    def _save_model_specific(self, filepath: Path, model_data: Dict) -> None:
        """Save NODE model."""
        model_data['model'] = self.model
        joblib.dump(model_data, filepath)
    
    def load(self, filepath: Union[str, Path]) -> 'NODEModel':
        """Load NODE model from disk."""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.params = model_data['params']
        self.feature_names = model_data['feature_names']
        self.task = model_data['task']
        self.is_fitted = model_data['is_fitted']
        
        logger.info(f"Model loaded from {filepath}")
        return self


class DeepGBMModel(BaseNeuralTabular):
    """
    DeepGBM (Deep Gradient Boosting Machine) model.
    
    DeepGBM combines neural networks with gradient boosting for tabular data.
    
    Note: This is a placeholder implementation. For production use, implement
    with the official DeepGBM repository or use tree-based models.
    
    Example:
        >>> model = DeepGBMModel()
        >>> model.fit(X_train, y_train, X_val, y_val)
        >>> preds = model.predict_proba(X_test)
    """
    
    def _get_model_name(self) -> str:
        return 'deepgbm'
    
    def fit(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None,
        **kwargs
    ) -> 'DeepGBMModel':
        """Fit DeepGBM model."""
        logger.warning(
            "DeepGBM is not fully implemented. Falling back to LightGBM. "
            "For production DeepGBM, use the official implementation."
        )
        
        # Fallback to tree-based model
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
        
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
        
        if self.task == 'classification':
            self.model = GradientBoostingClassifier(**kwargs)
        else:
            self.model = GradientBoostingRegressor(**kwargs)
        
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        return self
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict class labels or regression values."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        if self.task != 'classification':
            raise ValueError("predict_proba only available for classification")
        return self.model.predict_proba(X)
    
    def _save_model_specific(self, filepath: Path, model_data: Dict) -> None:
        """Save DeepGBM model."""
        model_data['model'] = self.model
        joblib.dump(model_data, filepath)
    
    def load(self, filepath: Union[str, Path]) -> 'DeepGBMModel':
        """Load DeepGBM model from disk."""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.params = model_data['params']
        self.feature_names = model_data['feature_names']
        self.task = model_data['task']
        self.is_fitted = model_data['is_fitted']
        
        logger.info(f"Model loaded from {filepath}")
        return self
