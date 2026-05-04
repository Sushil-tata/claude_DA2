"""
Collections NBA Payment Amount Estimator

Predicts expected payment amount using hurdle model approach with quantile regression
for confidence intervals and payment pattern analysis.

Example:
    >>> from src.use_cases.collections_nba.payment_estimator import PaymentEstimator
    >>> estimator = PaymentEstimator(config_path='config/model_config.yaml')
    >>> estimator.fit(X_train, y_train, propensity_scores_train)
    >>> predictions = estimator.predict(X_test, propensity_scores_test)
"""

import joblib
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from loguru import logger
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    logger.warning("LightGBM not available. Install with: pip install lightgbm")
    LIGHTGBM_AVAILABLE = False


class PaymentEstimatorConfig:
    """Configuration for payment estimator."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize payment estimator configuration.
        
        Args:
            config_path: Path to model_config.yaml
        """
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                full_config = yaml.safe_load(f)
                payment_config = full_config.get('payment_estimator', {})
        else:
            payment_config = {}
        
        self.use_hurdle = payment_config.get('use_hurdle', True)
        self.quantiles = payment_config.get('quantiles', [0.1, 0.25, 0.5, 0.75, 0.9])
        self.log_transform = payment_config.get('log_transform', True)
        self.clip_predictions = payment_config.get('clip_predictions', True)
        self.min_payment = payment_config.get('min_payment', 0.0)
        self.max_payment = payment_config.get('max_payment', None)
        
        self.model_params = payment_config.get('model_params', {
            'objective': 'regression',
            'metric': 'mae',
            'boosting': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.7,
            'bagging_freq': 5,
            'verbose': -1,
            'max_depth': 7,
            'min_child_samples': 20,
        })


class PaymentEstimator:
    """
    Payment amount estimator using hurdle model approach.
    
    Two-stage model:
    1. Propensity to pay (uses external propensity model)
    2. Amount to pay (conditional on paying)
    
    Provides point estimates and confidence intervals via quantile regression.
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        use_propensity_features: bool = True
    ):
        """
        Initialize payment estimator.
        
        Args:
            config_path: Path to model_config.yaml
            use_propensity_features: Whether to use propensity scores as features
            
        Example:
            >>> estimator = PaymentEstimator(
            ...     config_path="config/model_config.yaml",
            ...     use_propensity_features=True
            ... )
        """
        self.config = PaymentEstimatorConfig(config_path)
        self.use_propensity_features = use_propensity_features
        
        # Models
        self.amount_model: Optional[Any] = None
        self.quantile_models: Dict[float, Any] = {}
        
        # Preprocessing
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: Optional[List[str]] = None
        self.is_fitted = False
        
        # Stats for inverse transform
        self.y_mean: Optional[float] = None
        self.y_std: Optional[float] = None
        
        logger.info("Initialized PaymentEstimator")
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        propensity_scores: Optional[pd.Series] = None,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        propensity_scores_val: Optional[pd.Series] = None
    ) -> 'PaymentEstimator':
        """
        Fit payment amount estimator.
        
        Args:
            X: Training features
            y: Payment amounts (0 for non-payments)
            propensity_scores: Probability of payment (optional)
            X_val: Validation features
            y_val: Validation amounts
            propensity_scores_val: Validation propensity scores
            
        Returns:
            Self for method chaining
            
        Example:
            >>> estimator.fit(X_train, y_train, propensity_train, X_val, y_val, propensity_val)
        """
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is required. Install with: pip install lightgbm")
        
        # Add propensity scores as feature if provided
        if self.use_propensity_features and propensity_scores is not None:
            X = X.copy()
            X['propensity_score'] = propensity_scores.values
            
            if X_val is not None and propensity_scores_val is not None:
                X_val = X_val.copy()
                X_val['propensity_score'] = propensity_scores_val.values
        
        self.feature_names = list(X.columns)
        
        # Filter to only paying customers for hurdle model
        if self.config.use_hurdle:
            logger.info("Using hurdle model - training only on paying customers")
            pay_mask = y > 0
            X_pay = X[pay_mask]
            y_pay = y[pay_mask]
            
            if X_val is not None:
                val_pay_mask = y_val > 0
                X_val_pay = X_val[val_pay_mask]
                y_val_pay = y_val[val_pay_mask]
            else:
                X_val_pay = None
                y_val_pay = None
        else:
            X_pay = X
            y_pay = y
            X_val_pay = X_val
            y_val_pay = y_val
        
        logger.info(f"Training on {len(X_pay)} samples with payments")
        
        # Log transform if configured
        if self.config.log_transform:
            logger.info("Applying log transform to payment amounts")
            y_train = np.log1p(y_pay)
            y_val_transformed = np.log1p(y_val_pay) if y_val_pay is not None else None
            self.y_mean = y_train.mean()
            self.y_std = y_train.std()
        else:
            y_train = y_pay
            y_val_transformed = y_val_pay
            self.y_mean = y_train.mean()
            self.y_std = y_train.std()
        
        # Train main amount model (mean prediction)
        logger.info("Training main amount model")
        self.amount_model = lgb.LGBMRegressor(**self.config.model_params)
        
        eval_set = [(X_val_pay, y_val_transformed)] if X_val_pay is not None and y_val_transformed is not None else None
        
        self.amount_model.fit(
            X_pay, y_train,
            eval_set=eval_set,
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)] if eval_set else None
        )
        
        # Train quantile models for confidence intervals
        logger.info(f"Training quantile models for quantiles: {self.config.quantiles}")
        for quantile in self.config.quantiles:
            quantile_params = self.config.model_params.copy()
            quantile_params['objective'] = 'quantile'
            quantile_params['alpha'] = quantile
            
            model = lgb.LGBMRegressor(**quantile_params)
            model.fit(
                X_pay, y_train,
                eval_set=eval_set,
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)] if eval_set else None
            )
            self.quantile_models[quantile] = model
        
        # Log validation metrics
        if X_val_pay is not None and y_val_pay is not None:
            val_pred = self.amount_model.predict(X_val_pay)
            
            if self.config.log_transform:
                val_pred = np.expm1(val_pred)
                y_val_original = y_val_pay
            else:
                y_val_original = y_val_pay
            
            mae = mean_absolute_error(y_val_original, val_pred)
            rmse = np.sqrt(mean_squared_error(y_val_original, val_pred))
            r2 = r2_score(y_val_original, val_pred)
            
            logger.info(f"Validation MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")
        
        self.is_fitted = True
        logger.info("Payment estimator training complete")
        return self
    
    def predict(
        self,
        X: pd.DataFrame,
        propensity_scores: Optional[pd.Series] = None,
        return_intervals: bool = False,
        confidence_level: float = 0.8
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Predict payment amounts.
        
        Args:
            X: Features
            propensity_scores: Probability of payment (optional)
            return_intervals: Whether to return confidence intervals
            confidence_level: Confidence level for intervals (e.g., 0.8 for 80%)
            
        Returns:
            If return_intervals=False: Array of predicted amounts
            If return_intervals=True: Tuple of (predictions, lower_bounds, upper_bounds)
            
        Example:
            >>> amounts = estimator.predict(X_test, propensity_test)
            >>> amounts, lower, upper = estimator.predict(X_test, propensity_test, return_intervals=True)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Add propensity scores as feature if provided
        if self.use_propensity_features and propensity_scores is not None:
            X = X.copy()
            X['propensity_score'] = propensity_scores.values
        
        # Predict amounts
        predictions = self.amount_model.predict(X)
        
        # Inverse transform
        if self.config.log_transform:
            predictions = np.expm1(predictions)
        
        # Apply hurdle model (multiply by propensity)
        if self.config.use_hurdle and propensity_scores is not None:
            predictions = predictions * propensity_scores.values
        
        # Clip predictions
        if self.config.clip_predictions:
            predictions = np.clip(
                predictions,
                self.config.min_payment,
                self.config.max_payment if self.config.max_payment else np.inf
            )
        
        if not return_intervals:
            return predictions
        
        # Calculate confidence intervals using quantile models
        alpha = (1 - confidence_level) / 2
        lower_quantile = alpha
        upper_quantile = 1 - alpha
        
        # Find closest available quantiles
        available_quantiles = sorted(self.quantile_models.keys())
        lower_q = min(available_quantiles, key=lambda x: abs(x - lower_quantile))
        upper_q = min(available_quantiles, key=lambda x: abs(x - upper_quantile))
        
        lower_bounds = self.quantile_models[lower_q].predict(X)
        upper_bounds = self.quantile_models[upper_q].predict(X)
        
        # Inverse transform
        if self.config.log_transform:
            lower_bounds = np.expm1(lower_bounds)
            upper_bounds = np.expm1(upper_bounds)
        
        # Apply hurdle and clipping
        if self.config.use_hurdle and propensity_scores is not None:
            lower_bounds = lower_bounds * propensity_scores.values
            upper_bounds = upper_bounds * propensity_scores.values
        
        if self.config.clip_predictions:
            lower_bounds = np.clip(
                lower_bounds,
                self.config.min_payment,
                self.config.max_payment if self.config.max_payment else np.inf
            )
            upper_bounds = np.clip(
                upper_bounds,
                self.config.min_payment,
                self.config.max_payment if self.config.max_payment else np.inf
            )
        
        return predictions, lower_bounds, upper_bounds
    
    def predict_quantiles(
        self,
        X: pd.DataFrame,
        propensity_scores: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Predict all quantiles.
        
        Args:
            X: Features
            propensity_scores: Probability of payment (optional)
            
        Returns:
            DataFrame with columns for each quantile
            
        Example:
            >>> quantiles = estimator.predict_quantiles(X_test, propensity_test)
            >>> quantiles.columns  # ['q_0.1', 'q_0.25', 'q_0.5', 'q_0.75', 'q_0.9']
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Add propensity scores as feature if provided
        if self.use_propensity_features and propensity_scores is not None:
            X = X.copy()
            X['propensity_score'] = propensity_scores.values
        
        result = pd.DataFrame(index=X.index)
        
        for quantile, model in sorted(self.quantile_models.items()):
            predictions = model.predict(X)
            
            # Inverse transform
            if self.config.log_transform:
                predictions = np.expm1(predictions)
            
            # Apply hurdle
            if self.config.use_hurdle and propensity_scores is not None:
                predictions = predictions * propensity_scores.values
            
            # Clip
            if self.config.clip_predictions:
                predictions = np.clip(
                    predictions,
                    self.config.min_payment,
                    self.config.max_payment if self.config.max_payment else np.inf
                )
            
            result[f'q_{quantile}'] = predictions
        
        return result
    
    def analyze_payment_patterns(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        segment_col: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Analyze payment patterns by segment.
        
        Args:
            X: Features (must include segment_col if provided)
            y: Actual payment amounts
            segment_col: Column name for segmentation
            
        Returns:
            DataFrame with payment statistics by segment
            
        Example:
            >>> patterns = estimator.analyze_payment_patterns(X_test, y_test, 'risk_segment')
        """
        if segment_col is None or segment_col not in X.columns:
            segments = pd.Series(['all'] * len(X), index=X.index)
            segment_col = 'segment'
        else:
            segments = X[segment_col]
        
        results = []
        
        for segment in segments.unique():
            seg_mask = segments == segment
            y_seg = y[seg_mask]
            
            paying_mask = y_seg > 0
            
            result = {
                'segment': segment,
                'count': len(y_seg),
                'payment_rate': paying_mask.mean(),
                'avg_payment': y_seg[paying_mask].mean() if paying_mask.any() else 0,
                'median_payment': y_seg[paying_mask].median() if paying_mask.any() else 0,
                'total_collected': y_seg.sum(),
                'std_payment': y_seg[paying_mask].std() if paying_mask.any() else 0,
                'max_payment': y_seg.max(),
                'min_payment': y_seg[paying_mask].min() if paying_mask.any() else 0,
            }
            
            # Quantiles
            if paying_mask.any():
                for q in [0.25, 0.5, 0.75, 0.9]:
                    result[f'p{int(q*100)}'] = y_seg[paying_mask].quantile(q)
            
            results.append(result)
        
        return pd.DataFrame(results)
    
    def get_feature_importance(self, importance_type: str = 'gain') -> pd.DataFrame:
        """
        Get feature importance from amount model.
        
        Args:
            importance_type: 'gain' or 'split'
            
        Returns:
            DataFrame with features and importance scores
            
        Example:
            >>> importance = estimator.get_feature_importance()
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        importance = self.amount_model.feature_importances_
        
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
            >>> estimator.save("models/payment_estimator.pkl")
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        save_dict = {
            'amount_model': self.amount_model,
            'quantile_models': self.quantile_models,
            'feature_names': self.feature_names,
            'config': self.config,
            'use_propensity_features': self.use_propensity_features,
            'y_mean': self.y_mean,
            'y_std': self.y_std,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(save_dict, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'PaymentEstimator':
        """
        Load model from disk.
        
        Args:
            filepath: Path to saved model
            
        Returns:
            Loaded PaymentEstimator instance
            
        Example:
            >>> estimator = PaymentEstimator.load("models/payment_estimator.pkl")
        """
        save_dict = joblib.load(filepath)
        
        instance = cls(use_propensity_features=save_dict['use_propensity_features'])
        instance.amount_model = save_dict['amount_model']
        instance.quantile_models = save_dict['quantile_models']
        instance.feature_names = save_dict['feature_names']
        instance.config = save_dict['config']
        instance.y_mean = save_dict['y_mean']
        instance.y_std = save_dict['y_std']
        instance.is_fitted = save_dict['is_fitted']
        
        logger.info(f"Model loaded from {filepath}")
        return instance
