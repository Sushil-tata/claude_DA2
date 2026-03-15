"""
Supervised Fraud Detection Models

Implements production-ready supervised fraud classification with:
- Multi-model support (LightGBM, XGBoost, Neural Networks)
- Extreme class imbalance handling (fraud rate < 0.1%)
- Real-time scoring (<100ms latency)
- Graph feature integration
- Model calibration for accurate fraud probabilities
- Threshold optimization for precision/recall trade-off
- Feature engineering specific to fraud patterns

Example:
    >>> from src.use_cases.fraud_detection.supervised_fraud import FraudClassifier
    >>> classifier = FraudClassifier(model_type='lightgbm')
    >>> classifier.fit(X_train, y_train, X_val, y_val)
    >>> scores = classifier.predict_proba(X_test)
    >>> threshold = classifier.optimize_threshold(X_val, y_val, target_recall=0.8)
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
import warnings

from loguru import logger
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, roc_curve
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

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
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch not available. Install with: pip install torch")
    TORCH_AVAILABLE = False


class FraudFeatureEngineer:
    """
    Engineer fraud-specific features from transaction data.
    
    Creates velocity features, anomaly scores, and graph-based features
    optimized for fraud detection.
    """
    
    def __init__(self):
        """Initialize fraud feature engineer."""
        self.velocity_windows = [1, 6, 12, 24, 168]  # hours
        self.feature_names: List[str] = []
        logger.info("Initialized FraudFeatureEngineer")
    
    def create_velocity_features(
        self,
        transactions: pd.DataFrame,
        account_col: str = 'account_id',
        timestamp_col: str = 'timestamp',
        amount_col: str = 'amount'
    ) -> pd.DataFrame:
        """
        Create velocity features (transaction frequency/volume over time).
        
        Args:
            transactions: Transaction DataFrame
            account_col: Account ID column
            timestamp_col: Timestamp column
            amount_col: Amount column
            
        Returns:
            DataFrame with velocity features
            
        Example:
            >>> engineer = FraudFeatureEngineer()
            >>> features = engineer.create_velocity_features(transactions)
        """
        logger.info("Creating velocity features")
        
        df = transactions.copy()
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        df = df.sort_values([account_col, timestamp_col])
        
        features = []
        
        for window_hours in self.velocity_windows:
            window = pd.Timedelta(hours=window_hours)
            
            # Transaction count in window
            count_col = f'txn_count_{window_hours}h'
            df[count_col] = df.groupby(account_col)[timestamp_col].transform(
                lambda x: x.rolling(window, on=x).count()
            )
            features.append(count_col)
            
            # Transaction amount sum in window
            sum_col = f'txn_sum_{window_hours}h'
            df[sum_col] = df.groupby(account_col)[amount_col].transform(
                lambda x: x.rolling(window, on=df.loc[x.index, timestamp_col]).sum()
            )
            features.append(sum_col)
            
            # Average transaction amount in window
            avg_col = f'txn_avg_{window_hours}h'
            df[avg_col] = df[sum_col] / df[count_col].replace(0, 1)
            features.append(avg_col)
            
            # Max transaction amount in window
            max_col = f'txn_max_{window_hours}h'
            df[max_col] = df.groupby(account_col)[amount_col].transform(
                lambda x: x.rolling(window, on=df.loc[x.index, timestamp_col]).max()
            )
            features.append(max_col)
        
        # Velocity acceleration (change in velocity)
        if len(self.velocity_windows) >= 2:
            short_window = self.velocity_windows[0]
            long_window = self.velocity_windows[-1]
            df['velocity_acceleration'] = (
                df[f'txn_count_{short_window}h'] / 
                df[f'txn_count_{long_window}h'].replace(0, 1)
            )
            features.append('velocity_acceleration')
        
        self.feature_names.extend(features)
        logger.info(f"Created {len(features)} velocity features")
        
        return df[features].fillna(0)
    
    def create_anomaly_scores(
        self,
        transactions: pd.DataFrame,
        amount_col: str = 'amount',
        merchant_col: Optional[str] = 'merchant_id'
    ) -> pd.DataFrame:
        """
        Create anomaly scores for transactions.
        
        Args:
            transactions: Transaction DataFrame
            amount_col: Amount column
            merchant_col: Merchant ID column (optional)
            
        Returns:
            DataFrame with anomaly scores
        """
        logger.info("Creating anomaly scores")
        
        df = transactions.copy()
        features = {}
        
        # Amount z-score (global)
        features['amount_zscore'] = (
            (df[amount_col] - df[amount_col].mean()) / 
            df[amount_col].std()
        )
        
        # Amount percentile (global)
        features['amount_percentile'] = df[amount_col].rank(pct=True)
        
        # Merchant-specific anomaly (if available)
        if merchant_col and merchant_col in df.columns:
            merchant_stats = df.groupby(merchant_col)[amount_col].agg(['mean', 'std'])
            df = df.join(merchant_stats, on=merchant_col, rsuffix='_merchant')
            
            features['amount_zscore_merchant'] = (
                (df[amount_col] - df['mean']) / 
                df['std'].replace(0, 1)
            )
        
        # Time-of-day anomaly
        if 'timestamp' in df.columns:
            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
            hour_stats = df.groupby('hour')[amount_col].agg(['mean', 'std'])
            df = df.join(hour_stats, on='hour', rsuffix='_hour')
            
            features['amount_zscore_hour'] = (
                (df[amount_col] - df['mean']) / 
                df['std'].replace(0, 1)
            )
        
        result_df = pd.DataFrame(features).fillna(0)
        self.feature_names.extend(result_df.columns.tolist())
        
        logger.info(f"Created {len(features)} anomaly score features")
        return result_df
    
    def integrate_graph_features(
        self,
        base_features: pd.DataFrame,
        graph_features: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Integrate graph-based features with transaction features.
        
        Args:
            base_features: Base transaction features
            graph_features: Graph-based features
            
        Returns:
            Combined feature DataFrame
        """
        logger.info("Integrating graph features")
        
        combined = pd.concat([base_features, graph_features], axis=1)
        self.feature_names.extend(graph_features.columns.tolist())
        
        logger.info(f"Total features: {combined.shape[1]}")
        return combined


class FraudClassifier:
    """
    Production-ready fraud classifier with multiple model backends.
    
    Handles extreme class imbalance, provides calibrated probabilities,
    and supports real-time scoring with <100ms latency.
    
    Example:
        >>> classifier = FraudClassifier(model_type='lightgbm')
        >>> classifier.fit(X_train, y_train, X_val, y_val)
        >>> scores = classifier.predict_proba(X_test)
        >>> optimal_threshold = classifier.optimize_threshold(
        ...     X_val, y_val, target_recall=0.8
        ... )
    """
    
    def __init__(
        self,
        model_type: str = 'lightgbm',
        calibrate: bool = True,
        params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize fraud classifier.
        
        Args:
            model_type: Model type ('lightgbm', 'xgboost', 'neural')
            calibrate: Whether to calibrate probabilities
            params: Model hyperparameters
            
        Example:
            >>> classifier = FraudClassifier(
            ...     model_type='lightgbm',
            ...     calibrate=True
            ... )
        """
        self.model_type = model_type
        self.calibrate = calibrate
        self.params = params or self._get_default_params()
        self.model = None
        self.calibrator = None
        self.scaler = None
        self.feature_importance_: Optional[pd.DataFrame] = None
        self.threshold_: float = 0.5
        self.metrics_: Dict[str, float] = {}
        
        logger.info(f"Initialized FraudClassifier with {model_type}")
    
    def _get_default_params(self) -> Dict[str, Any]:
        """Get default parameters for the selected model type."""
        if self.model_type == 'lightgbm':
            return {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'max_depth': 7,
                'min_child_samples': 100,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'scale_pos_weight': 100,  # For class imbalance
                'verbose': -1
            }
        elif self.model_type == 'xgboost':
            return {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'max_depth': 7,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 10,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'scale_pos_weight': 100,
                'tree_method': 'hist'
            }
        elif self.model_type == 'neural':
            return {
                'hidden_dims': [128, 64, 32],
                'dropout': 0.3,
                'learning_rate': 0.001,
                'batch_size': 256,
                'epochs': 50,
                'early_stopping_rounds': 10
            }
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def fit(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None,
        sample_weight: Optional[np.ndarray] = None
    ) -> 'FraudClassifier':
        """
        Fit fraud classifier.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            sample_weight: Sample weights for training
            
        Returns:
            Self
            
        Example:
            >>> classifier.fit(X_train, y_train, X_val, y_val)
        """
        logger.info(f"Training {self.model_type} fraud classifier")
        logger.info(f"Training samples: {len(X_train)}, Fraud rate: {y_train.mean():.4f}")
        
        # Handle class imbalance with sample weights if not provided
        if sample_weight is None and hasattr(y_train, '__len__'):
            fraud_rate = np.mean(y_train)
            if fraud_rate < 0.001:  # Less than 0.1%
                logger.warning(f"Extreme class imbalance detected: {fraud_rate:.4%}")
                # Compute balanced sample weights
                pos_weight = (1 - fraud_rate) / fraud_rate
                sample_weight = np.where(y_train == 1, pos_weight, 1.0)
        
        if self.model_type == 'lightgbm':
            self._fit_lightgbm(X_train, y_train, X_val, y_val, sample_weight)
        elif self.model_type == 'xgboost':
            self._fit_xgboost(X_train, y_train, X_val, y_val, sample_weight)
        elif self.model_type == 'neural':
            self._fit_neural(X_train, y_train, X_val, y_val, sample_weight)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Calibrate probabilities
        if self.calibrate and X_val is not None and y_val is not None:
            logger.info("Calibrating probabilities")
            self._calibrate_probabilities(X_val, y_val)
        
        # Compute feature importance
        self._compute_feature_importance(X_train)
        
        # Evaluate on validation set
        if X_val is not None and y_val is not None:
            self._evaluate(X_val, y_val)
        
        logger.info("Training completed successfully")
        return self
    
    def _fit_lightgbm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
        sample_weight: Optional[np.ndarray]
    ):
        """Fit LightGBM model."""
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is required but not installed")
        
        train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weight)
        valid_sets = [train_data]
        
        if X_val is not None and y_val is not None:
            valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(valid_data)
        
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=1000,
            valid_sets=valid_sets,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=100)
            ]
        )
    
    def _fit_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
        sample_weight: Optional[np.ndarray]
    ):
        """Fit XGBoost model."""
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is required but not installed")
        
        dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weight)
        evals = [(dtrain, 'train')]
        
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals.append((dval, 'valid'))
        
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=1000,
            evals=evals,
            early_stopping_rounds=50,
            verbose_eval=100
        )
    
    def _fit_neural(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
        sample_weight: Optional[np.ndarray]
    ):
        """Fit neural network model."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required but not installed")
        
        # Scale features for neural network
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
        else:
            X_val_scaled = None
        
        # Create and train neural network
        self.model = FraudNeuralNet(
            input_dim=X_train.shape[1],
            hidden_dims=self.params['hidden_dims'],
            dropout=self.params['dropout']
        )
        
        self.model.fit(
            X_train_scaled, y_train,
            X_val_scaled, y_val,
            sample_weight=sample_weight,
            learning_rate=self.params['learning_rate'],
            batch_size=self.params['batch_size'],
            epochs=self.params['epochs'],
            early_stopping_rounds=self.params['early_stopping_rounds']
        )
    
    def _calibrate_probabilities(self, X_val: np.ndarray, y_val: np.ndarray):
        """Calibrate probability predictions using validation set."""
        val_preds = self._predict_proba_raw(X_val)
        
        # Use isotonic calibration for better calibration on imbalanced data
        from sklearn.isotonic import IsotonicRegression
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        self.calibrator.fit(val_preds, y_val)
        
        logger.info("Probability calibration completed")
    
    def predict_proba(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        Predict fraud probabilities.
        
        Args:
            X: Features
            
        Returns:
            Fraud probabilities
            
        Example:
            >>> proba = classifier.predict_proba(X_test)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        scores = self._predict_proba_raw(X)
        
        if self.calibrator is not None:
            scores = self.calibrator.predict(scores)
        
        return scores
    
    def _predict_proba_raw(self, X: np.ndarray) -> np.ndarray:
        """Get raw probability predictions from model."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        if self.model_type == 'lightgbm':
            return self.model.predict(X, num_iteration=self.model.best_iteration)
        elif self.model_type == 'xgboost':
            dtest = xgb.DMatrix(X)
            return self.model.predict(dtest)
        elif self.model_type == 'neural':
            if self.scaler is not None:
                X = self.scaler.transform(X)
            return self.model.predict(X)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        threshold: Optional[float] = None
    ) -> np.ndarray:
        """
        Predict fraud labels.
        
        Args:
            X: Features
            threshold: Decision threshold (uses optimized threshold if None)
            
        Returns:
            Binary predictions
        """
        proba = self.predict_proba(X)
        threshold = threshold or self.threshold_
        return (proba >= threshold).astype(int)
    
    def optimize_threshold(
        self,
        X_val: Union[pd.DataFrame, np.ndarray],
        y_val: Union[pd.Series, np.ndarray],
        target_recall: Optional[float] = None,
        target_precision: Optional[float] = None,
        optimize_f1: bool = False
    ) -> float:
        """
        Optimize decision threshold for desired metrics.
        
        Args:
            X_val: Validation features
            y_val: Validation labels
            target_recall: Target recall (fraud detection rate)
            target_precision: Target precision
            optimize_f1: Optimize F1 score
            
        Returns:
            Optimal threshold
            
        Example:
            >>> # Optimize for 80% recall
            >>> threshold = classifier.optimize_threshold(
            ...     X_val, y_val, target_recall=0.8
            ... )
        """
        logger.info("Optimizing decision threshold")
        
        proba = self.predict_proba(X_val)
        precision, recall, thresholds = precision_recall_curve(y_val, proba)
        
        if target_recall is not None:
            # Find threshold that achieves target recall
            idx = np.argmin(np.abs(recall - target_recall))
            optimal_threshold = thresholds[idx]
            logger.info(
                f"Threshold for {target_recall:.1%} recall: {optimal_threshold:.4f} "
                f"(precision: {precision[idx]:.4f})"
            )
        elif target_precision is not None:
            # Find threshold that achieves target precision
            idx = np.argmin(np.abs(precision - target_precision))
            optimal_threshold = thresholds[idx]
            logger.info(
                f"Threshold for {target_precision:.1%} precision: {optimal_threshold:.4f} "
                f"(recall: {recall[idx]:.4f})"
            )
        elif optimize_f1:
            # Find threshold that maximizes F1 score
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
            idx = np.argmax(f1_scores)
            optimal_threshold = thresholds[idx]
            logger.info(
                f"Threshold for max F1: {optimal_threshold:.4f} "
                f"(F1: {f1_scores[idx]:.4f}, precision: {precision[idx]:.4f}, "
                f"recall: {recall[idx]:.4f})"
            )
        else:
            # Default: balance precision and recall
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
            idx = np.argmax(f1_scores)
            optimal_threshold = thresholds[idx]
            logger.info(f"Balanced threshold: {optimal_threshold:.4f}")
        
        self.threshold_ = optimal_threshold
        return optimal_threshold
    
    def _compute_feature_importance(self, X: Union[pd.DataFrame, np.ndarray]):
        """Compute feature importance."""
        if self.model_type == 'lightgbm':
            importance = self.model.feature_importance(importance_type='gain')
            if isinstance(X, pd.DataFrame):
                feature_names = X.columns.tolist()
            else:
                feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            
            self.feature_importance_ = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
        elif self.model_type == 'xgboost':
            importance = self.model.get_score(importance_type='gain')
            self.feature_importance_ = pd.DataFrame([
                {'feature': k, 'importance': v}
                for k, v in importance.items()
            ]).sort_values('importance', ascending=False)
    
    def _evaluate(self, X_val: np.ndarray, y_val: np.ndarray):
        """Evaluate model on validation set."""
        proba = self.predict_proba(X_val)
        preds = self.predict(X_val, threshold=self.threshold_)
        
        self.metrics_ = {
            'auc_roc': roc_auc_score(y_val, proba),
            'auc_pr': average_precision_score(y_val, proba),
            'recall': np.mean(preds[y_val == 1]),
            'precision': np.mean(y_val[preds == 1]) if preds.sum() > 0 else 0,
        }
        
        logger.info(f"Validation metrics: {self.metrics_}")
    
    def save(self, path: Union[str, Path]):
        """
        Save model to disk.
        
        Args:
            path: Save path
            
        Example:
            >>> classifier.save('models/fraud_classifier.pkl')
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model_type': self.model_type,
            'model': self.model,
            'calibrator': self.calibrator,
            'scaler': self.scaler,
            'threshold': self.threshold_,
            'params': self.params,
            'metrics': self.metrics_,
            'feature_importance': self.feature_importance_
        }
        
        joblib.dump(model_data, path)
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'FraudClassifier':
        """
        Load model from disk.
        
        Args:
            path: Model path
            
        Returns:
            Loaded FraudClassifier
            
        Example:
            >>> classifier = FraudClassifier.load('models/fraud_classifier.pkl')
        """
        model_data = joblib.load(path)
        
        classifier = cls(
            model_type=model_data['model_type'],
            params=model_data['params']
        )
        classifier.model = model_data['model']
        classifier.calibrator = model_data['calibrator']
        classifier.scaler = model_data['scaler']
        classifier.threshold_ = model_data['threshold']
        classifier.metrics_ = model_data['metrics']
        classifier.feature_importance_ = model_data['feature_importance']
        
        logger.info(f"Model loaded from {path}")
        return classifier


class FraudNeuralNet(nn.Module):
    """Neural network for fraud detection."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [128, 64, 32],
        dropout: float = 0.3
    ):
        """
        Initialize fraud neural network.
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: Hidden layer dimensions
            dropout: Dropout rate
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None,
        learning_rate: float = 0.001,
        batch_size: int = 256,
        epochs: int = 50,
        early_stopping_rounds: int = 10
    ):
        """
        Fit neural network.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            sample_weight: Sample weights
            learning_rate: Learning rate
            batch_size: Batch size
            epochs: Number of epochs
            early_stopping_rounds: Early stopping patience
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.BCELoss()
        
        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(self.device)
        
        if sample_weight is not None:
            sample_weight_t = torch.FloatTensor(sample_weight).to(self.device)
        else:
            sample_weight_t = None
        
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            self.train()
            total_loss = 0
            
            # Mini-batch training
            indices = np.random.permutation(len(X_train))
            for i in range(0, len(X_train), batch_size):
                batch_indices = indices[i:i+batch_size]
                X_batch = X_train_t[batch_indices]
                y_batch = y_train_t[batch_indices]
                
                optimizer.zero_grad()
                outputs = self(X_batch)
                
                if sample_weight_t is not None:
                    weights = sample_weight_t[batch_indices].unsqueeze(1)
                    loss = (criterion(outputs, y_batch) * weights).mean()
                else:
                    loss = criterion(outputs, y_batch)
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            # Validation
            if X_val is not None and y_val is not None:
                self.eval()
                with torch.no_grad():
                    X_val_t = torch.FloatTensor(X_val).to(self.device)
                    y_val_t = torch.FloatTensor(y_val).unsqueeze(1).to(self.device)
                    val_outputs = self(X_val_t)
                    val_loss = criterion(val_outputs, y_val_t).item()
                
                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= early_stopping_rounds:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
                
                if (epoch + 1) % 10 == 0:
                    logger.info(
                        f"Epoch {epoch+1}/{epochs}: "
                        f"train_loss={total_loss/len(X_train):.4f}, "
                        f"val_loss={val_loss:.4f}"
                    )
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        self.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.device)
            outputs = self(X_t)
            return outputs.cpu().numpy().flatten()
