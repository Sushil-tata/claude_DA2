"""
Learning to rank models for recommendation systems.

This module provides various ranking algorithms optimized for different ranking
metrics (NDCG, MAP, MRR) with support for position bias correction and A/B testing.

Example:
    >>> from src.recommender.ranking_model import LambdaMARTRanker
    >>> ranker = LambdaMARTRanker(n_estimators=100)
    >>> ranker.fit(X_train, y_train, group_train)
    >>> rankings = ranker.predict(X_test)
"""

import joblib
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from loguru import logger
from sklearn.base import BaseEstimator
from sklearn.metrics import ndcg_score

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    logger.warning("LightGBM not available")
    LIGHTGBM_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    logger.warning("XGBoost not available")
    XGBOOST_AVAILABLE = False


class BaseRankingModel:
    """Base class for ranking models."""
    
    def __init__(self):
        """Initialize base ranking model."""
        self.is_fitted = False
        
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: np.ndarray,
        group: np.ndarray,
        **kwargs
    ) -> 'BaseRankingModel':
        """
        Fit ranking model.
        
        Args:
            X: Features
            y: Relevance labels (higher is better)
            group: Query/group identifiers or sizes
            **kwargs: Additional arguments
            
        Returns:
            Fitted model
        """
        raise NotImplementedError
        
    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        group: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Predict ranking scores.
        
        Args:
            X: Features
            group: Optional query/group identifiers
            
        Returns:
            Ranking scores
        """
        raise NotImplementedError
        
    def rank(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        group: np.ndarray,
        k: Optional[int] = None
    ) -> List[np.ndarray]:
        """
        Rank items within each group.
        
        Args:
            X: Features
            group: Query/group identifiers
            k: Return top-k items per group (None = all)
            
        Returns:
            List of ranked indices per group
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before ranking")
            
        scores = self.predict(X, group)
        
        # Group items and rank
        unique_groups = np.unique(group)
        rankings = []
        
        for g in unique_groups:
            group_mask = group == g
            group_scores = scores[group_mask]
            group_indices = np.where(group_mask)[0]
            
            # Sort by score (descending)
            sorted_idx = np.argsort(-group_scores)
            
            if k is not None:
                sorted_idx = sorted_idx[:k]
                
            rankings.append(group_indices[sorted_idx])
            
        return rankings
        
    def save(self, filepath: Union[str, Path]) -> None:
        """Save model to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        logger.info(f"Ranking model saved to {filepath}")
        
    @staticmethod
    def load(filepath: Union[str, Path]) -> 'BaseRankingModel':
        """Load model from disk."""
        model = joblib.load(filepath)
        logger.info(f"Ranking model loaded from {filepath}")
        return model


class LambdaMARTRanker(BaseRankingModel):
    """
    LambdaMART ranker using LightGBM.
    
    Gradient boosted ranking using LambdaRank algorithm optimized for NDCG.
    
    Example:
        >>> ranker = LambdaMARTRanker(n_estimators=100, learning_rate=0.1)
        >>> ranker.fit(X_train, y_train, group_train)
        >>> scores = ranker.predict(X_test)
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        num_leaves: int = 31,
        min_child_samples: int = 20,
        ndcg_at: List[int] = [1, 3, 5, 10],
        random_state: Optional[int] = None
    ):
        """
        Initialize LambdaMART ranker.
        
        Args:
            n_estimators: Number of boosting iterations
            learning_rate: Learning rate
            max_depth: Maximum tree depth
            num_leaves: Maximum leaves per tree
            min_child_samples: Minimum samples per leaf
            ndcg_at: Positions to evaluate NDCG
            random_state: Random seed
        """
        super().__init__()
        
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM required. Install with: pip install lightgbm")
            
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.ndcg_at = ndcg_at
        self.random_state = random_state
        
        self.model = None
        self.feature_importance_: Optional[np.ndarray] = None
        
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: np.ndarray,
        group: np.ndarray,
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[np.ndarray] = None,
        group_val: Optional[np.ndarray] = None,
        early_stopping_rounds: int = 50,
        verbose: int = 100
    ) -> 'LambdaMARTRanker':
        """
        Fit LambdaMART ranker.
        
        Args:
            X: Training features
            y: Training relevance labels
            group: Training group sizes or identifiers
            X_val: Validation features
            y_val: Validation labels
            group_val: Validation groups
            early_stopping_rounds: Early stopping patience
            verbose: Verbosity
            
        Returns:
            Fitted ranker
        """
        X = self._convert_to_array(X)
        y = np.asarray(y)
        
        # Convert group identifiers to group sizes if needed
        if len(group.shape) == 1 and len(np.unique(group)) < len(group):
            group_sizes = self._get_group_sizes(group)
        else:
            group_sizes = group
            
        logger.info(
            f"Fitting LambdaMART: {X.shape[0]} samples, "
            f"{len(group_sizes)} groups"
        )
        
        # Create LightGBM dataset
        train_data = lgb.Dataset(X, label=y, group=group_sizes)
        
        params = {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'ndcg_eval_at': self.ndcg_at,
            'boosting_type': 'gbdt',
            'num_leaves': self.num_leaves,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'min_child_samples': self.min_child_samples,
            'random_state': self.random_state,
            'verbose': -1
        }
        
        # Prepare validation set
        valid_sets = [train_data]
        valid_names = ['train']
        
        if X_val is not None and y_val is not None and group_val is not None:
            X_val = self._convert_to_array(X_val)
            y_val = np.asarray(y_val)
            
            if len(group_val.shape) == 1 and len(np.unique(group_val)) < len(group_val):
                group_val_sizes = self._get_group_sizes(group_val)
            else:
                group_val_sizes = group_val
                
            val_data = lgb.Dataset(X_val, label=y_val, group=group_val_sizes)
            valid_sets.append(val_data)
            valid_names.append('valid')
            
        # Train model
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=self.n_estimators,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=[
                lgb.early_stopping(early_stopping_rounds, verbose=False),
                lgb.log_evaluation(verbose)
            ]
        )
        
        self.feature_importance_ = self.model.feature_importance(importance_type='gain')
        self.is_fitted = True
        
        logger.info("LambdaMART fitted successfully")
        return self
        
    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        group: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Predict ranking scores."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        X = self._convert_to_array(X)
        scores = self.model.predict(X)
        
        return scores
        
    def get_feature_importance(
        self,
        importance_type: str = 'gain',
        feature_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Get feature importance.
        
        Args:
            importance_type: 'gain' or 'split'
            feature_names: Optional feature names
            
        Returns:
            DataFrame with feature importance
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted")
            
        importance = self.model.feature_importance(importance_type=importance_type)
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importance))]
            
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        })
        
        return df.sort_values('importance', ascending=False)
        
    @staticmethod
    def _convert_to_array(X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Convert input to numpy array."""
        if isinstance(X, pd.DataFrame):
            return X.values
        return np.asarray(X)
        
    @staticmethod
    def _get_group_sizes(group: np.ndarray) -> np.ndarray:
        """Convert group identifiers to group sizes."""
        unique_groups, counts = np.unique(group, return_counts=True)
        return counts


class PairwiseRanker(BaseRankingModel):
    """
    Pairwise ranking using gradient boosting.
    
    Learns to compare pairs of items using pairwise loss.
    
    Example:
        >>> ranker = PairwiseRanker()
        >>> ranker.fit(X_train, y_train, group_train)
        >>> scores = ranker.predict(X_test)
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        random_state: Optional[int] = None
    ):
        """
        Initialize Pairwise ranker.
        
        Args:
            n_estimators: Number of boosting iterations
            learning_rate: Learning rate
            max_depth: Maximum tree depth
            random_state: Random seed
        """
        super().__init__()
        
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost required. Install with: pip install xgboost")
            
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        
        self.model = None
        
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: np.ndarray,
        group: np.ndarray,
        **kwargs
    ) -> 'PairwiseRanker':
        """Fit pairwise ranker."""
        X = self._convert_to_array(X)
        y = np.asarray(y)
        
        # Convert group identifiers to group sizes if needed
        if len(group.shape) == 1 and len(np.unique(group)) < len(group):
            group_sizes = self._get_group_sizes(group)
        else:
            group_sizes = group
            
        logger.info(
            f"Fitting Pairwise ranker: {X.shape[0]} samples, "
            f"{len(group_sizes)} groups"
        )
        
        # Create XGBoost DMatrix
        dtrain = xgb.DMatrix(X, label=y)
        dtrain.set_group(group_sizes)
        
        params = {
            'objective': 'rank:pairwise',
            'eval_metric': 'ndcg',
            'eta': self.learning_rate,
            'max_depth': self.max_depth,
            'seed': self.random_state
        }
        
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=self.n_estimators,
            verbose_eval=False
        )
        
        self.is_fitted = True
        logger.info("Pairwise ranker fitted successfully")
        
        return self
        
    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        group: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Predict ranking scores."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        X = self._convert_to_array(X)
        dtest = xgb.DMatrix(X)
        scores = self.model.predict(dtest)
        
        return scores
        
    @staticmethod
    def _convert_to_array(X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Convert input to numpy array."""
        if isinstance(X, pd.DataFrame):
            return X.values
        return np.asarray(X)
        
    @staticmethod
    def _get_group_sizes(group: np.ndarray) -> np.ndarray:
        """Convert group identifiers to group sizes."""
        unique_groups, counts = np.unique(group, return_counts=True)
        return counts


class ListwiseRanker(BaseRankingModel):
    """
    Listwise ranking model.
    
    Optimizes listwise loss considering entire ranked lists.
    
    Example:
        >>> ranker = ListwiseRanker()
        >>> ranker.fit(X_train, y_train, group_train)
        >>> scores = ranker.predict(X_test)
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        random_state: Optional[int] = None
    ):
        """
        Initialize Listwise ranker.
        
        Args:
            n_estimators: Number of boosting iterations
            learning_rate: Learning rate
            max_depth: Maximum tree depth
            random_state: Random seed
        """
        super().__init__()
        
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost required")
            
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        
        self.model = None
        
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: np.ndarray,
        group: np.ndarray,
        **kwargs
    ) -> 'ListwiseRanker':
        """Fit listwise ranker."""
        X = self._convert_to_array(X)
        y = np.asarray(y)
        
        if len(group.shape) == 1 and len(np.unique(group)) < len(group):
            group_sizes = self._get_group_sizes(group)
        else:
            group_sizes = group
            
        logger.info(
            f"Fitting Listwise ranker: {X.shape[0]} samples, "
            f"{len(group_sizes)} groups"
        )
        
        dtrain = xgb.DMatrix(X, label=y)
        dtrain.set_group(group_sizes)
        
        params = {
            'objective': 'rank:ndcg',
            'eval_metric': 'ndcg',
            'eta': self.learning_rate,
            'max_depth': self.max_depth,
            'seed': self.random_state
        }
        
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=self.n_estimators,
            verbose_eval=False
        )
        
        self.is_fitted = True
        logger.info("Listwise ranker fitted successfully")
        
        return self
        
    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        group: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Predict ranking scores."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        X = self._convert_to_array(X)
        dtest = xgb.DMatrix(X)
        scores = self.model.predict(dtest)
        
        return scores
        
    @staticmethod
    def _convert_to_array(X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Convert input to numpy array."""
        if isinstance(X, pd.DataFrame):
            return X.values
        return np.asarray(X)
        
    @staticmethod
    def _get_group_sizes(group: np.ndarray) -> np.ndarray:
        """Convert group identifiers to group sizes."""
        unique_groups, counts = np.unique(group, return_counts=True)
        return counts


class NDCGOptimizer(BaseRankingModel):
    """
    Direct NDCG optimization using custom objective.
    
    Optimizes directly for NDCG metric at specific k values.
    
    Example:
        >>> ranker = NDCGOptimizer(k=10)
        >>> ranker.fit(X_train, y_train, group_train)
        >>> scores = ranker.predict(X_test)
    """
    
    def __init__(
        self,
        k: int = 10,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        random_state: Optional[int] = None
    ):
        """
        Initialize NDCG optimizer.
        
        Args:
            k: Top-k for NDCG calculation
            n_estimators: Number of boosting iterations
            learning_rate: Learning rate
            random_state: Random seed
        """
        super().__init__()
        
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM required")
            
        self.k = k
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        
        self.model = None
        
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: np.ndarray,
        group: np.ndarray,
        **kwargs
    ) -> 'NDCGOptimizer':
        """Fit NDCG optimizer."""
        X = self._convert_to_array(X)
        y = np.asarray(y)
        
        if len(group.shape) == 1 and len(np.unique(group)) < len(group):
            group_sizes = self._get_group_sizes(group)
        else:
            group_sizes = group
            
        logger.info(
            f"Fitting NDCG@{self.k} optimizer: {X.shape[0]} samples, "
            f"{len(group_sizes)} groups"
        )
        
        train_data = lgb.Dataset(X, label=y, group=group_sizes)
        
        params = {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'ndcg_eval_at': [self.k],
            'learning_rate': self.learning_rate,
            'random_state': self.random_state,
            'verbose': -1
        }
        
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=self.n_estimators
        )
        
        self.is_fitted = True
        logger.info(f"NDCG@{self.k} optimizer fitted successfully")
        
        return self
        
    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        group: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Predict ranking scores."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        X = self._convert_to_array(X)
        scores = self.model.predict(X)
        
        return scores
        
    @staticmethod
    def _convert_to_array(X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Convert input to numpy array."""
        if isinstance(X, pd.DataFrame):
            return X.values
        return np.asarray(X)
        
    @staticmethod
    def _get_group_sizes(group: np.ndarray) -> np.ndarray:
        """Convert group identifiers to group sizes."""
        unique_groups, counts = np.unique(group, return_counts=True)
        return counts


class RankingEnsemble:
    """
    Ensemble of ranking models.
    
    Combines multiple rankers for robust predictions.
    
    Example:
        >>> models = [
        ...     LambdaMARTRanker(),
        ...     PairwiseRanker(),
        ...     ListwiseRanker()
        ... ]
        >>> ensemble = RankingEnsemble(models)
        >>> ensemble.fit(X_train, y_train, group_train)
        >>> scores = ensemble.predict(X_test)
    """
    
    def __init__(
        self,
        models: List[BaseRankingModel],
        weights: Optional[List[float]] = None
    ):
        """
        Initialize ranking ensemble.
        
        Args:
            models: List of ranking models
            weights: Optional model weights (uniform if None)
        """
        self.models = models
        
        if weights is None:
            self.weights = np.ones(len(models)) / len(models)
        else:
            self.weights = np.array(weights)
            self.weights = self.weights / self.weights.sum()
            
        self.is_fitted = False
        
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: np.ndarray,
        group: np.ndarray,
        **kwargs
    ) -> 'RankingEnsemble':
        """Fit all models in ensemble."""
        logger.info(f"Fitting {len(self.models)} ranking models")
        
        for i, model in enumerate(self.models):
            logger.info(f"Fitting model {i+1}/{len(self.models)}: {type(model).__name__}")
            model.fit(X, y, group, **kwargs)
            
        self.is_fitted = True
        logger.info("Ensemble fitted successfully")
        
        return self
        
    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        group: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Predict weighted average of scores."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
            
        predictions = np.array([
            model.predict(X, group) for model in self.models
        ])
        
        weighted_pred = np.average(predictions, axis=0, weights=self.weights)
        return weighted_pred
        
    def save(self, directory: Union[str, Path]) -> None:
        """Save ensemble to directory."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        for i, model in enumerate(self.models):
            model.save(directory / f"model_{i}.pkl")
            
        config = {
            'weights': self.weights,
            'is_fitted': self.is_fitted,
            'n_models': len(self.models)
        }
        joblib.dump(config, directory / 'ensemble_config.pkl')
        
        logger.info(f"Ensemble saved to {directory}")
        
    @staticmethod
    def load(directory: Union[str, Path]) -> 'RankingEnsemble':
        """Load ensemble from directory."""
        directory = Path(directory)
        config = joblib.load(directory / 'ensemble_config.pkl')
        
        models = []
        for i in range(config['n_models']):
            model = BaseRankingModel.load(directory / f"model_{i}.pkl")
            models.append(model)
            
        ensemble = RankingEnsemble(models=models, weights=config['weights'])
        ensemble.is_fitted = config['is_fitted']
        
        logger.info(f"Ensemble loaded from {directory}")
        return ensemble


class PositionBiasCorrector:
    """
    Corrects for position bias in ranking data.
    
    Accounts for the fact that items at higher positions receive more attention.
    
    Example:
        >>> corrector = PositionBiasCorrector()
        >>> corrector.fit(positions, clicks)
        >>> corrected_relevance = corrector.correct(y, positions)
    """
    
    def __init__(self, method: str = 'inverse_propensity'):
        """
        Initialize position bias corrector.
        
        Args:
            method: 'inverse_propensity' or 'examination_hypothesis'
        """
        self.method = method
        self.position_probs: Optional[Dict[int, float]] = None
        self.is_fitted = False
        
    def fit(
        self,
        positions: np.ndarray,
        clicks: np.ndarray,
        max_position: int = 20
    ) -> 'PositionBiasCorrector':
        """
        Fit position bias model.
        
        Args:
            positions: Item positions (0-indexed)
            clicks: Click indicators
            max_position: Maximum position to model
            
        Returns:
            Fitted corrector
        """
        positions = np.asarray(positions)
        clicks = np.asarray(clicks)
        
        # Calculate click-through rate by position
        self.position_probs = {}
        
        for pos in range(max_position):
            pos_mask = positions == pos
            if pos_mask.sum() > 0:
                ctr = clicks[pos_mask].mean()
                self.position_probs[pos] = max(ctr, 0.01)  # Avoid division by zero
            else:
                # Use position decay for unseen positions
                self.position_probs[pos] = 1.0 / np.log2(pos + 2)
                
        self.is_fitted = True
        logger.info(f"Position bias corrector fitted for {max_position} positions")
        
        return self
        
    def correct(
        self,
        relevance: np.ndarray,
        positions: np.ndarray
    ) -> np.ndarray:
        """
        Correct relevance scores for position bias.
        
        Args:
            relevance: Original relevance scores
            positions: Item positions
            
        Returns:
            Corrected relevance scores
        """
        if not self.is_fitted:
            raise ValueError("Corrector must be fitted before use")
            
        relevance = np.asarray(relevance)
        positions = np.asarray(positions)
        
        if self.method == 'inverse_propensity':
            # Inverse propensity weighting
            corrected = np.zeros_like(relevance, dtype=float)
            
            for i, pos in enumerate(positions):
                prob = self.position_probs.get(pos, self.position_probs.get(max(self.position_probs.keys())))
                corrected[i] = relevance[i] / prob
                
            return corrected
        else:
            raise ValueError(f"Unknown method: {self.method}")


class RankingMetrics:
    """
    Ranking evaluation metrics.
    
    Provides NDCG, MAP, MRR and other ranking metrics.
    
    Example:
        >>> metrics = RankingMetrics()
        >>> ndcg = metrics.ndcg(y_true, y_pred, k=10)
        >>> map_score = metrics.mean_average_precision(y_true, y_pred)
    """
    
    @staticmethod
    def ndcg(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        k: Optional[int] = None,
        group: Optional[np.ndarray] = None
    ) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain.
        
        Args:
            y_true: True relevance scores
            y_pred: Predicted scores
            k: Top-k for NDCG calculation
            group: Query groups
            
        Returns:
            NDCG score
        """
        if group is None:
            # Single query
            return ndcg_score([y_true], [y_pred], k=k)
        else:
            # Multiple queries
            unique_groups = np.unique(group)
            ndcg_scores = []
            
            for g in unique_groups:
                group_mask = group == g
                y_true_g = y_true[group_mask]
                y_pred_g = y_pred[group_mask]
                
                if len(y_true_g) > 0:
                    score = ndcg_score([y_true_g], [y_pred_g], k=k)
                    ndcg_scores.append(score)
                    
            return np.mean(ndcg_scores)
            
    @staticmethod
    def mean_average_precision(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        group: Optional[np.ndarray] = None
    ) -> float:
        """
        Calculate Mean Average Precision.
        
        Args:
            y_true: True relevance (binary)
            y_pred: Predicted scores
            group: Query groups
            
        Returns:
            MAP score
        """
        if group is None:
            # Single query
            sorted_idx = np.argsort(-y_pred)
            y_true_sorted = y_true[sorted_idx]
            
            relevant_positions = np.where(y_true_sorted == 1)[0] + 1
            if len(relevant_positions) == 0:
                return 0.0
                
            precisions = np.arange(1, len(relevant_positions) + 1) / relevant_positions
            return np.mean(precisions)
        else:
            # Multiple queries
            unique_groups = np.unique(group)
            ap_scores = []
            
            for g in unique_groups:
                group_mask = group == g
                y_true_g = y_true[group_mask]
                y_pred_g = y_pred[group_mask]
                
                if len(y_true_g) > 0:
                    ap = RankingMetrics.mean_average_precision(y_true_g, y_pred_g)
                    ap_scores.append(ap)
                    
            return np.mean(ap_scores)
            
    @staticmethod
    def mean_reciprocal_rank(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        group: Optional[np.ndarray] = None
    ) -> float:
        """
        Calculate Mean Reciprocal Rank.
        
        Args:
            y_true: True relevance (binary)
            y_pred: Predicted scores
            group: Query groups
            
        Returns:
            MRR score
        """
        if group is None:
            # Single query
            sorted_idx = np.argsort(-y_pred)
            y_true_sorted = y_true[sorted_idx]
            
            relevant_positions = np.where(y_true_sorted == 1)[0]
            if len(relevant_positions) == 0:
                return 0.0
                
            return 1.0 / (relevant_positions[0] + 1)
        else:
            # Multiple queries
            unique_groups = np.unique(group)
            rr_scores = []
            
            for g in unique_groups:
                group_mask = group == g
                y_true_g = y_true[group_mask]
                y_pred_g = y_pred[group_mask]
                
                if len(y_true_g) > 0:
                    rr = RankingMetrics.mean_reciprocal_rank(y_true_g, y_pred_g)
                    rr_scores.append(rr)
                    
            return np.mean(rr_scores)
            
    @staticmethod
    def precision_at_k(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        k: int,
        group: Optional[np.ndarray] = None
    ) -> float:
        """
        Calculate Precision@k.
        
        Args:
            y_true: True relevance (binary)
            y_pred: Predicted scores
            k: Top-k
            group: Query groups
            
        Returns:
            Precision@k score
        """
        if group is None:
            # Single query
            sorted_idx = np.argsort(-y_pred)[:k]
            y_true_topk = y_true[sorted_idx]
            return y_true_topk.sum() / k
        else:
            # Multiple queries
            unique_groups = np.unique(group)
            precision_scores = []
            
            for g in unique_groups:
                group_mask = group == g
                y_true_g = y_true[group_mask]
                y_pred_g = y_pred[group_mask]
                
                if len(y_true_g) > 0:
                    p_at_k = RankingMetrics.precision_at_k(y_true_g, y_pred_g, k)
                    precision_scores.append(p_at_k)
                    
            return np.mean(precision_scores)
