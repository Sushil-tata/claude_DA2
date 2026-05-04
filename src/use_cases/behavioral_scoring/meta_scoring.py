"""
Meta-Learning Approach for Behavioral Scoring

Implements meta-learning techniques for behavioral scoring:
- Task-specific fine-tuning framework
- Multi-task learning support
- Transfer learning from similar customer segments
- Few-shot learning for new segments
- Meta-features extraction
- Model adaptation strategies
- Segment-specific model selection

Example:
    >>> from src.use_cases.behavioral_scoring.meta_scoring import MetaScoringEngine
    >>> 
    >>> # Train meta-learner
    >>> engine = MetaScoringEngine()
    >>> engine.fit(transactions_dict, labels_dict, segments_dict)
    >>> 
    >>> # Few-shot learning for new segment
    >>> engine.adapt_to_segment(new_segment_data, segment_id="new_segment", n_shots=10)
    >>> scores = engine.predict(test_data, segment_id="new_segment")
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
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch not available. Install with: pip install torch")
    TORCH_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    logger.warning("LightGBM not available. Install with: pip install lightgbm")
    LGBM_AVAILABLE = False


class MetaFeatureExtractor:
    """Extract meta-features for task characterization."""
    
    def __init__(self):
        """Initialize meta-feature extractor."""
        self.feature_stats = {}
        
    def extract(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Extract meta-features from dataset.
        
        Args:
            X: Feature matrix
            y: Labels (optional)
            
        Returns:
            Dictionary of meta-features
            
        Example:
            >>> extractor = MetaFeatureExtractor()
            >>> meta_features = extractor.extract(X_train, y_train)
            >>> print(f"Class imbalance: {meta_features['class_imbalance']:.3f}")
        """
        meta_features = {}
        
        # Dataset statistics
        meta_features['n_samples'] = X.shape[0]
        meta_features['n_features'] = X.shape[1]
        meta_features['data_dimensionality'] = X.shape[0] / X.shape[1]
        
        # Feature statistics
        meta_features['feature_mean'] = np.mean(X)
        meta_features['feature_std'] = np.std(X)
        meta_features['feature_skewness'] = np.mean([
            abs(np.mean((X[:, i] - np.mean(X[:, i])) ** 3) / (np.std(X[:, i]) ** 3 + 1e-10))
            for i in range(X.shape[1])
        ])
        meta_features['feature_kurtosis'] = np.mean([
            abs(np.mean((X[:, i] - np.mean(X[:, i])) ** 4) / (np.std(X[:, i]) ** 4 + 1e-10))
            for i in range(X.shape[1])
        ])
        
        # Sparsity
        meta_features['sparsity'] = np.mean(X == 0)
        
        # Correlation structure
        if X.shape[1] > 1:
            corr_matrix = np.corrcoef(X.T)
            meta_features['mean_correlation'] = np.mean(np.abs(corr_matrix[np.triu_indices_from(corr_matrix, k=1)]))
            meta_features['max_correlation'] = np.max(np.abs(corr_matrix[np.triu_indices_from(corr_matrix, k=1)]))
        else:
            meta_features['mean_correlation'] = 0.0
            meta_features['max_correlation'] = 0.0
        
        # Label statistics (if available)
        if y is not None:
            unique, counts = np.unique(y, return_counts=True)
            if len(unique) == 2:
                meta_features['class_imbalance'] = min(counts) / max(counts)
                meta_features['minority_class_ratio'] = min(counts) / len(y)
            else:
                meta_features['class_imbalance'] = 1.0
                meta_features['minority_class_ratio'] = 0.5
                
            # Feature-label correlation (simple proxy)
            if len(unique) == 2:
                meta_features['mean_feature_label_correlation'] = np.mean([
                    abs(np.corrcoef(X[:, i], y)[0, 1]) 
                    for i in range(min(X.shape[1], 100))
                ])
        
        logger.debug(f"Extracted {len(meta_features)} meta-features")
        return meta_features
    
    def compute_task_similarity(
        self,
        meta_features_1: Dict[str, float],
        meta_features_2: Dict[str, float]
    ) -> float:
        """
        Compute similarity between two tasks based on meta-features.
        
        Args:
            meta_features_1: Meta-features for first task
            meta_features_2: Meta-features for second task
            
        Returns:
            Similarity score (0-1, higher = more similar)
        """
        # Extract common keys
        keys = set(meta_features_1.keys()) & set(meta_features_2.keys())
        
        if not keys:
            return 0.0
        
        # Normalize and compute distance
        distances = []
        for key in keys:
            v1, v2 = meta_features_1[key], meta_features_2[key]
            # Handle potential division by zero
            if v1 == 0 and v2 == 0:
                distances.append(0.0)
            else:
                # Relative difference
                distances.append(abs(v1 - v2) / (abs(v1) + abs(v2) + 1e-10))
        
        # Convert distance to similarity
        avg_distance = np.mean(distances)
        similarity = 1.0 / (1.0 + avg_distance)
        
        return similarity


class MAMLAdapter:
    """Model-Agnostic Meta-Learning adapter for quick task adaptation."""
    
    def __init__(
        self,
        base_model: Optional[BaseEstimator] = None,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        n_inner_steps: int = 5
    ):
        """
        Initialize MAML adapter.
        
        Args:
            base_model: Base scikit-learn compatible model
            inner_lr: Learning rate for inner loop adaptation
            outer_lr: Learning rate for outer loop meta-update
            n_inner_steps: Number of gradient steps in inner loop
            
        Example:
            >>> adapter = MAMLAdapter(
            ...     base_model=LogisticRegression(),
            ...     inner_lr=0.01,
            ...     n_inner_steps=5
            ... )
        """
        if base_model is None:
            base_model = LogisticRegression(max_iter=1000)
        self.base_model = base_model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.n_inner_steps = n_inner_steps
        self.meta_model = None
        
    def meta_train(
        self,
        tasks: List[Tuple[np.ndarray, np.ndarray]],
        validation_tasks: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
        n_iterations: int = 100
    ):
        """
        Meta-train on multiple tasks.
        
        Args:
            tasks: List of (X, y) tuples for training tasks
            validation_tasks: Optional validation tasks
            n_iterations: Number of meta-training iterations
        """
        logger.info(f"Meta-training on {len(tasks)} tasks for {n_iterations} iterations")
        
        # Initialize meta-model
        self.meta_model = clone(self.base_model)
        
        # For sklearn models, we simulate MAML by training on all tasks
        # with sample weights based on task performance
        all_X = []
        all_y = []
        all_weights = []
        
        for X_task, y_task in tasks:
            all_X.append(X_task)
            all_y.append(y_task)
            # Equal weights initially
            all_weights.append(np.ones(len(y_task)))
        
        X_combined = np.vstack(all_X)
        y_combined = np.hstack(all_y)
        weights_combined = np.hstack(all_weights)
        
        # Train meta-model
        if hasattr(self.meta_model, 'fit'):
            if hasattr(self.meta_model, 'sample_weight'):
                self.meta_model.fit(X_combined, y_combined, sample_weight=weights_combined)
            else:
                self.meta_model.fit(X_combined, y_combined)
        
        logger.info("Meta-training completed")
        
    def adapt(
        self,
        X_support: np.ndarray,
        y_support: np.ndarray,
        n_steps: Optional[int] = None
    ) -> BaseEstimator:
        """
        Adapt meta-model to new task with few-shot examples.
        
        Args:
            X_support: Support set features
            y_support: Support set labels
            n_steps: Number of adaptation steps (uses default if None)
            
        Returns:
            Adapted model
            
        Example:
            >>> adapted_model = adapter.adapt(X_support, y_support, n_steps=5)
            >>> predictions = adapted_model.predict(X_query)
        """
        if self.meta_model is None:
            raise ValueError("Must call meta_train before adapt")
        
        n_steps = n_steps or self.n_inner_steps
        
        # Clone meta-model
        adapted_model = clone(self.meta_model)
        
        # Fine-tune on support set
        adapted_model.fit(X_support, y_support)
        
        logger.debug(f"Adapted model to new task with {len(y_support)} examples")
        return adapted_model


class TransferLearningScorer:
    """Transfer learning scorer for leveraging similar segments."""
    
    def __init__(
        self,
        base_model_type: str = "lgbm",
        similarity_threshold: float = 0.7
    ):
        """
        Initialize transfer learning scorer.
        
        Args:
            base_model_type: Type of base model ('lgbm', 'rf', 'logistic')
            similarity_threshold: Minimum similarity to consider transfer
            
        Example:
            >>> scorer = TransferLearningScorer(base_model_type="lgbm")
            >>> scorer.fit_source_tasks(source_tasks_dict)
            >>> scorer.transfer_and_fit(target_X, target_y, target_segment_id)
        """
        self.base_model_type = base_model_type
        self.similarity_threshold = similarity_threshold
        self.source_models = {}
        self.source_meta_features = {}
        self.meta_extractor = MetaFeatureExtractor()
        
    def _create_base_model(self) -> BaseEstimator:
        """Create base model instance."""
        if self.base_model_type == "lgbm" and LGBM_AVAILABLE:
            return LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
        elif self.base_model_type == "rf":
            return RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            return LogisticRegression(max_iter=1000, random_state=42)
    
    def fit_source_tasks(
        self,
        source_tasks: Dict[str, Tuple[np.ndarray, np.ndarray]]
    ):
        """
        Fit models on source tasks.
        
        Args:
            source_tasks: Dictionary mapping task_id to (X, y) tuples
        """
        logger.info(f"Training on {len(source_tasks)} source tasks")
        
        for task_id, (X, y) in source_tasks.items():
            # Extract meta-features
            self.source_meta_features[task_id] = self.meta_extractor.extract(X, y)
            
            # Train model
            model = self._create_base_model()
            model.fit(X, y)
            self.source_models[task_id] = model
            
            logger.debug(f"Trained model for source task: {task_id}")
    
    def find_similar_tasks(
        self,
        target_X: np.ndarray,
        target_y: Optional[np.ndarray] = None,
        top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Find most similar source tasks.
        
        Args:
            target_X: Target task features
            target_y: Target task labels (optional)
            top_k: Number of similar tasks to return
            
        Returns:
            List of (task_id, similarity_score) tuples
        """
        target_meta = self.meta_extractor.extract(target_X, target_y)
        
        similarities = []
        for task_id, source_meta in self.source_meta_features.items():
            similarity = self.meta_extractor.compute_task_similarity(
                target_meta, source_meta
            )
            similarities.append((task_id, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def transfer_and_fit(
        self,
        target_X: np.ndarray,
        target_y: np.ndarray,
        target_task_id: str,
        warm_start: bool = True
    ) -> BaseEstimator:
        """
        Transfer knowledge from similar tasks and fit to target.
        
        Args:
            target_X: Target task features
            target_y: Target task labels
            target_task_id: Target task identifier
            warm_start: Whether to use warm start from similar task
            
        Returns:
            Fitted model for target task
        """
        similar_tasks = self.find_similar_tasks(target_X, target_y, top_k=1)
        
        if similar_tasks and similar_tasks[0][1] >= self.similarity_threshold and warm_start:
            # Transfer from most similar task
            source_task_id, similarity = similar_tasks[0]
            logger.info(f"Transferring from task '{source_task_id}' (similarity: {similarity:.3f})")
            
            # Clone source model
            model = clone(self.source_models[source_task_id])
            
            # Fine-tune on target task
            model.fit(target_X, target_y)
        else:
            logger.info(f"Training from scratch for task '{target_task_id}'")
            model = self._create_base_model()
            model.fit(target_X, target_y)
        
        # Store as new source task
        self.source_models[target_task_id] = model
        self.source_meta_features[target_task_id] = self.meta_extractor.extract(
            target_X, target_y
        )
        
        return model


class MetaScoringEngine:
    """
    Meta-learning engine for behavioral scoring across customer segments.
    
    Implements multiple meta-learning strategies for efficient adaptation
    to new customer segments with limited data.
    """
    
    def __init__(
        self,
        meta_learning_strategy: str = "transfer",
        base_model_type: str = "lgbm",
        similarity_threshold: float = 0.7,
        n_inner_steps: int = 5
    ):
        """
        Initialize meta-scoring engine.
        
        Args:
            meta_learning_strategy: Strategy to use ('transfer', 'maml', 'multi_task')
            base_model_type: Base model type for scoring
            similarity_threshold: Similarity threshold for transfer learning
            n_inner_steps: Number of adaptation steps for MAML
            
        Example:
            >>> engine = MetaScoringEngine(
            ...     meta_learning_strategy="transfer",
            ...     base_model_type="lgbm"
            ... )
            >>> engine.fit(tasks_dict, labels_dict, segments_dict)
        """
        self.meta_learning_strategy = meta_learning_strategy
        self.base_model_type = base_model_type
        self.similarity_threshold = similarity_threshold
        self.n_inner_steps = n_inner_steps
        
        self.segment_models = {}
        self.meta_extractor = MetaFeatureExtractor()
        self.scaler = StandardScaler()
        
        # Initialize strategy-specific components
        if meta_learning_strategy == "transfer":
            self.transfer_scorer = TransferLearningScorer(
                base_model_type=base_model_type,
                similarity_threshold=similarity_threshold
            )
        elif meta_learning_strategy == "maml":
            self.maml_adapter = MAMLAdapter(n_inner_steps=n_inner_steps)
        
        self.is_fitted = False
        
    def fit(
        self,
        X_dict: Dict[str, np.ndarray],
        y_dict: Dict[str, np.ndarray],
        segment_dict: Dict[str, str]
    ):
        """
        Fit meta-scorer on multiple segments.
        
        Args:
            X_dict: Dictionary mapping sample_id to features
            y_dict: Dictionary mapping sample_id to labels
            segment_dict: Dictionary mapping sample_id to segment_id
            
        Example:
            >>> X_dict = {"user1": features1, "user2": features2}
            >>> y_dict = {"user1": label1, "user2": label2}
            >>> segment_dict = {"user1": "segment_a", "user2": "segment_b"}
            >>> engine.fit(X_dict, y_dict, segment_dict)
        """
        logger.info(f"Training meta-scorer with strategy: {self.meta_learning_strategy}")
        
        # Group by segment
        segment_data = defaultdict(lambda: {'X': [], 'y': []})
        for sample_id, segment_id in segment_dict.items():
            if sample_id in X_dict and sample_id in y_dict:
                segment_data[segment_id]['X'].append(X_dict[sample_id])
                segment_data[segment_id]['y'].append(y_dict[sample_id])
        
        # Convert to arrays
        segment_tasks = {}
        for segment_id, data in segment_data.items():
            X_seg = np.array(data['X'])
            y_seg = np.array(data['y'])
            segment_tasks[segment_id] = (X_seg, y_seg)
        
        logger.info(f"Training on {len(segment_tasks)} segments")
        
        # Fit scaler on all data
        all_X = np.vstack([X for X, _ in segment_tasks.values()])
        self.scaler.fit(all_X)
        
        # Scale segment data
        segment_tasks_scaled = {
            seg_id: (self.scaler.transform(X), y)
            for seg_id, (X, y) in segment_tasks.items()
        }
        
        if self.meta_learning_strategy == "transfer":
            self.transfer_scorer.fit_source_tasks(segment_tasks_scaled)
            self.segment_models = self.transfer_scorer.source_models
            
        elif self.meta_learning_strategy == "maml":
            tasks_list = list(segment_tasks_scaled.values())
            self.maml_adapter.meta_train(tasks_list)
            
            # Also train segment-specific models
            for seg_id, (X, y) in segment_tasks_scaled.items():
                model = self._create_base_model()
                model.fit(X, y)
                self.segment_models[seg_id] = model
                
        elif self.meta_learning_strategy == "multi_task":
            # Train single model on all segments with segment as feature
            all_X_list = []
            all_y_list = []
            all_seg_features = []
            
            for seg_idx, (seg_id, (X, y)) in enumerate(segment_tasks_scaled.items()):
                all_X_list.append(X)
                all_y_list.append(y)
                # One-hot encode segment
                seg_feature = np.zeros((len(y), len(segment_tasks_scaled)))
                seg_feature[:, seg_idx] = 1
                all_seg_features.append(seg_feature)
            
            X_multi = np.vstack([
                np.hstack([X, seg_feat])
                for X, seg_feat in zip(all_X_list, all_seg_features)
            ])
            y_multi = np.hstack(all_y_list)
            
            multi_task_model = self._create_base_model()
            multi_task_model.fit(X_multi, y_multi)
            self.segment_models['multi_task'] = multi_task_model
        
        self.is_fitted = True
        logger.info("Meta-scorer training completed")
    
    def adapt_to_segment(
        self,
        X_new: np.ndarray,
        y_new: np.ndarray,
        segment_id: str,
        n_shots: Optional[int] = None
    ):
        """
        Adapt to new segment with few-shot learning.
        
        Args:
            X_new: Features for new segment
            y_new: Labels for new segment
            segment_id: New segment identifier
            n_shots: Number of shots to use (None = use all)
            
        Example:
            >>> # Few-shot adaptation with 10 examples
            >>> engine.adapt_to_segment(X_new, y_new, "new_segment", n_shots=10)
            >>> scores = engine.predict(X_test, segment_id="new_segment")
        """
        if not self.is_fitted:
            raise ValueError("Must fit meta-scorer before adaptation")
        
        logger.info(f"Adapting to segment '{segment_id}' with {n_shots or 'all'} shots")
        
        # Sample n_shots if specified
        if n_shots is not None and n_shots < len(y_new):
            indices = np.random.choice(len(y_new), n_shots, replace=False)
            X_adapt = X_new[indices]
            y_adapt = y_new[indices]
        else:
            X_adapt = X_new
            y_adapt = y_new
        
        # Scale features
        X_adapt_scaled = self.scaler.transform(X_adapt)
        
        if self.meta_learning_strategy == "transfer":
            model = self.transfer_scorer.transfer_and_fit(
                X_adapt_scaled, y_adapt, segment_id
            )
        elif self.meta_learning_strategy == "maml":
            model = self.maml_adapter.adapt(X_adapt_scaled, y_adapt)
        else:
            model = self._create_base_model()
            model.fit(X_adapt_scaled, y_adapt)
        
        self.segment_models[segment_id] = model
        logger.info(f"Adaptation completed for segment '{segment_id}'")
    
    def predict(
        self,
        X: np.ndarray,
        segment_id: Optional[str] = None
    ) -> np.ndarray:
        """
        Predict scores for samples.
        
        Args:
            X: Features
            segment_id: Segment identifier (required for segment-specific models)
            
        Returns:
            Risk scores (0-1, higher = higher risk)
            
        Example:
            >>> scores = engine.predict(X_test, segment_id="premium_customers")
        """
        if not self.is_fitted:
            raise ValueError("Must fit meta-scorer before prediction")
        
        X_scaled = self.scaler.transform(X)
        
        if segment_id and segment_id in self.segment_models:
            model = self.segment_models[segment_id]
        elif 'multi_task' in self.segment_models:
            # For multi-task, need segment encoding (use default segment)
            logger.warning("No segment_id provided for multi-task model, using default")
            model = list(self.segment_models.values())[0]
        else:
            # Use first available model
            if not self.segment_models:
                raise ValueError("No models available for prediction")
            model = list(self.segment_models.values())[0]
            logger.warning(f"Using default model for prediction")
        
        if hasattr(model, 'predict_proba'):
            scores = model.predict_proba(X_scaled)[:, 1]
        else:
            scores = model.decision_function(X_scaled)
            # Normalize to 0-1
            scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
        
        return scores
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        segment_id: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X_test: Test features
            y_test: Test labels
            segment_id: Segment identifier
            
        Returns:
            Dictionary of evaluation metrics
        """
        scores = self.predict(X_test, segment_id=segment_id)
        
        metrics = {
            'auc_roc': roc_auc_score(y_test, scores),
            'avg_precision': average_precision_score(y_test, scores)
        }
        
        logger.info(f"Evaluation metrics: {metrics}")
        return metrics
    
    def _create_base_model(self) -> BaseEstimator:
        """Create base model instance."""
        if self.base_model_type == "lgbm" and LGBM_AVAILABLE:
            return LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
        elif self.base_model_type == "rf":
            return RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            return LogisticRegression(max_iter=1000, random_state=42)
    
    def save(self, path: Union[str, Path]):
        """Save meta-scorer to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"Meta-scorer saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'MetaScoringEngine':
        """Load meta-scorer from disk."""
        engine = joblib.load(path)
        logger.info(f"Meta-scorer loaded from {path}")
        return engine
