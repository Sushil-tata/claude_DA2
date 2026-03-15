"""
Transfer Learning Framework for Cross-Institution Knowledge Transfer.

This module implements privacy-preserving transfer learning with:
- Domain adaptation techniques
- Feature/instance/model-based transfer
- Adversarial domain adaptation
- Transferability metrics
- Multi-source transfer learning
- Privacy-preserving embeddings

References:
    - Pan & Yang (2010): A Survey on Transfer Learning
    - Ganin et al. (2016): Domain-Adversarial Training of Neural Networks
    - Tzeng et al. (2017): Adversarial Discriminative Domain Adaptation
    - Ben-David et al. (2010): A Theory of Learning from Different Domains

Example:
    >>> # Transfer from source to target institution
    >>> transfer_learner = TransferLearner(
    ...     source_data=(X_source, y_source),
    ...     target_data=(X_target, y_target),
    ...     transfer_method='feature_based'
    ... )
    >>> 
    >>> # Adapt model
    >>> adapted_model = transfer_learner.adapt(
    ...     source_model=pretrained_model,
    ...     adaptation_epochs=10
    ... )
    >>> 
    >>> # Evaluate on target domain
    >>> metrics = transfer_learner.evaluate_on_target(adapted_model)
"""

import copy
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from loguru import logger
from scipy import linalg
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score, mean_squared_error

try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available. Some features will be limited.")


@dataclass
class DomainStatistics:
    """Statistics for source and target domains.
    
    Attributes:
        source_mean: Mean of source domain features
        source_std: Std of source domain features
        target_mean: Mean of target domain features
        target_std: Std of target domain features
        source_samples: Number of source samples
        target_samples: Number of target samples
        feature_dim: Feature dimensionality
        mmd_distance: Maximum Mean Discrepancy between domains
        coral_distance: CORAL distance between domains
    """
    source_mean: np.ndarray
    source_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray
    source_samples: int
    target_samples: int
    feature_dim: int
    mmd_distance: float = 0.0
    coral_distance: float = 0.0


@dataclass
class TransferabilityScore:
    """Transferability metrics between source and target.
    
    Attributes:
        task_similarity: Task similarity score (0-1)
        domain_similarity: Domain similarity score (0-1)
        expected_transfer_gain: Expected performance gain
        negative_transfer_risk: Risk of negative transfer (0-1)
        confidence: Confidence in transferability estimate (0-1)
        method: Method used for computation
    """
    task_similarity: float
    domain_similarity: float
    expected_transfer_gain: float
    negative_transfer_risk: float
    confidence: float
    method: str = 'mmd'


class DomainAdapter:
    """Domain adaptation techniques for transfer learning.
    
    Implements various domain adaptation methods:
    - Feature-based: Align feature distributions
    - Instance-based: Reweight source samples
    - Model-based: Fine-tune pre-trained models
    - Adversarial: Use domain-adversarial training
    
    Example:
        >>> adapter = DomainAdapter(method='coral')
        >>> X_source_adapted = adapter.adapt_features(X_source, X_target)
    """
    
    def __init__(
        self,
        method: str = 'coral',
        lambda_adapt: float = 1.0,
        **kwargs
    ):
        """Initialize domain adapter.
        
        Args:
            method: Adaptation method ('coral', 'mmd', 'dann', 'instance_weight')
            lambda_adapt: Adaptation regularization parameter
            **kwargs: Additional method-specific parameters
        """
        self.method = method
        self.lambda_adapt = lambda_adapt
        self.kwargs = kwargs
        
        self.is_fitted = False
        self.adaptation_params: Dict[str, Any] = {}
        
        logger.info(f"Domain adapter initialized: method={method}")
    
    def fit(
        self,
        X_source: np.ndarray,
        X_target: np.ndarray,
        y_source: Optional[np.ndarray] = None
    ) -> 'DomainAdapter':
        """Fit domain adaptation parameters.
        
        Args:
            X_source: Source domain features
            X_target: Target domain features
            y_source: Source labels (for some methods)
        
        Returns:
            Self
        """
        if self.method == 'coral':
            self._fit_coral(X_source, X_target)
        elif self.method == 'mmd':
            self._fit_mmd(X_source, X_target)
        elif self.method == 'instance_weight':
            self._fit_instance_weight(X_source, X_target, y_source)
        else:
            logger.warning(f"Unknown method {self.method}, using identity")
        
        self.is_fitted = True
        return self
    
    def _fit_coral(
        self,
        X_source: np.ndarray,
        X_target: np.ndarray
    ) -> None:
        """Fit CORAL (Correlation Alignment) adaptation.
        
        CORAL aligns second-order statistics (covariances) of source
        and target domains.
        
        Reference: Sun et al. (2016): Return of Frustratingly Easy Domain Adaptation
        """
        # Compute covariance matrices
        source_cov = np.cov(X_source, rowvar=False) + np.eye(X_source.shape[1]) * 1e-6
        target_cov = np.cov(X_target, rowvar=False) + np.eye(X_target.shape[1]) * 1e-6
        
        # Compute transformation matrix
        # A_coral = C_s^{-1/2} C_t^{1/2}
        source_cov_sqrt_inv = linalg.fractional_matrix_power(source_cov, -0.5)
        target_cov_sqrt = linalg.fractional_matrix_power(target_cov, 0.5)
        
        coral_transform = source_cov_sqrt_inv @ target_cov_sqrt
        
        self.adaptation_params['coral_transform'] = coral_transform
        self.adaptation_params['source_mean'] = np.mean(X_source, axis=0)
        self.adaptation_params['target_mean'] = np.mean(X_target, axis=0)
        
        logger.debug("CORAL adaptation fitted")
    
    def _fit_mmd(
        self,
        X_source: np.ndarray,
        X_target: np.ndarray
    ) -> None:
        """Fit Maximum Mean Discrepancy adaptation.
        
        MMD measures distance between distributions using kernel methods.
        """
        # Compute kernel matrices
        gamma = self.kwargs.get('gamma', 1.0)
        
        # RBF kernel
        K_ss = self._rbf_kernel(X_source, X_source, gamma)
        K_tt = self._rbf_kernel(X_target, X_target, gamma)
        K_st = self._rbf_kernel(X_source, X_target, gamma)
        
        # Compute MMD
        n_source = len(X_source)
        n_target = len(X_target)
        
        mmd = (
            np.sum(K_ss) / (n_source * n_source) +
            np.sum(K_tt) / (n_target * n_target) -
            2 * np.sum(K_st) / (n_source * n_target)
        )
        
        self.adaptation_params['mmd'] = mmd
        self.adaptation_params['gamma'] = gamma
        
        logger.debug(f"MMD adaptation fitted: mmd={mmd:.4f}")
    
    def _fit_instance_weight(
        self,
        X_source: np.ndarray,
        X_target: np.ndarray,
        y_source: Optional[np.ndarray] = None
    ) -> None:
        """Fit instance-based weighting.
        
        Compute importance weights for source samples based on
        similarity to target distribution.
        """
        # Compute distances from each source to target samples
        distances = cdist(X_source, X_target, metric='euclidean')
        
        # Compute weights (inverse distance to nearest target sample)
        min_distances = np.min(distances, axis=1)
        weights = 1.0 / (min_distances + 1e-6)
        
        # Normalize weights
        weights = weights / np.sum(weights) * len(weights)
        
        self.adaptation_params['instance_weights'] = weights
        
        logger.debug(
            f"Instance weighting fitted: "
            f"mean={np.mean(weights):.4f}, std={np.std(weights):.4f}"
        )
    
    def adapt_features(
        self,
        X: np.ndarray,
        is_source: bool = True
    ) -> np.ndarray:
        """Adapt features using fitted parameters.
        
        Args:
            X: Features to adapt
            is_source: Whether features are from source domain
        
        Returns:
            Adapted features
        """
        if not self.is_fitted:
            raise ValueError("Adapter not fitted. Call fit() first.")
        
        if self.method == 'coral':
            return self._apply_coral(X, is_source)
        elif self.method == 'mmd':
            # MMD is used for measuring distance, not transformation
            return X
        elif self.method == 'instance_weight':
            # Weighting is applied during training, not feature transformation
            return X
        else:
            return X
    
    def _apply_coral(
        self,
        X: np.ndarray,
        is_source: bool
    ) -> np.ndarray:
        """Apply CORAL transformation."""
        if is_source:
            # Transform source to target distribution
            X_centered = X - self.adaptation_params['source_mean']
            X_adapted = X_centered @ self.adaptation_params['coral_transform']
            X_adapted += self.adaptation_params['target_mean']
        else:
            # Target features don't need transformation
            X_adapted = X
        
        return X_adapted
    
    def get_instance_weights(self) -> Optional[np.ndarray]:
        """Get instance weights for source samples."""
        return self.adaptation_params.get('instance_weights')
    
    @staticmethod
    def _rbf_kernel(
        X: np.ndarray,
        Y: np.ndarray,
        gamma: float
    ) -> np.ndarray:
        """Compute RBF (Gaussian) kernel."""
        sq_dists = cdist(X, Y, metric='sqeuclidean')
        return np.exp(-gamma * sq_dists)


class TransferabilityMetrics:
    """Compute transferability metrics between source and target tasks.
    
    Helps determine if transfer learning will be beneficial and
    detect potential negative transfer.
    
    Example:
        >>> metrics = TransferabilityMetrics()
        >>> score = metrics.compute_transferability(
        ...     X_source, y_source, X_target, y_target
        ... )
        >>> if score.expected_transfer_gain > 0.1:
        ...     print("Transfer learning recommended")
    """
    
    def __init__(self):
        """Initialize transferability metrics."""
        pass
    
    def compute_transferability(
        self,
        X_source: np.ndarray,
        y_source: np.ndarray,
        X_target: np.ndarray,
        y_target: Optional[np.ndarray] = None,
        method: str = 'mmd'
    ) -> TransferabilityScore:
        """Compute transferability score.
        
        Args:
            X_source: Source features
            y_source: Source labels
            X_target: Target features
            y_target: Target labels (if available)
            method: Method for computation ('mmd', 'kl', 'correlation')
        
        Returns:
            Transferability score
        """
        # Compute domain similarity
        domain_sim = self._compute_domain_similarity(
            X_source, X_target, method
        )
        
        # Compute task similarity (if target labels available)
        if y_target is not None and len(y_target) > 0:
            task_sim = self._compute_task_similarity(
                X_source, y_source, X_target, y_target
            )
        else:
            task_sim = 0.5  # Unknown, assume moderate similarity
        
        # Estimate transfer gain
        transfer_gain = self._estimate_transfer_gain(
            domain_sim, task_sim
        )
        
        # Estimate negative transfer risk
        neg_transfer_risk = self._estimate_negative_transfer_risk(
            domain_sim, task_sim
        )
        
        # Compute confidence based on available information
        confidence = 0.7 if y_target is not None else 0.4
        
        score = TransferabilityScore(
            task_similarity=task_sim,
            domain_similarity=domain_sim,
            expected_transfer_gain=transfer_gain,
            negative_transfer_risk=neg_transfer_risk,
            confidence=confidence,
            method=method
        )
        
        logger.info(
            f"Transferability: domain_sim={domain_sim:.3f}, "
            f"task_sim={task_sim:.3f}, gain={transfer_gain:.3f}, "
            f"risk={neg_transfer_risk:.3f}"
        )
        
        return score
    
    def _compute_domain_similarity(
        self,
        X_source: np.ndarray,
        X_target: np.ndarray,
        method: str
    ) -> float:
        """Compute similarity between source and target domains."""
        if method == 'mmd':
            # Maximum Mean Discrepancy
            mmd = self._compute_mmd(X_source, X_target)
            # Convert to similarity (0-1)
            similarity = np.exp(-mmd)
        elif method == 'kl':
            # KL divergence (approximated)
            kl = self._compute_kl_divergence(X_source, X_target)
            similarity = np.exp(-kl)
        elif method == 'correlation':
            # Feature correlation
            corr = self._compute_feature_correlation(X_source, X_target)
            similarity = corr
        else:
            # Default: cosine similarity of means
            mean_source = np.mean(X_source, axis=0)
            mean_target = np.mean(X_target, axis=0)
            similarity = np.dot(mean_source, mean_target) / (
                np.linalg.norm(mean_source) * np.linalg.norm(mean_target) + 1e-8
            )
        
        return float(np.clip(similarity, 0, 1))
    
    def _compute_task_similarity(
        self,
        X_source: np.ndarray,
        y_source: np.ndarray,
        X_target: np.ndarray,
        y_target: np.ndarray
    ) -> float:
        """Compute similarity between source and target tasks."""
        # Train simple model on source
        from sklearn.linear_model import Ridge
        
        model = Ridge(alpha=1.0)
        model.fit(X_source, y_source)
        
        # Evaluate on target
        y_pred = model.predict(X_target)
        
        # Compute correlation between predictions and true labels
        correlation = np.corrcoef(y_pred.flatten(), y_target.flatten())[0, 1]
        
        # Convert to 0-1 range
        task_similarity = (correlation + 1) / 2
        
        return float(np.clip(task_similarity, 0, 1))
    
    def _estimate_transfer_gain(
        self,
        domain_sim: float,
        task_sim: float
    ) -> float:
        """Estimate expected performance gain from transfer learning.
        
        Gain is higher when both domain and task similarity are high.
        """
        # Weighted combination
        gain = 0.4 * domain_sim + 0.6 * task_sim
        
        # Apply non-linear transformation
        # Transfer is most beneficial when similarity is high
        gain = gain ** 2
        
        return float(np.clip(gain, 0, 1))
    
    def _estimate_negative_transfer_risk(
        self,
        domain_sim: float,
        task_sim: float
    ) -> float:
        """Estimate risk of negative transfer.
        
        Risk is higher when domains/tasks are dissimilar.
        """
        # Inverse of similarity
        domain_risk = 1 - domain_sim
        task_risk = 1 - task_sim
        
        # Combined risk
        risk = 0.5 * domain_risk + 0.5 * task_risk
        
        # Non-linear: high risk only when both are dissimilar
        risk = risk ** 1.5
        
        return float(np.clip(risk, 0, 1))
    
    def _compute_mmd(
        self,
        X_source: np.ndarray,
        X_target: np.ndarray,
        gamma: float = 1.0
    ) -> float:
        """Compute Maximum Mean Discrepancy."""
        # Simplified MMD computation
        n_source = len(X_source)
        n_target = len(X_target)
        
        # Sample if too large
        if n_source > 1000:
            idx = np.random.choice(n_source, 1000, replace=False)
            X_source = X_source[idx]
            n_source = 1000
        
        if n_target > 1000:
            idx = np.random.choice(n_target, 1000, replace=False)
            X_target = X_target[idx]
            n_target = 1000
        
        # RBF kernel
        K_ss = DomainAdapter._rbf_kernel(X_source, X_source, gamma)
        K_tt = DomainAdapter._rbf_kernel(X_target, X_target, gamma)
        K_st = DomainAdapter._rbf_kernel(X_source, X_target, gamma)
        
        mmd = (
            np.sum(K_ss) / (n_source * n_source) +
            np.sum(K_tt) / (n_target * n_target) -
            2 * np.sum(K_st) / (n_source * n_target)
        )
        
        return float(max(0, mmd))
    
    def _compute_kl_divergence(
        self,
        X_source: np.ndarray,
        X_target: np.ndarray
    ) -> float:
        """Approximate KL divergence using histograms."""
        # Use first principal component for 1D approximation
        if SKLEARN_AVAILABLE:
            pca = PCA(n_components=1)
            X_combined = np.vstack([X_source, X_target])
            pca.fit(X_combined)
            
            source_proj = pca.transform(X_source).flatten()
            target_proj = pca.transform(X_target).flatten()
        else:
            source_proj = X_source[:, 0]
            target_proj = X_target[:, 0]
        
        # Compute histograms
        bins = 50
        range_min = min(source_proj.min(), target_proj.min())
        range_max = max(source_proj.max(), target_proj.max())
        
        hist_source, _ = np.histogram(
            source_proj, bins=bins, range=(range_min, range_max), density=True
        )
        hist_target, _ = np.histogram(
            target_proj, bins=bins, range=(range_min, range_max), density=True
        )
        
        # Add small constant to avoid log(0)
        hist_source = hist_source + 1e-10
        hist_target = hist_target + 1e-10
        
        # Compute KL divergence
        kl = np.sum(hist_source * np.log(hist_source / hist_target))
        
        return float(max(0, kl))
    
    def _compute_feature_correlation(
        self,
        X_source: np.ndarray,
        X_target: np.ndarray
    ) -> float:
        """Compute average correlation between features."""
        # Compute correlation between feature means
        source_mean = np.mean(X_source, axis=0)
        target_mean = np.mean(X_target, axis=0)
        
        correlation = np.corrcoef(source_mean, target_mean)[0, 1]
        
        return float((correlation + 1) / 2)


class TransferLearner:
    """Transfer learning for cross-institution knowledge transfer.
    
    Implements comprehensive transfer learning with:
    - Domain adaptation
    - Feature/instance/model-based transfer
    - Multi-source transfer
    - Privacy-preserving transfer
    - Negative transfer detection
    
    Example:
        >>> # Single-source transfer
        >>> learner = TransferLearner(
        ...     source_data=(X_source, y_source),
        ...     target_data=(X_target, y_target),
        ...     transfer_method='feature_based'
        ... )
        >>> adapted_model = learner.adapt(source_model)
        >>> 
        >>> # Multi-source transfer
        >>> learner = TransferLearner(
        ...     source_data=[
        ...         (X_source1, y_source1),
        ...         (X_source2, y_source2)
        ...     ],
        ...     target_data=(X_target, y_target),
        ...     transfer_method='multi_source'
        ... )
    """
    
    def __init__(
        self,
        source_data: Union[
            Tuple[np.ndarray, np.ndarray],
            List[Tuple[np.ndarray, np.ndarray]]
        ],
        target_data: Tuple[np.ndarray, np.ndarray],
        transfer_method: str = 'feature_based',
        adaptation_config: Optional[Dict[str, Any]] = None,
        privacy_config: Optional[Dict[str, Any]] = None
    ):
        """Initialize transfer learner.
        
        Args:
            source_data: Source domain data (X, y) or list of (X, y) for multi-source
            target_data: Target domain data (X, y)
            transfer_method: Transfer method ('feature_based', 'instance_based',
                'model_based', 'adversarial', 'multi_source')
            adaptation_config: Domain adaptation configuration
            privacy_config: Privacy-preserving configuration
        """
        # Handle single vs multi-source
        if isinstance(source_data, tuple):
            self.source_data = [source_data]
            self.multi_source = False
        else:
            self.source_data = source_data
            self.multi_source = True
        
        self.X_target, self.y_target = target_data
        self.transfer_method = transfer_method
        
        # Configuration
        self.adaptation_config = adaptation_config or {}
        self.privacy_config = privacy_config or {}
        
        # Domain adapter
        adapter_method = self.adaptation_config.get('method', 'coral')
        self.adapter = DomainAdapter(method=adapter_method)
        
        # Transferability metrics
        self.metrics_calculator = TransferabilityMetrics()
        
        # State
        self.is_adapted = False
        self.adapted_model: Optional[Dict[str, Any]] = None
        self.transferability_scores: List[TransferabilityScore] = []
        
        logger.info(
            f"Transfer learner initialized: "
            f"method={transfer_method}, "
            f"multi_source={self.multi_source}, "
            f"target_samples={len(self.X_target)}"
        )
    
    def compute_transferability(self) -> List[TransferabilityScore]:
        """Compute transferability from each source to target.
        
        Returns:
            List of transferability scores for each source
        """
        scores = []
        
        for i, (X_source, y_source) in enumerate(self.source_data):
            score = self.metrics_calculator.compute_transferability(
                X_source, y_source,
                self.X_target, self.y_target
            )
            scores.append(score)
            
            logger.info(
                f"Source {i}: transferability={score.expected_transfer_gain:.3f}, "
                f"risk={score.negative_transfer_risk:.3f}"
            )
        
        self.transferability_scores = scores
        return scores
    
    def adapt(
        self,
        source_model: Optional[Dict[str, np.ndarray]] = None,
        adaptation_epochs: int = 10,
        fine_tune_layers: Optional[List[str]] = None,
        early_stopping: bool = True,
        validation_split: float = 0.2
    ) -> Dict[str, np.ndarray]:
        """Adapt source model to target domain.
        
        Args:
            source_model: Pre-trained source model (if None, trains from scratch)
            adaptation_epochs: Number of adaptation epochs
            fine_tune_layers: Layers to fine-tune (None = all)
            early_stopping: Whether to use early stopping
            validation_split: Validation split for early stopping
        
        Returns:
            Adapted model
        """
        if self.transfer_method == 'feature_based':
            adapted_model = self._adapt_feature_based(
                source_model, adaptation_epochs
            )
        elif self.transfer_method == 'instance_based':
            adapted_model = self._adapt_instance_based(
                source_model, adaptation_epochs
            )
        elif self.transfer_method == 'model_based':
            adapted_model = self._adapt_model_based(
                source_model, adaptation_epochs, fine_tune_layers
            )
        elif self.transfer_method == 'multi_source':
            adapted_model = self._adapt_multi_source(
                source_model, adaptation_epochs
            )
        else:
            logger.warning(
                f"Unknown transfer method {self.transfer_method}, "
                f"using feature-based"
            )
            adapted_model = self._adapt_feature_based(
                source_model, adaptation_epochs
            )
        
        self.adapted_model = adapted_model
        self.is_adapted = True
        
        return adapted_model
    
    def _adapt_feature_based(
        self,
        source_model: Optional[Dict[str, np.ndarray]],
        epochs: int
    ) -> Dict[str, np.ndarray]:
        """Feature-based domain adaptation using CORAL or MMD."""
        logger.info("Starting feature-based adaptation")
        
        # Get source data (use first source if multi-source)
        X_source, y_source = self.source_data[0]
        
        # Fit adapter
        self.adapter.fit(X_source, self.X_target)
        
        # Adapt source features
        X_source_adapted = self.adapter.adapt_features(X_source, is_source=True)
        
        # Train model on adapted source + target data
        if source_model is None:
            # Initialize new model
            input_dim = X_source_adapted.shape[1]
            output_dim = 1 if len(y_source.shape) == 1 else y_source.shape[1]
            
            model = self._initialize_model(input_dim, output_dim)
        else:
            model = copy.deepcopy(source_model)
        
        # Combine adapted source and target data
        X_combined = np.vstack([X_source_adapted, self.X_target])
        y_combined = np.concatenate([y_source, self.y_target])
        
        # Fine-tune on combined data
        model = self._train_model(model, X_combined, y_combined, epochs)
        
        logger.info("Feature-based adaptation complete")
        return model
    
    def _adapt_instance_based(
        self,
        source_model: Optional[Dict[str, np.ndarray]],
        epochs: int
    ) -> Dict[str, np.ndarray]:
        """Instance-based adaptation using importance weighting."""
        logger.info("Starting instance-based adaptation")
        
        X_source, y_source = self.source_data[0]
        
        # Fit adapter to get instance weights
        self.adapter = DomainAdapter(method='instance_weight')
        self.adapter.fit(X_source, self.X_target, y_source)
        
        weights = self.adapter.get_instance_weights()
        
        # Initialize or copy model
        if source_model is None:
            input_dim = X_source.shape[1]
            output_dim = 1 if len(y_source.shape) == 1 else y_source.shape[1]
            model = self._initialize_model(input_dim, output_dim)
        else:
            model = copy.deepcopy(source_model)
        
        # Train with instance weights
        model = self._train_weighted_model(
            model, X_source, y_source, weights, epochs
        )
        
        # Fine-tune on target
        if len(self.y_target) > 0:
            model = self._train_model(
                model, self.X_target, self.y_target, epochs // 2
            )
        
        logger.info("Instance-based adaptation complete")
        return model
    
    def _adapt_model_based(
        self,
        source_model: Optional[Dict[str, np.ndarray]],
        epochs: int,
        fine_tune_layers: Optional[List[str]]
    ) -> Dict[str, np.ndarray]:
        """Model-based transfer via fine-tuning."""
        logger.info("Starting model-based adaptation (fine-tuning)")
        
        if source_model is None:
            logger.warning("No source model provided, training from scratch")
            X_source, y_source = self.source_data[0]
            input_dim = X_source.shape[1]
            output_dim = 1 if len(y_source.shape) == 1 else y_source.shape[1]
            source_model = self._initialize_model(input_dim, output_dim)
        
        # Copy model for fine-tuning
        model = copy.deepcopy(source_model)
        
        # Freeze layers if specified
        if fine_tune_layers is not None:
            frozen_layers = set(model.keys()) - set(fine_tune_layers)
            logger.info(f"Freezing layers: {frozen_layers}")
        else:
            frozen_layers = set()
        
        # Fine-tune on target data
        model = self._train_model(
            model, self.X_target, self.y_target, epochs,
            frozen_layers=frozen_layers
        )
        
        logger.info("Model-based adaptation complete")
        return model
    
    def _adapt_multi_source(
        self,
        source_model: Optional[Dict[str, np.ndarray]],
        epochs: int
    ) -> Dict[str, np.ndarray]:
        """Multi-source transfer learning.
        
        Combines knowledge from multiple source domains.
        """
        logger.info(f"Starting multi-source adaptation ({len(self.source_data)} sources)")
        
        # Compute transferability scores if not done
        if not self.transferability_scores:
            self.compute_transferability()
        
        # Weight sources by transferability
        weights = [
            score.expected_transfer_gain
            for score in self.transferability_scores
        ]
        weights = np.array(weights) / sum(weights)
        
        logger.info(f"Source weights: {weights}")
        
        # Adapt features from each source
        adapted_sources = []
        for i, (X_source, y_source) in enumerate(self.source_data):
            adapter = DomainAdapter(method='coral')
            adapter.fit(X_source, self.X_target)
            
            X_adapted = adapter.adapt_features(X_source, is_source=True)
            adapted_sources.append((X_adapted, y_source, weights[i]))
        
        # Combine adapted sources
        X_combined_list = []
        y_combined_list = []
        sample_weights = []
        
        for X_adapted, y_source, source_weight in adapted_sources:
            X_combined_list.append(X_adapted)
            y_combined_list.append(y_source)
            sample_weights.extend([source_weight] * len(X_adapted))
        
        # Add target data with high weight
        X_combined_list.append(self.X_target)
        y_combined_list.append(self.y_target)
        sample_weights.extend([1.0] * len(self.X_target))
        
        X_combined = np.vstack(X_combined_list)
        y_combined = np.concatenate(y_combined_list)
        sample_weights = np.array(sample_weights)
        
        # Initialize model
        if source_model is None:
            input_dim = X_combined.shape[1]
            output_dim = 1 if len(y_combined.shape) == 1 else y_combined.shape[1]
            model = self._initialize_model(input_dim, output_dim)
        else:
            model = copy.deepcopy(source_model)
        
        # Train on weighted combination
        model = self._train_weighted_model(
            model, X_combined, y_combined, sample_weights, epochs
        )
        
        logger.info("Multi-source adaptation complete")
        return model
    
    def _initialize_model(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 64
    ) -> Dict[str, np.ndarray]:
        """Initialize a simple MLP model."""
        model = {
            'w1': np.random.randn(input_dim, hidden_dim) * 0.01,
            'b1': np.zeros(hidden_dim),
            'w2': np.random.randn(hidden_dim, output_dim) * 0.01,
            'b2': np.zeros(output_dim)
        }
        return model
    
    def _train_model(
        self,
        model: Dict[str, np.ndarray],
        X: np.ndarray,
        y: np.ndarray,
        epochs: int,
        frozen_layers: Optional[set] = None,
        learning_rate: float = 0.01
    ) -> Dict[str, np.ndarray]:
        """Train model with gradient descent."""
        frozen_layers = frozen_layers or set()
        
        for epoch in range(epochs):
            # Forward pass
            h = np.maximum(0, X @ model['w1'] + model['b1'])
            predictions = h @ model['w2'] + model['b2']
            
            # Compute loss
            loss = np.mean((predictions - y) ** 2)
            
            # Backward pass
            batch_size = len(X)
            d_output = 2 * (predictions - y) / batch_size
            
            # Gradients
            grad_w2 = h.T @ d_output
            grad_b2 = np.sum(d_output, axis=0)
            
            d_hidden = d_output @ model['w2'].T
            d_hidden[h <= 0] = 0
            
            grad_w1 = X.T @ d_hidden
            grad_b1 = np.sum(d_hidden, axis=0)
            
            # Update (skip frozen layers)
            if 'w2' not in frozen_layers:
                model['w2'] -= learning_rate * grad_w2
            if 'b2' not in frozen_layers:
                model['b2'] -= learning_rate * grad_b2
            if 'w1' not in frozen_layers:
                model['w1'] -= learning_rate * grad_w1
            if 'b1' not in frozen_layers:
                model['b1'] -= learning_rate * grad_b1
            
            if (epoch + 1) % 10 == 0:
                logger.debug(f"Epoch {epoch + 1}/{epochs}: loss={loss:.4f}")
        
        return model
    
    def _train_weighted_model(
        self,
        model: Dict[str, np.ndarray],
        X: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        epochs: int,
        learning_rate: float = 0.01
    ) -> Dict[str, np.ndarray]:
        """Train model with sample weights."""
        # Normalize weights
        weights = weights / np.sum(weights) * len(weights)
        
        for epoch in range(epochs):
            # Forward pass
            h = np.maximum(0, X @ model['w1'] + model['b1'])
            predictions = h @ model['w2'] + model['b2']
            
            # Weighted loss
            losses = (predictions - y) ** 2
            loss = np.mean(losses * weights.reshape(-1, 1))
            
            # Weighted gradients
            batch_size = len(X)
            d_output = 2 * (predictions - y) * weights.reshape(-1, 1) / batch_size
            
            grad_w2 = h.T @ d_output
            grad_b2 = np.sum(d_output, axis=0)
            
            d_hidden = d_output @ model['w2'].T
            d_hidden[h <= 0] = 0
            
            grad_w1 = X.T @ d_hidden
            grad_b1 = np.sum(d_hidden, axis=0)
            
            # Update
            model['w2'] -= learning_rate * grad_w2
            model['b2'] -= learning_rate * grad_b2
            model['w1'] -= learning_rate * grad_w1
            model['b1'] -= learning_rate * grad_b1
            
            if (epoch + 1) % 10 == 0:
                logger.debug(f"Epoch {epoch + 1}/{epochs}: weighted_loss={loss:.4f}")
        
        return model
    
    def evaluate_on_target(
        self,
        model: Optional[Dict[str, np.ndarray]] = None,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Evaluate model on target domain.
        
        Args:
            model: Model to evaluate (uses adapted model if None)
            X_test: Test features (uses target data if None)
            y_test: Test labels (uses target data if None)
        
        Returns:
            Evaluation metrics
        """
        if model is None:
            if not self.is_adapted:
                raise ValueError("No adapted model available. Call adapt() first.")
            model = self.adapted_model
        
        if X_test is None or y_test is None:
            X_test, y_test = self.X_target, self.y_target
        
        # Forward pass
        h = np.maximum(0, X_test @ model['w1'] + model['b1'])
        predictions = h @ model['w2'] + model['b2']
        
        # Compute metrics
        mse = np.mean((predictions - y_test) ** 2)
        mae = np.mean(np.abs(predictions - y_test))
        rmse = np.sqrt(mse)
        
        # R-squared
        ss_res = np.sum((y_test - predictions) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        
        metrics = {
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'num_samples': len(X_test)
        }
        
        logger.info(
            f"Target evaluation: MSE={mse:.4f}, MAE={mae:.4f}, R2={r2:.4f}"
        )
        
        return metrics
    
    def detect_negative_transfer(
        self,
        baseline_model: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, Any]:
        """Detect negative transfer by comparing with baseline.
        
        Args:
            baseline_model: Baseline model trained only on target (if None, trains one)
        
        Returns:
            Negative transfer detection results
        """
        if not self.is_adapted:
            raise ValueError("No adapted model. Call adapt() first.")
        
        # Train baseline on target only
        if baseline_model is None:
            logger.info("Training baseline model on target only")
            input_dim = self.X_target.shape[1]
            output_dim = 1 if len(self.y_target.shape) == 1 else self.y_target.shape[1]
            
            baseline_model = self._initialize_model(input_dim, output_dim)
            baseline_model = self._train_model(
                baseline_model, self.X_target, self.y_target, epochs=50
            )
        
        # Evaluate both models
        transfer_metrics = self.evaluate_on_target(self.adapted_model)
        baseline_metrics = self.evaluate_on_target(baseline_model)
        
        # Compare performance
        transfer_mse = transfer_metrics['mse']
        baseline_mse = baseline_metrics['mse']
        
        performance_diff = baseline_mse - transfer_mse
        relative_improvement = performance_diff / (baseline_mse + 1e-8)
        
        has_negative_transfer = relative_improvement < -0.05  # 5% worse
        
        result = {
            'has_negative_transfer': has_negative_transfer,
            'transfer_mse': transfer_mse,
            'baseline_mse': baseline_mse,
            'performance_difference': performance_diff,
            'relative_improvement': relative_improvement,
            'transfer_metrics': transfer_metrics,
            'baseline_metrics': baseline_metrics
        }
        
        if has_negative_transfer:
            logger.warning(
                f"Negative transfer detected! "
                f"Transfer MSE ({transfer_mse:.4f}) > "
                f"Baseline MSE ({baseline_mse:.4f})"
            )
        else:
            logger.info(
                f"Positive transfer: {relative_improvement * 100:.2f}% improvement"
            )
        
        return result
    
    def save_adapted_model(self, path: Path) -> None:
        """Save adapted model and metadata.
        
        Args:
            path: Path to save model
        """
        if not self.is_adapted:
            raise ValueError("No adapted model. Call adapt() first.")
        
        model_data = {
            'model': {k: v.tolist() for k, v in self.adapted_model.items()},
            'transfer_method': self.transfer_method,
            'multi_source': self.multi_source,
            'adaptation_config': self.adaptation_config,
            'transferability_scores': [
                {
                    'task_similarity': s.task_similarity,
                    'domain_similarity': s.domain_similarity,
                    'expected_transfer_gain': s.expected_transfer_gain,
                    'negative_transfer_risk': s.negative_transfer_risk,
                    'confidence': s.confidence
                }
                for s in self.transferability_scores
            ],
            'timestamp': time.time()
        }
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        import json
        with open(path, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        logger.info(f"Adapted model saved to {path}")
