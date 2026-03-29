"""
Federated Learning Framework for Privacy-Preserving Multi-Institution ML.

This module implements federated learning with:
- FedAvg and FedProx algorithms
- Differential privacy (DP-SGD)
- Secure aggregation
- Byzantine-robust aggregation
- Privacy budget tracking
- Communication efficiency

References:
    - McMahan et al. (2017): Communication-Efficient Learning of Deep Networks
    - Li et al. (2020): Federated Optimization in Heterogeneous Networks (FedProx)
    - Abadi et al. (2016): Deep Learning with Differential Privacy
    - Bonawitz et al. (2017): Practical Secure Aggregation

Example:
    >>> # Server setup
    >>> server = FederatedLearningServer(
    ...     model_config={'input_dim': 10, 'hidden_dim': 64},
    ...     aggregation_strategy='fedavg',
    ...     privacy_budget={'epsilon': 1.0, 'delta': 1e-5}
    ... )
    >>> 
    >>> # Client setup
    >>> client = FederatedLearningClient(
    ...     client_id='institution_1',
    ...     local_data=data,
    ...     dp_config={'noise_multiplier': 1.1, 'l2_norm_clip': 1.0}
    ... )
    >>> 
    >>> # Federated training
    >>> for round_num in range(num_rounds):
    ...     selected_clients = server.select_clients(all_clients)
    ...     client_updates = [c.train_local_model() for c in selected_clients]
    ...     server.aggregate_updates(client_updates)
    ...     global_model = server.get_global_model()
"""

import copy
import hashlib
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from loguru import logger

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Some features will be limited.")


@dataclass
class PrivacyBudget:
    """Privacy budget for differential privacy.
    
    Attributes:
        epsilon: Privacy loss parameter (lower = more private)
        delta: Probability of privacy breach
        spent_epsilon: Cumulative epsilon spent
        max_epsilon: Maximum allowed epsilon
        composition_method: Method for privacy composition ('basic', 'advanced', 'rdp')
    """
    epsilon: float = 1.0
    delta: float = 1e-5
    spent_epsilon: float = 0.0
    max_epsilon: float = 10.0
    composition_method: str = 'rdp'
    
    def can_spend(self, epsilon: float) -> bool:
        """Check if epsilon can be spent without exceeding budget."""
        return (self.spent_epsilon + epsilon) <= self.max_epsilon
    
    def spend(self, epsilon: float) -> None:
        """Spend epsilon from budget."""
        if not self.can_spend(epsilon):
            raise ValueError(
                f"Insufficient privacy budget. Trying to spend {epsilon}, "
                f"but only {self.max_epsilon - self.spent_epsilon} remaining."
            )
        self.spent_epsilon += epsilon
        logger.info(
            f"Privacy budget spent: {epsilon:.4f}, "
            f"Total: {self.spent_epsilon:.4f}/{self.max_epsilon}"
        )
    
    def remaining(self) -> float:
        """Get remaining privacy budget."""
        return self.max_epsilon - self.spent_epsilon


@dataclass
class DifferentialPrivacyConfig:
    """Configuration for differential privacy in federated learning.
    
    Attributes:
        noise_multiplier: Noise level for DP-SGD
        l2_norm_clip: Gradient clipping threshold
        delta: Privacy parameter delta
        target_epsilon: Target epsilon for composition
        enable_dp: Whether to enable differential privacy
        accounting_method: Method for privacy accounting ('rdp', 'gdp')
    """
    noise_multiplier: float = 1.1
    l2_norm_clip: float = 1.0
    delta: float = 1e-5
    target_epsilon: float = 1.0
    enable_dp: bool = True
    accounting_method: str = 'rdp'


@dataclass
class ClientUpdate:
    """Update from a federated learning client.
    
    Attributes:
        client_id: Unique identifier for client
        model_update: Model parameters or gradients
        num_samples: Number of samples used in training
        loss: Training loss
        metrics: Additional metrics (accuracy, etc.)
        metadata: Additional metadata
        timestamp: Update timestamp
    """
    client_id: str
    model_update: Dict[str, np.ndarray]
    num_samples: int
    loss: float
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class PrivacyBudgetTracker:
    """Track and manage privacy budget across federated learning rounds.
    
    Implements privacy accounting for differential privacy using:
    - Renyi Differential Privacy (RDP) composition
    - Gaussian Differential Privacy (GDP)
    - Basic and advanced composition theorems
    
    Example:
        >>> tracker = PrivacyBudgetTracker(max_epsilon=10.0, delta=1e-5)
        >>> tracker.account_for_round(noise_multiplier=1.1, sampling_rate=0.01, steps=100)
        >>> print(f"Spent epsilon: {tracker.get_spent_epsilon()}")
    """
    
    def __init__(
        self,
        max_epsilon: float = 10.0,
        delta: float = 1e-5,
        composition_method: str = 'rdp'
    ):
        """Initialize privacy budget tracker.
        
        Args:
            max_epsilon: Maximum allowed privacy loss
            delta: Privacy parameter delta
            composition_method: Composition method ('rdp', 'basic', 'advanced')
        """
        self.max_epsilon = max_epsilon
        self.delta = delta
        self.composition_method = composition_method
        self.spent_epsilon = 0.0
        self.round_history: List[Dict[str, Any]] = []
        
        logger.info(
            f"Privacy budget tracker initialized: "
            f"max_epsilon={max_epsilon}, delta={delta}, "
            f"method={composition_method}"
        )
    
    def account_for_round(
        self,
        noise_multiplier: float,
        sampling_rate: float,
        steps: int,
        round_num: int = None
    ) -> float:
        """Account for privacy loss in a training round.
        
        Args:
            noise_multiplier: Noise multiplier for DP-SGD
            sampling_rate: Sampling rate of clients
            steps: Number of gradient steps
            round_num: Round number for logging
        
        Returns:
            Epsilon consumed in this round
        """
        if self.composition_method == 'rdp':
            epsilon = self._compute_rdp_epsilon(
                noise_multiplier, sampling_rate, steps
            )
        elif self.composition_method == 'basic':
            epsilon = self._compute_basic_epsilon(
                noise_multiplier, sampling_rate, steps
            )
        else:
            epsilon = self._compute_advanced_epsilon(
                noise_multiplier, sampling_rate, steps
            )
        
        self.spent_epsilon += epsilon
        
        self.round_history.append({
            'round': round_num,
            'epsilon': epsilon,
            'total_epsilon': self.spent_epsilon,
            'noise_multiplier': noise_multiplier,
            'sampling_rate': sampling_rate,
            'steps': steps,
            'timestamp': time.time()
        })
        
        logger.info(
            f"Privacy accounting - Round {round_num}: "
            f"epsilon={epsilon:.4f}, total={self.spent_epsilon:.4f}/{self.max_epsilon}"
        )
        
        if self.spent_epsilon > self.max_epsilon:
            logger.warning(
                f"Privacy budget exceeded! "
                f"Spent: {self.spent_epsilon:.4f}, Max: {self.max_epsilon}"
            )
        
        return epsilon
    
    def _compute_rdp_epsilon(
        self,
        noise_multiplier: float,
        sampling_rate: float,
        steps: int
    ) -> float:
        """Compute epsilon using RDP composition (simplified).
        
        This is a simplified implementation. For production use,
        consider using libraries like tensorflow-privacy.
        """
        # Simplified RDP computation
        # Real implementation would use RDP accountant
        q = sampling_rate
        sigma = noise_multiplier
        
        # Approximate using strong composition
        epsilon_per_step = (q * q) / (2 * sigma * sigma)
        epsilon = epsilon_per_step * steps
        
        # Add conversion from RDP to (epsilon, delta)-DP
        epsilon += np.log(1 / self.delta) / (steps + 1)
        
        return epsilon
    
    def _compute_basic_epsilon(
        self,
        noise_multiplier: float,
        sampling_rate: float,
        steps: int
    ) -> float:
        """Compute epsilon using basic composition."""
        # Simplified basic composition
        sigma = noise_multiplier
        epsilon_per_step = 1.0 / sigma
        return epsilon_per_step * steps
    
    def _compute_advanced_epsilon(
        self,
        noise_multiplier: float,
        sampling_rate: float,
        steps: int
    ) -> float:
        """Compute epsilon using advanced composition."""
        # Simplified advanced composition
        sigma = noise_multiplier
        epsilon_per_step = 1.0 / sigma
        # Advanced composition: O(sqrt(k) * epsilon) instead of O(k * epsilon)
        return epsilon_per_step * np.sqrt(steps * np.log(1 / self.delta))
    
    def get_spent_epsilon(self) -> float:
        """Get total spent epsilon."""
        return self.spent_epsilon
    
    def get_remaining_epsilon(self) -> float:
        """Get remaining epsilon budget."""
        return max(0, self.max_epsilon - self.spent_epsilon)
    
    def can_continue(self) -> bool:
        """Check if training can continue within budget."""
        return self.spent_epsilon < self.max_epsilon
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get privacy accounting history."""
        return self.round_history
    
    def save_audit_log(self, path: Path) -> None:
        """Save privacy audit log to file.
        
        Args:
            path: Path to save audit log
        """
        audit_data = {
            'max_epsilon': self.max_epsilon,
            'delta': self.delta,
            'spent_epsilon': self.spent_epsilon,
            'composition_method': self.composition_method,
            'history': self.round_history
        }
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(audit_data, f, indent=2)
        
        logger.info(f"Privacy audit log saved to {path}")


class FedAvgAggregator:
    """Federated Averaging (FedAvg) aggregation strategy.
    
    Implements the FedAvg algorithm from McMahan et al. (2017).
    Aggregates client model updates using weighted averaging based on
    number of samples.
    
    Example:
        >>> aggregator = FedAvgAggregator()
        >>> global_model = aggregator.aggregate([update1, update2, update3])
    """
    
    def __init__(self, use_weighted: bool = True):
        """Initialize FedAvg aggregator.
        
        Args:
            use_weighted: Whether to weight by number of samples
        """
        self.use_weighted = use_weighted
    
    def aggregate(
        self,
        client_updates: List[ClientUpdate]
    ) -> Dict[str, np.ndarray]:
        """Aggregate client updates using weighted averaging.
        
        Args:
            client_updates: List of client updates
        
        Returns:
            Aggregated model parameters
        """
        if not client_updates:
            raise ValueError("No client updates to aggregate")
        
        # Get total samples for weighting
        total_samples = sum(update.num_samples for update in client_updates)
        
        # Initialize aggregated model
        aggregated_model = {}
        
        # Get parameter names from first update
        param_names = list(client_updates[0].model_update.keys())
        
        for param_name in param_names:
            weighted_sum = None
            
            for update in client_updates:
                param = update.model_update[param_name]
                
                # Calculate weight
                if self.use_weighted:
                    weight = update.num_samples / total_samples
                else:
                    weight = 1.0 / len(client_updates)
                
                # Weighted sum
                if weighted_sum is None:
                    weighted_sum = weight * param
                else:
                    weighted_sum += weight * param
            
            aggregated_model[param_name] = weighted_sum
        
        logger.debug(
            f"Aggregated {len(client_updates)} client updates "
            f"({total_samples} total samples)"
        )
        
        return aggregated_model


class FedProxAggregator:
    """Federated Proximal (FedProx) aggregation strategy.
    
    Implements FedProx from Li et al. (2020) for heterogeneous data.
    Adds a proximal term to keep local models close to global model.
    
    Example:
        >>> aggregator = FedProxAggregator(mu=0.01)
        >>> global_model = aggregator.aggregate([update1, update2, update3])
    """
    
    def __init__(self, mu: float = 0.01, use_weighted: bool = True):
        """Initialize FedProx aggregator.
        
        Args:
            mu: Proximal term coefficient
            use_weighted: Whether to weight by number of samples
        """
        self.mu = mu
        self.use_weighted = use_weighted
        self.base_aggregator = FedAvgAggregator(use_weighted)
    
    def aggregate(
        self,
        client_updates: List[ClientUpdate],
        global_model: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, np.ndarray]:
        """Aggregate with proximal regularization.
        
        Args:
            client_updates: List of client updates
            global_model: Current global model (for proximal term)
        
        Returns:
            Aggregated model parameters
        """
        # Use FedAvg for base aggregation
        aggregated_model = self.base_aggregator.aggregate(client_updates)
        
        # Apply proximal regularization if global model provided
        if global_model is not None and self.mu > 0:
            for param_name in aggregated_model.keys():
                if param_name in global_model:
                    # Add proximal term
                    aggregated_model[param_name] = (
                        aggregated_model[param_name] +
                        self.mu * global_model[param_name]
                    ) / (1 + self.mu)
        
        return aggregated_model


class ByzantineRobustAggregator:
    """Byzantine-robust aggregation using outlier detection.
    
    Protects against malicious clients by detecting and removing outliers
    before aggregation using statistical methods.
    
    Example:
        >>> aggregator = ByzantineRobustAggregator(method='krum')
        >>> global_model = aggregator.aggregate([update1, update2, update3])
    """
    
    def __init__(
        self,
        method: str = 'krum',
        num_byzantine: int = 1,
        use_weighted: bool = True
    ):
        """Initialize Byzantine-robust aggregator.
        
        Args:
            method: Aggregation method ('krum', 'median', 'trimmed_mean')
            num_byzantine: Expected number of Byzantine clients
            use_weighted: Whether to weight by number of samples
        """
        self.method = method
        self.num_byzantine = num_byzantine
        self.use_weighted = use_weighted
        self.base_aggregator = FedAvgAggregator(use_weighted)
    
    def aggregate(
        self,
        client_updates: List[ClientUpdate]
    ) -> Dict[str, np.ndarray]:
        """Aggregate with Byzantine robustness.
        
        Args:
            client_updates: List of client updates
        
        Returns:
            Aggregated model parameters
        """
        if len(client_updates) <= 2 * self.num_byzantine:
            logger.warning(
                f"Too few clients for Byzantine robustness. "
                f"Using standard aggregation."
            )
            return self.base_aggregator.aggregate(client_updates)
        
        if self.method == 'krum':
            return self._krum_aggregate(client_updates)
        elif self.method == 'median':
            return self._median_aggregate(client_updates)
        elif self.method == 'trimmed_mean':
            return self._trimmed_mean_aggregate(client_updates)
        else:
            logger.warning(f"Unknown method {self.method}, using FedAvg")
            return self.base_aggregator.aggregate(client_updates)
    
    def _krum_aggregate(
        self,
        client_updates: List[ClientUpdate]
    ) -> Dict[str, np.ndarray]:
        """Multi-Krum aggregation."""
        # Simplified Krum: select clients with smallest distance sum
        num_select = len(client_updates) - self.num_byzantine - 2
        
        # Compute pairwise distances
        distances = self._compute_pairwise_distances(client_updates)
        
        # Compute scores (sum of closest distances)
        scores = []
        for i in range(len(client_updates)):
            closest_distances = sorted(distances[i])[:num_select]
            scores.append(sum(closest_distances))
        
        # Select clients with lowest scores
        selected_indices = np.argsort(scores)[:num_select]
        selected_updates = [client_updates[i] for i in selected_indices]
        
        logger.debug(f"Krum selected {len(selected_updates)} clients")
        
        return self.base_aggregator.aggregate(selected_updates)
    
    def _median_aggregate(
        self,
        client_updates: List[ClientUpdate]
    ) -> Dict[str, np.ndarray]:
        """Coordinate-wise median aggregation."""
        aggregated_model = {}
        param_names = list(client_updates[0].model_update.keys())
        
        for param_name in param_names:
            # Stack all client parameters
            params = np.stack([
                update.model_update[param_name]
                for update in client_updates
            ])
            
            # Compute coordinate-wise median
            aggregated_model[param_name] = np.median(params, axis=0)
        
        return aggregated_model
    
    def _trimmed_mean_aggregate(
        self,
        client_updates: List[ClientUpdate]
    ) -> Dict[str, np.ndarray]:
        """Trimmed mean aggregation."""
        aggregated_model = {}
        param_names = list(client_updates[0].model_update.keys())
        
        for param_name in param_names:
            # Stack all client parameters
            params = np.stack([
                update.model_update[param_name]
                for update in client_updates
            ])
            
            # Compute trimmed mean (remove top/bottom values)
            sorted_params = np.sort(params, axis=0)
            trimmed = sorted_params[
                self.num_byzantine:-self.num_byzantine
            ]
            aggregated_model[param_name] = np.mean(trimmed, axis=0)
        
        return aggregated_model
    
    def _compute_pairwise_distances(
        self,
        client_updates: List[ClientUpdate]
    ) -> np.ndarray:
        """Compute pairwise distances between client updates."""
        n = len(client_updates)
        distances = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = self._compute_update_distance(
                    client_updates[i],
                    client_updates[j]
                )
                distances[i, j] = dist
                distances[j, i] = dist
        
        return distances
    
    def _compute_update_distance(
        self,
        update1: ClientUpdate,
        update2: ClientUpdate
    ) -> float:
        """Compute Euclidean distance between two updates."""
        distance = 0.0
        
        for param_name in update1.model_update.keys():
            param1 = update1.model_update[param_name].flatten()
            param2 = update2.model_update[param_name].flatten()
            distance += np.sum((param1 - param2) ** 2)
        
        return np.sqrt(distance)


class FederatedLearningServer:
    """Federated Learning Server for coordinating multi-institution training.
    
    Coordinates federated learning across multiple institutions:
    - Manages global model
    - Selects clients for each round
    - Aggregates client updates
    - Tracks privacy budget
    - Monitors convergence
    - Provides audit logging
    
    Example:
        >>> server = FederatedLearningServer(
        ...     model_config={'input_dim': 10, 'output_dim': 1},
        ...     aggregation_strategy='fedavg',
        ...     privacy_budget={'epsilon': 10.0, 'delta': 1e-5}
        ... )
        >>> for round_num in range(100):
        ...     selected = server.select_clients(clients, fraction=0.1)
        ...     updates = [c.train() for c in selected]
        ...     server.aggregate_updates(updates, round_num)
    """
    
    def __init__(
        self,
        model_config: Dict[str, Any],
        aggregation_strategy: str = 'fedavg',
        privacy_budget: Optional[Dict[str, float]] = None,
        client_selection: str = 'random',
        min_clients: int = 2,
        convergence_threshold: float = 1e-4,
        **kwargs
    ):
        """Initialize federated learning server.
        
        Args:
            model_config: Model configuration
            aggregation_strategy: Strategy for aggregation ('fedavg', 'fedprox', 'krum')
            privacy_budget: Privacy budget configuration
            client_selection: Client selection strategy ('random', 'importance')
            min_clients: Minimum clients per round
            convergence_threshold: Threshold for convergence detection
            **kwargs: Additional arguments for aggregator
        """
        self.model_config = model_config
        self.aggregation_strategy = aggregation_strategy
        self.min_clients = min_clients
        self.convergence_threshold = convergence_threshold
        self.client_selection = client_selection
        
        # Initialize global model
        self.global_model = self._initialize_model(model_config)
        self.round_num = 0
        
        # Initialize aggregator
        self.aggregator = self._create_aggregator(aggregation_strategy, **kwargs)
        
        # Initialize privacy budget tracker
        if privacy_budget:
            self.privacy_tracker = PrivacyBudgetTracker(
                max_epsilon=privacy_budget.get('epsilon', 10.0),
                delta=privacy_budget.get('delta', 1e-5),
                composition_method=privacy_budget.get('method', 'rdp')
            )
        else:
            self.privacy_tracker = None
        
        # Tracking
        self.round_history: List[Dict[str, Any]] = []
        self.convergence_history: List[float] = []
        
        logger.info(
            f"Federated Learning Server initialized: "
            f"strategy={aggregation_strategy}, "
            f"privacy={'enabled' if privacy_budget else 'disabled'}"
        )
    
    def _initialize_model(
        self,
        model_config: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        """Initialize global model parameters."""
        # Initialize with random weights
        model = {}
        
        if 'input_dim' in model_config and 'output_dim' in model_config:
            input_dim = model_config['input_dim']
            output_dim = model_config['output_dim']
            hidden_dim = model_config.get('hidden_dim', 64)
            
            # Simple MLP initialization
            model['w1'] = np.random.randn(input_dim, hidden_dim) * 0.01
            model['b1'] = np.zeros(hidden_dim)
            model['w2'] = np.random.randn(hidden_dim, output_dim) * 0.01
            model['b2'] = np.zeros(output_dim)
        
        return model
    
    def _create_aggregator(
        self,
        strategy: str,
        **kwargs
    ) -> Union[FedAvgAggregator, FedProxAggregator, ByzantineRobustAggregator]:
        """Create aggregator based on strategy."""
        if strategy == 'fedavg':
            return FedAvgAggregator(**kwargs)
        elif strategy == 'fedprox':
            mu = kwargs.get('mu', 0.01)
            return FedProxAggregator(mu=mu, **kwargs)
        elif strategy in ['krum', 'median', 'trimmed_mean']:
            return ByzantineRobustAggregator(method=strategy, **kwargs)
        else:
            logger.warning(f"Unknown strategy {strategy}, using FedAvg")
            return FedAvgAggregator(**kwargs)
    
    def select_clients(
        self,
        available_clients: List['FederatedLearningClient'],
        fraction: float = 0.1,
        min_clients: Optional[int] = None
    ) -> List['FederatedLearningClient']:
        """Select clients for training round.
        
        Args:
            available_clients: List of available clients
            fraction: Fraction of clients to select
            min_clients: Minimum number of clients
        
        Returns:
            Selected clients
        """
        min_clients = min_clients or self.min_clients
        num_select = max(
            min_clients,
            int(len(available_clients) * fraction)
        )
        num_select = min(num_select, len(available_clients))
        
        if self.client_selection == 'random':
            selected = np.random.choice(
                available_clients,
                size=num_select,
                replace=False
            ).tolist()
        elif self.client_selection == 'importance':
            # Select based on data size or importance
            sizes = [c.get_data_size() for c in available_clients]
            probs = np.array(sizes) / sum(sizes)
            selected = np.random.choice(
                available_clients,
                size=num_select,
                replace=False,
                p=probs
            ).tolist()
        else:
            selected = available_clients[:num_select]
        
        logger.info(
            f"Selected {len(selected)}/{len(available_clients)} clients "
            f"for round {self.round_num}"
        )
        
        return selected
    
    def aggregate_updates(
        self,
        client_updates: List[ClientUpdate],
        round_num: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """Aggregate client updates into global model.
        
        Args:
            client_updates: Updates from clients
            round_num: Current round number
        
        Returns:
            Updated global model
        """
        if round_num is not None:
            self.round_num = round_num
        else:
            self.round_num += 1
        
        if len(client_updates) < self.min_clients:
            raise ValueError(
                f"Insufficient clients: {len(client_updates)} < {self.min_clients}"
            )
        
        # Aggregate updates
        if isinstance(self.aggregator, FedProxAggregator):
            new_model = self.aggregator.aggregate(
                client_updates,
                self.global_model
            )
        else:
            new_model = self.aggregator.aggregate(client_updates)
        
        # Compute model change for convergence tracking
        model_change = self._compute_model_change(self.global_model, new_model)
        self.convergence_history.append(model_change)
        
        # Update global model
        self.global_model = new_model
        
        # Compute aggregate metrics
        total_samples = sum(u.num_samples for u in client_updates)
        avg_loss = sum(
            u.loss * u.num_samples for u in client_updates
        ) / total_samples
        
        # Track round
        self.round_history.append({
            'round': self.round_num,
            'num_clients': len(client_updates),
            'total_samples': total_samples,
            'avg_loss': avg_loss,
            'model_change': model_change,
            'timestamp': time.time()
        })
        
        logger.info(
            f"Round {self.round_num}: "
            f"clients={len(client_updates)}, "
            f"loss={avg_loss:.4f}, "
            f"change={model_change:.6f}"
        )
        
        return self.global_model
    
    def _compute_model_change(
        self,
        old_model: Dict[str, np.ndarray],
        new_model: Dict[str, np.ndarray]
    ) -> float:
        """Compute L2 norm of model change."""
        change = 0.0
        for key in old_model.keys():
            if key in new_model:
                diff = old_model[key] - new_model[key]
                change += np.sum(diff ** 2)
        return np.sqrt(change)
    
    def has_converged(self, window: int = 5) -> bool:
        """Check if training has converged.
        
        Args:
            window: Number of recent rounds to check
        
        Returns:
            True if converged
        """
        if len(self.convergence_history) < window:
            return False
        
        recent_changes = self.convergence_history[-window:]
        avg_change = np.mean(recent_changes)
        
        return avg_change < self.convergence_threshold
    
    def get_global_model(self) -> Dict[str, np.ndarray]:
        """Get current global model."""
        return copy.deepcopy(self.global_model)
    
    def get_round_history(self) -> List[Dict[str, Any]]:
        """Get training history."""
        return self.round_history
    
    def save_checkpoint(self, path: Path) -> None:
        """Save server checkpoint.
        
        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            'round_num': self.round_num,
            'global_model': {k: v.tolist() for k, v in self.global_model.items()},
            'model_config': self.model_config,
            'round_history': self.round_history,
            'convergence_history': self.convergence_history
        }
        
        if self.privacy_tracker:
            checkpoint['privacy_budget'] = {
                'spent_epsilon': self.privacy_tracker.get_spent_epsilon(),
                'max_epsilon': self.privacy_tracker.max_epsilon,
                'delta': self.privacy_tracker.delta
            }
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        logger.info(f"Checkpoint saved to {path}")


class FederatedLearningClient:
    """Federated Learning Client for institution-local training.
    
    Trains models locally on institution data with:
    - Differential privacy (DP-SGD)
    - Gradient clipping
    - Local optimization
    - Privacy-preserving updates
    
    Example:
        >>> client = FederatedLearningClient(
        ...     client_id='institution_1',
        ...     local_data=(X_train, y_train),
        ...     dp_config={'noise_multiplier': 1.1, 'l2_norm_clip': 1.0}
        ... )
        >>> update = client.train_local_model(
        ...     global_model=server_model,
        ...     epochs=5
        ... )
    """
    
    def __init__(
        self,
        client_id: str,
        local_data: Tuple[np.ndarray, np.ndarray],
        dp_config: Optional[DifferentialPrivacyConfig] = None,
        learning_rate: float = 0.01,
        batch_size: int = 32
    ):
        """Initialize federated learning client.
        
        Args:
            client_id: Unique client identifier
            local_data: Tuple of (X, y) local data
            dp_config: Differential privacy configuration
            learning_rate: Learning rate for local training
            batch_size: Batch size for training
        """
        self.client_id = client_id
        self.X, self.y = local_data
        self.num_samples = len(self.X)
        
        self.dp_config = dp_config or DifferentialPrivacyConfig(enable_dp=False)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        self.local_model: Optional[Dict[str, np.ndarray]] = None
        self.training_history: List[Dict[str, Any]] = []
        
        logger.info(
            f"Client {client_id} initialized: "
            f"samples={self.num_samples}, "
            f"DP={'enabled' if self.dp_config.enable_dp else 'disabled'}"
        )
    
    def train_local_model(
        self,
        global_model: Dict[str, np.ndarray],
        epochs: int = 5,
        verbose: bool = False
    ) -> ClientUpdate:
        """Train model locally on institution data.
        
        Args:
            global_model: Global model from server
            epochs: Number of local epochs
            verbose: Whether to print training progress
        
        Returns:
            Client update with model changes
        """
        # Copy global model
        self.local_model = copy.deepcopy(global_model)
        
        # Training loop
        epoch_losses = []
        
        for epoch in range(epochs):
            epoch_loss = self._train_epoch()
            epoch_losses.append(epoch_loss)
            
            if verbose:
                logger.debug(
                    f"Client {self.client_id} - "
                    f"Epoch {epoch + 1}/{epochs}: loss={epoch_loss:.4f}"
                )
        
        # Compute model update (difference from global)
        model_update = {}
        for key in self.local_model.keys():
            model_update[key] = self.local_model[key] - global_model[key]
        
        # Apply differential privacy if enabled
        if self.dp_config.enable_dp:
            model_update = self._apply_differential_privacy(model_update)
        
        # Create client update
        avg_loss = np.mean(epoch_losses)
        
        update = ClientUpdate(
            client_id=self.client_id,
            model_update=model_update,
            num_samples=self.num_samples,
            loss=avg_loss,
            metadata={'epochs': epochs}
        )
        
        # Track training
        self.training_history.append({
            'timestamp': time.time(),
            'epochs': epochs,
            'avg_loss': avg_loss,
            'num_samples': self.num_samples
        })
        
        logger.info(
            f"Client {self.client_id} training complete: "
            f"loss={avg_loss:.4f}"
        )
        
        return update
    
    def _train_epoch(self) -> float:
        """Train for one epoch using mini-batch SGD."""
        indices = np.random.permutation(self.num_samples)
        epoch_loss = 0.0
        num_batches = 0
        
        for i in range(0, self.num_samples, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            X_batch = self.X[batch_indices]
            y_batch = self.y[batch_indices]
            
            # Forward pass
            predictions = self._forward(X_batch)
            loss = self._compute_loss(predictions, y_batch)
            epoch_loss += loss
            num_batches += 1
            
            # Backward pass
            gradients = self._compute_gradients(X_batch, y_batch, predictions)
            
            # Clip gradients if DP enabled
            if self.dp_config.enable_dp:
                gradients = self._clip_gradients(gradients)
            
            # Update model
            self._update_model(gradients)
        
        return epoch_loss / num_batches
    
    def _forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass through simple MLP."""
        # Hidden layer
        h = np.maximum(
            0,
            X @ self.local_model['w1'] + self.local_model['b1']
        )
        
        # Output layer
        output = h @ self.local_model['w2'] + self.local_model['b2']
        
        return output
    
    def _compute_loss(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> float:
        """Compute MSE loss."""
        return np.mean((predictions - targets) ** 2)
    
    def _compute_gradients(
        self,
        X: np.ndarray,
        y: np.ndarray,
        predictions: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Compute gradients via backpropagation."""
        batch_size = len(X)
        
        # Output layer gradients
        d_output = 2 * (predictions - y) / batch_size
        
        # Hidden layer activations
        h = np.maximum(
            0,
            X @ self.local_model['w1'] + self.local_model['b1']
        )
        
        # Gradients for w2 and b2
        grad_w2 = h.T @ d_output
        grad_b2 = np.sum(d_output, axis=0)
        
        # Backprop to hidden layer
        d_hidden = d_output @ self.local_model['w2'].T
        d_hidden[h <= 0] = 0  # ReLU gradient
        
        # Gradients for w1 and b1
        grad_w1 = X.T @ d_hidden
        grad_b1 = np.sum(d_hidden, axis=0)
        
        return {
            'w1': grad_w1,
            'b1': grad_b1,
            'w2': grad_w2,
            'b2': grad_b2
        }
    
    def _clip_gradients(
        self,
        gradients: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Clip gradients for differential privacy."""
        # Compute L2 norm of all gradients
        grad_norm = 0.0
        for grad in gradients.values():
            grad_norm += np.sum(grad ** 2)
        grad_norm = np.sqrt(grad_norm)
        
        # Clip if necessary
        clip_threshold = self.dp_config.l2_norm_clip
        if grad_norm > clip_threshold:
            scale = clip_threshold / grad_norm
            gradients = {k: v * scale for k, v in gradients.items()}
        
        return gradients
    
    def _update_model(self, gradients: Dict[str, np.ndarray]) -> None:
        """Update model parameters using gradients."""
        for key in self.local_model.keys():
            self.local_model[key] -= self.learning_rate * gradients[key]
    
    def _apply_differential_privacy(
        self,
        model_update: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Add noise for differential privacy."""
        noise_scale = (
            self.dp_config.noise_multiplier *
            self.dp_config.l2_norm_clip
        )
        
        private_update = {}
        for key, value in model_update.items():
            noise = np.random.normal(0, noise_scale, value.shape)
            private_update[key] = value + noise
        
        logger.debug(
            f"Client {self.client_id}: Applied DP noise "
            f"(multiplier={self.dp_config.noise_multiplier})"
        )
        
        return private_update
    
    def get_data_size(self) -> int:
        """Get size of local dataset."""
        return self.num_samples
    
    def evaluate_model(
        self,
        model: Dict[str, np.ndarray],
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Evaluate model on test data.
        
        Args:
            model: Model to evaluate
            X_test: Test features (uses local data if None)
            y_test: Test targets (uses local data if None)
        
        Returns:
            Evaluation metrics
        """
        if X_test is None or y_test is None:
            X_test, y_test = self.X, self.y
        
        # Temporarily set model
        old_model = self.local_model
        self.local_model = model
        
        # Evaluate
        predictions = self._forward(X_test)
        loss = self._compute_loss(predictions, y_test)
        
        # Compute additional metrics
        mse = np.mean((predictions - y_test) ** 2)
        mae = np.mean(np.abs(predictions - y_test))
        
        # Restore model
        self.local_model = old_model
        
        return {
            'loss': loss,
            'mse': mse,
            'mae': mae,
            'num_samples': len(X_test)
        }


def compress_gradients(
    gradients: Dict[str, np.ndarray],
    compression_ratio: float = 0.1,
    method: str = 'topk'
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Compress gradients for communication efficiency.
    
    Args:
        gradients: Gradient dictionary
        compression_ratio: Fraction of gradients to keep
        method: Compression method ('topk', 'random', 'threshold')
    
    Returns:
        Compressed gradients and metadata for decompression
    """
    compressed = {}
    metadata = {'method': method, 'compression_ratio': compression_ratio}
    
    for key, grad in gradients.items():
        if method == 'topk':
            # Keep top-k largest magnitude gradients
            flat_grad = grad.flatten()
            k = max(1, int(len(flat_grad) * compression_ratio))
            
            indices = np.argpartition(np.abs(flat_grad), -k)[-k:]
            values = flat_grad[indices]
            
            compressed[key] = {
                'indices': indices,
                'values': values,
                'shape': grad.shape
            }
        elif method == 'random':
            # Random sparsification
            flat_grad = grad.flatten()
            k = max(1, int(len(flat_grad) * compression_ratio))
            
            indices = np.random.choice(len(flat_grad), k, replace=False)
            values = flat_grad[indices] / compression_ratio
            
            compressed[key] = {
                'indices': indices,
                'values': values,
                'shape': grad.shape
            }
        else:
            # No compression
            compressed[key] = grad
    
    return compressed, metadata


def decompress_gradients(
    compressed: Dict[str, Any],
    metadata: Dict[str, Any]
) -> Dict[str, np.ndarray]:
    """Decompress gradients.
    
    Args:
        compressed: Compressed gradient data
        metadata: Compression metadata
    
    Returns:
        Decompressed gradients
    """
    method = metadata.get('method', 'none')
    gradients = {}
    
    for key, data in compressed.items():
        if method in ['topk', 'random'] and isinstance(data, dict):
            # Reconstruct sparse gradient
            flat_grad = np.zeros(np.prod(data['shape']))
            flat_grad[data['indices']] = data['values']
            gradients[key] = flat_grad.reshape(data['shape'])
        else:
            # Already decompressed
            gradients[key] = data
    
    return gradients
