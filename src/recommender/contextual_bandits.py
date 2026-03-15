"""
Contextual bandit algorithms for next best action recommendation.

This module provides multiple contextual bandit implementations for real-time
decision-making under uncertainty, with support for online learning, A/B testing,
and exploration-exploitation balance.

Example:
    >>> from src.recommender.contextual_bandits import LinUCBBandit
    >>> bandit = LinUCBBandit(n_actions=5, n_features=10, alpha=1.0)
    >>> action = bandit.select_action(context)
    >>> bandit.update(action, reward, context)
"""

import joblib
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from loguru import logger
from scipy.stats import beta
from sklearn.linear_model import Ridge


class BaseContextualBandit:
    """Base class for contextual bandit algorithms."""
    
    def __init__(self, n_actions: int, random_state: Optional[int] = None):
        """
        Initialize base contextual bandit.
        
        Args:
            n_actions: Number of available actions
            random_state: Random seed for reproducibility
        """
        self.n_actions = n_actions
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        
        # Tracking metrics
        self.n_pulls = np.zeros(n_actions, dtype=int)
        self.total_reward = np.zeros(n_actions)
        self.rewards_history: List[float] = []
        self.actions_history: List[int] = []
        self.regret_history: List[float] = []
        
    def select_action(self, context: Optional[np.ndarray] = None) -> int:
        """Select an action given context."""
        raise NotImplementedError
        
    def update(
        self,
        action: int,
        reward: float,
        context: Optional[np.ndarray] = None
    ) -> None:
        """Update bandit with observed reward."""
        self.n_pulls[action] += 1
        self.total_reward[action] += reward
        self.rewards_history.append(reward)
        self.actions_history.append(action)
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        metrics = {
            'n_pulls': self.n_pulls.tolist(),
            'total_reward': self.total_reward.tolist(),
            'avg_reward_per_action': (self.total_reward / np.maximum(self.n_pulls, 1)).tolist(),
            'total_actions': len(self.actions_history),
            'cumulative_reward': sum(self.rewards_history),
            'avg_reward': np.mean(self.rewards_history) if self.rewards_history else 0.0,
            'cumulative_regret': sum(self.regret_history) if self.regret_history else 0.0
        }
        return metrics
        
    def calculate_regret(self, optimal_reward: float) -> float:
        """Calculate regret for last action."""
        if not self.rewards_history:
            return 0.0
        regret = optimal_reward - self.rewards_history[-1]
        self.regret_history.append(regret)
        return regret
        
    def save(self, filepath: Union[str, Path]) -> None:
        """Save bandit to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.__dict__, filepath)
        logger.info(f"Bandit saved to {filepath}")
        
    def load(self, filepath: Union[str, Path]) -> 'BaseContextualBandit':
        """Load bandit from disk."""
        data = joblib.load(filepath)
        self.__dict__.update(data)
        logger.info(f"Bandit loaded from {filepath}")
        return self


class EpsilonGreedyBandit(BaseContextualBandit):
    """
    Epsilon-greedy bandit with decaying exploration.
    
    Balances exploration and exploitation using epsilon-greedy strategy.
    
    Example:
        >>> bandit = EpsilonGreedyBandit(n_actions=5, epsilon=0.1)
        >>> action = bandit.select_action()
        >>> bandit.update(action, reward=1.0)
    """
    
    def __init__(
        self,
        n_actions: int,
        epsilon: float = 0.1,
        decay_rate: float = 0.99,
        min_epsilon: float = 0.01,
        random_state: Optional[int] = None
    ):
        """
        Initialize epsilon-greedy bandit.
        
        Args:
            n_actions: Number of available actions
            epsilon: Initial exploration probability
            decay_rate: Rate at which epsilon decays
            min_epsilon: Minimum epsilon value
            random_state: Random seed
        """
        super().__init__(n_actions, random_state)
        self.epsilon = epsilon
        self.initial_epsilon = epsilon
        self.decay_rate = decay_rate
        self.min_epsilon = min_epsilon
        
    def select_action(self, context: Optional[np.ndarray] = None) -> int:
        """Select action using epsilon-greedy strategy."""
        if self.rng.random() < self.epsilon:
            # Explore: random action
            action = self.rng.randint(0, self.n_actions)
        else:
            # Exploit: best action so far
            avg_rewards = self.total_reward / np.maximum(self.n_pulls, 1)
            action = int(np.argmax(avg_rewards))
            
        return action
        
    def update(
        self,
        action: int,
        reward: float,
        context: Optional[np.ndarray] = None
    ) -> None:
        """Update with reward and decay epsilon."""
        super().update(action, reward, context)
        # Decay epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate)


class UCBBandit(BaseContextualBandit):
    """
    Upper Confidence Bound (UCB) bandit.
    
    Selects actions based on upper confidence bounds of reward estimates.
    
    Example:
        >>> bandit = UCBBandit(n_actions=5, c=2.0)
        >>> action = bandit.select_action()
        >>> bandit.update(action, reward=1.0)
    """
    
    def __init__(
        self,
        n_actions: int,
        c: float = 2.0,
        random_state: Optional[int] = None
    ):
        """
        Initialize UCB bandit.
        
        Args:
            n_actions: Number of available actions
            c: Exploration parameter (higher = more exploration)
            random_state: Random seed
        """
        super().__init__(n_actions, random_state)
        self.c = c
        self.t = 0
        
    def select_action(self, context: Optional[np.ndarray] = None) -> int:
        """Select action using UCB strategy."""
        self.t += 1
        
        # Pull each arm once first
        if np.any(self.n_pulls == 0):
            action = int(np.argmin(self.n_pulls))
            return action
            
        # Calculate UCB for each action
        avg_rewards = self.total_reward / self.n_pulls
        ucb_values = avg_rewards + self.c * np.sqrt(np.log(self.t) / self.n_pulls)
        
        action = int(np.argmax(ucb_values))
        return action


class ThompsonSamplingBandit(BaseContextualBandit):
    """
    Thompson Sampling bandit with Beta prior.
    
    Bayesian approach using Beta distribution for reward probabilities.
    
    Example:
        >>> bandit = ThompsonSamplingBandit(n_actions=5)
        >>> action = bandit.select_action()
        >>> bandit.update(action, reward=1.0)
    """
    
    def __init__(
        self,
        n_actions: int,
        alpha_prior: float = 1.0,
        beta_prior: float = 1.0,
        random_state: Optional[int] = None
    ):
        """
        Initialize Thompson Sampling bandit.
        
        Args:
            n_actions: Number of available actions
            alpha_prior: Prior parameter for Beta distribution (successes)
            beta_prior: Prior parameter for Beta distribution (failures)
            random_state: Random seed
        """
        super().__init__(n_actions, random_state)
        self.alpha = np.ones(n_actions) * alpha_prior
        self.beta = np.ones(n_actions) * beta_prior
        
    def select_action(self, context: Optional[np.ndarray] = None) -> int:
        """Select action using Thompson Sampling."""
        # Sample from Beta distribution for each action
        samples = [
            self.rng.beta(self.alpha[i], self.beta[i])
            for i in range(self.n_actions)
        ]
        action = int(np.argmax(samples))
        return action
        
    def update(
        self,
        action: int,
        reward: float,
        context: Optional[np.ndarray] = None
    ) -> None:
        """Update Beta parameters based on reward."""
        super().update(action, reward, context)
        # Update Beta distribution parameters
        self.alpha[action] += reward
        self.beta[action] += (1 - reward)
        
    def get_posterior_means(self) -> np.ndarray:
        """Get posterior mean estimates for each action."""
        return self.alpha / (self.alpha + self.beta)
        
    def get_credible_intervals(self, confidence: float = 0.95) -> np.ndarray:
        """Get credible intervals for reward probabilities."""
        intervals = np.zeros((self.n_actions, 2))
        alpha_level = (1 - confidence) / 2
        
        for i in range(self.n_actions):
            intervals[i, 0] = beta.ppf(alpha_level, self.alpha[i], self.beta[i])
            intervals[i, 1] = beta.ppf(1 - alpha_level, self.alpha[i], self.beta[i])
            
        return intervals


class LinUCBBandit(BaseContextualBandit):
    """
    Linear Upper Confidence Bound (LinUCB) bandit.
    
    Uses linear models to estimate rewards based on context features.
    
    Example:
        >>> bandit = LinUCBBandit(n_actions=5, n_features=10, alpha=1.0)
        >>> context = np.random.randn(10)
        >>> action = bandit.select_action(context)
        >>> bandit.update(action, reward=1.0, context=context)
    """
    
    def __init__(
        self,
        n_actions: int,
        n_features: int,
        alpha: float = 1.0,
        ridge_lambda: float = 1.0,
        random_state: Optional[int] = None
    ):
        """
        Initialize LinUCB bandit.
        
        Args:
            n_actions: Number of available actions
            n_features: Dimension of context features
            alpha: Exploration parameter
            ridge_lambda: Ridge regularization parameter
            random_state: Random seed
        """
        super().__init__(n_actions, random_state)
        self.n_features = n_features
        self.alpha = alpha
        self.ridge_lambda = ridge_lambda
        
        # Initialize design matrices and vectors for each action
        self.A = [
            np.eye(n_features) * ridge_lambda
            for _ in range(n_actions)
        ]
        self.b = [
            np.zeros(n_features)
            for _ in range(n_actions)
        ]
        
    def select_action(self, context: np.ndarray) -> int:
        """Select action using LinUCB."""
        if context is None:
            raise ValueError("LinUCB requires context features")
            
        context = np.asarray(context).flatten()
        if len(context) != self.n_features:
            raise ValueError(
                f"Context dimension {len(context)} doesn't match n_features {self.n_features}"
            )
        
        ucb_values = np.zeros(self.n_actions)
        
        for a in range(self.n_actions):
            # Solve for theta_a
            A_inv = np.linalg.inv(self.A[a])
            theta_a = A_inv @ self.b[a]
            
            # Calculate UCB
            pred_reward = theta_a @ context
            uncertainty = self.alpha * np.sqrt(context @ A_inv @ context)
            ucb_values[a] = pred_reward + uncertainty
            
        action = int(np.argmax(ucb_values))
        return action
        
    def update(
        self,
        action: int,
        reward: float,
        context: np.ndarray
    ) -> None:
        """Update model parameters."""
        super().update(action, reward, context)
        
        context = np.asarray(context).flatten()
        
        # Update design matrix and vector
        self.A[action] += np.outer(context, context)
        self.b[action] += reward * context
        
    def get_coefficients(self) -> List[np.ndarray]:
        """Get learned coefficients for each action."""
        coefs = []
        for a in range(self.n_actions):
            A_inv = np.linalg.inv(self.A[a])
            theta_a = A_inv @ self.b[a]
            coefs.append(theta_a)
        return coefs


class ContextualBanditOrchestrator:
    """
    Orchestrator for managing multiple bandit algorithms and A/B testing.
    
    Provides model selection, comparison, and deployment capabilities.
    
    Example:
        >>> bandits = {
        ...     'thompson': ThompsonSamplingBandit(n_actions=5),
        ...     'ucb': UCBBandit(n_actions=5),
        ...     'linucb': LinUCBBandit(n_actions=5, n_features=10)
        ... }
        >>> orchestrator = ContextualBanditOrchestrator(bandits)
        >>> action = orchestrator.select_action('linucb', context)
        >>> orchestrator.update('linucb', action, reward, context)
    """
    
    def __init__(
        self,
        bandits: Dict[str, BaseContextualBandit],
        default_bandit: Optional[str] = None
    ):
        """
        Initialize orchestrator.
        
        Args:
            bandits: Dictionary of bandit name to bandit instance
            default_bandit: Name of default bandit to use
        """
        self.bandits = bandits
        self.default_bandit = default_bandit or list(bandits.keys())[0]
        self.ab_test_splits: Dict[str, float] = {}
        
    def select_action(
        self,
        bandit_name: Optional[str] = None,
        context: Optional[np.ndarray] = None
    ) -> Tuple[int, str]:
        """
        Select action using specified bandit.
        
        Args:
            bandit_name: Name of bandit to use (uses default if None)
            context: Context features
            
        Returns:
            Tuple of (action, bandit_name_used)
        """
        if bandit_name is None:
            bandit_name = self._select_bandit_ab()
            
        if bandit_name not in self.bandits:
            raise ValueError(f"Unknown bandit: {bandit_name}")
            
        action = self.bandits[bandit_name].select_action(context)
        return action, bandit_name
        
    def update(
        self,
        bandit_name: str,
        action: int,
        reward: float,
        context: Optional[np.ndarray] = None,
        optimal_reward: Optional[float] = None
    ) -> None:
        """Update specified bandit with reward."""
        if bandit_name not in self.bandits:
            raise ValueError(f"Unknown bandit: {bandit_name}")
            
        self.bandits[bandit_name].update(action, reward, context)
        
        if optimal_reward is not None:
            self.bandits[bandit_name].calculate_regret(optimal_reward)
            
    def setup_ab_test(self, splits: Dict[str, float]) -> None:
        """
        Setup A/B test splits.
        
        Args:
            splits: Dictionary of bandit name to traffic proportion
        """
        total = sum(splits.values())
        if not np.isclose(total, 1.0):
            raise ValueError(f"Split proportions must sum to 1.0, got {total}")
            
        for name in splits:
            if name not in self.bandits:
                raise ValueError(f"Unknown bandit in splits: {name}")
                
        self.ab_test_splits = splits
        logger.info(f"A/B test configured: {splits}")
        
    def _select_bandit_ab(self) -> str:
        """Select bandit based on A/B test split."""
        if not self.ab_test_splits:
            return self.default_bandit
            
        r = np.random.random()
        cumsum = 0.0
        
        for name, proportion in self.ab_test_splits.items():
            cumsum += proportion
            if r < cumsum:
                return name
                
        return self.default_bandit
        
    def get_comparison_metrics(self) -> pd.DataFrame:
        """Get comparison metrics across all bandits."""
        metrics_list = []
        
        for name, bandit in self.bandits.items():
            metrics = bandit.get_metrics()
            metrics['bandit_name'] = name
            metrics_list.append(metrics)
            
        df = pd.DataFrame(metrics_list)
        return df
        
    def get_best_bandit(self, metric: str = 'avg_reward') -> str:
        """
        Identify best performing bandit.
        
        Args:
            metric: Metric to use for comparison
            
        Returns:
            Name of best bandit
        """
        df = self.get_comparison_metrics()
        best_idx = df[metric].idxmax()
        return df.loc[best_idx, 'bandit_name']
        
    def save(self, directory: Union[str, Path]) -> None:
        """Save all bandits to directory."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        # Save each bandit
        for name, bandit in self.bandits.items():
            filepath = directory / f"{name}.pkl"
            bandit.save(filepath)
            
        # Save orchestrator config
        config = {
            'default_bandit': self.default_bandit,
            'ab_test_splits': self.ab_test_splits,
            'bandit_names': list(self.bandits.keys())
        }
        joblib.dump(config, directory / 'orchestrator_config.pkl')
        
        logger.info(f"Orchestrator saved to {directory}")
        
    def load(self, directory: Union[str, Path]) -> 'ContextualBanditOrchestrator':
        """Load all bandits from directory."""
        directory = Path(directory)
        config = joblib.load(directory / 'orchestrator_config.pkl')
        
        self.default_bandit = config['default_bandit']
        self.ab_test_splits = config['ab_test_splits']
        
        # Load each bandit
        for name in config['bandit_names']:
            filepath = directory / f"{name}.pkl"
            # Reconstruct bandit (this is simplified, you'd need to store class info)
            self.bandits[name].load(filepath)
            
        logger.info(f"Orchestrator loaded from {directory}")
        return self


class OnlineBanditTrainer:
    """
    Online trainer for streaming bandit updates.
    
    Handles mini-batch updates and performance monitoring.
    
    Example:
        >>> trainer = OnlineBanditTrainer(bandit, batch_size=32)
        >>> for context, action, reward in stream:
        ...     trainer.add_observation(action, reward, context)
    """
    
    def __init__(
        self,
        bandit: BaseContextualBandit,
        batch_size: int = 1,
        performance_window: int = 100
    ):
        """
        Initialize online trainer.
        
        Args:
            bandit: Bandit to train
            batch_size: Number of observations before update
            performance_window: Window for calculating rolling metrics
        """
        self.bandit = bandit
        self.batch_size = batch_size
        self.performance_window = performance_window
        
        self.batch_actions: List[int] = []
        self.batch_rewards: List[float] = []
        self.batch_contexts: List[np.ndarray] = []
        
    def add_observation(
        self,
        action: int,
        reward: float,
        context: Optional[np.ndarray] = None
    ) -> bool:
        """
        Add observation to batch.
        
        Returns:
            True if batch was processed
        """
        self.batch_actions.append(action)
        self.batch_rewards.append(reward)
        if context is not None:
            self.batch_contexts.append(context)
            
        if len(self.batch_actions) >= self.batch_size:
            self._process_batch()
            return True
            
        return False
        
    def _process_batch(self) -> None:
        """Process accumulated batch."""
        for i in range(len(self.batch_actions)):
            context = self.batch_contexts[i] if self.batch_contexts else None
            self.bandit.update(
                self.batch_actions[i],
                self.batch_rewards[i],
                context
            )
            
        logger.debug(
            f"Processed batch of {len(self.batch_actions)} observations. "
            f"Avg reward: {np.mean(self.batch_rewards):.3f}"
        )
        
        # Clear batch
        self.batch_actions = []
        self.batch_rewards = []
        self.batch_contexts = []
        
    def get_rolling_metrics(self) -> Dict[str, float]:
        """Get rolling performance metrics."""
        if len(self.bandit.rewards_history) < self.performance_window:
            window = self.bandit.rewards_history
        else:
            window = self.bandit.rewards_history[-self.performance_window:]
            
        return {
            'rolling_avg_reward': np.mean(window),
            'rolling_std_reward': np.std(window),
            'rolling_min_reward': np.min(window),
            'rolling_max_reward': np.max(window),
            'window_size': len(window)
        }
