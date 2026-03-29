"""
Markov Chain Modeling for the Principal Data Science Decision Agent.

This module provides comprehensive Markov chain modeling capabilities for state transitions,
roll rate analysis, and delinquency progression modeling. It's particularly useful for
modeling customer transitions through various states in collections and credit risk.

Features:
    - State transition matrix estimation
    - Roll rate modeling (DPD 0→30→60→90→120→charge-off)
    - Steady-state distribution calculation
    - Multi-step transition probabilities
    - Absorbing state analysis
    - Time-to-event estimation
    - Segment-specific transition matrices
    - Migration analysis utilities

Example:
    >>> from src.simulation.markov_chains import MarkovChainModel
    >>> import pandas as pd
    >>> 
    >>> # Initialize model
    >>> model = MarkovChainModel(states=['Current', 'DPD30', 'DPD60', 'DPD90', 'ChargeOff'])
    >>> 
    >>> # Estimate transition matrix from data
    >>> model.fit(historical_data, state_col='dpd_status', id_col='account_id')
    >>> 
    >>> # Predict future state distribution
    >>> future_dist = model.predict_distribution(initial_state='DPD30', n_steps=6)
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
from dataclasses import dataclass
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import linalg
from collections import defaultdict
import json
from pathlib import Path


@dataclass
class MarkovConfig:
    """Configuration for Markov chain models."""
    states: List[str]
    absorbing_states: Optional[List[str]] = None
    time_unit: str = 'month'
    smoothing: float = 0.01  # Laplace smoothing
    min_observations: int = 10
    
    def __post_init__(self):
        if self.absorbing_states is None:
            self.absorbing_states = []


class MarkovChainModel:
    """
    Markov chain model for state transition analysis.
    
    This class provides comprehensive Markov chain modeling capabilities including
    transition matrix estimation, steady-state calculation, and multi-step predictions.
    Particularly useful for delinquency roll rate modeling and customer state transitions.
    
    Attributes:
        config (MarkovConfig): Model configuration
        transition_matrix (np.ndarray): State transition probability matrix
        state_to_idx (Dict[str, int]): Mapping from state names to indices
        idx_to_state (Dict[int, str]): Mapping from indices to state names
        
    Example:
        >>> model = MarkovChainModel(
        ...     states=['Current', 'DPD30', 'DPD60', 'DPD90', 'ChargeOff'],
        ...     absorbing_states=['ChargeOff']
        ... )
        >>> model.fit(data, state_col='status', id_col='account_id')
        >>> prob = model.transition_probability('DPD30', 'ChargeOff', n_steps=3)
    """
    
    def __init__(
        self,
        states: List[str],
        absorbing_states: Optional[List[str]] = None,
        time_unit: str = 'month',
        smoothing: float = 0.01
    ):
        """
        Initialize Markov chain model.
        
        Args:
            states: List of state names
            absorbing_states: List of absorbing state names
            time_unit: Time unit for transitions ('month', 'day', 'week')
            smoothing: Laplace smoothing parameter
        """
        self.config = MarkovConfig(
            states=states,
            absorbing_states=absorbing_states or [],
            time_unit=time_unit,
            smoothing=smoothing
        )
        
        # State mappings
        self.state_to_idx = {state: idx for idx, state in enumerate(states)}
        self.idx_to_state = {idx: state for idx, state in enumerate(states)}
        self.n_states = len(states)
        
        # Initialize empty transition matrix
        self.transition_matrix: Optional[np.ndarray] = None
        self.transition_counts: Optional[np.ndarray] = None
        self.steady_state: Optional[np.ndarray] = None
        
        logger.info(
            f"Initialized MarkovChainModel with {self.n_states} states, "
            f"absorbing_states={absorbing_states}, time_unit={time_unit}"
        )
    
    def fit(
        self,
        data: pd.DataFrame,
        state_col: str,
        id_col: str,
        time_col: Optional[str] = None,
        segment_col: Optional[str] = None
    ) -> 'MarkovChainModel':
        """
        Estimate transition matrix from historical data.
        
        Args:
            data: DataFrame with state transition history
            state_col: Column name for state values
            id_col: Column name for entity ID
            time_col: Column name for time/sequence ordering
            segment_col: Optional column for segment-specific matrices
            
        Returns:
            Self for method chaining
            
        Example:
            >>> data = pd.DataFrame({
            ...     'account_id': [1, 1, 1, 2, 2],
            ...     'month': [1, 2, 3, 1, 2],
            ...     'status': ['Current', 'DPD30', 'Current', 'Current', 'DPD30']
            ... })
            >>> model.fit(data, state_col='status', id_col='account_id', time_col='month')
        """
        logger.info(f"Estimating transition matrix from {len(data)} observations")
        
        # Sort by ID and time
        if time_col:
            data = data.sort_values([id_col, time_col])
        else:
            data = data.sort_values(id_col)
        
        # Initialize transition count matrix
        self.transition_counts = np.zeros((self.n_states, self.n_states))
        
        # Count transitions
        for entity_id, group in data.groupby(id_col):
            states = group[state_col].values
            
            # Count consecutive state transitions
            for i in range(len(states) - 1):
                from_state = states[i]
                to_state = states[i + 1]
                
                if from_state in self.state_to_idx and to_state in self.state_to_idx:
                    from_idx = self.state_to_idx[from_state]
                    to_idx = self.state_to_idx[to_state]
                    self.transition_counts[from_idx, to_idx] += 1
        
        # Apply smoothing and normalize
        smoothed_counts = self.transition_counts + self.config.smoothing
        row_sums = smoothed_counts.sum(axis=1, keepdims=True)
        self.transition_matrix = smoothed_counts / row_sums
        
        # Enforce absorbing states (probability 1.0 to stay)
        for state in self.config.absorbing_states:
            if state in self.state_to_idx:
                idx = self.state_to_idx[state]
                self.transition_matrix[idx, :] = 0
                self.transition_matrix[idx, idx] = 1.0
        
        logger.info("Transition matrix estimation complete")
        self._log_transition_summary()
        
        return self
    
    def _log_transition_summary(self) -> None:
        """Log summary of transition matrix."""
        if self.transition_matrix is None:
            return
        
        # Find most common transitions
        top_transitions = []
        for i in range(self.n_states):
            for j in range(self.n_states):
                if i != j:  # Exclude self-transitions
                    top_transitions.append({
                        'from': self.idx_to_state[i],
                        'to': self.idx_to_state[j],
                        'probability': self.transition_matrix[i, j]
                    })
        
        top_transitions.sort(key=lambda x: x['probability'], reverse=True)
        
        logger.info("Top 5 state transitions:")
        for trans in top_transitions[:5]:
            logger.info(
                f"  {trans['from']} → {trans['to']}: {trans['probability']:.3f}"
            )
    
    def fit_roll_rates(
        self,
        data: pd.DataFrame,
        dpd_col: str,
        id_col: str,
        time_col: str,
        dpd_buckets: Optional[List[int]] = None
    ) -> 'MarkovChainModel':
        """
        Fit model using delinquency roll rates.
        
        Args:
            data: DataFrame with DPD history
            dpd_col: Column name for days past due
            id_col: Column name for account ID
            time_col: Column name for time
            dpd_buckets: DPD bucket boundaries (default: [0, 30, 60, 90, 120])
            
        Returns:
            Self for method chaining
            
        Example:
            >>> model.fit_roll_rates(
            ...     data,
            ...     dpd_col='days_past_due',
            ...     id_col='account_id',
            ...     time_col='month'
            ... )
        """
        if dpd_buckets is None:
            dpd_buckets = [0, 30, 60, 90, 120]
        
        # Create DPD status column
        data = data.copy()
        data['dpd_status'] = pd.cut(
            data[dpd_col],
            bins=[-np.inf] + dpd_buckets + [np.inf],
            labels=[f'DPD{b}' for b in dpd_buckets] + ['ChargeOff']
        )
        
        # Update states if needed
        unique_states = data['dpd_status'].unique()
        if set(unique_states) != set(self.config.states):
            logger.warning("Updating states based on observed DPD statuses")
            self.config.states = sorted(unique_states)
            self.state_to_idx = {state: idx for idx, state in enumerate(self.config.states)}
            self.idx_to_state = {idx: state for idx, state in enumerate(self.config.states)}
            self.n_states = len(self.config.states)
        
        # Fit model
        return self.fit(
            data,
            state_col='dpd_status',
            id_col=id_col,
            time_col=time_col
        )
    
    def predict_distribution(
        self,
        initial_state: Union[str, np.ndarray],
        n_steps: int = 1
    ) -> np.ndarray:
        """
        Predict state distribution after n steps.
        
        Args:
            initial_state: Starting state name or distribution vector
            n_steps: Number of time steps
            
        Returns:
            Probability distribution over states
            
        Example:
            >>> dist = model.predict_distribution('DPD30', n_steps=6)
            >>> print(f"Prob of ChargeOff: {dist[model.state_to_idx['ChargeOff']]:.2%}")
        """
        if self.transition_matrix is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Convert initial state to distribution
        if isinstance(initial_state, str):
            initial_dist = np.zeros(self.n_states)
            initial_dist[self.state_to_idx[initial_state]] = 1.0
        else:
            initial_dist = initial_state
        
        # Apply transition matrix n times
        current_dist = initial_dist
        for _ in range(n_steps):
            current_dist = current_dist @ self.transition_matrix
        
        return current_dist
    
    def transition_probability(
        self,
        from_state: str,
        to_state: str,
        n_steps: int = 1
    ) -> float:
        """
        Calculate probability of transition from one state to another in n steps.
        
        Args:
            from_state: Starting state
            to_state: Target state
            n_steps: Number of steps
            
        Returns:
            Transition probability
            
        Example:
            >>> prob = model.transition_probability('Current', 'ChargeOff', n_steps=12)
        """
        dist = self.predict_distribution(from_state, n_steps)
        return dist[self.state_to_idx[to_state]]
    
    def calculate_steady_state(self) -> np.ndarray:
        """
        Calculate steady-state distribution.
        
        Returns:
            Steady-state probability distribution
            
        Example:
            >>> steady = model.calculate_steady_state()
            >>> for state, prob in zip(model.config.states, steady):
            ...     print(f"{state}: {prob:.2%}")
        """
        if self.transition_matrix is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Find eigenvalue 1 (steady state)
        eigenvalues, eigenvectors = linalg.eig(self.transition_matrix.T)
        
        # Find index of eigenvalue closest to 1
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        
        # Get corresponding eigenvector and normalize
        steady_state = np.real(eigenvectors[:, idx])
        steady_state = steady_state / steady_state.sum()
        
        self.steady_state = steady_state
        
        logger.info("Calculated steady-state distribution:")
        for state, prob in zip(self.config.states, steady_state):
            logger.info(f"  {state}: {prob:.3f}")
        
        return steady_state
    
    def time_to_absorption(
        self,
        from_state: str,
        absorbing_state: str,
        max_steps: int = 100
    ) -> Dict[str, float]:
        """
        Calculate expected time to reach absorbing state.
        
        Args:
            from_state: Starting state
            absorbing_state: Target absorbing state
            max_steps: Maximum number of steps to consider
            
        Returns:
            Dictionary with mean, median, and distribution of time to absorption
            
        Example:
            >>> tta = model.time_to_absorption('DPD30', 'ChargeOff')
            >>> print(f"Expected time to charge-off: {tta['mean']:.1f} months")
        """
        if absorbing_state not in self.config.absorbing_states:
            logger.warning(f"{absorbing_state} is not marked as absorbing state")
        
        # Calculate probability of absorption at each time step
        absorption_probs = []
        cumulative_prob = 0
        
        for step in range(1, max_steps + 1):
            dist = self.predict_distribution(from_state, n_steps=step)
            prob_absorbed = dist[self.state_to_idx[absorbing_state]]
            
            # Probability of absorption at this step (not before)
            prob_at_step = prob_absorbed - cumulative_prob
            absorption_probs.append(prob_at_step)
            cumulative_prob = prob_absorbed
            
            if cumulative_prob > 0.999:
                break
        
        absorption_probs = np.array(absorption_probs)
        time_steps = np.arange(1, len(absorption_probs) + 1)
        
        # Calculate statistics
        mean_time = np.sum(time_steps * absorption_probs)
        
        # Median (50th percentile)
        cumsum = np.cumsum(absorption_probs)
        median_idx = np.searchsorted(cumsum, 0.5)
        median_time = time_steps[min(median_idx, len(time_steps) - 1)]
        
        return {
            'mean': mean_time,
            'median': median_time,
            'distribution': absorption_probs,
            'time_steps': time_steps,
            'total_absorption_prob': cumulative_prob
        }
    
    def analyze_absorbing_states(self) -> pd.DataFrame:
        """
        Analyze absorption probabilities from all transient states.
        
        Returns:
            DataFrame with absorption probabilities
            
        Example:
            >>> absorption = model.analyze_absorbing_states()
            >>> print(absorption)
        """
        if not self.config.absorbing_states:
            raise ValueError("No absorbing states defined")
        
        results = []
        
        transient_states = [
            s for s in self.config.states 
            if s not in self.config.absorbing_states
        ]
        
        for from_state in transient_states:
            row = {'from_state': from_state}
            
            for abs_state in self.config.absorbing_states:
                # Calculate long-term absorption probability
                prob = self.transition_probability(
                    from_state,
                    abs_state,
                    n_steps=100
                )
                row[f'prob_{abs_state}'] = prob
                
                # Calculate expected time to absorption
                tta = self.time_to_absorption(from_state, abs_state)
                row[f'time_to_{abs_state}'] = tta['mean']
            
            results.append(row)
        
        return pd.DataFrame(results)
    
    def segment_specific_matrices(
        self,
        data: pd.DataFrame,
        state_col: str,
        id_col: str,
        segment_col: str,
        time_col: Optional[str] = None
    ) -> Dict[str, np.ndarray]:
        """
        Estimate separate transition matrices for each segment.
        
        Args:
            data: DataFrame with state transitions
            state_col: Column name for states
            id_col: Column name for entity ID
            segment_col: Column name for segments
            time_col: Optional time column
            
        Returns:
            Dictionary mapping segment names to transition matrices
            
        Example:
            >>> matrices = model.segment_specific_matrices(
            ...     data,
            ...     state_col='status',
            ...     id_col='account_id',
            ...     segment_col='risk_tier'
            ... )
        """
        segment_matrices = {}
        
        for segment, segment_data in data.groupby(segment_col):
            logger.info(f"Estimating transition matrix for segment: {segment}")
            
            # Create temporary model for this segment
            segment_model = MarkovChainModel(
                states=self.config.states,
                absorbing_states=self.config.absorbing_states,
                time_unit=self.config.time_unit,
                smoothing=self.config.smoothing
            )
            
            segment_model.fit(
                segment_data,
                state_col=state_col,
                id_col=id_col,
                time_col=time_col
            )
            
            segment_matrices[segment] = segment_model.transition_matrix
        
        return segment_matrices
    
    def migration_analysis(
        self,
        data: pd.DataFrame,
        state_col: str,
        id_col: str,
        time_col: str,
        start_period: Any,
        end_period: Any
    ) -> pd.DataFrame:
        """
        Analyze state migrations between two time periods.
        
        Args:
            data: DataFrame with state history
            state_col: Column name for states
            id_col: Column name for entity ID
            time_col: Column name for time
            start_period: Starting time period
            end_period: Ending time period
            
        Returns:
            Migration matrix as DataFrame
            
        Example:
            >>> migration = model.migration_analysis(
            ...     data,
            ...     state_col='status',
            ...     id_col='account_id',
            ...     time_col='month',
            ...     start_period='2023-01',
            ...     end_period='2023-12'
            ... )
        """
        # Get states at start and end periods
        start_data = data[data[time_col] == start_period][[id_col, state_col]]
        end_data = data[data[time_col] == end_period][[id_col, state_col]]
        
        # Merge to get transitions
        migrations = start_data.merge(
            end_data,
            on=id_col,
            suffixes=('_start', '_end')
        )
        
        # Create migration matrix
        migration_matrix = pd.crosstab(
            migrations[f'{state_col}_start'],
            migrations[f'{state_col}_end'],
            normalize='index'
        )
        
        logger.info(f"Migration analysis: {start_period} → {end_period}")
        logger.info(f"Total entities tracked: {len(migrations)}")
        
        return migration_matrix
    
    def plot_transition_diagram(
        self,
        threshold: float = 0.05,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot state transition diagram.
        
        Args:
            threshold: Minimum probability to show transition
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        if self.transition_matrix is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot heatmap
        sns.heatmap(
            self.transition_matrix,
            annot=True,
            fmt='.3f',
            cmap='YlOrRd',
            xticklabels=self.config.states,
            yticklabels=self.config.states,
            ax=ax,
            cbar_kws={'label': 'Transition Probability'}
        )
        
        ax.set_title('State Transition Matrix', fontsize=14, fontweight='bold')
        ax.set_xlabel('To State', fontsize=12)
        ax.set_ylabel('From State', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved transition diagram to {save_path}")
        
        return fig
    
    def plot_evolution(
        self,
        initial_state: Union[str, np.ndarray],
        n_steps: int = 12,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot evolution of state distribution over time.
        
        Args:
            initial_state: Starting state or distribution
            n_steps: Number of steps to simulate
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        if self.transition_matrix is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Simulate distribution evolution
        distributions = []
        for step in range(n_steps + 1):
            dist = self.predict_distribution(initial_state, n_steps=step)
            distributions.append(dist)
        
        distributions = np.array(distributions)
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(n_steps + 1)
        
        # Stack area plot
        ax.stackplot(
            x,
            *distributions.T,
            labels=self.config.states,
            alpha=0.7
        )
        
        ax.set_xlabel(f'Time Steps ({self.config.time_unit}s)', fontsize=12)
        ax.set_ylabel('Probability', fontsize=12)
        ax.set_title(f'State Distribution Evolution from {initial_state}', fontsize=14)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved evolution plot to {save_path}")
        
        return fig
    
    def save_model(self, filepath: str) -> None:
        """
        Save model to file.
        
        Args:
            filepath: Path to save model
        """
        filepath = Path(filepath)
        
        model_data = {
            'config': {
                'states': self.config.states,
                'absorbing_states': self.config.absorbing_states,
                'time_unit': self.config.time_unit,
                'smoothing': self.config.smoothing
            },
            'transition_matrix': self.transition_matrix.tolist() if self.transition_matrix is not None else None,
            'transition_counts': self.transition_counts.tolist() if self.transition_counts is not None else None
        }
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        logger.info(f"Saved model to {filepath}")
    
    def load_model(self, filepath: str) -> 'MarkovChainModel':
        """
        Load model from file.
        
        Args:
            filepath: Path to model file
            
        Returns:
            Self for method chaining
        """
        with open(filepath, 'r') as f:
            model_data = json.load(f)
        
        # Restore configuration
        config = model_data['config']
        self.config = MarkovConfig(**config)
        self.state_to_idx = {state: idx for idx, state in enumerate(config['states'])}
        self.idx_to_state = {idx: state for idx, state in enumerate(config['states'])}
        self.n_states = len(config['states'])
        
        # Restore matrices
        if model_data['transition_matrix']:
            self.transition_matrix = np.array(model_data['transition_matrix'])
        if model_data['transition_counts']:
            self.transition_counts = np.array(model_data['transition_counts'])
        
        logger.info(f"Loaded model from {filepath}")
        
        return self


if __name__ == "__main__":
    # Example usage
    logger.info("Running Markov chain example")
    
    # Create sample data
    np.random.seed(42)
    n_accounts = 1000
    n_months = 12
    
    data = []
    for account_id in range(n_accounts):
        current_state = 'Current'
        for month in range(n_months):
            data.append({
                'account_id': account_id,
                'month': month,
                'status': current_state
            })
            
            # Simple transition logic
            if current_state == 'Current':
                current_state = np.random.choice(
                    ['Current', 'DPD30'],
                    p=[0.85, 0.15]
                )
            elif current_state == 'DPD30':
                current_state = np.random.choice(
                    ['Current', 'DPD30', 'DPD60'],
                    p=[0.40, 0.35, 0.25]
                )
            elif current_state == 'DPD60':
                current_state = np.random.choice(
                    ['DPD30', 'DPD60', 'DPD90'],
                    p=[0.20, 0.40, 0.40]
                )
            elif current_state == 'DPD90':
                current_state = np.random.choice(
                    ['DPD60', 'DPD90', 'ChargeOff'],
                    p=[0.10, 0.40, 0.50]
                )
            elif current_state == 'ChargeOff':
                pass  # Absorbing state
    
    df = pd.DataFrame(data)
    
    # Initialize and fit model
    model = MarkovChainModel(
        states=['Current', 'DPD30', 'DPD60', 'DPD90', 'ChargeOff'],
        absorbing_states=['ChargeOff']
    )
    
    model.fit(df, state_col='status', id_col='account_id', time_col='month')
    
    print("\nTransition Matrix:")
    print(pd.DataFrame(
        model.transition_matrix,
        index=model.config.states,
        columns=model.config.states
    ))
    
    # Predict distribution
    dist = model.predict_distribution('DPD30', n_steps=6)
    print(f"\nDistribution after 6 months from DPD30:")
    for state, prob in zip(model.config.states, dist):
        print(f"  {state}: {prob:.2%}")
    
    # Time to absorption
    tta = model.time_to_absorption('DPD30', 'ChargeOff')
    print(f"\nExpected time to charge-off from DPD30: {tta['mean']:.1f} months")
