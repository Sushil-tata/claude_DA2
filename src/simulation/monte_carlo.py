"""
Monte Carlo Simulation Engine for the Principal Data Science Decision Agent.

This module provides comprehensive Monte Carlo simulation capabilities for repayment modeling,
scenario analysis, and risk assessment. It supports multiple scenarios with configurable
parameters and provides statistical analysis of simulation results.

Features:
    - Multiple scenario types (baseline, optimistic, pessimistic, stress)
    - Configurable iteration counts (1000, 10000, 100000)
    - Confidence interval calculations (5th, 50th, 95th percentiles)
    - Net Present Value (NPV) calculations
    - Recovery rate simulations
    - Scenario comparison utilities
    - Distribution visualization

Example:
    >>> from src.simulation.monte_carlo import MonteCarloSimulator
    >>> import pandas as pd
    >>> 
    >>> # Initialize simulator
    >>> simulator = MonteCarloSimulator(
    ...     n_simulations=10000,
    ...     random_seed=42
    ... )
    >>> 
    >>> # Run repayment simulation
    >>> results = simulator.simulate_repayments(
    ...     principal=100000,
    ...     default_prob=0.15,
    ...     recovery_rate=0.40,
    ...     scenario='baseline'
    ... )
    >>> 
    >>> # Calculate confidence intervals
    >>> ci = simulator.calculate_confidence_intervals(results['npv'])
    >>> print(f"NPV 95% CI: [{ci['p5']:.2f}, {ci['p95']:.2f}]")
"""

from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import numpy as np
import pandas as pd
from dataclasses import dataclass
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
from pathlib import Path


@dataclass
class SimulationConfig:
    """Configuration for Monte Carlo simulations."""
    n_simulations: int = 10000
    random_seed: Optional[int] = None
    confidence_levels: List[float] = None
    discount_rate: float = 0.10
    time_horizon: int = 36  # months
    pre_default_payment_factor: float = 0.1  # Fraction of principal paid monthly before default
    
    def __post_init__(self):
        if self.confidence_levels is None:
            self.confidence_levels = [0.05, 0.50, 0.95]


@dataclass
class ScenarioParameters:
    """Parameters for different simulation scenarios."""
    name: str
    default_prob: float
    default_prob_std: float
    recovery_rate: float
    recovery_rate_std: float
    payment_rate: float
    payment_rate_std: float
    time_to_default: float  # months
    time_to_default_std: float
    

class MonteCarloSimulator:
    """
    Monte Carlo simulation engine for financial modeling and risk analysis.
    
    This class provides comprehensive Monte Carlo simulation capabilities for modeling
    repayment scenarios, default events, recovery rates, and NPV calculations.
    
    Attributes:
        config (SimulationConfig): Configuration parameters for simulations
        scenarios (Dict[str, ScenarioParameters]): Pre-defined scenario parameters
        rng (np.random.Generator): Random number generator
        
    Example:
        >>> simulator = MonteCarloSimulator(n_simulations=10000, random_seed=42)
        >>> results = simulator.simulate_repayments(
        ...     principal=50000,
        ...     default_prob=0.10,
        ...     recovery_rate=0.45,
        ...     scenario='baseline'
        ... )
        >>> print(f"Expected NPV: ${results['expected_npv']:.2f}")
    """
    
    def __init__(
        self,
        n_simulations: int = 10000,
        random_seed: Optional[int] = None,
        discount_rate: float = 0.10,
        time_horizon: int = 36
    ):
        """
        Initialize Monte Carlo simulator.
        
        Args:
            n_simulations: Number of Monte Carlo iterations
            random_seed: Seed for reproducibility
            discount_rate: Annual discount rate for NPV calculations
            time_horizon: Simulation time horizon in months
        """
        self.config = SimulationConfig(
            n_simulations=n_simulations,
            random_seed=random_seed,
            discount_rate=discount_rate,
            time_horizon=time_horizon
        )
        
        # Initialize random number generator
        self.rng = np.random.default_rng(random_seed)
        
        # Define standard scenarios
        self.scenarios = self._initialize_scenarios()
        
        logger.info(
            f"Initialized MonteCarloSimulator with {n_simulations:,} simulations, "
            f"discount_rate={discount_rate:.2%}, time_horizon={time_horizon} months"
        )
    
    def _initialize_scenarios(self) -> Dict[str, ScenarioParameters]:
        """Initialize pre-defined scenario parameters."""
        return {
            'baseline': ScenarioParameters(
                name='Baseline',
                default_prob=0.15,
                default_prob_std=0.05,
                recovery_rate=0.40,
                recovery_rate_std=0.10,
                payment_rate=0.85,
                payment_rate_std=0.10,
                time_to_default=12.0,
                time_to_default_std=6.0
            ),
            'optimistic': ScenarioParameters(
                name='Optimistic',
                default_prob=0.08,
                default_prob_std=0.03,
                recovery_rate=0.55,
                recovery_rate_std=0.08,
                payment_rate=0.92,
                payment_rate_std=0.05,
                time_to_default=18.0,
                time_to_default_std=8.0
            ),
            'pessimistic': ScenarioParameters(
                name='Pessimistic',
                default_prob=0.25,
                default_prob_std=0.08,
                recovery_rate=0.25,
                recovery_rate_std=0.12,
                payment_rate=0.70,
                payment_rate_std=0.15,
                time_to_default=8.0,
                time_to_default_std=4.0
            ),
            'stress': ScenarioParameters(
                name='Stress',
                default_prob=0.35,
                default_prob_std=0.10,
                recovery_rate=0.15,
                recovery_rate_std=0.15,
                payment_rate=0.55,
                payment_rate_std=0.20,
                time_to_default=6.0,
                time_to_default_std=3.0
            )
        }
    
    def simulate_repayments(
        self,
        principal: float,
        default_prob: Optional[float] = None,
        recovery_rate: Optional[float] = None,
        payment_rate: Optional[float] = None,
        scenario: str = 'baseline',
        custom_params: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Simulate loan repayment scenarios using Monte Carlo.
        
        Args:
            principal: Initial principal amount
            default_prob: Probability of default (overrides scenario)
            recovery_rate: Recovery rate if default occurs (overrides scenario)
            payment_rate: Monthly payment rate (overrides scenario)
            scenario: Scenario name ('baseline', 'optimistic', 'pessimistic', 'stress')
            custom_params: Custom scenario parameters
            
        Returns:
            Dictionary containing simulation results:
                - default_events: Boolean array of default occurrences
                - recovery_amounts: Array of recovery amounts
                - payment_amounts: Array of total payments
                - npv: Array of NPV values
                - expected_npv: Expected NPV
                - confidence_intervals: NPV confidence intervals
                - scenario_params: Parameters used for simulation
        
        Example:
            >>> results = simulator.simulate_repayments(
            ...     principal=100000,
            ...     scenario='pessimistic'
            ... )
            >>> print(f"Default rate: {results['default_events'].mean():.2%}")
        """
        logger.info(
            f"Starting repayment simulation: principal=${principal:,.2f}, "
            f"scenario={scenario}, n_simulations={self.config.n_simulations:,}"
        )
        
        # Get scenario parameters
        if custom_params:
            params = ScenarioParameters(**custom_params)
        else:
            params = self.scenarios.get(scenario)
            if params is None:
                raise ValueError(f"Unknown scenario: {scenario}")
        
        # Override with specific parameters if provided
        if default_prob is not None:
            params.default_prob = default_prob
        if recovery_rate is not None:
            params.recovery_rate = recovery_rate
        if payment_rate is not None:
            params.payment_rate = payment_rate
        
        # Generate random variables
        n = self.config.n_simulations
        
        # Default events (Bernoulli)
        default_probs = self.rng.normal(
            params.default_prob,
            params.default_prob_std,
            n
        )
        default_probs = np.clip(default_probs, 0, 1)
        default_events = self.rng.random(n) < default_probs
        
        # Time to default (log-normal)
        time_to_default = self.rng.lognormal(
            mean=np.log(params.time_to_default),
            sigma=params.time_to_default_std / params.time_to_default,
            size=n
        )
        time_to_default = np.clip(time_to_default, 1, self.config.time_horizon)
        
        # Recovery rates (beta distribution for bounded [0,1])
        recovery_rates = self._generate_beta_random(
            params.recovery_rate,
            params.recovery_rate_std,
            n
        )
        
        # Payment rates (beta distribution)
        payment_rates = self._generate_beta_random(
            params.payment_rate,
            params.payment_rate_std,
            n
        )
        
        # Calculate outcomes
        recovery_amounts = np.zeros(n)
        payment_amounts = np.zeros(n)
        npv_values = np.zeros(n)
        
        for i in range(n):
            if default_events[i]:
                # Default scenario: partial recovery at default time
                recovery_amounts[i] = principal * recovery_rates[i]
                default_month = int(time_to_default[i])
                
                # Some payments before default (reduced payment rate before defaulting)
                pre_default_payments = (
                    principal * payment_rates[i] * 
                    self.config.pre_default_payment_factor * default_month
                )
                payment_amounts[i] = pre_default_payments
                
                # NPV calculation
                npv_values[i] = self._calculate_npv(
                    cash_flows=[pre_default_payments / default_month] * default_month + [recovery_amounts[i]],
                    times=list(range(1, default_month + 1)) + [default_month]
                )
            else:
                # No default: regular payments
                monthly_payment = principal * payment_rates[i] / self.config.time_horizon
                payment_amounts[i] = monthly_payment * self.config.time_horizon
                
                # NPV calculation
                cash_flows = [monthly_payment] * self.config.time_horizon
                times = list(range(1, self.config.time_horizon + 1))
                npv_values[i] = self._calculate_npv(cash_flows, times)
        
        # Calculate statistics
        expected_npv = np.mean(npv_values)
        confidence_intervals = self.calculate_confidence_intervals(npv_values)
        
        results = {
            'default_events': default_events,
            'recovery_amounts': recovery_amounts,
            'payment_amounts': payment_amounts,
            'npv': npv_values,
            'expected_npv': expected_npv,
            'confidence_intervals': confidence_intervals,
            'scenario_params': params,
            'default_rate': default_events.mean(),
            'avg_recovery': recovery_amounts[default_events].mean() if default_events.any() else 0,
            'avg_payment': payment_amounts.mean()
        }
        
        logger.info(
            f"Simulation complete: default_rate={results['default_rate']:.2%}, "
            f"expected_npv=${expected_npv:,.2f}"
        )
        
        return results
    
    def _generate_beta_random(
        self,
        mean: float,
        std: float,
        size: int
    ) -> np.ndarray:
        """
        Generate random samples from beta distribution.
        
        Args:
            mean: Target mean (between 0 and 1)
            std: Target standard deviation
            size: Number of samples
            
        Returns:
            Array of random samples from beta distribution
        """
        mean = np.clip(mean, 0.01, 0.99)
        
        # Ensure std doesn't exceed theoretical maximum
        max_std = np.sqrt(mean * (1 - mean))
        std = min(std, max_std * 0.9)
        
        # Calculate beta distribution parameters
        var = std ** 2
        alpha = mean * (mean * (1 - mean) / var - 1)
        beta = (1 - mean) * (mean * (1 - mean) / var - 1)
        
        # Ensure valid parameters
        alpha = max(alpha, 0.1)
        beta = max(beta, 0.1)
        
        return self.rng.beta(alpha, beta, size)
    
    def _calculate_npv(
        self,
        cash_flows: List[float],
        times: List[int]
    ) -> float:
        """
        Calculate Net Present Value of cash flows.
        
        Args:
            cash_flows: List of cash flow amounts
            times: List of time periods (months)
            
        Returns:
            Net present value
        """
        monthly_rate = (1 + self.config.discount_rate) ** (1/12) - 1
        discount_factors = [(1 + monthly_rate) ** (-t) for t in times]
        return sum(cf * df for cf, df in zip(cash_flows, discount_factors))
    
    def calculate_confidence_intervals(
        self,
        values: np.ndarray,
        confidence_levels: Optional[List[float]] = None
    ) -> Dict[str, float]:
        """
        Calculate confidence intervals for simulation results.
        
        Args:
            values: Array of simulated values
            confidence_levels: List of confidence levels (e.g., [0.05, 0.50, 0.95])
            
        Returns:
            Dictionary with percentile values
            
        Example:
            >>> ci = simulator.calculate_confidence_intervals(npv_values)
            >>> print(f"Median NPV: ${ci['p50']:,.2f}")
        """
        if confidence_levels is None:
            confidence_levels = self.config.confidence_levels
        
        percentiles = {
            f'p{int(level*100)}': np.percentile(values, level * 100)
            for level in confidence_levels
        }
        
        # Add mean and std
        percentiles['mean'] = np.mean(values)
        percentiles['std'] = np.std(values)
        
        return percentiles
    
    def simulate_portfolio(
        self,
        portfolio_df: pd.DataFrame,
        principal_col: str = 'principal',
        default_prob_col: str = 'default_prob',
        scenario: str = 'baseline',
        aggregate: bool = True
    ) -> Union[pd.DataFrame, Dict[str, Any]]:
        """
        Simulate entire portfolio of loans.
        
        Args:
            portfolio_df: DataFrame with loan information
            principal_col: Column name for principal amounts
            default_prob_col: Column name for default probabilities
            scenario: Scenario to use for simulation
            aggregate: If True, return aggregated results; else return loan-level
            
        Returns:
            Aggregated statistics or loan-level simulation results
            
        Example:
            >>> portfolio = pd.DataFrame({
            ...     'loan_id': [1, 2, 3],
            ...     'principal': [10000, 20000, 15000],
            ...     'default_prob': [0.10, 0.15, 0.12]
            ... })
            >>> results = simulator.simulate_portfolio(portfolio)
            >>> print(f"Portfolio NPV: ${results['total_npv']:,.2f}")
        """
        logger.info(f"Simulating portfolio with {len(portfolio_df)} loans")
        
        loan_results = []
        
        for idx, row in portfolio_df.iterrows():
            result = self.simulate_repayments(
                principal=row[principal_col],
                default_prob=row.get(default_prob_col),
                scenario=scenario
            )
            
            loan_results.append({
                'loan_index': idx,
                'expected_npv': result['expected_npv'],
                'default_rate': result['default_rate'],
                'npv_p5': result['confidence_intervals']['p5'],
                'npv_p50': result['confidence_intervals']['p50'],
                'npv_p95': result['confidence_intervals']['p95']
            })
        
        results_df = pd.DataFrame(loan_results)
        
        if aggregate:
            return {
                'total_npv': results_df['expected_npv'].sum(),
                'avg_default_rate': results_df['default_rate'].mean(),
                'total_npv_p5': results_df['npv_p5'].sum(),
                'total_npv_p50': results_df['npv_p50'].sum(),
                'total_npv_p95': results_df['npv_p95'].sum(),
                'n_loans': len(portfolio_df),
                'loan_level_results': results_df
            }
        else:
            return results_df
    
    def compare_scenarios(
        self,
        principal: float,
        scenarios: List[str] = None
    ) -> pd.DataFrame:
        """
        Compare multiple scenarios side-by-side.
        
        Args:
            principal: Principal amount
            scenarios: List of scenario names to compare
            
        Returns:
            DataFrame with comparison metrics
            
        Example:
            >>> comparison = simulator.compare_scenarios(
            ...     principal=100000,
            ...     scenarios=['baseline', 'optimistic', 'pessimistic']
            ... )
            >>> print(comparison)
        """
        if scenarios is None:
            scenarios = ['baseline', 'optimistic', 'pessimistic', 'stress']
        
        logger.info(f"Comparing scenarios: {scenarios}")
        
        results = []
        
        for scenario in scenarios:
            sim_result = self.simulate_repayments(
                principal=principal,
                scenario=scenario
            )
            
            results.append({
                'scenario': scenario,
                'expected_npv': sim_result['expected_npv'],
                'npv_std': sim_result['confidence_intervals']['std'],
                'npv_p5': sim_result['confidence_intervals']['p5'],
                'npv_p50': sim_result['confidence_intervals']['p50'],
                'npv_p95': sim_result['confidence_intervals']['p95'],
                'default_rate': sim_result['default_rate'],
                'avg_recovery': sim_result['avg_recovery']
            })
        
        return pd.DataFrame(results)
    
    def plot_distribution(
        self,
        values: np.ndarray,
        title: str = "Simulation Distribution",
        xlabel: str = "Value",
        show_confidence: bool = True,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot distribution of simulation results.
        
        Args:
            values: Array of simulated values
            title: Plot title
            xlabel: X-axis label
            show_confidence: Whether to show confidence interval lines
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
            
        Example:
            >>> results = simulator.simulate_repayments(principal=100000)
            >>> fig = simulator.plot_distribution(
            ...     results['npv'],
            ...     title="NPV Distribution",
            ...     xlabel="Net Present Value ($)"
            ... )
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot histogram and KDE
        ax.hist(values, bins=50, alpha=0.6, density=True, label='Histogram')
        
        # Fit and plot KDE
        kde = stats.gaussian_kde(values)
        x_range = np.linspace(values.min(), values.max(), 200)
        ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
        
        # Add confidence intervals
        if show_confidence:
            ci = self.calculate_confidence_intervals(values)
            for level, color in [('p5', 'orange'), ('p50', 'green'), ('p95', 'orange')]:
                ax.axvline(ci[level], color=color, linestyle='--', 
                          label=f'{level.upper()}: {ci[level]:,.2f}')
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Density')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")
        
        return fig
    
    def sensitivity_analysis(
        self,
        principal: float,
        param_name: str,
        param_range: np.ndarray,
        scenario: str = 'baseline'
    ) -> pd.DataFrame:
        """
        Perform sensitivity analysis on a parameter.
        
        Args:
            principal: Principal amount
            param_name: Parameter to vary ('default_prob', 'recovery_rate', etc.)
            param_range: Array of parameter values to test
            scenario: Base scenario
            
        Returns:
            DataFrame with sensitivity results
            
        Example:
            >>> sensitivity = simulator.sensitivity_analysis(
            ...     principal=100000,
            ...     param_name='default_prob',
            ...     param_range=np.linspace(0.05, 0.30, 10)
            ... )
        """
        logger.info(f"Running sensitivity analysis on {param_name}")
        
        results = []
        
        for param_value in param_range:
            kwargs = {param_name: param_value}
            sim_result = self.simulate_repayments(
                principal=principal,
                scenario=scenario,
                **kwargs
            )
            
            results.append({
                param_name: param_value,
                'expected_npv': sim_result['expected_npv'],
                'default_rate': sim_result['default_rate'],
                'npv_std': sim_result['confidence_intervals']['std']
            })
        
        return pd.DataFrame(results)
    
    def save_results(
        self,
        results: Dict[str, Any],
        filepath: str
    ) -> None:
        """
        Save simulation results to file.
        
        Args:
            results: Simulation results dictionary
            filepath: Path to save file (.json or .npz)
        """
        filepath = Path(filepath)
        
        if filepath.suffix == '.json':
            # Convert numpy arrays to lists for JSON serialization
            json_results = {}
            for key, value in results.items():
                if isinstance(value, np.ndarray):
                    json_results[key] = value.tolist()
                elif isinstance(value, ScenarioParameters):
                    json_results[key] = value.__dict__
                else:
                    json_results[key] = value
            
            with open(filepath, 'w') as f:
                json.dump(json_results, f, indent=2)
        
        elif filepath.suffix == '.npz':
            # Save as numpy archive
            arrays = {k: v for k, v in results.items() if isinstance(v, np.ndarray)}
            np.savez(filepath, **arrays)
        
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        logger.info(f"Saved simulation results to {filepath}")
    
    def load_results(self, filepath: str) -> Dict[str, Any]:
        """
        Load simulation results from file.
        
        Args:
            filepath: Path to results file
            
        Returns:
            Dictionary of simulation results
        """
        filepath = Path(filepath)
        
        if filepath.suffix == '.json':
            with open(filepath, 'r') as f:
                results = json.load(f)
            
            # Convert lists back to numpy arrays
            for key, value in results.items():
                if isinstance(value, list):
                    results[key] = np.array(value)
        
        elif filepath.suffix == '.npz':
            loaded = np.load(filepath)
            results = {key: loaded[key] for key in loaded.files}
        
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        logger.info(f"Loaded simulation results from {filepath}")
        return results


if __name__ == "__main__":
    # Example usage
    logger.info("Running Monte Carlo simulation example")
    
    # Initialize simulator
    simulator = MonteCarloSimulator(n_simulations=10000, random_seed=42)
    
    # Run single loan simulation
    results = simulator.simulate_repayments(
        principal=100000,
        scenario='baseline'
    )
    
    print(f"\nSimulation Results:")
    print(f"Expected NPV: ${results['expected_npv']:,.2f}")
    print(f"Default Rate: {results['default_rate']:.2%}")
    print(f"95% Confidence Interval: [${results['confidence_intervals']['p5']:,.2f}, "
          f"${results['confidence_intervals']['p95']:,.2f}]")
    
    # Compare scenarios
    comparison = simulator.compare_scenarios(principal=100000)
    print("\nScenario Comparison:")
    print(comparison.to_string())
