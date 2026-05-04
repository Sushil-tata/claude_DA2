"""
Policy Impact Simulation for the Principal Data Science Decision Agent.

This module provides comprehensive policy simulation capabilities for A/B testing,
treatment effect estimation, and policy change impact analysis. It's designed for
evaluating collections strategies, channel optimization, and decision rule changes.

Features:
    - Treatment effect simulation
    - Population-level impact estimation
    - Channel optimization (email, SMS, call, letter)
    - Capacity constraint modeling
    - Cost-benefit analysis
    - Uplift measurement
    - Scenario comparison
    - Sensitivity analysis

Example:
    >>> from src.simulation.policy_simulator import PolicySimulator
    >>> 
    >>> simulator = PolicySimulator()
    >>> 
    >>> # Simulate policy change impact
    >>> results = simulator.simulate_policy_impact(
    ...     population_df=customers,
    ...     baseline_policy='reactive',
    ...     new_policy='proactive',
    ...     channels=['email', 'sms', 'call']
    ... )
"""

from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import json


@dataclass
class ChannelConfig:
    """Configuration for communication channels."""
    name: str
    cost: float
    response_rate: float
    capacity_per_day: Optional[int] = None
    success_rate: float = 0.5
    contact_rate: float = 0.8
    

@dataclass
class PolicyConfig:
    """Configuration for policy rules."""
    name: str
    target_segments: List[str]
    channels: List[str]
    contact_frequency: int  # days between contacts
    max_contacts: int
    prioritization_rule: str = 'risk_score'
    treatment_effect: float = 0.15  # expected uplift
    

class PolicySimulator:
    """
    Policy impact simulation engine for A/B testing and strategy evaluation.
    
    This class provides comprehensive capabilities for simulating the impact of
    policy changes, treatment strategies, and channel optimizations on business
    outcomes such as recovery rates, contact costs, and overall ROI.
    
    Attributes:
        channels (Dict[str, ChannelConfig]): Available communication channels
        policies (Dict[str, PolicyConfig]): Defined policies
        rng (np.random.Generator): Random number generator
        
    Example:
        >>> simulator = PolicySimulator(random_seed=42)
        >>> results = simulator.simulate_ab_test(
        ...     population=customers,
        ...     control_policy='current',
        ...     treatment_policy='enhanced',
        ...     test_duration=90
        ... )
        >>> print(f"Uplift: {results['uplift']:.2%}")
    """
    
    def __init__(
        self,
        random_seed: Optional[int] = None
    ):
        """
        Initialize policy simulator.
        
        Args:
            random_seed: Seed for reproducibility
        """
        self.rng = np.random.default_rng(random_seed)
        
        # Initialize default channels
        self.channels = self._initialize_channels()
        
        # Initialize default policies
        self.policies = self._initialize_policies()
        
        logger.info(
            f"Initialized PolicySimulator with {len(self.channels)} channels "
            f"and {len(self.policies)} default policies"
        )
    
    def _initialize_channels(self) -> Dict[str, ChannelConfig]:
        """Initialize default communication channels."""
        return {
            'email': ChannelConfig(
                name='email',
                cost=0.10,
                response_rate=0.15,
                capacity_per_day=100000,
                success_rate=0.45,
                contact_rate=0.90
            ),
            'sms': ChannelConfig(
                name='sms',
                cost=0.25,
                response_rate=0.25,
                capacity_per_day=50000,
                success_rate=0.55,
                contact_rate=0.95
            ),
            'call': ChannelConfig(
                name='call',
                cost=5.00,
                response_rate=0.55,
                capacity_per_day=1000,
                success_rate=0.70,
                contact_rate=0.60
            ),
            'letter': ChannelConfig(
                name='letter',
                cost=1.50,
                response_rate=0.20,
                capacity_per_day=10000,
                success_rate=0.50,
                contact_rate=0.95
            ),
            'ivr': ChannelConfig(
                name='ivr',
                cost=0.50,
                response_rate=0.30,
                capacity_per_day=20000,
                success_rate=0.48,
                contact_rate=0.70
            )
        }
    
    def _initialize_policies(self) -> Dict[str, PolicyConfig]:
        """Initialize default policies."""
        return {
            'reactive': PolicyConfig(
                name='reactive',
                target_segments=['all'],
                channels=['letter', 'call'],
                contact_frequency=30,
                max_contacts=3,
                treatment_effect=0.10
            ),
            'proactive': PolicyConfig(
                name='proactive',
                target_segments=['high_risk'],
                channels=['email', 'sms', 'call'],
                contact_frequency=14,
                max_contacts=6,
                treatment_effect=0.20
            ),
            'aggressive': PolicyConfig(
                name='aggressive',
                target_segments=['high_value', 'high_risk'],
                channels=['call', 'sms', 'email'],
                contact_frequency=7,
                max_contacts=10,
                treatment_effect=0.25
            ),
            'minimal': PolicyConfig(
                name='minimal',
                target_segments=['low_risk'],
                channels=['email'],
                contact_frequency=60,
                max_contacts=2,
                treatment_effect=0.05
            )
        }
    
    def simulate_policy_impact(
        self,
        population_df: pd.DataFrame,
        baseline_policy: str,
        new_policy: str,
        outcome_col: str = 'will_pay',
        baseline_rate: Optional[float] = None,
        time_horizon: int = 90
    ) -> Dict[str, Any]:
        """
        Simulate impact of policy change on population.
        
        Args:
            population_df: DataFrame with customer/account information
            baseline_policy: Name of baseline policy
            new_policy: Name of new policy
            outcome_col: Column name for outcome (if available)
            baseline_rate: Baseline success rate (if outcome_col not available)
            time_horizon: Simulation time horizon in days
            
        Returns:
            Dictionary with simulation results including uplift and costs
            
        Example:
            >>> results = simulator.simulate_policy_impact(
            ...     population_df=accounts,
            ...     baseline_policy='reactive',
            ...     new_policy='proactive'
            ... )
        """
        logger.info(
            f"Simulating policy impact: {baseline_policy} → {new_policy}, "
            f"population={len(population_df):,}"
        )
        
        # Get policy configs
        baseline_cfg = self.policies.get(baseline_policy)
        new_cfg = self.policies.get(new_policy)
        
        if not baseline_cfg or not new_cfg:
            raise ValueError(f"Unknown policy: {baseline_policy} or {new_policy}")
        
        # Determine baseline success rate
        if outcome_col in population_df.columns:
            baseline_success = population_df[outcome_col].mean()
        elif baseline_rate is not None:
            baseline_success = baseline_rate
        else:
            baseline_success = 0.30  # Default assumption
        
        # Simulate baseline policy
        baseline_results = self._simulate_single_policy(
            population_df,
            baseline_cfg,
            baseline_success,
            time_horizon
        )
        
        # Simulate new policy (with treatment effect)
        new_success = baseline_success * (1 + new_cfg.treatment_effect)
        new_results = self._simulate_single_policy(
            population_df,
            new_cfg,
            new_success,
            time_horizon
        )
        
        # Calculate impact metrics
        absolute_uplift = new_results['success_rate'] - baseline_results['success_rate']
        relative_uplift = absolute_uplift / baseline_results['success_rate']
        
        incremental_successes = (
            new_results['total_successes'] - baseline_results['total_successes']
        )
        incremental_cost = new_results['total_cost'] - baseline_results['total_cost']
        
        cost_per_incremental = (
            incremental_cost / incremental_successes 
            if incremental_successes > 0 else np.inf
        )
        
        results = {
            'baseline': baseline_results,
            'new_policy': new_results,
            'absolute_uplift': absolute_uplift,
            'relative_uplift': relative_uplift,
            'incremental_successes': incremental_successes,
            'incremental_cost': incremental_cost,
            'cost_per_incremental_success': cost_per_incremental,
            'roi': (incremental_successes / incremental_cost 
                   if incremental_cost > 0 else np.inf),
            'population_size': len(population_df)
        }
        
        logger.info(
            f"Policy impact: uplift={relative_uplift:.2%}, "
            f"incremental_successes={incremental_successes:,.0f}, "
            f"cost_per_incremental=${cost_per_incremental:.2f}"
        )
        
        return results
    
    def _simulate_single_policy(
        self,
        population_df: pd.DataFrame,
        policy: PolicyConfig,
        base_success_rate: float,
        time_horizon: int
    ) -> Dict[str, Any]:
        """Simulate a single policy on population."""
        n = len(population_df)
        
        # Calculate number of contacts
        max_contact_rounds = min(
            policy.max_contacts,
            time_horizon // policy.contact_frequency
        )
        
        # Simulate channel effectiveness
        total_cost = 0
        total_contacts = 0
        successes = np.zeros(n, dtype=bool)
        
        for contact_round in range(max_contact_rounds):
            # Only contact accounts that haven't succeeded yet
            eligible_for_contact = ~successes
            
            if not eligible_for_contact.any():
                # All accounts have succeeded, no need for more contacts
                break
            
            # Select channel for this round
            channel_name = policy.channels[contact_round % len(policy.channels)]
            channel = self.channels[channel_name]
            
            # Simulate contact success (only for eligible accounts)
            contact_attempts = np.zeros(n, dtype=bool)
            contact_attempts[eligible_for_contact] = (
                self.rng.random(eligible_for_contact.sum()) < channel.contact_rate
            )
            
            # Among contacted, simulate response
            response_attempts = np.zeros(n, dtype=bool)
            response_attempts[contact_attempts] = (
                self.rng.random(contact_attempts.sum()) < channel.response_rate
            )
            
            # Among responded, simulate success
            round_successes = np.zeros(n, dtype=bool)
            round_successes[response_attempts] = (
                self.rng.random(response_attempts.sum()) < base_success_rate
            )
            
            # Update successes
            successes = successes | round_successes
            
            # Calculate costs (only for actual contacts made)
            contacts_made = contact_attempts.sum()
            total_contacts += contacts_made
            total_cost += contacts_made * channel.cost
        
        return {
            'success_rate': successes.mean(),
            'total_successes': successes.sum(),
            'total_contacts': total_contacts,
            'total_cost': total_cost,
            'cost_per_contact': total_cost / total_contacts if total_contacts > 0 else 0,
            'cost_per_success': total_cost / successes.sum() if successes.any() else np.inf,
            'contact_rounds': max_contact_rounds
        }
    
    def simulate_ab_test(
        self,
        population_df: pd.DataFrame,
        control_policy: str,
        treatment_policy: str,
        treatment_fraction: float = 0.5,
        outcome_col: str = 'will_pay',
        baseline_rate: Optional[float] = None,
        test_duration: int = 90,
        min_sample_size: int = 1000
    ) -> Dict[str, Any]:
        """
        Simulate A/B test between two policies.
        
        Args:
            population_df: Population for testing
            control_policy: Control group policy
            treatment_policy: Treatment group policy
            treatment_fraction: Fraction assigned to treatment
            outcome_col: Outcome column name
            baseline_rate: Baseline success rate
            test_duration: Test duration in days
            min_sample_size: Minimum sample size per group
            
        Returns:
            A/B test results with statistical significance
            
        Example:
            >>> ab_results = simulator.simulate_ab_test(
            ...     population=customers,
            ...     control_policy='current',
            ...     treatment_policy='new',
            ...     test_duration=60
            ... )
        """
        logger.info(
            f"Simulating A/B test: {control_policy} vs {treatment_policy}, "
            f"n={len(population_df):,}"
        )
        
        # Check sample size
        treatment_size = int(len(population_df) * treatment_fraction)
        control_size = len(population_df) - treatment_size
        
        if treatment_size < min_sample_size or control_size < min_sample_size:
            logger.warning(
                f"Sample size too small: control={control_size}, "
                f"treatment={treatment_size}, min={min_sample_size}"
            )
        
        # Randomly assign to groups
        is_treatment = self.rng.random(len(population_df)) < treatment_fraction
        
        control_df = population_df[~is_treatment]
        treatment_df = population_df[is_treatment]
        
        # Simulate both policies
        impact = self.simulate_policy_impact(
            population_df=population_df,
            baseline_policy=control_policy,
            new_policy=treatment_policy,
            outcome_col=outcome_col,
            baseline_rate=baseline_rate,
            time_horizon=test_duration
        )
        
        # Calculate statistical significance
        control_success = impact['baseline']['total_successes']
        treatment_success = impact['new_policy']['total_successes']
        
        # Proportion z-test
        p1 = control_success / control_size
        p2 = treatment_success / treatment_size
        
        p_pooled = (control_success + treatment_success) / (control_size + treatment_size)
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/control_size + 1/treatment_size))
        
        z_score = (p2 - p1) / se if se > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        # Confidence interval for uplift
        se_diff = np.sqrt(p1*(1-p1)/control_size + p2*(1-p2)/treatment_size)
        ci_lower = (p2 - p1) - 1.96 * se_diff
        ci_upper = (p2 - p1) + 1.96 * se_diff
        
        return {
            'control': {
                'policy': control_policy,
                'size': control_size,
                'successes': control_success,
                'success_rate': p1,
                'total_cost': impact['baseline']['total_cost']
            },
            'treatment': {
                'policy': treatment_policy,
                'size': treatment_size,
                'successes': treatment_success,
                'success_rate': p2,
                'total_cost': impact['new_policy']['total_cost']
            },
            'uplift': p2 - p1,
            'relative_uplift': (p2 - p1) / p1 if p1 > 0 else 0,
            'z_score': z_score,
            'p_value': p_value,
            'is_significant': p_value < 0.05,
            'confidence_interval': (ci_lower, ci_upper),
            'incremental_cost': impact['incremental_cost'],
            'cost_per_incremental': impact['cost_per_incremental_success']
        }
    
    def optimize_channel_mix(
        self,
        population_df: pd.DataFrame,
        available_channels: List[str],
        budget: float,
        target_contacts: Optional[int] = None,
        outcome_col: str = 'will_pay',
        baseline_rate: float = 0.30
    ) -> Dict[str, Any]:
        """
        Optimize channel mix given budget and constraints.
        
        Args:
            population_df: Target population
            available_channels: List of channel names
            budget: Total budget available
            target_contacts: Target number of contacts
            outcome_col: Outcome column
            baseline_rate: Baseline success rate
            
        Returns:
            Optimal channel allocation
            
        Example:
            >>> optimal = simulator.optimize_channel_mix(
            ...     population=accounts,
            ...     available_channels=['email', 'sms', 'call'],
            ...     budget=100000
            ... )
        """
        logger.info(
            f"Optimizing channel mix: budget=${budget:,.2f}, "
            f"channels={available_channels}"
        )
        
        n = len(population_df)
        
        # Calculate efficiency (successes per dollar) for each channel
        channel_efficiency = {}
        
        for channel_name in available_channels:
            channel = self.channels[channel_name]
            
            # Expected successes per contact
            expected_success = (
                channel.contact_rate * 
                channel.response_rate * 
                baseline_rate
            )
            
            # Efficiency = successes per dollar
            efficiency = expected_success / channel.cost
            
            channel_efficiency[channel_name] = {
                'efficiency': efficiency,
                'cost': channel.cost,
                'expected_success_rate': expected_success,
                'channel': channel
            }
        
        # Sort channels by efficiency
        sorted_channels = sorted(
            channel_efficiency.items(),
            key=lambda x: x[1]['efficiency'],
            reverse=True
        )
        
        # Allocate budget (greedy algorithm)
        allocation = {}
        remaining_budget = budget
        total_contacts = 0
        total_expected_successes = 0
        
        for channel_name, info in sorted_channels:
            channel = info['channel']
            
            # Calculate max contacts for this channel
            if target_contacts:
                max_by_target = max(0, target_contacts - total_contacts)
            else:
                max_by_target = n
            
            max_by_budget = int(remaining_budget / channel.cost)
            
            if channel.capacity_per_day:
                max_by_capacity = channel.capacity_per_day
            else:
                max_by_capacity = n
            
            contacts = min(max_by_target, max_by_budget, max_by_capacity, n)
            
            if contacts > 0:
                cost = contacts * channel.cost
                expected_successes = contacts * info['expected_success_rate']
                
                allocation[channel_name] = {
                    'contacts': contacts,
                    'cost': cost,
                    'expected_successes': expected_successes,
                    'cost_per_success': cost / expected_successes if expected_successes > 0 else np.inf
                }
                
                remaining_budget -= cost
                total_contacts += contacts
                total_expected_successes += expected_successes
            
            if remaining_budget <= 0 or (target_contacts and total_contacts >= target_contacts):
                break
        
        return {
            'allocation': allocation,
            'total_budget_used': budget - remaining_budget,
            'remaining_budget': remaining_budget,
            'total_contacts': total_contacts,
            'total_expected_successes': total_expected_successes,
            'overall_cost_per_success': (
                (budget - remaining_budget) / total_expected_successes 
                if total_expected_successes > 0 else np.inf
            ),
            'budget_utilization': (budget - remaining_budget) / budget
        }
    
    def simulate_capacity_constraints(
        self,
        population_df: pd.DataFrame,
        policy: str,
        channel_capacities: Dict[str, int],
        time_horizon: int = 30
    ) -> Dict[str, Any]:
        """
        Simulate policy with capacity constraints.
        
        Args:
            population_df: Target population
            policy: Policy name
            channel_capacities: Daily capacity for each channel
            time_horizon: Simulation horizon in days
            
        Returns:
            Simulation results with capacity utilization
        """
        logger.info(f"Simulating capacity constraints for policy: {policy}")
        
        policy_cfg = self.policies.get(policy)
        if not policy_cfg:
            raise ValueError(f"Unknown policy: {policy}")
        
        # Update channel capacities
        for channel_name, capacity in channel_capacities.items():
            if channel_name in self.channels:
                self.channels[channel_name].capacity_per_day = capacity
        
        # Simulate policy
        results = self._simulate_single_policy(
            population_df,
            policy_cfg,
            base_success_rate=0.30,
            time_horizon=time_horizon
        )
        
        # Calculate capacity utilization
        days_active = time_horizon // policy_cfg.contact_frequency
        
        capacity_utilization = {}
        for channel_name in policy_cfg.channels:
            channel = self.channels[channel_name]
            if channel.capacity_per_day:
                total_capacity = channel.capacity_per_day * days_active
                utilization = min(1.0, results['total_contacts'] / total_capacity)
                capacity_utilization[channel_name] = utilization
        
        results['capacity_utilization'] = capacity_utilization
        results['days_active'] = days_active
        
        return results
    
    def sensitivity_analysis(
        self,
        population_df: pd.DataFrame,
        policy: str,
        parameter: str,
        parameter_range: np.ndarray,
        baseline_rate: float = 0.30
    ) -> pd.DataFrame:
        """
        Perform sensitivity analysis on policy parameter.
        
        Args:
            population_df: Target population
            policy: Policy name
            parameter: Parameter to vary
            parameter_range: Range of parameter values
            baseline_rate: Baseline success rate
            
        Returns:
            DataFrame with sensitivity results
        """
        logger.info(f"Running sensitivity analysis on {parameter}")
        
        results = []
        
        for param_value in parameter_range:
            # Create modified policy
            policy_cfg = self.policies[policy]
            modified_cfg = PolicyConfig(
                name=f"{policy}_{parameter}_{param_value}",
                target_segments=policy_cfg.target_segments,
                channels=policy_cfg.channels,
                contact_frequency=policy_cfg.contact_frequency,
                max_contacts=policy_cfg.max_contacts,
                treatment_effect=policy_cfg.treatment_effect
            )
            
            # Modify parameter
            if parameter == 'treatment_effect':
                modified_cfg.treatment_effect = param_value
            elif parameter == 'contact_frequency':
                modified_cfg.contact_frequency = int(param_value)
            elif parameter == 'max_contacts':
                modified_cfg.max_contacts = int(param_value)
            
            # Simulate
            sim_result = self._simulate_single_policy(
                population_df,
                modified_cfg,
                baseline_rate * (1 + modified_cfg.treatment_effect),
                time_horizon=90
            )
            
            results.append({
                parameter: param_value,
                'success_rate': sim_result['success_rate'],
                'total_cost': sim_result['total_cost'],
                'cost_per_success': sim_result['cost_per_success']
            })
        
        return pd.DataFrame(results)
    
    def plot_policy_comparison(
        self,
        comparison_results: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot comparison of policies."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        baseline = comparison_results['baseline']
        new_policy = comparison_results['new_policy']
        
        # Success rates
        ax = axes[0, 0]
        policies = ['Baseline', 'New Policy']
        success_rates = [baseline['success_rate'], new_policy['success_rate']]
        ax.bar(policies, success_rates, color=['steelblue', 'coral'])
        ax.set_ylabel('Success Rate')
        ax.set_title('Success Rate Comparison')
        ax.set_ylim([0, max(success_rates) * 1.2])
        
        for i, v in enumerate(success_rates):
            ax.text(i, v + max(success_rates)*0.02, f'{v:.2%}', ha='center')
        
        # Costs
        ax = axes[0, 1]
        costs = [baseline['total_cost'], new_policy['total_cost']]
        ax.bar(policies, costs, color=['steelblue', 'coral'])
        ax.set_ylabel('Total Cost ($)')
        ax.set_title('Cost Comparison')
        
        # Cost per success
        ax = axes[1, 0]
        cps = [baseline['cost_per_success'], new_policy['cost_per_success']]
        ax.bar(policies, cps, color=['steelblue', 'coral'])
        ax.set_ylabel('Cost per Success ($)')
        ax.set_title('Efficiency Comparison')
        
        # Uplift
        ax = axes[1, 1]
        metrics = ['Absolute\nUplift', 'Relative\nUplift', 'Incremental\nSuccesses']
        values = [
            comparison_results['absolute_uplift'],
            comparison_results['relative_uplift'],
            comparison_results['incremental_successes'] / 1000
        ]
        ax.bar(metrics, values, color='green', alpha=0.7)
        ax.set_title('Impact Metrics')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved comparison plot to {save_path}")
        
        return fig
    
    def save_simulation(self, results: Dict[str, Any], filepath: str) -> None:
        """Save simulation results."""
        filepath = Path(filepath)
        
        # Convert to JSON-serializable format
        json_results = {}
        for key, value in results.items():
            if isinstance(value, (np.ndarray, pd.Series)):
                json_results[key] = value.tolist()
            elif isinstance(value, pd.DataFrame):
                json_results[key] = value.to_dict('records')
            elif isinstance(value, (ChannelConfig, PolicyConfig)):
                json_results[key] = value.__dict__
            else:
                json_results[key] = value
        
        with open(filepath, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Saved simulation results to {filepath}")


if __name__ == "__main__":
    # Example usage
    logger.info("Running policy simulation example")
    
    # Create sample population
    np.random.seed(42)
    n = 10000
    
    population = pd.DataFrame({
        'account_id': range(n),
        'balance': np.random.uniform(100, 10000, n),
        'risk_score': np.random.uniform(0, 1, n),
        'segment': np.random.choice(['low_risk', 'medium_risk', 'high_risk'], n),
        'will_pay': np.random.random(n) < 0.30
    })
    
    # Initialize simulator
    simulator = PolicySimulator(random_seed=42)
    
    # Simulate policy impact
    results = simulator.simulate_policy_impact(
        population,
        baseline_policy='reactive',
        new_policy='proactive'
    )
    
    print(f"\nPolicy Impact Results:")
    print(f"Relative Uplift: {results['relative_uplift']:.2%}")
    print(f"Incremental Successes: {results['incremental_successes']:,.0f}")
    print(f"Cost per Incremental Success: ${results['cost_per_incremental_success']:.2f}")
    
    # Optimize channel mix
    optimal = simulator.optimize_channel_mix(
        population,
        available_channels=['email', 'sms', 'call'],
        budget=50000
    )
    
    print(f"\nOptimal Channel Mix:")
    for channel, alloc in optimal['allocation'].items():
        print(f"  {channel}: {alloc['contacts']:,} contacts, ${alloc['cost']:,.2f}")
