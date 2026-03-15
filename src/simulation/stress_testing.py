"""
Stress Testing Engine for the Principal Data Science Decision Agent.

This module provides comprehensive stress testing capabilities for risk assessment,
regulatory compliance, and scenario analysis. It supports economic stress scenarios,
PD/LGD stress testing, vintage analysis, and portfolio-level assessments.

Features:
    - Economic stress scenarios (recession, boom, crisis)
    - PD (Probability of Default) stress testing
    - LGD (Loss Given Default) stress testing
    - Vintage analysis under stress
    - Portfolio-level stress tests
    - Correlation stress scenarios
    - Regulatory stress scenarios (CCAR/DFAST style)
    - Reverse stress testing
    - Comprehensive reporting and visualization

Example:
    >>> from src.simulation.stress_testing import StressTestEngine
    >>> 
    >>> engine = StressTestEngine()
    >>> 
    >>> # Run stress test
    >>> results = engine.run_stress_test(
    ...     portfolio_df=loans,
    ...     scenario='severe_recession',
    ...     metrics=['pd', 'lgd', 'ead']
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
from datetime import datetime


@dataclass
class StressScenario:
    """Configuration for stress test scenario."""
    name: str
    description: str
    pd_multiplier: float  # Multiplier for PD
    lgd_multiplier: float  # Multiplier for LGD
    ead_multiplier: float  # Multiplier for EAD
    recovery_rate_adjustment: float  # Additive adjustment
    unemployment_rate: float
    gdp_growth: float
    house_price_change: float
    correlation_increase: float = 0.0
    duration_months: int = 24
    

@dataclass  
class StressTestConfig:
    """Configuration for stress testing."""
    confidence_level: float = 0.99
    time_horizon: int = 12  # months
    n_simulations: int = 10000
    include_correlation: bool = True
    regulatory_framework: str = 'CCAR'  # CCAR, DFAST, Basel
    default_lgd: float = 0.45  # Default Loss Given Default assumption (45% loss rate = 55% recovery)


class StressTestEngine:
    """
    Comprehensive stress testing engine for credit risk and portfolio analysis.
    
    This class provides extensive stress testing capabilities including economic
    scenario analysis, PD/LGD stress testing, vintage analysis, and regulatory
    compliance testing (CCAR/DFAST style).
    
    Attributes:
        config (StressTestConfig): Configuration parameters
        scenarios (Dict[str, StressScenario]): Pre-defined stress scenarios
        rng (np.random.Generator): Random number generator
        
    Example:
        >>> engine = StressTestEngine()
        >>> results = engine.run_stress_test(
        ...     portfolio=loans,
        ...     scenario='severe_recession'
        ... )
        >>> print(f"Stressed PD: {results['stressed_pd']:.2%}")
    """
    
    def __init__(
        self,
        confidence_level: float = 0.99,
        time_horizon: int = 12,
        n_simulations: int = 10000,
        random_seed: Optional[int] = None
    ):
        """
        Initialize stress testing engine.
        
        Args:
            confidence_level: Confidence level for VaR calculations
            time_horizon: Stress test time horizon in months
            n_simulations: Number of Monte Carlo simulations
            random_seed: Seed for reproducibility
        """
        self.config = StressTestConfig(
            confidence_level=confidence_level,
            time_horizon=time_horizon,
            n_simulations=n_simulations
        )
        
        self.rng = np.random.default_rng(random_seed)
        
        # Initialize stress scenarios
        self.scenarios = self._initialize_scenarios()
        
        logger.info(
            f"Initialized StressTestEngine: confidence={confidence_level:.2%}, "
            f"horizon={time_horizon} months, simulations={n_simulations:,}"
        )
    
    def _initialize_scenarios(self) -> Dict[str, StressScenario]:
        """Initialize pre-defined stress scenarios."""
        return {
            'baseline': StressScenario(
                name='Baseline',
                description='Normal economic conditions',
                pd_multiplier=1.0,
                lgd_multiplier=1.0,
                ead_multiplier=1.0,
                recovery_rate_adjustment=0.0,
                unemployment_rate=4.5,
                gdp_growth=2.5,
                house_price_change=3.0,
                duration_months=12
            ),
            'moderate_recession': StressScenario(
                name='Moderate Recession',
                description='Moderate economic downturn',
                pd_multiplier=1.5,
                lgd_multiplier=1.2,
                ead_multiplier=1.1,
                recovery_rate_adjustment=-0.10,
                unemployment_rate=7.5,
                gdp_growth=-1.0,
                house_price_change=-5.0,
                correlation_increase=0.15,
                duration_months=18
            ),
            'severe_recession': StressScenario(
                name='Severe Recession',
                description='Severe economic downturn similar to 2008',
                pd_multiplier=2.5,
                lgd_multiplier=1.5,
                ead_multiplier=1.3,
                recovery_rate_adjustment=-0.20,
                unemployment_rate=10.0,
                gdp_growth=-3.5,
                house_price_change=-15.0,
                correlation_increase=0.30,
                duration_months=24
            ),
            'financial_crisis': StressScenario(
                name='Financial Crisis',
                description='Extreme financial market disruption',
                pd_multiplier=3.5,
                lgd_multiplier=2.0,
                ead_multiplier=1.5,
                recovery_rate_adjustment=-0.30,
                unemployment_rate=12.0,
                gdp_growth=-5.0,
                house_price_change=-25.0,
                correlation_increase=0.50,
                duration_months=36
            ),
            'stagflation': StressScenario(
                name='Stagflation',
                description='High inflation with economic stagnation',
                pd_multiplier=2.0,
                lgd_multiplier=1.4,
                ead_multiplier=1.2,
                recovery_rate_adjustment=-0.15,
                unemployment_rate=8.0,
                gdp_growth=0.0,
                house_price_change=-8.0,
                correlation_increase=0.20,
                duration_months=24
            ),
            'rapid_recovery': StressScenario(
                name='Rapid Recovery',
                description='Strong economic recovery',
                pd_multiplier=0.7,
                lgd_multiplier=0.8,
                ead_multiplier=0.9,
                recovery_rate_adjustment=0.10,
                unemployment_rate=3.5,
                gdp_growth=5.0,
                house_price_change=8.0,
                correlation_increase=-0.10,
                duration_months=12
            )
        }
    
    def run_stress_test(
        self,
        portfolio_df: pd.DataFrame,
        scenario: str,
        pd_col: str = 'pd',
        lgd_col: str = 'lgd',
        ead_col: str = 'ead',
        segment_col: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run comprehensive stress test on portfolio.
        
        Args:
            portfolio_df: Portfolio DataFrame
            scenario: Scenario name
            pd_col: Column name for PD
            lgd_col: Column name for LGD
            ead_col: Column name for EAD
            segment_col: Optional segment column
            
        Returns:
            Comprehensive stress test results
            
        Example:
            >>> results = engine.run_stress_test(
            ...     portfolio=loans,
            ...     scenario='severe_recession',
            ...     pd_col='default_prob',
            ...     ead_col='balance'
            ... )
        """
        logger.info(
            f"Running stress test: scenario={scenario}, "
            f"portfolio_size={len(portfolio_df):,}"
        )
        
        scenario_cfg = self.scenarios.get(scenario)
        if not scenario_cfg:
            raise ValueError(f"Unknown scenario: {scenario}")
        
        # Extract baseline values
        baseline_pd = portfolio_df[pd_col].values
        baseline_lgd = (
            portfolio_df[lgd_col].values if lgd_col in portfolio_df.columns 
            else np.full(len(portfolio_df), self.config.default_lgd)
        )
        baseline_ead = portfolio_df[ead_col].values
        
        # Apply stress
        stressed_pd = self._stress_pd(baseline_pd, scenario_cfg)
        stressed_lgd = self._stress_lgd(baseline_lgd, scenario_cfg)
        stressed_ead = self._stress_ead(baseline_ead, scenario_cfg)
        
        # Calculate expected loss
        baseline_el = baseline_pd * baseline_lgd * baseline_ead
        stressed_el = stressed_pd * stressed_lgd * stressed_ead
        
        # Portfolio-level metrics
        total_baseline_el = baseline_el.sum()
        total_stressed_el = stressed_el.sum()
        total_ead = baseline_ead.sum()
        
        # Calculate VaR and CVaR
        var_baseline = self._calculate_var(baseline_el, self.config.confidence_level)
        var_stressed = self._calculate_var(stressed_el, self.config.confidence_level)
        
        cvar_baseline = self._calculate_cvar(baseline_el, self.config.confidence_level)
        cvar_stressed = self._calculate_cvar(stressed_el, self.config.confidence_level)
        
        results = {
            'scenario': scenario_cfg,
            'portfolio_metrics': {
                'total_ead': total_ead,
                'baseline_el': total_baseline_el,
                'stressed_el': total_stressed_el,
                'el_increase': total_stressed_el - total_baseline_el,
                'el_increase_pct': (total_stressed_el - total_baseline_el) / total_baseline_el if total_baseline_el > 0 else 0,
                'baseline_el_rate': total_baseline_el / total_ead if total_ead > 0 else 0,
                'stressed_el_rate': total_stressed_el / total_ead if total_ead > 0 else 0
            },
            'risk_metrics': {
                'var_baseline': var_baseline,
                'var_stressed': var_stressed,
                'var_increase': var_stressed - var_baseline,
                'cvar_baseline': cvar_baseline,
                'cvar_stressed': cvar_stressed,
                'cvar_increase': cvar_stressed - cvar_baseline
            },
            'component_impacts': {
                'pd_impact': (stressed_pd.mean() - baseline_pd.mean()) / baseline_pd.mean() if baseline_pd.mean() > 0 else 0,
                'lgd_impact': (stressed_lgd.mean() - baseline_lgd.mean()) / baseline_lgd.mean() if baseline_lgd.mean() > 0 else 0,
                'ead_impact': (stressed_ead.mean() - baseline_ead.mean()) / baseline_ead.mean() if baseline_ead.mean() > 0 else 0
            },
            'distributions': {
                'baseline_pd': baseline_pd,
                'stressed_pd': stressed_pd,
                'baseline_lgd': baseline_lgd,
                'stressed_lgd': stressed_lgd,
                'baseline_el': baseline_el,
                'stressed_el': stressed_el
            }
        }
        
        # Segment analysis if requested
        if segment_col and segment_col in portfolio_df.columns:
            results['segment_analysis'] = self._analyze_by_segment(
                portfolio_df,
                segment_col,
                baseline_el,
                stressed_el,
                baseline_ead
            )
        
        logger.info(
            f"Stress test complete: EL increase={results['portfolio_metrics']['el_increase_pct']:.2%}, "
            f"stressed_EL=${total_stressed_el:,.0f}"
        )
        
        return results
    
    def _stress_pd(
        self,
        baseline_pd: np.ndarray,
        scenario: StressScenario
    ) -> np.ndarray:
        """Apply PD stress."""
        stressed = baseline_pd * scenario.pd_multiplier
        
        # Add some random variation
        variation = self.rng.normal(0, 0.05 * scenario.pd_multiplier, len(baseline_pd))
        stressed = stressed * (1 + variation)
        
        # Ensure valid probabilities
        return np.clip(stressed, 0, 1)
    
    def _stress_lgd(
        self,
        baseline_lgd: np.ndarray,
        scenario: StressScenario
    ) -> np.ndarray:
        """Apply LGD stress."""
        stressed = baseline_lgd * scenario.lgd_multiplier
        stressed = stressed - scenario.recovery_rate_adjustment
        
        # Add random variation
        variation = self.rng.normal(0, 0.03 * scenario.lgd_multiplier, len(baseline_lgd))
        stressed = stressed * (1 + variation)
        
        return np.clip(stressed, 0, 1)
    
    def _stress_ead(
        self,
        baseline_ead: np.ndarray,
        scenario: StressScenario
    ) -> np.ndarray:
        """Apply EAD stress."""
        stressed = baseline_ead * scenario.ead_multiplier
        
        # Add random variation
        variation = self.rng.normal(0, 0.02 * scenario.ead_multiplier, len(baseline_ead))
        stressed = stressed * (1 + variation)
        
        return np.maximum(stressed, 0)
    
    def _calculate_var(
        self,
        losses: np.ndarray,
        confidence_level: float
    ) -> float:
        """Calculate Value at Risk."""
        return np.percentile(losses, confidence_level * 100)
    
    def _calculate_cvar(
        self,
        losses: np.ndarray,
        confidence_level: float
    ) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)."""
        var = self._calculate_var(losses, confidence_level)
        return losses[losses >= var].mean()
    
    def _analyze_by_segment(
        self,
        portfolio_df: pd.DataFrame,
        segment_col: str,
        baseline_el: np.ndarray,
        stressed_el: np.ndarray,
        baseline_ead: np.ndarray
    ) -> pd.DataFrame:
        """Analyze stress test results by segment."""
        segments = portfolio_df[segment_col].values
        
        results = []
        for segment in np.unique(segments):
            mask = segments == segment
            
            results.append({
                'segment': segment,
                'count': mask.sum(),
                'total_ead': baseline_ead[mask].sum(),
                'baseline_el': baseline_el[mask].sum(),
                'stressed_el': stressed_el[mask].sum(),
                'el_increase': stressed_el[mask].sum() - baseline_el[mask].sum(),
                'el_increase_pct': (
                    (stressed_el[mask].sum() - baseline_el[mask].sum()) / baseline_el[mask].sum()
                    if baseline_el[mask].sum() > 0 else 0
                )
            })
        
        return pd.DataFrame(results).sort_values('el_increase', ascending=False)
    
    def run_vintage_stress_test(
        self,
        portfolio_df: pd.DataFrame,
        vintage_col: str,
        scenario: str,
        pd_col: str = 'pd',
        lgd_col: str = 'lgd',
        ead_col: str = 'ead'
    ) -> pd.DataFrame:
        """
        Run stress test by vintage cohort.
        
        Args:
            portfolio_df: Portfolio DataFrame
            vintage_col: Column name for vintage/origination period
            scenario: Scenario name
            pd_col: PD column name
            lgd_col: LGD column name
            ead_col: EAD column name
            
        Returns:
            Vintage-level stress test results
            
        Example:
            >>> vintage_results = engine.run_vintage_stress_test(
            ...     portfolio=loans,
            ...     vintage_col='origination_year',
            ...     scenario='severe_recession'
            ... )
        """
        logger.info(f"Running vintage stress test: scenario={scenario}")
        
        scenario_cfg = self.scenarios[scenario]
        
        results = []
        
        for vintage, vintage_df in portfolio_df.groupby(vintage_col):
            vintage_results = self.run_stress_test(
                vintage_df,
                scenario=scenario,
                pd_col=pd_col,
                lgd_col=lgd_col,
                ead_col=ead_col
            )
            
            results.append({
                'vintage': vintage,
                'count': len(vintage_df),
                'total_ead': vintage_results['portfolio_metrics']['total_ead'],
                'baseline_el': vintage_results['portfolio_metrics']['baseline_el'],
                'stressed_el': vintage_results['portfolio_metrics']['stressed_el'],
                'el_increase_pct': vintage_results['portfolio_metrics']['el_increase_pct'],
                'stressed_el_rate': vintage_results['portfolio_metrics']['stressed_el_rate']
            })
        
        return pd.DataFrame(results).sort_values('vintage')
    
    def compare_scenarios(
        self,
        portfolio_df: pd.DataFrame,
        scenarios: List[str],
        pd_col: str = 'pd',
        lgd_col: str = 'lgd',
        ead_col: str = 'ead'
    ) -> pd.DataFrame:
        """
        Compare multiple stress scenarios.
        
        Args:
            portfolio_df: Portfolio DataFrame
            scenarios: List of scenario names
            pd_col: PD column name
            lgd_col: LGD column name
            ead_col: EAD column name
            
        Returns:
            Comparison DataFrame
            
        Example:
            >>> comparison = engine.compare_scenarios(
            ...     portfolio=loans,
            ...     scenarios=['baseline', 'moderate_recession', 'severe_recession']
            ... )
        """
        logger.info(f"Comparing scenarios: {scenarios}")
        
        results = []
        
        for scenario in scenarios:
            scenario_results = self.run_stress_test(
                portfolio_df,
                scenario=scenario,
                pd_col=pd_col,
                lgd_col=lgd_col,
                ead_col=ead_col
            )
            
            results.append({
                'scenario': scenario,
                'total_ead': scenario_results['portfolio_metrics']['total_ead'],
                'expected_loss': scenario_results['portfolio_metrics']['stressed_el'],
                'el_rate': scenario_results['portfolio_metrics']['stressed_el_rate'],
                'var': scenario_results['risk_metrics']['var_stressed'],
                'cvar': scenario_results['risk_metrics']['cvar_stressed'],
                'pd_impact': scenario_results['component_impacts']['pd_impact'],
                'lgd_impact': scenario_results['component_impacts']['lgd_impact']
            })
        
        return pd.DataFrame(results)
    
    def reverse_stress_test(
        self,
        portfolio_df: pd.DataFrame,
        target_el_increase: float,
        pd_col: str = 'pd',
        lgd_col: str = 'lgd',
        ead_col: str = 'ead',
        max_iterations: int = 100
    ) -> Dict[str, Any]:
        """
        Reverse stress test: find scenario that produces target EL increase.
        
        Args:
            portfolio_df: Portfolio DataFrame
            target_el_increase: Target expected loss increase (e.g., 2.0 for 200%)
            pd_col: PD column name
            lgd_col: LGD column name
            ead_col: EAD column name
            max_iterations: Maximum optimization iterations
            
        Returns:
            Scenario parameters that achieve target
            
        Example:
            >>> reverse = engine.reverse_stress_test(
            ...     portfolio=loans,
            ...     target_el_increase=2.0  # 200% increase
            ... )
        """
        logger.info(f"Running reverse stress test: target_increase={target_el_increase:.2%}")
        
        # Get baseline
        baseline = self.run_stress_test(portfolio_df, 'baseline', pd_col, lgd_col, ead_col)
        baseline_el = baseline['portfolio_metrics']['baseline_el']
        target_el = baseline_el * (1 + target_el_increase)
        
        # Binary search for PD multiplier
        pd_multiplier_low = 1.0
        pd_multiplier_high = 5.0
        
        best_scenario = None
        best_diff = float('inf')
        
        for iteration in range(max_iterations):
            pd_multiplier = (pd_multiplier_low + pd_multiplier_high) / 2
            
            # Create test scenario
            test_scenario = StressScenario(
                name='Reverse_Test',
                description='Reverse stress test scenario',
                pd_multiplier=pd_multiplier,
                lgd_multiplier=1.2,
                ead_multiplier=1.1,
                recovery_rate_adjustment=-0.10,
                unemployment_rate=8.0,
                gdp_growth=-2.0,
                house_price_change=-10.0,
                duration_months=12
            )
            
            self.scenarios['reverse_test'] = test_scenario
            
            results = self.run_stress_test(
                portfolio_df,
                'reverse_test',
                pd_col,
                lgd_col,
                ead_col
            )
            
            stressed_el = results['portfolio_metrics']['stressed_el']
            diff = abs(stressed_el - target_el)
            
            if diff < best_diff:
                best_diff = diff
                best_scenario = test_scenario
            
            # Adjust search range
            if stressed_el < target_el:
                pd_multiplier_low = pd_multiplier
            else:
                pd_multiplier_high = pd_multiplier
            
            if diff < baseline_el * 0.01:  # Within 1% of target
                break
        
        logger.info(
            f"Reverse stress test complete: pd_multiplier={best_scenario.pd_multiplier:.2f}, "
            f"iterations={iteration + 1}"
        )
        
        return {
            'scenario': best_scenario,
            'iterations': iteration + 1,
            'target_el': target_el,
            'achieved_el': stressed_el,
            'difference': diff,
            'difference_pct': diff / target_el
        }
    
    def sensitivity_analysis(
        self,
        portfolio_df: pd.DataFrame,
        base_scenario: str,
        parameter: str,
        parameter_range: np.ndarray,
        pd_col: str = 'pd',
        lgd_col: str = 'lgd',
        ead_col: str = 'ead'
    ) -> pd.DataFrame:
        """
        Perform sensitivity analysis on scenario parameter.
        
        Args:
            portfolio_df: Portfolio DataFrame
            base_scenario: Base scenario name
            parameter: Parameter to vary
            parameter_range: Range of parameter values
            pd_col: PD column name
            lgd_col: LGD column name
            ead_col: EAD column name
            
        Returns:
            Sensitivity analysis results
        """
        logger.info(f"Running sensitivity analysis on {parameter}")
        
        base = self.scenarios[base_scenario]
        results = []
        
        for param_value in parameter_range:
            # Create modified scenario
            modified = StressScenario(
                name=f"{base_scenario}_{parameter}_{param_value}",
                description=base.description,
                pd_multiplier=base.pd_multiplier,
                lgd_multiplier=base.lgd_multiplier,
                ead_multiplier=base.ead_multiplier,
                recovery_rate_adjustment=base.recovery_rate_adjustment,
                unemployment_rate=base.unemployment_rate,
                gdp_growth=base.gdp_growth,
                house_price_change=base.house_price_change,
                duration_months=base.duration_months
            )
            
            # Modify parameter
            if parameter == 'pd_multiplier':
                modified.pd_multiplier = param_value
            elif parameter == 'lgd_multiplier':
                modified.lgd_multiplier = param_value
            elif parameter == 'unemployment_rate':
                modified.unemployment_rate = param_value
            elif parameter == 'gdp_growth':
                modified.gdp_growth = param_value
            
            self.scenarios['sensitivity_test'] = modified
            
            # Run stress test
            test_results = self.run_stress_test(
                portfolio_df,
                'sensitivity_test',
                pd_col,
                lgd_col,
                ead_col
            )
            
            results.append({
                parameter: param_value,
                'expected_loss': test_results['portfolio_metrics']['stressed_el'],
                'el_rate': test_results['portfolio_metrics']['stressed_el_rate'],
                'el_increase_pct': test_results['portfolio_metrics']['el_increase_pct']
            })
        
        return pd.DataFrame(results)
    
    def plot_stress_comparison(
        self,
        comparison_df: pd.DataFrame,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot comparison of stress scenarios."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Expected Loss
        ax = axes[0, 0]
        ax.bar(comparison_df['scenario'], comparison_df['expected_loss'] / 1e6)
        ax.set_ylabel('Expected Loss ($M)')
        ax.set_title('Expected Loss by Scenario')
        ax.tick_params(axis='x', rotation=45)
        
        # EL Rate
        ax = axes[0, 1]
        ax.bar(comparison_df['scenario'], comparison_df['el_rate'] * 100)
        ax.set_ylabel('EL Rate (%)')
        ax.set_title('Expected Loss Rate by Scenario')
        ax.tick_params(axis='x', rotation=45)
        
        # VaR
        ax = axes[1, 0]
        ax.bar(comparison_df['scenario'], comparison_df['var'] / 1e6)
        ax.set_ylabel('VaR ($M)')
        ax.set_title('Value at Risk by Scenario')
        ax.tick_params(axis='x', rotation=45)
        
        # Component Impacts
        ax = axes[1, 1]
        width = 0.35
        x = np.arange(len(comparison_df))
        ax.bar(x - width/2, comparison_df['pd_impact'] * 100, width, label='PD Impact')
        ax.bar(x + width/2, comparison_df['lgd_impact'] * 100, width, label='LGD Impact')
        ax.set_ylabel('Impact (%)')
        ax.set_title('Component Impacts')
        ax.set_xticks(x)
        ax.set_xticklabels(comparison_df['scenario'], rotation=45)
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved comparison plot to {save_path}")
        
        return fig
    
    def plot_distribution_comparison(
        self,
        stress_results: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot baseline vs stressed distributions."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        baseline_el = stress_results['distributions']['baseline_el']
        stressed_el = stress_results['distributions']['stressed_el']
        
        # PD Distribution
        ax = axes[0]
        baseline_pd = stress_results['distributions']['baseline_pd']
        stressed_pd = stress_results['distributions']['stressed_pd']
        
        ax.hist(baseline_pd, bins=50, alpha=0.5, label='Baseline', density=True)
        ax.hist(stressed_pd, bins=50, alpha=0.5, label='Stressed', density=True)
        ax.set_xlabel('Probability of Default')
        ax.set_ylabel('Density')
        ax.set_title('PD Distribution: Baseline vs Stressed')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # EL Distribution
        ax = axes[1]
        ax.hist(baseline_el, bins=50, alpha=0.5, label='Baseline', density=True)
        ax.hist(stressed_el, bins=50, alpha=0.5, label='Stressed', density=True)
        ax.set_xlabel('Expected Loss ($)')
        ax.set_ylabel('Density')
        ax.set_title('Expected Loss Distribution: Baseline vs Stressed')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved distribution plot to {save_path}")
        
        return fig
    
    def generate_report(
        self,
        stress_results: Dict[str, Any],
        output_path: str
    ) -> None:
        """
        Generate comprehensive stress test report.
        
        Args:
            stress_results: Stress test results
            output_path: Path to save report (JSON format)
        """
        scenario = stress_results['scenario']
        metrics = stress_results['portfolio_metrics']
        risk = stress_results['risk_metrics']
        
        report = {
            'report_date': datetime.now().isoformat(),
            'scenario': {
                'name': scenario.name,
                'description': scenario.description,
                'parameters': {
                    'pd_multiplier': scenario.pd_multiplier,
                    'lgd_multiplier': scenario.lgd_multiplier,
                    'ead_multiplier': scenario.ead_multiplier,
                    'unemployment_rate': scenario.unemployment_rate,
                    'gdp_growth': scenario.gdp_growth,
                    'house_price_change': scenario.house_price_change
                }
            },
            'portfolio_metrics': {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in metrics.items()
            },
            'risk_metrics': {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in risk.items()
            },
            'component_impacts': stress_results['component_impacts'],
            'summary': {
                'total_ead': metrics['total_ead'],
                'stressed_el': metrics['stressed_el'],
                'el_increase_pct': metrics['el_increase_pct'],
                'stressed_el_rate': metrics['stressed_el_rate'],
                'var_stressed': risk['var_stressed'],
                'cvar_stressed': risk['cvar_stressed']
            }
        }
        
        output_path = Path(output_path)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Generated stress test report: {output_path}")


if __name__ == "__main__":
    # Example usage
    logger.info("Running stress testing example")
    
    # Create sample portfolio
    np.random.seed(42)
    n = 10000
    
    portfolio = pd.DataFrame({
        'loan_id': range(n),
        'pd': np.random.beta(2, 18, n),  # Mean ~0.10
        'lgd': np.random.beta(4, 6, n),  # Mean ~0.40
        'ead': np.random.lognormal(10, 1, n),
        'vintage': np.random.choice(['2020', '2021', '2022', '2023'], n),
        'segment': np.random.choice(['Prime', 'Near-Prime', 'Subprime'], n)
    })
    
    # Initialize engine
    engine = StressTestEngine(random_seed=42)
    
    # Run stress test
    results = engine.run_stress_test(
        portfolio,
        scenario='severe_recession',
        segment_col='segment'
    )
    
    print(f"\nStress Test Results:")
    print(f"Scenario: {results['scenario'].name}")
    print(f"Baseline EL: ${results['portfolio_metrics']['baseline_el']:,.0f}")
    print(f"Stressed EL: ${results['portfolio_metrics']['stressed_el']:,.0f}")
    print(f"EL Increase: {results['portfolio_metrics']['el_increase_pct']:.2%}")
    print(f"Stressed VaR: ${results['risk_metrics']['var_stressed']:,.0f}")
    
    # Compare scenarios
    comparison = engine.compare_scenarios(
        portfolio,
        scenarios=['baseline', 'moderate_recession', 'severe_recession', 'financial_crisis']
    )
    
    print("\nScenario Comparison:")
    print(comparison.to_string(index=False))
