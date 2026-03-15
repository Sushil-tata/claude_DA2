"""
Simulation package for the Principal Data Science Decision Agent.

This package provides comprehensive simulation capabilities including:
- Monte Carlo simulation for repayment modeling and scenario analysis
- Markov chain modeling for state transitions and roll rate analysis
- Policy impact simulation for A/B testing and strategy evaluation
- Stress testing for risk assessment and regulatory compliance

Classes:
    MonteCarloSimulator: Monte Carlo simulation engine
    MarkovChainModel: Markov chain state transition modeling
    PolicySimulator: Policy impact and A/B testing simulation
    StressTestEngine: Comprehensive stress testing engine
"""

from src.simulation.monte_carlo import (
    MonteCarloSimulator,
    SimulationConfig,
    ScenarioParameters
)

from src.simulation.markov_chains import (
    MarkovChainModel,
    MarkovConfig
)

from src.simulation.policy_simulator import (
    PolicySimulator,
    ChannelConfig,
    PolicyConfig
)

from src.simulation.stress_testing import (
    StressTestEngine,
    StressScenario,
    StressTestConfig
)

__all__ = [
    # Monte Carlo
    'MonteCarloSimulator',
    'SimulationConfig',
    'ScenarioParameters',
    
    # Markov Chains
    'MarkovChainModel',
    'MarkovConfig',
    
    # Policy Simulation
    'PolicySimulator',
    'ChannelConfig',
    'PolicyConfig',
    
    # Stress Testing
    'StressTestEngine',
    'StressScenario',
    'StressTestConfig',
]
