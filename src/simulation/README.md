# Simulation Layer

The Simulation layer provides comprehensive simulation and stress testing capabilities for the Principal Data Science Decision Agent. This module enables Monte Carlo simulations, Markov chain modeling, policy impact analysis, and regulatory stress testing.

## Overview

The Simulation layer consists of four main components:

1. **Monte Carlo Simulation** - Probabilistic scenario analysis and risk modeling
2. **Markov Chain Modeling** - State transition and roll rate analysis
3. **Policy Simulation** - A/B testing and treatment effect estimation
4. **Stress Testing** - Economic scenario and regulatory compliance testing

## Modules

### 1. Monte Carlo Simulation (`monte_carlo.py`)

Comprehensive Monte Carlo simulation engine for repayment modeling and risk analysis.

#### Key Features
- Multiple pre-defined scenarios (baseline, optimistic, pessimistic, stress)
- Configurable simulation iterations (1,000 to 100,000+)
- Net Present Value (NPV) calculations with discounting
- Recovery rate simulations
- Confidence interval calculations (5th, 50th, 95th percentiles)
- Portfolio-level simulations
- Sensitivity analysis
- Distribution visualization

#### Usage Example

```python
from src.simulation.monte_carlo import MonteCarloSimulator
import pandas as pd

# Initialize simulator
simulator = MonteCarloSimulator(
    n_simulations=10000,
    random_seed=42,
    discount_rate=0.10,
    time_horizon=36  # months
)

# Run single loan simulation
results = simulator.simulate_repayments(
    principal=100000,
    default_prob=0.15,
    recovery_rate=0.40,
    scenario='baseline'
)

print(f"Expected NPV: ${results['expected_npv']:,.2f}")
print(f"Default Rate: {results['default_rate']:.2%}")
print(f"95% CI: [${results['confidence_intervals']['p5']:,.2f}, "
      f"${results['confidence_intervals']['p95']:,.2f}]")

# Compare scenarios
comparison = simulator.compare_scenarios(
    principal=100000,
    scenarios=['baseline', 'optimistic', 'pessimistic', 'stress']
)
print(comparison)

# Portfolio simulation
portfolio = pd.DataFrame({
    'loan_id': [1, 2, 3],
    'principal': [10000, 20000, 15000],
    'default_prob': [0.10, 0.15, 0.12]
})

portfolio_results = simulator.simulate_portfolio(
    portfolio_df=portfolio,
    scenario='baseline'
)
print(f"Portfolio NPV: ${portfolio_results['total_npv']:,.2f}")

# Visualize distribution
fig = simulator.plot_distribution(
    results['npv'],
    title="NPV Distribution",
    xlabel="Net Present Value ($)"
)
```

#### Custom Scenarios

```python
# Define custom scenario
custom_params = {
    'name': 'Custom',
    'default_prob': 0.20,
    'default_prob_std': 0.06,
    'recovery_rate': 0.35,
    'recovery_rate_std': 0.12,
    'payment_rate': 0.75,
    'payment_rate_std': 0.15,
    'time_to_default': 10.0,
    'time_to_default_std': 5.0
}

results = simulator.simulate_repayments(
    principal=50000,
    custom_params=custom_params
)
```

### 2. Markov Chain Modeling (`markov_chains.py`)

State transition modeling for delinquency progression and customer lifecycle analysis.

#### Key Features
- Transition matrix estimation from historical data
- Roll rate modeling (Current → DPD30 → DPD60 → DPD90 → Charge-off)
- Steady-state distribution calculation
- Multi-step transition probabilities
- Absorbing state analysis (cure, charge-off)
- Time-to-event estimation
- Segment-specific matrices
- Migration analysis between periods

#### Usage Example

```python
from src.simulation.markov_chains import MarkovChainModel
import pandas as pd

# Define states
states = ['Current', 'DPD30', 'DPD60', 'DPD90', 'ChargeOff']

# Initialize model
model = MarkovChainModel(
    states=states,
    absorbing_states=['ChargeOff'],
    time_unit='month'
)

# Prepare historical data
data = pd.DataFrame({
    'account_id': [1, 1, 1, 2, 2, 2],
    'month': [1, 2, 3, 1, 2, 3],
    'status': ['Current', 'DPD30', 'Current', 'Current', 'Current', 'DPD30']
})

# Fit model
model.fit(
    data,
    state_col='status',
    id_col='account_id',
    time_col='month'
)

# View transition matrix
print(pd.DataFrame(
    model.transition_matrix,
    index=states,
    columns=states
))

# Predict future distribution
dist = model.predict_distribution('DPD30', n_steps=6)
print(f"6-month distribution from DPD30:")
for state, prob in zip(states, dist):
    print(f"  {state}: {prob:.2%}")

# Calculate transition probability
prob_default = model.transition_probability('DPD30', 'ChargeOff', n_steps=12)
print(f"Prob of charge-off in 12 months from DPD30: {prob_default:.2%}")

# Time to absorption
tta = model.time_to_absorption('DPD30', 'ChargeOff')
print(f"Expected time to charge-off: {tta['mean']:.1f} months")

# Steady state
steady = model.calculate_steady_state()
print("Steady-state distribution:")
for state, prob in zip(states, steady):
    print(f"  {state}: {prob:.2%}")

# Visualize
fig = model.plot_transition_diagram()
fig = model.plot_evolution('Current', n_steps=12)
```

#### Roll Rate Analysis

```python
# Automatic DPD bucket creation
model.fit_roll_rates(
    data,
    dpd_col='days_past_due',
    id_col='account_id',
    time_col='month',
    dpd_buckets=[0, 30, 60, 90, 120]
)

# Analyze migrations
migration = model.migration_analysis(
    data,
    state_col='status',
    id_col='account_id',
    time_col='month',
    start_period='2023-01',
    end_period='2023-12'
)
print(migration)
```

### 3. Policy Simulation (`policy_simulator.py`)

A/B testing and policy impact simulation for collections strategies.

#### Key Features
- Treatment effect simulation
- Population-level impact estimation
- Channel optimization (email, SMS, call, letter, IVR)
- Capacity constraint modeling
- Cost-benefit analysis
- Uplift measurement
- Statistical significance testing
- Budget optimization

#### Usage Example

```python
from src.simulation.policy_simulator import PolicySimulator
import pandas as pd

# Initialize simulator
simulator = PolicySimulator(random_seed=42)

# Prepare population
population = pd.DataFrame({
    'account_id': range(10000),
    'balance': np.random.uniform(100, 10000, 10000),
    'risk_score': np.random.uniform(0, 1, 10000),
    'segment': np.random.choice(['low_risk', 'medium_risk', 'high_risk'], 10000)
})

# Simulate policy impact
results = simulator.simulate_policy_impact(
    population_df=population,
    baseline_policy='reactive',
    new_policy='proactive',
    baseline_rate=0.30,
    time_horizon=90
)

print(f"Relative Uplift: {results['relative_uplift']:.2%}")
print(f"Incremental Successes: {results['incremental_successes']:,.0f}")
print(f"Cost per Incremental: ${results['cost_per_incremental_success']:.2f}")

# A/B test simulation
ab_results = simulator.simulate_ab_test(
    population_df=population,
    control_policy='reactive',
    treatment_policy='proactive',
    treatment_fraction=0.5,
    test_duration=60
)

print(f"\nA/B Test Results:")
print(f"Uplift: {ab_results['uplift']:.2%}")
print(f"P-value: {ab_results['p_value']:.4f}")
print(f"Significant: {ab_results['is_significant']}")
print(f"95% CI: [{ab_results['confidence_interval'][0]:.2%}, "
      f"{ab_results['confidence_interval'][1]:.2%}]")

# Optimize channel mix
optimal = simulator.optimize_channel_mix(
    population_df=population,
    available_channels=['email', 'sms', 'call'],
    budget=50000
)

print(f"\nOptimal Channel Allocation:")
for channel, alloc in optimal['allocation'].items():
    print(f"  {channel}: {alloc['contacts']:,} contacts, ${alloc['cost']:,.2f}")
print(f"Expected Successes: {optimal['total_expected_successes']:,.0f}")
print(f"Cost per Success: ${optimal['overall_cost_per_success']:.2f}")

# Visualize comparison
fig = simulator.plot_policy_comparison(results)
```

#### Custom Policies

```python
from src.simulation.policy_simulator import PolicyConfig

# Define custom policy
custom_policy = PolicyConfig(
    name='custom_aggressive',
    target_segments=['high_value'],
    channels=['call', 'sms', 'email'],
    contact_frequency=7,
    max_contacts=8,
    prioritization_rule='balance',
    treatment_effect=0.22
)

simulator.policies['custom_aggressive'] = custom_policy
```

### 4. Stress Testing (`stress_testing.py`)

Comprehensive stress testing for credit risk and regulatory compliance.

#### Key Features
- Economic stress scenarios (recession, crisis, stagflation)
- PD, LGD, and EAD stress testing
- Value at Risk (VaR) and Conditional VaR (CVaR) calculations
- Vintage cohort analysis
- Segment-level impact assessment
- Reverse stress testing
- CCAR/DFAST-style regulatory scenarios
- Sensitivity analysis
- Comprehensive reporting

#### Usage Example

```python
from src.simulation.stress_testing import StressTestEngine
import pandas as pd

# Initialize engine
engine = StressTestEngine(
    confidence_level=0.99,
    time_horizon=12,
    n_simulations=10000,
    random_seed=42
)

# Prepare portfolio
portfolio = pd.DataFrame({
    'loan_id': range(10000),
    'pd': np.random.beta(2, 18, 10000),  # Mean ~0.10
    'lgd': np.random.beta(4, 6, 10000),   # Mean ~0.40
    'ead': np.random.lognormal(10, 1, 10000),
    'vintage': np.random.choice(['2020', '2021', '2022'], 10000),
    'segment': np.random.choice(['Prime', 'Subprime'], 10000)
})

# Run stress test
results = engine.run_stress_test(
    portfolio_df=portfolio,
    scenario='severe_recession',
    pd_col='pd',
    lgd_col='lgd',
    ead_col='ead',
    segment_col='segment'
)

print(f"\nStress Test Results:")
print(f"Scenario: {results['scenario'].name}")
print(f"Baseline EL: ${results['portfolio_metrics']['baseline_el']:,.0f}")
print(f"Stressed EL: ${results['portfolio_metrics']['stressed_el']:,.0f}")
print(f"EL Increase: {results['portfolio_metrics']['el_increase_pct']:.2%}")
print(f"VaR (99%): ${results['risk_metrics']['var_stressed']:,.0f}")
print(f"CVaR (99%): ${results['risk_metrics']['cvar_stressed']:,.0f}")

# Segment analysis
print("\nSegment Impact:")
print(results['segment_analysis'])

# Compare scenarios
comparison = engine.compare_scenarios(
    portfolio_df=portfolio,
    scenarios=['baseline', 'moderate_recession', 'severe_recession', 'financial_crisis']
)
print("\nScenario Comparison:")
print(comparison)

# Vintage analysis
vintage_results = engine.run_vintage_stress_test(
    portfolio_df=portfolio,
    vintage_col='vintage',
    scenario='severe_recession'
)
print("\nVintage Analysis:")
print(vintage_results)

# Reverse stress test
reverse = engine.reverse_stress_test(
    portfolio_df=portfolio,
    target_el_increase=2.0  # Find scenario for 200% EL increase
)
print(f"\nReverse Stress Test:")
print(f"Required PD Multiplier: {reverse['scenario'].pd_multiplier:.2f}")

# Sensitivity analysis
sensitivity = engine.sensitivity_analysis(
    portfolio_df=portfolio,
    base_scenario='moderate_recession',
    parameter='pd_multiplier',
    parameter_range=np.linspace(1.0, 3.0, 10)
)
print("\nSensitivity Analysis:")
print(sensitivity)

# Visualizations
fig = engine.plot_stress_comparison(comparison)
fig = engine.plot_distribution_comparison(results)

# Generate report
engine.generate_report(results, 'stress_test_report.json')
```

#### Custom Stress Scenarios

```python
from src.simulation.stress_testing import StressScenario

custom_scenario = StressScenario(
    name='Pandemic Scenario',
    description='Severe but short-lived economic shock',
    pd_multiplier=3.0,
    lgd_multiplier=1.3,
    ead_multiplier=1.2,
    recovery_rate_adjustment=-0.15,
    unemployment_rate=14.0,
    gdp_growth=-8.0,
    house_price_change=-5.0,
    correlation_increase=0.40,
    duration_months=18
)

engine.scenarios['pandemic'] = custom_scenario
```

## Integration Examples

### Combined Workflow Example

```python
from src.simulation import (
    MonteCarloSimulator,
    MarkovChainModel,
    PolicySimulator,
    StressTestEngine
)
import pandas as pd

# Load portfolio
portfolio = pd.read_csv('portfolio.csv')

# 1. Monte Carlo simulation for NPV analysis
mc_sim = MonteCarloSimulator(n_simulations=10000, random_seed=42)
npv_results = mc_sim.simulate_portfolio(
    portfolio_df=portfolio,
    principal_col='balance',
    scenario='baseline'
)

# 2. Markov chain for delinquency progression
markov = MarkovChainModel(
    states=['Current', 'DPD30', 'DPD60', 'DPD90', 'ChargeOff'],
    absorbing_states=['ChargeOff']
)
markov.fit(historical_data, state_col='status', id_col='account_id')
future_defaults = markov.predict_distribution('Current', n_steps=12)

# 3. Policy simulation for strategy optimization
policy_sim = PolicySimulator(random_seed=42)
policy_impact = policy_sim.simulate_policy_impact(
    population_df=portfolio,
    baseline_policy='reactive',
    new_policy='proactive'
)

# 4. Stress testing for risk assessment
stress = StressTestEngine(random_seed=42)
stress_results = stress.run_stress_test(
    portfolio_df=portfolio,
    scenario='severe_recession',
    segment_col='risk_tier'
)

# Combine insights
report = {
    'npv_analysis': npv_results,
    'delinquency_forecast': future_defaults,
    'policy_recommendation': policy_impact,
    'stress_assessment': stress_results
}
```

## Best Practices

### 1. Simulation Configuration

- **Sample Size**: Use 10,000+ simulations for production analysis
- **Random Seeds**: Always set random seeds for reproducibility
- **Validation**: Compare simulation results with historical data
- **Scenarios**: Test multiple scenarios to understand uncertainty

### 2. Markov Chain Modeling

- **State Definition**: Define states based on business logic
- **Data Requirements**: Ensure sufficient observations per transition
- **Validation**: Check if Markov assumption holds (test for memory)
- **Smoothing**: Use Laplace smoothing for rare transitions

### 3. Policy Simulation

- **Sample Size**: Ensure adequate sample size for statistical power
- **A/B Testing**: Use stratified sampling for balanced groups
- **Capacity**: Model realistic capacity constraints
- **Costs**: Include all relevant costs (contact + operational)

### 4. Stress Testing

- **Scenarios**: Include both regulatory and custom scenarios
- **Calibration**: Calibrate scenarios to historical crises
- **Segments**: Analyze impact by vintage and segment
- **Documentation**: Generate comprehensive reports for audits

## Performance Optimization

### Vectorization

All modules use NumPy vectorization for performance:

```python
# Efficient: Vectorized operations
stressed_pd = baseline_pd * scenario.pd_multiplier

# Inefficient: Loop-based operations
stressed_pd = np.array([pd * scenario.pd_multiplier for pd in baseline_pd])
```

### Parallel Processing

For large portfolios, consider parallel processing:

```python
from joblib import Parallel, delayed

# Simulate loans in parallel
results = Parallel(n_jobs=-1)(
    delayed(simulator.simulate_repayments)(
        principal=row['balance'],
        scenario='baseline'
    )
    for _, row in portfolio.iterrows()
)
```

## Validation and Testing

### Unit Tests

```python
import pytest
from src.simulation import MonteCarloSimulator

def test_monte_carlo_confidence_intervals():
    sim = MonteCarloSimulator(n_simulations=1000, random_seed=42)
    results = sim.simulate_repayments(principal=100000, scenario='baseline')
    
    ci = results['confidence_intervals']
    assert ci['p5'] < ci['p50'] < ci['p95']
    assert results['expected_npv'] > 0
```

### Backtesting

```python
# Validate Markov chain predictions
def backtest_markov(model, test_data):
    predictions = []
    actuals = []
    
    for account_id, group in test_data.groupby('account_id'):
        initial_state = group.iloc[0]['status']
        actual_state = group.iloc[-1]['status']
        
        predicted_dist = model.predict_distribution(initial_state, n_steps=len(group)-1)
        predictions.append(predicted_dist)
        actuals.append(actual_state)
    
    # Calculate prediction accuracy
    return accuracy_score(actuals, predictions)
```

## Troubleshooting

### Common Issues

1. **Memory Error**: Reduce `n_simulations` or process portfolio in batches
2. **Singular Matrix**: Increase smoothing parameter in Markov chains
3. **Slow Performance**: Use vectorized operations, reduce sample size
4. **Unrealistic Results**: Validate input parameters and scenario calibration

## References

- Monte Carlo Methods: Robert & Casella (2004)
- Markov Chains: Norris (1997)
- Stress Testing: Basel Committee on Banking Supervision
- A/B Testing: Kohavi & Longbotham (2017)

## Support

For questions or issues, please refer to the main project documentation or contact the development team.
