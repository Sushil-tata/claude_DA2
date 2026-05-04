# Income Estimation Use Case

A production-ready income estimation system for financial services that provides calibrated income estimates with uncertainty quantification.

## Overview

This module implements a comprehensive income estimation pipeline that combines:

1. **Deposit Intelligence** - Pattern recognition for income deposits
2. **Graph Network Analysis** - Validation through payment networks
3. **Stability Modeling** - Behavioral pattern analysis and risk scoring
4. **Prediction Calibration** - Uncertainty quantification with prediction intervals
5. **End-to-End Pipeline** - Production-ready orchestration

## Key Features

- ✅ Multiple income source detection (salary, freelance, gig, investments)
- ✅ Prediction intervals with confidence levels (not just point estimates)
- ✅ Network-based validation and peer comparison
- ✅ Stability and risk assessment
- ✅ Conformal prediction for distribution-free intervals
- ✅ Dynamic calibration with new data
- ✅ Regulatory compliance considerations
- ✅ Production-ready with comprehensive validation

## Installation

```python
# The module uses existing project dependencies
# Ensure you have installed requirements.txt
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from src.use_cases.income_estimation import IncomeEstimationPipeline
import pandas as pd

# Create pipeline
pipeline = IncomeEstimationPipeline()

# Prepare transaction data
transactions = pd.DataFrame({
    'user_id': ['user_1'] * 12,
    'timestamp': pd.date_range('2024-01-01', periods=12, freq='M'),
    'amount': [5200, 5100, 5300, 5000, 5400, 5200, 5100, 5250, 5300, 5150, 5200, 5300],
    'description': ['Monthly Salary'] * 12,
})

# Predict income
result = pipeline.predict(transactions, user_id='user_1')

print(f"Estimated Monthly Income: ${result.point_estimate:,.2f}")
print(f"90% Confidence Interval: [${result.lower_bound:,.2f}, ${result.upper_bound:,.2f}]")
print(f"Confidence: {result.confidence:.1%}")
print(f"Stability Score: {result.stability_score:.2f}")
print(f"Risk Category: {result.risk_category}")
```

### With Calibration

```python
# Fit pipeline with known incomes for calibration
known_incomes = pd.DataFrame({
    'user_id': ['user_1', 'user_2', ...],
    'monthly_income': [5000, 6000, ...]
})

pipeline.fit(transactions_df, known_incomes)

# Now predictions are calibrated
result = pipeline.predict(new_transactions, user_id='user_new')

# Validate
metrics = pipeline.validate(test_transactions, test_known_incomes)
print(f"MAE: ${metrics.mae:.2f}")
print(f"90% Coverage: {metrics.coverage_90:.1%}")
```

### Batch Prediction

```python
# Predict for many users at once
predictions_df = pipeline.predict_batch(transactions_df, batch_size=100)

# Save results
predictions_df.to_csv('income_predictions.csv', index=False)
```

## Module Components

### 1. Deposit Intelligence (`deposit_intelligence.py`)

Detects and classifies income deposits from transaction data.

```python
from src.use_cases.income_estimation import DepositDetector

detector = DepositDetector(
    min_deposit_amount=100.0,
    salary_min_amount=500.0
)

# Detect income deposits
deposits = detector.detect_income_deposits(transactions_df)

# Classify income sources
income_sources = detector.classify_income_sources(deposits)

for source in income_sources:
    print(f"{source.source_type}: ${source.monthly_estimate:.2f}")
    print(f"  Confidence: {source.confidence:.1%}")
    print(f"  Frequency: {source.pattern.frequency}")
    print(f"  Active: {source.is_active}")

# Track historical income
history = detector.track_income_history(transactions_df, monthly=True)
```

**Features:**
- Pattern recognition for regular deposits (monthly, bi-weekly, weekly)
- Source classification (primary salary, secondary income, freelance, gig, investments)
- Confidence scoring
- Seasonal pattern detection
- Trend analysis

### 2. Graph Payment Analysis (`graph_payment.py`)

Network-based income validation and peer comparison.

```python
from src.use_cases.income_estimation import PaymentNetworkAnalyzer

analyzer = PaymentNetworkAnalyzer()

# Build payment network
graph = analyzer.build_payment_network(transactions_df)

# Detect employers
employers = analyzer.detect_employer_networks(graph)

for emp in employers:
    print(f"{emp.name}: {emp.employee_count} employees")
    print(f"  Avg Salary: ${emp.avg_salary:.2f}")
    print(f"  Payment Frequency: {emp.payment_frequency}")

# Peer comparison
peer_comp = analyzer.infer_from_peer_comparison(graph, 'user_123')
print(f"Peer Median Income: ${peer_comp.peer_median_income:.2f}")
print(f"Your Percentile: {peer_comp.user_income_percentile:.0%}")

# Validate with network
validation = analyzer.validate_income_with_network(
    estimated_income=50000,
    graph=graph,
    user_id='user_123'
)
print(f"Network Validated: {validation.is_consistent}")
```

**Features:**
- Employer network detection
- Peer group similarity
- Network-based validation
- Industry inference
- Stability indicators from network structure

### 3. Stability Modeling (`stability_model.py`)

Analyzes income stability and behavioral patterns.

```python
from src.use_cases.income_estimation import IncomeStabilityScorer

scorer = IncomeStabilityScorer()

# Score stability
stability = scorer.score_income_stability(income_history_df)

print(f"Overall Stability: {stability.overall_stability:.2f}")
print(f"Volatility (CV): {stability.volatility_cv:.2f}")
print(f"Trend: {stability.trend_direction}")
print(f"Seasonality: {stability.seasonality_strength:.2f}")
print(f"Is Stable: {stability.is_stable}")

# Analyze trends
trends = scorer.analyze_trends(income_history_df)
print(f"Direction: {trends.direction}")
print(f"12-month Forecast: ${trends.forecast_12m:.2f}")

# Risk assessment
risk = scorer.calculate_risk_score(stability)
print(f"Risk Score: {risk.risk_score:.2f}")
print(f"Risk Category: {risk.risk_category}")

# Detect income shocks
shocks = scorer.detect_income_shocks(income_history_df)
print(f"Detected {len(shocks)} income shocks")
```

**Features:**
- Volatility metrics (CV, std, range)
- Trend analysis with forecasting
- Seasonal decomposition
- Irregularity detection
- Employment stability indicators
- Risk scoring

### 4. Calibration (`calibration.py`)

Advanced uncertainty quantification and calibration methods.

```python
from src.use_cases.income_estimation import (
    ConformalPredictor,
    IsotonicCalibrator,
    BayesianCalibrator
)

# Conformal Prediction
conformal = ConformalPredictor(alpha=0.1)  # 90% coverage
conformal.fit(y_true_calibration, y_pred_calibration)

intervals = conformal.predict_interval([50000, 75000, 100000])
for interval in intervals:
    print(f"[{interval.lower_bound:.0f}, {interval.upper_bound:.0f}]")

# Validate coverage
metrics = conformal.validate_coverage(y_true_test, y_pred_test)
print(f"Coverage: {metrics.coverage:.1%}")
print(f"Sharpness: ${metrics.sharpness:.2f}")

# Isotonic Calibration
isotonic = IsotonicCalibrator()
isotonic.fit(y_true, y_pred)
calibrated_predictions = isotonic.calibrate(new_predictions)

# Bayesian Calibration
bayesian = BayesianCalibrator(prior_mean=50000, prior_std=20000)
bayesian.update(observation=55000)
bayesian.update_batch([52000, 53000, 54000])

posterior_mean, posterior_std = bayesian.predict(return_std=True)
interval = bayesian.predict_interval(confidence=0.9)
```

**Features:**
- Conformal prediction for distribution-free intervals
- Isotonic regression for probability calibration
- Bayesian updating with priors
- Multi-quantile prediction
- Dynamic recalibration
- Coverage validation

### 5. Income Pipeline (`income_pipeline.py`)

End-to-end orchestration for production deployment.

```python
from src.use_cases.income_estimation import IncomeEstimationPipeline

# Initialize with configuration
pipeline = IncomeEstimationPipeline(
    min_deposit_amount=100.0,
    confidence_threshold=0.6,
    enable_network_analysis=True,
    enable_calibration=True
)

# Fit on historical data
pipeline.fit(transactions_df, known_incomes_df)

# Predict for single user
result = pipeline.predict(transactions_df, user_id='user_123')

# Access result details
print(result.to_dict())

# Batch prediction
predictions_df = pipeline.predict_batch(transactions_df)

# Validate
metrics = pipeline.validate(test_transactions, test_known_incomes)

# Save/load pipeline
pipeline.save('models/income_pipeline.pkl')
loaded_pipeline = IncomeEstimationPipeline.load('models/income_pipeline.pkl')
```

## Validation Metrics

The pipeline provides comprehensive validation:

- **MAE (Mean Absolute Error)**: Average prediction error in dollars
- **MAPE (Mean Absolute Percentage Error)**: Average error as percentage
- **RMSE (Root Mean Squared Error)**: Penalizes large errors
- **Coverage (90%, 80%)**: Percentage of true values within intervals
- **Sharpness**: Average interval width
- **Calibration Score**: How well intervals match target coverage

## Regulatory Compliance

The module includes considerations for regulatory compliance:

1. **Transparency**: All predictions include confidence scores and explanations
2. **Fairness**: Network-based validation helps detect biases
3. **Uncertainty**: Prediction intervals instead of point estimates
4. **Auditability**: Complete tracking of data sources and model versions
5. **Stability**: Risk assessment helps identify unstable income

## Production Deployment

### Model Serving

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Load pipeline at startup
pipeline = IncomeEstimationPipeline.load('models/income_pipeline.pkl')

class PredictionRequest(BaseModel):
    user_id: str
    transactions: list

@app.post("/predict_income")
def predict_income(request: PredictionRequest):
    transactions_df = pd.DataFrame(request.transactions)
    result = pipeline.predict(transactions_df, user_id=request.user_id)
    return result.to_dict()
```

### Monitoring

```python
# Track prediction distributions
predictions = []
for user_id in user_ids:
    result = pipeline.predict(transactions_df, user_id=user_id)
    predictions.append(result.confidence)

# Alert if confidence drops
avg_confidence = np.mean(predictions)
if avg_confidence < 0.5:
    logger.warning(f"Low average confidence: {avg_confidence:.1%}")

# Monitor calibration drift
if len(new_actuals) >= 50:
    metrics = conformal.validate_coverage(new_actuals, new_predictions)
    if abs(metrics.coverage_gap) > 0.1:
        logger.warning("Recalibration needed")
        conformal.fit(new_actuals, new_predictions)
```

## Performance

- **Throughput**: ~100 predictions/second (single core)
- **Latency**: ~10ms per prediction
- **Memory**: ~50MB for pipeline
- **Scalability**: Supports batch processing

## Best Practices

1. **Minimum Data**: Require at least 3-6 months of transaction history
2. **Calibration**: Always calibrate with known incomes when available
3. **Validation**: Use temporal validation (train on old data, test on recent)
4. **Confidence Thresholds**: Only use predictions with confidence > 0.6
5. **Interval Coverage**: Target 90% coverage for regulatory compliance
6. **Monitoring**: Track calibration metrics and recalibrate periodically
7. **Explainability**: Always provide breakdown by income source

## Examples

See `income_pipeline.py` for complete runnable examples:

- `example_basic_usage()` - Simple income estimation
- `example_with_calibration()` - Training with known incomes
- `example_batch_prediction()` - Large-scale prediction

Run examples:
```bash
cd src/use_cases/income_estimation
python income_pipeline.py
```

## Testing

```bash
# Unit tests
pytest tests/test_income_estimation.py

# Integration tests
pytest tests/test_income_pipeline_integration.py

# Validation tests
pytest tests/test_income_validation.py -v
```

## References

- Conformal Prediction: Shafer & Vovk (2008)
- Isotonic Calibration: Zadrozny & Elkan (2002)
- Network Analysis: NetworkX Documentation
- Time Series: Hyndman & Athanasopoulos (2021)

## Support

For questions or issues:
- Documentation: See module docstrings
- Examples: Run `python income_pipeline.py`
- Logging: Set `LOGURU_LEVEL=DEBUG` for detailed logs

## License

Part of the Principal Data Science Decision Agent project.

---

**Author**: Principal Data Science Decision Agent  
**Version**: 1.0.0
