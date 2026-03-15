# Validation Framework for Principal Data Science Decision Agent

## Overview

This validation framework provides comprehensive model validation capabilities following industry best practices and regulatory requirements. It implements state-of-the-art techniques for ensuring model stability, fairness, calibration, and governance compliance.

## Modules

### 1. **stability_testing.py** - Model Stability Testing

Monitors population and characteristic stability using PSI (Population Stability Index) and CSI (Characteristic Stability Index).

**Key Features:**
- PSI calculation with configurable thresholds
- Feature-level CSI analysis
- Segment-based stability testing
- Traffic light warning system (green/yellow/red)
- Time series stability tracking
- Comprehensive reporting and visualization

**Usage:**
```python
from validation.stability_testing import StabilityTester

# Initialize tester
tester = StabilityTester(
    n_bins=10,
    psi_threshold_yellow=0.1,
    psi_threshold_red=0.25
)

# Analyze stability
result = tester.analyze_stability(
    baseline_scores=train_predictions,
    production_scores=prod_predictions,
    baseline_features=X_train,
    production_features=X_prod,
    segments={
        'baseline': train_segments_df,
        'production': prod_segments_df
    }
)

# Generate report
tester.generate_report(result, 'stability_report.html')
```

**Interpretation:**
- **PSI < 0.1:** No significant shift (GREEN)
- **0.1 ≤ PSI < 0.25:** Moderate shift, investigate (YELLOW)
- **PSI ≥ 0.25:** Significant shift, retrain recommended (RED)

---

### 2. **adversarial_validation.py** - Train/Test Distribution Validation

Detects distribution differences, temporal drift, and data leakage using adversarial validation.

**Key Features:**
- Train vs. test discriminator using Random Forest
- AUC-based distribution similarity scoring
- Feature importance for drift attribution
- Temporal validation support
- Sampling bias detection
- Actionable mitigation recommendations

**Usage:**
```python
from validation.adversarial_validation import AdversarialValidator

# Initialize validator
validator = AdversarialValidator(
    n_estimators=100,
    auc_threshold=0.85,
    auc_critical=0.95
)

# Validate distributions
result = validator.validate(
    train_data=X_train,
    test_data=X_test,
    categorical_features=['category', 'region']
)

# Check for issues
if result.status == 'fail':
    print(f"Critical: AUC={result.auc_score:.3f}")
    print(f"Top drift features: {result.drift_features[:5]}")

# Temporal validation
temporal_result = validator.validate_temporal(
    data=df,
    time_column='date',
    train_end_date='2023-12-31',
    test_start_date='2024-01-01'
)
```

**Interpretation:**
- **AUC ≈ 0.5:** Similar distributions (PASS)
- **0.85 ≤ AUC < 0.95:** Moderate shift (WARNING)
- **AUC ≥ 0.95:** Severe shift or leakage (FAIL)

---

### 3. **drift_monitor.py** - Production Drift Monitoring

Real-time monitoring of data drift and concept drift in production.

**Key Features:**
- Data drift detection (KS test, Chi-square test)
- Concept drift detection (target distribution changes)
- Feature-level drift scoring
- Alert generation with severity levels
- Historical drift tracking
- Distribution comparison visualizations

**Usage:**
```python
from validation.drift_monitor import DriftMonitor

# Initialize monitor
monitor = DriftMonitor(
    reference_data=train_df,
    alert_threshold=0.05,
    critical_threshold=0.01
)

# Detect drift
drift_report = monitor.detect_drift(
    production_data=prod_df,
    features=['feature1', 'feature2'],
    categorical_features=['category'],
    target_col='target',
    production_target=prod_target
)

# Check alerts
if drift_report.has_drift:
    print(f"Alert! {len(drift_report.drifted_features)} features drifting")
    monitor.send_alert(drift_report)

# Visualize
monitor.visualize_drift(drift_report, output_path='drift_viz.png')
```

**Key Metrics:**
- **P-value < 0.05:** Significant drift detected
- **P-value < 0.01:** Critical drift
- **Severity Levels:** low, medium, high, critical

---

### 4. **calibration_validator.py** - Model Calibration Validation

Validates probabilistic model calibration with multiple statistical tests.

**Key Features:**
- Hosmer-Lemeshow goodness-of-fit test
- Brier score calculation
- Expected Calibration Error (ECE)
- Maximum Calibration Error (MCE)
- Calibration curves and reliability diagrams
- Segment-level calibration analysis
- Conformal prediction intervals

**Usage:**
```python
from validation.calibration_validator import CalibrationValidator

# Initialize validator
validator = CalibrationValidator(
    n_bins=10,
    ece_threshold=0.1
)

# Validate calibration
result = validator.validate_calibration(
    y_true=y_test,
    y_pred_proba=y_pred_proba
)

# Check calibration
if not result.is_well_calibrated:
    print("Model is poorly calibrated")
    print(f"ECE: {result.expected_calibration_error:.4f}")
    print("Recommendations:", result.recommendations)

# Visualize
validator.plot_calibration_curve(result, 'calibration.png')
validator.plot_calibration_by_decile(y_true, y_pred_proba, 'deciles.png')
```

**Interpretation:**
- **Brier Score:** < 0.1 excellent, 0.1-0.25 acceptable
- **ECE:** < 0.1 well-calibrated
- **Hosmer-Lemeshow p-value:** > 0.05 good calibration

---

### 5. **governance_report.py** - Regulatory Reporting & Model Cards

Comprehensive model governance, documentation, and regulatory compliance.

**Key Features:**
- Google-style model cards
- Fairness analysis (demographic parity, equal opportunity)
- Bias detection across protected groups
- Explainability summaries (SHAP, feature importance)
- Regulatory compliance tracking (SR 11-7, GDPR, FCRA, ECOA)
- Audit trail generation
- Version control and model lineage

**Usage:**
```python
from validation.governance_report import GovernanceReporter

# Initialize reporter
reporter = GovernanceReporter(
    model_name='Credit Risk Model',
    model_version='2.1',
    regulatory_framework=['SR 11-7', 'ECOA', 'FCRA']
)

# Generate model card
model_card = reporter.generate_model_card(
    model_details={
        'type': 'XGBoost',
        'architecture': 'Gradient Boosting',
        'training_data': '500K loans, 2020-2023',
        'owner': 'Data Science Team',
        'contact': 'ds@company.com'
    },
    intended_use={
        'description': 'Credit risk assessment for consumer loans',
        'users': ['Credit Officers', 'Risk Management'],
        'out_of_scope': ['Business loans', 'Mortgage decisions']
    },
    performance_metrics={
        'auc': 0.85,
        'accuracy': 0.82,
        'precision': 0.78
    }
)

# Analyze fairness
fairness = reporter.analyze_fairness(
    y_true=y_test,
    y_pred=y_pred,
    protected_attribute=age_group  # 0: young, 1: old
)

print(f"Demographic Parity: {fairness['demographic_parity']:.3f}")
print(f"Equal Opportunity: {fairness['equal_opportunity']:.3f}")
print(f"Assessment: {fairness['fairness_assessment']}")

# Save model card
reporter.save_model_card(model_card, 'model_card.html')
reporter.save_model_card(model_card, 'model_card.json', format='json')

# Audit logging
reporter.log_audit('model_deployment', 'john@company.com', 
                  {'environment': 'production', 'version': '2.1'})
```

**Fairness Metrics:**
- **Demographic Parity:** P(Ŷ=1|A=0) / P(Ŷ=1|A=1)
- **Equal Opportunity:** TPR parity across groups
- **Equalized Odds:** TPR and FPR parity
- **80% Rule:** Ratio should be 0.8-1.25 for fairness

---

## Regulatory Context

### SR 11-7: Guidance on Model Risk Management (Federal Reserve)
- **Requirements:**
  - Comprehensive model documentation
  - Independent model validation
  - Ongoing performance monitoring
  - Model inventory maintenance

- **Framework Support:**
  - Model cards for documentation
  - Stability and drift monitoring for ongoing validation
  - Audit trails for governance
  - Performance reporting by segment

### GDPR: General Data Protection Regulation
- **Requirements:**
  - Right to explanation for automated decisions
  - Data protection impact assessments
  - Privacy by design

- **Framework Support:**
  - Explainability summaries (SHAP, feature importance)
  - Bias detection and fairness analysis
  - Audit trails for accountability

### FCRA: Fair Credit Reporting Act
- **Requirements:**
  - Adverse action notices
  - Model interpretability for credit decisions

- **Framework Support:**
  - Calibration validation for probability accuracy
  - Feature importance for explainability
  - Segment-level performance analysis

### ECOA: Equal Credit Opportunity Act
- **Requirements:**
  - Fair lending analysis
  - Disparate impact testing
  - Prohibition of discrimination

- **Framework Support:**
  - Comprehensive fairness metrics
  - Bias detection across protected groups
  - Segment-level stability and performance monitoring

---

## Best Practices

### 1. Model Development Phase

**Adversarial Validation:**
```python
# Check for data leakage before training
validator = AdversarialValidator()
result = validator.validate(X_train, X_test)

if result.auc_score > 0.85:
    print("WARNING: Potential data leakage!")
    print("Top features:", result.drift_features[:5])
```

**Calibration Validation:**
```python
# Validate calibration on hold-out set
cal_validator = CalibrationValidator()
result = cal_validator.validate_calibration(y_val, y_pred_proba_val)

if not result.is_well_calibrated:
    # Apply calibration
    from sklearn.calibration import CalibratedClassifierCV
    calibrated_model = CalibratedClassifierCV(model, method='isotonic')
```

### 2. Pre-Deployment Phase

**Stability Testing:**
```python
# Test stability on OOT (Out-of-Time) data
tester = StabilityTester()
result = tester.analyze_stability(
    baseline_scores=train_scores,
    production_scores=oot_scores,
    baseline_features=X_train,
    production_features=X_oot
)

if result.traffic_light != 'green':
    print("WARNING: Population shift detected")
```

**Fairness Analysis:**
```python
# Validate fairness before deployment
reporter = GovernanceReporter('Model v2.0')
fairness = reporter.analyze_fairness(y_test, y_pred, protected_attr)

if fairness['fairness_assessment'] != 'fair':
    print("ALERT: Bias detected!")
    # Consider model adjustments or constraints
```

### 3. Production Monitoring

**Continuous Drift Monitoring:**
```python
# Set up production monitoring
monitor = DriftMonitor(reference_data=train_df)

# Weekly monitoring
drift_report = monitor.detect_drift(
    production_data=weekly_prod_df,
    target_col='actual_outcome',
    production_target=weekly_outcomes
)

if drift_report.has_drift or drift_report.concept_drift:
    # Trigger retraining pipeline
    monitor.send_alert(drift_report, alert_callback=trigger_retraining)
```

**Calibration Monitoring:**
```python
# Monthly calibration checks
result = cal_validator.validate_calibration(
    monthly_actuals,
    monthly_predictions
)

if result.brier_score > 0.25:
    # Recalibrate model
    print("Model requires recalibration")
```

### 4. Model Governance

**Model Cards:**
```python
# Maintain comprehensive documentation
reporter = GovernanceReporter('Production Model')
model_card = reporter.generate_model_card(...)

# Update with each version
model_card.previous_version = '1.9'
model_card.changes_from_previous = [
    'Added new features: payment_history, credit_utilization',
    'Improved fairness metrics by 15%',
    'Reduced false positive rate from 12% to 8%'
]

reporter.save_model_card(model_card, 'model_card_v2.0.html')
```

**Audit Trails:**
```python
# Log all model lifecycle events
reporter.log_audit('model_training', 'data_scientist@company.com',
                  {'dataset_version': 'v3', 'samples': 500000})
reporter.log_audit('model_validation', 'validator@company.com',
                  {'auc': 0.85, 'status': 'approved'})
reporter.log_audit('model_deployment', 'mlops@company.com',
                  {'environment': 'production', 'replicas': 3})
```

---

## Integration with Existing Framework

### With Models Module
```python
from models.ensemble_engine import EnsembleEngine
from validation.stability_testing import StabilityTester

# Train ensemble
ensemble = EnsembleEngine()
ensemble.fit(X_train, y_train)
predictions = ensemble.predict_proba(X_test)

# Validate stability
tester = StabilityTester()
result = tester.analyze_stability(
    baseline_scores=ensemble.predict_proba(X_train)[:, 1],
    production_scores=predictions[:, 1]
)
```

### With Use Cases
```python
from use_cases.credit_risk import CreditRiskModel
from validation.governance_report import GovernanceReporter

# Credit risk model with governance
model = CreditRiskModel()
model.train(X_train, y_train)

# Generate compliance documentation
reporter = GovernanceReporter('Credit Risk Model',
                             regulatory_framework=['SR 11-7', 'ECOA'])
fairness = reporter.analyze_fairness(y_test, model.predict(X_test),
                                    protected_attribute=age_group)

model_card = reporter.generate_model_card(
    model_details={'type': 'Ensemble', 'features': model.feature_names},
    intended_use={'description': 'Consumer credit risk assessment'},
    performance_metrics=model.get_metrics(),
    fairness_metrics=fairness
)
```

---

## Statistical Tests Reference

### Kolmogorov-Smirnov Test (Numerical Drift)
- **Null Hypothesis:** Distributions are identical
- **Test Statistic:** Maximum distance between CDFs
- **Interpretation:** p < 0.05 indicates significant drift

### Chi-Square Test (Categorical Drift)
- **Null Hypothesis:** Distributions are independent
- **Test Statistic:** χ² from contingency table
- **Interpretation:** p < 0.05 indicates significant association change

### Hosmer-Lemeshow Test (Calibration)
- **Null Hypothesis:** Model is well-calibrated
- **Test Statistic:** χ² across probability bins
- **Interpretation:** p > 0.05 indicates good calibration

---

## Visualization Capabilities

All modules include rich visualization:

1. **Stability Testing:**
   - PSI trends over time
   - Segment comparison bar charts
   - Distribution overlays

2. **Adversarial Validation:**
   - ROC curves
   - Feature importance plots
   - Distribution comparisons

3. **Drift Monitoring:**
   - Drift score heatmaps
   - Time series drift tracking
   - Feature distribution Q-Q plots

4. **Calibration:**
   - Calibration curves (reliability diagrams)
   - Decile analysis
   - Perfect calibration reference lines

---

## Output Formats

All reports support multiple formats:

- **HTML:** Rich, interactive reports with styling
- **JSON:** Machine-readable for integration
- **Markdown:** Version-control friendly documentation

```python
# HTML for stakeholders
reporter.save_model_card(card, 'report.html', format='html')

# JSON for APIs/databases
reporter.save_model_card(card, 'report.json', format='json')

# Markdown for Git
reporter.save_model_card(card, 'report.md', format='markdown')
```

---

## Performance Considerations

- **Minimum Sample Sizes:**
  - PSI calculation: 1000+ samples per segment
  - Calibration testing: 100+ samples per bin
  - Fairness analysis: 500+ samples per group

- **Computational Efficiency:**
  - Adversarial validation uses n_jobs=-1 for parallelization
  - Statistical tests optimized for large datasets
  - Visualization caching for repeated generation

---

## Error Handling

All modules include comprehensive error handling:

```python
try:
    result = validator.validate_calibration(y_true, y_pred)
except ValueError as e:
    logger.error(f"Validation failed: {e}")
    # Graceful degradation or fallback
```

---

## Logging

Uses `loguru` for structured logging:

```python
from loguru import logger

# Configure logging
logger.add("validation_{time}.log", rotation="1 day")

# Logs automatically generated by all modules
# INFO: Validation started
# DEBUG: Calculated PSI: 0.0842
# WARNING: Drift detected in feature 'age'
# ERROR: Insufficient samples for calibration test
```

---

## Testing

Comprehensive test coverage:

```bash
pytest tests/validation/test_stability_testing.py
pytest tests/validation/test_adversarial_validation.py
pytest tests/validation/test_drift_monitor.py
pytest tests/validation/test_calibration_validator.py
pytest tests/validation/test_governance_report.py
```

---

## References

1. **Model Cards:** Mitchell et al., "Model Cards for Model Reporting", FAT* 2019
2. **SR 11-7:** Federal Reserve Guidance on Model Risk Management
3. **Fairness Metrics:** Mehrabi et al., "A Survey on Bias and Fairness in Machine Learning", 2021
4. **PSI:** Siddiqi, "Credit Risk Scorecards", Wiley 2006
5. **Calibration:** Guo et al., "On Calibration of Modern Neural Networks", ICML 2017

---

## License

See main repository LICENSE file.

## Support

For questions or issues, contact the Data Science team or open an issue in the repository.
