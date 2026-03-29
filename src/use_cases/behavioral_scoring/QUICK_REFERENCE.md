# Behavioral Scoring Quick Reference

## Installation

```bash
# Core dependencies
pip install numpy pandas scikit-learn scipy loguru joblib

# Optional: Deep Learning
pip install torch  # or tensorflow

# Optional: Advanced Models
pip install lightgbm xgboost catboost

# Optional: Interpretability
pip install shap
```

## Quick Start

### 1. Simple Ensemble Scoring

```python
import numpy as np
from src.use_cases.behavioral_scoring import BehavioralEnsembleScorer

# Prepare data
X_train = np.random.randn(1000, 20)
y_train = np.random.randint(0, 2, 1000)

# Train ensemble
ensemble = BehavioralEnsembleScorer(ensemble_type="stacking")
ensemble.fit(X_train, y_train)

# Predict
X_test = np.random.randn(100, 20)
scores = ensemble.predict(X_test)
```

### 2. Meta-Learning for Segments

```python
from src.use_cases.behavioral_scoring import MetaScoringEngine

# Prepare segment data
X_dict = {f'user_{i}': features[i] for i in range(100)}
y_dict = {f'user_{i}': labels[i] for i in range(100)}
seg_dict = {f'user_{i}': segment_ids[i] for i in range(100)}

# Train meta-learner
meta_engine = MetaScoringEngine(meta_learning_strategy="transfer")
meta_engine.fit(X_dict, y_dict, seg_dict)

# Few-shot adaptation to new segment
meta_engine.adapt_to_segment(X_new, y_new, "new_segment", n_shots=10)
scores = meta_engine.predict(X_test, segment_id="new_segment")
```

### 3. Deep Learning on Sequences

```python
from src.use_cases.behavioral_scoring import LSTMScoringModel

# Prepare sequences (list of variable-length arrays)
sequences = [np.random.randn(50, 10), np.random.randn(30, 10), ...]
labels = np.array([0, 1, ...])

# Train LSTM
model = LSTMScoringModel(input_size=10, hidden_size=128)
history = model.fit(sequences, labels, epochs=20)

# Predict
new_sequences = [np.random.randn(40, 10), ...]
scores = model.predict(new_sequences)
```

### 4. Complete Pipeline

```python
import pandas as pd
from src.use_cases.behavioral_scoring import BehavioralScoringPipeline

# Prepare DataFrames
transactions = pd.DataFrame({
    'user_id': [...],
    'timestamp': [...],
    'amount': [...]
})

labels = pd.DataFrame({
    'user_id': [...],
    'label': [...]
})

# Initialize and train
pipeline = BehavioralScoringPipeline(
    feature_config="config/feature_config.yaml",
    use_deep_learning=True,
    ensemble_type="stacking"
)

history = pipeline.fit(
    transactions_df=transactions,
    labels_df=labels,
    validation_strategy="temporal"
)

# Real-time scoring
score = pipeline.score_single(
    user_id="12345",
    transaction_data=user_transactions
)

# Batch scoring
scores_df = pipeline.score_batch(test_transactions)

# Save/Load
pipeline.save("models/scoring_pipeline")
loaded_pipeline = BehavioralScoringPipeline.load("models/scoring_pipeline")
```

## Common Patterns

### Pattern 1: Segment-Aware Scoring

```python
# Train with segments
ensemble = BehavioralEnsembleScorer(
    ensemble_type="stacking",
    segment_aware=True
)
ensemble.fit(X_train, y_train, segments=train_segments)

# Score with segments
scores = ensemble.predict(X_test, segments=test_segments)

# Evaluate per segment
metrics = ensemble.evaluate(X_test, y_test, segments=test_segments)
# Returns segment-specific metrics
```

### Pattern 2: Model Monitoring

```python
from src.use_cases.behavioral_scoring import PerformanceMonitor

monitor = PerformanceMonitor()

# Log metrics periodically
for batch in new_data_batches:
    scores = model.predict(batch['X'])
    monitor.log_metrics(batch['y'], scores, segment=batch['segment'])

# Check for degradation
if monitor.detect_performance_degradation():
    print("Time to retrain!")
    model.fit(updated_data, updated_labels)
```

### Pattern 3: Temporal Validation

```python
from src.use_cases.behavioral_scoring import TemporalValidator

validator = TemporalValidator(n_splits=5, test_size_days=30)

for train_idx, test_idx in validator.split(data_df, timestamp_col='date'):
    train_data = data_df.iloc[train_idx]
    test_data = data_df.iloc[test_idx]
    
    model.fit(train_data['X'], train_data['y'])
    scores = model.predict(test_data['X'])
    
    auc = roc_auc_score(test_data['y'], scores)
    print(f"Fold AUC: {auc:.4f}")
```

### Pattern 4: Ensemble Calibration

```python
# Train with calibration
ensemble = BehavioralEnsembleScorer(
    ensemble_type="stacking",
    calibration_method="isotonic"
)
ensemble.fit(X_train, y_train)

# Predictions are automatically calibrated
scores = ensemble.predict(X_test)  # Well-calibrated probabilities

# Manual calibration
raw_scores = model.predict(X_test)
calibrated = ensemble.calibrate_scores(raw_scores, y_true)
```

### Pattern 5: Model Interpretation

```python
from src.use_cases.behavioral_scoring import DeepScoringInterpreter

# For deep learning models
interpreter = DeepScoringInterpreter(lstm_model)

# Feature importance
importance = interpreter.feature_importance(X_test, method="gradient")

# Attention visualization (for Transformer)
attention = transformer_model.get_attention_weights(sequences)
import matplotlib.pyplot as plt
plt.imshow(attention[0], cmap='hot')
plt.title("Transaction Attention")
```

## Configuration Examples

### Feature Config (config/feature_config.yaml)

```yaml
behavioral:
  velocity_windows: [7, 14, 30, 60, 90]
  momentum_windows: [7, 14, 30]
  volatility_windows: [30, 60, 90]
  stability_windows: [60, 90, 180]

temporal:
  rolling_windows:
    short_term: [7, 14]
    medium_term: [30, 60]
    long_term: [90, 180]
  lag_features: [1, 7, 14, 30]

persona:
  clustering_method: "hdbscan"
  n_clusters: 5
```

## Performance Tips

### 1. Speed Optimization

```python
# Disable deep learning for faster training
pipeline = BehavioralScoringPipeline(use_deep_learning=False)

# Use fewer base models
ensemble = BehavioralEnsembleScorer(
    base_models=["rf", "logistic"]  # Instead of all models
)

# Reduce sequence length
preprocessor = SequencePreprocessor(max_sequence_length=30)  # Default is 100
```

### 2. Memory Optimization

```python
# Batch scoring
batch_size = 1000
for i in range(0, len(data), batch_size):
    batch = data.iloc[i:i+batch_size]
    scores = pipeline.score_batch(batch)
    save_scores(scores)  # Save and clear memory
```

### 3. Accuracy Optimization

```python
# Use more diverse models
ensemble = BehavioralEnsembleScorer(
    base_models=["lgbm", "xgb", "catboost", "rf", "gb"],
    ensemble_type="stacking"
)

# Enable calibration
ensemble = BehavioralEnsembleScorer(
    calibration_method="isotonic"
)

# Use segment-specific models
ensemble = BehavioralEnsembleScorer(segment_aware=True)
```

## Troubleshooting

### Issue: "PyTorch not available"
**Solution:** Install PyTorch or disable deep learning:
```python
pipeline = BehavioralScoringPipeline(use_deep_learning=False)
```

### Issue: "Feature modules not available"
**Solution:** System will use basic aggregations automatically. No action needed.

### Issue: "Memory error with large datasets"
**Solution:** Use batch processing:
```python
scores_df = pipeline.score_batch(data.iloc[:10000])  # Process in chunks
```

### Issue: "Model performance degraded"
**Solution:** Retrain with recent data:
```python
if monitor.detect_performance_degradation():
    pipeline.fit(new_transactions, new_labels)
```

## Key Metrics

- **AUC-ROC**: Overall discriminative power (target: >0.75)
- **Average Precision**: Precision-recall trade-off (target: >0.3 for 10% base rate)
- **Brier Score**: Calibration quality (lower is better, target: <0.15)
- **Diversity Score**: Ensemble diversity (target: >0.3)

## Best Practices

1. **Always use temporal validation** for time-series data
2. **Calibrate scores** for probability interpretation
3. **Monitor performance** regularly (weekly/monthly)
4. **Use segments** when customer behavior differs significantly
5. **Start simple** (ensemble) then add complexity (meta-learning, deep learning)
6. **Save models** after training for reproducibility
7. **Document assumptions** especially for label alignment

## API Reference Summary

### Core Classes

- `BehavioralEnsembleScorer`: Main ensemble scoring engine
- `MetaScoringEngine`: Meta-learning for segments
- `LSTMScoringModel`: Deep learning on sequences
- `BehavioralScoringPipeline`: End-to-end orchestrator

### Utilities

- `TransactionPreprocessor`: Data cleaning
- `TemporalValidator`: Time-aware validation
- `PerformanceMonitor`: Model monitoring
- `SequencePreprocessor`: Sequence preparation
- `EnsembleCalibrator`: Probability calibration

### Key Methods

- `.fit(X, y)`: Train model
- `.predict(X)`: Get scores
- `.evaluate(X, y)`: Compute metrics
- `.save(path)`: Persist model
- `.load(path)`: Load model

## Getting Help

1. Check module docstrings: `help(BehavioralEnsembleScorer)`
2. Review README.md for detailed documentation
3. Check examples in each module's `__main__` block
4. Review code comments for implementation details

## Citation

```bibtex
@software{behavioral_scoring_2024,
  title={Behavioral Scoring with Meta-Learning and Deep Learning},
  author={Principal Data Science Decision Agent},
  year={2024}
}
```
