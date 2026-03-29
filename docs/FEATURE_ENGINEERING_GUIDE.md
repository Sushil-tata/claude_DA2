# Feature Engineering Quick Reference Guide

## Overview
This guide provides quick examples for using each feature engineering module.

## 1. Behavioral Features

```python
from features import BehavioralFeatureEngine
import pandas as pd

# Initialize
engine = BehavioralFeatureEngine(
    config_path="config/feature_config.yaml",
    entity_col="user_id",
    timestamp_col="timestamp",
    value_col="amount"
)

# Sample data
df = pd.DataFrame({
    'user_id': [1, 1, 1, 1],
    'timestamp': pd.date_range('2024-01-01', periods=4, freq='D'),
    'amount': [100, 120, 150, 140]
})

# Compute specific features
velocity_df = engine.compute_velocity_features(df, windows=[7, 14, 30])
momentum_df = engine.compute_momentum_features(df, windows=[7, 14])
volatility_df = engine.compute_volatility_features(df, windows=[30, 60])
stability_df = engine.compute_stability_features(df, windows=[60, 90])

# Or compute all at once
all_features = engine.compute_all_features(df)
```

## 2. Temporal Features

```python
from features import TemporalFeatureEngine

# Initialize
engine = TemporalFeatureEngine(
    config_path="config/feature_config.yaml",
    entity_col="user_id",
    timestamp_col="timestamp",
    value_col="amount"
)

# Compute specific features
rolling_df = engine.compute_rolling_window_features(df)
lag_df = engine.compute_lag_features(df, lags=[1, 7, 14, 30])
multi_res_df = engine.compute_multi_resolution_features(df, resolutions=['daily', 'weekly', 'monthly'])
trend_df = engine.compute_trend_seasonality_features(df, period=7)

# Event-triggered features
df['is_payment'] = [0, 1, 0, 0]
event_df = engine.compute_event_triggered_features(df, event_col='is_payment', event_value=1)

# Compute all
all_features = engine.compute_all_features(df)
```

## 3. Liquidity Features

```python
from features import LiquidityFeatureEngine

# Initialize
engine = LiquidityFeatureEngine(
    config_path="config/feature_config.yaml",
    entity_col="user_id",
    timestamp_col="timestamp"
)

# Sample data with credit info
df = pd.DataFrame({
    'user_id': [1, 1, 1],
    'timestamp': pd.date_range('2024-01-01', periods=3, freq='D'),
    'credit_limit': [10000, 10000, 12000],
    'outstanding_balance': [3000, 4000, 3500],
    'due_date': pd.date_range('2024-01-15', periods=3, freq='M'),
    'payment_amount': [500, 500, 550],
    'emi_amount': [1000, 1000, 1200],
    'income': [5000, 5000, 5500]
})

# Compute specific features
otb_df = engine.compute_otb_features(df)
repay_df = engine.compute_repayment_buffer_features(df, income_col='income')
install_df = engine.compute_installment_lock_features(df, income_col='income')
util_df = engine.compute_utilization_pattern_features(df)

# Compute all
all_features = engine.compute_all_features(df)
```

## 4. Persona Features

```python
from features import PersonaFeatureEngine

# Initialize
engine = PersonaFeatureEngine(
    config_path="config/feature_config.yaml",
    entity_col="user_id",
    timestamp_col="timestamp"
)

# Sample data with categories and merchants
df = pd.DataFrame({
    'user_id': [1, 1, 2, 2],
    'timestamp': pd.date_range('2024-01-01', periods=4, freq='D'),
    'amount': [50, 75, 20, 25],
    'category': ['Food', 'Retail', 'Food', 'Food'],
    'merchant': ['Starbucks', 'Walmart', 'McDonalds', 'Starbucks'],
    'description': ['Coffee at Starbucks', 'Groceries', 'Lunch', 'Coffee']
})

# Compute specific features
desc_df = engine.compute_transaction_description_features(df, description_col='description')
merchant_df = engine.compute_merchant_segmentation_features(df, merchant_col='merchant')
persona_df = engine.compute_behavioral_persona_features(df, category_col='category')
conc_df = engine.compute_category_concentration_features(df, category_col='category')
div_df = engine.compute_merchant_diversity_features(df, merchant_col='merchant')

# Compute all
all_features = engine.compute_all_features(
    df,
    category_col='category',
    merchant_col='merchant',
    description_col='description'
)
```

## 5. Graph Features

```python
from features import GraphFeatureEngine
import networkx as nx

# Initialize
engine = GraphFeatureEngine(
    config_path="config/feature_config.yaml",
    entity_col="user_id"
)

# Build graph from edge list
edges_df = pd.DataFrame({
    'source': [1, 2, 3, 1],
    'target': [2, 3, 4, 4],
    'weight': [1.0, 2.0, 1.5, 0.5]
})

G = engine.build_graph_from_edges(edges_df, weight_col='weight')

# Or use existing NetworkX graph
G = nx.karate_club_graph()

# Compute specific features
centrality_df = engine.compute_centrality_features(G)
community_df = engine.compute_community_features(G)
structural_df = engine.compute_structural_features(G)

# Risk propagation
risk_scores = {0: 1.0, 1: 0.8}  # Initial risk scores
risk_df = engine.compute_risk_propagation_features(G, risk_scores)

# Network statistics
stats = engine.compute_network_statistics(G)

# Compute all (excluding embeddings for speed)
all_features = engine.compute_all_features(G, include_embeddings=False)
```

## 6. Feature Store

```python
from features import FeatureStore
from sklearn.ensemble import RandomForestClassifier

# Initialize
store = FeatureStore(
    config_path="config/feature_config.yaml",
    registry_path="feature_registry.json"
)

# Sample data
df = pd.DataFrame({
    'user_id': range(100),
    'timestamp': pd.date_range('2024-01-01', periods=100, freq='H'),
    'feature1': np.random.randn(100),
    'feature2': np.random.gamma(2, 1, 100),
    'feature3': np.random.choice(['A', 'B', 'C'], 100),
    'target': np.random.binomial(1, 0.3, 100)
})

feature_cols = ['feature1', 'feature2', 'feature3']

# Register features
store.register_features_from_dataframe(df, feature_cols, source="transactions")

# Quality checks
quality_results = store.run_quality_checks(df, feature_cols)
print("Missing rates:", quality_results['missing_rates'])
print("Cardinalities:", quality_results['cardinalities'])
print("Variances:", quality_results['variances'])

# Leakage detection
leakage_results = store.detect_leakage(
    df,
    feature_cols,
    target_col='target',
    timestamp_col='timestamp'
)
print("Temporal leakage:", leakage_results.get('temporal'))
print("Correlation leakage:", leakage_results.get('correlation'))

# Train a model for importance tracking
X = df[feature_cols].fillna(0)
y = df['target']

# Encode categorical
from sklearn.preprocessing import LabelEncoder
for col in X.select_dtypes(include=['object']).columns:
    X[col] = LabelEncoder().fit_transform(X[col])

model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Track importance
importance_results = store.track_importance(model, X, y, feature_cols)
print("Gain importance:", importance_results.get('gain'))
print("Permutation importance:", importance_results.get('permutation'))

# Save registry
store.registry.save_registry()

# Load registry later
store.registry.load_registry()
```

## Configuration Example (feature_config.yaml)

```yaml
# Behavioral Features
behavioral:
  velocity_windows: [7, 14, 30, 60, 90]
  momentum_windows: [7, 14, 30]
  volatility_windows: [30, 60, 90]
  stability_windows: [60, 90, 180]

# Temporal Features
temporal:
  rolling_windows:
    short_term: [7, 14]
    medium_term: [30, 60]
    long_term: [90, 180]
  lag_features: [1, 7, 14, 30]
  lead_features: [7, 14, 30]

# Liquidity Features
liquidity:
  otb_metrics:
    - 'otb_utilization'
    - 'otb_velocity'

# Persona Features
persona:
  clustering:
    method: 'kmeans'
    n_clusters: [5, 10, 15]
  nlp_features:
    enabled: true
    max_features: 1000

# Graph Features
graph:
  centrality_measures:
    - 'degree_centrality'
    - 'betweenness_centrality'
    - 'pagerank'

# Feature Store
feature_store:
  leakage_detection:
    enabled: true
    thresholds:
      correlation_threshold: 0.95
      auc_threshold: 0.85
  quality_checks:
    missing_threshold: 0.5
    cardinality_threshold: 0.95
    variance_threshold: 0.01
```

## Best Practices

1. **Always validate input data** before feature engineering
2. **Use configuration files** for easy parameter tuning
3. **Monitor feature quality** using the feature store
4. **Check for leakage** before training models
5. **Track feature importance** to understand model behavior
6. **Version your features** using the registry
7. **Handle missing values** appropriately for your use case
8. **Test on sample data** before running on full datasets
9. **Use appropriate time windows** based on your domain
10. **Document feature logic** in the registry

## Performance Tips

1. **Filter data** before feature engineering when possible
2. **Use chunking** for very large datasets
3. **Parallelize** independent feature computations
4. **Cache** frequently used features
5. **Profile** to identify bottlenecks
6. **Use vectorized operations** (already done internally)
7. **Limit rolling windows** to necessary periods
8. **Consider sampling** for exploratory analysis

## Common Patterns

### Pattern 1: Complete Feature Pipeline
```python
# Import all engines
from features import (
    BehavioralFeatureEngine,
    TemporalFeatureEngine,
    LiquidityFeatureEngine,
    PersonaFeatureEngine,
    FeatureStore
)

# Initialize
behavioral = BehavioralFeatureEngine()
temporal = TemporalFeatureEngine()
liquidity = LiquidityFeatureEngine()
persona = PersonaFeatureEngine()
store = FeatureStore()

# Compute all features
df = behavioral.compute_all_features(df)
df = temporal.compute_all_features(df)
df = liquidity.compute_all_features(df)
df = persona.compute_all_features(df, category_col='category')

# Register and validate
feature_cols = [col for col in df.columns if col not in ['user_id', 'timestamp', 'target']]
store.register_features_from_dataframe(df, feature_cols)
store.run_quality_checks(df, feature_cols)
store.detect_leakage(df, feature_cols, 'target', 'timestamp')
```

### Pattern 2: Incremental Feature Addition
```python
# Start with base features
features = df.copy()

# Add behavioral features one by one
behavioral = BehavioralFeatureEngine()
features = behavioral.compute_velocity_features(features)
features = behavioral.compute_momentum_features(features)

# Validate after each addition
feature_cols = [col for col in features.columns if col not in df.columns]
quality = store.run_quality_checks(features, feature_cols)
```

### Pattern 3: Feature Selection Workflow
```python
# Generate all features
all_features = engine.compute_all_features(df)

# Track importance
importance = store.track_importance(model, X, y)

# Select top N features
top_features = sorted(
    importance['gain'].items(),
    key=lambda x: x[1],
    reverse=True
)[:50]

# Use only top features
selected_df = df[[name for name, _ in top_features]]
```

## Troubleshooting

**Issue**: Features have high missing rates
- **Solution**: Check data quality, adjust time windows, or handle missing values before engineering

**Issue**: Leakage detected
- **Solution**: Review feature computation logic, check for future information usage

**Issue**: Slow performance
- **Solution**: Reduce time windows, sample data, or use incremental processing

**Issue**: Memory errors
- **Solution**: Process in chunks, reduce feature count, or use streaming

**Issue**: Import errors for optional dependencies
- **Solution**: Install optional packages (hdbscan, node2vec, karateclub, shap) or use alternative methods

## Support

For issues, questions, or contributions, please refer to the main documentation or contact the data science team.
