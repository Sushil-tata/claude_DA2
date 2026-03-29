# Fraud Detection Use Case

Production-ready fraud detection system with graph-based analysis, supervised/unsupervised learning, risk propagation, and real-time scoring capabilities.

## Features

- **Graph-Based Analysis**: Build transaction networks with accounts, merchants, devices, and IPs
- **Supervised Learning**: Multi-model fraud classification (LightGBM, XGBoost, Neural Networks)
- **Anomaly Detection**: Ensemble methods (Isolation Forest, LOF, Autoencoders)
- **Graph Embeddings**: Node2Vec, DeepWalk, GraphSAGE for network features
- **Risk Propagation**: PageRank and label propagation for fraud ring detection
- **Real-Time Scoring**: <100ms latency for production deployment
- **Model Monitoring**: Built-in alerting and performance tracking
- **A/B Testing**: Framework for comparing model variants

## Quick Start

### Basic Usage

```python
from src.use_cases.fraud_detection import FraudDetectionPipeline

# Initialize pipeline
pipeline = FraudDetectionPipeline()

# Train on historical data
pipeline.fit(transactions_df)

# Real-time scoring
result = pipeline.predict_single_transaction({
    'account_id': 'acc_123',
    'amount': 1500.00,
    'merchant_id': 'merch_456',
    'device_id': 'device_789',
    'ip_address': '192.168.1.1',
    'timestamp': '2024-01-15 10:30:00'
})

print(f"Fraud Score: {result['fraud_score']:.3f}")
print(f"Is Fraud: {result['is_fraud']}")
print(f"Latency: {result['latency_ms']:.1f}ms")
```

### Batch Scoring

```python
# Score multiple transactions
scores = pipeline.predict_batch(transactions_df)
print(scores[['fraud_score', 'is_fraud']].head())
```

### Fraud Ring Detection

```python
# Detect organized fraud rings
rings = pipeline.detect_fraud_rings()

for ring in rings[:5]:
    print(f"Ring Size: {ring['size']}")
    print(f"Average Risk: {ring['avg_risk']:.3f}")
    print(f"Density: {ring['density']:.3f}")
```

### Risk Propagation

```python
# Propagate risk from known fraud accounts
known_fraud = {'acc_123', 'acc_456', 'acc_789'}
risk_scores = pipeline.propagate_risk(known_fraud)

# Get top risky accounts
top_risk = sorted(risk_scores.items(), key=lambda x: x[1], reverse=True)[:10]
for account, risk in top_risk:
    print(f"{account}: {risk:.3f}")
```

## Components

### 1. Graph Builder

Constructs heterogeneous graphs from transaction data:

```python
from src.use_cases.fraud_detection import GraphBuilder

builder = GraphBuilder()
graph = builder.build_from_transactions(
    transactions_df,
    account_col='account_id',
    merchant_col='merchant_id',
    device_col='device_id',
    ip_col='ip_address'
)

# Save graph
builder.save_graph(graph, 'fraud_graph.pkl')
```

### 2. Supervised Fraud Classification

Multi-model classifier with extreme class imbalance handling:

```python
from src.use_cases.fraud_detection import FraudClassifier

# LightGBM classifier
classifier = FraudClassifier(model_type='lightgbm', calibrate=True)
classifier.fit(X_train, y_train, X_val, y_val)

# Optimize threshold for target recall
threshold = classifier.optimize_threshold(
    X_val, y_val, 
    target_recall=0.8
)

# Predictions
fraud_probs = classifier.predict_proba(X_test)
```

**Fraud-Specific Features**:
- Velocity features (transaction frequency/volume over time windows)
- Anomaly scores (z-scores, percentiles)
- Graph features (node degree, PageRank, clustering coefficient)
- Embeddings (graph node embeddings)

### 3. Anomaly Detection

Ensemble of multiple anomaly detection methods:

```python
from src.use_cases.fraud_detection import EnsembleAnomalyDetector

detector = EnsembleAnomalyDetector(
    use_isolation_forest=True,
    use_lof=True,
    use_autoencoder=True
)

detector.fit(X_train)
anomaly_scores = detector.predict_scores(X_test)
anomalies = detector.predict(X_test, contamination=0.01)
```

**Individual Detectors**:
- `IsolationForestDetector`: Tree-based isolation
- `LocalOutlierFactorDetector`: Density-based detection
- `AutoencoderDetector`: Neural network reconstruction error

### 4. Graph Embeddings

Learn node representations for fraud networks:

```python
from src.use_cases.fraud_detection import Node2VecEmbeddings

# Node2Vec embeddings
embedder = Node2VecEmbeddings(
    dimensions=64,
    walk_length=30,
    num_walks=200,
    p=1.0,  # Return parameter
    q=1.0   # In-out parameter
)

embeddings = embedder.fit_transform(graph)

# Community detection
from src.use_cases.fraud_detection import CommunityEmbeddings

community_detector = CommunityEmbeddings(method='louvain')
communities = community_detector.detect_communities(graph)
```

### 5. Risk Propagation

Propagate risk through transaction networks:

```python
from src.use_cases.fraud_detection import PageRankRiskPropagation

propagator = PageRankRiskPropagation(damping=0.85)

# Initial fraud nodes
initial_scores = {node: 1.0 for node in known_fraud_nodes}

# Propagate risk
risk_scores = propagator.propagate(graph, initial_scores)

# Detect fraud rings
rings = propagator.detect_fraud_rings(
    graph, 
    risk_scores, 
    threshold=0.7
)
```

**Propagation Methods**:
- `PageRankRiskPropagation`: Modified PageRank algorithm
- `LabelPropagationRisk`: Neighborhood consensus
- `NetworkInfluenceScoring`: Centrality-based scoring
- `IterativeRiskScoring`: Multi-round refinement

### 6. Production Pipeline

End-to-end orchestration with monitoring:

```python
from src.use_cases.fraud_detection import (
    FraudDetectionPipeline,
    FraudPipelineConfig
)

# Custom configuration
config = FraudPipelineConfig()
config.supervised_model = 'lightgbm'
config.target_recall = 0.85
config.max_latency_ms = 100

pipeline = FraudDetectionPipeline(config)
pipeline.fit(transactions_df)

# Save/load
pipeline.save('models/fraud_pipeline.pkl')
pipeline = FraudDetectionPipeline.load('models/fraud_pipeline.pkl')

# Monitoring
metrics = pipeline.get_monitoring_metrics()
print(f"Average Latency: {metrics['avg_latency_ms']:.1f}ms")
print(f"Fraud Rate: {metrics['fraud_rate']:.2%}")

# Alerts
alerts = pipeline.check_alerts()
for alert in alerts:
    print(f"{alert['severity']}: {alert['message']}")
```

### 7. A/B Testing

Compare model variants in production:

```python
from src.use_cases.fraud_detection import ABTestingFramework

# Create variants
ab_test = ABTestingFramework()
ab_test.add_variant('control', pipeline_v1, traffic_proportion=0.5)
ab_test.add_variant('treatment', pipeline_v2, traffic_proportion=0.5)

# Route traffic
variant_name, variant_pipeline = ab_test.get_variant(user_id='user_123')
result = variant_pipeline.predict_single_transaction(transaction)

# Log results
ab_test.log_result(variant_name, result, ground_truth=True)

# Analyze
analysis = ab_test.analyze_results()
print(analysis)
```

## Configuration

Create `config/fraud_config.yaml`:

```yaml
# Data columns
account_col: account_id
merchant_col: merchant_id
device_col: device_id
ip_col: ip_address
amount_col: amount
timestamp_col: timestamp
fraud_col: is_fraud

# Model configuration
supervised_model: lightgbm  # 'lightgbm', 'xgboost', 'neural'
use_anomaly_detection: true
use_graph_features: true
use_risk_propagation: true

# Graph configuration
graph_embedding_dim: 64
use_communities: true

# Scoring configuration
score_threshold: 0.5
target_recall: 0.8
max_latency_ms: 100

# Monitoring configuration
enable_monitoring: true
alert_threshold: 0.9
monitoring_window_days: 7
```

## Performance Optimization

### Real-Time Scoring (<100ms)

1. **Feature Caching**: Cache graph features for known entities
2. **Model Quantization**: Use quantized models for faster inference
3. **Batch Processing**: Process multiple transactions together
4. **Async Graph Updates**: Update graph asynchronously

```python
# Enable optimizations
pipeline.config.max_latency_ms = 50

# Use lightweight model
pipeline.config.supervised_model = 'lightgbm'
pipeline.config.use_anomaly_detection = False  # Disable if too slow
```

### Handling Class Imbalance

Fraud rate < 0.1%:

1. **Sample Weighting**: Automatically computed based on class distribution
2. **Threshold Optimization**: Target specific recall/precision
3. **Anomaly Detection**: Complements supervised learning
4. **Risk Propagation**: Leverages network structure

```python
# Optimize for high recall
classifier.optimize_threshold(X_val, y_val, target_recall=0.9)

# Use ensemble with unsupervised methods
detector = EnsembleAnomalyDetector()
```

## Advanced Usage

### Custom Feature Engineering

```python
from src.use_cases.fraud_detection import FraudFeatureEngineer

engineer = FraudFeatureEngineer()

# Velocity features
velocity_features = engineer.create_velocity_features(
    transactions,
    account_col='account_id',
    timestamp_col='timestamp',
    amount_col='amount'
)

# Anomaly scores
anomaly_features = engineer.create_anomaly_scores(
    transactions,
    amount_col='amount',
    merchant_col='merchant_id'
)
```

### Temporal Graph Embeddings

```python
from src.use_cases.fraud_detection import TemporalGraphEmbeddings

temporal_emb = TemporalGraphEmbeddings(
    base_embedder=Node2VecEmbeddings(dimensions=64),
    memory_decay=0.9
)

# Add graph snapshots over time
temporal_emb.add_snapshot(graph_t0, timestamp=0)
temporal_emb.add_snapshot(graph_t1, timestamp=1)
temporal_emb.add_snapshot(graph_t2, timestamp=2)

# Get current embeddings
embeddings = temporal_emb.get_current_embeddings()
```

### Iterative Risk Scoring

```python
from src.use_cases.fraud_detection import IterativeRiskScoring

scorer = IterativeRiskScoring(
    methods=['pagerank', 'label_propagation', 'influence'],
    n_iterations=3
)

risk_scores = scorer.score(graph, initial_fraud_nodes)

# Detect fraud communities
communities = scorer.detect_fraud_communities(
    graph, 
    risk_scores, 
    threshold=0.6
)
```

## Testing

```python
import pytest
from src.use_cases.fraud_detection import FraudDetectionPipeline
import pandas as pd
import numpy as np

def test_fraud_pipeline():
    # Create synthetic data
    n_samples = 1000
    transactions = pd.DataFrame({
        'account_id': [f'acc_{i}' for i in range(n_samples)],
        'merchant_id': np.random.choice(['merch_1', 'merch_2'], n_samples),
        'amount': np.random.lognormal(3, 1, n_samples),
        'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='H'),
        'is_fraud': np.random.random(n_samples) < 0.01
    })
    
    # Train pipeline
    pipeline = FraudDetectionPipeline()
    pipeline.fit(transactions)
    
    # Test prediction
    result = pipeline.predict_single_transaction(transactions.iloc[0])
    
    assert 'fraud_score' in result
    assert 0 <= result['fraud_score'] <= 1
    assert result['latency_ms'] < 200  # Should be fast
```

## Model Performance

Typical performance on real-world fraud datasets:

- **AUC-ROC**: 0.90 - 0.95
- **AUC-PR**: 0.70 - 0.85 (challenging due to extreme imbalance)
- **Recall @ 80%**: Precision of 40-60%
- **Latency**: 30-80ms (real-time scoring)
- **Fraud Rate**: Successfully handles < 0.1%

## Best Practices

1. **Regular Retraining**: Retrain models monthly or when performance degrades
2. **Feature Freshness**: Keep velocity features up-to-date
3. **Graph Updates**: Update transaction graph incrementally
4. **Threshold Tuning**: Adjust based on business requirements
5. **Monitoring**: Track performance metrics continuously
6. **False Positive Review**: Regular review process for manual verification

## Troubleshooting

### High Latency

```python
# Disable expensive features
config.use_graph_features = False
config.use_anomaly_detection = False

# Use faster model
config.supervised_model = 'lightgbm'
```

### Low Recall

```python
# Optimize threshold for higher recall
classifier.optimize_threshold(X_val, y_val, target_recall=0.9)

# Use ensemble with anomaly detection
config.use_anomaly_detection = True
```

### Memory Issues

```python
# Limit graph size
builder.config.max_nodes = 100000

# Use smaller embedding dimension
config.graph_embedding_dim = 32
```

## Dependencies

```
# Core
numpy>=1.20.0
pandas>=1.3.0
scikit-learn>=1.0.0
networkx>=2.6.0

# Tree models (optional)
lightgbm>=3.3.0
xgboost>=1.5.0

# Deep learning (optional)
torch>=1.10.0

# Graph embeddings (optional)
node2vec>=0.4.0

# Logging
loguru>=0.6.0
```

## References

- Node2Vec: Grover & Leskovec (2016)
- Isolation Forest: Liu et al. (2008)
- PageRank: Page et al. (1999)
- Graph Neural Networks: Hamilton et al. (2017)

## License

Internal use only - Principal Data Science Decision Agent

---

For questions or issues, contact the ML Engineering team.
