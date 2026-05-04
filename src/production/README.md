# Production Infrastructure - ML Model Deployment & Operations

## Overview

This module provides a comprehensive, production-grade infrastructure for deploying and operating machine learning models at scale. It includes deployment orchestration, real-time monitoring, automated retraining, and low-latency feature serving.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Production ML System                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Deployment  │  │  Monitoring  │  │  Retraining  │      │
│  │     API      │  │   Dashboard  │  │  Pipeline    │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                 │                  │               │
│  ┌──────▼──────────────────▼──────────────────▼───────┐    │
│  │           Model Registry & Version Control          │    │
│  └──────────────────────────────────────────────────────┘   │
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │Feature Server│  │ Drift Monitor│  │Alert Manager │      │
│  │   (Redis)    │  │  (Real-time) │  │  (Webhook)   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Modules

### 1. deployment.py - Model Deployment Orchestration

Comprehensive deployment system supporting multiple strategies:

**Key Features:**
- ✅ REST API deployment (FastAPI)
- ✅ Batch scoring deployment
- ✅ Model versioning and registry
- ✅ A/B testing (champion vs. challenger)
- ✅ Canary deployments
- ✅ Blue-green deployments
- ✅ Model packaging (pickle, joblib, ONNX)
- ✅ Docker containerization
- ✅ Health check endpoints
- ✅ Prometheus metrics
- ✅ Deployment validation

**Example Usage:**

```python
from production.deployment import ModelDeployer, ModelRegistry, DeploymentStrategy

# Initialize registry
registry = ModelRegistry(base_path="./models")

# Register a model
metadata = registry.register_model(
    model=trained_model,
    model_name="credit_risk",
    version="v1.0.0",
    metrics={"auc": 0.85, "ks_statistic": 0.42},
    features=feature_list,
    environment=EnvironmentType.PROD
)

# Deploy as REST API
deployer = ModelDeployer()
deployer.deploy_rest_api(
    model_name="credit_risk",
    version="v1.0.0",
    port=8000,
    enable_ab_testing=True,
    ab_models={
        "control": ("v1.0.0", 80.0),
        "treatment": ("v1.1.0", 20.0)
    }
)

# Batch deployment
results = deployer.deploy_batch(
    model_name="credit_risk",
    input_path="data/to_score.csv",
    output_path="data/predictions.csv",
    batch_size=1000
)

# Create Docker image
image_tag = deployer.create_docker_image(
    model_name="credit_risk",
    version="v1.0.0"
)
```

### 2. monitoring.py - Production Model Monitoring

Real-time monitoring framework with comprehensive metrics:

**Key Features:**
- ✅ Performance monitoring (AUC, KS, Gini degradation)
- ✅ Data drift detection integration
- ✅ Prediction distribution monitoring
- ✅ Latency tracking (p50, p95, p99)
- ✅ Error rate monitoring
- ✅ Volume metrics (RPS)
- ✅ Alert generation with severity levels
- ✅ Prometheus metrics export
- ✅ SLA compliance tracking
- ✅ Dashboard data export

**Example Usage:**

```python
from production.monitoring import ModelMonitor, SLAConfig, MetricType

# Initialize monitor
sla_config = SLAConfig(
    max_latency_p95_ms=50.0,
    max_latency_p99_ms=100.0,
    min_auc=0.75,
    max_auc_degradation=0.05,
    max_error_rate=0.1
)

monitor = ModelMonitor(
    model_name="fraud_detection",
    model_version="v1.0.0",
    reference_data=train_df,
    reference_predictions=train_preds,
    reference_actuals=train_actuals,
    metric_type=MetricType.CLASSIFICATION,
    sla_config=sla_config,
    storage_path="./monitoring_data"
)

# Log predictions
monitor.log_predictions(
    features=features_df,
    predictions=predictions,
    actuals=actuals,
    prediction_probabilities=probabilities,
    latency_ms=5.2
)

# Check for drift
drift_report = monitor.check_drift(
    production_data=prod_features,
    production_predictions=prod_preds
)

# Check for alerts
alerts = monitor.check_alerts()
if alerts:
    monitor.send_notifications(
        alerts=alerts,
        webhook_url="https://hooks.slack.com/...",
        email_recipients=["team@company.com"]
    )

# Get Prometheus metrics
prometheus_metrics = monitor.get_prometheus_metrics()

# Get SLA compliance
compliance = monitor.get_sla_compliance()
```

### 3. retraining.py - Automated Retraining Pipeline

Intelligent retraining orchestration with multiple triggers:

**Key Features:**
- ✅ Performance degradation triggers
- ✅ Data drift triggers (PSI, feature drift)
- ✅ Scheduled retraining (daily, weekly, monthly, quarterly)
- ✅ Manual triggers
- ✅ Automated data pipeline integration
- ✅ Feature engineering pipeline
- ✅ Hyperparameter optimization
- ✅ OOT validation
- ✅ Champion-challenger comparison
- ✅ Automated rollback on failure
- ✅ Job history tracking

**Example Usage:**

```python
from production.retraining import RetrainingOrchestrator, TriggerConfig, ValidationConfig

# Define training pipeline
def training_pipeline(train_data, val_data):
    # Your training logic here
    model = train_model(train_data)
    metrics = evaluate_model(model, val_data)
    return model, metrics

# Configure triggers
trigger_config = TriggerConfig(
    enable_performance_trigger=True,
    max_auc_degradation=0.05,
    enable_drift_trigger=True,
    max_psi_threshold=0.25,
    enable_scheduled_trigger=True,
    schedule_frequency="monthly"
)

# Configure validation
validation_config = ValidationConfig(
    oot_validation_required=True,
    min_oot_samples=5000,
    min_improvement_threshold=0.01,
    business_metrics=["auc", "ks_statistic"]
)

# Initialize orchestrator
orchestrator = RetrainingOrchestrator(
    model_name="credit_risk",
    training_pipeline=training_pipeline,
    data_pipeline=fetch_training_data,
    feature_pipeline=engineer_features,
    trigger_config=trigger_config,
    validation_config=validation_config,
    monitor=monitor
)

# Schedule automatic retraining
orchestrator.schedule_retraining(schedule="monthly")

# Or trigger manually
job = orchestrator.trigger_retraining(
    trigger_type=TriggerType.MANUAL,
    reason="Model update with new features"
)

# Check job status
status = orchestrator.get_job_status(job.job_id)

# Get retraining history
history = orchestrator.get_retraining_history()
```

### 4. feature_serving.py - Real-time Feature Server

Low-latency feature serving with caching:

**Key Features:**
- ✅ Online/offline/realtime feature types
- ✅ Redis and in-memory caching
- ✅ LRU and TTL cache strategies
- ✅ Batch precomputation
- ✅ Real-time aggregations
- ✅ Feature versioning
- ✅ Fallback strategies
- ✅ Latency optimization (<10ms target)
- ✅ Cache hit rate tracking
- ✅ Async operations

**Example Usage:**

```python
from production.feature_serving import (
    FeatureServer, FeatureConfig, FeatureDefinition, FeatureType
)

# Define features
features = [
    FeatureDefinition(
        name="age",
        feature_type=FeatureType.OFFLINE,
        table_name="customers",
        column_name="age",
        cache_ttl_seconds=3600
    ),
    FeatureDefinition(
        name="credit_utilization",
        feature_type=FeatureType.ONLINE,
        computation_fn=compute_credit_utilization,
        dependencies=["credit_used", "credit_limit"],
        cache_ttl_seconds=300
    ),
    FeatureDefinition(
        name="transactions_24h",
        feature_type=FeatureType.REALTIME,
        aggregation_window="24h",
        aggregation_function="count",
        cache_ttl_seconds=60
    )
]

# Configure feature server
config = FeatureConfig(
    features=features,
    cache_strategy=CacheStrategy.LRU,
    max_cache_size=10000,
    latency_target_ms=10.0
)

# Initialize server
server = FeatureServer(
    feature_config=config,
    cache_backend="redis",
    redis_url="redis://localhost:6379"
)

await server.initialize()

# Get features for single entity
features = await server.get_features(
    entity_id="customer_12345",
    feature_names=["age", "credit_utilization", "transactions_24h"]
)

# Batch feature retrieval
batch_features = await server.get_features_batch(
    entity_ids=customer_ids,
    feature_names=["age", "credit_utilization"]
)

# Precompute batch features
await server.precompute_batch_features(
    entity_ids=high_priority_customers,
    feature_names=["transactions_24h", "credit_utilization"]
)

# Get metrics
metrics = server.get_metrics()
print(f"Cache hit rate: {metrics['cache_hit_rate']:.2f}%")
```

## Deployment Strategies

### 1. Direct Deployment
Immediate deployment of new model version.

### 2. Blue-Green Deployment
Zero-downtime deployment with instant rollback capability.

```python
# Deploy new version (green)
deployer.deploy_rest_api(
    model_name="credit_risk",
    version="v2.0.0",
    strategy=DeploymentStrategy.BLUE_GREEN
)
# Traffic switches immediately after validation
```

### 3. Canary Deployment
Gradual rollout to subset of traffic.

```python
deployer.deploy_rest_api(
    model_name="credit_risk",
    version="v2.0.0",
    strategy=DeploymentStrategy.CANARY,
    canary_percentage=10.0  # 10% of traffic
)
# Monitor, then increase gradually
```

### 4. A/B Testing
Compare multiple model versions.

```python
deployer.deploy_rest_api(
    model_name="credit_risk",
    enable_ab_testing=True,
    ab_models={
        "control": ("v1.0.0", 50.0),
        "variant_a": ("v2.0.0", 25.0),
        "variant_b": ("v2.1.0", 25.0)
    }
)
```

## Integration Examples

### Complete Production Pipeline

```python
from production import *

# 1. Register model
registry = ModelRegistry()
metadata = registry.register_model(
    model=trained_model,
    model_name="fraud_detector",
    version="v1.0.0",
    metrics={"auc": 0.92, "precision": 0.88}
)

# 2. Deploy with monitoring
deployer = ModelDeployer(model_registry=registry)
monitor = ModelMonitor(
    model_name="fraud_detector",
    reference_data=train_df,
    sla_config=SLAConfig(max_latency_p95_ms=50)
)

app = deployer.create_rest_api(
    model_name="fraud_detector",
    version="v1.0.0"
)

# 3. Setup automated retraining
orchestrator = RetrainingOrchestrator(
    model_name="fraud_detector",
    training_pipeline=my_training_fn,
    monitor=monitor
)
orchestrator.schedule_retraining("monthly")

# 4. Setup feature serving
feature_server = FeatureServer(config=feature_config)

# 5. Integrate into prediction endpoint
@app.post("/predict")
async def predict(entity_id: str):
    # Get features
    features = await feature_server.get_features(
        entity_id=entity_id,
        feature_names=required_features
    )
    
    # Make prediction
    prediction = model.predict([features])
    
    # Log for monitoring
    monitor.log_predictions(
        features=pd.DataFrame([features]),
        predictions=prediction,
        latency_ms=response_time
    )
    
    return {"prediction": prediction[0]}
```

### Docker Deployment

```bash
# Build Docker image
python -c "
from production.deployment import ModelDeployer
deployer = ModelDeployer()
deployer.create_docker_image('fraud_detector', 'v1.0.0')
"

# Run container
docker run -p 8000:8000 fraud_detector:v1.0.0

# Health check
curl http://localhost:8000/health

# Make prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"age": 35, "income": 50000}}'

# Check metrics
curl http://localhost:8000/metrics
```

### Kubernetes Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fraud-detector
spec:
  replicas: 3
  selector:
    matchLabels:
      app: fraud-detector
  template:
    metadata:
      labels:
        app: fraud-detector
    spec:
      containers:
      - name: model-server
        image: fraud-detector:v1.0.0
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: fraud-detector-service
spec:
  selector:
    app: fraud-detector
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

## Monitoring Dashboard

### Grafana Configuration

```json
{
  "dashboard": {
    "title": "ML Model Monitoring",
    "panels": [
      {
        "title": "Model Performance",
        "targets": [
          {"expr": "model_auc{model=\"fraud_detector\"}"},
          {"expr": "model_accuracy{model=\"fraud_detector\"}"}
        ]
      },
      {
        "title": "Latency",
        "targets": [
          {"expr": "model_latency_ms_p95{model=\"fraud_detector\"}"},
          {"expr": "model_latency_ms_p99{model=\"fraud_detector\"}"}
        ]
      },
      {
        "title": "Error Rate",
        "targets": [
          {"expr": "rate(model_errors_total[5m])"}
        ]
      }
    ]
  }
}
```

## Configuration

### Environment Variables

```bash
# Deployment
MODEL_REGISTRY_PATH=/app/models
DEPLOYMENT_ENVIRONMENT=production

# Monitoring
MONITORING_STORAGE_PATH=/app/monitoring
PROMETHEUS_PORT=9090
ALERT_WEBHOOK_URL=https://hooks.slack.com/...

# Feature Serving
REDIS_URL=redis://localhost:6379
FEATURE_CACHE_SIZE=10000
FEATURE_CACHE_TTL=3600

# Retraining
RETRAINING_SCHEDULE=monthly
RETRAINING_DAY=1
RETRAINING_TIME=02:00
```

## Best Practices

### 1. Model Versioning
- Use semantic versioning (v1.0.0, v1.1.0, v2.0.0)
- Tag models with metadata (training date, data version, features)
- Maintain champion-challenger separation

### 2. Monitoring
- Set realistic SLA thresholds based on business requirements
- Monitor both technical (latency, errors) and business metrics (AUC, precision)
- Alert on trends, not single data points

### 3. Retraining
- Validate on out-of-time (OOT) data before deployment
- Require minimum improvement threshold to replace champion
- Implement automatic rollback on deployment failure

### 4. Feature Serving
- Cache expensive features with appropriate TTL
- Precompute batch features for known entities
- Implement fallback strategies for missing features

### 5. Deployment
- Use canary deployments for high-risk changes
- Validate deployment before switching traffic
- Maintain rollback capability

## Troubleshooting

### High Latency
```python
# Check feature serving metrics
metrics = feature_server.get_metrics()
slow_features = [
    name for name, m in metrics['feature_metrics'].items()
    if m['avg_latency_ms'] > 10
]

# Increase cache TTL or precompute
for feature in slow_features:
    server.precompute_batch_features(
        entity_ids=active_users,
        feature_names=[feature]
    )
```

### Model Degradation
```python
# Check drift report
drift = monitor.check_drift(production_data)
if drift.has_drift:
    print(f"Drifted features: {drift.drifted_features}")
    
    # Trigger retraining
    orchestrator.trigger_retraining(
        trigger_type=TriggerType.DATA_DRIFT,
        reason=f"Drift in {len(drift.drifted_features)} features"
    )
```

### Deployment Failures
```python
# Validate before deployment
is_valid = deployer.validate_deployment(
    model_name="fraud_detector",
    version="v2.0.0",
    test_data=validation_df
)

if not is_valid:
    # Fix issues before deploying
    pass
```

## Performance Benchmarks

| Component | Metric | Target | Typical |
|-----------|--------|--------|---------|
| Feature Serving | P95 Latency | <10ms | 5-8ms |
| Model Prediction | P95 Latency | <50ms | 20-30ms |
| Cache Hit Rate | Hit Rate | >80% | 85-95% |
| API Throughput | RPS | >1000 | 2000-5000 |
| Monitoring Overhead | Latency Impact | <5% | 2-3% |

## Cloud Provider Support

All components are cloud-agnostic and support:
- ✅ AWS (S3, ECS, Lambda, SageMaker)
- ✅ GCP (GCS, Cloud Run, Vertex AI)
- ✅ Azure (Blob Storage, AKS, Azure ML)

## License

See LICENSE file for details.

## Contributing

See CONTRIBUTING.md for guidelines.
