# Production Deployment Configuration Examples

## Example 1: Simple REST API Deployment

```python
from production import ModelDeployer, ModelRegistry, EnvironmentType

# Initialize components
registry = ModelRegistry(base_path="./models")
deployer = ModelDeployer(model_registry_path="./models")

# Register model
metadata = registry.register_model(
    model=my_trained_model,
    model_name="credit_risk",
    version="v1.0.0",
    framework="lightgbm",
    metrics={
        "auc": 0.85,
        "ks_statistic": 0.42,
        "gini": 0.70
    },
    features=["age", "income", "credit_score", "debt_ratio"],
    target="default_flag",
    environment=EnvironmentType.PROD
)

# Deploy REST API
deployer.deploy_rest_api(
    model_name="credit_risk",
    version="v1.0.0",
    port=8000,
    host="0.0.0.0"
)
```

## Example 2: A/B Testing Deployment

```python
from production import ModelDeployer, DeploymentStrategy

deployer = ModelDeployer()

# Deploy with A/B testing
deployer.deploy_rest_api(
    model_name="fraud_detector",
    enable_ab_testing=True,
    ab_models={
        "control": ("v1.0.0", 70.0),      # 70% traffic
        "challenger": ("v1.1.0", 30.0)     # 30% traffic
    },
    port=8000
)
```

## Example 3: Canary Deployment

```python
from production import DeploymentConfig, DeploymentStrategy, EnvironmentType

config = DeploymentConfig(
    model_name="recommendation_engine",
    version="v2.0.0",
    strategy=DeploymentStrategy.CANARY,
    environment=EnvironmentType.PROD,
    canary_percentage=10.0,  # Start with 10%
    rollback_on_error=True,
    health_check_interval=30
)

deployer.deploy_rest_api(
    model_name="recommendation_engine",
    version="v2.0.0",
    port=8000
)

# Monitor canary performance
# If successful, gradually increase traffic
# If issues detected, automatic rollback
```

## Example 4: Batch Scoring

```python
from production import ModelDeployer

deployer = ModelDeployer()

# Batch prediction on large dataset
results = deployer.deploy_batch(
    model_name="customer_churn",
    version="v1.0.0",
    input_path="data/customers_to_score.parquet",
    output_path="data/churn_predictions.parquet",
    batch_size=5000,  # Process in batches of 5000
    schedule="0 2 * * *"  # Daily at 2 AM
)

print(f"Scored {len(results)} customers")
print(f"Predicted churn rate: {results['prediction'].mean():.2%}")
```

## Example 5: Complete Monitoring Setup

```python
from production import ModelMonitor, SLAConfig, MetricType

# Define SLA thresholds
sla_config = SLAConfig(
    max_latency_p95_ms=50.0,      # P95 < 50ms
    max_latency_p99_ms=100.0,     # P99 < 100ms
    min_availability=99.9,         # 99.9% uptime
    max_error_rate=0.1,           # < 0.1% errors
    min_auc=0.75,                 # AUC > 0.75
    max_auc_degradation=0.05,     # Max 5% degradation
    max_psi_threshold=0.25        # PSI < 0.25
)

# Initialize monitor
monitor = ModelMonitor(
    model_name="fraud_detector",
    model_version="v1.0.0",
    reference_data=train_df,
    reference_predictions=train_predictions,
    reference_actuals=train_actuals,
    metric_type=MetricType.CLASSIFICATION,
    sla_config=sla_config,
    monitoring_window_hours=24,
    storage_path="./monitoring_data"
)

# Log predictions in production
def make_prediction(features, actuals=None):
    import time
    start = time.time()
    
    prediction = model.predict([features])
    probabilities = model.predict_proba([features])
    
    latency_ms = (time.time() - start) * 1000
    
    # Log for monitoring
    monitor.log_predictions(
        features=pd.DataFrame([features]),
        predictions=prediction,
        actuals=actuals if actuals else None,
        prediction_probabilities=probabilities,
        latency_ms=latency_ms
    )
    
    return prediction[0]

# Periodic drift check (run hourly)
def check_drift_job():
    recent_data = get_recent_production_data()
    drift_report = monitor.check_drift(recent_data)
    
    if drift_report.has_drift:
        logger.warning(f"Drift detected: {drift_report.drifted_features}")
        send_alert(drift_report)

# Check alerts (run every 5 minutes)
def check_alerts_job():
    alerts = monitor.check_alerts()
    
    if alerts:
        monitor.send_notifications(
            alerts=alerts,
            webhook_url="https://hooks.slack.com/services/YOUR/WEBHOOK",
            email_recipients=["ml-team@company.com"]
        )
```

## Example 6: Automated Retraining Pipeline

```python
from production import (
    RetrainingOrchestrator, 
    TriggerConfig, 
    ValidationConfig,
    TriggerType
)

# Define training pipeline
def training_pipeline(train_data, val_data):
    """Your model training logic."""
    from lightgbm import LGBMClassifier
    from sklearn.metrics import roc_auc_score
    
    # Prepare data
    X_train = train_data.drop('target', axis=1)
    y_train = train_data['target']
    X_val = val_data.drop('target', axis=1)
    y_val = val_data['target']
    
    # Train model
    model = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=7
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    val_proba = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_proba)
    
    metrics = {
        'auc': auc,
        'n_samples': len(train_data)
    }
    
    return model, metrics

# Define data pipeline
def data_pipeline():
    """Fetch fresh training data."""
    # Load from database, data lake, etc.
    train_df = load_training_data(days=365)
    val_df = load_validation_data(days=30)
    return train_df, val_df

# Configure triggers
trigger_config = TriggerConfig(
    # Performance-based retraining
    enable_performance_trigger=True,
    min_auc_threshold=0.75,
    max_auc_degradation=0.05,
    min_sample_size=1000,
    
    # Drift-based retraining
    enable_drift_trigger=True,
    max_psi_threshold=0.25,
    max_drifted_features=5,
    
    # Scheduled retraining
    enable_scheduled_trigger=True,
    schedule_frequency="monthly",
    schedule_day=1,
    schedule_time="02:00"
)

# Configure validation
validation_config = ValidationConfig(
    oot_validation_required=True,
    min_oot_samples=5000,
    oot_period_days=30,
    min_improvement_threshold=0.01,  # Min 1% improvement
    max_degradation_threshold=0.02,  # Max 2% degradation
    business_metrics=['auc', 'ks_statistic', 'gini']
)

# Initialize orchestrator
orchestrator = RetrainingOrchestrator(
    model_name="credit_risk",
    training_pipeline=training_pipeline,
    data_pipeline=data_pipeline,
    trigger_config=trigger_config,
    validation_config=validation_config,
    monitor=monitor,
    storage_path="./retraining_jobs"
)

# Start scheduler for automatic retraining
orchestrator.schedule_retraining()

# Or trigger manually
job = orchestrator.trigger_retraining(
    trigger_type=TriggerType.MANUAL,
    reason="Adding new features"
)

# Monitor job progress
status = orchestrator.get_job_status(job.job_id)
print(f"Status: {status.status}")
print(f"Metrics: {status.challenger_metrics}")
```

## Example 7: Feature Serving Setup

```python
from production import (
    FeatureServer,
    FeatureConfig,
    FeatureDefinition,
    FeatureType,
    CacheStrategy
)

# Define feature computation functions
def compute_debt_ratio(entity_id: str, total_debt: float, annual_income: float) -> float:
    """Compute debt-to-income ratio."""
    if annual_income == 0:
        return 0.0
    return total_debt / annual_income

async def fetch_credit_score(entity_id: str) -> int:
    """Fetch credit score from external service."""
    # Call credit bureau API
    response = await credit_bureau_api.get_score(entity_id)
    return response['score']

# Define features
features = [
    # Offline features (precomputed in database)
    FeatureDefinition(
        name="age",
        feature_type=FeatureType.OFFLINE,
        table_name="customer_profiles",
        column_name="age",
        cache_ttl_seconds=3600,  # Cache for 1 hour
        description="Customer age in years"
    ),
    
    FeatureDefinition(
        name="annual_income",
        feature_type=FeatureType.OFFLINE,
        table_name="customer_profiles",
        column_name="annual_income",
        cache_ttl_seconds=3600
    ),
    
    # Online features (computed on-demand)
    FeatureDefinition(
        name="debt_ratio",
        feature_type=FeatureType.ONLINE,
        computation_fn=compute_debt_ratio,
        dependencies=["total_debt", "annual_income"],
        cache_ttl_seconds=300,  # Cache for 5 minutes
        description="Debt-to-income ratio"
    ),
    
    FeatureDefinition(
        name="credit_score",
        feature_type=FeatureType.ONLINE,
        computation_fn=fetch_credit_score,
        cache_ttl_seconds=1800,  # Cache for 30 minutes
        description="Credit bureau score"
    ),
    
    # Realtime aggregation features
    FeatureDefinition(
        name="transactions_24h",
        feature_type=FeatureType.REALTIME,
        aggregation_window="24h",
        aggregation_function="count",
        cache_ttl_seconds=60,  # Cache for 1 minute
        description="Transaction count in last 24 hours"
    ),
    
    FeatureDefinition(
        name="avg_transaction_7d",
        feature_type=FeatureType.REALTIME,
        aggregation_window="7d",
        aggregation_function="avg",
        cache_ttl_seconds=300,
        description="Average transaction amount in last 7 days"
    )
]

# Configure feature server
config = FeatureConfig(
    features=features,
    entity_key="customer_id",
    cache_strategy=CacheStrategy.LRU,
    max_cache_size=10000,
    default_ttl_seconds=3600,
    enable_fallback=True,
    latency_target_ms=10.0
)

# Initialize server
server = FeatureServer(
    feature_config=config,
    cache_backend="redis",
    redis_url="redis://localhost:6379"
)

await server.initialize()

# Get features for prediction
async def get_prediction_features(customer_id: str):
    features = await server.get_features(
        entity_id=customer_id,
        feature_names=[
            "age",
            "annual_income",
            "debt_ratio",
            "credit_score",
            "transactions_24h",
            "avg_transaction_7d"
        ]
    )
    return features

# Precompute features for batch scoring
async def precompute_features():
    high_value_customers = get_high_value_customers()
    
    await server.precompute_batch_features(
        entity_ids=high_value_customers,
        feature_names=["debt_ratio", "transactions_24h"]
    )
    
    logger.info(f"Precomputed features for {len(high_value_customers)} customers")

# Monitor feature serving performance
metrics = server.get_metrics()
print(f"Cache hit rate: {metrics['cache_hit_rate']:.2f}%")
print(f"P95 latency: {metrics['p95_latency_ms']:.2f}ms")
```

## Example 8: End-to-End Production System

```python
"""
Complete production ML system integrating all components.
"""
from production import *
import asyncio

class ProductionMLSystem:
    def __init__(self, model_name: str):
        self.model_name = model_name
        
        # Initialize components
        self.registry = ModelRegistry()
        self.deployer = ModelDeployer(model_registry=self.registry)
        self.monitor = ModelMonitor(
            model_name=model_name,
            sla_config=SLAConfig(
                max_latency_p95_ms=50,
                min_auc=0.75
            )
        )
        self.orchestrator = RetrainingOrchestrator(
            model_name=model_name,
            training_pipeline=self.train_model,
            monitor=self.monitor
        )
        self.feature_server = FeatureServer(
            feature_config=self.get_feature_config()
        )
    
    def get_feature_config(self):
        # Define features
        return FeatureConfig(features=[...])
    
    def train_model(self, train_data, val_data):
        # Training logic
        return model, metrics
    
    async def predict(self, entity_id: str):
        """Make a prediction with full monitoring."""
        import time
        start = time.time()
        
        # Get features
        features = await self.feature_server.get_features(
            entity_id=entity_id,
            feature_names=self.get_required_features()
        )
        
        # Load model
        model, metadata = self.registry.get_model(self.model_name)
        
        # Predict
        prediction = model.predict([list(features.values())])
        probabilities = model.predict_proba([list(features.values())])
        
        # Log for monitoring
        latency_ms = (time.time() - start) * 1000
        self.monitor.log_predictions(
            features=pd.DataFrame([features]),
            predictions=prediction,
            prediction_probabilities=probabilities,
            latency_ms=latency_ms
        )
        
        return {
            'prediction': int(prediction[0]),
            'probability': float(probabilities[0][1]),
            'latency_ms': latency_ms
        }
    
    async def health_check(self):
        """System health check."""
        # Check alerts
        alerts = self.monitor.check_alerts()
        
        # Check feature server
        feature_metrics = self.feature_server.get_metrics()
        
        # Check model
        models = self.registry.list_models()
        
        return {
            'healthy': len(alerts) == 0,
            'alerts': len(alerts),
            'feature_cache_hit_rate': feature_metrics.get('cache_hit_rate', 0),
            'models_registered': len(models)
        }
    
    def start(self):
        """Start all services."""
        # Start retraining scheduler
        self.orchestrator.schedule_retraining("monthly")
        
        # Deploy API
        self.deployer.deploy_rest_api(
            model_name=self.model_name,
            port=8000
        )

# Usage
if __name__ == "__main__":
    system = ProductionMLSystem("credit_risk")
    system.start()
```

## Configuration Files

### config.yaml
```yaml
deployment:
  model_registry_path: /app/models
  environment: production
  replicas: 3
  port: 8000

monitoring:
  storage_path: /app/monitoring
  window_hours: 24
  sla:
    max_latency_p95_ms: 50.0
    max_latency_p99_ms: 100.0
    min_auc: 0.75
    max_error_rate: 0.1

retraining:
  schedule_frequency: monthly
  schedule_day: 1
  schedule_time: "02:00"
  triggers:
    performance_degradation: true
    data_drift: true
    max_auc_degradation: 0.05
    max_psi_threshold: 0.25

feature_serving:
  cache_backend: redis
  redis_url: redis://redis:6379
  cache_size: 10000
  latency_target_ms: 10.0
```

### docker-compose.yml
```yaml
version: '3.8'

services:
  model-server:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_NAME=credit_risk
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
    volumes:
      - ./models:/app/models
      - ./monitoring:/app/monitoring

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-data:/var/lib/grafana
      - ./grafana-dashboards:/etc/grafana/provisioning/dashboards

volumes:
  redis-data:
  prometheus-data:
  grafana-data:
```
