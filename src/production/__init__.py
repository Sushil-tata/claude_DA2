"""
Production Infrastructure for ML Models

This package provides comprehensive production infrastructure including:
- Model deployment (REST API, batch, Docker)
- Real-time monitoring and alerting
- Automated retraining pipelines
- Low-latency feature serving

Example:
    >>> from production import ModelDeployer, ModelMonitor, RetrainingOrchestrator
    >>> deployer = ModelDeployer()
    >>> deployer.deploy_rest_api(model_name="credit_risk", port=8000)
"""

# Deployment
from production.deployment import (
    ModelDeployer,
    ModelRegistry,
    ModelMetadata,
    DeploymentConfig,
    DeploymentStrategy,
    ModelFormat,
    EnvironmentType,
    PredictionRequest,
    PredictionResponse,
    HealthResponse
)

# Monitoring
from production.monitoring import (
    ModelMonitor,
    PerformanceMetrics,
    LatencyMetrics,
    VolumeMetrics,
    Alert,
    AlertSeverity,
    MetricType,
    SLAConfig
)

# Retraining
from production.retraining import (
    RetrainingOrchestrator,
    RetrainingJob,
    RetrainingStatus,
    TriggerType,
    TriggerConfig,
    ValidationConfig
)

# Feature Serving
from production.feature_serving import (
    FeatureServer,
    FeatureConfig,
    FeatureDefinition,
    FeatureType,
    FeatureMetrics,
    CacheStrategy,
    LRUCache,
    TTLCache
)

__all__ = [
    # Deployment
    'ModelDeployer',
    'ModelRegistry',
    'ModelMetadata',
    'DeploymentConfig',
    'DeploymentStrategy',
    'ModelFormat',
    'EnvironmentType',
    'PredictionRequest',
    'PredictionResponse',
    'HealthResponse',
    
    # Monitoring
    'ModelMonitor',
    'PerformanceMetrics',
    'LatencyMetrics',
    'VolumeMetrics',
    'Alert',
    'AlertSeverity',
    'MetricType',
    'SLAConfig',
    
    # Retraining
    'RetrainingOrchestrator',
    'RetrainingJob',
    'RetrainingStatus',
    'TriggerType',
    'TriggerConfig',
    'ValidationConfig',
    
    # Feature Serving
    'FeatureServer',
    'FeatureConfig',
    'FeatureDefinition',
    'FeatureType',
    'FeatureMetrics',
    'CacheStrategy',
    'LRUCache',
    'TTLCache',
]

__version__ = '1.0.0'
