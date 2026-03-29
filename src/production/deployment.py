"""
Production Deployment Architecture for ML Models

This module provides comprehensive deployment orchestration for ML models including
REST API deployment, batch scoring, model versioning, A/B testing, canary deployments,
and blue-green deployment patterns.

Usage Example:
    >>> from deployment import ModelDeployer
    >>> deployer = ModelDeployer(model_registry_path="./models")
    >>> # REST API deployment
    >>> deployer.deploy_rest_api(
    ...     model_name="credit_risk_v1",
    ...     port=8000,
    ...     enable_ab_testing=True
    ... )
    >>> # Batch deployment
    >>> deployer.deploy_batch(
    ...     model_name="credit_risk_v1",
    ...     schedule="0 2 * * *"  # Daily at 2 AM
    ... )
"""

from typing import Dict, List, Optional, Union, Any, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import pickle
import joblib
import json
import asyncio
from enum import Enum
import hashlib
import shutil
import subprocess
import tempfile

import numpy as np
import pandas as pd
from loguru import logger
from fastapi import FastAPI, HTTPException, Request, Response, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import uvicorn

# Optional dependencies
try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logger.warning("ONNX not available. Install onnx and onnxruntime for ONNX support.")


class DeploymentStrategy(str, Enum):
    """Deployment strategy options."""
    DIRECT = "direct"
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    AB_TEST = "ab_test"
    SHADOW = "shadow"


class ModelFormat(str, Enum):
    """Model serialization format."""
    PICKLE = "pickle"
    JOBLIB = "joblib"
    ONNX = "onnx"
    CUSTOM = "custom"


class EnvironmentType(str, Enum):
    """Deployment environment type."""
    DEV = "dev"
    STAGING = "staging"
    PROD = "prod"
    CANARY = "canary"


@dataclass
class ModelMetadata:
    """Model metadata for registry."""
    model_name: str
    version: str
    created_at: datetime
    model_format: ModelFormat
    framework: str  # sklearn, xgboost, lightgbm, pytorch, etc.
    metrics: Dict[str, float] = field(default_factory=dict)
    features: List[str] = field(default_factory=list)
    target: Optional[str] = None
    model_hash: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)
    environment: EnvironmentType = EnvironmentType.DEV
    deployment_strategy: Optional[DeploymentStrategy] = None
    traffic_percentage: float = 100.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'model_name': self.model_name,
            'version': self.version,
            'created_at': self.created_at.isoformat(),
            'model_format': self.model_format.value,
            'framework': self.framework,
            'metrics': self.metrics,
            'features': self.features,
            'target': self.target,
            'model_hash': self.model_hash,
            'tags': self.tags,
            'environment': self.environment.value,
            'deployment_strategy': self.deployment_strategy.value if self.deployment_strategy else None,
            'traffic_percentage': self.traffic_percentage
        }


@dataclass
class DeploymentConfig:
    """Configuration for model deployment."""
    model_name: str
    version: str
    strategy: DeploymentStrategy = DeploymentStrategy.DIRECT
    environment: EnvironmentType = EnvironmentType.DEV
    replicas: int = 1
    resources: Dict[str, Any] = field(default_factory=dict)
    health_check_interval: int = 30  # seconds
    traffic_split: Dict[str, float] = field(default_factory=dict)  # For A/B testing
    canary_percentage: float = 10.0
    rollback_on_error: bool = True
    enable_monitoring: bool = True
    enable_caching: bool = True
    timeout: int = 30  # seconds
    batch_size: int = 1000
    
    def validate(self) -> bool:
        """Validate deployment configuration."""
        if self.strategy == DeploymentStrategy.CANARY:
            if not 0 < self.canary_percentage <= 100:
                raise ValueError(f"Invalid canary percentage: {self.canary_percentage}")
        
        if self.strategy == DeploymentStrategy.AB_TEST:
            if not self.traffic_split:
                raise ValueError("Traffic split required for A/B testing")
            if abs(sum(self.traffic_split.values()) - 100.0) > 0.01:
                raise ValueError("Traffic split must sum to 100%")
        
        return True


class PredictionRequest(BaseModel):
    """Request schema for predictions."""
    features: Union[Dict[str, Any], List[Dict[str, Any]]]
    model_version: Optional[str] = None
    return_probabilities: bool = False
    return_explanations: bool = False
    
    class Config:
        schema_extra = {
            "example": {
                "features": {
                    "age": 35,
                    "income": 50000,
                    "credit_score": 720
                },
                "return_probabilities": True
            }
        }


class PredictionResponse(BaseModel):
    """Response schema for predictions."""
    predictions: Union[List[float], List[int], List[Dict[str, float]]]
    model_name: str
    model_version: str
    timestamp: str
    latency_ms: float
    probabilities: Optional[List[Dict[str, float]]] = None
    explanations: Optional[List[Dict[str, Any]]] = None
    
    class Config:
        schema_extra = {
            "example": {
                "predictions": [0, 1, 0],
                "model_name": "credit_risk_model",
                "model_version": "v1.0.0",
                "timestamp": "2024-01-01T12:00:00",
                "latency_ms": 5.2,
                "probabilities": [
                    {"class_0": 0.8, "class_1": 0.2},
                    {"class_0": 0.3, "class_1": 0.7}
                ]
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    uptime_seconds: float
    requests_served: int
    avg_latency_ms: float


class ModelRegistry:
    """
    Model registry for version control and metadata management.
    
    Supports model versioning, metadata tracking, and model retrieval.
    
    Example:
        >>> registry = ModelRegistry(base_path="./models")
        >>> registry.register_model(
        ...     model=trained_model,
        ...     model_name="credit_risk",
        ...     version="v1.0.0",
        ...     metrics={"auc": 0.85}
        ... )
        >>> model, metadata = registry.get_model("credit_risk", "v1.0.0")
    """
    
    def __init__(self, base_path: Union[str, Path] = "./model_registry"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.base_path / "registry_metadata.json"
        self.metadata_cache: Dict[str, List[ModelMetadata]] = {}
        self._load_metadata()
        logger.info(f"Initialized ModelRegistry at {self.base_path}")
    
    def _load_metadata(self) -> None:
        """Load registry metadata from disk."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                data = json.load(f)
                for model_name, versions in data.items():
                    self.metadata_cache[model_name] = [
                        ModelMetadata(
                            model_name=v['model_name'],
                            version=v['version'],
                            created_at=datetime.fromisoformat(v['created_at']),
                            model_format=ModelFormat(v['model_format']),
                            framework=v['framework'],
                            metrics=v.get('metrics', {}),
                            features=v.get('features', []),
                            target=v.get('target'),
                            model_hash=v.get('model_hash'),
                            tags=v.get('tags', {}),
                            environment=EnvironmentType(v.get('environment', 'dev')),
                            deployment_strategy=DeploymentStrategy(v['deployment_strategy']) if v.get('deployment_strategy') else None,
                            traffic_percentage=v.get('traffic_percentage', 100.0)
                        )
                        for v in versions
                    ]
            logger.info(f"Loaded metadata for {len(self.metadata_cache)} models")
    
    def _save_metadata(self) -> None:
        """Save registry metadata to disk."""
        data = {
            model_name: [v.to_dict() for v in versions]
            for model_name, versions in self.metadata_cache.items()
        }
        with open(self.metadata_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _compute_model_hash(self, model_path: Path) -> str:
        """Compute hash of model file for integrity checking."""
        sha256_hash = hashlib.sha256()
        with open(model_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def register_model(
        self,
        model: Any,
        model_name: str,
        version: str,
        model_format: ModelFormat = ModelFormat.JOBLIB,
        framework: str = "sklearn",
        metrics: Optional[Dict[str, float]] = None,
        features: Optional[List[str]] = None,
        target: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        environment: EnvironmentType = EnvironmentType.DEV,
        deployment_strategy: Optional[DeploymentStrategy] = None
    ) -> ModelMetadata:
        """
        Register a model in the registry.
        
        Args:
            model: Model object to register
            model_name: Name of the model
            version: Version string (e.g., 'v1.0.0')
            model_format: Serialization format
            framework: ML framework used
            metrics: Performance metrics
            features: Feature list
            target: Target variable name
            tags: Additional metadata tags
            environment: Deployment environment
            deployment_strategy: Deployment strategy
            
        Returns:
            ModelMetadata object
        """
        # Create model directory
        model_dir = self.base_path / model_name / version
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        if model_format == ModelFormat.PICKLE:
            model_path = model_dir / "model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        elif model_format == ModelFormat.JOBLIB:
            model_path = model_dir / "model.joblib"
            joblib.dump(model, model_path)
        elif model_format == ModelFormat.ONNX:
            if not ONNX_AVAILABLE:
                raise RuntimeError("ONNX not available")
            model_path = model_dir / "model.onnx"
            # Assume model is already in ONNX format
            onnx.save(model, str(model_path))
        else:
            raise ValueError(f"Unsupported model format: {model_format}")
        
        # Compute hash
        model_hash = self._compute_model_hash(model_path)
        
        # Create metadata
        metadata = ModelMetadata(
            model_name=model_name,
            version=version,
            created_at=datetime.now(),
            model_format=model_format,
            framework=framework,
            metrics=metrics or {},
            features=features or [],
            target=target,
            model_hash=model_hash,
            tags=tags or {},
            environment=environment,
            deployment_strategy=deployment_strategy
        )
        
        # Save metadata to model directory
        with open(model_dir / "metadata.json", 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)
        
        # Update registry cache
        if model_name not in self.metadata_cache:
            self.metadata_cache[model_name] = []
        self.metadata_cache[model_name].append(metadata)
        self._save_metadata()
        
        logger.info(f"Registered model {model_name} version {version} ({model_format.value})")
        return metadata
    
    def get_model(
        self,
        model_name: str,
        version: Optional[str] = None
    ) -> Tuple[Any, ModelMetadata]:
        """
        Retrieve a model from the registry.
        
        Args:
            model_name: Name of the model
            version: Version string (defaults to latest)
            
        Returns:
            Tuple of (model, metadata)
        """
        if model_name not in self.metadata_cache:
            raise ValueError(f"Model {model_name} not found in registry")
        
        versions = self.metadata_cache[model_name]
        if version is None:
            # Get latest version
            metadata = sorted(versions, key=lambda x: x.created_at)[-1]
        else:
            metadata = next((v for v in versions if v.version == version), None)
            if metadata is None:
                raise ValueError(f"Version {version} not found for model {model_name}")
        
        # Load model
        model_dir = self.base_path / model_name / metadata.version
        
        if metadata.model_format == ModelFormat.PICKLE:
            model_path = model_dir / "model.pkl"
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        elif metadata.model_format == ModelFormat.JOBLIB:
            model_path = model_dir / "model.joblib"
            model = joblib.load(model_path)
        elif metadata.model_format == ModelFormat.ONNX:
            model_path = model_dir / "model.onnx"
            model = ort.InferenceSession(str(model_path))
        else:
            raise ValueError(f"Unsupported model format: {metadata.model_format}")
        
        # Verify hash
        current_hash = self._compute_model_hash(model_path)
        if current_hash != metadata.model_hash:
            logger.warning(f"Model hash mismatch for {model_name} {metadata.version}")
        
        logger.info(f"Loaded model {model_name} version {metadata.version}")
        return model, metadata
    
    def list_models(self) -> Dict[str, List[str]]:
        """List all models and their versions."""
        return {
            model_name: [v.version for v in versions]
            for model_name, versions in self.metadata_cache.items()
        }
    
    def get_latest_version(self, model_name: str) -> str:
        """Get latest version of a model."""
        if model_name not in self.metadata_cache:
            raise ValueError(f"Model {model_name} not found")
        versions = self.metadata_cache[model_name]
        latest = sorted(versions, key=lambda x: x.created_at)[-1]
        return latest.version
    
    def promote_model(
        self,
        model_name: str,
        version: str,
        to_environment: EnvironmentType
    ) -> None:
        """Promote model to different environment."""
        if model_name not in self.metadata_cache:
            raise ValueError(f"Model {model_name} not found")
        
        for metadata in self.metadata_cache[model_name]:
            if metadata.version == version:
                metadata.environment = to_environment
                self._save_metadata()
                logger.info(f"Promoted {model_name} {version} to {to_environment.value}")
                return
        
        raise ValueError(f"Version {version} not found for model {model_name}")


class ModelDeployer:
    """
    Comprehensive model deployment orchestrator.
    
    Supports multiple deployment strategies including REST API, batch scoring,
    A/B testing, canary deployments, and blue-green deployments.
    
    Example:
        >>> deployer = ModelDeployer()
        >>> deployer.deploy_rest_api(
        ...     model_name="credit_risk",
        ...     version="v1.0.0",
        ...     port=8000,
        ...     strategy=DeploymentStrategy.BLUE_GREEN
        ... )
    """
    
    def __init__(
        self,
        model_registry_path: Union[str, Path] = "./model_registry",
        deployment_path: Union[str, Path] = "./deployments"
    ):
        self.registry = ModelRegistry(model_registry_path)
        self.deployment_path = Path(deployment_path)
        self.deployment_path.mkdir(parents=True, exist_ok=True)
        self.active_deployments: Dict[str, Dict[str, Any]] = {}
        self.app: Optional[FastAPI] = None
        self.start_time = datetime.now()
        self.request_count = 0
        self.latencies: List[float] = []
        logger.info("Initialized ModelDeployer")
    
    def create_rest_api(
        self,
        model_name: str,
        version: Optional[str] = None,
        enable_ab_testing: bool = False,
        ab_models: Optional[Dict[str, Tuple[str, float]]] = None
    ) -> FastAPI:
        """
        Create FastAPI application for model serving.
        
        Args:
            model_name: Name of the model
            version: Model version (defaults to latest)
            enable_ab_testing: Enable A/B testing
            ab_models: Dict mapping variant names to (version, traffic_percentage)
            
        Returns:
            FastAPI application
        """
        app = FastAPI(
            title=f"{model_name} Prediction API",
            description="Production ML model serving API",
            version="1.0.0"
        )
        
        # Load model
        model, metadata = self.registry.get_model(model_name, version)
        self.current_model = model
        self.current_metadata = metadata
        
        # A/B testing setup
        self.ab_testing_enabled = enable_ab_testing
        self.ab_models = {}
        if enable_ab_testing and ab_models:
            for variant, (ver, traffic) in ab_models.items():
                m, md = self.registry.get_model(model_name, ver)
                self.ab_models[variant] = {
                    'model': m,
                    'metadata': md,
                    'traffic': traffic
                }
            logger.info(f"A/B testing enabled with {len(self.ab_models)} variants")
        
        @app.get("/", response_model=Dict[str, str])
        async def root():
            """Root endpoint."""
            return {
                "message": f"{model_name} Prediction API",
                "version": metadata.version,
                "status": "active"
            }
        
        @app.get("/health", response_model=HealthResponse)
        async def health():
            """Health check endpoint."""
            uptime = (datetime.now() - self.start_time).total_seconds()
            avg_latency = np.mean(self.latencies) if self.latencies else 0.0
            
            return HealthResponse(
                status="healthy",
                model_loaded=True,
                model_name=model_name,
                model_version=metadata.version,
                uptime_seconds=uptime,
                requests_served=self.request_count,
                avg_latency_ms=avg_latency
            )
        
        @app.post("/predict", response_model=PredictionResponse)
        async def predict(request: PredictionRequest):
            """Prediction endpoint."""
            start_time = datetime.now()
            
            try:
                # Select model (A/B testing)
                selected_model = self.current_model
                selected_metadata = self.current_metadata
                
                if self.ab_testing_enabled and self.ab_models:
                    # Simple random routing based on traffic split
                    rand = np.random.random() * 100
                    cumulative = 0
                    for variant, config in self.ab_models.items():
                        cumulative += config['traffic']
                        if rand < cumulative:
                            selected_model = config['model']
                            selected_metadata = config['metadata']
                            break
                
                # Prepare features
                if isinstance(request.features, dict):
                    # Single prediction
                    features_list = [request.features]
                else:
                    features_list = request.features
                
                # Convert to DataFrame
                X = pd.DataFrame(features_list)
                
                # Make predictions
                if hasattr(selected_model, 'predict'):
                    predictions = selected_model.predict(X).tolist()
                else:
                    raise ValueError("Model does not have predict method")
                
                # Get probabilities if requested
                probabilities = None
                if request.return_probabilities and hasattr(selected_model, 'predict_proba'):
                    proba = selected_model.predict_proba(X)
                    probabilities = [
                        {f"class_{i}": float(p) for i, p in enumerate(prob)}
                        for prob in proba
                    ]
                
                # Calculate latency
                latency = (datetime.now() - start_time).total_seconds() * 1000
                self.latencies.append(latency)
                self.request_count += 1
                
                return PredictionResponse(
                    predictions=predictions,
                    model_name=model_name,
                    model_version=selected_metadata.version,
                    timestamp=datetime.now().isoformat(),
                    latency_ms=latency,
                    probabilities=probabilities
                )
                
            except Exception as e:
                logger.error(f"Prediction error: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/model/info")
        async def model_info():
            """Get model information."""
            return {
                "model_name": model_name,
                "version": metadata.version,
                "framework": metadata.framework,
                "features": metadata.features,
                "metrics": metadata.metrics,
                "created_at": metadata.created_at.isoformat(),
                "environment": metadata.environment.value
            }
        
        @app.get("/metrics")
        async def metrics():
            """Get performance metrics (Prometheus format)."""
            uptime = (datetime.now() - self.start_time).total_seconds()
            avg_latency = np.mean(self.latencies) if self.latencies else 0.0
            p95_latency = np.percentile(self.latencies, 95) if self.latencies else 0.0
            p99_latency = np.percentile(self.latencies, 99) if self.latencies else 0.0
            
            prometheus_metrics = f"""# HELP model_requests_total Total number of prediction requests
# TYPE model_requests_total counter
model_requests_total{{{{"model":"{model_name}","version":"{metadata.version}"}}}} {self.request_count}

# HELP model_latency_ms Prediction latency in milliseconds
# TYPE model_latency_ms gauge
model_latency_ms_avg{{{{"model":"{model_name}","version":"{metadata.version}"}}}} {avg_latency:.2f}
model_latency_ms_p95{{{{"model":"{model_name}","version":"{metadata.version}"}}}} {p95_latency:.2f}
model_latency_ms_p99{{{{"model":"{model_name}","version":"{metadata.version}"}}}} {p99_latency:.2f}

# HELP model_uptime_seconds Model uptime in seconds
# TYPE model_uptime_seconds gauge
model_uptime_seconds{{{{"model":"{model_name}","version":"{metadata.version}"}}}} {uptime:.2f}
"""
            return Response(content=prometheus_metrics, media_type="text/plain")
        
        self.app = app
        return app
    
    def deploy_rest_api(
        self,
        model_name: str,
        version: Optional[str] = None,
        port: int = 8000,
        host: str = "0.0.0.0",
        enable_ab_testing: bool = False,
        ab_models: Optional[Dict[str, Tuple[str, float]]] = None,
        reload: bool = False
    ) -> None:
        """
        Deploy model as REST API.
        
        Args:
            model_name: Name of the model
            version: Model version
            port: Port number
            host: Host address
            enable_ab_testing: Enable A/B testing
            ab_models: A/B test configuration
            reload: Auto-reload on code changes
        """
        app = self.create_rest_api(
            model_name=model_name,
            version=version,
            enable_ab_testing=enable_ab_testing,
            ab_models=ab_models
        )
        
        logger.info(f"Starting REST API server on {host}:{port}")
        uvicorn.run(app, host=host, port=port, reload=reload)
    
    def deploy_batch(
        self,
        model_name: str,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        version: Optional[str] = None,
        batch_size: int = 1000,
        schedule: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Deploy model for batch scoring.
        
        Args:
            model_name: Name of the model
            input_path: Path to input data
            output_path: Path to save predictions
            version: Model version
            batch_size: Batch size for predictions
            schedule: Cron schedule (optional, for scheduled jobs)
            
        Returns:
            DataFrame with predictions
        """
        logger.info(f"Starting batch deployment for {model_name}")
        
        # Load model
        model, metadata = self.registry.get_model(model_name, version)
        
        # Load data
        input_path = Path(input_path)
        if input_path.suffix == '.csv':
            data = pd.read_csv(input_path)
        elif input_path.suffix == '.parquet':
            data = pd.read_parquet(input_path)
        else:
            raise ValueError(f"Unsupported file format: {input_path.suffix}")
        
        logger.info(f"Loaded {len(data)} records from {input_path}")
        
        # Make predictions in batches
        predictions = []
        probabilities = []
        
        for i in range(0, len(data), batch_size):
            batch = data.iloc[i:i+batch_size]
            batch_preds = model.predict(batch)
            predictions.extend(batch_preds)
            
            if hasattr(model, 'predict_proba'):
                batch_proba = model.predict_proba(batch)
                probabilities.extend(batch_proba)
            
            if (i // batch_size + 1) % 10 == 0:
                logger.info(f"Processed {i + len(batch)}/{len(data)} records")
        
        # Create output DataFrame
        result = data.copy()
        result['prediction'] = predictions
        
        if probabilities:
            proba_df = pd.DataFrame(
                probabilities,
                columns=[f'prob_class_{i}' for i in range(len(probabilities[0]))]
            )
            result = pd.concat([result, proba_df], axis=1)
        
        result['model_version'] = metadata.version
        result['prediction_timestamp'] = datetime.now()
        
        # Save results
        output_path = Path(output_path)
        if output_path.suffix == '.csv':
            result.to_csv(output_path, index=False)
        elif output_path.suffix == '.parquet':
            result.to_parquet(output_path, index=False)
        
        logger.info(f"Saved predictions to {output_path}")
        return result
    
    def create_docker_image(
        self,
        model_name: str,
        version: Optional[str] = None,
        base_image: str = "python:3.10-slim",
        tag: Optional[str] = None
    ) -> str:
        """
        Create Docker image for model deployment.
        
        Args:
            model_name: Name of the model
            version: Model version
            base_image: Base Docker image
            tag: Docker image tag
            
        Returns:
            Docker image tag
        """
        if version is None:
            version = self.registry.get_latest_version(model_name)
        
        if tag is None:
            tag = f"{model_name}:{version}"
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Copy model files
            model_dir = self.registry.base_path / model_name / version
            shutil.copytree(model_dir, tmpdir / "model")
            
            # Create requirements.txt
            requirements = [
                "fastapi>=0.103.0",
                "uvicorn>=0.23.0",
                "pandas>=2.0.0",
                "numpy>=1.24.0",
                "scikit-learn>=1.3.0",
                "joblib>=1.3.0",
                "loguru>=0.7.0",
                "pydantic>=2.3.0"
            ]
            with open(tmpdir / "requirements.txt", 'w') as f:
                f.write('\n'.join(requirements))
            
            # Create Dockerfile
            dockerfile = f"""FROM {base_image}

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model
COPY model/ /app/model/

# Copy deployment script
COPY deployment.py /app/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
  CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run API server
CMD ["python", "-m", "uvicorn", "deployment:app", "--host", "0.0.0.0", "--port", "8000"]
"""
            with open(tmpdir / "Dockerfile", 'w') as f:
                f.write(dockerfile)
            
            # Copy deployment script
            shutil.copy(__file__, tmpdir / "deployment.py")
            
            # Build Docker image
            logger.info(f"Building Docker image {tag}")
            result = subprocess.run(
                ["docker", "build", "-t", tag, "."],
                cwd=tmpdir,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                logger.error(f"Docker build failed: {result.stderr}")
                raise RuntimeError(f"Docker build failed: {result.stderr}")
            
            logger.info(f"Successfully built Docker image {tag}")
            return tag
    
    def validate_deployment(
        self,
        model_name: str,
        version: str,
        test_data: pd.DataFrame,
        expected_predictions: Optional[np.ndarray] = None,
        tolerance: float = 1e-6
    ) -> bool:
        """
        Validate deployment by comparing predictions.
        
        Args:
            model_name: Name of the model
            version: Model version
            test_data: Test data
            expected_predictions: Expected predictions for validation
            tolerance: Tolerance for prediction comparison
            
        Returns:
            True if validation passes
        """
        logger.info(f"Validating deployment for {model_name} {version}")
        
        # Load model
        model, metadata = self.registry.get_model(model_name, version)
        
        # Make predictions
        predictions = model.predict(test_data)
        
        # Compare with expected if provided
        if expected_predictions is not None:
            if not np.allclose(predictions, expected_predictions, atol=tolerance):
                logger.error("Validation failed: predictions do not match expected")
                return False
        
        # Check prediction shape
        if len(predictions) != len(test_data):
            logger.error(f"Validation failed: prediction count mismatch")
            return False
        
        # Check for NaN/Inf
        if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
            logger.error("Validation failed: predictions contain NaN or Inf")
            return False
        
        logger.info("Deployment validation passed")
        return True
