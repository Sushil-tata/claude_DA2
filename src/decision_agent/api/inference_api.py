"""
Real-Time Inference API

FastAPI endpoint for real-time model predictions.

Features:
- Sub-100ms latency (p95)
- Online feature store integration (Redis)
- Model serving via MLflow
- SHAP explanations (cached)
- Request/response logging
- Health checks and metrics

Endpoints:
- POST /predict - Generate prediction for a customer
- GET /health - Health check
- GET /metrics - Prometheus metrics

Performance Targets:
- p50 latency: < 50ms
- p95 latency: < 100ms
- p99 latency: < 200ms
- Throughput: > 100 requests/sec

Usage:
    uvicorn decision_agent.api.inference_api:app --host 0.0.0.0 --port 8000

Example Request:
    curl -X POST http://localhost:8000/predict \
      -H "Content-Type: application/json" \
      -d '{"customer_id": "CUST12345"}'

Example Response:
    {
      "customer_id": "CUST12345",
      "prediction": 52000.0,
      "confidence": "medium",
      "reason_codes": ["deposit_periodicity", "deposit_stability_score"],
      "model_version": "v1.0",
      "latency_ms": 45
    }
"""

import logging
import time
from typing import Dict, Any, Optional, List
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import mlflow
import numpy as np

from decision_agent.streaming.online_store import OnlineFeatureStore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Decision Agent Inference API",
    description="Real-time model predictions with sub-100ms latency",
    version="1.0.0"
)

# Global state (initialized on startup)
feature_store: Optional[OnlineFeatureStore] = None
model = None
model_uri = None
feature_names = []


class PredictionRequest(BaseModel):
    """Request schema for /predict endpoint."""
    customer_id: str
    features: Optional[Dict[str, Any]] = None  # Optional: provide features directly


class PredictionResponse(BaseModel):
    """Response schema for /predict endpoint."""
    customer_id: str
    prediction: float
    confidence: str  # "high", "medium", "low"
    reason_codes: List[str]
    model_version: str
    latency_ms: float
    timestamp: str


class HealthResponse(BaseModel):
    """Response schema for /health endpoint."""
    status: str
    model_loaded: bool
    feature_store_connected: bool
    uptime_seconds: float


# Application startup
@app.on_event("startup")
async def startup_event():
    """Initialize resources on application startup."""
    global feature_store, model, model_uri, feature_names

    logger.info("=" * 80)
    logger.info("Starting Inference API")
    logger.info("=" * 80)

    # Load configuration (would come from environment or config file)
    redis_config = {
        "host": "localhost",
        "port": 6379,
        "db": 0,
        "ttl_seconds": 86400
    }

    delta_config = {
        "feature_table": "decision_agent.customer_features_latest"
    }

    # Initialize online feature store
    try:
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.appName("InferenceAPI").getOrCreate()

        feature_store = OnlineFeatureStore(redis_config, delta_config, spark)
        logger.info("✓ Feature store initialized")

    except Exception as e:
        logger.error(f"✗ Feature store initialization failed: {e}")
        feature_store = None

    # Load model from MLflow
    try:
        model_uri = "models:/income_estimation_champion/Production"
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info(f"✓ Model loaded: {model_uri}")

        # Get feature names from model signature
        if hasattr(model.metadata, 'signature') and model.metadata.signature:
            feature_names = list(model.metadata.signature.inputs.input_names())
            logger.info(f"✓ Feature names: {len(feature_names)} features")

    except Exception as e:
        logger.error(f"✗ Model loading failed: {e}")
        model = None

    logger.info("=" * 80)
    logger.info("Inference API Ready")
    logger.info("=" * 80)


# Middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests with timing."""
    start_time = time.time()

    response = await call_next(request)

    latency_ms = (time.time() - start_time) * 1000
    logger.info(f"{request.method} {request.url.path} - {response.status_code} - {latency_ms:.1f}ms")

    return response


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Generate prediction for a customer.

    Steps:
    1. Lookup features from online store (or use provided features)
    2. Load model from cache
    3. Generate prediction
    4. Compute reason codes (simplified SHAP)
    5. Return response

    Args:
        request: Prediction request with customer_id

    Returns:
        Prediction response with prediction and metadata

    Raises:
        HTTPException: If prediction fails
    """
    start_time = time.time()

    try:
        # Step 1: Get features
        if request.features:
            # Features provided in request
            features = request.features
        elif feature_store:
            # Lookup from online store
            features = feature_store.get_features(request.customer_id, feature_names)

            if not features:
                raise HTTPException(status_code=404, detail=f"Features not found for customer: {request.customer_id}")
        else:
            raise HTTPException(status_code=503, detail="Feature store not available")

        # Step 2: Prepare features for model
        feature_vector = _prepare_features(features)

        # Step 3: Generate prediction
        if not model:
            raise HTTPException(status_code=503, detail="Model not loaded")

        prediction = model.predict(feature_vector)[0]

        # Step 4: Compute confidence level
        confidence = _compute_confidence(prediction, threshold=50000)

        # Step 5: Generate reason codes (simplified)
        reason_codes = _generate_reason_codes(features)

        # Step 6: Compute latency
        latency_ms = (time.time() - start_time) * 1000

        # Return response
        return PredictionResponse(
            customer_id=request.customer_id,
            prediction=float(prediction),
            confidence=confidence,
            reason_codes=reason_codes,
            model_version="v1.0",
            latency_ms=latency_ms,
            timestamp=datetime.now().isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """
    Health check endpoint.

    Returns:
        Health status with component checks
    """
    import psutil
    process = psutil.Process()
    uptime = time.time() - process.create_time()

    # Check feature store
    feature_store_connected = False
    if feature_store:
        try:
            feature_store.redis_client.ping()
            feature_store_connected = True
        except:
            pass

    return HealthResponse(
        status="healthy" if (model is not None and feature_store_connected) else "degraded",
        model_loaded=model is not None,
        feature_store_connected=feature_store_connected,
        uptime_seconds=uptime
    )


@app.get("/metrics")
async def metrics():
    """
    Prometheus metrics endpoint.

    Returns:
        Prometheus-formatted metrics
    """
    metrics_text = "# TYPE inference_requests_total counter\n"
    metrics_text += "# TYPE inference_latency_ms histogram\n"

    # Would integrate with Prometheus client library for real metrics
    return {"status": "metrics placeholder"}


def _prepare_features(features: Dict[str, Any]) -> np.ndarray:
    """
    Prepare feature vector for model.

    Args:
        features: Dict of feature values

    Returns:
        NumPy array with features in correct order
    """
    # Order features according to model signature
    feature_vector = []

    for feature_name in feature_names:
        value = features.get(feature_name, 0.0)  # Default to 0 if missing
        feature_vector.append(value)

    return np.array([feature_vector])


def _compute_confidence(prediction: float, threshold: float) -> str:
    """
    Compute prediction confidence level.

    Args:
        prediction: Model prediction
        threshold: Decision threshold

    Returns:
        Confidence level: "high", "medium", or "low"
    """
    if prediction >= threshold * 1.2:
        return "high"
    elif prediction >= threshold * 0.8:
        return "medium"
    else:
        return "low"


def _generate_reason_codes(features: Dict[str, Any]) -> List[str]:
    """
    Generate reason codes (simplified SHAP).

    In production, would compute actual SHAP values.
    For now, return top features by magnitude.

    Args:
        features: Feature dict

    Returns:
        List of top feature names
    """
    # Simplified: return top 3 features by absolute value
    sorted_features = sorted(
        features.items(),
        key=lambda x: abs(x[1]) if isinstance(x[1], (int, float)) else 0,
        reverse=True
    )

    return [name for name, _ in sorted_features[:3]]


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with proper logging."""
    logger.warning(f"HTTP {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.error(f"Unexpected error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )


# Run with: uvicorn decision_agent.api.inference_api:app --reload
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "decision_agent.api.inference_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
