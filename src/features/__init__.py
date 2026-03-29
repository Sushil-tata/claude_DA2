"""
Features Engineering Module

Comprehensive feature engineering layer for the Principal Data Science Decision Agent.

Modules:
    - behavioral_features: Velocity, momentum, volatility, elasticity, stability features
    - temporal_features: Rolling windows, lags, leads, multi-resolution signals
    - liquidity_features: OTB, repayment buffers, installment impacts, utilization
    - persona_features: Clustering, segmentation, NLP-based features
    - graph_features: Node embeddings, centrality, community detection
    - feature_store: Registry, leakage detection, importance tracking, quality checks

Author: Principal Data Science Decision Agent
"""

from .behavioral_features import BehavioralFeatureEngine, BehavioralFeatureConfig
from .temporal_features import TemporalFeatureEngine, TemporalFeatureConfig
from .liquidity_features import LiquidityFeatureEngine, LiquidityFeatureConfig
from .persona_features import PersonaFeatureEngine, PersonaFeatureConfig
from .graph_features import GraphFeatureEngine, GraphFeatureConfig
from .feature_store import (
    FeatureStore,
    FeatureStoreConfig,
    FeatureRegistry,
    FeatureMetadata,
    LeakageDetector,
    FeatureImportanceTracker,
    FeatureQualityChecker,
)

__all__ = [
    # Behavioral
    "BehavioralFeatureEngine",
    "BehavioralFeatureConfig",
    # Temporal
    "TemporalFeatureEngine",
    "TemporalFeatureConfig",
    # Liquidity
    "LiquidityFeatureEngine",
    "LiquidityFeatureConfig",
    # Persona
    "PersonaFeatureEngine",
    "PersonaFeatureConfig",
    # Graph
    "GraphFeatureEngine",
    "GraphFeatureConfig",
    # Feature Store
    "FeatureStore",
    "FeatureStoreConfig",
    "FeatureRegistry",
    "FeatureMetadata",
    "LeakageDetector",
    "FeatureImportanceTracker",
    "FeatureQualityChecker",
]

__version__ = "1.0.0"
