"""
Fraud Detection Use Case

Complete fraud detection system with graph-based analysis, supervised and
unsupervised learning, risk propagation, and production-ready deployment.

Example:
    >>> from src.use_cases.fraud_detection import FraudDetectionPipeline
    >>> 
    >>> # Train pipeline
    >>> pipeline = FraudDetectionPipeline()
    >>> pipeline.fit(transactions_df)
    >>> 
    >>> # Real-time scoring
    >>> result = pipeline.predict_single_transaction(transaction)
    >>> print(f"Fraud score: {result['fraud_score']:.3f}")
"""

from src.use_cases.fraud_detection.graph_builder import GraphBuilder
from src.use_cases.fraud_detection.supervised_fraud import (
    FraudClassifier,
    FraudFeatureEngineer,
    FraudNeuralNet
)
from src.use_cases.fraud_detection.anomaly_detection import (
    IsolationForestDetector,
    LocalOutlierFactorDetector,
    AutoencoderDetector,
    EnsembleAnomalyDetector,
    AnomalyScoreCalibrator
)
from src.use_cases.fraud_detection.graph_embeddings import (
    Node2VecEmbeddings,
    DeepWalkEmbeddings,
    GraphSAGEEmbeddings,
    CommunityEmbeddings,
    EmbeddingQualityMetrics,
    TemporalGraphEmbeddings
)
from src.use_cases.fraud_detection.risk_propagation import (
    PageRankRiskPropagation,
    LabelPropagationRisk,
    NetworkInfluenceScoring,
    IterativeRiskScoring,
    FraudRingDetector
)
from src.use_cases.fraud_detection.fraud_pipeline import (
    FraudDetectionPipeline,
    FraudPipelineConfig,
    ABTestingFramework
)

__all__ = [
    # Graph
    'GraphBuilder',
    
    # Supervised
    'FraudClassifier',
    'FraudFeatureEngineer',
    'FraudNeuralNet',
    
    # Anomaly Detection
    'IsolationForestDetector',
    'LocalOutlierFactorDetector',
    'AutoencoderDetector',
    'EnsembleAnomalyDetector',
    'AnomalyScoreCalibrator',
    
    # Graph Embeddings
    'Node2VecEmbeddings',
    'DeepWalkEmbeddings',
    'GraphSAGEEmbeddings',
    'CommunityEmbeddings',
    'EmbeddingQualityMetrics',
    'TemporalGraphEmbeddings',
    
    # Risk Propagation
    'PageRankRiskPropagation',
    'LabelPropagationRisk',
    'NetworkInfluenceScoring',
    'IterativeRiskScoring',
    'FraudRingDetector',
    
    # Pipeline
    'FraudDetectionPipeline',
    'FraudPipelineConfig',
    'ABTestingFramework'
]
