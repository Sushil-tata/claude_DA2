"""
Behavioral Scoring Use Case

Complete behavioral scoring system with meta-learning, deep learning,
ensemble methods, and production-ready deployment.

This use case implements advanced behavioral scoring techniques for financial
risk assessment, leveraging transaction sequences, behavioral patterns, and
customer segments.

Key Features:
- Meta-learning for quick adaptation to new customer segments
- Deep learning models (LSTM, Transformer, CNN) for temporal patterns
- Multi-model ensembles with diversity optimization
- End-to-end pipeline with feature engineering integration
- Real-time and batch scoring capabilities
- Performance monitoring and model degradation detection
- Temporal validation with walk-forward testing
- Score calibration for reliable probability estimates

Example:
    >>> from src.use_cases.behavioral_scoring import BehavioralScoringPipeline
    >>> 
    >>> # Initialize pipeline
    >>> pipeline = BehavioralScoringPipeline(
    ...     feature_config="config/feature_config.yaml",
    ...     use_deep_learning=True,
    ...     use_meta_learning=True,
    ...     ensemble_type="stacking"
    ... )
    >>> 
    >>> # Train on historical data
    >>> pipeline.fit(
    ...     transactions_df=transactions,
    ...     labels_df=labels,
    ...     segments_df=segments
    ... )
    >>> 
    >>> # Real-time scoring
    >>> score = pipeline.score_single(
    ...     user_id="12345",
    ...     transaction_data=recent_transactions,
    ...     segment="premium"
    ... )
    >>> print(f"Risk score: {score['score']:.3f}")
    >>> 
    >>> # Batch scoring
    >>> scores_df = pipeline.score_batch(
    ...     transactions_df=test_transactions,
    ...     segments_df=test_segments
    ... )
    >>> 
    >>> # Monitor performance
    >>> metrics = pipeline.monitor_performance(
    ...     transactions_df=new_data,
    ...     labels_df=new_labels
    ... )
    >>> if metrics['performance_degraded']:
    ...     print("Model needs retraining")
"""

# Meta-learning
from src.use_cases.behavioral_scoring.meta_scoring import (
    MetaFeatureExtractor,
    MAMLAdapter,
    TransferLearningScorer,
    MetaScoringEngine
)

# Deep learning
from src.use_cases.behavioral_scoring.deep_scoring import (
    SequencePreprocessor,
    DeepScoringInterpreter
)

try:
    from src.use_cases.behavioral_scoring.deep_scoring import (
        LSTMScoringModel,
        TransformerScoringModel,
        TemporalCNN
    )
    DEEP_MODELS_AVAILABLE = True
except ImportError:
    DEEP_MODELS_AVAILABLE = False

# Ensemble
from src.use_cases.behavioral_scoring.ensemble_scoring import (
    DiversityOptimizer,
    TemporalWeightCalculator,
    EnsembleCalibrator,
    BehavioralEnsembleScorer
)

# Pipeline
from src.use_cases.behavioral_scoring.scoring_pipeline import (
    TransactionPreprocessor,
    TemporalValidator,
    PerformanceMonitor,
    BehavioralScoringPipeline
)

__all__ = [
    # Meta-learning
    'MetaFeatureExtractor',
    'MAMLAdapter',
    'TransferLearningScorer',
    'MetaScoringEngine',
    
    # Deep learning utilities
    'SequencePreprocessor',
    'DeepScoringInterpreter',
    
    # Ensemble
    'DiversityOptimizer',
    'TemporalWeightCalculator',
    'EnsembleCalibrator',
    'BehavioralEnsembleScorer',
    
    # Pipeline
    'TransactionPreprocessor',
    'TemporalValidator',
    'PerformanceMonitor',
    'BehavioralScoringPipeline',
]

# Add deep learning models if available
if DEEP_MODELS_AVAILABLE:
    __all__.extend([
        'LSTMScoringModel',
        'TransformerScoringModel',
        'TemporalCNN'
    ])
