"""
Recommender Systems module for Principal Data Science Decision Agent.

This module provides state-of-the-art recommendation algorithms including
contextual bandits, uplift modeling, and learning-to-rank models.

Modules:
    - contextual_bandits: Multi-armed bandits for next best action
    - uplift_model: Causal uplift modeling for treatment optimization
    - ranking_model: Learning-to-rank models for recommendation
"""

# Contextual Bandits
from .contextual_bandits import (
    BaseContextualBandit,
    EpsilonGreedyBandit,
    UCBBandit,
    ThompsonSamplingBandit,
    LinUCBBandit,
    ContextualBanditOrchestrator,
    OnlineBanditTrainer
)

# Uplift Models
from .uplift_model import (
    BaseUpliftModel,
    TLearner,
    SLearner,
    XLearner,
    CausalForestUplift,
    UpliftEnsemble,
    UpliftValidator
)

# Ranking Models
from .ranking_model import (
    BaseRankingModel,
    LambdaMARTRanker,
    PairwiseRanker,
    ListwiseRanker,
    NDCGOptimizer,
    RankingEnsemble,
    PositionBiasCorrector,
    RankingMetrics
)

__all__ = [
    # Contextual Bandits
    'BaseContextualBandit',
    'EpsilonGreedyBandit',
    'UCBBandit',
    'ThompsonSamplingBandit',
    'LinUCBBandit',
    'ContextualBanditOrchestrator',
    'OnlineBanditTrainer',
    
    # Uplift Models
    'BaseUpliftModel',
    'TLearner',
    'SLearner',
    'XLearner',
    'CausalForestUplift',
    'UpliftEnsemble',
    'UpliftValidator',
    
    # Ranking Models
    'BaseRankingModel',
    'LambdaMARTRanker',
    'PairwiseRanker',
    'ListwiseRanker',
    'NDCGOptimizer',
    'RankingEnsemble',
    'PositionBiasCorrector',
    'RankingMetrics',
]

__version__ = '1.0.0'
