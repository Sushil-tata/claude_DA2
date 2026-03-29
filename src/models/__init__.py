"""
Models package for the Principal Data Science Decision Agent.

This package provides a comprehensive suite of machine learning models including:
- Tree-based models (LightGBM, XGBoost, CatBoost, RandomForest)
- Neural tabular models (TabNet, TabPFN, NODE, DeepGBM)
- Ensemble methods (Weighted Average, Stacking, Blending, Segment-wise, Hybrid)
- Unsupervised learning (Clustering, Dimensionality Reduction, Autoencoder Clustering)
- AutoML and meta-learning (Bayesian Optimization, Multi-objective Optimization)
"""

# Tree-based models
from src.models.tree_models import (
    LightGBMModel,
    XGBoostModel,
    CatBoostModel,
    RandomForestModel,
    BaseTreeModel
)

# Neural tabular models
from src.models.neural_tabular import (
    TabNetModel,
    TabPFNModel,
    NODEModel,
    DeepGBMModel,
    BaseNeuralTabular
)

# Ensemble methods
from src.models.ensemble_engine import (
    WeightedAverageEnsemble,
    StackingEnsemble,
    BlendingEnsemble,
    SegmentWiseEnsemble,
    HybridRuleMLEnsemble,
    compare_ensemble_performance
)

# Unsupervised learning
from src.models.unsupervised import (
    ClusteringEngine,
    DimensionalityReduction,
    AutoencoderClustering
)

# Meta-learning and AutoML
from src.models.meta_learner import (
    BayesianOptimizer,
    GeneticAlgorithmOptimizer,
    MultiObjectiveOptimizer,
    HyperparameterSearchSpace,
    AutoMLEngine
)

__all__ = [
    # Tree models
    'LightGBMModel',
    'XGBoostModel',
    'CatBoostModel',
    'RandomForestModel',
    'BaseTreeModel',
    
    # Neural tabular
    'TabNetModel',
    'TabPFNModel',
    'NODEModel',
    'DeepGBMModel',
    'BaseNeuralTabular',
    
    # Ensemble
    'WeightedAverageEnsemble',
    'StackingEnsemble',
    'BlendingEnsemble',
    'SegmentWiseEnsemble',
    'HybridRuleMLEnsemble',
    'compare_ensemble_performance',
    
    # Unsupervised
    'ClusteringEngine',
    'DimensionalityReduction',
    'AutoencoderClustering',
    
    # Meta-learning
    'BayesianOptimizer',
    'GeneticAlgorithmOptimizer',
    'MultiObjectiveOptimizer',
    'HyperparameterSearchSpace',
    'AutoMLEngine',
]
