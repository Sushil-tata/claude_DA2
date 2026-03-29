"""
Privacy-Preserving Machine Learning Layer.

This module provides comprehensive privacy-preserving ML capabilities including:
- Federated learning with secure aggregation
- Transfer learning with privacy guarantees
- Differential privacy mechanisms
- Privacy budget tracking and compliance
- Multi-institution coordination

References:
    - McMahan et al. (2017): Communication-Efficient Learning of Deep Networks
    - Abadi et al. (2016): Deep Learning with Differential Privacy
    - Dwork & Roth (2014): The Algorithmic Foundations of Differential Privacy
"""

from .federated_learning import (
    FederatedLearningServer,
    FederatedLearningClient,
    FedAvgAggregator,
    FedProxAggregator,
    PrivacyBudgetTracker,
    DifferentialPrivacyConfig,
    PrivacyBudget,
    ClientUpdate,
)

from .transfer_learning import (
    TransferLearner,
    DomainAdapter,
    TransferabilityMetrics,
    DomainStatistics,
    TransferabilityScore,
)

__version__ = "1.0.0"

__all__ = [
    "FederatedLearningServer",
    "FederatedLearningClient",
    "FedAvgAggregator",
    "FedProxAggregator",
    "PrivacyBudgetTracker",
    "DifferentialPrivacyConfig",
    "PrivacyBudget",
    "ClientUpdate",
    "TransferLearner",
    "DomainAdapter",
    "TransferabilityMetrics",
    "DomainStatistics",
    "TransferabilityScore",
]
