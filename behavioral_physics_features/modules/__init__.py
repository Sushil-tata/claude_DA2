"""
Behavioral Physics Feature Factory - Module Exports
===================================================

Production-ready PySpark feature engineering system using
behavioral physics concepts.

Author: Behavioral Physics Team
Version: 1.0.0
"""

__version__ = "1.0.0"

# Core configuration
from .config import (
    get_config,
    StateConfig,
    LenderTypeConfig,
    WindowConfig,
    ThresholdConfig,
    QualityConfig
)

# Schema adapter
from .bureau_schema_adapter import BureauSchemaAdapter

# Feature engines
from .state_builder import StateBuilder
from .trajectory_engine import TrajectoryEngine
from .lender_ecology import LenderEcologyEngine
from .repayment_dynamics import RepaymentDynamicsEngine
from .enquiries_engine import EnquiriesEngine
from .cardx_bureau_interactions import CardXBureauInteractionsEngine
from .legal_actions import LegalActionsEngine
from .tdr_restructuring import TDRRestructuringEngine

# Registry and orchestration
from .feature_registry import (
    FeatureRegistry,
    FeatureMetadata,
    FeatureType,
    FeatureStability
)
from .main_pipeline import BehavioralPhysicsPipeline

# Module registry
MODULES = {
    "config": get_config,
    "state_builder": StateBuilder,
    "trajectory_engine": TrajectoryEngine,
    "lender_ecology": LenderEcologyEngine,
    "repayment_dynamics": RepaymentDynamicsEngine,
    "enquiries_engine": EnquiriesEngine,
    "cardx_bureau_interactions": CardXBureauInteractionsEngine,
    "legal_actions": LegalActionsEngine,
    "tdr_restructuring": TDRRestructuringEngine,
    "feature_registry": FeatureRegistry,
    "main_pipeline": BehavioralPhysicsPipeline
}

# Feature counts by module
FEATURE_COUNTS = {
    "state_builder": 8,                    # State and regime assignments
    "trajectory_engine": 60,               # Velocity, acceleration, transitions, entropy
    "lender_ecology": 25,                  # Cross-lender dynamics
    "repayment_dynamics": 35,              # NORMAL/STRESSED/delta features
    "enquiries_engine": 12,                # Enquiry patterns
    "cardx_bureau_interactions": 20,       # CardX-Bureau interactions
    "legal_actions": 18,                   # Legal status and settlements
    "tdr_restructuring": 22,               # TDR/restructuring dynamics
    "total": 200
}

__all__ = [
    # Version
    "__version__",

    # Config
    "get_config",
    "StateConfig",
    "LenderTypeConfig",
    "WindowConfig",
    "ThresholdConfig",
    "QualityConfig",

    # Schema Adapter
    "BureauSchemaAdapter",

    # Engines
    "StateBuilder",
    "TrajectoryEngine",
    "LenderEcologyEngine",
    "RepaymentDynamicsEngine",
    "EnquiriesEngine",
    "CardXBureauInteractionsEngine",
    "LegalActionsEngine",
    "TDRRestructuringEngine",

    # Registry
    "FeatureRegistry",
    "FeatureMetadata",
    "FeatureType",
    "FeatureStability",

    # Pipeline
    "BehavioralPhysicsPipeline",

    # Module info
    "MODULES",
    "FEATURE_COUNTS"
]
