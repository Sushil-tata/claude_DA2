"""
Behavioral Feature Engineering (BFE) Repository
================================================

A self-contained, versioned module that enriches Decision Engine capabilities
with behavioral features.

PHILOSOPHY:
- ADDITIVE: Augments, never overwrites existing features
- QUERYABLE: Any agent can call get_features()
- VERSIONED: Features tagged with BFE_v{major}.{minor}
- AUDITABLE: Every computation is traceable
- POINT-IN-TIME SAFE: No look-ahead bias

USAGE:
    from decision_engine.feature_store.behavioral_feature_engine import get_features, list_features

    # Get features for an account
    features = get_features(
        account_id="ACC123456",
        feature_sets=["delinquency", "payment"],
        as_of_date="2024-01-31",
        account_history=account_df
    )

    # List available features
    feature_catalog = list_features()
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from datetime import datetime, date
import json
from pathlib import Path

from .modules import DelinquencyFeatureEngine, PaymentFeatureEngine
from .modules import get_delinquency_metadata, get_payment_metadata
from .modules.vintage import VintageFeatureEngine
from .modules.interactions import InteractionFeatureEngine


__version__ = "1.2.0"
__author__ = "Decision Engine Team"


class BehavioralFeatureEngine:
    """
    Unified interface for behavioral feature engineering.

    Orchestrates all feature modules and provides a single API for
    Decision Engine integration.
    """

    VERSION = "BFE_v1.2"

    # Available modules
    MODULES = {
        "delinquency": DelinquencyFeatureEngine,
        "payment": PaymentFeatureEngine,
        "vintage": VintageFeatureEngine,
        "interactions": InteractionFeatureEngine
        # Future modules will be added here:
        # "repayment_term_loan": RepaymentTermLoanFeatureEngine,
        # "utilization_revolving": UtilizationRevolvingFeatureEngine,
        # "balance_exposure": BalanceExposureFeatureEngine,
        # "bureau": BureauFeatureEngine,
        # "cross_product": CrossProductFeatureEngine,
        # "composite_indices": CompositeIndicesFeatureEngine
    }

    def __init__(self):
        """Initialize BFE repository"""
        self.module_instances = {}
        self._initialize_modules()

    def _initialize_modules(self):
        """Lazy initialization of feature modules"""
        for module_name, module_class in self.MODULES.items():
            # Modules are initialized on first use
            self.module_instances[module_name] = None

    def _get_module(self, module_name: str):
        """Get or initialize a feature module"""
        if module_name not in self.MODULES:
            raise ValueError(f"Unknown feature module: {module_name}. Available: {list(self.MODULES.keys())}")

        if self.module_instances[module_name] is None:
            # Initialize on first use
            module_class = self.MODULES[module_name]
            self.module_instances[module_name] = module_class()

        return self.module_instances[module_name]

    def get_features(
        self,
        account_id: str,
        account_history: Dict[str, pd.DataFrame],
        as_of_date: Union[str, datetime, date],
        feature_sets: List[str] = None
    ) -> Dict[str, float]:
        """
        Get behavioral features for an account.

        Args:
            account_id: Account identifier
            account_history: Dictionary of DataFrames with historical data
                {
                    "delinquency": DataFrame with DPD history,
                    "payment": DataFrame with payment history,
                    "balance": DataFrame with balance history,
                    ...
                }
            as_of_date: Reference date (point-in-time)
            feature_sets: List of feature sets to compute (default: ["all"])
                Options: "delinquency", "payment", "repayment_term_loan",
                        "utilization_revolving", "balance_exposure", "bureau",
                        "cross_product", "composite_indices", "all"

        Returns:
            Dictionary of features with BFE_v{version} tagging

        Example:
            features = bfe.get_features(
                account_id="ACC123456",
                account_history={
                    "delinquency": dpd_df,
                    "payment": payment_df
                },
                as_of_date="2024-01-31",
                feature_sets=["delinquency", "payment"]
            )
        """
        # Parse as_of_date
        if isinstance(as_of_date, str):
            as_of_date = pd.to_datetime(as_of_date).date()
        elif isinstance(as_of_date, datetime):
            as_of_date = as_of_date.date()

        # Determine which feature sets to compute
        feature_sets = feature_sets or ["all"]
        if "all" in feature_sets:
            feature_sets = list(self.MODULES.keys())

        # Validate feature sets
        for feature_set in feature_sets:
            if feature_set not in self.MODULES:
                raise ValueError(f"Unknown feature set: {feature_set}. Available: {list(self.MODULES.keys())}")

        # Compute features from each module
        all_features = {}

        # Build schema mappings from account_history keys
        schema_mappings = {
            "delinquency": {"account_id": "account_id", "date": "date", "dpd": "dpd"},
            "payment": {"account_id": "account_id", "date": "date",
                       "payment_amount": "payment_amount", "amount_due": "amount_due"}
        }

        for feature_set in feature_sets:
            # Skip interactions - computed after all base modules
            if feature_set == "interactions":
                continue

            # Check if data is available for this module
            if feature_set not in account_history and feature_set != "vintage":
                print(f"⚠️  No data provided for '{feature_set}' module. Skipping...")
                continue

            # Get module instance
            module = self._get_module(feature_set)

            # Compute features
            try:
                # Vintage module needs special handling
                if feature_set == "vintage":
                    module_features = module.compute_features(
                        account_history=account_history,
                        account_id=account_id,
                        as_of_date=as_of_date,
                        schema_mappings=schema_mappings
                    )
                else:
                    module_features = module.compute_features(
                        account_history=account_history[feature_set],
                        account_id=account_id,
                        as_of_date=as_of_date
                    )

                # Add prefix to avoid name collisions
                prefixed_features = {
                    f"{feature_set}.{key}": value
                    for key, value in module_features.items()
                    if key not in ["module_version", "module_name", "as_of_date", "months_of_history"]
                }

                all_features.update(prefixed_features)

            except Exception as e:
                print(f"❌ Error computing {feature_set} features: {str(e)}")
                # Continue with other modules rather than failing completely
                continue

        # Compute interactions last (needs all base features)
        if "interactions" in feature_sets:
            try:
                module = self._get_module("interactions")
                interaction_features = module.compute_features(all_features)

                # Add prefix
                prefixed_interactions = {
                    f"interactions.{key}": value
                    for key, value in interaction_features.items()
                }

                all_features.update(prefixed_interactions)
            except Exception as e:
                print(f"❌ Error computing interaction features: {str(e)}")

        # Add global metadata
        all_features["bfe_version"] = self.VERSION
        all_features["account_id"] = account_id
        all_features["as_of_date"] = as_of_date
        all_features["computed_at"] = datetime.now()

        return all_features

    def get_features_batch(
        self,
        accounts_history: Dict[str, Dict[str, pd.DataFrame]],
        as_of_date: Union[str, datetime, date],
        feature_sets: List[str] = None
    ) -> pd.DataFrame:
        """
        Get features for multiple accounts (batch processing).

        Args:
            accounts_history: Dictionary of account histories
                {
                    "ACC123456": {"delinquency": df1, "payment": df2, ...},
                    "ACC789012": {"delinquency": df3, "payment": df4, ...},
                    ...
                }
            as_of_date: Reference date
            feature_sets: List of feature sets to compute

        Returns:
            DataFrame with one row per account
        """
        all_account_features = []

        for account_id, account_history in accounts_history.items():
            try:
                features = self.get_features(
                    account_id=account_id,
                    account_history=account_history,
                    as_of_date=as_of_date,
                    feature_sets=feature_sets
                )
                all_account_features.append(features)
            except Exception as e:
                print(f"❌ Error processing account {account_id}: {str(e)}")
                continue

        return pd.DataFrame(all_account_features)

    def list_features(self, module: Optional[str] = None) -> Dict[str, Dict]:
        """
        List all available features with metadata.

        Args:
            module: Optional module name to filter (default: all modules)

        Returns:
            Dictionary of feature metadata
        """
        all_metadata = {}

        modules_to_list = [module] if module else list(self.MODULES.keys())

        for mod_name in modules_to_list:
            if mod_name not in self.MODULES:
                continue

            # Get metadata from module
            if mod_name == "delinquency":
                metadata = get_delinquency_metadata()
            elif mod_name == "payment":
                metadata = get_payment_metadata()
            else:
                metadata = {}

            # Add module prefix
            for feature_name, feature_meta in metadata.items():
                all_metadata[f"{mod_name}.{feature_name}"] = feature_meta

        return all_metadata

    def save_feature_registry(self, output_path: str = None):
        """
        Save feature registry to JSON file.

        Args:
            output_path: Path to save registry (default: feature_registry.json)
        """
        output_path = output_path or "feature_registry.json"

        registry = self.list_features()

        with open(output_path, 'w') as f:
            json.dump(registry, f, indent=2, default=str)

        print(f"✓ Feature registry saved to {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Global API Functions
# ─────────────────────────────────────────────────────────────────────────────

# Singleton instance
_bfe_engine = None

def _get_engine() -> BehavioralFeatureEngine:
    """Get or create global BFE engine instance"""
    global _bfe_engine
    if _bfe_engine is None:
        _bfe_engine = BehavioralFeatureEngine()
    return _bfe_engine


def get_features(
    account_id: str,
    account_history: Dict[str, pd.DataFrame],
    as_of_date: Union[str, datetime, date],
    feature_sets: List[str] = None
) -> Dict[str, float]:
    """
    Get behavioral features for an account.

    This is the main entry point for Decision Engine integration.

    Args:
        account_id: Account identifier
        account_history: Dictionary of historical data DataFrames
        as_of_date: Reference date (point-in-time)
        feature_sets: List of feature sets to compute

    Returns:
        Dictionary of features

    Example:
        >>> features = get_features(
        ...     account_id="ACC123456",
        ...     account_history={
        ...         "delinquency": dpd_df,
        ...         "payment": payment_df
        ...     },
        ...     as_of_date="2024-01-31",
        ...     feature_sets=["delinquency", "payment"]
        ... )
    """
    engine = _get_engine()
    return engine.get_features(account_id, account_history, as_of_date, feature_sets)


def get_features_batch(
    accounts_history: Dict[str, Dict[str, pd.DataFrame]],
    as_of_date: Union[str, datetime, date],
    feature_sets: List[str] = None
) -> pd.DataFrame:
    """
    Get features for multiple accounts (batch processing).

    Args:
        accounts_history: Dictionary of account histories
        as_of_date: Reference date
        feature_sets: List of feature sets to compute

    Returns:
        DataFrame with one row per account
    """
    engine = _get_engine()
    return engine.get_features_batch(accounts_history, as_of_date, feature_sets)


def list_features(module: Optional[str] = None) -> Dict[str, Dict]:
    """
    List all available features with metadata.

    Args:
        module: Optional module name to filter

    Returns:
        Dictionary of feature metadata
    """
    engine = _get_engine()
    return engine.list_features(module)


def save_feature_registry(output_path: str = None):
    """
    Save feature registry to JSON file.

    Args:
        output_path: Path to save registry
    """
    engine = _get_engine()
    engine.save_feature_registry(output_path)


def reset_schema_mappings():
    """Reset all schema mappings (for re-configuration)"""
    from .utils import get_schema_mapper
    mapper = get_schema_mapper()
    mapper.reset_all()


__all__ = [
    "get_features",
    "get_features_batch",
    "list_features",
    "save_feature_registry",
    "reset_schema_mappings",
    "BehavioralFeatureEngine"
]
