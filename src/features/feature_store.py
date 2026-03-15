"""
Feature Store and Registry Module

Implements feature store and registry with:
- Feature registry with metadata tracking
- Leakage detection framework (temporal validation, target correlation, adversarial validation)
- Feature importance tracking (SHAP, permutation, gain)
- Feature versioning
- Feature quality checks (missing threshold, cardinality, variance)
- Feature lineage tracking

Author: Principal Data Science Decision Agent
"""

import hashlib
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from pathlib import Path
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not available. Install with: pip install shap")


class FeatureStoreConfig:
    """Configuration for feature store."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize feature store configuration.

        Args:
            config_path: Path to feature_config.yaml. If None, uses defaults.

        Example:
            >>> config = FeatureStoreConfig("config/feature_config.yaml")
            >>> config.leakage_detection_enabled
            True
        """
        if config_path and Path(config_path).exists():
            with open(config_path, "r") as f:
                full_config = yaml.safe_load(f)
                store_config = full_config.get("feature_store", {})
        else:
            store_config = {}

        # Registry settings
        registry_config = store_config.get("registry", {})
        self.registry_enabled = registry_config.get("enabled", True)
        self.versioning_enabled = registry_config.get("versioning", True)
        self.metadata_tracking = registry_config.get("metadata_tracking", True)

        # Leakage detection settings
        leakage_config = store_config.get("leakage_detection", {})
        self.leakage_detection_enabled = leakage_config.get("enabled", True)
        self.leakage_methods = leakage_config.get(
            "methods", ["temporal_validation", "target_correlation", "adversarial_validation"]
        )

        thresholds = leakage_config.get("thresholds", {})
        self.correlation_threshold = thresholds.get("correlation_threshold", 0.95)
        self.auc_threshold = thresholds.get("auc_threshold", 0.85)

        # Importance tracking settings
        importance_config = store_config.get("importance_tracking", {})
        self.importance_tracking_enabled = importance_config.get("enabled", True)
        self.importance_methods = importance_config.get(
            "methods", ["shap", "permutation", "gain"]
        )

        # Quality checks settings
        quality_config = store_config.get("quality_checks", {})
        self.missing_threshold = quality_config.get("missing_threshold", 0.5)
        self.cardinality_threshold = quality_config.get("cardinality_threshold", 0.95)
        self.variance_threshold = quality_config.get("variance_threshold", 0.01)


class FeatureMetadata:
    """Metadata for a feature."""

    def __init__(
        self,
        name: str,
        dtype: str,
        description: str = "",
        version: str = "1.0",
        created_at: Optional[datetime] = None,
        source: str = "",
        computation_logic: str = "",
        dependencies: Optional[List[str]] = None,
    ):
        """
        Initialize feature metadata.

        Args:
            name: Feature name
            dtype: Data type
            description: Feature description
            version: Feature version
            created_at: Creation timestamp
            source: Source system or table
            computation_logic: How feature is computed
            dependencies: List of dependent features
        """
        self.name = name
        self.dtype = dtype
        self.description = description
        self.version = version
        self.created_at = created_at or datetime.now()
        self.source = source
        self.computation_logic = computation_logic
        self.dependencies = dependencies or []

        # Quality metrics (to be populated)
        self.missing_rate: Optional[float] = None
        self.cardinality: Optional[int] = None
        self.variance: Optional[float] = None

        # Importance scores (to be populated)
        self.importance_scores: Dict[str, float] = {}

        # Leakage flags
        self.has_leakage: bool = False
        self.leakage_details: Dict[str, Any] = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "name": self.name,
            "dtype": self.dtype,
            "description": self.description,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "source": self.source,
            "computation_logic": self.computation_logic,
            "dependencies": self.dependencies,
            "missing_rate": self.missing_rate,
            "cardinality": self.cardinality,
            "variance": self.variance,
            "importance_scores": self.importance_scores,
            "has_leakage": self.has_leakage,
            "leakage_details": self.leakage_details,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureMetadata":
        """Create metadata from dictionary."""
        metadata = cls(
            name=data["name"],
            dtype=data["dtype"],
            description=data.get("description", ""),
            version=data.get("version", "1.0"),
            created_at=datetime.fromisoformat(data["created_at"]),
            source=data.get("source", ""),
            computation_logic=data.get("computation_logic", ""),
            dependencies=data.get("dependencies", []),
        )

        metadata.missing_rate = data.get("missing_rate")
        metadata.cardinality = data.get("cardinality")
        metadata.variance = data.get("variance")
        metadata.importance_scores = data.get("importance_scores", {})
        metadata.has_leakage = data.get("has_leakage", False)
        metadata.leakage_details = data.get("leakage_details", {})

        return metadata


class FeatureRegistry:
    """Registry for tracking features and their metadata."""

    def __init__(self, registry_path: Optional[str] = None):
        """
        Initialize feature registry.

        Args:
            registry_path: Path to save/load registry JSON file
        """
        self.registry_path = registry_path
        self.features: Dict[str, FeatureMetadata] = {}

        if registry_path and Path(registry_path).exists():
            self.load_registry()

        logger.info(f"Initialized FeatureRegistry with {len(self.features)} features")

    def register_feature(self, metadata: FeatureMetadata) -> None:
        """
        Register a feature in the registry.

        Args:
            metadata: Feature metadata object
        """
        self.features[metadata.name] = metadata
        logger.info(f"Registered feature: {metadata.name} (v{metadata.version})")

    def get_feature(self, name: str) -> Optional[FeatureMetadata]:
        """
        Get feature metadata by name.

        Args:
            name: Feature name

        Returns:
            Feature metadata if found, None otherwise
        """
        return self.features.get(name)

    def list_features(self, filter_by: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        List all features, optionally filtered.

        Args:
            filter_by: Dictionary of filters (e.g., {'dtype': 'float64'})

        Returns:
            List of feature names
        """
        if not filter_by:
            return list(self.features.keys())

        filtered = []
        for name, metadata in self.features.items():
            match = True
            for key, value in filter_by.items():
                if getattr(metadata, key, None) != value:
                    match = False
                    break
            if match:
                filtered.append(name)

        return filtered

    def save_registry(self) -> None:
        """Save registry to file."""
        if not self.registry_path:
            logger.warning("No registry path specified, cannot save")
            return

        registry_data = {
            name: metadata.to_dict() for name, metadata in self.features.items()
        }

        with open(self.registry_path, "w") as f:
            json.dump(registry_data, f, indent=2)

        logger.info(f"Saved registry with {len(self.features)} features to {self.registry_path}")

    def load_registry(self) -> None:
        """Load registry from file."""
        if not self.registry_path or not Path(self.registry_path).exists():
            logger.warning(f"Registry file not found: {self.registry_path}")
            return

        with open(self.registry_path, "r") as f:
            registry_data = json.load(f)

        self.features = {
            name: FeatureMetadata.from_dict(data)
            for name, data in registry_data.items()
        }

        logger.info(f"Loaded registry with {len(self.features)} features from {self.registry_path}")


class LeakageDetector:
    """Detector for feature leakage."""

    def __init__(self, config: FeatureStoreConfig):
        """
        Initialize leakage detector.

        Args:
            config: Feature store configuration
        """
        self.config = config
        logger.info("Initialized LeakageDetector")

    def detect_temporal_leakage(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        timestamp_col: str,
        target_col: str,
        test_size: float = 0.2,
    ) -> Dict[str, Any]:
        """
        Detect temporal leakage using time-based split validation.

        Features should not be predictive when using future information.

        Args:
            df: Input dataframe
            feature_cols: List of feature columns to check
            timestamp_col: Timestamp column name
            target_col: Target column name
            test_size: Fraction for test set

        Returns:
            Dictionary with leakage detection results
        """
        logger.info("Detecting temporal leakage")

        results = {}

        # Sort by timestamp
        df_sorted = df.sort_values(timestamp_col)

        # Split at temporal boundary
        split_idx = int(len(df_sorted) * (1 - test_size))
        train_df = df_sorted.iloc[:split_idx]
        test_df = df_sorted.iloc[split_idx:]

        # Check if features are available in training data
        for feature in feature_cols:
            train_missing = train_df[feature].isna().mean()
            test_missing = test_df[feature].isna().mean()

            # If feature suddenly becomes available in test set, it's leakage
            if train_missing > 0.9 and test_missing < 0.1:
                results[feature] = {
                    "has_temporal_leakage": True,
                    "reason": "Feature availability changes drastically between train/test",
                    "train_missing_rate": train_missing,
                    "test_missing_rate": test_missing,
                }
            else:
                results[feature] = {
                    "has_temporal_leakage": False,
                    "train_missing_rate": train_missing,
                    "test_missing_rate": test_missing,
                }

        logger.info(f"Temporal leakage check completed for {len(feature_cols)} features")

        return results

    def detect_target_correlation_leakage(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
    ) -> Dict[str, Any]:
        """
        Detect leakage via suspiciously high correlation with target.

        Args:
            df: Input dataframe
            feature_cols: List of feature columns to check
            target_col: Target column name

        Returns:
            Dictionary with correlation leakage results
        """
        logger.info("Detecting target correlation leakage")

        results = {}

        for feature in feature_cols:
            try:
                # Handle different data types
                if pd.api.types.is_numeric_dtype(df[feature]):
                    correlation = df[[feature, target_col]].corr().iloc[0, 1]
                else:
                    # For categorical, use Cramér's V
                    confusion_matrix = pd.crosstab(df[feature], df[target_col])
                    chi2 = stats.chi2_contingency(confusion_matrix)[0]
                    n = confusion_matrix.sum().sum()
                    correlation = np.sqrt(chi2 / (n * (min(confusion_matrix.shape) - 1)))

                has_leakage = abs(correlation) > self.config.correlation_threshold

                results[feature] = {
                    "has_correlation_leakage": has_leakage,
                    "correlation": float(correlation),
                    "threshold": self.config.correlation_threshold,
                }

                if has_leakage:
                    logger.warning(
                        f"Feature {feature} has suspicious correlation: {correlation:.3f}"
                    )

            except Exception as e:
                logger.warning(f"Failed to compute correlation for {feature}: {str(e)}")
                results[feature] = {"has_correlation_leakage": False, "error": str(e)}

        logger.info(f"Correlation leakage check completed for {len(feature_cols)} features")

        return results

    def detect_adversarial_validation_leakage(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_cols: List[str],
    ) -> Dict[str, Any]:
        """
        Detect leakage via adversarial validation.

        If we can easily predict train/test split, features may contain leakage.

        Args:
            train_df: Training dataframe
            test_df: Test dataframe
            feature_cols: List of feature columns to check

        Returns:
            Dictionary with adversarial validation results
        """
        logger.info("Detecting leakage via adversarial validation")

        # Create train/test labels
        train_df = train_df.copy()
        test_df = test_df.copy()

        train_df["is_test"] = 0
        test_df["is_test"] = 1

        combined = pd.concat([train_df, test_df], ignore_index=True)

        # Prepare features
        X = combined[feature_cols].copy()

        # Handle missing values
        X = X.fillna(-999)

        # Encode categorical variables
        for col in X.columns:
            if X[col].dtype == "object":
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))

        y = combined["is_test"]

        # Train model to distinguish train from test
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)

            # Evaluate
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_pred_proba)

            has_leakage = auc > self.config.auc_threshold

            # Feature importances
            importances = dict(zip(feature_cols, model.feature_importances_))
            top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]

            results = {
                "has_adversarial_leakage": has_leakage,
                "auc": float(auc),
                "threshold": self.config.auc_threshold,
                "top_leaky_features": top_features,
            }

            if has_leakage:
                logger.warning(
                    f"Adversarial validation AUC is suspiciously high: {auc:.3f}"
                )

        except Exception as e:
            logger.error(f"Adversarial validation failed: {str(e)}")
            results = {"has_adversarial_leakage": False, "error": str(e)}

        logger.info("Adversarial validation completed")

        return results


class FeatureImportanceTracker:
    """Tracker for feature importance across different methods."""

    def __init__(self, config: FeatureStoreConfig):
        """
        Initialize feature importance tracker.

        Args:
            config: Feature store configuration
        """
        self.config = config
        logger.info("Initialized FeatureImportanceTracker")

    def compute_shap_importance(
        self,
        model: Any,
        X: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Compute SHAP-based feature importance.

        Args:
            model: Trained model
            X: Feature dataframe
            feature_cols: Feature columns. If None, uses all columns.

        Returns:
            Dictionary of feature importances
        """
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available, skipping")
            return {}

        feature_cols = feature_cols or list(X.columns)

        logger.info(f"Computing SHAP importance for {len(feature_cols)} features")

        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X[feature_cols])

            # Handle multi-class output
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Use positive class for binary

            # Mean absolute SHAP values
            importance = np.abs(shap_values).mean(axis=0)

            importances = dict(zip(feature_cols, importance))

            logger.info("SHAP importance computation completed")

            return importances

        except Exception as e:
            logger.error(f"SHAP importance computation failed: {str(e)}")
            return {}

    def compute_permutation_importance(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        feature_cols: Optional[List[str]] = None,
        n_repeats: int = 10,
    ) -> Dict[str, float]:
        """
        Compute permutation-based feature importance.

        Args:
            model: Trained model
            X: Feature dataframe
            y: Target series
            feature_cols: Feature columns. If None, uses all columns.
            n_repeats: Number of permutation repeats

        Returns:
            Dictionary of feature importances
        """
        feature_cols = feature_cols or list(X.columns)

        logger.info(
            f"Computing permutation importance for {len(feature_cols)} features"
        )

        try:
            result = permutation_importance(
                model, X[feature_cols], y, n_repeats=n_repeats, random_state=42, n_jobs=-1
            )

            importances = dict(zip(feature_cols, result.importances_mean))

            logger.info("Permutation importance computation completed")

            return importances

        except Exception as e:
            logger.error(f"Permutation importance computation failed: {str(e)}")
            return {}

    def compute_gain_importance(
        self,
        model: Any,
        feature_cols: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Compute gain-based feature importance from tree model.

        Args:
            model: Trained tree-based model
            feature_cols: Feature columns. If None, uses model's feature names.

        Returns:
            Dictionary of feature importances
        """
        logger.info("Computing gain-based importance")

        try:
            if hasattr(model, "feature_importances_"):
                importances_array = model.feature_importances_

                if feature_cols is None:
                    if hasattr(model, "feature_name_"):
                        feature_cols = model.feature_name_
                    elif hasattr(model, "feature_names_in_"):
                        feature_cols = model.feature_names_in_
                    else:
                        feature_cols = [f"feature_{i}" for i in range(len(importances_array))]

                importances = dict(zip(feature_cols, importances_array))

                logger.info("Gain importance computation completed")

                return importances
            else:
                logger.warning("Model does not have feature_importances_ attribute")
                return {}

        except Exception as e:
            logger.error(f"Gain importance computation failed: {str(e)}")
            return {}


class FeatureQualityChecker:
    """Checker for feature quality metrics."""

    def __init__(self, config: FeatureStoreConfig):
        """
        Initialize feature quality checker.

        Args:
            config: Feature store configuration
        """
        self.config = config
        logger.info("Initialized FeatureQualityChecker")

    def check_missing_rate(self, df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, float]:
        """
        Check missing rate for features.

        Args:
            df: Input dataframe
            feature_cols: Feature columns to check

        Returns:
            Dictionary of missing rates
        """
        logger.info(f"Checking missing rates for {len(feature_cols)} features")

        missing_rates = {}
        for col in feature_cols:
            missing_rates[col] = df[col].isna().mean()

        # Flag high missing rates
        high_missing = {
            col: rate
            for col, rate in missing_rates.items()
            if rate > self.config.missing_threshold
        }

        if high_missing:
            logger.warning(
                f"{len(high_missing)} features have high missing rates (>{self.config.missing_threshold})"
            )

        return missing_rates

    def check_cardinality(self, df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, int]:
        """
        Check cardinality (unique values) for features.

        Args:
            df: Input dataframe
            feature_cols: Feature columns to check

        Returns:
            Dictionary of cardinalities
        """
        logger.info(f"Checking cardinality for {len(feature_cols)} features")

        cardinalities = {}
        for col in feature_cols:
            cardinalities[col] = df[col].nunique()

        # Flag high cardinality (approaching row count)
        high_cardinality = {
            col: card
            for col, card in cardinalities.items()
            if card / len(df) > self.config.cardinality_threshold
        }

        if high_cardinality:
            logger.warning(
                f"{len(high_cardinality)} features have high cardinality "
                f"(>{self.config.cardinality_threshold * 100}% of rows)"
            )

        return cardinalities

    def check_variance(self, df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, float]:
        """
        Check variance for numeric features.

        Args:
            df: Input dataframe
            feature_cols: Feature columns to check

        Returns:
            Dictionary of variances
        """
        logger.info(f"Checking variance for {len(feature_cols)} features")

        variances = {}
        for col in feature_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                variances[col] = df[col].var()

        # Flag low variance features
        low_variance = {
            col: var
            for col, var in variances.items()
            if var < self.config.variance_threshold
        }

        if low_variance:
            logger.warning(
                f"{len(low_variance)} features have low variance "
                f"(<{self.config.variance_threshold})"
            )

        return variances


class FeatureStore:
    """
    Comprehensive feature store with registry, leakage detection,
    importance tracking, and quality checks.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        registry_path: Optional[str] = None,
    ):
        """
        Initialize feature store.

        Args:
            config_path: Path to feature_config.yaml
            registry_path: Path to save/load feature registry

        Example:
            >>> store = FeatureStore(
            ...     config_path="config/feature_config.yaml",
            ...     registry_path="feature_registry.json"
            ... )
        """
        self.config = FeatureStoreConfig(config_path)
        self.registry = FeatureRegistry(registry_path)
        self.leakage_detector = LeakageDetector(self.config)
        self.importance_tracker = FeatureImportanceTracker(self.config)
        self.quality_checker = FeatureQualityChecker(self.config)

        logger.info("Initialized FeatureStore")

    def register_features_from_dataframe(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        source: str = "",
    ) -> None:
        """
        Register features from a dataframe.

        Args:
            df: Input dataframe
            feature_cols: Feature columns to register
            source: Source description
        """
        logger.info(f"Registering {len(feature_cols)} features from dataframe")

        for col in feature_cols:
            metadata = FeatureMetadata(
                name=col,
                dtype=str(df[col].dtype),
                source=source,
                version="1.0",
            )

            # Compute quality metrics
            metadata.missing_rate = df[col].isna().mean()
            metadata.cardinality = df[col].nunique()

            if pd.api.types.is_numeric_dtype(df[col]):
                metadata.variance = df[col].var()

            self.registry.register_feature(metadata)

        # Save registry
        if self.registry.registry_path:
            self.registry.save_registry()

    def run_quality_checks(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
    ) -> Dict[str, Any]:
        """
        Run all quality checks on features.

        Args:
            df: Input dataframe
            feature_cols: Feature columns to check

        Returns:
            Dictionary with quality check results
        """
        logger.info(f"Running quality checks on {len(feature_cols)} features")

        results = {
            "missing_rates": self.quality_checker.check_missing_rate(df, feature_cols),
            "cardinalities": self.quality_checker.check_cardinality(df, feature_cols),
            "variances": self.quality_checker.check_variance(df, feature_cols),
        }

        # Update registry with quality metrics
        for col in feature_cols:
            metadata = self.registry.get_feature(col)
            if metadata:
                metadata.missing_rate = results["missing_rates"].get(col)
                metadata.cardinality = results["cardinalities"].get(col)
                metadata.variance = results["variances"].get(col)

        return results

    def detect_leakage(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        timestamp_col: Optional[str] = None,
        test_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Run all leakage detection checks.

        Args:
            df: Input dataframe (training data)
            feature_cols: Feature columns to check
            target_col: Target column name
            timestamp_col: Optional timestamp column for temporal checks
            test_df: Optional test dataframe for adversarial validation

        Returns:
            Dictionary with leakage detection results
        """
        logger.info(f"Running leakage detection on {len(feature_cols)} features")

        results = {}

        # Temporal leakage
        if timestamp_col and "temporal_validation" in self.config.leakage_methods:
            results["temporal"] = self.leakage_detector.detect_temporal_leakage(
                df, feature_cols, timestamp_col, target_col
            )

        # Target correlation leakage
        if "target_correlation" in self.config.leakage_methods:
            results["correlation"] = (
                self.leakage_detector.detect_target_correlation_leakage(
                    df, feature_cols, target_col
                )
            )

        # Adversarial validation
        if test_df is not None and "adversarial_validation" in self.config.leakage_methods:
            results["adversarial"] = (
                self.leakage_detector.detect_adversarial_validation_leakage(
                    df, test_df, feature_cols
                )
            )

        # Update registry with leakage information
        for col in feature_cols:
            metadata = self.registry.get_feature(col)
            if metadata:
                has_leakage = False

                # Check all leakage types
                if "temporal" in results and results["temporal"].get(col, {}).get(
                    "has_temporal_leakage"
                ):
                    has_leakage = True

                if "correlation" in results and results["correlation"].get(col, {}).get(
                    "has_correlation_leakage"
                ):
                    has_leakage = True

                metadata.has_leakage = has_leakage
                metadata.leakage_details = {
                    "temporal": results.get("temporal", {}).get(col),
                    "correlation": results.get("correlation", {}).get(col),
                }

        return results

    def track_importance(
        self,
        model: Any,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        feature_cols: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Track feature importance using multiple methods.

        Args:
            model: Trained model
            X: Feature dataframe
            y: Target series (required for permutation importance)
            feature_cols: Feature columns. If None, uses all columns.

        Returns:
            Dictionary with importance scores from different methods
        """
        feature_cols = feature_cols or list(X.columns)

        logger.info(f"Tracking importance for {len(feature_cols)} features")

        results = {}

        if "gain" in self.config.importance_methods:
            results["gain"] = self.importance_tracker.compute_gain_importance(
                model, feature_cols
            )

        if "shap" in self.config.importance_methods and SHAP_AVAILABLE:
            results["shap"] = self.importance_tracker.compute_shap_importance(
                model, X, feature_cols
            )

        if "permutation" in self.config.importance_methods and y is not None:
            results["permutation"] = (
                self.importance_tracker.compute_permutation_importance(
                    model, X, y, feature_cols
                )
            )

        # Update registry with importance scores
        for col in feature_cols:
            metadata = self.registry.get_feature(col)
            if metadata:
                for method, importances in results.items():
                    if col in importances:
                        metadata.importance_scores[method] = importances[col]

        return results


if __name__ == "__main__":
    # Example usage
    logger.info("Running feature store example")

    # Create sample data
    np.random.seed(42)
    n_samples = 1000

    df = pd.DataFrame(
        {
            "user_id": range(n_samples),
            "timestamp": pd.date_range("2024-01-01", periods=n_samples, freq="H"),
            "feature1": np.random.randn(n_samples),
            "feature2": np.random.gamma(2, 1, n_samples),
            "feature3": np.random.choice(["A", "B", "C"], n_samples),
            "target": np.random.binomial(1, 0.3, n_samples),
        }
    )

    # Initialize feature store
    store = FeatureStore(registry_path="feature_registry_example.json")

    # Register features
    feature_cols = ["feature1", "feature2", "feature3"]
    store.register_features_from_dataframe(df, feature_cols, source="example_data")

    # Run quality checks
    quality_results = store.run_quality_checks(df, feature_cols)
    logger.info(f"Quality check results: {quality_results}")

    # Detect leakage
    leakage_results = store.detect_leakage(
        df, feature_cols, target_col="target", timestamp_col="timestamp"
    )
    logger.info(f"Leakage detection completed")

    logger.info("Feature store example completed")
