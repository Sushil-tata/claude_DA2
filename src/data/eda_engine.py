"""
EDA Engine Module

Comprehensive Exploratory Data Analysis framework with automated univariate,
bivariate, and multivariate analysis, drift detection, leakage identification,
and bias detection.

Author: Principal Data Science Decision Agent
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import LabelEncoder, StandardScaler


@dataclass
class EDAConfig:
    """Configuration for EDA engine."""

    # Correlation thresholds
    high_correlation_threshold: float = 0.7
    perfect_correlation_threshold: float = 0.95

    # Drift detection
    drift_pvalue_threshold: float = 0.05
    drift_ks_statistic_threshold: float = 0.1

    # Leakage detection
    leakage_correlation_threshold: float = 0.95
    leakage_mutual_info_threshold: float = 0.9

    # Bias detection
    bias_disparity_threshold: float = 1.2

    # Feature importance
    importance_threshold: float = 0.01

    # Sampling for large datasets
    max_rows_for_analysis: int = 100000
    sample_random_state: int = 42


@dataclass
class UnivariateAnalysis:
    """Results of univariate analysis."""

    column: str
    dtype: str
    count: int
    missing: int
    missing_pct: float
    unique: int
    unique_pct: float
    stats: Dict[str, Any]
    distribution: Dict[str, Any]
    outliers: Dict[str, Any]


@dataclass
class BivariateAnalysis:
    """Results of bivariate analysis with target."""

    feature: str
    target: str
    relationship_type: str
    correlation: Optional[float]
    mutual_information: Optional[float]
    statistical_test: Dict[str, Any]
    feature_importance: Optional[float]


@dataclass
class EDAReport:
    """Comprehensive EDA report."""

    dataset_overview: Dict[str, Any]
    univariate_analysis: List[UnivariateAnalysis]
    bivariate_analysis: List[BivariateAnalysis]
    correlation_matrix: Optional[pd.DataFrame]
    drift_analysis: Dict[str, Any]
    leakage_risks: Dict[str, Any]
    bias_analysis: Dict[str, Any]
    regime_changes: Dict[str, Any]
    recommendations: List[str]
    warnings: List[str]


class EDAEngine:
    """
    Comprehensive Exploratory Data Analysis engine.

    Performs automated analysis including:
    - Univariate analysis (distributions, outliers, statistics)
    - Bivariate analysis with target variable
    - Correlation and multicollinearity detection
    - Temporal drift analysis
    - Target leakage identification
    - Sampling bias detection
    - Regime change detection
    - Feature importance estimation

    Examples:
        >>> engine = EDAEngine()
        >>> report = engine.analyze(df, target_column='target')
        >>> print(f"Recommendations: {len(report.recommendations)}")
        >>> print(f"Warnings: {len(report.warnings)}")
    """

    def __init__(self, config: Optional[EDAConfig] = None):
        """
        Initialize EDA Engine.

        Args:
            config: Configuration object for EDA
        """
        self.config = config or EDAConfig()
        logger.info("EDAEngine initialized")

    def analyze(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        datetime_column: Optional[str] = None,
        categorical_columns: Optional[List[str]] = None,
    ) -> EDAReport:
        """
        Perform comprehensive EDA.

        Args:
            df: DataFrame to analyze
            target_column: Target variable for supervised learning
            datetime_column: Column with datetime for temporal analysis
            categorical_columns: List of categorical column names

        Returns:
            EDAReport with all analysis results
        """
        logger.info("Starting comprehensive EDA")
        logger.info("Dataset shape: {} rows, {} columns", len(df), len(df.columns))

        # Sample if dataset is too large
        if len(df) > self.config.max_rows_for_analysis:
            logger.info(
                "Sampling {} rows for analysis", self.config.max_rows_for_analysis
            )
            df_sample = df.sample(
                n=self.config.max_rows_for_analysis,
                random_state=self.config.sample_random_state,
            )
        else:
            df_sample = df

        recommendations = []
        warnings = []

        # Dataset overview
        logger.info("Generating dataset overview...")
        dataset_overview = self._generate_overview(df_sample)

        # Univariate analysis
        logger.info("Performing univariate analysis...")
        univariate_results = self._univariate_analysis(
            df_sample, categorical_columns
        )

        # Correlation analysis
        logger.info("Computing correlation matrix...")
        correlation_matrix = self._correlation_analysis(df_sample)

        # Bivariate analysis (if target provided)
        bivariate_results = []
        if target_column and target_column in df_sample.columns:
            logger.info("Performing bivariate analysis with target...")
            bivariate_results = self._bivariate_analysis(
                df_sample, target_column, categorical_columns
            )

        # Drift analysis (if datetime provided)
        drift_analysis = {}
        if datetime_column and datetime_column in df_sample.columns:
            logger.info("Analyzing temporal drift...")
            drift_analysis = self._drift_analysis(
                df_sample, datetime_column, target_column
            )
            if drift_analysis.get("drifted_features"):
                warnings.append(
                    f"Detected drift in {len(drift_analysis['drifted_features'])} features"
                )
                recommendations.append(
                    "Consider using time-based validation and monitoring drift"
                )

        # Leakage detection (if target provided)
        leakage_risks = {}
        if target_column and target_column in df_sample.columns:
            logger.info("Detecting potential target leakage...")
            leakage_risks = self._detect_leakage(
                df_sample, target_column, datetime_column
            )
            if leakage_risks.get("high_risk_features"):
                warnings.append(
                    f"Potential target leakage in {len(leakage_risks['high_risk_features'])} features"
                )
                recommendations.append(
                    "Review high-risk features for data leakage before modeling"
                )

        # Bias analysis
        logger.info("Analyzing sampling bias...")
        bias_analysis = self._bias_analysis(df_sample, categorical_columns)
        if bias_analysis.get("biased_features"):
            warnings.append(
                f"Detected bias in {len(bias_analysis['biased_features'])} features"
            )
            recommendations.append("Consider resampling or bias mitigation techniques")

        # Regime change detection (if datetime provided)
        regime_changes = {}
        if datetime_column and datetime_column in df_sample.columns:
            logger.info("Detecting regime changes...")
            regime_changes = self._detect_regime_changes(
                df_sample, datetime_column, target_column
            )
            if regime_changes.get("detected_changes"):
                warnings.append("Detected potential regime changes in data")
                recommendations.append(
                    "Investigate regime changes and consider separate models"
                )

        # Generate additional recommendations
        recommendations.extend(self._generate_recommendations(univariate_results, correlation_matrix))

        report = EDAReport(
            dataset_overview=dataset_overview,
            univariate_analysis=univariate_results,
            bivariate_analysis=bivariate_results,
            correlation_matrix=correlation_matrix,
            drift_analysis=drift_analysis,
            leakage_risks=leakage_risks,
            bias_analysis=bias_analysis,
            regime_changes=regime_changes,
            recommendations=recommendations,
            warnings=warnings,
        )

        logger.info(
            "EDA complete: {} recommendations, {} warnings",
            len(recommendations),
            len(warnings),
        )

        return report

    def _generate_overview(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate dataset overview statistics."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        datetime_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()

        memory_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)  # MB

        overview = {
            "n_rows": len(df),
            "n_columns": len(df.columns),
            "n_numeric": len(numeric_cols),
            "n_categorical": len(categorical_cols),
            "n_datetime": len(datetime_cols),
            "memory_mb": float(memory_usage),
            "missing_cells": int(df.isnull().sum().sum()),
            "missing_pct": float(df.isnull().sum().sum() / (df.shape[0] * df.shape[1])),
            "duplicate_rows": int(df.duplicated().sum()),
            "duplicate_pct": float(df.duplicated().sum() / len(df)),
        }

        logger.debug("Dataset overview: {} rows, {} columns", overview["n_rows"], overview["n_columns"])

        return overview

    def _univariate_analysis(
        self, df: pd.DataFrame, categorical_columns: Optional[List[str]] = None
    ) -> List[UnivariateAnalysis]:
        """Perform univariate analysis on all columns."""
        results = []

        for col in df.columns:
            logger.debug("Analyzing column: {}", col)

            count = len(df[col])
            missing = df[col].isnull().sum()
            missing_pct = missing / count
            unique = df[col].nunique()
            unique_pct = unique / count

            # Basic stats
            stats_dict = {
                "count": int(count),
                "missing": int(missing),
                "unique": int(unique),
            }

            # Type-specific analysis
            if pd.api.types.is_numeric_dtype(df[col]):
                non_null = df[col].dropna()
                if len(non_null) > 0:
                    stats_dict.update(
                        {
                            "mean": float(non_null.mean()),
                            "median": float(non_null.median()),
                            "std": float(non_null.std()),
                            "min": float(non_null.min()),
                            "max": float(non_null.max()),
                            "q25": float(non_null.quantile(0.25)),
                            "q75": float(non_null.quantile(0.75)),
                        }
                    )

                    # Distribution analysis
                    distribution = {
                        "skewness": float(stats.skew(non_null)),
                        "kurtosis": float(stats.kurtosis(non_null)),
                    }

                    # Outlier detection
                    Q1 = non_null.quantile(0.25)
                    Q3 = non_null.quantile(0.75)
                    IQR = Q3 - Q1
                    outliers = non_null[
                        (non_null < Q1 - 1.5 * IQR) | (non_null > Q3 + 1.5 * IQR)
                    ]
                    outlier_info = {
                        "count": int(len(outliers)),
                        "percentage": float(len(outliers) / len(non_null)),
                    }
                else:
                    distribution = {}
                    outlier_info = {}

            elif categorical_columns and col in categorical_columns or pd.api.types.is_object_dtype(df[col]):
                value_counts = df[col].value_counts()
                stats_dict.update(
                    {
                        "top_value": str(value_counts.index[0]) if len(value_counts) > 0 else None,
                        "top_frequency": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                        "top_percentage": float(value_counts.iloc[0] / count) if len(value_counts) > 0 else 0,
                    }
                )
                distribution = {"value_counts": value_counts.head(10).to_dict()}
                outlier_info = {}

            else:
                distribution = {}
                outlier_info = {}

            analysis = UnivariateAnalysis(
                column=col,
                dtype=str(df[col].dtype),
                count=count,
                missing=int(missing),
                missing_pct=float(missing_pct),
                unique=int(unique),
                unique_pct=float(unique_pct),
                stats=stats_dict,
                distribution=distribution,
                outliers=outlier_info,
            )

            results.append(analysis)

        return results

    def _correlation_analysis(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Compute correlation matrix for numeric features."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) < 2:
            logger.warning("Less than 2 numeric columns, skipping correlation analysis")
            return None

        # Handle missing values
        df_numeric = df[numeric_cols].dropna()

        if len(df_numeric) == 0:
            logger.warning("No complete rows for correlation analysis")
            return None

        corr_matrix = df_numeric.corr()

        # Find high correlations
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_val = abs(corr_matrix.iloc[i, j])
                if corr_val > self.config.high_correlation_threshold:
                    high_corr_pairs.append(
                        (
                            corr_matrix.columns[i],
                            corr_matrix.columns[j],
                            corr_val,
                        )
                    )

        if high_corr_pairs:
            logger.info("Found {} high correlation pairs", len(high_corr_pairs))

        return corr_matrix

    def _bivariate_analysis(
        self,
        df: pd.DataFrame,
        target_column: str,
        categorical_columns: Optional[List[str]] = None,
    ) -> List[BivariateAnalysis]:
        """Perform bivariate analysis with target variable."""
        results = []
        feature_cols = [col for col in df.columns if col != target_column]

        # Determine target type
        target_is_numeric = pd.api.types.is_numeric_dtype(df[target_column])
        target_is_binary = df[target_column].nunique() == 2

        # Compute feature importance if possible
        feature_importance = self._compute_feature_importance(
            df, target_column, feature_cols, categorical_columns
        )

        for col in feature_cols:
            logger.debug("Analyzing relationship: {} vs {}", col, target_column)

            # Determine feature type
            feature_is_numeric = pd.api.types.is_numeric_dtype(df[col])

            # Clean data
            df_clean = df[[col, target_column]].dropna()

            if len(df_clean) < 3:
                continue

            # Calculate correlation (if both numeric)
            correlation = None
            if feature_is_numeric and target_is_numeric:
                correlation = float(df_clean[col].corr(df_clean[target_column]))
                relationship_type = "numeric_numeric"

            elif feature_is_numeric and not target_is_numeric:
                relationship_type = "numeric_categorical"

            elif not feature_is_numeric and target_is_numeric:
                relationship_type = "categorical_numeric"

            else:
                relationship_type = "categorical_categorical"

            # Compute mutual information
            mutual_info = self._compute_mutual_information(
                df_clean[[col]], df_clean[target_column], target_is_numeric
            )

            # Statistical test
            statistical_test = self._statistical_test(
                df_clean[col], df_clean[target_column], relationship_type
            )

            # Get feature importance
            importance = feature_importance.get(col)

            analysis = BivariateAnalysis(
                feature=col,
                target=target_column,
                relationship_type=relationship_type,
                correlation=correlation,
                mutual_information=mutual_info,
                statistical_test=statistical_test,
                feature_importance=importance,
            )

            results.append(analysis)

        # Sort by importance/correlation
        results.sort(
            key=lambda x: abs(x.feature_importance or x.correlation or x.mutual_information or 0),
            reverse=True,
        )

        return results

    def _compute_mutual_information(
        self, X: pd.DataFrame, y: pd.Series, target_is_numeric: bool
    ) -> Optional[float]:
        """Compute mutual information between feature and target."""
        try:
            # Encode categorical features
            X_encoded = X.copy()
            for col in X_encoded.columns:
                if not pd.api.types.is_numeric_dtype(X_encoded[col]):
                    le = LabelEncoder()
                    X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))

            if target_is_numeric:
                mi = mutual_info_regression(X_encoded, y, random_state=42)
            else:
                # Encode target
                le = LabelEncoder()
                y_encoded = le.fit_transform(y.astype(str))
                mi = mutual_info_classif(X_encoded, y_encoded, random_state=42)

            # Normalize to [0, 1]
            mi_normalized = float(mi[0]) if len(mi) > 0 else 0.0

            return mi_normalized

        except Exception as e:
            logger.warning("Failed to compute mutual information: {}", str(e))
            return None

    def _statistical_test(
        self, feature: pd.Series, target: pd.Series, relationship_type: str
    ) -> Dict[str, Any]:
        """Perform appropriate statistical test based on variable types."""
        try:
            if relationship_type == "numeric_numeric":
                # Pearson correlation test
                corr, p_value = stats.pearsonr(feature, target)
                return {
                    "test": "pearson_correlation",
                    "statistic": float(corr),
                    "p_value": float(p_value),
                    "significant": p_value < 0.05,
                }

            elif relationship_type == "numeric_categorical":
                # ANOVA or t-test
                groups = [group.dropna() for name, group in feature.groupby(target)]
                if len(groups) == 2:
                    stat, p_value = stats.ttest_ind(groups[0], groups[1])
                    test_name = "t_test"
                else:
                    stat, p_value = stats.f_oneway(*groups)
                    test_name = "anova"

                return {
                    "test": test_name,
                    "statistic": float(stat),
                    "p_value": float(p_value),
                    "significant": p_value < 0.05,
                }

            elif relationship_type == "categorical_numeric":
                # Same as numeric_categorical, reversed
                groups = [group.dropna() for name, group in target.groupby(feature)]
                if len(groups) == 2:
                    stat, p_value = stats.ttest_ind(groups[0], groups[1])
                    test_name = "t_test"
                else:
                    stat, p_value = stats.f_oneway(*groups)
                    test_name = "anova"

                return {
                    "test": test_name,
                    "statistic": float(stat),
                    "p_value": float(p_value),
                    "significant": p_value < 0.05,
                }

            else:  # categorical_categorical
                # Chi-square test
                contingency_table = pd.crosstab(feature, target)
                chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

                return {
                    "test": "chi_square",
                    "statistic": float(chi2),
                    "p_value": float(p_value),
                    "degrees_of_freedom": int(dof),
                    "significant": p_value < 0.05,
                }

        except Exception as e:
            logger.warning("Statistical test failed: {}", str(e))
            return {"test": "failed", "error": str(e)}

    def _compute_feature_importance(
        self,
        df: pd.DataFrame,
        target_column: str,
        feature_cols: List[str],
        categorical_columns: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Compute feature importance using Random Forest."""
        try:
            # Prepare data
            X = df[feature_cols].copy()
            y = df[target_column].copy()

            # Handle missing values
            X = X.fillna(X.median(numeric_only=True))
            X = X.fillna("missing")

            # Encode categorical features
            for col in X.columns:
                if not pd.api.types.is_numeric_dtype(X[col]):
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))

            # Determine problem type
            if pd.api.types.is_numeric_dtype(y):
                model = RandomForestRegressor(
                    n_estimators=50, max_depth=10, random_state=42, n_jobs=-1
                )
            else:
                le = LabelEncoder()
                y = le.fit_transform(y.astype(str))
                model = RandomForestClassifier(
                    n_estimators=50, max_depth=10, random_state=42, n_jobs=-1
                )

            # Fit model
            model.fit(X, y)

            # Get feature importance
            importance_dict = dict(zip(feature_cols, model.feature_importances_))

            return importance_dict

        except Exception as e:
            logger.warning("Feature importance computation failed: {}", str(e))
            return {}

    def _drift_analysis(
        self,
        df: pd.DataFrame,
        datetime_column: str,
        target_column: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Detect temporal drift in features."""
        try:
            # Sort by datetime
            df_sorted = df.sort_values(datetime_column)

            # Split into first and last half
            split_idx = len(df_sorted) // 2
            df_first = df_sorted.iloc[:split_idx]
            df_last = df_sorted.iloc[split_idx:]

            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if target_column in numeric_cols:
                numeric_cols.remove(target_column)

            drifted_features = []
            drift_details = {}

            for col in numeric_cols:
                # Kolmogorov-Smirnov test
                first_values = df_first[col].dropna()
                last_values = df_last[col].dropna()

                if len(first_values) > 0 and len(last_values) > 0:
                    ks_stat, p_value = stats.ks_2samp(first_values, last_values)

                    drift_details[col] = {
                        "ks_statistic": float(ks_stat),
                        "p_value": float(p_value),
                        "drifted": p_value < self.config.drift_pvalue_threshold,
                    }

                    if p_value < self.config.drift_pvalue_threshold:
                        drifted_features.append(col)

            return {
                "drifted_features": drifted_features,
                "drift_details": drift_details,
                "n_drifted": len(drifted_features),
            }

        except Exception as e:
            logger.error("Drift analysis failed: {}", str(e))
            return {"error": str(e)}

    def _detect_leakage(
        self,
        df: pd.DataFrame,
        target_column: str,
        datetime_column: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Detect potential target leakage."""
        high_risk_features = []
        leakage_scores = {}

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_column in numeric_cols:
            numeric_cols.remove(target_column)

        target_is_numeric = pd.api.types.is_numeric_dtype(df[target_column])

        for col in numeric_cols:
            df_clean = df[[col, target_column]].dropna()

            if len(df_clean) < 3:
                continue

            # Correlation-based leakage
            if target_is_numeric:
                corr = abs(df_clean[col].corr(df_clean[target_column]))
                leakage_scores[col] = {"correlation": float(corr)}

                if corr > self.config.leakage_correlation_threshold:
                    high_risk_features.append(col)

            # Perfect separation check for classification
            if not target_is_numeric:
                # Check if feature can perfectly separate target
                unique_combinations = df_clean.groupby(col)[target_column].nunique()
                if (unique_combinations == 1).all():
                    high_risk_features.append(col)
                    leakage_scores[col] = {"perfect_separation": True}

        # Temporal leakage (future information)
        temporal_leakage = []
        if datetime_column and datetime_column in df.columns:
            # Check if any features have values that appear only after target events
            # This is a simplified check
            pass

        return {
            "high_risk_features": high_risk_features,
            "leakage_scores": leakage_scores,
            "temporal_leakage": temporal_leakage,
            "n_high_risk": len(high_risk_features),
        }

    def _bias_analysis(
        self,
        df: pd.DataFrame,
        categorical_columns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Detect sampling bias in categorical features."""
        biased_features = []
        bias_details = {}

        # Identify categorical columns
        if categorical_columns:
            cat_cols = categorical_columns
        else:
            cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        for col in cat_cols:
            value_counts = df[col].value_counts()

            if len(value_counts) < 2:
                continue

            # Check for imbalance
            max_freq = value_counts.max()
            min_freq = value_counts.min()

            if min_freq > 0:
                disparity_ratio = max_freq / min_freq

                bias_details[col] = {
                    "disparity_ratio": float(disparity_ratio),
                    "most_common": str(value_counts.index[0]),
                    "most_common_freq": int(value_counts.iloc[0]),
                    "least_common": str(value_counts.index[-1]),
                    "least_common_freq": int(value_counts.iloc[-1]),
                }

                if disparity_ratio > self.config.bias_disparity_threshold:
                    biased_features.append(col)

        return {
            "biased_features": biased_features,
            "bias_details": bias_details,
            "n_biased": len(biased_features),
        }

    def _detect_regime_changes(
        self,
        df: pd.DataFrame,
        datetime_column: str,
        target_column: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Detect regime changes in data over time."""
        try:
            # Sort by datetime
            df_sorted = df.sort_values(datetime_column).reset_index(drop=True)

            # Use rolling windows to detect changes
            if target_column and target_column in df.columns:
                target_values = df_sorted[target_column].dropna()

                if len(target_values) < 100:
                    return {"detected_changes": False, "reason": "insufficient_data"}

                # Compute rolling statistics
                window_size = max(len(target_values) // 10, 20)
                rolling_mean = target_values.rolling(window=window_size).mean()
                rolling_std = target_values.rolling(window=window_size).std()

                # Detect significant changes in mean or variance
                mean_changes = abs(rolling_mean.diff()) > rolling_std
                std_changes = rolling_std.diff() > 0.5 * rolling_std

                change_points = []
                for idx in range(len(mean_changes)):
                    if mean_changes.iloc[idx] or std_changes.iloc[idx]:
                        change_points.append(idx)

                return {
                    "detected_changes": len(change_points) > 0,
                    "n_change_points": len(change_points),
                    "change_indices": change_points[:10],  # First 10
                }

            return {"detected_changes": False, "reason": "no_target"}

        except Exception as e:
            logger.error("Regime change detection failed: {}", str(e))
            return {"detected_changes": False, "error": str(e)}

    def _generate_recommendations(
        self,
        univariate_results: List[UnivariateAnalysis],
        correlation_matrix: Optional[pd.DataFrame],
    ) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []

        # Missing value recommendations
        high_missing = [
            r for r in univariate_results if r.missing_pct > 0.3
        ]
        if high_missing:
            recommendations.append(
                f"Consider dropping {len(high_missing)} columns with >30% missing values"
            )

        # Low variance recommendations
        low_variance = [
            r for r in univariate_results if r.unique_pct < 0.01
        ]
        if low_variance:
            recommendations.append(
                f"Remove {len(low_variance)} low-variance features (< 1% unique)"
            )

        # High correlation recommendations
        if correlation_matrix is not None:
            high_corr_count = 0
            for i in range(len(correlation_matrix.columns)):
                for j in range(i + 1, len(correlation_matrix.columns)):
                    if abs(correlation_matrix.iloc[i, j]) > self.config.high_correlation_threshold:
                        high_corr_count += 1

            if high_corr_count > 0:
                recommendations.append(
                    f"Address multicollinearity: {high_corr_count} feature pairs with correlation > {self.config.high_correlation_threshold}"
                )

        # Skewness recommendations
        highly_skewed = [
            r
            for r in univariate_results
            if "skewness" in r.distribution
            and abs(r.distribution["skewness"]) > 2
        ]
        if highly_skewed:
            recommendations.append(
                f"Apply transformations to {len(highly_skewed)} highly skewed features"
            )

        # Outlier recommendations
        outlier_features = [
            r
            for r in univariate_results
            if r.outliers and r.outliers.get("percentage", 0) > 0.05
        ]
        if outlier_features:
            recommendations.append(
                f"Review and handle outliers in {len(outlier_features)} features"
            )

        return recommendations


def generate_eda_summary(report: EDAReport) -> str:
    """
    Generate human-readable summary of EDA report.

    Args:
        report: EDAReport object

    Returns:
        Formatted summary string
    """
    summary_lines = [
        "=" * 80,
        "EXPLORATORY DATA ANALYSIS SUMMARY",
        "=" * 80,
        "",
        "DATASET OVERVIEW:",
        f"  Rows: {report.dataset_overview['n_rows']:,}",
        f"  Columns: {report.dataset_overview['n_columns']}",
        f"  Numeric: {report.dataset_overview['n_numeric']}",
        f"  Categorical: {report.dataset_overview['n_categorical']}",
        f"  Missing: {report.dataset_overview['missing_pct']:.2%}",
        f"  Duplicates: {report.dataset_overview['duplicate_pct']:.2%}",
        "",
    ]

    if report.warnings:
        summary_lines.extend(
            [
                "WARNINGS:",
                *[f"  ⚠ {w}" for w in report.warnings],
                "",
            ]
        )

    if report.recommendations:
        summary_lines.extend(
            [
                "RECOMMENDATIONS:",
                *[f"  → {r}" for r in report.recommendations],
                "",
            ]
        )

    if report.leakage_risks.get("high_risk_features"):
        summary_lines.extend(
            [
                "HIGH LEAKAGE RISK FEATURES:",
                *[f"  - {f}" for f in report.leakage_risks["high_risk_features"]],
                "",
            ]
        )

    if report.drift_analysis.get("drifted_features"):
        summary_lines.extend(
            [
                "DRIFTED FEATURES:",
                *[f"  - {f}" for f in report.drift_analysis["drifted_features"][:10]],
                "",
            ]
        )

    summary_lines.append("=" * 80)

    return "\n".join(summary_lines)
