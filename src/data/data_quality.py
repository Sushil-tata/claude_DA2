"""
Data Quality Module

Comprehensive data quality assessment including missing value analysis,
outlier detection, distribution analysis, consistency checks, and quality scoring.

Author: Principal Data Science Decision Agent
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


@dataclass
class DataQualityConfig:
    """Configuration for data quality checks."""

    # Missing value thresholds
    missing_threshold_warning: float = 0.05  # 5%
    missing_threshold_critical: float = 0.30  # 30%

    # Outlier detection
    iqr_multiplier: float = 1.5
    z_score_threshold: float = 3.0
    isolation_forest_contamination: float = 0.1

    # Distribution analysis
    skewness_threshold: float = 1.0
    kurtosis_threshold: float = 3.0
    normality_alpha: float = 0.05

    # Consistency checks
    duplicate_threshold: float = 0.01  # 1%

    # Quality scoring weights
    weight_completeness: float = 0.25
    weight_validity: float = 0.25
    weight_consistency: float = 0.25
    weight_uniqueness: float = 0.25


@dataclass
class QualityReport:
    """Container for data quality assessment results."""

    overall_score: float
    dimension_scores: Dict[str, float]
    missing_analysis: Dict[str, Any]
    outlier_analysis: Dict[str, Any]
    distribution_analysis: Dict[str, Any]
    consistency_analysis: Dict[str, Any]
    issues: List[Dict[str, Any]]
    recommendations: List[str]


class DataQualityAnalyzer:
    """
    Comprehensive data quality analyzer.

    Performs multiple types of quality checks:
    - Missing value analysis
    - Outlier detection (IQR, Z-score, Isolation Forest)
    - Distribution analysis
    - Consistency checks
    - Data quality scoring

    Examples:
        >>> analyzer = DataQualityAnalyzer()
        >>> report = analyzer.analyze(df)
        >>> print(f"Quality Score: {report.overall_score:.2f}")
        >>> print(f"Issues: {len(report.issues)}")
    """

    def __init__(self, config: Optional[DataQualityConfig] = None):
        """
        Initialize DataQualityAnalyzer.

        Args:
            config: Configuration object for quality checks
        """
        self.config = config or DataQualityConfig()
        logger.info("DataQualityAnalyzer initialized")

    def analyze(
        self, df: pd.DataFrame, target_column: Optional[str] = None
    ) -> QualityReport:
        """
        Perform comprehensive data quality analysis.

        Args:
            df: DataFrame to analyze
            target_column: Optional target column name for supervised tasks

        Returns:
            QualityReport with all analysis results
        """
        logger.info("Starting comprehensive data quality analysis")
        logger.info("Dataset shape: {} rows, {} columns", len(df), len(df.columns))

        issues = []
        recommendations = []

        # Missing value analysis
        logger.info("Analyzing missing values...")
        missing_analysis = self.analyze_missing_values(df)
        issues.extend(missing_analysis.get("issues", []))
        recommendations.extend(missing_analysis.get("recommendations", []))

        # Outlier analysis
        logger.info("Detecting outliers...")
        outlier_analysis = self.detect_outliers(df)
        issues.extend(outlier_analysis.get("issues", []))
        recommendations.extend(outlier_analysis.get("recommendations", []))

        # Distribution analysis
        logger.info("Analyzing distributions...")
        distribution_analysis = self.analyze_distributions(df)
        issues.extend(distribution_analysis.get("issues", []))
        recommendations.extend(distribution_analysis.get("recommendations", []))

        # Consistency analysis
        logger.info("Checking consistency...")
        consistency_analysis = self.check_consistency(df, target_column)
        issues.extend(consistency_analysis.get("issues", []))
        recommendations.extend(consistency_analysis.get("recommendations", []))

        # Calculate dimension scores
        dimension_scores = self._calculate_dimension_scores(
            missing_analysis, outlier_analysis, distribution_analysis, consistency_analysis
        )

        # Calculate overall score
        overall_score = self._calculate_overall_score(dimension_scores)

        report = QualityReport(
            overall_score=overall_score,
            dimension_scores=dimension_scores,
            missing_analysis=missing_analysis,
            outlier_analysis=outlier_analysis,
            distribution_analysis=distribution_analysis,
            consistency_analysis=consistency_analysis,
            issues=issues,
            recommendations=recommendations,
        )

        logger.info("Data quality analysis complete. Overall score: {:.2f}", overall_score)
        return report

    def analyze_missing_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive missing value analysis.

        Args:
            df: DataFrame to analyze

        Returns:
            Dictionary with missing value analysis results
        """
        total_cells = df.shape[0] * df.shape[1]
        total_missing = df.isnull().sum().sum()

        # Per-column analysis
        column_missing = []
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_pct = missing_count / len(df)

            if missing_count > 0:
                column_missing.append(
                    {
                        "column": col,
                        "missing_count": int(missing_count),
                        "missing_percentage": float(missing_pct),
                        "dtype": str(df[col].dtype),
                    }
                )

        # Sort by missing percentage
        column_missing = sorted(
            column_missing, key=lambda x: x["missing_percentage"], reverse=True
        )

        # Identify patterns
        missing_patterns = self._identify_missing_patterns(df)

        # Generate issues and recommendations
        issues = []
        recommendations = []

        for item in column_missing:
            if item["missing_percentage"] > self.config.missing_threshold_critical:
                issues.append(
                    {
                        "type": "missing_values",
                        "severity": "critical",
                        "column": item["column"],
                        "message": f"Column has {item['missing_percentage']:.1%} missing values",
                    }
                )
                recommendations.append(
                    f"Consider dropping column '{item['column']}' or using advanced imputation"
                )
            elif item["missing_percentage"] > self.config.missing_threshold_warning:
                issues.append(
                    {
                        "type": "missing_values",
                        "severity": "warning",
                        "column": item["column"],
                        "message": f"Column has {item['missing_percentage']:.1%} missing values",
                    }
                )
                recommendations.append(
                    f"Investigate missing pattern in '{item['column']}' and consider imputation"
                )

        analysis = {
            "total_missing": int(total_missing),
            "total_cells": int(total_cells),
            "overall_missing_rate": float(total_missing / total_cells),
            "columns_with_missing": len(column_missing),
            "column_details": column_missing,
            "missing_patterns": missing_patterns,
            "completeness_score": 1.0 - (total_missing / total_cells),
            "issues": issues,
            "recommendations": recommendations,
        }

        logger.debug(
            "Missing value analysis: {:.1%} missing overall",
            analysis["overall_missing_rate"],
        )

        return analysis

    def detect_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect outliers using multiple methods.

        Args:
            df: DataFrame to analyze

        Returns:
            Dictionary with outlier detection results
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if not numeric_cols:
            logger.warning("No numeric columns found for outlier detection")
            return {
                "methods": {},
                "summary": {},
                "issues": [],
                "recommendations": [],
            }

        results = {}
        issues = []
        recommendations = []

        for col in numeric_cols:
            col_data = df[col].dropna()

            if len(col_data) == 0:
                continue

            col_results = {
                "iqr_outliers": self._detect_outliers_iqr(col_data),
                "zscore_outliers": self._detect_outliers_zscore(col_data),
            }

            results[col] = col_results

            # Generate issues
            total_outliers = len(
                set(col_results["iqr_outliers"]) | set(col_results["zscore_outliers"])
            )
            outlier_pct = total_outliers / len(col_data)

            if outlier_pct > 0.05:  # More than 5% outliers
                issues.append(
                    {
                        "type": "outliers",
                        "severity": "warning" if outlier_pct < 0.10 else "critical",
                        "column": col,
                        "message": f"{outlier_pct:.1%} of values are outliers",
                    }
                )
                recommendations.append(
                    f"Review outliers in '{col}' - consider capping, transformation, or removal"
                )

        # Isolation Forest for multivariate outliers
        if len(numeric_cols) > 1:
            isolation_outliers = self._detect_outliers_isolation_forest(
                df[numeric_cols].dropna()
            )
            results["multivariate"] = {"isolation_forest_outliers": isolation_outliers}

        # Calculate summary
        total_outlier_flags = sum(
            len(col_res["iqr_outliers"]) + len(col_res["zscore_outliers"])
            for col_res in results.values()
            if isinstance(col_res, dict) and "iqr_outliers" in col_res
        )

        summary = {
            "total_outlier_flags": int(total_outlier_flags),
            "columns_analyzed": len(numeric_cols),
            "outlier_rate": float(total_outlier_flags / (len(df) * len(numeric_cols))),
        }

        analysis = {
            "methods": results,
            "summary": summary,
            "issues": issues,
            "recommendations": recommendations,
        }

        logger.debug("Outlier detection: {} flags across {} columns", total_outlier_flags, len(numeric_cols))

        return analysis

    def analyze_distributions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze statistical distributions of numeric columns.

        Args:
            df: DataFrame to analyze

        Returns:
            Dictionary with distribution analysis results
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if not numeric_cols:
            logger.warning("No numeric columns found for distribution analysis")
            return {"distributions": {}, "issues": [], "recommendations": []}

        distributions = {}
        issues = []
        recommendations = []

        for col in numeric_cols:
            col_data = df[col].dropna()

            if len(col_data) < 3:
                continue

            # Calculate statistics
            skewness = float(stats.skew(col_data))
            kurtosis = float(stats.kurtosis(col_data))

            # Normality test (Shapiro-Wilk for small samples, Anderson-Darling for large)
            if len(col_data) <= 5000:
                _, p_value = stats.shapiro(col_data)
                normality_test = "shapiro"
            else:
                # Use Anderson-Darling for larger samples
                result = stats.anderson(col_data)
                p_value = 0.05 if result.statistic > result.critical_values[2] else 0.10
                normality_test = "anderson"

            is_normal = p_value > self.config.normality_alpha

            distributions[col] = {
                "mean": float(col_data.mean()),
                "median": float(col_data.median()),
                "std": float(col_data.std()),
                "min": float(col_data.min()),
                "max": float(col_data.max()),
                "skewness": skewness,
                "kurtosis": kurtosis,
                "is_normal": is_normal,
                "normality_test": normality_test,
                "normality_p_value": float(p_value),
            }

            # Generate issues
            if abs(skewness) > self.config.skewness_threshold:
                severity = "warning" if abs(skewness) < 2 else "critical"
                issues.append(
                    {
                        "type": "distribution",
                        "severity": severity,
                        "column": col,
                        "message": f"High skewness: {skewness:.2f}",
                    }
                )
                recommendations.append(
                    f"Consider log or Box-Cox transformation for '{col}' to reduce skewness"
                )

            if abs(kurtosis) > self.config.kurtosis_threshold:
                issues.append(
                    {
                        "type": "distribution",
                        "severity": "info",
                        "column": col,
                        "message": f"High kurtosis: {kurtosis:.2f}",
                    }
                )

        analysis = {
            "distributions": distributions,
            "issues": issues,
            "recommendations": recommendations,
        }

        logger.debug("Distribution analysis complete for {} columns", len(numeric_cols))

        return analysis

    def check_consistency(
        self, df: pd.DataFrame, target_column: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Check data consistency including duplicates and referential integrity.

        Args:
            df: DataFrame to analyze
            target_column: Optional target column for additional checks

        Returns:
            Dictionary with consistency analysis results
        """
        issues = []
        recommendations = []

        # Duplicate analysis
        duplicate_rows = df.duplicated().sum()
        duplicate_pct = duplicate_rows / len(df)

        # Per-column unique value analysis
        uniqueness = {}
        for col in df.columns:
            unique_count = df[col].nunique()
            unique_pct = unique_count / len(df)

            uniqueness[col] = {
                "unique_count": int(unique_count),
                "unique_percentage": float(unique_pct),
                "is_unique": unique_pct == 1.0,
                "is_constant": unique_count == 1,
            }

            # Flag constant columns
            if unique_count == 1:
                issues.append(
                    {
                        "type": "consistency",
                        "severity": "warning",
                        "column": col,
                        "message": "Column has only one unique value",
                    }
                )
                recommendations.append(f"Consider removing constant column '{col}'")

        # Check for duplicate columns
        duplicate_columns = self._find_duplicate_columns(df)

        if duplicate_columns:
            for col1, col2 in duplicate_columns:
                issues.append(
                    {
                        "type": "consistency",
                        "severity": "warning",
                        "column": f"{col1}, {col2}",
                        "message": "Columns have identical values",
                    }
                )
                recommendations.append(
                    f"Consider removing duplicate column '{col2}' (duplicate of '{col1}')"
                )

        # Duplicate row issues
        if duplicate_pct > self.config.duplicate_threshold:
            issues.append(
                {
                    "type": "consistency",
                    "severity": "warning",
                    "column": "all",
                    "message": f"{duplicate_pct:.1%} of rows are duplicates",
                }
            )
            recommendations.append("Review and remove duplicate rows if appropriate")

        # Target leakage check (if target column specified)
        leakage_risk = {}
        if target_column and target_column in df.columns:
            leakage_risk = self._check_target_leakage(df, target_column)
            if leakage_risk["high_risk_columns"]:
                issues.append(
                    {
                        "type": "consistency",
                        "severity": "critical",
                        "column": ", ".join(leakage_risk["high_risk_columns"]),
                        "message": "Potential target leakage detected",
                    }
                )
                recommendations.append(
                    "Review high-correlation columns for potential target leakage"
                )

        # Calculate consistency score
        consistency_score = 1.0 - duplicate_pct
        if duplicate_columns:
            consistency_score *= 0.9

        analysis = {
            "duplicate_rows": int(duplicate_rows),
            "duplicate_percentage": float(duplicate_pct),
            "uniqueness": uniqueness,
            "duplicate_columns": duplicate_columns,
            "leakage_risk": leakage_risk,
            "consistency_score": float(consistency_score),
            "issues": issues,
            "recommendations": recommendations,
        }

        logger.debug(
            "Consistency check: {} duplicates ({:.1%})",
            duplicate_rows,
            duplicate_pct,
        )

        return analysis

    def calculate_quality_score(self, report: QualityReport) -> float:
        """
        Calculate overall data quality score.

        Args:
            report: QualityReport object

        Returns:
            Quality score between 0 and 1
        """
        return report.overall_score

    def _identify_missing_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Identify patterns in missing data."""
        # Find rows with multiple missing values
        missing_per_row = df.isnull().sum(axis=1)
        rows_with_missing = (missing_per_row > 0).sum()

        # Find columns that tend to be missing together
        if rows_with_missing > 0:
            missing_mask = df.isnull()
            # Calculate correlation between missing patterns
            missing_corr = missing_mask.corr()

            # Find highly correlated missing patterns
            correlated_missing = []
            for i in range(len(missing_corr.columns)):
                for j in range(i + 1, len(missing_corr.columns)):
                    if missing_corr.iloc[i, j] > 0.7:  # High correlation
                        correlated_missing.append(
                            {
                                "column1": missing_corr.columns[i],
                                "column2": missing_corr.columns[j],
                                "correlation": float(missing_corr.iloc[i, j]),
                            }
                        )

            return {
                "rows_with_missing": int(rows_with_missing),
                "correlated_missing": correlated_missing,
            }

        return {"rows_with_missing": 0, "correlated_missing": []}

    def _detect_outliers_iqr(self, data: pd.Series) -> List[int]:
        """Detect outliers using IQR method."""
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - self.config.iqr_multiplier * IQR
        upper_bound = Q3 + self.config.iqr_multiplier * IQR

        outliers = data[(data < lower_bound) | (data > upper_bound)]
        return outliers.index.tolist()

    def _detect_outliers_zscore(self, data: pd.Series) -> List[int]:
        """Detect outliers using Z-score method."""
        z_scores = np.abs(stats.zscore(data))
        outliers = data[z_scores > self.config.z_score_threshold]
        return outliers.index.tolist()

    def _detect_outliers_isolation_forest(
        self, df: pd.DataFrame
    ) -> List[int]:
        """Detect multivariate outliers using Isolation Forest."""
        if len(df) < 10:
            return []

        try:
            clf = IsolationForest(
                contamination=self.config.isolation_forest_contamination,
                random_state=42,
            )
            predictions = clf.fit_predict(df)
            outliers = df[predictions == -1]
            return outliers.index.tolist()
        except Exception as e:
            logger.warning("Isolation Forest failed: {}", str(e))
            return []

    def _find_duplicate_columns(self, df: pd.DataFrame) -> List[Tuple[str, str]]:
        """Find columns with identical values."""
        duplicates = []
        columns = df.columns.tolist()

        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                if df[columns[i]].equals(df[columns[j]]):
                    duplicates.append((columns[i], columns[j]))

        return duplicates

    def _check_target_leakage(
        self, df: pd.DataFrame, target_column: str
    ) -> Dict[str, Any]:
        """Check for potential target leakage."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if target_column not in numeric_cols:
            return {"high_risk_columns": [], "correlations": {}}

        # Calculate correlations with target
        correlations = {}
        high_risk_columns = []

        for col in numeric_cols:
            if col != target_column:
                corr = df[[col, target_column]].corr().iloc[0, 1]
                if not np.isnan(corr):
                    correlations[col] = float(abs(corr))

                    # Flag very high correlations as potential leakage
                    if abs(corr) > 0.95:
                        high_risk_columns.append(col)

        return {
            "high_risk_columns": high_risk_columns,
            "correlations": correlations,
        }

    def _calculate_dimension_scores(
        self,
        missing_analysis: Dict,
        outlier_analysis: Dict,
        distribution_analysis: Dict,
        consistency_analysis: Dict,
    ) -> Dict[str, float]:
        """Calculate scores for each quality dimension."""
        # Completeness score (from missing analysis)
        completeness = missing_analysis.get("completeness_score", 1.0)

        # Validity score (based on outliers and distributions)
        outlier_rate = outlier_analysis.get("summary", {}).get("outlier_rate", 0)
        validity = 1.0 - min(outlier_rate, 0.5)  # Cap penalty at 50%

        # Consistency score
        consistency = consistency_analysis.get("consistency_score", 1.0)

        # Uniqueness score
        duplicate_pct = consistency_analysis.get("duplicate_percentage", 0)
        uniqueness = 1.0 - duplicate_pct

        return {
            "completeness": completeness,
            "validity": validity,
            "consistency": consistency,
            "uniqueness": uniqueness,
        }

    def _calculate_overall_score(self, dimension_scores: Dict[str, float]) -> float:
        """Calculate weighted overall quality score."""
        score = (
            dimension_scores["completeness"] * self.config.weight_completeness
            + dimension_scores["validity"] * self.config.weight_validity
            + dimension_scores["consistency"] * self.config.weight_consistency
            + dimension_scores["uniqueness"] * self.config.weight_uniqueness
        )

        return float(np.clip(score, 0, 1))


def generate_quality_report_html(report: QualityReport) -> str:
    """
    Generate HTML report from quality analysis.

    Args:
        report: QualityReport object

    Returns:
        HTML string
    """
    html = f"""
    <html>
    <head>
        <title>Data Quality Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .score {{ font-size: 48px; font-weight: bold; }}
            .good {{ color: green; }}
            .warning {{ color: orange; }}
            .critical {{ color: red; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #4CAF50; color: white; }}
        </style>
    </head>
    <body>
        <h1>Data Quality Report</h1>
        <div class="score {'good' if report.overall_score > 0.8 else 'warning' if report.overall_score > 0.6 else 'critical'}">
            Overall Score: {report.overall_score:.2f}
        </div>
        
        <h2>Dimension Scores</h2>
        <table>
            <tr><th>Dimension</th><th>Score</th></tr>
            {''.join(f'<tr><td>{k}</td><td>{v:.2f}</td></tr>' for k, v in report.dimension_scores.items())}
        </table>
        
        <h2>Issues ({len(report.issues)})</h2>
        <table>
            <tr><th>Type</th><th>Severity</th><th>Column</th><th>Message</th></tr>
            {''.join(f'<tr><td>{i["type"]}</td><td class="{i["severity"]}">{i["severity"]}</td><td>{i.get("column", "")}</td><td>{i["message"]}</td></tr>' for i in report.issues)}
        </table>
        
        <h2>Recommendations</h2>
        <ul>
            {''.join(f'<li>{r}</li>' for r in report.recommendations)}
        </ul>
    </body>
    </html>
    """
    return html
