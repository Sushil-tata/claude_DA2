"""
Production Drift Monitoring Framework

This module provides comprehensive drift monitoring for production models,
including data drift, concept drift, and feature distribution monitoring.

Usage Example:
    >>> from drift_monitor import DriftMonitor
    >>> monitor = DriftMonitor(reference_data=train_df, alert_threshold=0.05)
    >>> drift_report = monitor.detect_drift(production_df)
    >>> if drift_report.has_drift:
    ...     print(f"Alert! {len(drift_report.drifted_features)} features drifting")
    ...     monitor.send_alert(drift_report)
"""

from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ks_2samp, chi2_contingency
import warnings


@dataclass
class DriftReport:
    """Container for drift detection results."""
    has_drift: bool
    drifted_features: List[str] = field(default_factory=list)
    drift_scores: Dict[str, float] = field(default_factory=dict)
    p_values: Dict[str, float] = field(default_factory=dict)
    severity_levels: Dict[str, str] = field(default_factory=dict)  # low, medium, high, critical
    concept_drift: bool = False
    concept_drift_metrics: Dict[str, float] = field(default_factory=dict)
    alerts: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    reference_period: Optional[str] = None
    monitoring_period: Optional[str] = None


class DriftMonitor:
    """
    Comprehensive production drift monitoring system.
    
    Detects both data drift (feature distribution changes) and concept drift
    (relationship between features and target changes).
    
    Attributes:
        reference_data: Reference dataset (typically training data)
        alert_threshold: P-value threshold for drift alerts (default: 0.05)
        critical_threshold: P-value for critical alerts (default: 0.01)
        min_samples: Minimum samples for statistical tests (default: 100)
        categorical_threshold: Threshold for categorical drift (default: 0.1)
    
    Example:
        >>> monitor = DriftMonitor(reference_data=train_df)
        >>> drift_report = monitor.detect_drift(prod_df)
        >>> monitor.visualize_drift(drift_report, output_path='drift_report.png')
    """
    
    def __init__(
        self,
        reference_data: Optional[pd.DataFrame] = None,
        alert_threshold: float = 0.05,
        critical_threshold: float = 0.01,
        min_samples: int = 100,
        categorical_threshold: float = 0.1
    ):
        self.reference_data = reference_data
        self.alert_threshold = alert_threshold
        self.critical_threshold = critical_threshold
        self.min_samples = min_samples
        self.categorical_threshold = categorical_threshold
        self.drift_history: List[DriftReport] = []
        
        logger.info(f"Initialized DriftMonitor with thresholds: "
                   f"alert={alert_threshold}, critical={critical_threshold}")
    
    def set_reference_data(self, reference_data: pd.DataFrame) -> None:
        """Set or update reference data."""
        self.reference_data = reference_data.copy()
        logger.info(f"Reference data set with {len(reference_data)} samples, "
                   f"{len(reference_data.columns)} features")
    
    def detect_drift(
        self,
        production_data: pd.DataFrame,
        features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        target_col: Optional[str] = None,
        production_target: Optional[pd.Series] = None
    ) -> DriftReport:
        """
        Detect drift in production data.
        
        Args:
            production_data: Production dataset to monitor
            features: Features to monitor (if None, uses all common features)
            categorical_features: List of categorical feature names
            target_col: Target column name for concept drift detection
            production_target: Production target values for concept drift
        
        Returns:
            DriftReport with comprehensive drift analysis
        """
        if self.reference_data is None:
            raise ValueError("Reference data not set. Call set_reference_data() first.")
        
        logger.info(f"Starting drift detection on {len(production_data)} samples")
        
        # Determine features to monitor
        if features is None:
            features = [col for col in self.reference_data.columns
                       if col in production_data.columns and col != target_col]
        
        if categorical_features is None:
            categorical_features = []
        
        # Validate sample sizes
        if len(production_data) < self.min_samples:
            logger.warning(f"Production data ({len(production_data)}) below minimum "
                         f"samples ({self.min_samples})")
        
        # Initialize report
        report = DriftReport(has_drift=False)
        
        # Detect data drift for each feature
        for feature in features:
            if feature not in self.reference_data.columns or \
               feature not in production_data.columns:
                continue
            
            ref_values = self.reference_data[feature].dropna()
            prod_values = production_data[feature].dropna()
            
            if len(ref_values) == 0 or len(prod_values) == 0:
                continue
            
            # Choose appropriate test based on feature type
            if feature in categorical_features or \
               self.reference_data[feature].dtype == 'object':
                drift_score, p_value = self._test_categorical_drift(
                    ref_values, prod_values
                )
            else:
                drift_score, p_value = self._test_numerical_drift(
                    ref_values, prod_values
                )
            
            report.drift_scores[feature] = drift_score
            report.p_values[feature] = p_value
            
            # Determine severity
            severity = self._get_severity(p_value)
            report.severity_levels[feature] = severity
            
            # Check if drift detected
            if p_value < self.alert_threshold:
                report.drifted_features.append(feature)
                report.has_drift = True
                
                alert_msg = (f"Drift detected in '{feature}': "
                           f"score={drift_score:.4f}, p-value={p_value:.4e}, "
                           f"severity={severity}")
                report.alerts.append(alert_msg)
                logger.warning(alert_msg)
        
        # Detect concept drift if target provided
        if target_col and production_target is not None:
            self._detect_concept_drift(report, target_col, production_target)
        
        # Store in history
        self.drift_history.append(report)
        
        logger.info(f"Drift detection complete. Drift detected: {report.has_drift}, "
                   f"Drifted features: {len(report.drifted_features)}")
        
        return report
    
    def _test_numerical_drift(
        self,
        reference: pd.Series,
        production: pd.Series
    ) -> Tuple[float, float]:
        """
        Test for drift in numerical features using Kolmogorov-Smirnov test.
        
        Returns:
            Tuple of (KS statistic, p-value)
        """
        try:
            statistic, p_value = ks_2samp(reference, production)
            return float(statistic), float(p_value)
        except Exception as e:
            logger.error(f"Error in KS test: {e}")
            return 0.0, 1.0
    
    def _test_categorical_drift(
        self,
        reference: pd.Series,
        production: pd.Series
    ) -> Tuple[float, float]:
        """
        Test for drift in categorical features using Chi-square test.
        
        Returns:
            Tuple of (Chi-square statistic, p-value)
        """
        try:
            # Get value counts
            ref_counts = reference.value_counts()
            prod_counts = production.value_counts()
            
            # Align categories
            all_categories = set(ref_counts.index) | set(prod_counts.index)
            
            ref_aligned = [ref_counts.get(cat, 0) for cat in all_categories]
            prod_aligned = [prod_counts.get(cat, 0) for cat in all_categories]
            
            # Create contingency table
            contingency_table = np.array([ref_aligned, prod_aligned])
            
            # Chi-square test
            chi2, p_value, dof, expected = chi2_contingency(contingency_table)
            
            return float(chi2), float(p_value)
        
        except Exception as e:
            logger.error(f"Error in Chi-square test: {e}")
            return 0.0, 1.0
    
    def _get_severity(self, p_value: float) -> str:
        """Determine severity level based on p-value."""
        if p_value < self.critical_threshold:
            return 'critical'
        elif p_value < self.alert_threshold:
            return 'high'
        elif p_value < self.alert_threshold * 2:
            return 'medium'
        else:
            return 'low'
    
    def _detect_concept_drift(
        self,
        report: DriftReport,
        target_col: str,
        production_target: pd.Series
    ) -> None:
        """
        Detect concept drift by comparing target distributions.
        
        Concept drift occurs when the relationship between features and target changes.
        """
        if target_col not in self.reference_data.columns:
            logger.warning(f"Target column '{target_col}' not in reference data")
            return
        
        ref_target = self.reference_data[target_col].dropna()
        prod_target = production_target.dropna()
        
        if len(ref_target) == 0 or len(prod_target) == 0:
            return
        
        # Test target distribution
        if ref_target.dtype in ['int64', 'float64']:
            statistic, p_value = ks_2samp(ref_target, prod_target)
            test_name = 'KS'
        else:
            statistic, p_value = self._test_categorical_drift(ref_target, prod_target)
            test_name = 'Chi-square'
        
        report.concept_drift_metrics = {
            'statistic': float(statistic),
            'p_value': float(p_value),
            'test': test_name,
            'reference_mean': float(ref_target.mean()) if ref_target.dtype in ['int64', 'float64'] else None,
            'production_mean': float(prod_target.mean()) if prod_target.dtype in ['int64', 'float64'] else None
        }
        
        if p_value < self.alert_threshold:
            report.concept_drift = True
            report.alerts.append(
                f"CONCEPT DRIFT detected in target '{target_col}': "
                f"{test_name} p-value={p_value:.4e}"
            )
            logger.warning(f"Concept drift detected in target '{target_col}'")
    
    def monitor_feature_distributions(
        self,
        production_data: pd.DataFrame,
        features: List[str],
        bins: int = 30
    ) -> Dict[str, Dict[str, Any]]:
        """
        Monitor and compare feature distributions.
        
        Args:
            production_data: Production data
            features: Features to monitor
            bins: Number of bins for histograms
        
        Returns:
            Dictionary with distribution statistics for each feature
        """
        if self.reference_data is None:
            raise ValueError("Reference data not set")
        
        distribution_stats = {}
        
        for feature in features:
            if feature not in self.reference_data.columns or \
               feature not in production_data.columns:
                continue
            
            ref_values = self.reference_data[feature].dropna()
            prod_values = production_data[feature].dropna()
            
            stats_dict = {
                'reference': {
                    'mean': float(ref_values.mean()) if ref_values.dtype in ['int64', 'float64'] else None,
                    'std': float(ref_values.std()) if ref_values.dtype in ['int64', 'float64'] else None,
                    'min': float(ref_values.min()) if ref_values.dtype in ['int64', 'float64'] else None,
                    'max': float(ref_values.max()) if ref_values.dtype in ['int64', 'float64'] else None,
                    'missing_rate': float(self.reference_data[feature].isna().mean())
                },
                'production': {
                    'mean': float(prod_values.mean()) if prod_values.dtype in ['int64', 'float64'] else None,
                    'std': float(prod_values.std()) if prod_values.dtype in ['int64', 'float64'] else None,
                    'min': float(prod_values.min()) if prod_values.dtype in ['int64', 'float64'] else None,
                    'max': float(prod_values.max()) if prod_values.dtype in ['int64', 'float64'] else None,
                    'missing_rate': float(production_data[feature].isna().mean())
                }
            }
            
            distribution_stats[feature] = stats_dict
        
        return distribution_stats
    
    def calculate_drift_score(
        self,
        production_data: pd.DataFrame,
        features: Optional[List[str]] = None
    ) -> float:
        """
        Calculate overall drift score across all features.
        
        Args:
            production_data: Production data
            features: Features to include in score
        
        Returns:
            Composite drift score (0-1, higher = more drift)
        """
        report = self.detect_drift(production_data, features)
        
        if not report.drift_scores:
            return 0.0
        
        # Calculate weighted drift score
        drift_values = list(report.drift_scores.values())
        mean_drift = np.mean(drift_values)
        max_drift = np.max(drift_values)
        
        # Composite score: 70% mean, 30% max
        composite_score = 0.7 * mean_drift + 0.3 * max_drift
        
        return float(composite_score)
    
    def visualize_drift(
        self,
        drift_report: DriftReport,
        top_n: int = 10,
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 6)
    ) -> None:
        """
        Visualize drift scores for top drifting features.
        
        Args:
            drift_report: DriftReport from detect_drift
            top_n: Number of top features to show
            output_path: Path to save plot
            figsize: Figure size
        """
        if not drift_report.drift_scores:
            logger.warning("No drift scores to visualize")
            return
        
        # Sort by drift score
        sorted_features = sorted(
            drift_report.drift_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]
        
        features = [f[0] for f in sorted_features]
        scores = [f[1] for f in sorted_features]
        severities = [drift_report.severity_levels.get(f, 'low') for f in features]
        
        # Color by severity
        color_map = {
            'critical': '#e74c3c',
            'high': '#e67e22',
            'medium': '#f39c12',
            'low': '#3498db'
        }
        colors = [color_map.get(s, 'gray') for s in severities]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        bars = ax.barh(range(len(features)), scores, color=colors, alpha=0.7,
                      edgecolor='black')
        
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features)
        ax.set_xlabel('Drift Score', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        ax.set_title('Top Features by Drift Score', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels and p-values
        for i, (bar, score, feature) in enumerate(zip(bars, scores, features)):
            p_val = drift_report.p_values.get(feature, 1.0)
            ax.text(score, i, f' {score:.3f} (p={p_val:.2e})',
                   va='center', fontsize=9)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=color_map['critical'], label='Critical'),
            Patch(facecolor=color_map['high'], label='High'),
            Patch(facecolor=color_map['medium'], label='Medium'),
            Patch(facecolor=color_map['low'], label='Low')
        ]
        ax.legend(handles=legend_elements, loc='lower right', title='Severity')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Drift visualization saved to {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def visualize_distribution_comparison(
        self,
        production_data: pd.DataFrame,
        feature: str,
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6)
    ) -> None:
        """
        Visualize distribution comparison for a specific feature.
        
        Args:
            production_data: Production data
            feature: Feature name to visualize
            output_path: Path to save plot
            figsize: Figure size
        """
        if self.reference_data is None:
            raise ValueError("Reference data not set")
        
        if feature not in self.reference_data.columns or \
           feature not in production_data.columns:
            raise ValueError(f"Feature '{feature}' not found in data")
        
        ref_values = self.reference_data[feature].dropna()
        prod_values = production_data[feature].dropna()
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Histogram comparison
        if ref_values.dtype in ['int64', 'float64']:
            # Numerical feature
            axes[0].hist(ref_values, bins=30, alpha=0.5, label='Reference',
                        color='blue', density=True)
            axes[0].hist(prod_values, bins=30, alpha=0.5, label='Production',
                        color='red', density=True)
            axes[0].set_xlabel(feature, fontsize=11)
            axes[0].set_ylabel('Density', fontsize=11)
            axes[0].set_title('Distribution Comparison', fontsize=12, fontweight='bold')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Q-Q plot
            from scipy.stats import probplot
            axes[1].scatter(
                np.sort(ref_values.sample(min(1000, len(ref_values)))),
                np.sort(prod_values.sample(min(1000, len(prod_values)))),
                alpha=0.5
            )
            min_val = min(ref_values.min(), prod_values.min())
            max_val = max(ref_values.max(), prod_values.max())
            axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
            axes[1].set_xlabel('Reference Quantiles', fontsize=11)
            axes[1].set_ylabel('Production Quantiles', fontsize=11)
            axes[1].set_title('Q-Q Plot', fontsize=12, fontweight='bold')
            axes[1].grid(True, alpha=0.3)
        
        else:
            # Categorical feature
            ref_counts = ref_values.value_counts(normalize=True).head(10)
            prod_counts = prod_values.value_counts(normalize=True).head(10)
            
            categories = list(set(ref_counts.index) | set(prod_counts.index))
            
            x = np.arange(len(categories))
            width = 0.35
            
            axes[0].bar(x - width/2, [ref_counts.get(c, 0) for c in categories],
                       width, label='Reference', color='blue', alpha=0.7)
            axes[0].bar(x + width/2, [prod_counts.get(c, 0) for c in categories],
                       width, label='Production', color='red', alpha=0.7)
            
            axes[0].set_xlabel('Category', fontsize=11)
            axes[0].set_ylabel('Proportion', fontsize=11)
            axes[0].set_title('Distribution Comparison', fontsize=12, fontweight='bold')
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(categories, rotation=45, ha='right')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3, axis='y')
            
            # Hide second subplot for categorical
            axes[1].axis('off')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Distribution comparison saved to {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def track_drift_over_time(self) -> pd.DataFrame:
        """
        Create timeline of drift metrics from history.
        
        Returns:
            DataFrame with drift metrics over time
        """
        if not self.drift_history:
            logger.warning("No drift history available")
            return pd.DataFrame()
        
        timeline = []
        for report in self.drift_history:
            timeline.append({
                'timestamp': report.timestamp,
                'has_drift': report.has_drift,
                'num_drifted_features': len(report.drifted_features),
                'concept_drift': report.concept_drift,
                'num_alerts': len(report.alerts),
                'mean_drift_score': np.mean(list(report.drift_scores.values()))
                                   if report.drift_scores else 0.0
            })
        
        return pd.DataFrame(timeline)
    
    def send_alert(
        self,
        drift_report: DriftReport,
        alert_callback: Optional[Callable] = None
    ) -> None:
        """
        Send drift alert (placeholder for integration with monitoring systems).
        
        Args:
            drift_report: DriftReport with alerts
            alert_callback: Optional callback function for custom alerting
        """
        if not drift_report.alerts:
            logger.info("No alerts to send")
            return
        
        logger.warning(f"DRIFT ALERT: {len(drift_report.alerts)} issues detected")
        
        for alert in drift_report.alerts:
            logger.warning(f"  - {alert}")
        
        if alert_callback:
            alert_callback(drift_report)
    
    def generate_report(
        self,
        drift_report: DriftReport,
        output_path: str = 'drift_report.html'
    ) -> str:
        """
        Generate comprehensive HTML drift monitoring report.
        
        Args:
            drift_report: DriftReport from detect_drift
            output_path: Output file path
        
        Returns:
            Path to generated report
        """
        status_color = '#e74c3c' if drift_report.has_drift else '#2ecc71'
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Drift Monitoring Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
                .container {{ background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
                h2 {{ color: #34495e; margin-top: 30px; }}
                .metric {{ background-color: #ecf0f1; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                .metric-label {{ font-weight: bold; color: #7f8c8d; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: {status_color}; }}
                .alert {{ background-color: #f8d7da; border-left: 4px solid #e74c3c; padding: 10px; margin: 10px 0; }}
                .warning {{ background-color: #fff3cd; border-left: 4px solid #f39c12; padding: 10px; margin: 10px 0; }}
                .success {{ background-color: #d4edda; border-left: 4px solid #2ecc71; padding: 10px; margin: 10px 0; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #3498db; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .critical {{ color: #e74c3c; font-weight: bold; }}
                .high {{ color: #e67e22; font-weight: bold; }}
                .medium {{ color: #f39c12; }}
                .low {{ color: #3498db; }}
                .timestamp {{ color: #95a5a6; font-size: 14px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Drift Monitoring Report</h1>
                <p class="timestamp">Generated: {drift_report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <div class="metric">
                    <div class="metric-label">Drift Status</div>
                    <div class="metric-value">
                        {"DRIFT DETECTED" if drift_report.has_drift else "NO DRIFT"}
                    </div>
                    <div>Drifted Features: <strong>{len(drift_report.drifted_features)}</strong></div>
                </div>
        """
        
        # Concept drift
        if drift_report.concept_drift:
            html_content += """
                <div class="alert">
                    <strong>⚠ CONCEPT DRIFT DETECTED</strong><br>
                    The relationship between features and target has changed.
                    Model retraining recommended.
                </div>
            """
            
            if drift_report.concept_drift_metrics:
                html_content += f"""
                    <div class="metric">
                        <div class="metric-label">Concept Drift Metrics</div>
                        <div>Test: {drift_report.concept_drift_metrics.get('test', 'N/A')}</div>
                        <div>P-value: {drift_report.concept_drift_metrics.get('p_value', 0):.4e}</div>
                    </div>
                """
        
        # Alerts
        if drift_report.alerts:
            html_content += "<h2>Alerts</h2>"
            for alert in drift_report.alerts:
                severity = 'alert' if 'CRITICAL' in alert or 'CONCEPT DRIFT' in alert else 'warning'
                html_content += f'<div class="{severity}">{alert}</div>'
        else:
            html_content += '<div class="success">No drift alerts. System operating normally.</div>'
        
        # Drifted features
        if drift_report.drifted_features:
            html_content += """
                <h2>Drifted Features</h2>
                <table>
                    <tr>
                        <th>Feature</th>
                        <th>Drift Score</th>
                        <th>P-Value</th>
                        <th>Severity</th>
                    </tr>
            """
            
            for feature in drift_report.drifted_features:
                score = drift_report.drift_scores.get(feature, 0)
                p_val = drift_report.p_values.get(feature, 1)
                severity = drift_report.severity_levels.get(feature, 'low')
                
                html_content += f"""
                    <tr>
                        <td>{feature}</td>
                        <td>{score:.4f}</td>
                        <td>{p_val:.4e}</td>
                        <td class="{severity}">{severity.upper()}</td>
                    </tr>
                """
            
            html_content += "</table>"
        
        # All features summary
        if drift_report.drift_scores:
            html_content += """
                <h2>All Features Summary</h2>
                <table>
                    <tr>
                        <th>Feature</th>
                        <th>Drift Score</th>
                        <th>P-Value</th>
                        <th>Severity</th>
                    </tr>
            """
            
            for feature, score in sorted(drift_report.drift_scores.items(),
                                        key=lambda x: x[1], reverse=True):
                p_val = drift_report.p_values.get(feature, 1)
                severity = drift_report.severity_levels.get(feature, 'low')
                
                html_content += f"""
                    <tr>
                        <td>{feature}</td>
                        <td>{score:.4f}</td>
                        <td>{p_val:.4e}</td>
                        <td class="{severity}">{severity.upper()}</td>
                    </tr>
                """
            
            html_content += "</table>"
        
        html_content += """
            </div>
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Drift monitoring report generated: {output_path}")
        return output_path
