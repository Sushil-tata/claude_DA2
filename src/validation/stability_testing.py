"""
Model Stability Testing Framework

This module provides comprehensive stability analysis for machine learning models,
including PSI (Population Stability Index), CSI (Characteristic Stability Index),
and sub-segment stability analysis.

Usage Example:
    >>> from stability_testing import StabilityTester
    >>> tester = StabilityTester(n_bins=10, psi_threshold_yellow=0.1, psi_threshold_red=0.25)
    >>> stability_report = tester.analyze_stability(
    ...     baseline_scores, production_scores,
    ...     baseline_features, production_features,
    ...     segments={'age': baseline_age, 'region': baseline_region}
    ... )
    >>> tester.generate_report(stability_report, output_path='stability_report.html')
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime
import warnings
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


@dataclass
class StabilityResult:
    """Container for stability analysis results."""
    psi_value: float
    csi_values: Dict[str, float] = field(default_factory=dict)
    segment_psi: Dict[str, float] = field(default_factory=dict)
    traffic_light: str = "green"  # green, yellow, red
    timestamp: datetime = field(default_factory=datetime.now)
    warnings: List[str] = field(default_factory=list)
    segment_sizes: Dict[str, int] = field(default_factory=dict)
    drift_features: List[str] = field(default_factory=list)


class StabilityTester:
    """
    Comprehensive model stability testing framework.
    
    Implements PSI, CSI, and segment-level stability analysis with
    traffic light warnings for model monitoring.
    
    Attributes:
        n_bins: Number of bins for score discretization (default: 10)
        psi_threshold_yellow: PSI threshold for yellow warning (default: 0.1)
        psi_threshold_red: PSI threshold for red alert (default: 0.25)
        min_segment_size: Minimum required segment size (default: 1000)
        csi_threshold: CSI threshold for feature drift warning (default: 0.15)
    
    Example:
        >>> tester = StabilityTester(n_bins=20, psi_threshold_yellow=0.05)
        >>> result = tester.calculate_psi(baseline_scores, production_scores)
        >>> print(f"PSI: {result:.4f} - Status: {tester.get_traffic_light(result)}")
    """
    
    def __init__(
        self,
        n_bins: int = 10,
        psi_threshold_yellow: float = 0.1,
        psi_threshold_red: float = 0.25,
        min_segment_size: int = 1000,
        csi_threshold: float = 0.15
    ):
        self.n_bins = n_bins
        self.psi_threshold_yellow = psi_threshold_yellow
        self.psi_threshold_red = psi_threshold_red
        self.min_segment_size = min_segment_size
        self.csi_threshold = csi_threshold
        
        logger.info(f"Initialized StabilityTester with {n_bins} bins, "
                   f"PSI thresholds: {psi_threshold_yellow}/{psi_threshold_red}")
    
    def calculate_psi(
        self,
        baseline: np.ndarray,
        production: np.ndarray,
        bins: Optional[np.ndarray] = None
    ) -> float:
        """
        Calculate Population Stability Index (PSI).
        
        PSI = Σ (% production - % baseline) × ln(% production / % baseline)
        
        Args:
            baseline: Baseline/training scores or predictions
            production: Production/test scores or predictions
            bins: Custom bin edges (if None, uses quantiles from baseline)
        
        Returns:
            PSI value (float)
        
        Interpretation:
            PSI < 0.1: No significant population shift
            0.1 ≤ PSI < 0.25: Moderate shift, investigation needed
            PSI ≥ 0.25: Significant shift, model may need retraining
        """
        baseline = np.asarray(baseline).flatten()
        production = np.asarray(production).flatten()
        
        if len(baseline) == 0 or len(production) == 0:
            raise ValueError("Baseline and production arrays cannot be empty")
        
        # Create bins based on baseline quantiles if not provided
        if bins is None:
            bins = np.percentile(baseline, np.linspace(0, 100, self.n_bins + 1))
            bins = np.unique(bins)  # Remove duplicate edges
        
        # Ensure bins cover full range
        bins[0] = min(bins[0], baseline.min(), production.min()) - 1e-6
        bins[-1] = max(bins[-1], baseline.max(), production.max()) + 1e-6
        
        # Calculate distributions
        baseline_counts, _ = np.histogram(baseline, bins=bins)
        production_counts, _ = np.histogram(production, bins=bins)
        
        # Convert to proportions with small epsilon to avoid log(0)
        epsilon = 1e-6
        baseline_prop = (baseline_counts + epsilon) / (len(baseline) + epsilon * len(bins))
        production_prop = (production_counts + epsilon) / (len(production) + epsilon * len(bins))
        
        # Calculate PSI
        psi = np.sum((production_prop - baseline_prop) * np.log(production_prop / baseline_prop))
        
        logger.debug(f"Calculated PSI: {psi:.4f}")
        return float(psi)
    
    def calculate_csi(
        self,
        baseline_features: pd.DataFrame,
        production_features: pd.DataFrame,
        categorical_features: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Calculate Characteristic Stability Index (CSI) for each feature.
        
        CSI measures distribution shifts for individual features,
        similar to PSI but applied to input features.
        
        Args:
            baseline_features: Baseline feature dataset
            production_features: Production feature dataset
            categorical_features: List of categorical feature names
        
        Returns:
            Dictionary mapping feature names to CSI values
        """
        if categorical_features is None:
            categorical_features = []
        
        csi_dict = {}
        
        for col in baseline_features.columns:
            if col not in production_features.columns:
                logger.warning(f"Feature '{col}' not found in production data")
                continue
            
            try:
                baseline_col = baseline_features[col].dropna()
                production_col = production_features[col].dropna()
                
                if len(baseline_col) == 0 or len(production_col) == 0:
                    logger.warning(f"Feature '{col}' has no valid values")
                    continue
                
                if col in categorical_features or baseline_features[col].dtype == 'object':
                    # For categorical features, use value counts
                    csi_dict[col] = self._calculate_categorical_csi(
                        baseline_col, production_col
                    )
                else:
                    # For numerical features, use binning
                    csi_dict[col] = self.calculate_psi(
                        baseline_col.values, production_col.values
                    )
            
            except Exception as e:
                logger.error(f"Error calculating CSI for feature '{col}': {e}")
                continue
        
        logger.info(f"Calculated CSI for {len(csi_dict)} features")
        return csi_dict
    
    def _calculate_categorical_csi(
        self,
        baseline: pd.Series,
        production: pd.Series
    ) -> float:
        """Calculate CSI for categorical features."""
        # Get all unique categories
        all_categories = set(baseline.unique()) | set(production.unique())
        
        epsilon = 1e-6
        baseline_dist = baseline.value_counts(normalize=True)
        production_dist = production.value_counts(normalize=True)
        
        csi = 0.0
        for cat in all_categories:
            baseline_prop = baseline_dist.get(cat, 0) + epsilon
            production_prop = production_dist.get(cat, 0) + epsilon
            
            csi += (production_prop - baseline_prop) * np.log(production_prop / baseline_prop)
        
        return float(csi)
    
    def analyze_segment_stability(
        self,
        baseline_scores: np.ndarray,
        production_scores: np.ndarray,
        baseline_segments: pd.DataFrame,
        production_segments: pd.DataFrame,
        segment_columns: List[str]
    ) -> Dict[str, Any]:
        """
        Analyze stability across different sub-segments.
        
        Args:
            baseline_scores: Baseline prediction scores
            production_scores: Production prediction scores
            baseline_segments: DataFrame with segment identifiers for baseline
            production_segments: DataFrame with segment identifiers for production
            segment_columns: List of column names to segment by
        
        Returns:
            Dictionary with segment-level PSI values and warnings
        """
        segment_results = {}
        warnings_list = []
        
        for segment_col in segment_columns:
            if segment_col not in baseline_segments.columns:
                logger.warning(f"Segment column '{segment_col}' not in baseline data")
                continue
            
            if segment_col not in production_segments.columns:
                logger.warning(f"Segment column '{segment_col}' not in production data")
                continue
            
            segment_psi = {}
            segment_sizes = {}
            
            # Get unique segment values
            unique_values = set(baseline_segments[segment_col].unique()) | \
                          set(production_segments[segment_col].unique())
            
            for segment_value in unique_values:
                baseline_mask = baseline_segments[segment_col] == segment_value
                production_mask = production_segments[segment_col] == segment_value
                
                baseline_segment_scores = baseline_scores[baseline_mask]
                production_segment_scores = production_scores[production_mask]
                
                segment_size_baseline = len(baseline_segment_scores)
                segment_size_production = len(production_segment_scores)
                
                # Check minimum segment size
                if segment_size_baseline < self.min_segment_size:
                    warnings_list.append(
                        f"Segment {segment_col}={segment_value} baseline size "
                        f"({segment_size_baseline}) below minimum ({self.min_segment_size})"
                    )
                
                if segment_size_production < self.min_segment_size:
                    warnings_list.append(
                        f"Segment {segment_col}={segment_value} production size "
                        f"({segment_size_production}) below minimum ({self.min_segment_size})"
                    )
                
                if len(baseline_segment_scores) > 0 and len(production_segment_scores) > 0:
                    psi = self.calculate_psi(baseline_segment_scores, production_segment_scores)
                    segment_psi[str(segment_value)] = psi
                    segment_sizes[str(segment_value)] = {
                        'baseline': segment_size_baseline,
                        'production': segment_size_production
                    }
            
            segment_results[segment_col] = {
                'psi': segment_psi,
                'sizes': segment_sizes
            }
        
        return {
            'segment_results': segment_results,
            'warnings': warnings_list
        }
    
    def get_traffic_light(self, psi_value: float) -> str:
        """
        Determine traffic light status based on PSI value.
        
        Args:
            psi_value: PSI value to evaluate
        
        Returns:
            'green', 'yellow', or 'red'
        """
        if psi_value < self.psi_threshold_yellow:
            return 'green'
        elif psi_value < self.psi_threshold_red:
            return 'yellow'
        else:
            return 'red'
    
    def analyze_stability(
        self,
        baseline_scores: np.ndarray,
        production_scores: np.ndarray,
        baseline_features: Optional[pd.DataFrame] = None,
        production_features: Optional[pd.DataFrame] = None,
        segments: Optional[Dict[str, pd.DataFrame]] = None,
        categorical_features: Optional[List[str]] = None
    ) -> StabilityResult:
        """
        Comprehensive stability analysis.
        
        Args:
            baseline_scores: Baseline model scores
            production_scores: Production model scores
            baseline_features: Baseline features (optional)
            production_features: Production features (optional)
            segments: Dictionary of segment DataFrames (optional)
            categorical_features: List of categorical feature names (optional)
        
        Returns:
            StabilityResult object with comprehensive analysis
        """
        logger.info("Starting comprehensive stability analysis")
        
        # Calculate overall PSI
        psi = self.calculate_psi(baseline_scores, production_scores)
        traffic_light = self.get_traffic_light(psi)
        
        result = StabilityResult(
            psi_value=psi,
            traffic_light=traffic_light
        )
        
        # Calculate CSI if features provided
        if baseline_features is not None and production_features is not None:
            csi_values = self.calculate_csi(
                baseline_features, production_features, categorical_features
            )
            result.csi_values = csi_values
            
            # Identify drifting features
            drift_features = [
                feat for feat, csi in csi_values.items()
                if csi > self.csi_threshold
            ]
            result.drift_features = drift_features
            
            if drift_features:
                result.warnings.append(
                    f"{len(drift_features)} features showing significant drift: "
                    f"{', '.join(drift_features[:5])}"
                )
        
        # Segment analysis if provided
        if segments is not None:
            baseline_segments = segments.get('baseline')
            production_segments = segments.get('production')
            
            if baseline_segments is not None and production_segments is not None:
                segment_analysis = self.analyze_segment_stability(
                    baseline_scores, production_scores,
                    baseline_segments, production_segments,
                    list(baseline_segments.columns)
                )
                
                result.warnings.extend(segment_analysis['warnings'])
                
                # Flatten segment PSI results
                for seg_col, seg_data in segment_analysis['segment_results'].items():
                    for seg_val, seg_psi in seg_data['psi'].items():
                        key = f"{seg_col}_{seg_val}"
                        result.segment_psi[key] = seg_psi
                        result.segment_sizes[key] = seg_data['sizes'][seg_val]
        
        # Overall warnings
        if traffic_light == 'yellow':
            result.warnings.append(
                f"Moderate population shift detected (PSI={psi:.4f}). "
                "Investigation recommended."
            )
        elif traffic_light == 'red':
            result.warnings.append(
                f"Significant population shift detected (PSI={psi:.4f}). "
                "Model retraining strongly recommended."
            )
        
        logger.info(f"Stability analysis complete. PSI={psi:.4f}, Status={traffic_light}")
        return result
    
    def track_stability_over_time(
        self,
        baseline_scores: np.ndarray,
        time_series_scores: Dict[str, np.ndarray]
    ) -> pd.DataFrame:
        """
        Track stability metrics over time.
        
        Args:
            baseline_scores: Reference baseline scores
            time_series_scores: Dictionary mapping timestamps to score arrays
        
        Returns:
            DataFrame with PSI values over time
        """
        stability_timeline = []
        
        for timestamp, scores in sorted(time_series_scores.items()):
            psi = self.calculate_psi(baseline_scores, scores)
            traffic_light = self.get_traffic_light(psi)
            
            stability_timeline.append({
                'timestamp': timestamp,
                'psi': psi,
                'traffic_light': traffic_light,
                'sample_size': len(scores)
            })
        
        df = pd.DataFrame(stability_timeline)
        logger.info(f"Tracked stability across {len(df)} time periods")
        
        return df
    
    def visualize_psi_trends(
        self,
        stability_timeline: pd.DataFrame,
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 6)
    ) -> None:
        """
        Visualize PSI trends over time.
        
        Args:
            stability_timeline: DataFrame from track_stability_over_time
            output_path: Path to save plot (if None, displays plot)
            figsize: Figure size tuple
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot PSI over time
        ax.plot(stability_timeline['timestamp'], stability_timeline['psi'],
                marker='o', linewidth=2, markersize=6)
        
        # Add threshold lines
        ax.axhline(y=self.psi_threshold_yellow, color='orange', linestyle='--',
                  label=f'Yellow threshold ({self.psi_threshold_yellow})')
        ax.axhline(y=self.psi_threshold_red, color='red', linestyle='--',
                  label=f'Red threshold ({self.psi_threshold_red})')
        
        # Color background based on traffic light zones
        ax.axhspan(0, self.psi_threshold_yellow, alpha=0.1, color='green')
        ax.axhspan(self.psi_threshold_yellow, self.psi_threshold_red,
                  alpha=0.1, color='yellow')
        ax.axhspan(self.psi_threshold_red, ax.get_ylim()[1], alpha=0.1, color='red')
        
        ax.set_xlabel('Time Period', fontsize=12)
        ax.set_ylabel('PSI Value', fontsize=12)
        ax.set_title('Population Stability Index Over Time', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"PSI trend plot saved to {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def visualize_segment_comparison(
        self,
        segment_psi: Dict[str, float],
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6)
    ) -> None:
        """
        Visualize PSI across different segments.
        
        Args:
            segment_psi: Dictionary of segment PSI values
            output_path: Path to save plot
            figsize: Figure size tuple
        """
        if not segment_psi:
            logger.warning("No segment PSI data to visualize")
            return
        
        segments = list(segment_psi.keys())
        psi_values = list(segment_psi.values())
        colors = [self._get_traffic_light_color(psi) for psi in psi_values]
        
        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.barh(segments, psi_values, color=colors, alpha=0.7, edgecolor='black')
        
        # Add threshold lines
        ax.axvline(x=self.psi_threshold_yellow, color='orange', linestyle='--',
                  label=f'Yellow threshold')
        ax.axvline(x=self.psi_threshold_red, color='red', linestyle='--',
                  label=f'Red threshold')
        
        ax.set_xlabel('PSI Value', fontsize=12)
        ax.set_ylabel('Segment', fontsize=12)
        ax.set_title('PSI by Segment', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Segment comparison plot saved to {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def _get_traffic_light_color(self, psi: float) -> str:
        """Get color for traffic light status."""
        traffic_light = self.get_traffic_light(psi)
        color_map = {
            'green': '#2ecc71',
            'yellow': '#f39c12',
            'red': '#e74c3c'
        }
        return color_map.get(traffic_light, 'gray')
    
    def generate_report(
        self,
        stability_result: StabilityResult,
        output_path: str = 'stability_report.html'
    ) -> str:
        """
        Generate comprehensive HTML stability report.
        
        Args:
            stability_result: StabilityResult from analyze_stability
            output_path: Output file path
        
        Returns:
            Path to generated report
        """
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Stability Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
                .container {{ background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
                h2 {{ color: #34495e; margin-top: 30px; }}
                .metric {{ background-color: #ecf0f1; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                .metric-label {{ font-weight: bold; color: #7f8c8d; }}
                .metric-value {{ font-size: 24px; font-weight: bold; }}
                .green {{ color: #2ecc71; }}
                .yellow {{ color: #f39c12; }}
                .red {{ color: #e74c3c; }}
                .warning {{ background-color: #fff3cd; border-left: 4px solid #f39c12; padding: 10px; margin: 10px 0; }}
                .alert {{ background-color: #f8d7da; border-left: 4px solid #e74c3c; padding: 10px; margin: 10px 0; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #3498db; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .timestamp {{ color: #95a5a6; font-size: 14px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Model Stability Report</h1>
                <p class="timestamp">Generated: {stability_result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <div class="metric">
                    <div class="metric-label">Overall Population Stability Index (PSI)</div>
                    <div class="metric-value {stability_result.traffic_light}">
                        {stability_result.psi_value:.4f}
                    </div>
                    <div>Status: <strong class="{stability_result.traffic_light}">
                        {stability_result.traffic_light.upper()}
                    </strong></div>
                </div>
                
                <h2>Interpretation Guidelines</h2>
                <ul>
                    <li><strong class="green">PSI &lt; 0.1:</strong> No significant population shift</li>
                    <li><strong class="yellow">0.1 ≤ PSI &lt; 0.25:</strong> Moderate shift, investigation needed</li>
                    <li><strong class="red">PSI ≥ 0.25:</strong> Significant shift, model retraining recommended</li>
                </ul>
        """
        
        # Add warnings
        if stability_result.warnings:
            html_content += "<h2>Warnings and Recommendations</h2>"
            for warning in stability_result.warnings:
                severity = 'alert' if 'Significant' in warning or 'red' in warning.lower() else 'warning'
                html_content += f'<div class="{severity}">{warning}</div>'
        
        # Add CSI results
        if stability_result.csi_values:
            html_content += """
                <h2>Feature Stability (CSI)</h2>
                <table>
                    <tr>
                        <th>Feature</th>
                        <th>CSI Value</th>
                        <th>Status</th>
                    </tr>
            """
            
            for feature, csi in sorted(stability_result.csi_values.items(),
                                      key=lambda x: x[1], reverse=True):
                traffic_light = self.get_traffic_light(csi)
                html_content += f"""
                    <tr>
                        <td>{feature}</td>
                        <td>{csi:.4f}</td>
                        <td class="{traffic_light}">{traffic_light.upper()}</td>
                    </tr>
                """
            
            html_content += "</table>"
        
        # Add segment results
        if stability_result.segment_psi:
            html_content += """
                <h2>Segment-Level Stability</h2>
                <table>
                    <tr>
                        <th>Segment</th>
                        <th>PSI Value</th>
                        <th>Baseline Size</th>
                        <th>Production Size</th>
                        <th>Status</th>
                    </tr>
            """
            
            for segment, psi in sorted(stability_result.segment_psi.items(),
                                      key=lambda x: x[1], reverse=True):
                traffic_light = self.get_traffic_light(psi)
                sizes = stability_result.segment_sizes.get(segment, {'baseline': 0, 'production': 0})
                html_content += f"""
                    <tr>
                        <td>{segment}</td>
                        <td>{psi:.4f}</td>
                        <td>{sizes.get('baseline', 'N/A')}</td>
                        <td>{sizes.get('production', 'N/A')}</td>
                        <td class="{traffic_light}">{traffic_light.upper()}</td>
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
        
        logger.info(f"Stability report generated: {output_path}")
        return output_path


def calculate_psi_quick(baseline: np.ndarray, production: np.ndarray, n_bins: int = 10) -> float:
    """
    Quick PSI calculation utility function.
    
    Args:
        baseline: Baseline scores
        production: Production scores
        n_bins: Number of bins
    
    Returns:
        PSI value
    """
    tester = StabilityTester(n_bins=n_bins)
    return tester.calculate_psi(baseline, production)
