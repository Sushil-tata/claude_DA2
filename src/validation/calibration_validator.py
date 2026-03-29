"""
Model Calibration Validation Framework

This module provides comprehensive calibration validation for probabilistic models,
including calibration curves, reliability diagrams, and statistical tests.

Usage Example:
    >>> from calibration_validator import CalibrationValidator
    >>> validator = CalibrationValidator(n_bins=10)
    >>> result = validator.validate_calibration(y_true, y_pred_proba)
    >>> validator.plot_calibration_curve(result, output_path='calibration.png')
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss
from scipy import stats
import warnings


@dataclass
class CalibrationResult:
    """Container for calibration validation results."""
    brier_score: float
    log_loss: float
    hosmer_lemeshow_statistic: float
    hosmer_lemeshow_pvalue: float
    expected_calibration_error: float
    maximum_calibration_error: float
    calibration_curve_data: Dict[str, np.ndarray] = field(default_factory=dict)
    is_well_calibrated: bool = True
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    segment_calibration: Dict[str, Dict[str, float]] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class CalibrationValidator:
    """
    Comprehensive model calibration validation framework.
    
    Validates whether predicted probabilities reflect true probabilities.
    A well-calibrated model predicts 70% probability for events that occur
    70% of the time.
    
    Attributes:
        n_bins: Number of bins for calibration curve (default: 10)
        hl_bins: Number of bins for Hosmer-Lemeshow test (default: 10)
        alpha: Significance level for statistical tests (default: 0.05)
        ece_threshold: Threshold for Expected Calibration Error (default: 0.1)
    
    Example:
        >>> validator = CalibrationValidator(n_bins=20)
        >>> result = validator.validate_calibration(y_true, y_pred_proba)
        >>> print(f"Brier Score: {result.brier_score:.4f}")
        >>> print(f"Is calibrated: {result.is_well_calibrated}")
    """
    
    def __init__(
        self,
        n_bins: int = 10,
        hl_bins: int = 10,
        alpha: float = 0.05,
        ece_threshold: float = 0.1,
        mce_threshold: float = 0.15
    ):
        self.n_bins = n_bins
        self.hl_bins = hl_bins
        self.alpha = alpha
        self.ece_threshold = ece_threshold
        self.mce_threshold = mce_threshold
        
        logger.info(f"Initialized CalibrationValidator with {n_bins} bins, "
                   f"ECE threshold: {ece_threshold}")
    
    def validate_calibration(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        sample_weight: Optional[np.ndarray] = None
    ) -> CalibrationResult:
        """
        Comprehensive calibration validation.
        
        Args:
            y_true: True binary labels (0/1)
            y_pred_proba: Predicted probabilities (0-1)
            sample_weight: Optional sample weights
        
        Returns:
            CalibrationResult with comprehensive calibration metrics
        """
        logger.info(f"Starting calibration validation on {len(y_true)} samples")
        
        y_true = np.asarray(y_true).flatten()
        y_pred_proba = np.asarray(y_pred_proba).flatten()
        
        # Validate inputs
        if len(y_true) != len(y_pred_proba):
            raise ValueError("y_true and y_pred_proba must have same length")
        
        if not np.all((y_true == 0) | (y_true == 1)):
            raise ValueError("y_true must be binary (0/1)")
        
        if not np.all((y_pred_proba >= 0) & (y_pred_proba <= 1)):
            raise ValueError("y_pred_proba must be in range [0, 1]")
        
        # Calculate Brier score
        brier = brier_score_loss(y_true, y_pred_proba, sample_weight=sample_weight)
        
        # Calculate log loss
        eps = 1e-15
        y_pred_clipped = np.clip(y_pred_proba, eps, 1 - eps)
        logloss = log_loss(y_true, y_pred_clipped, sample_weight=sample_weight)
        
        # Hosmer-Lemeshow test
        hl_stat, hl_pvalue = self._hosmer_lemeshow_test(y_true, y_pred_proba)
        
        # Calibration curve
        prob_true, prob_pred = calibration_curve(
            y_true, y_pred_proba, n_bins=self.n_bins, strategy='uniform'
        )
        
        # Expected and Maximum Calibration Error
        ece = self._calculate_ece(y_true, y_pred_proba)
        mce = self._calculate_mce(y_true, y_pred_proba)
        
        # Create result
        result = CalibrationResult(
            brier_score=float(brier),
            log_loss=float(logloss),
            hosmer_lemeshow_statistic=float(hl_stat),
            hosmer_lemeshow_pvalue=float(hl_pvalue),
            expected_calibration_error=float(ece),
            maximum_calibration_error=float(mce),
            calibration_curve_data={
                'prob_true': prob_true,
                'prob_pred': prob_pred
            }
        )
        
        # Determine if well-calibrated
        result.is_well_calibrated = self._assess_calibration(result)
        
        # Add warnings and recommendations
        self._add_warnings_and_recommendations(result)
        
        logger.info(f"Calibration validation complete. "
                   f"Brier={brier:.4f}, ECE={ece:.4f}, "
                   f"Well-calibrated={result.is_well_calibrated}")
        
        return result
    
    def _hosmer_lemeshow_test(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray
    ) -> Tuple[float, float]:
        """
        Perform Hosmer-Lemeshow goodness-of-fit test.
        
        Tests whether predicted probabilities match observed frequencies
        across bins of predicted probability.
        
        Returns:
            Tuple of (chi-square statistic, p-value)
        """
        # Create bins based on predicted probabilities
        bin_edges = np.linspace(0, 1, self.hl_bins + 1)
        bin_indices = np.digitize(y_pred_proba, bin_edges[1:-1])
        
        observed = []
        expected = []
        
        for i in range(self.hl_bins):
            mask = bin_indices == i
            
            if not np.any(mask):
                continue
            
            obs_events = np.sum(y_true[mask])
            exp_events = np.sum(y_pred_proba[mask])
            n_samples = np.sum(mask)
            
            observed.extend([obs_events, n_samples - obs_events])
            expected.extend([exp_events, n_samples - exp_events])
        
        observed = np.array(observed)
        expected = np.array(expected)
        
        # Avoid division by zero
        expected = np.maximum(expected, 1e-10)
        
        # Chi-square statistic
        chi_square = np.sum((observed - expected) ** 2 / expected)
        
        # Degrees of freedom: number of bins - 2
        dof = self.hl_bins - 2
        
        # P-value
        p_value = 1 - stats.chi2.cdf(chi_square, dof)
        
        return float(chi_square), float(p_value)
    
    def _calculate_ece(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray
    ) -> float:
        """
        Calculate Expected Calibration Error (ECE).
        
        ECE is the weighted average of calibration errors across bins.
        """
        bin_edges = np.linspace(0, 1, self.n_bins + 1)
        bin_indices = np.digitize(y_pred_proba, bin_edges[1:-1])
        
        ece = 0.0
        n_total = len(y_true)
        
        for i in range(self.n_bins):
            mask = bin_indices == i
            
            if not np.any(mask):
                continue
            
            bin_accuracy = np.mean(y_true[mask])
            bin_confidence = np.mean(y_pred_proba[mask])
            bin_size = np.sum(mask)
            
            ece += (bin_size / n_total) * np.abs(bin_accuracy - bin_confidence)
        
        return float(ece)
    
    def _calculate_mce(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray
    ) -> float:
        """
        Calculate Maximum Calibration Error (MCE).
        
        MCE is the maximum calibration error across all bins.
        """
        bin_edges = np.linspace(0, 1, self.n_bins + 1)
        bin_indices = np.digitize(y_pred_proba, bin_edges[1:-1])
        
        max_error = 0.0
        
        for i in range(self.n_bins):
            mask = bin_indices == i
            
            if not np.any(mask):
                continue
            
            bin_accuracy = np.mean(y_true[mask])
            bin_confidence = np.mean(y_pred_proba[mask])
            
            error = np.abs(bin_accuracy - bin_confidence)
            max_error = max(max_error, error)
        
        return float(max_error)
    
    def _assess_calibration(self, result: CalibrationResult) -> bool:
        """Determine if model is well-calibrated."""
        # Multiple criteria for good calibration
        criteria = [
            result.hosmer_lemeshow_pvalue > self.alpha,  # Pass HL test
            result.expected_calibration_error < self.ece_threshold,  # Low ECE
            result.maximum_calibration_error < self.mce_threshold  # Low MCE
        ]
        
        # Model is well-calibrated if it passes majority of criteria
        return sum(criteria) >= 2
    
    def _add_warnings_and_recommendations(self, result: CalibrationResult) -> None:
        """Add warnings and recalibration recommendations."""
        if not result.is_well_calibrated:
            result.warnings.append(
                "Model is poorly calibrated. Predicted probabilities do not "
                "reflect true probabilities."
            )
            result.recommendations.extend([
                "Apply Platt scaling (logistic calibration)",
                "Use isotonic regression for non-parametric calibration",
                "Consider temperature scaling for neural networks",
                "Collect more calibration data",
                "Review model training process for probability estimates"
            ])
        
        if result.hosmer_lemeshow_pvalue < self.alpha:
            result.warnings.append(
                f"Hosmer-Lemeshow test failed (p={result.hosmer_lemeshow_pvalue:.4f}). "
                "Significant deviation from perfect calibration."
            )
        
        if result.expected_calibration_error > self.ece_threshold:
            result.warnings.append(
                f"High Expected Calibration Error ({result.expected_calibration_error:.4f}). "
                "Average calibration across bins is poor."
            )
        
        if result.maximum_calibration_error > self.mce_threshold:
            result.warnings.append(
                f"High Maximum Calibration Error ({result.maximum_calibration_error:.4f}). "
                "Some probability bins are severely miscalibrated."
            )
        
        # Brier score interpretation
        if result.brier_score < 0.1:
            result.recommendations.append("Brier score is excellent (<0.1)")
        elif result.brier_score < 0.25:
            result.recommendations.append("Brier score is acceptable (0.1-0.25)")
        else:
            result.warnings.append(
                f"High Brier score ({result.brier_score:.4f}). "
                "Model predictions have poor accuracy and calibration."
            )
    
    def validate_by_segment(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        segments: pd.Series,
        segment_name: str = 'segment'
    ) -> Dict[str, CalibrationResult]:
        """
        Validate calibration across different segments.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            segments: Series with segment identifiers
            segment_name: Name for the segment
        
        Returns:
            Dictionary mapping segment values to CalibrationResults
        """
        logger.info(f"Validating calibration by {segment_name}")
        
        segment_results = {}
        
        for segment_value in segments.unique():
            mask = segments == segment_value
            
            if np.sum(mask) < 50:  # Minimum samples for reliable calibration
                logger.warning(f"Segment {segment_value} has insufficient samples")
                continue
            
            try:
                result = self.validate_calibration(
                    y_true[mask],
                    y_pred_proba[mask]
                )
                segment_results[str(segment_value)] = result
            
            except Exception as e:
                logger.error(f"Error validating segment {segment_value}: {e}")
        
        logger.info(f"Validated calibration for {len(segment_results)} segments")
        return segment_results
    
    def plot_calibration_curve(
        self,
        result: CalibrationResult,
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6)
    ) -> None:
        """
        Plot calibration curve (reliability diagram).
        
        Args:
            result: CalibrationResult from validate_calibration
            output_path: Path to save plot
            figsize: Figure size
        """
        prob_true = result.calibration_curve_data['prob_true']
        prob_pred = result.calibration_curve_data['prob_pred']
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration')
        
        # Plot actual calibration curve
        ax.plot(prob_pred, prob_true, 's-', linewidth=2, markersize=8,
               label=f'Model (ECE={result.expected_calibration_error:.3f})')
        
        # Shade gap between perfect and actual
        ax.fill_between(prob_pred, prob_true, prob_pred, alpha=0.2,
                       color='red' if not result.is_well_calibrated else 'green')
        
        ax.set_xlabel('Mean Predicted Probability', fontsize=12)
        ax.set_ylabel('Fraction of Positives', fontsize=12)
        ax.set_title('Calibration Curve (Reliability Diagram)',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        # Add text box with metrics
        textstr = f'Brier Score: {result.brier_score:.4f}\n'
        textstr += f'ECE: {result.expected_calibration_error:.4f}\n'
        textstr += f'MCE: {result.maximum_calibration_error:.4f}\n'
        textstr += f'HL p-value: {result.hosmer_lemeshow_pvalue:.4f}'
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Calibration curve saved to {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_calibration_by_decile(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 6)
    ) -> None:
        """
        Plot calibration metrics by decile of predicted probability.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            output_path: Path to save plot
            figsize: Figure size
        """
        # Create deciles
        deciles = pd.qcut(y_pred_proba, q=10, labels=False, duplicates='drop')
        
        decile_stats = []
        
        for decile in np.unique(deciles):
            mask = deciles == decile
            
            decile_stats.append({
                'decile': decile + 1,
                'mean_predicted': np.mean(y_pred_proba[mask]),
                'observed_rate': np.mean(y_true[mask]),
                'count': np.sum(mask)
            })
        
        df = pd.DataFrame(decile_stats)
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: Predicted vs Observed
        x = np.arange(len(df))
        width = 0.35
        
        axes[0].bar(x - width/2, df['mean_predicted'], width,
                   label='Mean Predicted', alpha=0.7, color='blue')
        axes[0].bar(x + width/2, df['observed_rate'], width,
                   label='Observed Rate', alpha=0.7, color='red')
        
        axes[0].set_xlabel('Decile', fontsize=11)
        axes[0].set_ylabel('Probability', fontsize=11)
        axes[0].set_title('Predicted vs Observed by Decile',
                         fontsize=12, fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(df['decile'])
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Sample counts
        axes[1].bar(x, df['count'], alpha=0.7, color='green')
        axes[1].set_xlabel('Decile', fontsize=11)
        axes[1].set_ylabel('Sample Count', fontsize=11)
        axes[1].set_title('Sample Distribution by Decile',
                         fontsize=12, fontweight='bold')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(df['decile'])
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Decile calibration plot saved to {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def calculate_conformal_intervals(
        self,
        y_calib_true: np.ndarray,
        y_calib_pred: np.ndarray,
        y_test_pred: np.ndarray,
        alpha: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate conformal prediction intervals.
        
        Provides distribution-free coverage guarantees for predictions.
        
        Args:
            y_calib_true: Calibration set true labels
            y_calib_pred: Calibration set predictions
            y_test_pred: Test set predictions
            alpha: Desired error rate (1-alpha coverage)
        
        Returns:
            Tuple of (lower_bounds, upper_bounds) for test predictions
        """
        # Calculate calibration scores (absolute errors)
        calib_scores = np.abs(y_calib_true - y_calib_pred)
        
        # Find quantile
        n = len(calib_scores)
        q = np.ceil((n + 1) * (1 - alpha)) / n
        quantile = np.quantile(calib_scores, q)
        
        # Create prediction intervals
        lower_bounds = y_test_pred - quantile
        upper_bounds = y_test_pred + quantile
        
        logger.info(f"Calculated conformal intervals with {(1-alpha)*100:.1f}% coverage, "
                   f"width={quantile*2:.4f}")
        
        return lower_bounds, upper_bounds
    
    def generate_report(
        self,
        result: CalibrationResult,
        output_path: str = 'calibration_report.html'
    ) -> str:
        """
        Generate comprehensive HTML calibration report.
        
        Args:
            result: CalibrationResult from validate_calibration
            output_path: Output file path
        
        Returns:
            Path to generated report
        """
        status_color = '#2ecc71' if result.is_well_calibrated else '#e74c3c'
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Calibration Validation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
                .container {{ background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
                h2 {{ color: #34495e; margin-top: 30px; }}
                .metric {{ background-color: #ecf0f1; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                .metric-label {{ font-weight: bold; color: #7f8c8d; }}
                .metric-value {{ font-size: 20px; font-weight: bold; }}
                .warning {{ background-color: #fff3cd; border-left: 4px solid #f39c12; padding: 10px; margin: 10px 0; }}
                .alert {{ background-color: #f8d7da; border-left: 4px solid #e74c3c; padding: 10px; margin: 10px 0; }}
                .success {{ background-color: #d4edda; border-left: 4px solid #2ecc71; padding: 10px; margin: 10px 0; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #3498db; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .timestamp {{ color: #95a5a6; font-size: 14px; }}
                ul {{ line-height: 1.8; }}
                .status-badge {{ display: inline-block; padding: 5px 15px; border-radius: 5px; 
                               background-color: {status_color}; color: white; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Model Calibration Validation Report</h1>
                <p class="timestamp">Generated: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <div class="metric">
                    <div class="metric-label">Calibration Status</div>
                    <div class="metric-value">
                        <span class="status-badge">
                            {"WELL CALIBRATED" if result.is_well_calibrated else "POORLY CALIBRATED"}
                        </span>
                    </div>
                </div>
                
                <h2>Calibration Metrics</h2>
                
                <div class="metric">
                    <div class="metric-label">Brier Score</div>
                    <div class="metric-value">{result.brier_score:.4f}</div>
                    <div style="font-size: 12px; color: #7f8c8d; margin-top: 5px;">
                        Lower is better. Range: [0, 1]. Good: &lt; 0.1, Acceptable: 0.1-0.25
                    </div>
                </div>
                
                <div class="metric">
                    <div class="metric-label">Log Loss</div>
                    <div class="metric-value">{result.log_loss:.4f}</div>
                    <div style="font-size: 12px; color: #7f8c8d; margin-top: 5px;">
                        Lower is better. Penalizes confident wrong predictions heavily.
                    </div>
                </div>
                
                <div class="metric">
                    <div class="metric-label">Expected Calibration Error (ECE)</div>
                    <div class="metric-value">{result.expected_calibration_error:.4f}</div>
                    <div style="font-size: 12px; color: #7f8c8d; margin-top: 5px;">
                        Average calibration error across bins. Good: &lt; 0.1
                    </div>
                </div>
                
                <div class="metric">
                    <div class="metric-label">Maximum Calibration Error (MCE)</div>
                    <div class="metric-value">{result.maximum_calibration_error:.4f}</div>
                    <div style="font-size: 12px; color: #7f8c8d; margin-top: 5px;">
                        Worst-case calibration error. Good: &lt; 0.15
                    </div>
                </div>
                
                <div class="metric">
                    <div class="metric-label">Hosmer-Lemeshow Test</div>
                    <div class="metric-value">
                        χ² = {result.hosmer_lemeshow_statistic:.4f}, 
                        p = {result.hosmer_lemeshow_pvalue:.4f}
                    </div>
                    <div style="font-size: 12px; color: #7f8c8d; margin-top: 5px;">
                        Tests goodness of fit. p &gt; 0.05 indicates good calibration.
                    </div>
                </div>
        """
        
        # Warnings
        if result.warnings:
            html_content += "<h2>Findings</h2>"
            for warning in result.warnings:
                css_class = 'alert' if 'High' in warning or 'failed' in warning else 'warning'
                html_content += f'<div class="{css_class}">{warning}</div>'
        else:
            html_content += '<div class="success">Model is well-calibrated. No issues detected.</div>'
        
        # Recommendations
        if result.recommendations:
            html_content += "<h2>Recommendations</h2><ul>"
            for rec in result.recommendations:
                html_content += f"<li>{rec}</li>"
            html_content += "</ul>"
        
        # Interpretation guide
        html_content += """
                <h2>Calibration Metrics Guide</h2>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Description</th>
                        <th>Good Range</th>
                    </tr>
                    <tr>
                        <td><strong>Brier Score</strong></td>
                        <td>Mean squared error of probability predictions</td>
                        <td>&lt; 0.1 (excellent), 0.1-0.25 (acceptable)</td>
                    </tr>
                    <tr>
                        <td><strong>ECE</strong></td>
                        <td>Weighted average calibration error across bins</td>
                        <td>&lt; 0.1</td>
                    </tr>
                    <tr>
                        <td><strong>MCE</strong></td>
                        <td>Maximum calibration error in any bin</td>
                        <td>&lt; 0.15</td>
                    </tr>
                    <tr>
                        <td><strong>Hosmer-Lemeshow</strong></td>
                        <td>Chi-square goodness of fit test</td>
                        <td>p-value &gt; 0.05</td>
                    </tr>
                </table>
                
                <h2>Recalibration Techniques</h2>
                <ul>
                    <li><strong>Platt Scaling:</strong> Fit logistic regression on predictions (parametric)</li>
                    <li><strong>Isotonic Regression:</strong> Non-parametric monotonic calibration</li>
                    <li><strong>Temperature Scaling:</strong> Single parameter scaling for neural networks</li>
                    <li><strong>Beta Calibration:</strong> Generalization of Platt scaling</li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Calibration report generated: {output_path}")
        return output_path


def quick_calibration_check(y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
    """
    Quick calibration check with key metrics.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
    
    Returns:
        Dictionary with calibration metrics
    """
    validator = CalibrationValidator()
    result = validator.validate_calibration(y_true, y_pred_proba)
    
    return {
        'brier_score': result.brier_score,
        'ece': result.expected_calibration_error,
        'mce': result.maximum_calibration_error,
        'is_well_calibrated': result.is_well_calibrated
    }
