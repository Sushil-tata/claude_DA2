"""
Adversarial Validation Framework

This module implements adversarial validation to detect distribution differences
between training and test datasets, data leakage, and temporal drift.

Usage Example:
    >>> from adversarial_validation import AdversarialValidator
    >>> validator = AdversarialValidator(n_estimators=100, auc_threshold=0.85)
    >>> result = validator.validate(train_df, test_df, target_col='is_test')
    >>> validator.generate_report(result, 'adversarial_report.html')
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder
import warnings


@dataclass
class AdversarialResult:
    """Container for adversarial validation results."""
    auc_score: float
    feature_importance: Dict[str, float] = field(default_factory=dict)
    status: str = "pass"  # pass, warning, fail
    warnings: List[str] = field(default_factory=list)
    cv_scores: List[float] = field(default_factory=list)
    predictions: np.ndarray = field(default_factory=lambda: np.array([]))
    timestamp: datetime = field(default_factory=datetime.now)
    mitigation_recommendations: List[str] = field(default_factory=list)
    drift_features: List[str] = field(default_factory=list)


class AdversarialValidator:
    """
    Adversarial validation for detecting train-test distribution differences.
    
    Uses a classifier to distinguish between train and test sets. If the classifier
    achieves high AUC (>0.85), it indicates significant distribution differences,
    potential data leakage, or temporal drift.
    
    Attributes:
        n_estimators: Number of trees in Random Forest (default: 100)
        auc_threshold: AUC threshold for warning (default: 0.85)
        auc_critical: AUC threshold for critical alert (default: 0.95)
        cv_folds: Number of cross-validation folds (default: 5)
        random_state: Random seed for reproducibility (default: 42)
    
    Example:
        >>> validator = AdversarialValidator(auc_threshold=0.80)
        >>> result = validator.validate(X_train, X_test)
        >>> if result.status == 'fail':
        ...     print(f"Critical issue! AUC={result.auc_score:.3f}")
        ...     print(f"Top drift features: {result.drift_features[:5]}")
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        auc_threshold: float = 0.85,
        auc_critical: float = 0.95,
        cv_folds: int = 5,
        random_state: int = 42
    ):
        self.n_estimators = n_estimators
        self.auc_threshold = auc_threshold
        self.auc_critical = auc_critical
        self.cv_folds = cv_folds
        self.random_state = random_state
        
        logger.info(f"Initialized AdversarialValidator with AUC thresholds: "
                   f"{auc_threshold} (warning), {auc_critical} (critical)")
    
    def validate(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None
    ) -> AdversarialResult:
        """
        Perform adversarial validation.
        
        Creates a binary classification problem where:
        - Label 0: training data
        - Label 1: test data
        
        High classification performance indicates distribution shift.
        
        Args:
            train_data: Training dataset
            test_data: Test dataset
            features: List of features to use (if None, uses all columns)
            categorical_features: List of categorical feature names
        
        Returns:
            AdversarialResult with validation metrics and recommendations
        """
        logger.info("Starting adversarial validation")
        
        # Prepare data
        if features is None:
            features = list(train_data.columns)
        
        train_subset = train_data[features].copy()
        test_subset = test_data[features].copy()
        
        # Handle categorical features
        if categorical_features is not None:
            train_subset, test_subset = self._encode_categorical(
                train_subset, test_subset, categorical_features
            )
        
        # Create adversarial labels
        train_subset['is_test'] = 0
        test_subset['is_test'] = 1
        
        # Combine datasets
        combined = pd.concat([train_subset, test_subset], axis=0, ignore_index=True)
        
        # Handle missing values
        combined = combined.fillna(-999)
        
        X = combined.drop('is_test', axis=1)
        y = combined['is_test']
        
        # Train classifier
        clf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10
        )
        
        # Cross-validation
        logger.info("Performing cross-validation")
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True,
                            random_state=self.random_state)
        cv_scores = cross_val_score(clf, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
        
        # Fit final model
        clf.fit(X, y)
        predictions = clf.predict_proba(X)[:, 1]
        
        # Calculate metrics
        auc_score = roc_auc_score(y, predictions)
        feature_importance = dict(zip(X.columns, clf.feature_importances_))
        
        # Sort by importance
        feature_importance = dict(
            sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        )
        
        # Create result
        result = AdversarialResult(
            auc_score=auc_score,
            feature_importance=feature_importance,
            cv_scores=cv_scores.tolist(),
            predictions=predictions
        )
        
        # Determine status
        result.status = self._determine_status(auc_score)
        
        # Identify drift features
        result.drift_features = [
            feat for feat, importance in list(feature_importance.items())[:10]
            if importance > 0.05
        ]
        
        # Generate warnings and recommendations
        self._add_warnings_and_recommendations(result)
        
        logger.info(f"Adversarial validation complete. AUC={auc_score:.4f}, "
                   f"Status={result.status}")
        
        return result
    
    def _encode_categorical(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        categorical_features: List[str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Encode categorical features using label encoding."""
        train_encoded = train_df.copy()
        test_encoded = test_df.copy()
        
        for col in categorical_features:
            if col not in train_df.columns:
                continue
            
            le = LabelEncoder()
            
            # Combine to fit encoder on all possible values
            combined = pd.concat([
                train_df[col].astype(str),
                test_df[col].astype(str)
            ])
            le.fit(combined)
            
            train_encoded[col] = le.transform(train_df[col].astype(str))
            test_encoded[col] = le.transform(test_df[col].astype(str))
        
        return train_encoded, test_encoded
    
    def _determine_status(self, auc_score: float) -> str:
        """Determine validation status based on AUC."""
        if auc_score >= self.auc_critical:
            return 'fail'
        elif auc_score >= self.auc_threshold:
            return 'warning'
        else:
            return 'pass'
    
    def _add_warnings_and_recommendations(self, result: AdversarialResult) -> None:
        """Add warnings and mitigation recommendations."""
        auc = result.auc_score
        
        if result.status == 'fail':
            result.warnings.append(
                f"CRITICAL: Very high AUC ({auc:.4f}) indicates severe distribution "
                "shift or data leakage. Model may not generalize."
            )
            result.mitigation_recommendations.extend([
                "Investigate data collection process for potential leakage",
                "Review feature engineering pipeline for look-ahead bias",
                "Consider using temporal validation instead of random split",
                "Examine top important features for unexpected patterns",
                "Verify that test data represents true production distribution"
            ])
        
        elif result.status == 'warning':
            result.warnings.append(
                f"WARNING: Elevated AUC ({auc:.4f}) suggests moderate distribution "
                "shift between train and test."
            )
            result.mitigation_recommendations.extend([
                "Review feature distributions between train and test",
                "Consider stratified sampling to balance distributions",
                "Investigate temporal patterns in data",
                "Apply domain adaptation techniques if appropriate",
                "Monitor model performance on test set closely"
            ])
        
        else:
            result.warnings.append(
                f"PASS: AUC ({auc:.4f}) indicates similar distributions. "
                "No significant issues detected."
            )
        
        # Feature-specific recommendations
        if result.drift_features:
            top_drift = ', '.join(result.drift_features[:3])
            result.warnings.append(
                f"Features with highest drift: {top_drift}"
            )
            result.mitigation_recommendations.append(
                f"Investigate distribution of: {', '.join(result.drift_features[:5])}"
            )
    
    def validate_temporal(
        self,
        data: pd.DataFrame,
        time_column: str,
        train_end_date: str,
        test_start_date: str,
        features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None
    ) -> AdversarialResult:
        """
        Perform temporal adversarial validation.
        
        Useful for detecting temporal drift in time-series data.
        
        Args:
            data: Full dataset with time column
            time_column: Name of datetime column
            train_end_date: End date for training period
            test_start_date: Start date for test period
            features: List of features to use
            categorical_features: List of categorical feature names
        
        Returns:
            AdversarialResult with temporal validation results
        """
        logger.info(f"Performing temporal adversarial validation: "
                   f"train <= {train_end_date}, test >= {test_start_date}")
        
        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(data[time_column]):
            data[time_column] = pd.to_datetime(data[time_column])
        
        # Split data temporally
        train_data = data[data[time_column] <= train_end_date].copy()
        test_data = data[data[time_column] >= test_start_date].copy()
        
        logger.info(f"Train samples: {len(train_data)}, Test samples: {len(test_data)}")
        
        if len(train_data) == 0 or len(test_data) == 0:
            raise ValueError("Train or test data is empty after temporal split")
        
        # Remove time column from features
        if features is None:
            features = [col for col in data.columns if col != time_column]
        else:
            features = [col for col in features if col != time_column]
        
        return self.validate(train_data, test_data, features, categorical_features)
    
    def detect_sampling_bias(
        self,
        population: pd.DataFrame,
        sample: pd.DataFrame,
        features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None
    ) -> AdversarialResult:
        """
        Detect sampling bias by comparing sample to full population.
        
        Args:
            population: Full population dataset
            sample: Sampled subset
            features: Features to analyze
            categorical_features: Categorical feature names
        
        Returns:
            AdversarialResult indicating sampling bias
        """
        logger.info("Detecting sampling bias")
        
        result = self.validate(population, sample, features, categorical_features)
        
        # Adjust interpretation for sampling bias
        if result.status != 'pass':
            result.warnings.insert(0,
                "Sampling bias detected: sample distribution differs from population"
            )
            result.mitigation_recommendations.insert(0,
                "Use stratified sampling or weighting to match population distribution"
            )
        
        return result
    
    def visualize_roc_curve(
        self,
        result: AdversarialResult,
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6)
    ) -> None:
        """
        Visualize ROC curve for adversarial validation.
        
        Args:
            result: AdversarialResult from validation
            output_path: Path to save plot
            figsize: Figure size tuple
        """
        if len(result.predictions) == 0:
            logger.warning("No predictions available for ROC curve")
            return
        
        # Recreate labels (assuming half train, half test)
        n_samples = len(result.predictions)
        y_true = np.concatenate([
            np.zeros(n_samples // 2),
            np.ones(n_samples - n_samples // 2)
        ])
        
        fpr, tpr, thresholds = roc_curve(y_true, result.predictions)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot ROC curve
        ax.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {result.auc_score:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.5)')
        
        # Color based on status
        if result.status == 'fail':
            color = '#e74c3c'
        elif result.status == 'warning':
            color = '#f39c12'
        else:
            color = '#2ecc71'
        
        ax.fill_between(fpr, 0, tpr, alpha=0.2, color=color)
        
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('Adversarial Validation ROC Curve', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curve saved to {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def visualize_feature_importance(
        self,
        result: AdversarialResult,
        top_n: int = 15,
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8)
    ) -> None:
        """
        Visualize top features contributing to distribution differences.
        
        Args:
            result: AdversarialResult from validation
            top_n: Number of top features to display
            output_path: Path to save plot
            figsize: Figure size tuple
        """
        if not result.feature_importance:
            logger.warning("No feature importance available")
            return
        
        # Get top N features
        top_features = dict(list(result.feature_importance.items())[:top_n])
        
        features = list(top_features.keys())
        importances = list(top_features.values())
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create horizontal bar chart
        colors = ['#e74c3c' if imp > 0.1 else '#f39c12' if imp > 0.05 else '#3498db'
                 for imp in importances]
        
        bars = ax.barh(range(len(features)), importances, color=colors, alpha=0.7,
                      edgecolor='black')
        
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features)
        ax.set_xlabel('Feature Importance', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        ax.set_title('Top Features Contributing to Distribution Differences',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, importances)):
            ax.text(val, i, f' {val:.3f}', va='center', fontsize=9)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def generate_report(
        self,
        result: AdversarialResult,
        output_path: str = 'adversarial_report.html'
    ) -> str:
        """
        Generate comprehensive HTML adversarial validation report.
        
        Args:
            result: AdversarialResult from validation
            output_path: Output file path
        
        Returns:
            Path to generated report
        """
        # Determine status styling
        status_color = {
            'pass': '#2ecc71',
            'warning': '#f39c12',
            'fail': '#e74c3c'
        }.get(result.status, 'gray')
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Adversarial Validation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
                .container {{ background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
                h2 {{ color: #34495e; margin-top: 30px; }}
                .metric {{ background-color: #ecf0f1; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                .metric-label {{ font-weight: bold; color: #7f8c8d; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: {status_color}; }}
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
                <h1>Adversarial Validation Report</h1>
                <p class="timestamp">Generated: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <div class="metric">
                    <div class="metric-label">Train-Test Discriminator AUC</div>
                    <div class="metric-value">{result.auc_score:.4f}</div>
                    <div>Status: <span class="status-badge">{result.status.upper()}</span></div>
                </div>
                
                <h2>Interpretation</h2>
                <ul>
                    <li><strong>AUC ≈ 0.5:</strong> Train and test distributions are similar (ideal)</li>
                    <li><strong>AUC = 0.85-0.95:</strong> Moderate distribution shift (investigate)</li>
                    <li><strong>AUC &gt; 0.95:</strong> Severe shift or data leakage (critical issue)</li>
                </ul>
                
                <h2>Cross-Validation Scores</h2>
                <div class="metric">
                    <div>Mean AUC: <strong>{np.mean(result.cv_scores):.4f}</strong></div>
                    <div>Std AUC: <strong>{np.std(result.cv_scores):.4f}</strong></div>
                    <div>Scores: {', '.join([f'{s:.4f}' for s in result.cv_scores])}</div>
                </div>
        """
        
        # Add warnings
        if result.warnings:
            html_content += "<h2>Findings</h2>"
            for warning in result.warnings:
                if 'CRITICAL' in warning or 'fail' in result.status:
                    css_class = 'alert'
                elif 'WARNING' in warning or 'warning' in result.status:
                    css_class = 'warning'
                else:
                    css_class = 'success'
                
                html_content += f'<div class="{css_class}">{warning}</div>'
        
        # Add top drift features
        if result.drift_features:
            html_content += """
                <h2>Top Features Contributing to Distribution Shift</h2>
                <table>
                    <tr>
                        <th>Rank</th>
                        <th>Feature</th>
                        <th>Importance</th>
                    </tr>
            """
            
            for rank, feature in enumerate(result.drift_features, 1):
                importance = result.feature_importance.get(feature, 0)
                html_content += f"""
                    <tr>
                        <td>{rank}</td>
                        <td>{feature}</td>
                        <td>{importance:.4f}</td>
                    </tr>
                """
            
            html_content += "</table>"
        
        # Add recommendations
        if result.mitigation_recommendations:
            html_content += "<h2>Mitigation Recommendations</h2><ul>"
            for rec in result.mitigation_recommendations:
                html_content += f"<li>{rec}</li>"
            html_content += "</ul>"
        
        # Add feature importance table
        if result.feature_importance:
            html_content += """
                <h2>All Feature Importance Scores</h2>
                <table>
                    <tr>
                        <th>Feature</th>
                        <th>Importance</th>
                    </tr>
            """
            
            for feature, importance in list(result.feature_importance.items())[:20]:
                html_content += f"""
                    <tr>
                        <td>{feature}</td>
                        <td>{importance:.4f}</td>
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
        
        logger.info(f"Adversarial validation report generated: {output_path}")
        return output_path


def quick_adversarial_check(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: Optional[List[str]] = None
) -> float:
    """
    Quick adversarial validation check.
    
    Args:
        train_df: Training data
        test_df: Test data
        features: Features to use
    
    Returns:
        AUC score
    """
    validator = AdversarialValidator()
    result = validator.validate(train_df, test_df, features)
    return result.auc_score
