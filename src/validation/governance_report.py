"""
Model Governance and Regulatory Reporting Framework

This module provides comprehensive model governance, including model cards,
regulatory documentation, fairness analysis, and audit trails.

Usage Example:
    >>> from governance_report import GovernanceReporter
    >>> reporter = GovernanceReporter(model_name='Credit Risk Model v2.1')
    >>> model_card = reporter.generate_model_card(
    ...     model_details={'type': 'XGBoost', 'version': '2.1'},
    ...     performance_metrics={'auc': 0.85, 'accuracy': 0.82},
    ...     fairness_metrics={'demographic_parity': 0.95}
    ... )
    >>> reporter.save_model_card(model_card, 'model_card.html')
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict
from datetime import datetime
import json
from pathlib import Path
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


@dataclass
class ModelCard:
    """
    Google-style Model Card for comprehensive model documentation.
    
    Based on: "Model Cards for Model Reporting" (Mitchell et al., 2019)
    """
    model_name: str
    model_version: str
    model_date: datetime = field(default_factory=datetime.now)
    
    # Model Details
    model_type: str = ""
    model_architecture: str = ""
    training_data_description: str = ""
    features_used: List[str] = field(default_factory=list)
    
    # Intended Use
    intended_use: str = ""
    intended_users: List[str] = field(default_factory=list)
    out_of_scope_uses: List[str] = field(default_factory=list)
    
    # Performance Metrics
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    performance_by_segment: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Fairness & Bias
    fairness_metrics: Dict[str, float] = field(default_factory=dict)
    bias_analysis: Dict[str, Any] = field(default_factory=dict)
    protected_groups: List[str] = field(default_factory=list)
    
    # Limitations & Risks
    limitations: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    mitigation_strategies: List[str] = field(default_factory=list)
    
    # Explainability
    feature_importance: Dict[str, float] = field(default_factory=dict)
    explainability_methods: List[str] = field(default_factory=list)
    
    # Regulatory & Compliance
    regulatory_requirements: List[str] = field(default_factory=list)
    compliance_status: Dict[str, str] = field(default_factory=dict)
    
    # Version Control & Lineage
    previous_version: Optional[str] = None
    changes_from_previous: List[str] = field(default_factory=list)
    training_job_id: Optional[str] = None
    
    # Contact & Ownership
    model_owner: str = ""
    contact_email: str = ""
    approval_status: str = "pending"  # pending, approved, rejected


@dataclass
class AuditTrail:
    """Audit trail entry for model governance."""
    timestamp: datetime
    action: str
    user: str
    details: Dict[str, Any] = field(default_factory=dict)
    model_version: str = ""


class GovernanceReporter:
    """
    Comprehensive model governance and regulatory reporting system.
    
    Provides tools for model documentation, fairness analysis, regulatory
    compliance, and audit trails following industry best practices.
    
    Attributes:
        model_name: Name of the model
        regulatory_framework: Regulatory framework to follow (e.g., 'SR 11-7', 'GDPR')
        audit_log_path: Path to store audit trail
    
    Example:
        >>> reporter = GovernanceReporter('Fraud Detection Model')
        >>> reporter.log_audit('model_training', 'data_scientist@company.com')
        >>> fairness = reporter.analyze_fairness(y_true, y_pred, protected_attr)
    """
    
    def __init__(
        self,
        model_name: str,
        model_version: str = "1.0",
        regulatory_framework: Optional[List[str]] = None,
        audit_log_path: Optional[str] = None
    ):
        self.model_name = model_name
        self.model_version = model_version
        self.regulatory_framework = regulatory_framework or []
        self.audit_log_path = audit_log_path
        self.audit_trail: List[AuditTrail] = []
        
        logger.info(f"Initialized GovernanceReporter for {model_name} v{model_version}")
    
    def generate_model_card(
        self,
        model_details: Dict[str, Any],
        intended_use: Dict[str, Any],
        performance_metrics: Dict[str, float],
        fairness_metrics: Optional[Dict[str, float]] = None,
        limitations: Optional[List[str]] = None,
        feature_importance: Optional[Dict[str, float]] = None
    ) -> ModelCard:
        """
        Generate comprehensive model card.
        
        Args:
            model_details: Dictionary with model architecture, type, etc.
            intended_use: Dictionary with intended use cases and users
            performance_metrics: Model performance metrics
            fairness_metrics: Fairness and bias metrics
            limitations: List of known limitations
            feature_importance: Feature importance scores
        
        Returns:
            ModelCard object with complete documentation
        """
        logger.info(f"Generating model card for {self.model_name}")
        
        card = ModelCard(
            model_name=self.model_name,
            model_version=self.model_version,
            model_type=model_details.get('type', ''),
            model_architecture=model_details.get('architecture', ''),
            training_data_description=model_details.get('training_data', ''),
            features_used=model_details.get('features', []),
            intended_use=intended_use.get('description', ''),
            intended_users=intended_use.get('users', []),
            out_of_scope_uses=intended_use.get('out_of_scope', []),
            performance_metrics=performance_metrics,
            fairness_metrics=fairness_metrics or {},
            limitations=limitations or [],
            feature_importance=feature_importance or {},
            model_owner=model_details.get('owner', ''),
            contact_email=model_details.get('contact', '')
        )
        
        # Add regulatory requirements based on framework
        card.regulatory_requirements = self._get_regulatory_requirements()
        
        logger.info("Model card generated successfully")
        return card
    
    def _get_regulatory_requirements(self) -> List[str]:
        """Get regulatory requirements based on framework."""
        requirements = []
        
        if 'SR 11-7' in self.regulatory_framework:
            requirements.extend([
                "Model Risk Management Framework Documentation",
                "Model Validation Independent Review",
                "Ongoing Performance Monitoring",
                "Model Inventory Documentation"
            ])
        
        if 'GDPR' in self.regulatory_framework:
            requirements.extend([
                "Right to Explanation for Automated Decisions",
                "Data Protection Impact Assessment",
                "Privacy by Design Implementation",
                "Data Subject Rights Compliance"
            ])
        
        if 'FCRA' in self.regulatory_framework:
            requirements.extend([
                "Adverse Action Notice Capability",
                "Model Interpretability for Credit Decisions",
                "Fair Credit Reporting Compliance"
            ])
        
        if 'ECOA' in self.regulatory_framework:
            requirements.extend([
                "Fair Lending Analysis",
                "Disparate Impact Testing",
                "Discrimination Prevention Controls"
            ])
        
        return requirements
    
    def analyze_fairness(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        protected_attribute: np.ndarray,
        favorable_outcome: int = 1
    ) -> Dict[str, float]:
        """
        Analyze model fairness across protected groups.
        
        Calculates multiple fairness metrics:
        - Demographic Parity: P(Ŷ=1|A=0) / P(Ŷ=1|A=1)
        - Equal Opportunity: P(Ŷ=1|Y=1,A=0) / P(Ŷ=1|Y=1,A=1) (TPR parity)
        - Equalized Odds Score: 1 - max(|1-TPR_ratio|, |1-FPR_ratio|)
          Both TPR and FPR should be similar across groups. Score of 1.0 is perfect.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            protected_attribute: Protected group membership (0/1)
            favorable_outcome: Label considered favorable (default: 1)
        
        Returns:
            Dictionary with fairness metrics
        """
        logger.info("Analyzing model fairness")
        
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        protected_attribute = np.asarray(protected_attribute)
        
        # Split by protected attribute
        mask_0 = protected_attribute == 0
        mask_1 = protected_attribute == 1
        
        # Demographic Parity (Statistical Parity)
        # Ratio should be close to 1.0
        positive_rate_0 = np.mean(y_pred[mask_0] == favorable_outcome)
        positive_rate_1 = np.mean(y_pred[mask_1] == favorable_outcome)
        demographic_parity = (positive_rate_0 / positive_rate_1
                            if positive_rate_1 > 0 else 0)
        
        # Equal Opportunity (True Positive Rate Parity)
        # Among true positives, equal prediction rates
        true_pos_mask_0 = mask_0 & (y_true == favorable_outcome)
        true_pos_mask_1 = mask_1 & (y_true == favorable_outcome)
        
        tpr_0 = (np.mean(y_pred[true_pos_mask_0] == favorable_outcome)
                if np.any(true_pos_mask_0) else 0)
        tpr_1 = (np.mean(y_pred[true_pos_mask_1] == favorable_outcome)
                if np.any(true_pos_mask_1) else 0)
        equal_opportunity = tpr_0 / tpr_1 if tpr_1 > 0 else 0
        
        # False Positive Rate Parity
        true_neg_mask_0 = mask_0 & (y_true != favorable_outcome)
        true_neg_mask_1 = mask_1 & (y_true != favorable_outcome)
        
        fpr_0 = (np.mean(y_pred[true_neg_mask_0] == favorable_outcome)
                if np.any(true_neg_mask_0) else 0)
        fpr_1 = (np.mean(y_pred[true_neg_mask_1] == favorable_outcome)
                if np.any(true_neg_mask_1) else 0)
        fpr_parity = fpr_0 / fpr_1 if fpr_1 > 0 else 0
        
        # Equalized Odds (combines TPR and FPR parity)
        # Both TPR and FPR should be similar across groups
        # We report the maximum deviation from parity
        tpr_deviation = abs(1.0 - equal_opportunity)
        fpr_deviation = abs(1.0 - fpr_parity)
        equalized_odds_score = 1.0 - max(tpr_deviation, fpr_deviation)
        
        # Acceptance rates by group
        acceptance_rate_0 = positive_rate_0
        acceptance_rate_1 = positive_rate_1
        
        fairness_metrics = {
            'demographic_parity': float(demographic_parity),
            'equal_opportunity': float(equal_opportunity),
            'equalized_odds_score': float(equalized_odds_score),
            'fpr_parity': float(fpr_parity),
            'acceptance_rate_group_0': float(acceptance_rate_0),
            'acceptance_rate_group_1': float(acceptance_rate_1),
            'tpr_group_0': float(tpr_0),
            'tpr_group_1': float(tpr_1),
            'fpr_group_0': float(fpr_0),
            'fpr_group_1': float(fpr_1)
        }
        
        # Assess fairness
        fairness_assessment = self._assess_fairness(fairness_metrics)
        fairness_metrics['fairness_assessment'] = fairness_assessment
        
        logger.info(f"Fairness analysis complete. Assessment: {fairness_assessment}")
        return fairness_metrics
    
    def _assess_fairness(self, metrics: Dict[str, float]) -> str:
        """
        Assess overall fairness based on metrics.
        
        Common threshold: ratio should be between 0.8 and 1.25 (80% rule)
        """
        # Check if key metrics are within acceptable range
        thresholds_met = 0
        total_checks = 0
        
        for metric in ['demographic_parity', 'equal_opportunity', 'equalized_odds_score']:
            if metric in metrics:
                total_checks += 1
                value = metrics[metric]
                # Check if ratio/score is within acceptable range
                # For equalized_odds_score, higher is better (closer to 1.0)
                if metric == 'equalized_odds_score':
                    if value >= 0.8:  # 80% or better means both TPR and FPR are within 20% parity
                        thresholds_met += 1
                else:
                    # For ratios, check if between 0.8 and 1.25
                    if 0.8 <= value <= 1.25:
                        thresholds_met += 1
        
        if total_checks == 0:
            return 'unknown'
        
        ratio = thresholds_met / total_checks
        
        if ratio >= 0.75:
            return 'fair'
        elif ratio >= 0.5:
            return 'moderate_bias'
        else:
            return 'significant_bias'
    
    def detect_bias(
        self,
        dataset: pd.DataFrame,
        protected_attributes: List[str],
        target_col: str,
        prediction_col: str
    ) -> Dict[str, Any]:
        """
        Comprehensive bias detection across multiple protected attributes.
        
        Args:
            dataset: Full dataset with features, target, and predictions
            protected_attributes: List of protected attribute column names
            target_col: Target column name
            prediction_col: Prediction column name
        
        Returns:
            Dictionary with bias analysis for each protected attribute
        """
        logger.info(f"Detecting bias across {len(protected_attributes)} protected attributes")
        
        bias_report = {}
        
        for attr in protected_attributes:
            if attr not in dataset.columns:
                logger.warning(f"Protected attribute '{attr}' not in dataset")
                continue
            
            # Analyze fairness for this attribute
            fairness = self.analyze_fairness(
                dataset[target_col].values,
                dataset[prediction_col].values,
                dataset[attr].values
            )
            
            bias_report[attr] = fairness
        
        return bias_report
    
    def generate_explainability_summary(
        self,
        feature_importance: Dict[str, float],
        shap_values: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        top_n: int = 10
    ) -> Dict[str, Any]:
        """
        Generate explainability summary from feature importance and SHAP values.
        
        Args:
            feature_importance: Dictionary of feature importance scores
            shap_values: SHAP values array (optional)
            feature_names: Feature names for SHAP values
            top_n: Number of top features to include
        
        Returns:
            Dictionary with explainability analysis
        """
        logger.info("Generating explainability summary")
        
        summary = {
            'top_features': {},
            'explainability_methods_used': []
        }
        
        # Feature importance
        if feature_importance:
            sorted_features = sorted(
                feature_importance.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:top_n]
            
            summary['top_features'] = dict(sorted_features)
            summary['explainability_methods_used'].append('feature_importance')
        
        # SHAP values
        if shap_values is not None and feature_names is not None:
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            shap_importance = dict(zip(feature_names, mean_abs_shap))
            
            sorted_shap = sorted(
                shap_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:top_n]
            
            summary['shap_importance'] = dict(sorted_shap)
            summary['explainability_methods_used'].append('SHAP')
        
        logger.info(f"Explainability summary generated with "
                   f"{len(summary['explainability_methods_used'])} methods")
        
        return summary
    
    def log_audit(
        self,
        action: str,
        user: str,
        details: Optional[Dict[str, Any]] = None
    ) -> AuditTrail:
        """
        Log action to audit trail.
        
        Args:
            action: Action performed (e.g., 'model_training', 'model_deployment')
            user: User who performed action
            details: Additional details
        
        Returns:
            AuditTrail entry
        """
        entry = AuditTrail(
            timestamp=datetime.now(),
            action=action,
            user=user,
            details=details or {},
            model_version=self.model_version
        )
        
        self.audit_trail.append(entry)
        
        logger.info(f"Audit log: {action} by {user}")
        
        # Save to file if path specified
        if self.audit_log_path:
            self._save_audit_log()
        
        return entry
    
    def _save_audit_log(self) -> None:
        """Save audit trail to file."""
        if not self.audit_log_path:
            return
        
        audit_data = [
            {
                'timestamp': entry.timestamp.isoformat(),
                'action': entry.action,
                'user': entry.user,
                'details': entry.details,
                'model_version': entry.model_version
            }
            for entry in self.audit_trail
        ]
        
        with open(self.audit_log_path, 'w') as f:
            json.dump(audit_data, f, indent=2)
    
    def get_audit_history(
        self,
        action_filter: Optional[str] = None,
        user_filter: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get audit history as DataFrame.
        
        Args:
            action_filter: Filter by action type
            user_filter: Filter by user
        
        Returns:
            DataFrame with audit history
        """
        if not self.audit_trail:
            return pd.DataFrame()
        
        df = pd.DataFrame([
            {
                'timestamp': entry.timestamp,
                'action': entry.action,
                'user': entry.user,
                'model_version': entry.model_version,
                'details': str(entry.details)
            }
            for entry in self.audit_trail
        ])
        
        if action_filter:
            df = df[df['action'] == action_filter]
        
        if user_filter:
            df = df[df['user'] == user_filter]
        
        return df
    
    def save_model_card(
        self,
        model_card: ModelCard,
        output_path: str = 'model_card.html',
        format: str = 'html'
    ) -> str:
        """
        Save model card to file.
        
        Args:
            model_card: ModelCard object
            output_path: Output file path
            format: Output format ('html', 'json', 'markdown')
        
        Returns:
            Path to saved file
        """
        if format == 'html':
            return self._save_model_card_html(model_card, output_path)
        elif format == 'json':
            return self._save_model_card_json(model_card, output_path)
        elif format == 'markdown':
            return self._save_model_card_markdown(model_card, output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _save_model_card_html(self, card: ModelCard, path: str) -> str:
        """Save model card as HTML."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Card: {card.model_name}</title>
            <style>
                body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
                .container {{ background-color: white; padding: 40px; border-radius: 10px; 
                            box-shadow: 0 2px 10px rgba(0,0,0,0.1); max-width: 1200px; margin: 0 auto; }}
                h1 {{ color: #2c3e50; border-bottom: 4px solid #3498db; padding-bottom: 15px; }}
                h2 {{ color: #34495e; margin-top: 35px; border-bottom: 2px solid #ecf0f1; padding-bottom: 10px; }}
                h3 {{ color: #7f8c8d; margin-top: 20px; }}
                .section {{ background-color: #f8f9fa; padding: 20px; margin: 15px 0; border-radius: 5px; 
                           border-left: 4px solid #3498db; }}
                .metric {{ display: inline-block; background-color: #ecf0f1; padding: 10px 20px; 
                          margin: 5px; border-radius: 5px; }}
                .metric-label {{ font-weight: bold; color: #7f8c8d; font-size: 12px; }}
                .metric-value {{ font-size: 20px; font-weight: bold; color: #2c3e50; }}
                .warning {{ background-color: #fff3cd; border-left: 4px solid #f39c12; padding: 15px; margin: 10px 0; }}
                .success {{ background-color: #d4edda; border-left: 4px solid #2ecc71; padding: 15px; margin: 10px 0; }}
                .alert {{ background-color: #f8d7da; border-left: 4px solid #e74c3c; padding: 15px; margin: 10px 0; }}
                table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #3498db; color: white; font-weight: bold; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .badge {{ display: inline-block; padding: 5px 12px; border-radius: 3px; 
                         color: white; font-size: 12px; font-weight: bold; }}
                .badge-pending {{ background-color: #f39c12; }}
                .badge-approved {{ background-color: #2ecc71; }}
                .badge-rejected {{ background-color: #e74c3c; }}
                ul {{ line-height: 1.8; }}
                .timestamp {{ color: #95a5a6; font-size: 14px; font-style: italic; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🎯 Model Card: {card.model_name}</h1>
                <p class="timestamp">Version: {card.model_version} | 
                   Generated: {card.model_date.strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <div class="section">
                    <p><strong>Approval Status:</strong> 
                       <span class="badge badge-{card.approval_status}">{card.approval_status.upper()}</span>
                    </p>
                    <p><strong>Owner:</strong> {card.model_owner}</p>
                    <p><strong>Contact:</strong> {card.contact_email}</p>
                </div>
                
                <h2>📋 Model Details</h2>
                <div class="section">
                    <p><strong>Type:</strong> {card.model_type}</p>
                    <p><strong>Architecture:</strong> {card.model_architecture}</p>
                    <p><strong>Training Data:</strong> {card.training_data_description}</p>
                    {f'<p><strong>Training Job ID:</strong> {card.training_job_id}</p>' if card.training_job_id else ''}
                </div>
                
                <h2>🎯 Intended Use</h2>
                <div class="section">
                    <p>{card.intended_use}</p>
                    {f'<h3>Intended Users:</h3><ul>{"".join([f"<li>{u}</li>" for u in card.intended_users])}</ul>' if card.intended_users else ''}
                    {f'<h3>Out of Scope Uses:</h3><ul>{"".join([f"<li>{u}</li>" for u in card.out_of_scope_uses])}</ul>' if card.out_of_scope_uses else ''}
                </div>
        """
        
        # Performance Metrics
        if card.performance_metrics:
            html_content += """
                <h2>📊 Performance Metrics</h2>
                <div class="section">
            """
            for metric, value in card.performance_metrics.items():
                html_content += f"""
                    <div class="metric">
                        <div class="metric-label">{metric.upper()}</div>
                        <div class="metric-value">{value:.4f}</div>
                    </div>
                """
            html_content += "</div>"
        
        # Performance by Segment
        if card.performance_by_segment:
            html_content += """
                <h3>Performance by Segment</h3>
                <table>
                    <tr><th>Segment</th><th>Metrics</th></tr>
            """
            for segment, metrics in card.performance_by_segment.items():
                metrics_str = ', '.join([f"{k}: {v:.4f}" for k, v in metrics.items()])
                html_content += f"<tr><td>{segment}</td><td>{metrics_str}</td></tr>"
            html_content += "</table>"
        
        # Fairness Analysis
        if card.fairness_metrics:
            html_content += """
                <h2>⚖️ Fairness & Bias Analysis</h2>
                <div class="section">
            """
            
            fairness_status = card.fairness_metrics.get('fairness_assessment', 'unknown')
            if fairness_status == 'fair':
                html_content += '<div class="success">✓ Model passes fairness criteria</div>'
            elif fairness_status == 'moderate_bias':
                html_content += '<div class="warning">⚠ Moderate bias detected</div>'
            elif fairness_status == 'significant_bias':
                html_content += '<div class="alert">⚠ Significant bias detected</div>'
            
            html_content += "<table><tr><th>Metric</th><th>Value</th><th>Assessment</th></tr>"
            for metric, value in card.fairness_metrics.items():
                if metric == 'fairness_assessment':
                    continue
                
                assessment = "✓ Fair" if 0.8 <= value <= 1.25 else "⚠ Biased"
                html_content += f"""
                    <tr>
                        <td>{metric.replace('_', ' ').title()}</td>
                        <td>{value:.4f}</td>
                        <td>{assessment}</td>
                    </tr>
                """
            html_content += "</table>"
            
            if card.protected_groups:
                html_content += f"""
                    <p><strong>Protected Groups Analyzed:</strong> 
                       {', '.join(card.protected_groups)}</p>
                """
            
            html_content += "</div>"
        
        # Feature Importance
        if card.feature_importance:
            html_content += """
                <h2>🔍 Explainability</h2>
                <div class="section">
                    <h3>Top Features</h3>
                    <table>
                        <tr><th>Rank</th><th>Feature</th><th>Importance</th></tr>
            """
            for rank, (feature, importance) in enumerate(
                sorted(card.feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:10], 1
            ):
                html_content += f"""
                    <tr>
                        <td>{rank}</td>
                        <td>{feature}</td>
                        <td>{importance:.4f}</td>
                    </tr>
                """
            html_content += "</table>"
            
            if card.explainability_methods:
                html_content += f"""
                    <p><strong>Methods Used:</strong> {', '.join(card.explainability_methods)}</p>
                """
            
            html_content += "</div>"
        
        # Limitations & Risks
        if card.limitations or card.risks:
            html_content += "<h2>⚠️ Limitations & Risks</h2><div class='section'>"
            
            if card.limitations:
                html_content += f"""
                    <h3>Known Limitations</h3>
                    <ul>{"".join([f"<li>{lim}</li>" for lim in card.limitations])}</ul>
                """
            
            if card.risks:
                html_content += f"""
                    <h3>Identified Risks</h3>
                    <ul>{"".join([f"<li>{risk}</li>" for risk in card.risks])}</ul>
                """
            
            if card.mitigation_strategies:
                html_content += f"""
                    <h3>Mitigation Strategies</h3>
                    <ul>{"".join([f"<li>{mit}</li>" for mit in card.mitigation_strategies])}</ul>
                """
            
            html_content += "</div>"
        
        # Regulatory Compliance
        if card.regulatory_requirements:
            html_content += f"""
                <h2>📜 Regulatory Compliance</h2>
                <div class="section">
                    <h3>Requirements</h3>
                    <ul>{"".join([f"<li>{req}</li>" for req in card.regulatory_requirements])}</ul>
            """
            
            if card.compliance_status:
                html_content += """
                    <h3>Compliance Status</h3>
                    <table>
                        <tr><th>Requirement</th><th>Status</th></tr>
                """
                for req, status in card.compliance_status.items():
                    html_content += f"<tr><td>{req}</td><td>{status}</td></tr>"
                html_content += "</table>"
            
            html_content += "</div>"
        
        # Version History
        if card.previous_version or card.changes_from_previous:
            html_content += "<h2>📝 Version History</h2><div class='section'>"
            
            if card.previous_version:
                html_content += f"<p><strong>Previous Version:</strong> {card.previous_version}</p>"
            
            if card.changes_from_previous:
                html_content += f"""
                    <h3>Changes from Previous Version</h3>
                    <ul>{"".join([f"<li>{change}</li>" for change in card.changes_from_previous])}</ul>
                """
            
            html_content += "</div>"
        
        html_content += """
            </div>
        </body>
        </html>
        """
        
        with open(path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Model card saved to {path}")
        return path
    
    def _save_model_card_json(self, card: ModelCard, path: str) -> str:
        """Save model card as JSON."""
        card_dict = asdict(card)
        card_dict['model_date'] = card.model_date.isoformat()
        
        with open(path, 'w') as f:
            json.dump(card_dict, f, indent=2)
        
        logger.info(f"Model card JSON saved to {path}")
        return path
    
    def _save_model_card_markdown(self, card: ModelCard, path: str) -> str:
        """Save model card as Markdown."""
        md_content = f"""# Model Card: {card.model_name}

**Version:** {card.model_version}  
**Date:** {card.model_date.strftime('%Y-%m-%d')}  
**Status:** {card.approval_status.upper()}  
**Owner:** {card.model_owner}  
**Contact:** {card.contact_email}

---

## Model Details

- **Type:** {card.model_type}
- **Architecture:** {card.model_architecture}
- **Training Data:** {card.training_data_description}

## Intended Use

{card.intended_use}

### Intended Users
"""
        for user in card.intended_users:
            md_content += f"- {user}\n"
        
        if card.out_of_scope_uses:
            md_content += "\n### Out of Scope Uses\n"
            for use in card.out_of_scope_uses:
                md_content += f"- {use}\n"
        
        if card.performance_metrics:
            md_content += "\n## Performance Metrics\n\n"
            for metric, value in card.performance_metrics.items():
                md_content += f"- **{metric}:** {value:.4f}\n"
        
        if card.fairness_metrics:
            md_content += "\n## Fairness Analysis\n\n"
            for metric, value in card.fairness_metrics.items():
                md_content += f"- **{metric}:** {value:.4f}\n"
        
        if card.limitations:
            md_content += "\n## Limitations\n\n"
            for lim in card.limitations:
                md_content += f"- {lim}\n"
        
        if card.risks:
            md_content += "\n## Risks\n\n"
            for risk in card.risks:
                md_content += f"- {risk}\n"
        
        with open(path, 'w') as f:
            f.write(md_content)
        
        logger.info(f"Model card Markdown saved to {path}")
        return path
    
    def generate_performance_report(
        self,
        metrics: Dict[str, float],
        segment_metrics: Optional[Dict[str, Dict[str, float]]] = None,
        output_path: str = 'performance_report.html'
    ) -> str:
        """
        Generate comprehensive performance report.
        
        Args:
            metrics: Overall performance metrics
            segment_metrics: Performance by segment
            output_path: Output file path
        
        Returns:
            Path to generated report
        """
        # Simple HTML report for performance metrics
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Performance Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #2c3e50; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #3498db; color: white; }}
            </style>
        </head>
        <body>
            <h1>Model Performance Report: {self.model_name}</h1>
            <h2>Overall Metrics</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
        """
        
        for metric, value in metrics.items():
            html_content += f"<tr><td>{metric}</td><td>{value:.4f}</td></tr>"
        
        html_content += "</table>"
        
        if segment_metrics:
            html_content += "<h2>Performance by Segment</h2>"
            for segment, seg_metrics in segment_metrics.items():
                html_content += f"<h3>{segment}</h3><table><tr><th>Metric</th><th>Value</th></tr>"
                for metric, value in seg_metrics.items():
                    html_content += f"<tr><td>{metric}</td><td>{value:.4f}</td></tr>"
                html_content += "</table>"
        
        html_content += "</body></html>"
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Performance report saved to {output_path}")
        return output_path
