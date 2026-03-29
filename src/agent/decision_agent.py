"""
Main Decision Agent for Principal Data Science in Financial Services.

This is the core agent that accepts problem definitions, routes to appropriate
pipelines, and produces structured decision outputs.
"""

from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import yaml

from .prompt_engine import PromptEngine
from .orchestrator import ModelOrchestrator, ModelCandidate


class UseCase(Enum):
    """Supported use cases."""
    COLLECTIONS_NBA = "collections_nba"
    FRAUD_DETECTION = "fraud_detection"
    BEHAVIORAL_SCORING = "behavioral_scoring"
    INCOME_ESTIMATION = "income_estimation"


@dataclass
class ProblemDefinition:
    """Defines a problem for the decision agent."""
    
    use_case: UseCase
    business_objective: str
    data_sources: List[str]
    target_variable: str
    evaluation_metrics: List[str]
    constraints: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionOutput:
    """Structured decision output from the agent."""
    
    problem_understanding: Dict[str, Any]
    data_architecture: Dict[str, Any]
    feature_blueprint: Dict[str, Any]
    modeling_blueprint: Dict[str, Any]
    optimization_strategy: Dict[str, Any]
    validation_blueprint: Dict[str, Any]
    simulation_policy: Dict[str, Any]
    production_design: Dict[str, Any]
    recommendations: List[str] = field(default_factory=list)
    trade_offs: Dict[str, List[str]] = field(default_factory=dict)


class DecisionAgent:
    """
    Principal Data Science Decision Agent.
    
    This agent operates at the level of a Head of AI / Chief Risk Scientist,
    providing comprehensive decision support for ML problems in financial services.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the decision agent.
        
        Args:
            config_path: Path to agent configuration file
        """
        self.config_path = config_path or self._get_default_config_path()
        self.config = self._load_config()
        self.prompt_engine = PromptEngine(self.config_path)
        self.orchestrator = ModelOrchestrator(self.config_path)
        
        # Setup logging
        self._setup_logging()
        
        logger.info("Principal Data Science Decision Agent initialized")
        logger.info(f"Operating rules: {self.config.get('agent', {}).get('operating_rules', [])}")
    
    @staticmethod
    def _get_default_config_path() -> Path:
        """Get default configuration path."""
        return Path(__file__).parent.parent.parent / "config" / "agent_config.yaml"
    
    def _load_config(self) -> Dict:
        """Load agent configuration."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        return {}
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        log_config = self.config.get('logging', {})
        log_level = log_config.get('level', 'INFO')
        
        logger.remove()  # Remove default handler
        logger.add(
            lambda msg: print(msg, end=''),
            level=log_level,
            format=log_config.get('format', '{time} | {level} | {message}')
        )
    
    def analyze_problem(self, problem: ProblemDefinition) -> DecisionOutput:
        """
        Analyze a problem and produce structured decision output.
        
        Args:
            problem: Problem definition
            
        Returns:
            Structured decision output
        """
        logger.info(f"Analyzing problem: {problem.use_case.value}")
        logger.info(f"Business objective: {problem.business_objective}")
        
        # Get relevant prompts
        system_prompt = self.prompt_engine.get_combined_prompt(problem.use_case.value)
        logger.debug(f"Using system prompt with {len(system_prompt)} characters")
        
        # Build decision output
        decision = DecisionOutput(
            problem_understanding=self._analyze_problem_understanding(problem),
            data_architecture=self._design_data_architecture(problem),
            feature_blueprint=self._design_feature_blueprint(problem),
            modeling_blueprint=self._design_modeling_blueprint(problem),
            optimization_strategy=self._design_optimization_strategy(problem),
            validation_blueprint=self._design_validation_blueprint(problem),
            simulation_policy=self._design_simulation_policy(problem),
            production_design=self._design_production_architecture(problem)
        )
        
        # Generate recommendations
        decision.recommendations = self._generate_recommendations(problem, decision)
        decision.trade_offs = self._analyze_trade_offs(problem, decision)
        
        logger.info("Problem analysis complete")
        return decision
    
    def _analyze_problem_understanding(self, problem: ProblemDefinition) -> Dict[str, Any]:
        """Analyze problem understanding."""
        return {
            'use_case': problem.use_case.value,
            'business_objective': problem.business_objective,
            'target_variable': problem.target_variable,
            'evaluation_metrics': problem.evaluation_metrics,
            'constraints': problem.constraints,
            'loss_function': self._infer_loss_function(problem),
            'success_criteria': self._define_success_criteria(problem)
        }
    
    def _infer_loss_function(self, problem: ProblemDefinition) -> str:
        """Infer appropriate loss function based on problem type."""
        if problem.use_case == UseCase.COLLECTIONS_NBA:
            return "Expected recovery value (probabilistic + amount estimation)"
        elif problem.use_case == UseCase.FRAUD_DETECTION:
            return "Weighted cross-entropy (accounting for fraud costs vs. false positive costs)"
        elif problem.use_case == UseCase.BEHAVIORAL_SCORING:
            return "Binary cross-entropy with calibration penalty"
        elif problem.use_case == UseCase.INCOME_ESTIMATION:
            return "Pinball loss for quantile regression (median + confidence intervals)"
        return "Binary cross-entropy"
    
    def _define_success_criteria(self, problem: ProblemDefinition) -> List[str]:
        """Define success criteria for the problem."""
        criteria = [
            f"Achieve target metrics: {', '.join(problem.evaluation_metrics)}",
            "Pass OOT validation (minimum 3 months hold-out)",
            "Sub-segment stability (PSI < 0.25 across all segments)",
            "Calibration test (Hosmer-Lemeshow p-value > 0.05)",
            "Business value positive (measured through simulation)"
        ]
        return criteria
    
    def _design_data_architecture(self, problem: ProblemDefinition) -> Dict[str, Any]:
        """Design data architecture."""
        return {
            'data_sources': problem.data_sources,
            'required_datasets': self._identify_required_datasets(problem),
            'schema_design': self._design_schema(problem),
            'quality_checks': [
                'Missing value analysis',
                'Outlier detection',
                'Distribution analysis',
                'Temporal drift detection',
                'Leakage risk assessment'
            ],
            'sampling_strategy': self._design_sampling_strategy(problem)
        }
    
    def _identify_required_datasets(self, problem: ProblemDefinition) -> List[str]:
        """Identify required datasets based on use case."""
        base_datasets = ['customer_master', 'account_master']
        
        if problem.use_case == UseCase.COLLECTIONS_NBA:
            return base_datasets + [
                'collections_history',
                'payment_history',
                'customer_communications',
                'treatment_outcomes',
                'account_balance_history'
            ]
        elif problem.use_case == UseCase.FRAUD_DETECTION:
            return base_datasets + [
                'transaction_history',
                'merchant_data',
                'device_fingerprints',
                'fraud_labels',
                'network_graph'
            ]
        elif problem.use_case == UseCase.BEHAVIORAL_SCORING:
            return base_datasets + [
                'transaction_history',
                'credit_history',
                'account_activity',
                'external_bureau_data'
            ]
        elif problem.use_case == UseCase.INCOME_ESTIMATION:
            return base_datasets + [
                'transaction_history',
                'deposit_patterns',
                'merchant_data',
                'employment_indicators'
            ]
        return base_datasets
    
    def _design_schema(self, problem: ProblemDefinition) -> Dict[str, Any]:
        """Design data schema."""
        return {
            'fact_tables': self._identify_required_datasets(problem),
            'dimension_tables': ['date_dimension', 'customer_dimension', 'product_dimension'],
            'grain': 'customer-month level',
            'temporal_coverage': '24 months historical + 6 months OOT'
        }
    
    def _design_sampling_strategy(self, problem: ProblemDefinition) -> Dict[str, Any]:
        """Design sampling strategy."""
        return {
            'train_period': '18 months',
            'validation_period': '3 months',
            'oot_period': '3 months',
            'sampling_method': 'stratified by target and key segments',
            'class_balancing': 'use class_weight or SMOTE if needed'
        }
    
    def _design_feature_blueprint(self, problem: ProblemDefinition) -> Dict[str, Any]:
        """Design feature engineering blueprint."""
        return {
            'feature_categories': self._identify_feature_categories(problem),
            'window_logic': {
                'short_term': [7, 14],
                'medium_term': [30, 60],
                'long_term': [90, 180]
            },
            'leakage_checks': [
                'Temporal validation',
                'Target correlation check (correlation > 0.95 flag)',
                'Adversarial validation (AUC > 0.85 flag)',
                'Feature availability in production'
            ],
            'feature_store': {
                'registry': 'enabled',
                'versioning': 'enabled',
                'lineage_tracking': 'enabled'
            }
        }
    
    def _identify_feature_categories(self, problem: ProblemDefinition) -> List[str]:
        """Identify relevant feature categories."""
        categories = ['demographic', 'account_attributes']
        
        if problem.use_case == UseCase.COLLECTIONS_NBA:
            categories.extend([
                'behavioral_velocity',
                'payment_patterns',
                'liquidity_features',
                'treatment_history',
                'temporal_features'
            ])
        elif problem.use_case == UseCase.FRAUD_DETECTION:
            categories.extend([
                'transaction_velocity',
                'graph_features',
                'device_features',
                'merchant_features',
                'anomaly_scores'
            ])
        elif problem.use_case == UseCase.BEHAVIORAL_SCORING:
            categories.extend([
                'transaction_patterns',
                'behavioral_stability',
                'persona_features',
                'temporal_sequences'
            ])
        elif problem.use_case == UseCase.INCOME_ESTIMATION:
            categories.extend([
                'deposit_intelligence',
                'transaction_patterns',
                'stability_metrics',
                'graph_payment_features'
            ])
        
        return categories
    
    def _design_modeling_blueprint(self, problem: ProblemDefinition) -> Dict[str, Any]:
        """Design modeling blueprint with multiple candidates."""
        return {
            'algorithm_candidates': self._select_algorithm_candidates(problem),
            'ensemble_strategy': 'Level-1 stacking with logistic regression meta-learner',
            'interpretability': 'SHAP values + feature importance + partial dependence plots',
            'champion_challenger_framework': {
                'champion': 'Best performing model on primary metric',
                'challengers': 'Top 2-3 alternative models',
                'evaluation_period': '3 months',
                'promotion_criteria': 'Consistent outperformance on multiple metrics'
            }
        }
    
    def _select_algorithm_candidates(self, problem: ProblemDefinition) -> List[Dict[str, Any]]:
        """Select algorithm candidates with justifications."""
        candidates = [
            {
                'name': 'LightGBM',
                'justification': 'Fast training, handles missing values, excellent performance',
                'pros': ['Speed', 'Accuracy', 'Feature importance', 'Handles categorical'],
                'cons': ['Overfitting risk', 'Hyperparameter sensitive']
            },
            {
                'name': 'XGBoost',
                'justification': 'Robust, well-established, excellent for structured data',
                'pros': ['Accuracy', 'Regularization', 'Parallel processing'],
                'cons': ['Slower than LightGBM', 'Memory intensive']
            },
            {
                'name': 'CatBoost',
                'justification': 'Handles categorical features natively, reduces overfitting',
                'pros': ['Categorical handling', 'Ordered boosting', 'Good defaults'],
                'cons': ['Slower training', 'Limited community support']
            }
        ]
        
        # Add use-case specific models
        if problem.use_case == UseCase.FRAUD_DETECTION:
            candidates.append({
                'name': 'Isolation Forest (for anomaly detection)',
                'justification': 'Unsupervised detection of rare fraud patterns',
                'pros': ['No labels needed', 'Detects novel patterns'],
                'cons': ['Lower precision', 'Requires tuning']
            })
        
        if problem.use_case == UseCase.BEHAVIORAL_SCORING:
            candidates.append({
                'name': 'TabNet',
                'justification': 'Deep learning with attention, captures complex patterns',
                'pros': ['Feature selection built-in', 'Complex interactions'],
                'cons': ['Longer training', 'Hyperparameter sensitive']
            })
        
        return candidates
    
    def _design_optimization_strategy(self, problem: ProblemDefinition) -> Dict[str, Any]:
        """Design hyperparameter optimization strategy."""
        return {
            'search_strategy': 'Bayesian optimization (Optuna)',
            'n_trials': 100,
            'timeout': '3600 seconds',
            'cv_strategy': '5-fold stratified time-series split',
            'optimization_metric': problem.evaluation_metrics[0] if problem.evaluation_metrics else 'auc',
            'multi_objective': {
                'enabled': True,
                'objectives': ['auc', 'stability_index', 'calibration_score', 'business_value']
            }
        }
    
    def _design_validation_blueprint(self, problem: ProblemDefinition) -> Dict[str, Any]:
        """Design validation framework."""
        return {
            'oot_testing': {
                'period': '3 months',
                'min_auc': self.config.get('validation', {}).get('min_auc', 0.65)
            },
            'sub_segment_analysis': {
                'segments': ['age_groups', 'account_vintage', 'product_type', 'risk_tier'],
                'min_segment_size': self.config.get('validation', {}).get('min_segment_size', 1000),
                'psi_threshold': self.config.get('validation', {}).get('psi_threshold', 0.25)
            },
            'calibration_tests': [
                'Hosmer-Lemeshow test',
                'Brier score',
                'Calibration plots by decile'
            ],
            'stability_metrics': ['PSI', 'CSI', 'Feature drift']
        }
    
    def _design_simulation_policy(self, problem: ProblemDefinition) -> Dict[str, Any]:
        """Design simulation and policy framework."""
        if problem.use_case == UseCase.COLLECTIONS_NBA:
            return {
                'simulation_type': 'Monte Carlo repayment simulation',
                'scenarios': ['baseline', 'optimistic', 'pessimistic', 'stress'],
                'economic_value_model': 'NPV of recovery - operational costs',
                'policy_rules': 'Treatment assignment based on propensity + expected payment'
            }
        elif problem.use_case == UseCase.FRAUD_DETECTION:
            return {
                'simulation_type': 'Adversarial simulation (fraud evolution)',
                'scenarios': ['current_patterns', 'emerging_patterns', 'coordinated_attacks'],
                'economic_value_model': 'Prevented fraud loss - false positive costs',
                'policy_rules': 'Risk-based decision thresholds by transaction type'
            }
        else:
            return {
                'simulation_type': 'Policy impact simulation',
                'scenarios': ['baseline', 'alternative_policies'],
                'economic_value_model': 'Business metric optimization',
                'policy_rules': 'Score-based decision rules'
            }
    
    def _design_production_architecture(self, problem: ProblemDefinition) -> Dict[str, Any]:
        """Design production deployment architecture."""
        return {
            'deployment_mode': self._select_deployment_mode(problem),
            'monitoring_framework': {
                'performance_metrics': problem.evaluation_metrics,
                'data_drift': ['PSI', 'KL divergence', 'Wasserstein distance'],
                'concept_drift': ['Model performance degradation'],
                'alerts': ['PSI > 0.25', 'AUC drop > 5%', 'Calibration degradation']
            },
            'retraining_triggers': [
                'PSI > 0.25 for 2 consecutive months',
                'AUC drop > 5%',
                'Scheduled quarterly retraining',
                'Significant business logic change'
            ],
            'feature_serving': {
                'online_features': 'Redis/feature store',
                'batch_features': 'Data warehouse',
                'latency_target': self._get_latency_target(problem)
            }
        }
    
    def _select_deployment_mode(self, problem: ProblemDefinition) -> str:
        """Select appropriate deployment mode."""
        if problem.use_case == UseCase.FRAUD_DETECTION:
            return "Real-time REST API (<100ms latency)"
        elif problem.use_case == UseCase.COLLECTIONS_NBA:
            return "Batch scoring (daily)"
        return "Hybrid (batch + on-demand API)"
    
    def _get_latency_target(self, problem: ProblemDefinition) -> str:
        """Get latency target based on use case."""
        if problem.use_case == UseCase.FRAUD_DETECTION:
            return "<100ms"
        elif problem.use_case == UseCase.COLLECTIONS_NBA:
            return "<1s (batch acceptable)"
        return "<500ms"
    
    def _generate_recommendations(
        self,
        problem: ProblemDefinition,
        decision: DecisionOutput
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = [
            f"Implement champion-challenger framework with {len(decision.modeling_blueprint['algorithm_candidates'])} model candidates",
            "Conduct OOT validation before production deployment",
            "Monitor sub-segment performance monthly",
            "Implement automated retraining pipeline",
            "Set up comprehensive monitoring dashboard"
        ]
        
        # Use case specific recommendations
        if problem.use_case == UseCase.FRAUD_DETECTION:
            recommendations.append("Implement real-time feature serving for <100ms latency")
            recommendations.append("Set up adversarial validation framework")
        
        return recommendations
    
    def _analyze_trade_offs(
        self,
        problem: ProblemDefinition,
        decision: DecisionOutput
    ) -> Dict[str, List[str]]:
        """Analyze trade-offs between different approaches."""
        return {
            'accuracy_vs_interpretability': [
                'Tree models: High accuracy, moderate interpretability',
                'Linear models: Lower accuracy, high interpretability',
                'Deep learning: Highest accuracy, lowest interpretability'
            ],
            'speed_vs_performance': [
                'LightGBM: Fastest training, excellent performance',
                'XGBoost: Moderate speed, excellent performance',
                'Deep learning: Slowest, potentially highest performance'
            ],
            'complexity_vs_maintainability': [
                'Ensemble: Higher performance, more complex to maintain',
                'Single model: Lower performance, easier to maintain',
                'AutoML: Highest automation, less control'
            ]
        }
    
    def generate_report(self, decision: DecisionOutput) -> str:
        """
        Generate comprehensive decision report.
        
        Args:
            decision: Decision output
            
        Returns:
            Formatted report string
        """
        template = self.prompt_engine.format_decision_template()
        
        report = "# PRINCIPAL DATA SCIENCE DECISION AGENT REPORT\n\n"
        report += f"## Problem Understanding\n"
        report += f"- Use Case: {decision.problem_understanding['use_case']}\n"
        report += f"- Business Objective: {decision.problem_understanding['business_objective']}\n"
        report += f"- Loss Function: {decision.problem_understanding['loss_function']}\n\n"
        
        report += "## Recommendations\n"
        for i, rec in enumerate(decision.recommendations, 1):
            report += f"{i}. {rec}\n"
        
        report += "\n## Trade-offs Analysis\n"
        for category, items in decision.trade_offs.items():
            report += f"\n### {category.replace('_', ' ').title()}\n"
            for item in items:
                report += f"- {item}\n"
        
        return report


def main():
    """Main entry point for the decision agent."""
    logger.info("Starting Principal Data Science Decision Agent")
    
    # Example usage
    agent = DecisionAgent()
    
    # Example problem
    problem = ProblemDefinition(
        use_case=UseCase.COLLECTIONS_NBA,
        business_objective="Maximize collections recovery while minimizing operational costs",
        data_sources=["collections_db", "payment_history", "customer_master"],
        target_variable="repayment_within_30d",
        evaluation_metrics=["auc", "ks", "recovery_rate"],
        constraints={"max_contact_attempts": 3, "regulatory_compliance": "FDCPA"}
    )
    
    decision = agent.analyze_problem(problem)
    report = agent.generate_report(decision)
    
    print(report)
    logger.info("Analysis complete")


if __name__ == "__main__":
    main()
