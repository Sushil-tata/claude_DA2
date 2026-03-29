"""
Multi-Model Orchestrator for Principal Data Science Decision Agent.

This module manages multiple model candidates, runs champion-challenger frameworks,
and produces comparison tables with trade-offs.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import yaml
from pathlib import Path
from loguru import logger


@dataclass
class ModelCandidate:
    """Represents a model candidate in the orchestration framework."""
    
    name: str
    model_type: str
    model_instance: Any
    hyperparameters: Dict[str, Any]
    metrics: Dict[str, float] = field(default_factory=dict)
    predictions: Optional[np.ndarray] = None
    feature_importance: Optional[pd.DataFrame] = None
    training_time: float = 0.0
    inference_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnsembleConfig:
    """Configuration for ensemble models."""
    
    method: str  # 'weighted_average', 'stacking', 'blending'
    weights: Optional[List[float]] = None
    meta_learner: Optional[Any] = None
    cv_folds: int = 5
    optimization_metric: str = 'auc'


class ModelOrchestrator:
    """
    Orchestrates multiple model candidates in a champion-challenger framework.
    
    This class manages:
    - Multiple model training and evaluation
    - Champion-challenger comparisons
    - Ensemble construction
    - Model selection recommendations
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the model orchestrator.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.candidates: List[ModelCandidate] = []
        self.champion: Optional[ModelCandidate] = None
        self.challengers: List[ModelCandidate] = []
        self.ensemble: Optional[ModelCandidate] = None
        
    def _load_config(self, config_path: Optional[Path]) -> Dict:
        """Load orchestration configuration."""
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "agent_config.yaml"
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                return config.get('orchestration', {})
        return {}
    
    def add_candidate(self, candidate: ModelCandidate) -> None:
        """
        Add a model candidate to the orchestration.
        
        Args:
            candidate: ModelCandidate instance
        """
        self.candidates.append(candidate)
        logger.info(f"Added candidate: {candidate.name} ({candidate.model_type})")
    
    def train_all_candidates(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        parallel: bool = True
    ) -> None:
        """
        Train all model candidates.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            parallel: Whether to train models in parallel
        """
        if parallel and self.config.get('parallel_execution', True):
            self._train_parallel(X_train, y_train, X_val, y_val)
        else:
            self._train_sequential(X_train, y_train, X_val, y_val)
    
    def _train_sequential(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame],
        y_val: Optional[pd.Series]
    ) -> None:
        """Train models sequentially."""
        for candidate in self.candidates:
            try:
                logger.info(f"Training {candidate.name}...")
                import time
                start_time = time.time()
                
                candidate.model_instance.fit(X_train, y_train)
                candidate.training_time = time.time() - start_time
                
                # Generate predictions
                if hasattr(candidate.model_instance, 'predict_proba'):
                    candidate.predictions = candidate.model_instance.predict_proba(X_val)[:, 1]
                else:
                    candidate.predictions = candidate.model_instance.predict(X_val)
                
                logger.info(f"Completed {candidate.name} in {candidate.training_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Error training {candidate.name}: {str(e)}")
    
    def _train_parallel(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame],
        y_val: Optional[pd.Series]
    ) -> None:
        """Train models in parallel."""
        max_workers = self.config.get('max_workers', 4)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for candidate in self.candidates:
                future = executor.submit(
                    self._train_single_candidate,
                    candidate, X_train, y_train, X_val
                )
                futures[future] = candidate
            
            for future in as_completed(futures):
                candidate = futures[future]
                try:
                    future.result()
                    logger.info(f"Completed training {candidate.name}")
                except Exception as e:
                    logger.error(f"Error training {candidate.name}: {str(e)}")
    
    def _train_single_candidate(
        self,
        candidate: ModelCandidate,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame
    ) -> None:
        """Train a single candidate (for parallel execution)."""
        import time
        start_time = time.time()
        
        candidate.model_instance.fit(X_train, y_train)
        candidate.training_time = time.time() - start_time
        
        # Generate predictions
        if hasattr(candidate.model_instance, 'predict_proba'):
            candidate.predictions = candidate.model_instance.predict_proba(X_val)[:, 1]
        else:
            candidate.predictions = candidate.model_instance.predict(X_val)
    
    def evaluate_all_candidates(
        self,
        y_true: pd.Series,
        metrics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Evaluate all candidates on specified metrics.
        
        Args:
            y_true: True labels
            metrics: List of metric names to compute
            
        Returns:
            DataFrame with evaluation results
        """
        if metrics is None:
            metrics = ['auc', 'ks', 'gini', 'accuracy', 'precision', 'recall']
        
        results = []
        for candidate in self.candidates:
            if candidate.predictions is None:
                logger.warning(f"No predictions for {candidate.name}, skipping evaluation")
                continue
            
            metrics_dict = self._compute_metrics(
                y_true, candidate.predictions, metrics
            )
            candidate.metrics = metrics_dict
            
            results.append({
                'model_name': candidate.name,
                'model_type': candidate.model_type,
                **metrics_dict,
                'training_time': candidate.training_time
            })
        
        return pd.DataFrame(results)
    
    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metrics: List[str]
    ) -> Dict[str, float]:
        """Compute specified metrics."""
        from sklearn.metrics import (
            roc_auc_score, accuracy_score, precision_score,
            recall_score, f1_score, log_loss
        )
        
        results = {}
        
        for metric in metrics:
            try:
                if metric == 'auc':
                    results['auc'] = roc_auc_score(y_true, y_pred)
                elif metric == 'ks':
                    results['ks'] = self._compute_ks(y_true, y_pred)
                elif metric == 'gini':
                    auc = roc_auc_score(y_true, y_pred)
                    results['gini'] = 2 * auc - 1
                elif metric == 'accuracy':
                    y_pred_binary = (y_pred > 0.5).astype(int)
                    results['accuracy'] = accuracy_score(y_true, y_pred_binary)
                elif metric == 'precision':
                    y_pred_binary = (y_pred > 0.5).astype(int)
                    results['precision'] = precision_score(y_true, y_pred_binary)
                elif metric == 'recall':
                    y_pred_binary = (y_pred > 0.5).astype(int)
                    results['recall'] = recall_score(y_true, y_pred_binary)
                elif metric == 'f1':
                    y_pred_binary = (y_pred > 0.5).astype(int)
                    results['f1'] = f1_score(y_true, y_pred_binary)
                elif metric == 'logloss':
                    results['logloss'] = log_loss(y_true, y_pred)
            except Exception as e:
                logger.warning(f"Could not compute {metric}: {str(e)}")
                results[metric] = np.nan
        
        return results
    
    @staticmethod
    def _compute_ks(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Kolmogorov-Smirnov statistic."""
        df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
        df = df.sort_values('y_pred', ascending=False).reset_index(drop=True)
        
        df['cumsum_pos'] = (df['y_true'] == 1).cumsum() / (df['y_true'] == 1).sum()
        df['cumsum_neg'] = (df['y_true'] == 0).cumsum() / (df['y_true'] == 0).sum()
        
        ks = (df['cumsum_pos'] - df['cumsum_neg']).abs().max()
        return ks
    
    def select_champion_challengers(
        self,
        primary_metric: str = 'auc',
        n_challengers: int = 2
    ) -> Tuple[ModelCandidate, List[ModelCandidate]]:
        """
        Select champion and challenger models.
        
        Args:
            primary_metric: Metric to use for selection
            n_challengers: Number of challengers to select
            
        Returns:
            Tuple of (champion, challengers)
        """
        # Sort candidates by primary metric
        sorted_candidates = sorted(
            self.candidates,
            key=lambda x: x.metrics.get(primary_metric, 0),
            reverse=True
        )
        
        self.champion = sorted_candidates[0]
        self.challengers = sorted_candidates[1:n_challengers+1]
        
        logger.info(f"Champion: {self.champion.name} ({primary_metric}={self.champion.metrics[primary_metric]:.4f})")
        for i, challenger in enumerate(self.challengers, 1):
            logger.info(f"Challenger {i}: {challenger.name} ({primary_metric}={challenger.metrics.get(primary_metric, 0):.4f})")
        
        return self.champion, self.challengers
    
    def create_comparison_table(self) -> pd.DataFrame:
        """
        Create detailed comparison table with trade-offs.
        
        Returns:
            DataFrame with model comparisons
        """
        comparison_data = []
        
        for candidate in self.candidates:
            row = {
                'Model': candidate.name,
                'Type': candidate.model_type,
                'AUC': candidate.metrics.get('auc', np.nan),
                'KS': candidate.metrics.get('ks', np.nan),
                'Gini': candidate.metrics.get('gini', np.nan),
                'Training Time (s)': candidate.training_time,
                'Champion': candidate == self.champion,
                'Challenger': candidate in self.challengers
            }
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('AUC', ascending=False)
        
        return df
    
    def generate_recommendation_report(self) -> str:
        """
        Generate detailed recommendation report.
        
        Returns:
            Formatted recommendation report
        """
        report = "# MODEL ORCHESTRATION RECOMMENDATION REPORT\n\n"
        
        if self.champion:
            report += f"## Champion Model: {self.champion.name}\n"
            report += f"- Type: {self.champion.model_type}\n"
            report += f"- AUC: {self.champion.metrics.get('auc', 'N/A'):.4f}\n"
            report += f"- KS: {self.champion.metrics.get('ks', 'N/A'):.4f}\n"
            report += f"- Gini: {self.champion.metrics.get('gini', 'N/A'):.4f}\n"
            report += f"- Training Time: {self.champion.training_time:.2f}s\n\n"
        
        if self.challengers:
            report += "## Challenger Models:\n"
            for i, challenger in enumerate(self.challengers, 1):
                report += f"\n### Challenger {i}: {challenger.name}\n"
                report += f"- Type: {challenger.model_type}\n"
                report += f"- AUC: {challenger.metrics.get('auc', 'N/A'):.4f}\n"
                report += f"- KS: {challenger.metrics.get('ks', 'N/A'):.4f}\n"
                report += f"- Gini: {challenger.metrics.get('gini', 'N/A'):.4f}\n"
                report += f"- Training Time: {challenger.training_time:.2f}s\n"
        
        report += "\n## Comparison Table:\n"
        report += self.create_comparison_table().to_markdown(index=False)
        
        return report
