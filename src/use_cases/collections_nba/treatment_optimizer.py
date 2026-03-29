"""
Collections NBA Treatment Optimizer

Selects optimal resolution path considering expected value, constraints, and
multi-armed bandit integration for continuous optimization.

Example:
    >>> from src.use_cases.collections_nba.treatment_optimizer import TreatmentOptimizer
    >>> optimizer = TreatmentOptimizer(config_path='config/model_config.yaml')
    >>> optimizer.fit(X_train, y_train, treatments_train, costs_train)
    >>> optimal_treatments = optimizer.recommend(X_test)
"""

import joblib
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import defaultdict

from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    logger.warning("LightGBM not available. Install with: pip install lightgbm")
    LIGHTGBM_AVAILABLE = False


class TreatmentOptimizerConfig:
    """Configuration for treatment optimizer."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize treatment optimizer configuration.
        
        Args:
            config_path: Path to model_config.yaml
        """
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                full_config = yaml.safe_load(f)
                treatment_config = full_config.get('treatment_optimizer', {})
        else:
            treatment_config = {}
        
        self.treatments = treatment_config.get('treatments', [
            'Legal', 'Settlement', 'Restructuring', 'DebtSale', 'SelfCure'
        ])
        
        # Treatment costs (relative units)
        self.treatment_costs = treatment_config.get('treatment_costs', {
            'Legal': 100,
            'Settlement': 50,
            'Restructuring': 75,
            'DebtSale': 25,
            'SelfCure': 5
        })
        
        # Treatment success rates (base rates, will be personalized)
        self.base_success_rates = treatment_config.get('base_success_rates', {
            'Legal': 0.3,
            'Settlement': 0.5,
            'Restructuring': 0.6,
            'DebtSale': 0.8,
            'SelfCure': 0.4
        })
        
        # Capacity constraints (max % of portfolio)
        self.capacity_constraints = treatment_config.get('capacity_constraints', {
            'Legal': 0.15,
            'Settlement': 0.30,
            'Restructuring': 0.20,
            'DebtSale': 0.25,
            'SelfCure': 1.0
        })
        
        # Regulatory constraints
        self.regulatory_constraints = treatment_config.get('regulatory_constraints', {
            'min_settlement_pct': 0.3,  # Min settlement as % of outstanding
            'max_legal_age_days': 180,  # Max age for legal action
        })
        
        # Multi-armed bandit parameters
        self.use_bandit = treatment_config.get('use_bandit', True)
        self.epsilon = treatment_config.get('epsilon', 0.1)  # Exploration rate
        self.bandit_window = treatment_config.get('bandit_window', 1000)  # Update frequency
        
        self.model_params = treatment_config.get('model_params', {
            'objective': 'multiclass',
            'num_class': len(self.treatments),
            'metric': 'multi_logloss',
            'boosting': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'verbose': -1,
        })


class TreatmentOptimizer:
    """
    Optimal treatment selection for collections NBA.
    
    Selects best resolution path based on:
    - Expected recovery value
    - Treatment costs
    - Capacity constraints
    - Regulatory compliance
    - Historical performance (via multi-armed bandit)
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize treatment optimizer.
        
        Args:
            config_path: Path to model_config.yaml
            
        Example:
            >>> optimizer = TreatmentOptimizer(config_path="config/model_config.yaml")
        """
        self.config = TreatmentOptimizerConfig(config_path)
        
        # Models for each treatment outcome
        self.outcome_models: Dict[str, Any] = {}
        
        # Multi-armed bandit stats
        self.treatment_counts: Dict[str, int] = defaultdict(int)
        self.treatment_rewards: Dict[str, float] = defaultdict(float)
        self.treatment_success: Dict[str, List[float]] = defaultdict(list)
        
        self.feature_names: Optional[List[str]] = None
        self.is_fitted = False
        
        logger.info(f"Initialized TreatmentOptimizer with treatments: {self.config.treatments}")
    
    def fit(
        self,
        X: pd.DataFrame,
        outcomes: pd.DataFrame,
        treatments: pd.Series,
        recovery_amounts: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        outcomes_val: Optional[pd.DataFrame] = None,
        treatments_val: Optional[pd.Series] = None
    ) -> 'TreatmentOptimizer':
        """
        Fit treatment outcome models.
        
        Args:
            X: Training features
            outcomes: Binary outcomes for each treatment (columns = treatments)
            treatments: Actual treatments applied
            recovery_amounts: Actual recovery amounts
            X_val: Validation features
            outcomes_val: Validation outcomes
            treatments_val: Validation treatments
            
        Returns:
            Self for method chaining
            
        Example:
            >>> optimizer.fit(X_train, outcomes_train, treatments_train, amounts_train)
        """
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is required. Install with: pip install lightgbm")
        
        self.feature_names = list(X.columns)
        
        # Train separate outcome model for each treatment
        logger.info(f"Training outcome models for {len(self.config.treatments)} treatments")
        
        for treatment in self.config.treatments:
            if treatment not in outcomes.columns:
                logger.warning(f"Treatment {treatment} not in outcomes, skipping")
                continue
            
            # Filter to samples where this treatment was applied
            treatment_mask = treatments == treatment
            X_treatment = X[treatment_mask]
            y_treatment = outcomes[treatment][treatment_mask]
            
            if len(X_treatment) < 50:
                logger.warning(f"Treatment {treatment} has only {len(X_treatment)} samples, using base rate")
                continue
            
            logger.info(f"Training model for {treatment}: {len(X_treatment)} samples, {y_treatment.mean():.2%} success rate")
            
            # Train outcome model
            model = lgb.LGBMClassifier(**{
                **self.config.model_params,
                'objective': 'binary',
                'metric': 'auc'
            })
            
            # Validation set for this treatment
            if X_val is not None and treatments_val is not None:
                val_mask = treatments_val == treatment
                X_val_treatment = X_val[val_mask]
                y_val_treatment = outcomes_val[treatment][val_mask] if outcomes_val is not None else None
                eval_set = [(X_val_treatment, y_val_treatment)] if len(X_val_treatment) > 0 and y_val_treatment is not None else None
            else:
                eval_set = None
            
            model.fit(
                X_treatment, y_treatment,
                eval_set=eval_set,
                callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)] if eval_set else None
            )
            
            self.outcome_models[treatment] = model
            
            # Initialize bandit stats
            self.treatment_counts[treatment] = len(X_treatment)
            self.treatment_rewards[treatment] = y_treatment.sum()
            self.treatment_success[treatment] = y_treatment.tolist()
        
        self.is_fitted = True
        logger.info("Treatment optimizer training complete")
        return self
    
    def predict_treatment_outcomes(
        self,
        X: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Predict success probability for each treatment.
        
        Args:
            X: Features
            
        Returns:
            DataFrame with columns for each treatment's success probability
            
        Example:
            >>> probs = optimizer.predict_treatment_outcomes(X_test)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        result = pd.DataFrame(index=X.index)
        
        for treatment in self.config.treatments:
            if treatment in self.outcome_models:
                probs = self.outcome_models[treatment].predict_proba(X)[:, 1]
            else:
                # Use base rate if no model
                probs = np.full(len(X), self.config.base_success_rates.get(treatment, 0.5))
            
            result[f'{treatment}_prob'] = probs
        
        return result
    
    def calculate_expected_value(
        self,
        X: pd.DataFrame,
        outstanding_amounts: pd.Series,
        treatment_probs: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Calculate expected value for each treatment.
        
        Args:
            X: Features
            outstanding_amounts: Outstanding debt amounts
            treatment_probs: Pre-computed treatment probabilities (optional)
            
        Returns:
            DataFrame with expected values for each treatment
            
        Example:
            >>> ev = optimizer.calculate_expected_value(X_test, outstanding_test)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if treatment_probs is None:
            treatment_probs = self.predict_treatment_outcomes(X)
        
        result = pd.DataFrame(index=X.index)
        
        for treatment in self.config.treatments:
            prob_col = f'{treatment}_prob'
            if prob_col not in treatment_probs.columns:
                continue
            
            # Expected recovery (simplified: prob * outstanding)
            expected_recovery = treatment_probs[prob_col] * outstanding_amounts
            
            # Subtract treatment cost
            treatment_cost = self.config.treatment_costs.get(treatment, 0)
            
            expected_value = expected_recovery - treatment_cost
            
            result[f'{treatment}_ev'] = expected_value
        
        return result
    
    def recommend(
        self,
        X: pd.DataFrame,
        outstanding_amounts: pd.Series,
        account_ages: Optional[pd.Series] = None,
        constraints: Optional[Dict[str, Any]] = None,
        use_exploration: bool = True
    ) -> pd.DataFrame:
        """
        Recommend optimal treatment for each account.
        
        Args:
            X: Features
            outstanding_amounts: Outstanding debt amounts
            account_ages: Account age in days (for regulatory constraints)
            constraints: Additional constraints dict
            use_exploration: Whether to use epsilon-greedy exploration
            
        Returns:
            DataFrame with recommended treatment and expected value
            
        Example:
            >>> recommendations = optimizer.recommend(X_test, outstanding_test, ages_test)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Get treatment probabilities and expected values
        treatment_probs = self.predict_treatment_outcomes(X)
        expected_values = self.calculate_expected_value(X, outstanding_amounts, treatment_probs)
        
        # Apply regulatory constraints
        if account_ages is not None:
            max_legal_age = self.config.regulatory_constraints.get('max_legal_age_days', 180)
            legal_ev_col = 'Legal_ev'
            if legal_ev_col in expected_values.columns:
                # Disqualify legal action for accounts beyond max age
                expected_values.loc[account_ages > max_legal_age, legal_ev_col] = -np.inf
        
        # Settlement constraints
        settlement_ev_col = 'Settlement_ev'
        if settlement_ev_col in expected_values.columns:
            min_settlement_pct = self.config.regulatory_constraints.get('min_settlement_pct', 0.3)
            # This is simplified - in practice, would check actual settlement terms
        
        result = pd.DataFrame(index=X.index)
        
        # Select best treatment
        ev_cols = [col for col in expected_values.columns if col.endswith('_ev')]
        
        # Epsilon-greedy exploration
        if use_exploration and self.config.use_bandit:
            explore_mask = np.random.random(len(X)) < self.config.epsilon
            
            # Exploitation: select best EV
            best_treatments = expected_values[ev_cols].idxmax(axis=1).str.replace('_ev', '')
            best_evs = expected_values[ev_cols].max(axis=1)
            
            # Exploration: random treatment
            random_treatments = np.random.choice(
                self.config.treatments,
                size=len(X)
            )
            
            # Combine
            result['treatment'] = np.where(explore_mask, random_treatments, best_treatments)
            result['expected_value'] = best_evs
            result['is_exploration'] = explore_mask
        else:
            # Pure exploitation
            result['treatment'] = expected_values[ev_cols].idxmax(axis=1).str.replace('_ev', '')
            result['expected_value'] = expected_values[ev_cols].max(axis=1)
            result['is_exploration'] = False
        
        # Add success probabilities
        for treatment in self.config.treatments:
            prob_col = f'{treatment}_prob'
            if prob_col in treatment_probs.columns:
                result[prob_col] = treatment_probs[prob_col]
        
        # Apply capacity constraints
        result = self._apply_capacity_constraints(result, expected_values)
        
        return result
    
    def _apply_capacity_constraints(
        self,
        recommendations: pd.DataFrame,
        expected_values: pd.DataFrame
    ) -> pd.DataFrame:
        """Apply capacity constraints to treatment recommendations."""
        treatment_counts = recommendations['treatment'].value_counts()
        total = len(recommendations)
        
        # Check each treatment against capacity
        for treatment, count in treatment_counts.items():
            capacity = self.config.capacity_constraints.get(treatment, 1.0)
            max_count = int(total * capacity)
            
            if count > max_count:
                logger.warning(
                    f"Treatment {treatment} exceeds capacity: {count} > {max_count}, "
                    f"reallocating excess"
                )
                
                # Find accounts assigned to this treatment
                treatment_mask = recommendations['treatment'] == treatment
                treatment_indices = recommendations[treatment_mask].index
                
                # Sort by expected value (lowest first for reallocation)
                ev_col = f'{treatment}_ev'
                if ev_col in expected_values.columns:
                    sorted_indices = expected_values.loc[treatment_indices, ev_col].sort_values().index
                else:
                    sorted_indices = treatment_indices
                
                # Reallocate excess to next best treatment
                excess_count = count - max_count
                excess_indices = sorted_indices[:excess_count]
                
                # Find next best treatment for excess accounts
                ev_cols = [col for col in expected_values.columns if col.endswith('_ev')]
                for idx in excess_indices:
                    # Get EVs for this account, excluding current treatment
                    account_evs = expected_values.loc[idx, ev_cols].copy()
                    account_evs[ev_col] = -np.inf
                    next_best = account_evs.idxmax().replace('_ev', '')
                    recommendations.loc[idx, 'treatment'] = next_best
                    recommendations.loc[idx, 'expected_value'] = account_evs.max()
        
        return recommendations
    
    def update_bandit_stats(
        self,
        treatments: pd.Series,
        outcomes: pd.Series
    ) -> None:
        """
        Update multi-armed bandit statistics with new results.
        
        Args:
            treatments: Applied treatments
            outcomes: Binary outcomes (1=success, 0=failure)
            
        Example:
            >>> optimizer.update_bandit_stats(treatments_actual, outcomes_actual)
        """
        for treatment in self.config.treatments:
            treatment_mask = treatments == treatment
            if treatment_mask.any():
                outcomes_treatment = outcomes[treatment_mask]
                
                self.treatment_counts[treatment] += len(outcomes_treatment)
                self.treatment_rewards[treatment] += outcomes_treatment.sum()
                self.treatment_success[treatment].extend(outcomes_treatment.tolist())
                
                # Keep only recent history
                if len(self.treatment_success[treatment]) > self.config.bandit_window:
                    self.treatment_success[treatment] = self.treatment_success[treatment][-self.config.bandit_window:]
        
        logger.info("Updated bandit statistics")
    
    def get_bandit_stats(self) -> pd.DataFrame:
        """
        Get current multi-armed bandit statistics.
        
        Returns:
            DataFrame with treatment statistics
            
        Example:
            >>> stats = optimizer.get_bandit_stats()
        """
        stats = []
        
        for treatment in self.config.treatments:
            count = self.treatment_counts[treatment]
            rewards = self.treatment_rewards[treatment]
            
            stat = {
                'treatment': treatment,
                'count': count,
                'total_rewards': rewards,
                'success_rate': rewards / count if count > 0 else 0,
                'recent_success_rate': np.mean(self.treatment_success[treatment]) if self.treatment_success[treatment] else 0,
                'base_rate': self.config.base_success_rates.get(treatment, 0),
                'cost': self.config.treatment_costs.get(treatment, 0),
                'capacity': self.config.capacity_constraints.get(treatment, 1.0)
            }
            
            stats.append(stat)
        
        return pd.DataFrame(stats)
    
    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save optimizer to disk.
        
        Args:
            filepath: Path to save optimizer
            
        Example:
            >>> optimizer.save("models/treatment_optimizer.pkl")
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        save_dict = {
            'outcome_models': self.outcome_models,
            'feature_names': self.feature_names,
            'config': self.config,
            'treatment_counts': dict(self.treatment_counts),
            'treatment_rewards': dict(self.treatment_rewards),
            'treatment_success': dict(self.treatment_success),
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(save_dict, filepath)
        logger.info(f"Optimizer saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'TreatmentOptimizer':
        """
        Load optimizer from disk.
        
        Args:
            filepath: Path to saved optimizer
            
        Returns:
            Loaded TreatmentOptimizer instance
            
        Example:
            >>> optimizer = TreatmentOptimizer.load("models/treatment_optimizer.pkl")
        """
        save_dict = joblib.load(filepath)
        
        instance = cls()
        instance.outcome_models = save_dict['outcome_models']
        instance.feature_names = save_dict['feature_names']
        instance.config = save_dict['config']
        instance.treatment_counts = defaultdict(int, save_dict['treatment_counts'])
        instance.treatment_rewards = defaultdict(float, save_dict['treatment_rewards'])
        instance.treatment_success = defaultdict(list, save_dict['treatment_success'])
        instance.is_fitted = save_dict['is_fitted']
        
        logger.info(f"Optimizer loaded from {filepath}")
        return instance
