"""
AutoML and meta-learning for automated model selection and hyperparameter tuning.

This module provides Bayesian optimization, genetic algorithms, multi-objective
optimization, and automated model selection with comprehensive tracking.

Example:
    >>> from src.models.meta_learner import AutoMLEngine
    >>> automl = AutoMLEngine(config_path='config/model_config.yaml')
    >>> best_model = automl.fit(X_train, y_train, X_val, y_val, n_trials=100)
    >>> predictions = best_model.predict_proba(X_test)
"""

import joblib
import json
import numpy as np
import pandas as pd
import yaml
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from loguru import logger
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    f1_score, mean_squared_error, mean_absolute_error, r2_score
)

try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    logger.warning("Optuna not available. Install with: pip install optuna")
    OPTUNA_AVAILABLE = False

try:
    from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
    HYPEROPT_AVAILABLE = True
except ImportError:
    logger.warning("Hyperopt not available. Install with: pip install hyperopt")
    HYPEROPT_AVAILABLE = False


class BaseOptimizer(ABC):
    """Base class for hyperparameter optimizers."""
    
    def __init__(self, metric: str = 'auc', direction: str = 'maximize'):
        """
        Initialize optimizer.
        
        Args:
            metric: Optimization metric
            direction: 'maximize' or 'minimize'
        """
        self.metric = metric
        self.direction = direction
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_score: Optional[float] = None
        self.trials_history: List[Dict[str, Any]] = []
    
    @abstractmethod
    def optimize(
        self,
        objective_fn: Any,
        search_space: Dict[str, Any],
        n_trials: int = 100,
        **kwargs
    ) -> Tuple[Dict[str, Any], float]:
        """Optimize hyperparameters."""
        pass
    
    def save_results(self, filepath: Union[str, Path]) -> None:
        """Save optimization results."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        results = {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'trials_history': self.trials_history,
            'metric': self.metric,
            'direction': self.direction
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Optimization results saved to {filepath}")


class BayesianOptimizer(BaseOptimizer):
    """
    Bayesian optimization using Optuna.
    
    Uses Tree-structured Parzen Estimator (TPE) for efficient hyperparameter search.
    
    Example:
        >>> optimizer = BayesianOptimizer(metric='auc', direction='maximize')
        >>> search_space = {
        ...     'learning_rate': ('float', 0.01, 0.3),
        ...     'max_depth': ('int', 3, 10),
        ...     'n_estimators': ('categorical', [100, 200, 500])
        ... }
        >>> best_params, best_score = optimizer.optimize(objective_fn, search_space, n_trials=100)
    """
    
    def optimize(
        self,
        objective_fn: Any,
        search_space: Dict[str, Any],
        n_trials: int = 100,
        timeout: Optional[int] = None,
        n_jobs: int = 1,
        **kwargs
    ) -> Tuple[Dict[str, Any], float]:
        """
        Optimize using Optuna.
        
        Args:
            objective_fn: Objective function to minimize/maximize
            search_space: Dictionary defining parameter search space
            n_trials: Number of trials
            timeout: Timeout in seconds
            n_jobs: Number of parallel jobs
            
        Returns:
            Best parameters and best score
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna not available. Install with: pip install optuna")
        
        def optuna_objective(trial):
            # Sample parameters from search space
            params = {}
            for param_name, param_config in search_space.items():
                param_type = param_config[0]
                
                if param_type == 'float':
                    params[param_name] = trial.suggest_float(
                        param_name, param_config[1], param_config[2],
                        log=param_config[3] if len(param_config) > 3 else False
                    )
                elif param_type == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name, param_config[1], param_config[2]
                    )
                elif param_type == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_config[1]
                    )
            
            # Evaluate objective
            score = objective_fn(params)
            
            # Store trial
            self.trials_history.append({
                'params': params,
                'score': score,
                'trial_number': trial.number
            })
            
            return score
        
        logger.info(f"Starting Bayesian optimization with {n_trials} trials...")
        
        study = optuna.create_study(
            direction=self.direction,
            sampler=TPESampler(seed=42)
        )
        
        study.optimize(
            optuna_objective,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
            show_progress_bar=True
        )
        
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        logger.info(f"Optimization complete. Best score: {self.best_score:.6f}")
        logger.info(f"Best parameters: {self.best_params}")
        
        return self.best_params, self.best_score


class GeneticAlgorithmOptimizer(BaseOptimizer):
    """
    Genetic algorithm for hyperparameter optimization.
    
    Note: This is a placeholder implementation. For production use, consider
    using DEAP or other established GA libraries.
    
    Example:
        >>> optimizer = GeneticAlgorithmOptimizer(metric='auc')
        >>> best_params, best_score = optimizer.optimize(objective_fn, search_space)
    """
    
    def optimize(
        self,
        objective_fn: Any,
        search_space: Dict[str, Any],
        n_trials: int = 100,
        population_size: int = 20,
        **kwargs
    ) -> Tuple[Dict[str, Any], float]:
        """
        Optimize using genetic algorithm.
        
        Note: Currently falls back to random search. Implement proper GA for production.
        """
        logger.warning("Genetic algorithm not fully implemented. Using random search.")
        
        # Simple random search as placeholder
        best_score = float('-inf') if self.direction == 'maximize' else float('inf')
        best_params = None
        
        for trial in range(n_trials):
            # Random parameter sampling
            params = {}
            for param_name, param_config in search_space.items():
                param_type = param_config[0]
                
                if param_type == 'float':
                    params[param_name] = np.random.uniform(param_config[1], param_config[2])
                elif param_type == 'int':
                    params[param_name] = np.random.randint(param_config[1], param_config[2] + 1)
                elif param_type == 'categorical':
                    params[param_name] = np.random.choice(param_config[1])
            
            # Evaluate
            score = objective_fn(params)
            
            # Update best
            if self.direction == 'maximize':
                if score > best_score:
                    best_score = score
                    best_params = params
            else:
                if score < best_score:
                    best_score = score
                    best_params = params
            
            self.trials_history.append({
                'params': params,
                'score': score,
                'trial_number': trial
            })
        
        self.best_params = best_params
        self.best_score = best_score
        
        logger.info(f"Optimization complete. Best score: {self.best_score:.6f}")
        return self.best_params, self.best_score


class MultiObjectiveOptimizer:
    """
    Multi-objective optimization for balancing multiple metrics.
    
    Optimizes for AUC, stability, calibration, and business value simultaneously.
    
    Example:
        >>> objectives = {
        ...     'auc': {'weight': 0.4, 'direction': 'maximize'},
        ...     'stability': {'weight': 0.3, 'direction': 'maximize'},
        ...     'calibration': {'weight': 0.2, 'direction': 'minimize'},
        ...     'business_value': {'weight': 0.1, 'direction': 'maximize'}
        ... }
        >>> optimizer = MultiObjectiveOptimizer(objectives)
        >>> best_params = optimizer.optimize(objective_fn, search_space, n_trials=100)
    """
    
    def __init__(self, objectives: Dict[str, Dict[str, Any]]):
        """
        Initialize multi-objective optimizer.
        
        Args:
            objectives: Dictionary of objective names to configs with 'weight' and 'direction'
        """
        self.objectives = objectives
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_weighted_score: Optional[float] = None
        self.pareto_front: List[Dict[str, Any]] = []
    
    def optimize(
        self,
        objective_fn: Any,
        search_space: Dict[str, Any],
        n_trials: int = 100,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Optimize multiple objectives.
        
        Args:
            objective_fn: Function that returns dict of objective scores
            search_space: Parameter search space
            n_trials: Number of trials
            
        Returns:
            Best parameters based on weighted score
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna not available. Install with: pip install optuna")
        
        def optuna_objective(trial):
            # Sample parameters
            params = {}
            for param_name, param_config in search_space.items():
                param_type = param_config[0]
                
                if param_type == 'float':
                    params[param_name] = trial.suggest_float(
                        param_name, param_config[1], param_config[2],
                        log=param_config[3] if len(param_config) > 3 else False
                    )
                elif param_type == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name, param_config[1], param_config[2]
                    )
                elif param_type == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_config[1]
                    )
            
            # Evaluate all objectives
            scores = objective_fn(params)
            
            # Calculate weighted score
            weighted_score = 0.0
            for obj_name, obj_config in self.objectives.items():
                score = scores.get(obj_name, 0.0)
                weight = obj_config['weight']
                direction = obj_config['direction']
                
                # Normalize to maximization
                if direction == 'minimize':
                    score = -score
                
                weighted_score += weight * score
            
            # Store in Pareto front
            self.pareto_front.append({
                'params': params,
                'scores': scores,
                'weighted_score': weighted_score
            })
            
            return weighted_score
        
        logger.info(f"Starting multi-objective optimization with {n_trials} trials...")
        
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42)
        )
        
        study.optimize(optuna_objective, n_trials=n_trials, show_progress_bar=True)
        
        self.best_params = study.best_params
        self.best_weighted_score = study.best_value
        
        logger.info(f"Multi-objective optimization complete")
        logger.info(f"Best weighted score: {self.best_weighted_score:.6f}")
        logger.info(f"Best parameters: {self.best_params}")
        
        return self.best_params
    
    def get_pareto_front(self, top_k: int = 10) -> pd.DataFrame:
        """
        Get top solutions from Pareto front.
        
        Args:
            top_k: Number of top solutions to return
            
        Returns:
            DataFrame with top solutions
        """
        # Sort by weighted score
        sorted_solutions = sorted(
            self.pareto_front,
            key=lambda x: x['weighted_score'],
            reverse=True
        )[:top_k]
        
        # Convert to DataFrame
        records = []
        for sol in sorted_solutions:
            record = {**sol['scores'], **sol['params']}
            record['weighted_score'] = sol['weighted_score']
            records.append(record)
        
        return pd.DataFrame(records)


class HyperparameterSearchSpace:
    """
    Hyperparameter search space builder from config.
    
    Example:
        >>> search_space = HyperparameterSearchSpace.from_config(
        ...     'config/model_config.yaml', 'lightgbm'
        ... )
        >>> print(search_space)
    """
    
    @staticmethod
    def from_config(
        config_path: Union[str, Path],
        model_name: str
    ) -> Dict[str, Any]:
        """
        Load search space from config file.
        
        Args:
            config_path: Path to config YAML
            model_name: Model name in config
            
        Returns:
            Search space dictionary
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if model_name not in config:
            raise ValueError(f"Model '{model_name}' not found in config")
        
        search_space_config = config[model_name].get('search_space', {})
        search_space = {}
        
        for param_name, param_values in search_space_config.items():
            if isinstance(param_values, list):
                if all(isinstance(v, (int, float)) for v in param_values):
                    # Numeric range
                    if all(isinstance(v, int) for v in param_values):
                        search_space[param_name] = ('int', min(param_values), max(param_values))
                    else:
                        search_space[param_name] = ('float', min(param_values), max(param_values))
                else:
                    # Categorical
                    search_space[param_name] = ('categorical', param_values)
        
        return search_space
    
    @staticmethod
    def get_default_params(
        config_path: Union[str, Path],
        model_name: str
    ) -> Dict[str, Any]:
        """Load default parameters from config."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config.get(model_name, {}).get('default', {})


class AutoMLEngine:
    """
    Automated machine learning engine for model selection and tuning.
    
    Automatically:
    - Tries multiple model types
    - Tunes hyperparameters
    - Performs cross-validation
    - Tracks and compares results
    - Selects best model
    
    Example:
        >>> automl = AutoMLEngine(config_path='config/model_config.yaml')
        >>> best_model = automl.fit(
        ...     X_train, y_train, X_val, y_val,
        ...     models=['lightgbm', 'xgboost', 'catboost'],
        ...     n_trials=50
        ... )
        >>> predictions = best_model.predict_proba(X_test)
    """
    
    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        task: str = 'classification',
        metric: str = 'auc',
        cv_folds: int = 5,
        random_state: int = 42
    ):
        """
        Initialize AutoML engine.
        
        Args:
            config_path: Path to model config
            task: 'classification' or 'regression'
            metric: Optimization metric
            cv_folds: Cross-validation folds
            random_state: Random seed
        """
        self.config_path = config_path
        self.task = task
        self.metric = metric
        self.cv_folds = cv_folds
        self.random_state = random_state
        
        self.best_model = None
        self.best_model_name = None
        self.best_score = None
        self.model_results: List[Dict[str, Any]] = []
    
    def fit(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None,
        models: Optional[List[str]] = None,
        n_trials: int = 50,
        timeout_per_model: Optional[int] = None
    ) -> Any:
        """
        Fit AutoML and find best model.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            models: List of model names to try
            n_trials: Trials per model
            timeout_per_model: Timeout per model in seconds
            
        Returns:
            Best fitted model
        """
        if models is None:
            models = ['lightgbm', 'xgboost', 'random_forest']
        
        logger.info(f"Starting AutoML with {len(models)} models and {n_trials} trials each")
        
        best_score = float('-inf')
        best_model = None
        best_model_name = None
        
        for model_name in models:
            logger.info(f"\n{'='*60}")
            logger.info(f"Optimizing {model_name}...")
            logger.info(f"{'='*60}")
            
            try:
                # Get search space
                search_space = HyperparameterSearchSpace.from_config(
                    self.config_path, model_name
                )
                
                # Define objective function
                def objective(params):
                    return self._train_and_evaluate(
                        model_name, params, X_train, y_train, X_val, y_val
                    )
                
                # Optimize
                optimizer = BayesianOptimizer(metric=self.metric, direction='maximize')
                best_params, score = optimizer.optimize(
                    objective, search_space, n_trials=n_trials, timeout=timeout_per_model
                )
                
                # Train final model with best params
                final_model = self._train_model(model_name, best_params, X_train, y_train, X_val, y_val)
                
                # Store results
                self.model_results.append({
                    'model_name': model_name,
                    'best_params': best_params,
                    'score': score,
                    'model': final_model
                })
                
                # Update best
                if score > best_score:
                    best_score = score
                    best_model = final_model
                    best_model_name = model_name
                
                logger.info(f"{model_name} best score: {score:.6f}")
                
            except Exception as e:
                logger.error(f"Error optimizing {model_name}: {e}")
                continue
        
        self.best_model = best_model
        self.best_model_name = best_model_name
        self.best_score = best_score
        
        logger.info(f"\n{'='*60}")
        logger.info(f"AutoML complete! Best model: {best_model_name} (score: {best_score:.6f})")
        logger.info(f"{'='*60}")
        
        return best_model
    
    def _train_and_evaluate(
        self,
        model_name: str,
        params: Dict[str, Any],
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_val: Optional[Union[pd.DataFrame, np.ndarray]],
        y_val: Optional[Union[pd.Series, np.ndarray]]
    ) -> float:
        """Train model and evaluate performance."""
        try:
            model = self._train_model(model_name, params, X_train, y_train, X_val, y_val)
            
            # Evaluate
            if X_val is not None and y_val is not None:
                if self.task == 'classification':
                    y_pred_proba = model.predict_proba(X_val)
                    if y_pred_proba.ndim == 2:
                        y_pred_proba = y_pred_proba[:, 1]
                    score = roc_auc_score(y_val, y_pred_proba)
                else:
                    y_pred = model.predict(X_val)
                    score = -mean_squared_error(y_val, y_pred)
            else:
                # Use cross-validation
                if self.task == 'classification':
                    kf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
                else:
                    kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
                
                scores = cross_val_score(
                    model.model if hasattr(model, 'model') else model,
                    X_train, y_train, cv=kf, scoring='roc_auc' if self.task == 'classification' else 'neg_mean_squared_error'
                )
                score = np.mean(scores)
            
            return score
            
        except Exception as e:
            logger.warning(f"Error in training: {e}")
            return float('-inf')
    
    def _train_model(
        self,
        model_name: str,
        params: Dict[str, Any],
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_val: Optional[Union[pd.DataFrame, np.ndarray]],
        y_val: Optional[Union[pd.Series, np.ndarray]]
    ) -> Any:
        """Train a specific model."""
        # Import models
        from src.models.tree_models import (
            LightGBMModel, XGBoostModel, CatBoostModel, RandomForestModel
        )
        
        # Create model
        if model_name == 'lightgbm':
            model = LightGBMModel(params=params, task=self.task)
        elif model_name == 'xgboost':
            model = XGBoostModel(params=params, task=self.task)
        elif model_name == 'catboost':
            model = CatBoostModel(params=params, task=self.task)
        elif model_name == 'random_forest':
            model = RandomForestModel(params=params, task=self.task)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Train
        model.fit(X_train, y_train, X_val, y_val, verbose=0)
        
        return model
    
    def get_leaderboard(self) -> pd.DataFrame:
        """
        Get leaderboard of all models tried.
        
        Returns:
            DataFrame sorted by score
        """
        records = []
        for result in self.model_results:
            records.append({
                'model_name': result['model_name'],
                'score': result['score'],
                'best_params': str(result['best_params'])
            })
        
        df = pd.DataFrame(records).sort_values('score', ascending=False)
        return df
    
    def save_results(self, filepath: Union[str, Path]) -> None:
        """Save AutoML results."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save best model
        if self.best_model is not None:
            model_path = filepath.parent / f"{filepath.stem}_best_model.pkl"
            self.best_model.save(model_path)
        
        # Save leaderboard
        leaderboard = self.get_leaderboard()
        leaderboard.to_csv(filepath.parent / f"{filepath.stem}_leaderboard.csv", index=False)
        
        # Save metadata
        metadata = {
            'best_model_name': self.best_model_name,
            'best_score': self.best_score,
            'task': self.task,
            'metric': self.metric,
            'cv_folds': self.cv_folds
        }
        
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"AutoML results saved to {filepath.parent}")
