"""
Collections NBA End-to-End Pipeline

Orchestrates propensity modeling, payment estimation, treatment optimization,
and channel selection for comprehensive Next Best Action recommendations.

Example:
    >>> from src.use_cases.collections_nba.nba_pipeline import CollectionsNBAPipeline
    >>> pipeline = CollectionsNBAPipeline(config_path='config/model_config.yaml')
    >>> pipeline.fit(training_data)
    >>> recommendations = pipeline.predict(new_accounts)
"""

import joblib
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime

from loguru import logger

from .propensity_model import PropensityModel
from .payment_estimator import PaymentEstimator
from .treatment_optimizer import TreatmentOptimizer
from .channel_optimizer import ChannelOptimizer


class CollectionsNBAConfig:
    """Configuration for Collections NBA pipeline."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize NBA pipeline configuration."""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                full_config = yaml.safe_load(f)
                nba_config = full_config.get('collections_nba', {})
        else:
            nba_config = {}
        
        self.primary_horizon = nba_config.get('primary_horizon', 30)
        self.min_expected_value = nba_config.get('min_expected_value', 10.0)
        self.enable_treatment_opt = nba_config.get('enable_treatment_opt', True)
        self.enable_channel_opt = nba_config.get('enable_channel_opt', True)
        self.simulation_iterations = nba_config.get('simulation_iterations', 1000)


class CollectionsNBAPipeline:
    """
    End-to-end Collections NBA pipeline.
    
    Orchestrates:
    1. Propensity modeling (multi-horizon)
    2. Payment amount estimation
    3. Treatment optimization
    4. Channel and timing optimization
    5. Final NBA recommendation
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        propensity_model: Optional[PropensityModel] = None,
        payment_estimator: Optional[PaymentEstimator] = None,
        treatment_optimizer: Optional[TreatmentOptimizer] = None,
        channel_optimizer: Optional[ChannelOptimizer] = None
    ):
        """
        Initialize Collections NBA pipeline.
        
        Args:
            config_path: Path to model_config.yaml
            propensity_model: Pre-trained propensity model (optional)
            payment_estimator: Pre-trained payment estimator (optional)
            treatment_optimizer: Pre-trained treatment optimizer (optional)
            channel_optimizer: Pre-trained channel optimizer (optional)
            
        Example:
            >>> pipeline = CollectionsNBAPipeline(config_path="config/model_config.yaml")
        """
        self.config = CollectionsNBAConfig(config_path)
        self.config_path = config_path
        
        # Initialize or use provided models
        self.propensity_model = propensity_model or PropensityModel(config_path=config_path)
        self.payment_estimator = payment_estimator or PaymentEstimator(config_path=config_path)
        self.treatment_optimizer = treatment_optimizer or TreatmentOptimizer(config_path=config_path)
        self.channel_optimizer = channel_optimizer or ChannelOptimizer(config_path=config_path)
        
        self.is_fitted = False
        
        logger.info("Initialized CollectionsNBAPipeline")
    
    def fit(
        self,
        training_data: Dict[str, pd.DataFrame],
        validation_data: Optional[Dict[str, pd.DataFrame]] = None
    ) -> 'CollectionsNBAPipeline':
        """
        Fit all pipeline components.
        
        Args:
            training_data: Dictionary with keys:
                - 'features': Feature DataFrame
                - 'propensity_targets': Dict of {horizon: target_series}
                - 'payment_amounts': Payment amount series
                - 'segments': Customer segments series
                - 'treatments': Applied treatment series
                - 'treatment_outcomes': Treatment outcome DataFrame
                - 'recovery_amounts': Recovery amount series
                - 'channels': Applied channel series
                - 'channel_responses': Channel response DataFrame
            validation_data: Same structure for validation
            
        Returns:
            Self for method chaining
            
        Example:
            >>> training_data = {
            ...     'features': X_train,
            ...     'propensity_targets': {7: y_7d, 14: y_14d, 30: y_30d},
            ...     'payment_amounts': amounts_train,
            ...     'segments': segments_train,
            ...     'treatments': treatments_train,
            ...     'treatment_outcomes': outcomes_train,
            ...     'channels': channels_train,
            ...     'channel_responses': responses_train
            ... }
            >>> pipeline.fit(training_data, validation_data)
        """
        logger.info("=" * 80)
        logger.info("Starting Collections NBA Pipeline Training")
        logger.info("=" * 80)
        
        X_train = training_data['features']
        X_val = validation_data['features'] if validation_data else None
        
        # 1. Train Propensity Model
        logger.info("\n[1/4] Training Propensity Model...")
        propensity_targets = training_data['propensity_targets']
        propensity_targets_val = validation_data.get('propensity_targets') if validation_data else None
        segments = training_data.get('segments')
        segments_val = validation_data.get('segments') if validation_data else None
        
        self.propensity_model.fit(
            X_train,
            propensity_targets,
            X_val,
            propensity_targets_val,
            segments
        )
        
        # Generate propensity scores for training
        propensity_train = self.propensity_model.predict_proba(
            X_train,
            horizon=self.config.primary_horizon,
            segments=segments
        )
        
        # 2. Train Payment Estimator
        logger.info("\n[2/4] Training Payment Estimator...")
        payment_amounts = training_data['payment_amounts']
        payment_amounts_val = validation_data.get('payment_amounts') if validation_data else None
        
        propensity_val = None
        if X_val is not None and propensity_targets_val:
            propensity_val = self.propensity_model.predict_proba(
                X_val,
                horizon=self.config.primary_horizon,
                segments=segments_val
            )
        
        self.payment_estimator.fit(
            X_train,
            payment_amounts,
            pd.Series(propensity_train, index=X_train.index),
            X_val,
            payment_amounts_val,
            pd.Series(propensity_val, index=X_val.index) if propensity_val is not None else None
        )
        
        # 3. Train Treatment Optimizer
        if self.config.enable_treatment_opt and 'treatments' in training_data:
            logger.info("\n[3/4] Training Treatment Optimizer...")
            
            self.treatment_optimizer.fit(
                X_train,
                training_data['treatment_outcomes'],
                training_data['treatments'],
                training_data.get('recovery_amounts', payment_amounts),
                X_val,
                validation_data.get('treatment_outcomes') if validation_data else None,
                validation_data.get('treatments') if validation_data else None
            )
        else:
            logger.info("\n[3/4] Skipping Treatment Optimizer (disabled or missing data)")
        
        # 4. Train Channel Optimizer
        if self.config.enable_channel_opt and 'channels' in training_data:
            logger.info("\n[4/4] Training Channel Optimizer...")
            
            self.channel_optimizer.fit(
                X_train,
                training_data['channel_responses'],
                training_data['channels'],
                timing_features=training_data.get('timing_features'),
                X_val=X_val,
                response_val=validation_data.get('channel_responses') if validation_data else None,
                channels_val=validation_data.get('channels') if validation_data else None
            )
        else:
            logger.info("\n[4/4] Skipping Channel Optimizer (disabled or missing data)")
        
        self.is_fitted = True
        
        logger.info("\n" + "=" * 80)
        logger.info("Collections NBA Pipeline Training Complete")
        logger.info("=" * 80)
        
        return self
    
    def predict(
        self,
        X: pd.DataFrame,
        outstanding_amounts: pd.Series,
        segments: Optional[pd.Series] = None,
        account_ages: Optional[pd.Series] = None,
        current_datetime: Optional[datetime] = None,
        return_details: bool = False
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]]:
        """
        Generate NBA recommendations for accounts.
        
        Args:
            X: Features
            outstanding_amounts: Outstanding debt amounts
            segments: Customer segments
            account_ages: Account age in days
            current_datetime: Current datetime for timing optimization
            return_details: Whether to return detailed predictions from each component
            
        Returns:
            If return_details=False: DataFrame with final recommendations
            If return_details=True: Tuple of (recommendations, details_dict)
            
        Example:
            >>> recommendations = pipeline.predict(
            ...     X_test, outstanding_test, segments_test, ages_test
            ... )
        """
        if not self.is_fitted:
            raise ValueError("Pipeline not fitted. Call fit() first.")
        
        logger.info(f"Generating NBA recommendations for {len(X)} accounts")
        
        details = {}
        
        # 1. Get propensity predictions
        logger.info("Step 1: Computing propensity scores...")
        propensity_multi = self.propensity_model.predict_multi_horizon(X, segments)
        propensity_primary = propensity_multi[f'prob_{self.config.primary_horizon}d']
        details['propensity'] = propensity_multi
        
        # 2. Get payment amount predictions
        logger.info("Step 2: Estimating payment amounts...")
        payment_pred, payment_lower, payment_upper = self.payment_estimator.predict(
            X,
            pd.Series(propensity_primary, index=X.index),
            return_intervals=True,
            confidence_level=0.8
        )
        
        details['payment'] = pd.DataFrame({
            'predicted_amount': payment_pred,
            'lower_bound': payment_lower,
            'upper_bound': payment_upper
        }, index=X.index)
        
        # 3. Get treatment recommendations
        if self.config.enable_treatment_opt and self.treatment_optimizer.is_fitted:
            logger.info("Step 3: Optimizing treatment selection...")
            treatment_rec = self.treatment_optimizer.recommend(
                X,
                outstanding_amounts,
                account_ages
            )
            details['treatment'] = treatment_rec
        else:
            treatment_rec = pd.DataFrame({
                'treatment': 'SelfCure',
                'expected_value': payment_pred
            }, index=X.index)
        
        # 4. Get channel recommendations
        if self.config.enable_channel_opt and self.channel_optimizer.is_fitted:
            logger.info("Step 4: Optimizing channel and timing...")
            channel_rec = self.channel_optimizer.recommend(
                X,
                pd.Series(propensity_primary, index=X.index),
                pd.Series(payment_pred, index=X.index),
                current_datetime
            )
            details['channel'] = channel_rec
        else:
            channel_rec = pd.DataFrame({
                'channel': 'Email',
                'expected_value': payment_pred
            }, index=X.index)
        
        # 5. Combine into final recommendations
        logger.info("Step 5: Generating final recommendations...")
        recommendations = pd.DataFrame(index=X.index)
        
        # Core predictions
        recommendations['propensity_to_pay'] = propensity_primary
        recommendations['expected_payment'] = payment_pred
        recommendations['payment_lower_bound'] = payment_lower
        recommendations['payment_upper_bound'] = payment_upper
        
        # Treatment
        recommendations['recommended_treatment'] = treatment_rec['treatment']
        recommendations['treatment_expected_value'] = treatment_rec['expected_value']
        
        # Channel
        recommendations['recommended_channel'] = channel_rec['channel']
        recommendations['channel_expected_value'] = channel_rec['expected_value']
        if 'recommended_hour' in channel_rec.columns:
            recommendations['recommended_hour'] = channel_rec['recommended_hour']
            recommendations['recommended_day'] = channel_rec['recommended_day']
        
        # Overall expected value (simplified combination)
        recommendations['total_expected_value'] = (
            propensity_primary * payment_pred * 
            treatment_rec.get('expected_value', 1.0) * 
            channel_rec.get('expected_value', 1.0)
        )
        
        # Priority ranking
        recommendations['priority_rank'] = recommendations['total_expected_value'].rank(
            ascending=False, method='first'
        ).astype(int)
        
        # Action flag (whether to take action)
        recommendations['take_action'] = (
            recommendations['total_expected_value'] >= self.config.min_expected_value
        )
        
        # Add multi-horizon propensities
        for col in propensity_multi.columns:
            if col != f'prob_{self.config.primary_horizon}d':
                recommendations[col] = propensity_multi[col]
        
        logger.info(
            f"Generated recommendations: {recommendations['take_action'].sum()} "
            f"accounts flagged for action"
        )
        
        if return_details:
            return recommendations, details
        return recommendations
    
    def backtest(
        self,
        test_data: Dict[str, pd.DataFrame],
        actual_outcomes: pd.Series
    ) -> pd.DataFrame:
        """
        Backtest pipeline performance.
        
        Args:
            test_data: Same structure as training_data
            actual_outcomes: Actual payment outcomes
            
        Returns:
            DataFrame with backtest metrics
            
        Example:
            >>> backtest_results = pipeline.backtest(test_data, actual_payments)
        """
        logger.info("Running backtest...")
        
        X_test = test_data['features']
        outstanding = test_data.get('outstanding_amounts', pd.Series(1000, index=X_test.index))
        segments = test_data.get('segments')
        
        # Get recommendations
        recommendations = self.predict(X_test, outstanding, segments)
        
        # Calculate metrics
        results = []
        
        # Overall metrics
        take_action_mask = recommendations['take_action']
        
        if take_action_mask.any():
            precision = actual_outcomes[take_action_mask].mean()
            total_recovery = actual_outcomes[take_action_mask].sum()
            avg_recovery = actual_outcomes[take_action_mask].mean()
        else:
            precision = 0
            total_recovery = 0
            avg_recovery = 0
        
        results.append({
            'segment': 'All',
            'count': len(X_test),
            'actions_recommended': take_action_mask.sum(),
            'precision': precision,
            'total_recovery': total_recovery,
            'avg_recovery': avg_recovery,
            'avg_expected_value': recommendations['total_expected_value'].mean()
        })
        
        # Per-segment metrics
        if segments is not None:
            for segment in segments.unique():
                seg_mask = segments == segment
                seg_rec = recommendations[seg_mask]
                seg_outcomes = actual_outcomes[seg_mask]
                seg_actions = seg_rec['take_action']
                
                if seg_actions.any():
                    seg_precision = seg_outcomes[seg_actions].mean()
                    seg_total = seg_outcomes[seg_actions].sum()
                    seg_avg = seg_outcomes[seg_actions].mean()
                else:
                    seg_precision = 0
                    seg_total = 0
                    seg_avg = 0
                
                results.append({
                    'segment': segment,
                    'count': seg_mask.sum(),
                    'actions_recommended': seg_actions.sum(),
                    'precision': seg_precision,
                    'total_recovery': seg_total,
                    'avg_recovery': seg_avg,
                    'avg_expected_value': seg_rec['total_expected_value'].mean()
                })
        
        return pd.DataFrame(results)
    
    def simulate(
        self,
        X: pd.DataFrame,
        outstanding_amounts: pd.Series,
        segments: Optional[pd.Series] = None,
        n_iterations: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Simulate NBA outcomes using Monte Carlo.
        
        Args:
            X: Features
            outstanding_amounts: Outstanding amounts
            segments: Customer segments
            n_iterations: Number of simulation iterations
            
        Returns:
            DataFrame with simulation statistics
            
        Example:
            >>> simulation = pipeline.simulate(X_test, outstanding_test, segments_test)
        """
        if n_iterations is None:
            n_iterations = self.config.simulation_iterations
        
        logger.info(f"Running {n_iterations} simulation iterations...")
        
        # Get base recommendations
        recommendations = self.predict(X, outstanding_amounts, segments)
        
        # Extract distributions
        propensity = recommendations['propensity_to_pay'].values
        payment_mean = recommendations['expected_payment'].values
        payment_lower = recommendations['payment_lower_bound'].values
        payment_upper = recommendations['payment_upper_bound'].values
        
        # Estimate payment std from confidence interval
        payment_std = (payment_upper - payment_lower) / 3.29  # Approximate for 80% CI
        
        # Run simulations
        simulated_recoveries = np.zeros((len(X), n_iterations))
        
        for i in range(n_iterations):
            # Simulate payment events
            pays = np.random.random(len(X)) < propensity
            
            # Simulate payment amounts (log-normal distribution)
            amounts = np.random.lognormal(
                mean=np.log(payment_mean + 1),
                sigma=np.log(payment_std + 1)
            )
            
            simulated_recoveries[:, i] = pays * amounts
        
        # Calculate statistics
        result = pd.DataFrame(index=X.index)
        result['mean_recovery'] = simulated_recoveries.mean(axis=1)
        result['median_recovery'] = np.median(simulated_recoveries, axis=1)
        result['std_recovery'] = simulated_recoveries.std(axis=1)
        result['p10_recovery'] = np.percentile(simulated_recoveries, 10, axis=1)
        result['p90_recovery'] = np.percentile(simulated_recoveries, 90, axis=1)
        result['prob_zero_recovery'] = (simulated_recoveries == 0).mean(axis=1)
        
        logger.info(
            f"Simulation complete: Mean recovery = {result['mean_recovery'].sum():.2f}, "
            f"Median = {result['median_recovery'].sum():.2f}"
        )
        
        return result
    
    def generate_report(
        self,
        recommendations: pd.DataFrame,
        segments: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Generate summary report from recommendations.
        
        Args:
            recommendations: Output from predict()
            segments: Customer segments
            
        Returns:
            Dictionary with report statistics
            
        Example:
            >>> report = pipeline.generate_report(recommendations, segments)
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_accounts': len(recommendations),
            'actions_recommended': recommendations['take_action'].sum(),
            'action_rate': recommendations['take_action'].mean(),
            'total_expected_value': recommendations['total_expected_value'].sum(),
            'avg_expected_value': recommendations['total_expected_value'].mean(),
            'median_propensity': recommendations['propensity_to_pay'].median(),
            'median_payment': recommendations['expected_payment'].median(),
        }
        
        # Treatment distribution
        treatment_dist = recommendations['recommended_treatment'].value_counts()
        report['treatment_distribution'] = treatment_dist.to_dict()
        
        # Channel distribution
        channel_dist = recommendations['recommended_channel'].value_counts()
        report['channel_distribution'] = channel_dist.to_dict()
        
        # Segment analysis
        if segments is not None:
            segment_stats = []
            for segment in segments.unique():
                seg_mask = segments == segment
                seg_rec = recommendations[seg_mask]
                
                segment_stats.append({
                    'segment': segment,
                    'count': seg_mask.sum(),
                    'action_rate': seg_rec['take_action'].mean(),
                    'avg_expected_value': seg_rec['total_expected_value'].mean(),
                    'avg_propensity': seg_rec['propensity_to_pay'].mean(),
                })
            
            report['segment_analysis'] = segment_stats
        
        return report
    
    def save(self, dirpath: Union[str, Path]) -> None:
        """
        Save pipeline to disk.
        
        Args:
            dirpath: Directory to save pipeline components
            
        Example:
            >>> pipeline.save("models/nba_pipeline/")
        """
        dirpath = Path(dirpath)
        dirpath.mkdir(parents=True, exist_ok=True)
        
        # Save each component
        self.propensity_model.save(dirpath / "propensity_model.pkl")
        self.payment_estimator.save(dirpath / "payment_estimator.pkl")
        
        if self.treatment_optimizer.is_fitted:
            self.treatment_optimizer.save(dirpath / "treatment_optimizer.pkl")
        
        if self.channel_optimizer.is_fitted:
            self.channel_optimizer.save(dirpath / "channel_optimizer.pkl")
        
        # Save pipeline metadata
        metadata = {
            'config': self.config,
            'is_fitted': self.is_fitted,
            'config_path': str(self.config_path) if self.config_path else None
        }
        
        joblib.dump(metadata, dirpath / "pipeline_metadata.pkl")
        
        logger.info(f"Pipeline saved to {dirpath}")
    
    @classmethod
    def load(cls, dirpath: Union[str, Path]) -> 'CollectionsNBAPipeline':
        """
        Load pipeline from disk.
        
        Args:
            dirpath: Directory containing saved pipeline
            
        Returns:
            Loaded CollectionsNBAPipeline instance
            
        Example:
            >>> pipeline = CollectionsNBAPipeline.load("models/nba_pipeline/")
        """
        dirpath = Path(dirpath)
        
        # Load components
        propensity_model = PropensityModel.load(dirpath / "propensity_model.pkl")
        payment_estimator = PaymentEstimator.load(dirpath / "payment_estimator.pkl")
        
        treatment_path = dirpath / "treatment_optimizer.pkl"
        treatment_optimizer = TreatmentOptimizer.load(treatment_path) if treatment_path.exists() else None
        
        channel_path = dirpath / "channel_optimizer.pkl"
        channel_optimizer = ChannelOptimizer.load(channel_path) if channel_path.exists() else None
        
        # Load metadata
        metadata = joblib.load(dirpath / "pipeline_metadata.pkl")
        
        # Create instance
        instance = cls(
            config_path=metadata.get('config_path'),
            propensity_model=propensity_model,
            payment_estimator=payment_estimator,
            treatment_optimizer=treatment_optimizer,
            channel_optimizer=channel_optimizer
        )
        
        instance.config = metadata['config']
        instance.is_fitted = metadata['is_fitted']
        
        logger.info(f"Pipeline loaded from {dirpath}")
        return instance
