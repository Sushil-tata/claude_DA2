"""
Collections NBA Channel Optimizer

Optimizes communication channel, offer, and timing for maximum response rate
with contact strategy sequencing.

Example:
    >>> from src.use_cases.collections_nba.channel_optimizer import ChannelOptimizer
    >>> optimizer = ChannelOptimizer(config_path='config/model_config.yaml')
    >>> optimizer.fit(X_train, response_train, channels_train)
    >>> recommendations = optimizer.recommend(X_test, propensity_test)
"""

import joblib
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, time

from loguru import logger
from sklearn.metrics import roc_auc_score

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    logger.warning("LightGBM not available. Install with: pip install lightgbm")
    LIGHTGBM_AVAILABLE = False


class ChannelOptimizerConfig:
    """Configuration for channel optimizer."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize channel optimizer configuration."""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                full_config = yaml.safe_load(f)
                channel_config = full_config.get('channel_optimizer', {})
        else:
            channel_config = {}
        
        self.channels = channel_config.get('channels', [
            'SMS', 'Email', 'Phone', 'App', 'Letter'
        ])
        
        self.channel_costs = channel_config.get('channel_costs', {
            'SMS': 0.05, 'Email': 0.01, 'Phone': 2.0, 'App': 0.0, 'Letter': 1.5
        })
        
        self.channel_capacity = channel_config.get('channel_capacity', {
            'SMS': 1.0, 'Email': 1.0, 'Phone': 0.2, 'App': 1.0, 'Letter': 0.5
        })
        
        # Optimal timing windows (hour of day)
        self.timing_windows = channel_config.get('timing_windows', {
            'SMS': [(9, 12), (18, 20)],
            'Email': [(8, 11), (14, 17)],
            'Phone': [(10, 12), (15, 17)],
            'App': [(0, 24)],  # Anytime
            'Letter': [(0, 24)]  # N/A
        })
        
        # Day of week preferences (0=Monday, 6=Sunday)
        self.day_preferences = channel_config.get('day_preferences', {
            'SMS': [1, 2, 3, 4],  # Tue-Fri
            'Email': [0, 1, 2, 3],  # Mon-Thu
            'Phone': [1, 2, 3],  # Tue-Thu
            'App': list(range(7)),
            'Letter': list(range(7))
        })
        
        # Contact frequency limits
        self.max_contacts_per_day = channel_config.get('max_contacts_per_day', 2)
        self.max_contacts_per_week = channel_config.get('max_contacts_per_week', 5)
        self.min_hours_between_contacts = channel_config.get('min_hours_between_contacts', 4)
        
        self.model_params = channel_config.get('model_params', {
            'objective': 'binary',
            'metric': 'auc',
            'boosting': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'verbose': -1,
        })


class ChannelOptimizer:
    """
    Channel and timing optimizer for collections NBA.
    
    Optimizes:
    - Communication channel selection
    - Message timing (hour, day of week)
    - Offer structure and amount
    - Contact sequencing strategy
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize channel optimizer.
        
        Args:
            config_path: Path to model_config.yaml
            
        Example:
            >>> optimizer = ChannelOptimizer(config_path="config/model_config.yaml")
        """
        self.config = ChannelOptimizerConfig(config_path)
        
        # Models for each channel's response rate
        self.channel_models: Dict[str, Any] = {}
        
        # Timing models (hour of day, day of week)
        self.timing_models: Dict[str, Any] = {}
        
        # Offer response models
        self.offer_model: Optional[Any] = None
        
        self.feature_names: Optional[List[str]] = None
        self.is_fitted = False
        
        logger.info(f"Initialized ChannelOptimizer with channels: {self.config.channels}")
    
    def fit(
        self,
        X: pd.DataFrame,
        response: pd.DataFrame,
        channels: pd.Series,
        timing_features: Optional[pd.DataFrame] = None,
        X_val: Optional[pd.DataFrame] = None,
        response_val: Optional[pd.DataFrame] = None,
        channels_val: Optional[pd.Series] = None
    ) -> 'ChannelOptimizer':
        """
        Fit channel response models.
        
        Args:
            X: Training features
            response: Binary response for each channel (columns = channels)
            channels: Actual channels used
            timing_features: Hour of day, day of week features
            X_val: Validation features
            response_val: Validation response
            channels_val: Validation channels
            
        Returns:
            Self for method chaining
            
        Example:
            >>> optimizer.fit(X_train, response_train, channels_train, timing_train)
        """
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is required. Install with: pip install lightgbm")
        
        self.feature_names = list(X.columns)
        
        # Add timing features if provided
        if timing_features is not None:
            X = pd.concat([X, timing_features], axis=1)
            if X_val is not None:
                X_val = pd.concat([X_val, timing_features], axis=1)
        
        # Train channel-specific response models
        logger.info(f"Training response models for {len(self.config.channels)} channels")
        
        for channel in self.config.channels:
            if channel not in response.columns:
                logger.warning(f"Channel {channel} not in response data, skipping")
                continue
            
            # Filter to samples where this channel was used
            channel_mask = channels == channel
            X_channel = X[channel_mask]
            y_channel = response[channel][channel_mask]
            
            if len(X_channel) < 50:
                logger.warning(f"Channel {channel} has only {len(X_channel)} samples, skipping")
                continue
            
            logger.info(
                f"Training model for {channel}: {len(X_channel)} samples, "
                f"{y_channel.mean():.2%} response rate"
            )
            
            model = lgb.LGBMClassifier(**self.config.model_params)
            
            # Validation set for this channel
            if X_val is not None and channels_val is not None:
                val_mask = channels_val == channel
                X_val_channel = X_val[val_mask]
                y_val_channel = response_val[channel][val_mask] if response_val is not None else None
                eval_set = [(X_val_channel, y_val_channel)] if len(X_val_channel) > 0 and y_val_channel is not None else None
            else:
                eval_set = None
            
            model.fit(
                X_channel, y_channel,
                eval_set=eval_set,
                callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)] if eval_set else None
            )
            
            self.channel_models[channel] = model
            
            # Log validation metrics
            if eval_set and len(X_val_channel) > 0:
                val_pred = model.predict_proba(X_val_channel)[:, 1]
                val_auc = roc_auc_score(y_val_channel, val_pred)
                logger.info(f"Channel {channel} validation AUC: {val_auc:.4f}")
        
        self.is_fitted = True
        logger.info("Channel optimizer training complete")
        return self
    
    def predict_channel_response(
        self,
        X: pd.DataFrame,
        timing_features: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Predict response probability for each channel.
        
        Args:
            X: Features
            timing_features: Hour of day, day of week features
            
        Returns:
            DataFrame with columns for each channel's response probability
            
        Example:
            >>> probs = optimizer.predict_channel_response(X_test, timing_test)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Add timing features if provided
        if timing_features is not None:
            X = pd.concat([X, timing_features], axis=1)
        
        result = pd.DataFrame(index=X.index)
        
        for channel in self.config.channels:
            if channel in self.channel_models:
                probs = self.channel_models[channel].predict_proba(X)[:, 1]
                result[f'{channel}_prob'] = probs
        
        return result
    
    def calculate_channel_value(
        self,
        X: pd.DataFrame,
        propensity_scores: pd.Series,
        payment_amounts: pd.Series,
        timing_features: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Calculate expected value for each channel.
        
        Args:
            X: Features
            propensity_scores: Probability of payment
            payment_amounts: Expected payment amounts
            timing_features: Timing features
            
        Returns:
            DataFrame with expected values for each channel
            
        Example:
            >>> values = optimizer.calculate_channel_value(X_test, propensity, amounts)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Get channel response probabilities
        response_probs = self.predict_channel_response(X, timing_features)
        
        result = pd.DataFrame(index=X.index)
        
        for channel in self.config.channels:
            prob_col = f'{channel}_prob'
            if prob_col not in response_probs.columns:
                continue
            
            # Expected value = response_prob * propensity * payment_amount - cost
            channel_cost = self.config.channel_costs.get(channel, 0)
            
            expected_value = (
                response_probs[prob_col] * 
                propensity_scores * 
                payment_amounts - 
                channel_cost
            )
            
            result[f'{channel}_value'] = expected_value
        
        return result
    
    def recommend(
        self,
        X: pd.DataFrame,
        propensity_scores: pd.Series,
        payment_amounts: pd.Series,
        current_datetime: Optional[datetime] = None,
        contact_history: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Recommend optimal channel and timing for each account.
        
        Args:
            X: Features
            propensity_scores: Probability of payment
            payment_amounts: Expected payment amounts
            current_datetime: Current date/time for timing optimization
            contact_history: Recent contact history (columns: user_id, timestamp, channel)
            
        Returns:
            DataFrame with channel, timing, and value recommendations
            
        Example:
            >>> recommendations = optimizer.recommend(
            ...     X_test, propensity_test, amounts_test,
            ...     current_datetime=datetime.now(),
            ...     contact_history=recent_contacts
            ... )
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if current_datetime is None:
            current_datetime = datetime.now()
        
        # Create timing features
        hour_of_day = current_datetime.hour
        day_of_week = current_datetime.weekday()
        
        timing_features = pd.DataFrame({
            'hour_of_day': hour_of_day,
            'day_of_week': day_of_week,
            'is_weekend': int(day_of_week >= 5),
            'is_business_hours': int(9 <= hour_of_day <= 17)
        }, index=X.index)
        
        # Calculate channel values
        channel_values = self.calculate_channel_value(
            X, propensity_scores, payment_amounts, timing_features
        )
        
        # Apply timing constraints
        for channel in self.config.channels:
            value_col = f'{channel}_value'
            if value_col not in channel_values.columns:
                continue
            
            # Check if current time is optimal for this channel
            timing_windows = self.config.timing_windows.get(channel, [(0, 24)])
            hour_ok = any(start <= hour_of_day < end for start, end in timing_windows)
            
            day_prefs = self.config.day_preferences.get(channel, list(range(7)))
            day_ok = day_of_week in day_prefs
            
            # Penalize channels used at suboptimal times
            if not hour_ok or not day_ok:
                channel_values[value_col] *= 0.7  # 30% penalty
        
        # Apply contact frequency constraints
        if contact_history is not None:
            channel_values = self._apply_contact_constraints(
                channel_values, contact_history, current_datetime
            )
        
        # Apply capacity constraints
        channel_values = self._apply_capacity_constraints(channel_values)
        
        # Select best channel
        value_cols = [col for col in channel_values.columns if col.endswith('_value')]
        
        result = pd.DataFrame(index=X.index)
        result['channel'] = channel_values[value_cols].idxmax(axis=1).str.replace('_value', '')
        result['expected_value'] = channel_values[value_cols].max(axis=1)
        
        # Add timing recommendations
        result['recommended_hour'] = result['channel'].map(
            lambda ch: self._get_optimal_hour(ch)
        )
        result['recommended_day'] = result['channel'].map(
            lambda ch: self._get_optimal_day(ch)
        )
        
        # Add all channel probabilities
        response_probs = self.predict_channel_response(X, timing_features)
        for col in response_probs.columns:
            result[col] = response_probs[col]
        
        return result
    
    def optimize_contact_sequence(
        self,
        X: pd.DataFrame,
        propensity_scores: pd.Series,
        payment_amounts: pd.Series,
        sequence_length: int = 3,
        time_horizon_days: int = 14
    ) -> pd.DataFrame:
        """
        Optimize multi-touch contact sequence.
        
        Args:
            X: Features
            propensity_scores: Probability of payment
            payment_amounts: Expected payment amounts
            sequence_length: Number of contacts in sequence
            time_horizon_days: Time horizon for sequence
            
        Returns:
            DataFrame with optimized contact sequence
            
        Example:
            >>> sequence = optimizer.optimize_contact_sequence(
            ...     X_test, propensity_test, amounts_test,
            ...     sequence_length=3, time_horizon_days=14
            ... )
        """
        result = pd.DataFrame(index=X.index)
        
        # Simple heuristic: alternate between high-value channels
        # In practice, this would use reinforcement learning or dynamic programming
        
        for i in range(sequence_length):
            day_offset = int((i / sequence_length) * time_horizon_days)
            
            # Get optimal channel for this touchpoint
            recommendations = self.recommend(X, propensity_scores, payment_amounts)
            
            result[f'contact_{i+1}_channel'] = recommendations['channel']
            result[f'contact_{i+1}_day'] = day_offset
            result[f'contact_{i+1}_value'] = recommendations['expected_value']
            
            # For subsequent contacts, reduce propensity (diminishing returns)
            propensity_scores = propensity_scores * 0.8
        
        result['total_sequence_value'] = sum(
            result[f'contact_{i+1}_value'] for i in range(sequence_length)
        )
        
        return result
    
    def _apply_contact_constraints(
        self,
        channel_values: pd.DataFrame,
        contact_history: pd.DataFrame,
        current_datetime: datetime
    ) -> pd.DataFrame:
        """Apply contact frequency constraints."""
        if contact_history is None or len(contact_history) == 0:
            return channel_values
        
        # This is simplified - in practice, would track per-user contact history
        # and apply constraints individually
        
        return channel_values
    
    def _apply_capacity_constraints(
        self,
        channel_values: pd.DataFrame
    ) -> pd.DataFrame:
        """Apply channel capacity constraints."""
        # Get best channel for each account
        value_cols = [col for col in channel_values.columns if col.endswith('_value')]
        best_channels = channel_values[value_cols].idxmax(axis=1).str.replace('_value', '')
        
        channel_counts = best_channels.value_counts()
        total = len(channel_values)
        
        # Penalize over-capacity channels
        for channel, count in channel_counts.items():
            capacity = self.config.channel_capacity.get(channel, 1.0)
            max_count = int(total * capacity)
            
            if count > max_count:
                # Apply penalty to least valuable assignments
                value_col = f'{channel}_value'
                if value_col in channel_values.columns:
                    channel_values[value_col] *= 0.5
        
        return channel_values
    
    def _get_optimal_hour(self, channel: str) -> int:
        """Get optimal hour for channel."""
        windows = self.config.timing_windows.get(channel, [(9, 17)])
        # Return midpoint of first window
        start, end = windows[0]
        return (start + end) // 2
    
    def _get_optimal_day(self, channel: str) -> int:
        """Get optimal day of week for channel."""
        days = self.config.day_preferences.get(channel, [1, 2, 3])
        # Return first preferred day
        return days[0] if days else 1
    
    def get_channel_stats(
        self,
        channels: pd.Series,
        responses: pd.Series
    ) -> pd.DataFrame:
        """
        Get channel performance statistics.
        
        Args:
            channels: Applied channels
            responses: Binary response outcomes
            
        Returns:
            DataFrame with channel statistics
            
        Example:
            >>> stats = optimizer.get_channel_stats(channels_actual, responses_actual)
        """
        stats = []
        
        for channel in self.config.channels:
            channel_mask = channels == channel
            if channel_mask.any():
                responses_channel = responses[channel_mask]
                
                stat = {
                    'channel': channel,
                    'count': len(responses_channel),
                    'response_rate': responses_channel.mean(),
                    'cost_per_contact': self.config.channel_costs.get(channel, 0),
                    'total_cost': len(responses_channel) * self.config.channel_costs.get(channel, 0),
                    'capacity': self.config.channel_capacity.get(channel, 1.0)
                }
                
                stats.append(stat)
        
        return pd.DataFrame(stats)
    
    def save(self, filepath: Union[str, Path]) -> None:
        """Save optimizer to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        save_dict = {
            'channel_models': self.channel_models,
            'timing_models': self.timing_models,
            'offer_model': self.offer_model,
            'feature_names': self.feature_names,
            'config': self.config,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(save_dict, filepath)
        logger.info(f"Optimizer saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'ChannelOptimizer':
        """Load optimizer from disk."""
        save_dict = joblib.load(filepath)
        
        instance = cls()
        instance.channel_models = save_dict['channel_models']
        instance.timing_models = save_dict['timing_models']
        instance.offer_model = save_dict['offer_model']
        instance.feature_names = save_dict['feature_names']
        instance.config = save_dict['config']
        instance.is_fitted = save_dict['is_fitted']
        
        logger.info(f"Optimizer loaded from {filepath}")
        return instance
