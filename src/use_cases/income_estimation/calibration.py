"""
Continuous Prediction Calibration for Income Estimation

Implements advanced calibration methods for uncertainty quantification:
- Conformal prediction for prediction intervals
- Isotonic calibration for probability calibration
- Bayesian calibration with prior updates
- Multi-quantile prediction (10th, 50th, 90th percentiles)
- Calibration validation metrics (coverage, sharpness)
- Dynamic recalibration based on new data
- Confidence interval generation
- Uncertainty quantification with distributional forecasts

Example:
    >>> from src.use_cases.income_estimation.calibration import (
    ...     ConformalPredictor, IsotonicCalibrator, BayesianCalibrator
    ... )
    >>> 
    >>> # Conformal prediction intervals
    >>> conformal = ConformalPredictor(alpha=0.1)
    >>> conformal.fit(y_train, predictions_train)
    >>> intervals = conformal.predict_interval(predictions_test)
    >>> 
    >>> # Isotonic calibration
    >>> isotonic = IsotonicCalibrator()
    >>> isotonic.fit(y_true, y_prob)
    >>> calibrated_probs = isotonic.calibrate(y_prob_new)

Author: Principal Data Science Decision Agent
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats
from sklearn.isotonic import IsotonicRegression


@dataclass
class PredictionInterval:
    """Container for prediction interval."""
    
    point_estimate: float
    lower_bound: float
    upper_bound: float
    confidence_level: float
    interval_width: float
    
    def contains(self, value: float) -> bool:
        """Check if value is within interval."""
        return self.lower_bound <= value <= self.upper_bound


@dataclass
class CalibrationMetrics:
    """Container for calibration validation metrics."""
    
    coverage: float  # Empirical coverage rate
    target_coverage: float  # Target coverage rate
    coverage_gap: float  # |coverage - target|
    sharpness: float  # Average interval width
    normalized_sharpness: float  # Width / mean(predictions)
    calibration_error: float  # Mean calibration error
    is_well_calibrated: bool  # coverage_gap < threshold


@dataclass
class QuantilePrediction:
    """Container for multi-quantile predictions."""
    
    q10: float  # 10th percentile
    q25: float  # 25th percentile
    q50: float  # 50th percentile (median)
    q75: float  # 75th percentile
    q90: float  # 90th percentile
    mean: float  # Expected value
    std: float  # Standard deviation
    
    def get_interval(self, confidence: float = 0.8) -> Tuple[float, float]:
        """Get prediction interval for given confidence level."""
        if confidence == 0.8:
            return (self.q10, self.q90)
        elif confidence == 0.5:
            return (self.q25, self.q75)
        else:
            # Approximate using normal distribution
            z = stats.norm.ppf(0.5 + confidence / 2)
            half_width = z * self.std
            return (self.mean - half_width, self.mean + half_width)


class ConformalPredictor:
    """
    Conformal prediction for distribution-free prediction intervals.
    
    Provides valid prediction intervals without distributional assumptions
    using the conformal prediction framework.
    """
    
    def __init__(
        self,
        alpha: float = 0.1,
        method: str = 'absolute'
    ):
        """
        Initialize conformal predictor.
        
        Args:
            alpha: Miscoverage rate (1 - coverage_level)
            method: 'absolute' or 'normalized' residuals
            
        Example:
            >>> predictor = ConformalPredictor(alpha=0.1)  # 90% coverage
            >>> predictor.fit(y_calibration, predictions_calibration)
            >>> intervals = predictor.predict_interval(predictions_test)
        """
        self.alpha = alpha
        self.method = method
        self.coverage_level = 1 - alpha
        self.quantile_value_ = None
        self.is_fitted_ = False
        
        logger.info(
            f"ConformalPredictor initialized with {self.coverage_level:.0%} coverage"
        )
    
    def fit(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> 'ConformalPredictor':
        """
        Fit conformal predictor on calibration set.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Self
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have same length")
        
        # Calculate nonconformity scores
        if self.method == 'absolute':
            scores = np.abs(y_true - y_pred)
        elif self.method == 'normalized':
            # Normalize by predicted value
            scores = np.abs(y_true - y_pred) / (np.abs(y_pred) + 1e-10)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Calculate quantile
        n = len(scores)
        q = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.quantile_value_ = np.quantile(scores, q)
        
        self.is_fitted_ = True
        
        logger.info(
            f"Fitted conformal predictor: quantile={self.quantile_value_:.4f}"
        )
        
        return self
    
    def predict_interval(
        self,
        y_pred: Union[np.ndarray, float]
    ) -> Union[List[PredictionInterval], PredictionInterval]:
        """
        Predict intervals for new predictions.
        
        Args:
            y_pred: Predicted values
            
        Returns:
            PredictionInterval or list of PredictionInterval
            
        Example:
            >>> intervals = predictor.predict_interval([50000, 75000, 100000])
            >>> for interval in intervals:
            ...     print(f"[{interval.lower_bound:.0f}, {interval.upper_bound:.0f}]")
        """
        if not self.is_fitted_:
            raise ValueError("Predictor not fitted. Call fit() first.")
        
        is_scalar = np.isscalar(y_pred)
        y_pred = np.atleast_1d(y_pred)
        
        intervals = []
        
        for pred in y_pred:
            if self.method == 'absolute':
                delta = self.quantile_value_
            else:  # normalized
                delta = self.quantile_value_ * (np.abs(pred) + 1e-10)
            
            lower = pred - delta
            upper = pred + delta
            
            interval = PredictionInterval(
                point_estimate=float(pred),
                lower_bound=float(max(lower, 0)),  # Non-negative income
                upper_bound=float(upper),
                confidence_level=self.coverage_level,
                interval_width=float(upper - lower)
            )
            
            intervals.append(interval)
        
        return intervals[0] if is_scalar else intervals
    
    def validate_coverage(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> CalibrationMetrics:
        """
        Validate coverage on test set.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            CalibrationMetrics
        """
        intervals = self.predict_interval(y_pred)
        
        # Calculate empirical coverage
        coverage = np.mean([
            interval.contains(y) for interval, y in zip(intervals, y_true)
        ])
        
        coverage_gap = abs(coverage - self.coverage_level)
        
        # Calculate sharpness (average interval width)
        widths = [interval.interval_width for interval in intervals]
        sharpness = float(np.mean(widths))
        normalized_sharpness = sharpness / (np.mean(y_pred) + 1e-10)
        
        # Well-calibrated if coverage gap < 0.05
        is_well_calibrated = coverage_gap < 0.05
        
        return CalibrationMetrics(
            coverage=float(coverage),
            target_coverage=self.coverage_level,
            coverage_gap=float(coverage_gap),
            sharpness=sharpness,
            normalized_sharpness=float(normalized_sharpness),
            calibration_error=float(coverage_gap),
            is_well_calibrated=is_well_calibrated
        )


class IsotonicCalibrator:
    """
    Isotonic regression for probability calibration.
    
    Calibrates predicted probabilities or scores to better match
    empirical frequencies using isotonic regression.
    """
    
    def __init__(
        self,
        y_min: Optional[float] = None,
        y_max: Optional[float] = None
    ):
        """
        Initialize isotonic calibrator.
        
        Args:
            y_min: Minimum output value (default: None)
            y_max: Maximum output value (default: None)
        """
        self.y_min = y_min
        self.y_max = y_max
        self.calibrator_ = None
        self.is_fitted_ = False
        
        logger.info("IsotonicCalibrator initialized")
    
    def fit(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> 'IsotonicCalibrator':
        """
        Fit isotonic calibrator.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Self
        """
        self.calibrator_ = IsotonicRegression(
            y_min=self.y_min,
            y_max=self.y_max,
            out_of_bounds='clip'
        )
        
        self.calibrator_.fit(y_pred, y_true)
        self.is_fitted_ = True
        
        logger.info("Fitted isotonic calibrator")
        
        return self
    
    def calibrate(
        self,
        y_pred: np.ndarray
    ) -> np.ndarray:
        """
        Calibrate predictions.
        
        Args:
            y_pred: Predictions to calibrate
            
        Returns:
            Calibrated predictions
        """
        if not self.is_fitted_:
            raise ValueError("Calibrator not fitted. Call fit() first.")
        
        return self.calibrator_.predict(y_pred)
    
    def validate_calibration(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_bins: int = 10
    ) -> CalibrationMetrics:
        """
        Validate calibration using binning.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            n_bins: Number of bins for calibration curve
            
        Returns:
            CalibrationMetrics
        """
        # Calibrate predictions
        y_calibrated = self.calibrate(y_pred)
        
        # Bin predictions
        bin_edges = np.linspace(
            y_calibrated.min(),
            y_calibrated.max(),
            n_bins + 1
        )
        
        bin_indices = np.digitize(y_calibrated, bin_edges[:-1]) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        # Calculate calibration error per bin
        calibration_errors = []
        
        for i in range(n_bins):
            mask = bin_indices == i
            if mask.sum() > 0:
                pred_mean = y_calibrated[mask].mean()
                true_mean = y_true[mask].mean()
                calibration_errors.append(abs(pred_mean - true_mean))
        
        # Mean calibration error
        calibration_error = float(np.mean(calibration_errors)) if calibration_errors else 0.0
        
        # Calculate other metrics
        residuals = y_true - y_calibrated
        mse = float(np.mean(residuals ** 2))
        mae = float(np.mean(np.abs(residuals)))
        
        return CalibrationMetrics(
            coverage=0.0,  # N/A for isotonic
            target_coverage=0.0,
            coverage_gap=0.0,
            sharpness=float(np.std(residuals)),
            normalized_sharpness=mae / (np.mean(y_true) + 1e-10),
            calibration_error=calibration_error,
            is_well_calibrated=calibration_error < 0.1 * np.mean(y_true)
        )


class BayesianCalibrator:
    """
    Bayesian calibration with prior updates.
    
    Uses Bayesian inference to calibrate predictions and quantify uncertainty,
    updating beliefs as new data arrives.
    """
    
    def __init__(
        self,
        prior_mean: float = 50000.0,
        prior_std: float = 20000.0,
        likelihood_std: float = 10000.0
    ):
        """
        Initialize Bayesian calibrator.
        
        Args:
            prior_mean: Prior mean for income
            prior_std: Prior standard deviation
            likelihood_std: Likelihood standard deviation (measurement noise)
        """
        self.prior_mean = prior_mean
        self.prior_std = prior_std
        self.likelihood_std = likelihood_std
        
        # Current posterior (starts as prior)
        self.posterior_mean_ = prior_mean
        self.posterior_std_ = prior_std
        self.n_updates_ = 0
        
        logger.info(
            f"BayesianCalibrator initialized with prior N({prior_mean}, {prior_std}²)"
        )
    
    def update(
        self,
        observation: float,
        observation_std: Optional[float] = None
    ) -> 'BayesianCalibrator':
        """
        Update posterior with new observation.
        
        Args:
            observation: Observed income value
            observation_std: Observation uncertainty (default: likelihood_std)
            
        Returns:
            Self
        """
        if observation_std is None:
            observation_std = self.likelihood_std
        
        # Bayesian update (Gaussian-Gaussian conjugate)
        prior_var = self.posterior_std_ ** 2
        obs_var = observation_std ** 2
        
        # Posterior parameters
        posterior_var = 1.0 / (1.0 / prior_var + 1.0 / obs_var)
        posterior_mean = posterior_var * (
            self.posterior_mean_ / prior_var + observation / obs_var
        )
        
        self.posterior_mean_ = posterior_mean
        self.posterior_std_ = np.sqrt(posterior_var)
        self.n_updates_ += 1
        
        logger.debug(
            f"Updated posterior: N({self.posterior_mean_:.2f}, {self.posterior_std_:.2f}²)"
        )
        
        return self
    
    def update_batch(
        self,
        observations: np.ndarray,
        observation_stds: Optional[np.ndarray] = None
    ) -> 'BayesianCalibrator':
        """
        Update posterior with batch of observations.
        
        Args:
            observations: Array of observed values
            observation_stds: Array of observation uncertainties
            
        Returns:
            Self
        """
        observations = np.asarray(observations)
        
        if observation_stds is None:
            observation_stds = np.full(len(observations), self.likelihood_std)
        else:
            observation_stds = np.asarray(observation_stds)
        
        for obs, obs_std in zip(observations, observation_stds):
            self.update(obs, obs_std)
        
        return self
    
    def predict(
        self,
        return_std: bool = False
    ) -> Union[float, Tuple[float, float]]:
        """
        Get current posterior prediction.
        
        Args:
            return_std: If True, return (mean, std) tuple
            
        Returns:
            Posterior mean or (mean, std)
        """
        if return_std:
            return (self.posterior_mean_, self.posterior_std_)
        else:
            return self.posterior_mean_
    
    def predict_interval(
        self,
        confidence: float = 0.9
    ) -> PredictionInterval:
        """
        Get prediction interval from posterior.
        
        Args:
            confidence: Confidence level (0-1)
            
        Returns:
            PredictionInterval
        """
        z = stats.norm.ppf(0.5 + confidence / 2)
        half_width = z * self.posterior_std_
        
        return PredictionInterval(
            point_estimate=self.posterior_mean_,
            lower_bound=max(self.posterior_mean_ - half_width, 0),
            upper_bound=self.posterior_mean_ + half_width,
            confidence_level=confidence,
            interval_width=2 * half_width
        )
    
    def reset(self) -> 'BayesianCalibrator':
        """Reset to prior."""
        self.posterior_mean_ = self.prior_mean
        self.posterior_std_ = self.prior_std
        self.n_updates_ = 0
        
        logger.info("Reset to prior")
        
        return self


class QuantileRegressor:
    """
    Multi-quantile regression for distributional predictions.
    
    Predicts multiple quantiles simultaneously to capture full
    predictive distribution.
    """
    
    def __init__(
        self,
        quantiles: List[float] = [0.1, 0.25, 0.5, 0.75, 0.9]
    ):
        """
        Initialize quantile regressor.
        
        Args:
            quantiles: List of quantiles to predict
        """
        self.quantiles = sorted(quantiles)
        self.models_ = {}
        self.is_fitted_ = False
        
        logger.info(f"QuantileRegressor initialized with quantiles: {quantiles}")
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> 'QuantileRegressor':
        """
        Fit quantile models.
        
        Args:
            X: Features
            y: Target values
            
        Returns:
            Self
        """
        from sklearn.linear_model import QuantileRegressor as SKQuantileRegressor
        
        for q in self.quantiles:
            model = SKQuantileRegressor(quantile=q, alpha=0.0)
            model.fit(X, y)
            self.models_[q] = model
        
        self.is_fitted_ = True
        
        logger.info("Fitted quantile models")
        
        return self
    
    def predict_quantiles(
        self,
        X: np.ndarray
    ) -> Dict[float, np.ndarray]:
        """
        Predict all quantiles.
        
        Args:
            X: Features
            
        Returns:
            Dictionary mapping quantile -> predictions
        """
        if not self.is_fitted_:
            raise ValueError("Model not fitted. Call fit() first.")
        
        predictions = {}
        for q, model in self.models_.items():
            predictions[q] = model.predict(X)
        
        return predictions
    
    def predict_distribution(
        self,
        X: np.ndarray
    ) -> List[QuantilePrediction]:
        """
        Predict full distribution for each sample.
        
        Args:
            X: Features
            
        Returns:
            List of QuantilePrediction objects
        """
        quantile_preds = self.predict_quantiles(X)
        
        results = []
        n_samples = len(X)
        
        for i in range(n_samples):
            # Extract quantiles for this sample
            q_dict = {q: quantile_preds[q][i] for q in self.quantiles}
            
            # Estimate mean and std from quantiles
            # Use median as mean estimate
            mean = q_dict.get(0.5, np.mean(list(q_dict.values())))
            
            # Estimate std from IQR
            if 0.25 in q_dict and 0.75 in q_dict:
                iqr = q_dict[0.75] - q_dict[0.25]
                std = iqr / 1.35  # For normal distribution
            else:
                std = (max(q_dict.values()) - min(q_dict.values())) / 4
            
            result = QuantilePrediction(
                q10=q_dict.get(0.1, mean - 1.28 * std),
                q25=q_dict.get(0.25, mean - 0.67 * std),
                q50=q_dict.get(0.5, mean),
                q75=q_dict.get(0.75, mean + 0.67 * std),
                q90=q_dict.get(0.9, mean + 1.28 * std),
                mean=mean,
                std=max(std, 0)
            )
            
            results.append(result)
        
        return results


class DynamicCalibrator:
    """
    Dynamic calibrator that adapts over time.
    
    Maintains calibration as data distribution shifts using
    sliding window and exponential weighting.
    """
    
    def __init__(
        self,
        window_size: int = 100,
        decay_factor: float = 0.95,
        recalibration_interval: int = 10
    ):
        """
        Initialize dynamic calibrator.
        
        Args:
            window_size: Size of sliding window
            decay_factor: Exponential decay for weighting
            recalibration_interval: Recalibrate every N observations
        """
        self.window_size = window_size
        self.decay_factor = decay_factor
        self.recalibration_interval = recalibration_interval
        
        self.calibrator_ = IsotonicCalibrator()
        self.buffer_y_true_ = []
        self.buffer_y_pred_ = []
        self.n_observations_ = 0
        self.is_fitted_ = False
        
        logger.info(
            f"DynamicCalibrator initialized with window={window_size}, "
            f"decay={decay_factor}"
        )
    
    def partial_fit(
        self,
        y_true: float,
        y_pred: float
    ) -> 'DynamicCalibrator':
        """
        Incrementally update calibrator.
        
        Args:
            y_true: True value
            y_pred: Predicted value
            
        Returns:
            Self
        """
        self.buffer_y_true_.append(y_true)
        self.buffer_y_pred_.append(y_pred)
        self.n_observations_ += 1
        
        # Keep only last window_size observations
        if len(self.buffer_y_true_) > self.window_size:
            self.buffer_y_true_.pop(0)
            self.buffer_y_pred_.pop(0)
        
        # Recalibrate periodically
        if self.n_observations_ % self.recalibration_interval == 0:
            self._recalibrate()
        
        return self
    
    def calibrate(
        self,
        y_pred: np.ndarray
    ) -> np.ndarray:
        """
        Calibrate predictions.
        
        Args:
            y_pred: Predictions to calibrate
            
        Returns:
            Calibrated predictions
        """
        if not self.is_fitted_:
            # Not yet calibrated, return as-is
            return y_pred
        
        return self.calibrator_.calibrate(y_pred)
    
    def _recalibrate(self) -> None:
        """Recalibrate using current buffer."""
        if len(self.buffer_y_true_) < 10:
            return  # Not enough data
        
        # Apply exponential weights (more recent = higher weight)
        n = len(self.buffer_y_true_)
        weights = np.array([self.decay_factor ** (n - i - 1) for i in range(n)])
        weights /= weights.sum()
        
        # Weighted resampling
        indices = np.random.choice(
            n,
            size=n,
            replace=True,
            p=weights
        )
        
        y_true_weighted = np.array(self.buffer_y_true_)[indices]
        y_pred_weighted = np.array(self.buffer_y_pred_)[indices]
        
        # Refit calibrator
        self.calibrator_.fit(y_true_weighted, y_pred_weighted)
        self.is_fitted_ = True
        
        logger.debug(f"Recalibrated with {len(self.buffer_y_true_)} observations")
