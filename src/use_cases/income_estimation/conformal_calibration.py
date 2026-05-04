"""
Conformal Calibration for Income Estimation
============================================
Distribution-free uncertainty quantification.

Classes:
  - ConformalPredictor   : absolute / normalized residual intervals
  - IsotonicCalibrator   : isotonic regression for bias correction
  - QuantileRegressor    : multi-quantile distributional predictions
  - DynamicCalibrator    : sliding-window adaptive recalibration

PredictionInterval and CalibrationMetrics are re-exported for downstream use.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.isotonic import IsotonicRegression

logger = logging.getLogger(__name__)


@dataclass
class PredictionInterval:
    point_estimate: float
    lower_bound: float
    upper_bound: float
    confidence_level: float
    interval_width: float

    def contains(self, value: float) -> bool:
        return self.lower_bound <= value <= self.upper_bound


@dataclass
class CalibrationMetrics:
    coverage: float
    target_coverage: float
    coverage_gap: float
    sharpness: float
    normalized_sharpness: float
    calibration_error: float
    is_well_calibrated: bool


class ConformalPredictor:
    """
    Conformal prediction for distribution-free prediction intervals.
    Provides valid coverage without distributional assumptions.
    """

    def __init__(self, alpha: float = 0.1, method: str = "absolute"):
        self.alpha          = alpha
        self.method         = method
        self.coverage_level = 1 - alpha
        self.quantile_value_: Optional[float] = None
        self.is_fitted_     = False

    def fit(self, y_true: np.ndarray, y_pred: np.ndarray) -> "ConformalPredictor":
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        if self.method == "absolute":
            scores = np.abs(y_true - y_pred)
        else:
            scores = np.abs(y_true - y_pred) / (np.abs(y_pred) + 1e-10)

        n = len(scores)
        q = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.quantile_value_ = float(np.quantile(scores, min(q, 1.0)))
        self.is_fitted_ = True

        logger.info("ConformalPredictor fitted: quantile=%.4f", self.quantile_value_)
        return self

    def predict_interval(
        self, y_pred: Union[np.ndarray, float]
    ) -> Union[List[PredictionInterval], PredictionInterval]:
        if not self.is_fitted_:
            raise RuntimeError("Call fit() first.")

        is_scalar = np.isscalar(y_pred)
        y_pred = np.atleast_1d(np.asarray(y_pred, dtype=float))

        intervals = []
        for pred in y_pred:
            delta = (
                self.quantile_value_
                if self.method == "absolute"
                else self.quantile_value_ * (abs(pred) + 1e-10)
            )
            intervals.append(PredictionInterval(
                point_estimate=float(pred),
                lower_bound=float(max(pred - delta, 0)),
                upper_bound=float(pred + delta),
                confidence_level=self.coverage_level,
                interval_width=float(2 * delta),
            ))

        return intervals[0] if is_scalar else intervals

    def validate_coverage(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> CalibrationMetrics:
        intervals = self.predict_interval(y_pred)
        coverage  = float(np.mean([
            iv.contains(y) for iv, y in zip(intervals, y_true)
        ]))
        gap       = abs(coverage - self.coverage_level)
        widths    = [iv.interval_width for iv in intervals]
        sharpness = float(np.mean(widths))

        return CalibrationMetrics(
            coverage=coverage,
            target_coverage=self.coverage_level,
            coverage_gap=gap,
            sharpness=sharpness,
            normalized_sharpness=sharpness / (np.mean(y_pred) + 1e-10),
            calibration_error=gap,
            is_well_calibrated=(gap < 0.05),
        )


class IsotonicCalibrator:
    """Isotonic regression to remove systematic bias from predictions."""

    def __init__(self, y_min: Optional[float] = None, y_max: Optional[float] = None):
        self.y_min = y_min
        self.y_max = y_max
        self._reg  = None
        self.is_fitted_ = False

    def fit(self, y_true: np.ndarray, y_pred: np.ndarray) -> "IsotonicCalibrator":
        self._reg = IsotonicRegression(
            y_min=self.y_min, y_max=self.y_max, out_of_bounds="clip"
        )
        self._reg.fit(y_pred, y_true)
        self.is_fitted_ = True
        return self

    def calibrate(self, y_pred: np.ndarray) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("Call fit() first.")
        return self._reg.predict(y_pred)


class QuantileRegressor:
    """Multi-quantile predictions for full distributional output."""

    def __init__(self, quantiles: List[float] = [0.1, 0.25, 0.5, 0.75, 0.9]):
        self.quantiles = sorted(quantiles)
        self._models: Dict[float, object] = {}
        self.is_fitted_ = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "QuantileRegressor":
        from sklearn.linear_model import QuantileRegressor as SKQ
        for q in self.quantiles:
            m = SKQ(quantile=q, alpha=0.0)
            m.fit(X, y)
            self._models[q] = m
        self.is_fitted_ = True
        return self

    def predict_quantiles(self, X: np.ndarray) -> Dict[float, np.ndarray]:
        if not self.is_fitted_:
            raise RuntimeError("Call fit() first.")
        return {q: m.predict(X) for q, m in self._models.items()}

    def predict_intervals(
        self, X: np.ndarray, lower_q: float = 0.1, upper_q: float = 0.9
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (lower_bound, upper_bound) arrays."""
        preds = self.predict_quantiles(X)
        lower = preds.get(lower_q, preds[min(self.quantiles)])
        upper = preds.get(upper_q, preds[max(self.quantiles)])
        return np.maximum(lower, 0), upper


class DynamicCalibrator:
    """Sliding-window calibrator that adapts as distribution shifts."""

    def __init__(
        self,
        window_size: int = 100,
        decay_factor: float = 0.95,
        recalibration_interval: int = 10,
    ):
        self.window_size            = window_size
        self.decay_factor           = decay_factor
        self.recalibration_interval = recalibration_interval

        self._iso               = IsotonicCalibrator()
        self._buf_y_true: List[float] = []
        self._buf_y_pred: List[float] = []
        self._n_obs             = 0
        self.is_fitted_         = False

    def partial_fit(self, y_true: float, y_pred: float) -> "DynamicCalibrator":
        self._buf_y_true.append(y_true)
        self._buf_y_pred.append(y_pred)
        self._n_obs += 1

        if len(self._buf_y_true) > self.window_size:
            self._buf_y_true.pop(0)
            self._buf_y_pred.pop(0)

        if self._n_obs % self.recalibration_interval == 0:
            self._recalibrate()

        return self

    def calibrate(self, y_pred: np.ndarray) -> np.ndarray:
        if not self.is_fitted_:
            return y_pred
        return self._iso.calibrate(y_pred)

    def _recalibrate(self) -> None:
        n = len(self._buf_y_true)
        if n < 10:
            return
        w  = np.array([self.decay_factor ** (n - i - 1) for i in range(n)])
        w /= w.sum()
        idx = np.random.choice(n, size=n, replace=True, p=w)
        self._iso.fit(
            np.array(self._buf_y_true)[idx],
            np.array(self._buf_y_pred)[idx],
        )
        self.is_fitted_ = True
