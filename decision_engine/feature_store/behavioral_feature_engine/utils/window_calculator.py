"""
Window Calculator - Reusable utilities for rolling statistics, slopes, momentum

Provides point-in-time safe aggregations across all time windows.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from scipy import stats as scipy_stats


class WindowCalculator:
    """
    Compute statistics over rolling windows with point-in-time safety.

    All computations respect as_of_date and only use historical data.
    """

    # Standard windows (months)
    WINDOWS = {
        "3M": 3,
        "6M": 6,
        "12M": 12,
        "24M": 24
    }

    @staticmethod
    def compute_statistics(
        series: pd.Series,
        windows: List[str] = None,
        statistics: List[str] = None
    ) -> Dict[str, float]:
        """
        Compute statistics over multiple windows.

        Args:
            series: Time series data (must be sorted by date)
            windows: List of window names (e.g., ["3M", "6M"])
            statistics: List of stats to compute (e.g., ["mean", "max", "std"])

        Returns:
            Dictionary with keys like "3M_mean", "6M_std", etc.
        """
        windows = windows or ["3M", "6M", "12M"]
        statistics = statistics or ["mean", "max", "min", "std"]

        results = {}

        for window in windows:
            window_months = WindowCalculator.WINDOWS.get(window, int(window.replace("M", "")))

            # Get last N months of data
            window_data = series.tail(window_months)

            if len(window_data) == 0:
                # Insufficient data
                for stat in statistics:
                    results[f"{window}_{stat}"] = np.nan
                continue

            # Compute each statistic
            for stat in statistics:
                key = f"{window}_{stat}"
                if stat == "mean":
                    results[key] = window_data.mean()
                elif stat == "max":
                    results[key] = window_data.max()
                elif stat == "min":
                    results[key] = window_data.min()
                elif stat == "std":
                    results[key] = window_data.std()
                elif stat == "median":
                    results[key] = window_data.median()
                elif stat == "sum":
                    results[key] = window_data.sum()
                elif stat == "count":
                    results[key] = window_data.count()
                elif stat == "cv":  # Coefficient of variation
                    mean_val = window_data.mean()
                    std_val = window_data.std()
                    results[key] = std_val / mean_val if mean_val != 0 else np.nan
                else:
                    results[key] = np.nan

        return results

    @staticmethod
    def compute_slope(
        series: pd.Series,
        windows: List[str] = None
    ) -> Dict[str, float]:
        """
        Compute linear regression slope over windows.

        Args:
            series: Time series data
            windows: List of window names

        Returns:
            Dictionary with keys like "3M_slope", "6M_slope"
        """
        windows = windows or ["3M", "6M", "12M"]
        results = {}

        for window in windows:
            window_months = WindowCalculator.WINDOWS.get(window, int(window.replace("M", "")))
            window_data = series.tail(window_months)

            if len(window_data) < 2:
                results[f"{window}_slope"] = np.nan
                continue

            # Linear regression: y = mx + b
            x = np.arange(len(window_data))
            y = window_data.values

            # Remove NaN
            mask = ~np.isnan(y)
            if mask.sum() < 2:
                results[f"{window}_slope"] = np.nan
                continue

            slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(
                x[mask], y[mask]
            )

            results[f"{window}_slope"] = slope
            results[f"{window}_r_squared"] = r_value ** 2

        return results

    @staticmethod
    def compute_momentum(
        series: pd.Series,
        windows: List[str] = None
    ) -> Dict[str, float]:
        """
        Compute momentum (change in slope) over windows.

        Momentum = slope(current_window) - slope(previous_window)

        Args:
            series: Time series data
            windows: List of window names

        Returns:
            Dictionary with keys like "3M_momentum", "6M_momentum"
        """
        windows = windows or ["3M", "6M", "12M"]
        results = {}

        for window in windows:
            window_months = WindowCalculator.WINDOWS.get(window, int(window.replace("M", "")))

            # Current window slope
            current_window = series.tail(window_months)
            current_slope = WindowCalculator._compute_single_slope(current_window)

            # Previous window slope
            previous_window = series.iloc[-2*window_months:-window_months] if len(series) >= 2*window_months else pd.Series()
            previous_slope = WindowCalculator._compute_single_slope(previous_window)

            # Momentum = change in slope
            if not np.isnan(current_slope) and not np.isnan(previous_slope):
                results[f"{window}_momentum"] = current_slope - previous_slope
            else:
                results[f"{window}_momentum"] = np.nan

        return results

    @staticmethod
    def _compute_single_slope(series: pd.Series) -> float:
        """Helper to compute slope for a single series"""
        if len(series) < 2:
            return np.nan

        x = np.arange(len(series))
        y = series.values

        mask = ~np.isnan(y)
        if mask.sum() < 2:
            return np.nan

        slope, _, _, _, _ = scipy_stats.linregress(x[mask], y[mask])
        return slope

    @staticmethod
    def compute_period_over_period_change(
        series: pd.Series,
        periods: List[str] = None
    ) -> Dict[str, float]:
        """
        Compute MoM, QoQ, YoY changes.

        Args:
            series: Time series data (monthly frequency)
            periods: List of periods (e.g., ["MoM", "QoQ", "YoY"])

        Returns:
            Dictionary with keys like "mom_change", "qoq_change"
        """
        periods = periods or ["MoM", "QoQ", "YoY"]
        results = {}

        period_lags = {
            "MoM": 1,
            "QoQ": 3,
            "YoY": 12
        }

        current_value = series.iloc[-1] if len(series) > 0 else np.nan

        for period in periods:
            lag = period_lags.get(period, 1)

            if len(series) > lag:
                previous_value = series.iloc[-1-lag]
                change = current_value - previous_value
                pct_change = (change / previous_value * 100) if previous_value != 0 else np.nan

                results[f"{period.lower()}_change"] = change
                results[f"{period.lower()}_pct_change"] = pct_change
            else:
                results[f"{period.lower()}_change"] = np.nan
                results[f"{period.lower()}_pct_change"] = np.nan

        return results

    @staticmethod
    def compute_streak(
        series: pd.Series,
        condition_func: callable
    ) -> int:
        """
        Compute current streak where condition is True.

        Args:
            series: Time series data
            condition_func: Function that returns True/False for each value

        Returns:
            Length of current streak
        """
        if len(series) == 0:
            return 0

        streak = 0
        for value in reversed(series):
            if condition_func(value):
                streak += 1
            else:
                break

        return streak

    @staticmethod
    def compute_max_streak(
        series: pd.Series,
        condition_func: callable
    ) -> int:
        """
        Compute maximum streak where condition was True.

        Args:
            series: Time series data
            condition_func: Function that returns True/False for each value

        Returns:
            Maximum streak length
        """
        if len(series) == 0:
            return 0

        max_streak = 0
        current_streak = 0

        for value in series:
            if condition_func(value):
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0

        return max_streak

    @staticmethod
    def compute_mode(series: pd.Series, window: str = "6M") -> Union[float, str]:
        """
        Compute most frequent value over window.

        Args:
            series: Time series data
            window: Window name

        Returns:
            Most frequent value
        """
        window_months = WindowCalculator.WINDOWS.get(window, int(window.replace("M", "")))
        window_data = series.tail(window_months)

        if len(window_data) == 0:
            return np.nan

        return window_data.mode().iloc[0] if len(window_data.mode()) > 0 else np.nan

    @staticmethod
    def count_occurrences(
        series: pd.Series,
        value: Union[float, str, bool],
        window: str = "6M"
    ) -> int:
        """
        Count occurrences of a value over window.

        Args:
            series: Time series data
            value: Value to count
            window: Window name

        Returns:
            Count of occurrences
        """
        window_months = WindowCalculator.WINDOWS.get(window, int(window.replace("M", "")))
        window_data = series.tail(window_months)

        return (window_data == value).sum()

    @staticmethod
    def compute_elasticity(
        dependent_series: pd.Series,
        independent_series: pd.Series,
        lag: int = 1
    ) -> float:
        """
        Compute elasticity: % change in Y / % change in X (with lag to prevent simultaneity bias)

        Args:
            dependent_series: Y variable
            independent_series: X variable (lagged)
            lag: Lag periods for independent variable

        Returns:
            Elasticity coefficient
        """
        if len(dependent_series) < 2 or len(independent_series) < 2 + lag:
            return np.nan

        # Lag independent variable
        independent_lagged = independent_series.shift(lag).dropna()

        # Align series
        aligned_dep = dependent_series.loc[independent_lagged.index]

        # Compute % changes
        dep_pct_change = aligned_dep.pct_change().dropna()
        indep_pct_change = independent_lagged.pct_change().dropna()

        # Align again after pct_change
        common_index = dep_pct_change.index.intersection(indep_pct_change.index)

        if len(common_index) < 2:
            return np.nan

        dep_pct = dep_pct_change.loc[common_index]
        indep_pct = indep_pct_change.loc[common_index]

        # Elasticity = correlation weighted by std ratio
        if indep_pct.std() == 0:
            return np.nan

        elasticity = (dep_pct.corr(indep_pct) * dep_pct.std()) / indep_pct.std()

        return elasticity
