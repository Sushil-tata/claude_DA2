"""
Temporal Validator - Enforces point-in-time safety and prevents look-ahead bias

Critical for ensuring features use only data available at decision time.
"""

import pandas as pd
import numpy as np
from datetime import datetime, date
from typing import Optional, Union, List
import warnings


class TemporalValidator:
    """
    Validates and enforces temporal constraints for feature computation.

    Prevents look-ahead bias by ensuring all data used for feature engineering
    has timestamps <= as_of_date.
    """

    @staticmethod
    def validate_no_future_data(
        data: pd.DataFrame,
        as_of_date: Union[str, datetime, date],
        timestamp_col: str = "date"
    ) -> bool:
        """
        Validate that no data points are from after as_of_date.

        Args:
            data: DataFrame with temporal data
            as_of_date: Reference date (decision time)
            timestamp_col: Name of timestamp column

        Returns:
            True if valid, raises ValueError if future data detected

        Raises:
            ValueError: If future data detected
        """
        as_of_dt = pd.to_datetime(as_of_date)

        if timestamp_col not in data.columns:
            raise ValueError(f"Timestamp column '{timestamp_col}' not found in data")

        data_dates = pd.to_datetime(data[timestamp_col])

        future_rows = data_dates > as_of_dt

        if future_rows.any():
            future_count = future_rows.sum()
            future_dates = data_dates[future_rows].unique()

            raise ValueError(
                f"LOOK-AHEAD BIAS DETECTED!\n"
                f"Found {future_count} rows with dates after as_of_date={as_of_date}\n"
                f"Future dates found: {future_dates[:5].tolist()}"
            )

        return True

    @staticmethod
    def filter_to_as_of_date(
        data: pd.DataFrame,
        as_of_date: Union[str, datetime, date],
        timestamp_col: str = "date",
        strict: bool = True
    ) -> pd.DataFrame:
        """
        Filter data to include only records <= as_of_date.

        Args:
            data: DataFrame with temporal data
            as_of_date: Reference date
            timestamp_col: Name of timestamp column
            strict: If True, raise error if future data found. If False, silently filter.

        Returns:
            Filtered DataFrame
        """
        as_of_dt = pd.to_datetime(as_of_date)
        data_dates = pd.to_datetime(data[timestamp_col])

        if strict:
            TemporalValidator.validate_no_future_data(data, as_of_date, timestamp_col)

        return data[data_dates <= as_of_dt].copy()

    @staticmethod
    def compute_data_lag(
        data_timestamp: Union[str, datetime, date],
        as_of_date: Union[str, datetime, date]
    ) -> int:
        """
        Compute lag in days between data timestamp and as_of_date.

        Args:
            data_timestamp: Timestamp of data
            as_of_date: Reference date

        Returns:
            Number of days lag (positive = data is older)
        """
        data_dt = pd.to_datetime(data_timestamp)
        as_of_dt = pd.to_datetime(as_of_date)

        lag_days = (as_of_dt - data_dt).days

        return lag_days

    @staticmethod
    def validate_minimum_history(
        data: pd.DataFrame,
        min_months: int,
        timestamp_col: str = "date",
        account_id_col: str = "account_id"
    ) -> pd.DataFrame:
        """
        Validate that each account has minimum required history.

        Args:
            data: DataFrame with temporal data
            min_months: Minimum required months of history
            timestamp_col: Name of timestamp column
            account_id_col: Name of account ID column

        Returns:
            DataFrame filtered to accounts with sufficient history
        """
        data_sorted = data.sort_values([account_id_col, timestamp_col])

        # Compute months of history per account
        history_lengths = []

        for account_id, group in data_sorted.groupby(account_id_col):
            dates = pd.to_datetime(group[timestamp_col])
            min_date = dates.min()
            max_date = dates.max()

            months_of_history = (max_date.year - min_date.year) * 12 + (max_date.month - min_date.month)

            history_lengths.append({
                account_id_col: account_id,
                "months_of_history": months_of_history,
                "has_sufficient_history": months_of_history >= min_months
            })

        history_df = pd.DataFrame(history_lengths)

        # Filter to accounts with sufficient history
        valid_accounts = history_df[history_df["has_sufficient_history"]][account_id_col]
        filtered_data = data[data[account_id_col].isin(valid_accounts)]

        if len(filtered_data) < len(data):
            warnings.warn(
                f"Filtered {len(data) - len(filtered_data)} records from "
                f"{len(data[account_id_col].unique()) - len(valid_accounts)} accounts "
                f"with insufficient history (<{min_months} months)"
            )

        return filtered_data

    @staticmethod
    def create_monthly_snapshot(
        data: pd.DataFrame,
        as_of_date: Union[str, datetime, date],
        lookback_months: int,
        timestamp_col: str = "date",
        account_id_col: str = "account_id"
    ) -> pd.DataFrame:
        """
        Create monthly snapshots for rolling window calculations.

        Args:
            data: DataFrame with temporal data
            as_of_date: Reference date
            lookback_months: Number of months to include
            timestamp_col: Name of timestamp column
            account_id_col: Name of account ID column

        Returns:
            DataFrame with monthly snapshots
        """
        as_of_dt = pd.to_datetime(as_of_date)

        # Calculate start date
        start_date = as_of_dt - pd.DateOffset(months=lookback_months)

        # Filter data
        data_dates = pd.to_datetime(data[timestamp_col])
        windowed_data = data[(data_dates >= start_date) & (data_dates <= as_of_dt)].copy()

        # Create monthly periods
        windowed_data["month_period"] = pd.to_datetime(windowed_data[timestamp_col]).dt.to_period("M")

        return windowed_data

    @staticmethod
    def validate_temporal_order(
        data: pd.DataFrame,
        timestamp_col: str = "date",
        account_id_col: str = "account_id"
    ) -> bool:
        """
        Validate that data is properly ordered by time within each account.

        Args:
            data: DataFrame with temporal data
            timestamp_col: Name of timestamp column
            account_id_col: Name of account ID column

        Returns:
            True if valid, raises ValueError if ordering issues detected
        """
        for account_id, group in data.groupby(account_id_col):
            dates = pd.to_datetime(group[timestamp_col])

            if not dates.is_monotonic_increasing:
                raise ValueError(
                    f"Temporal ordering violation detected for account {account_id}. "
                    f"Data must be sorted by {timestamp_col} within each account."
                )

        return True

    @staticmethod
    def check_staleness(
        last_update_date: Union[str, datetime, date],
        as_of_date: Union[str, datetime, date],
        max_staleness_days: int = 90
    ) -> dict:
        """
        Check if data is stale (e.g., bureau data older than 90 days).

        Args:
            last_update_date: Date of last data update
            as_of_date: Reference date
            max_staleness_days: Maximum acceptable staleness

        Returns:
            Dictionary with staleness info
        """
        lag_days = TemporalValidator.compute_data_lag(last_update_date, as_of_date)

        is_stale = lag_days > max_staleness_days

        return {
            "last_update_date": pd.to_datetime(last_update_date),
            "as_of_date": pd.to_datetime(as_of_date),
            "staleness_days": lag_days,
            "is_stale": is_stale,
            "staleness_flag": "STALE" if is_stale else "FRESH"
        }

    @staticmethod
    def create_lag_indicator(
        data: pd.DataFrame,
        as_of_date: Union[str, datetime, date],
        timestamp_col: str = "date",
        lag_thresholds: List[int] = None
    ) -> pd.DataFrame:
        """
        Create lag indicator columns based on thresholds.

        Args:
            data: DataFrame with temporal data
            as_of_date: Reference date
            timestamp_col: Name of timestamp column
            lag_thresholds: List of thresholds in days (e.g., [30, 90, 180])

        Returns:
            DataFrame with lag indicator columns added
        """
        lag_thresholds = lag_thresholds or [30, 90, 180]

        as_of_dt = pd.to_datetime(as_of_date)
        data_dates = pd.to_datetime(data[timestamp_col])

        result = data.copy()
        result["staleness_days"] = (as_of_dt - data_dates).dt.days

        for threshold in lag_thresholds:
            col_name = f"lag_gt_{threshold}d"
            result[col_name] = result["staleness_days"] > threshold

        return result
