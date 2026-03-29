"""
Null Handler - Graceful handling of missing data, edge cases, new accounts

Ensures BFE repository fails gracefully rather than raising errors.
"""

import numpy as np
import pandas as pd
from typing import Any, Optional, Union, Dict
import warnings


class NullHandler:
    """
    Handles missing data and edge cases in feature computation.

    Philosophy:
    - Return NaN with metadata rather than raise errors
    - Distinguish between "missing data" and "not applicable"
    - Special handling for new accounts (<3 months vintage)
    """

    @staticmethod
    def safe_divide(
        numerator: Union[float, pd.Series],
        denominator: Union[float, pd.Series],
        default: float = np.nan
    ) -> Union[float, pd.Series]:
        """
        Safely divide, handling zero denominators.

        Args:
            numerator: Numerator value(s)
            denominator: Denominator value(s)
            default: Value to return when denominator is zero

        Returns:
            Result of division or default
        """
        if isinstance(numerator, pd.Series) or isinstance(denominator, pd.Series):
            # Series division
            result = numerator / denominator
            result = result.replace([np.inf, -np.inf], default)
            return result
        else:
            # Scalar division
            if denominator == 0 or pd.isna(denominator):
                return default
            return numerator / denominator

    @staticmethod
    def safe_percentage(
        part: Union[float, pd.Series],
        whole: Union[float, pd.Series],
        default: float = 0.0
    ) -> Union[float, pd.Series]:
        """
        Safely compute percentage, handling zero whole.

        Args:
            part: Part value(s)
            whole: Whole value(s)
            default: Value to return when whole is zero

        Returns:
            Percentage (0-100) or default
        """
        result = NullHandler.safe_divide(part, whole, default=np.nan) * 100

        if isinstance(result, pd.Series):
            result = result.fillna(default)
        elif pd.isna(result):
            result = default

        return result

    @staticmethod
    def handle_insufficient_history(
        feature_value: Any,
        months_of_history: int,
        min_required_months: int,
        flag_suffix: str = ""
    ) -> Dict[str, Any]:
        """
        Handle features for accounts with insufficient history.

        Args:
            feature_value: Computed feature value
            months_of_history: Actual months of history
            min_required_months: Minimum required months
            flag_suffix: Suffix for flag column name

        Returns:
            Dictionary with feature value and metadata
        """
        has_sufficient_history = months_of_history >= min_required_months

        return {
            "value": feature_value if has_sufficient_history else np.nan,
            f"insufficient_history_flag{flag_suffix}": not has_sufficient_history,
            f"months_of_history{flag_suffix}": months_of_history,
            f"min_required_months{flag_suffix}": min_required_months
        }

    @staticmethod
    def handle_new_account(
        account_age_months: int,
        new_account_threshold: int = 3
    ) -> Dict[str, Any]:
        """
        Identify and flag new accounts.

        Args:
            account_age_months: Age of account in months
            new_account_threshold: Threshold for new account (default 3 months)

        Returns:
            Dictionary with new account flags
        """
        is_new_account = account_age_months < new_account_threshold

        return {
            "new_account_flag": is_new_account,
            "account_age_months": account_age_months,
            "behavioral_maturity": "immature" if is_new_account else "mature"
        }

    @staticmethod
    def impute_missing(
        value: Any,
        imputation_strategy: str = "nan",
        default_value: Any = None
    ) -> Any:
        """
        Impute missing values based on strategy.

        Args:
            value: Value to check and impute
            imputation_strategy: Strategy ("nan", "zero", "default", "forward_fill")
            default_value: Value to use for "default" strategy

        Returns:
            Imputed value
        """
        if not pd.isna(value):
            return value

        if imputation_strategy == "nan":
            return np.nan
        elif imputation_strategy == "zero":
            return 0.0
        elif imputation_strategy == "default":
            return default_value if default_value is not None else np.nan
        else:
            return np.nan

    @staticmethod
    def validate_range(
        value: Union[float, pd.Series],
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        clip: bool = False
    ) -> Union[float, pd.Series]:
        """
        Validate value is within expected range.

        Args:
            value: Value(s) to validate
            min_value: Minimum acceptable value
            max_value: Maximum acceptable value
            clip: If True, clip to range. If False, set out-of-range to NaN

        Returns:
            Validated value(s)
        """
        if isinstance(value, pd.Series):
            result = value.copy()

            if min_value is not None:
                if clip:
                    result = result.clip(lower=min_value)
                else:
                    result[result < min_value] = np.nan

            if max_value is not None:
                if clip:
                    result = result.clip(upper=max_value)
                else:
                    result[result > max_value] = np.nan

            return result
        else:
            if pd.isna(value):
                return value

            if min_value is not None and value < min_value:
                return min_value if clip else np.nan

            if max_value is not None and value > max_value:
                return max_value if clip else np.nan

            return value

    @staticmethod
    def winsorize(
        series: pd.Series,
        lower_percentile: float = 0.01,
        upper_percentile: float = 0.99
    ) -> pd.Series:
        """
        Winsorize series to handle outliers.

        Args:
            series: Series to winsorize
            lower_percentile: Lower percentile (e.g., 0.01 for 1%)
            upper_percentile: Upper percentile (e.g., 0.99 for 99%)

        Returns:
            Winsorized series
        """
        lower_bound = series.quantile(lower_percentile)
        upper_bound = series.quantile(upper_percentile)

        return series.clip(lower=lower_bound, upper=upper_bound)

    @staticmethod
    def handle_zero_variance(
        series: pd.Series,
        threshold: float = 1e-10
    ) -> Dict[str, Any]:
        """
        Detect and flag features with zero or near-zero variance.

        Args:
            series: Series to check
            threshold: Variance threshold below which variance is considered zero

        Returns:
            Dictionary with variance info and flag
        """
        variance = series.var()
        is_zero_variance = variance < threshold

        if is_zero_variance:
            warnings.warn(
                f"Feature has zero or near-zero variance ({variance:.2e}). "
                f"This feature may not be informative."
            )

        return {
            "variance": variance,
            "zero_variance_flag": is_zero_variance,
            "unique_values": series.nunique()
        }

    @staticmethod
    def create_null_metadata(
        value: Any,
        feature_name: str,
        reason: str = "missing_data"
    ) -> Dict[str, Any]:
        """
        Create metadata for null feature values.

        Args:
            value: Feature value (usually NaN)
            feature_name: Name of feature
            reason: Reason for null ("missing_data", "insufficient_history", "not_applicable")

        Returns:
            Dictionary with value and metadata
        """
        return {
            "feature_name": feature_name,
            "value": value,
            "is_null": pd.isna(value),
            "null_reason": reason if pd.isna(value) else None
        }

    @staticmethod
    def coalesce(*values: Any) -> Any:
        """
        Return first non-null value (SQL COALESCE).

        Args:
            *values: Variable number of values to check

        Returns:
            First non-null value or NaN if all null
        """
        for val in values:
            if not pd.isna(val):
                return val
        return np.nan

    @staticmethod
    def safe_log(
        value: Union[float, pd.Series],
        base: float = np.e,
        offset: float = 1.0
    ) -> Union[float, pd.Series]:
        """
        Safely compute logarithm, handling zero and negative values.

        Uses log(value + offset) to avoid log(0).

        Args:
            value: Value(s) to transform
            base: Logarithm base (default: natural log)
            offset: Offset to add before log (default: 1.0)

        Returns:
            Log-transformed value(s)
        """
        if isinstance(value, pd.Series):
            adjusted = value + offset
            result = np.log(adjusted) / np.log(base)
            return result.replace([np.inf, -np.inf], np.nan)
        else:
            if pd.isna(value):
                return np.nan
            adjusted = value + offset
            if adjusted <= 0:
                return np.nan
            return np.log(adjusted) / np.log(base)

    @staticmethod
    def fillna_with_context(
        series: pd.Series,
        strategy: str = "median",
        group_by: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Fill missing values with context-aware imputation.

        Args:
            series: Series with missing values
            strategy: Imputation strategy ("median", "mean", "mode", "zero")
            group_by: Optional grouping series for group-wise imputation

        Returns:
            Series with imputed values
        """
        if group_by is not None:
            # Group-wise imputation
            if strategy == "median":
                return series.fillna(series.groupby(group_by).transform("median"))
            elif strategy == "mean":
                return series.fillna(series.groupby(group_by).transform("mean"))
            elif strategy == "mode":
                return series.fillna(series.groupby(group_by).transform(lambda x: x.mode()[0] if len(x.mode()) > 0 else np.nan))
            elif strategy == "zero":
                return series.fillna(0)
        else:
            # Global imputation
            if strategy == "median":
                return series.fillna(series.median())
            elif strategy == "mean":
                return series.fillna(series.mean())
            elif strategy == "mode":
                mode_val = series.mode()[0] if len(series.mode()) > 0 else 0
                return series.fillna(mode_val)
            elif strategy == "zero":
                return series.fillna(0)

        return series
