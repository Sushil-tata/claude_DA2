"""
Behavioral Features Engineering Module

Implements behavioral feature engineering for financial decision making including:
- Velocity features (rate of change over multiple time windows)
- Momentum features (acceleration and second derivatives)
- Volatility features (std, coefficient of variation, range)
- Elasticity features (sensitivity to changes)
- Stability indices (consistency over time, trend strength)

All features support multiple aggregation windows and are optimized for performance.

Author: Principal Data Science Decision Agent
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from pathlib import Path
from scipy import stats
from sklearn.preprocessing import StandardScaler


class BehavioralFeatureConfig:
    """Configuration for behavioral feature engineering."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize behavioral feature configuration.

        Args:
            config_path: Path to feature_config.yaml. If None, uses defaults.

        Example:
            >>> config = BehavioralFeatureConfig("config/feature_config.yaml")
            >>> config.velocity_windows
            [7, 14, 30, 60, 90]
        """
        if config_path and Path(config_path).exists():
            with open(config_path, "r") as f:
                full_config = yaml.safe_load(f)
                behavioral_config = full_config.get("behavioral", {})
        else:
            behavioral_config = {}

        self.velocity_windows = behavioral_config.get(
            "velocity_windows", [7, 14, 30, 60, 90]
        )
        self.momentum_windows = behavioral_config.get("momentum_windows", [7, 14, 30])
        self.volatility_windows = behavioral_config.get(
            "volatility_windows", [30, 60, 90]
        )
        self.stability_windows = behavioral_config.get(
            "stability_windows", [60, 90, 180]
        )
        self.metrics = behavioral_config.get(
            "metrics",
            ["mean", "median", "std", "cv", "skew", "kurt", "min", "max",
             "percentile_25", "percentile_75"],
        )


class BehavioralFeatureEngine:
    """
    Engine for computing behavioral features from transactional data.

    Features include velocity, momentum, volatility, elasticity, and stability metrics
    across multiple time windows for comprehensive behavioral profiling.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        entity_col: str = "user_id",
        timestamp_col: str = "timestamp",
        value_col: str = "amount",
    ):
        """
        Initialize behavioral feature engine.

        Args:
            config_path: Path to feature_config.yaml
            entity_col: Column name for entity identifier
            timestamp_col: Column name for timestamp
            value_col: Column name for value to compute features on

        Example:
            >>> engine = BehavioralFeatureEngine(
            ...     config_path="config/feature_config.yaml",
            ...     entity_col="customer_id",
            ...     value_col="transaction_amount"
            ... )
        """
        self.config = BehavioralFeatureConfig(config_path)
        self.entity_col = entity_col
        self.timestamp_col = timestamp_col
        self.value_col = value_col
        logger.info("Initialized BehavioralFeatureEngine")

    def validate_data(self, df: pd.DataFrame) -> None:
        """
        Validate input dataframe has required columns and types.

        Args:
            df: Input dataframe

        Raises:
            ValueError: If required columns are missing or invalid
        """
        required_cols = [self.entity_col, self.timestamp_col, self.value_col]
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        if not pd.api.types.is_datetime64_any_dtype(df[self.timestamp_col]):
            raise ValueError(
                f"{self.timestamp_col} must be datetime type. "
                f"Got {df[self.timestamp_col].dtype}"
            )

        if not pd.api.types.is_numeric_dtype(df[self.value_col]):
            raise ValueError(
                f"{self.value_col} must be numeric type. "
                f"Got {df[self.value_col].dtype}"
            )

    def compute_velocity_features(
        self,
        df: pd.DataFrame,
        windows: Optional[List[int]] = None,
        prefix: str = "velocity",
    ) -> pd.DataFrame:
        """
        Compute velocity features (rate of change) over multiple windows.

        Velocity measures how quickly a metric is changing over time.
        Formula: (current_value - past_value) / days_elapsed

        Args:
            df: Input dataframe with entity, timestamp, and value columns
            windows: List of window sizes in days. If None, uses config.
            prefix: Prefix for feature names

        Returns:
            DataFrame with velocity features

        Example:
            >>> df = pd.DataFrame({
            ...     'user_id': [1, 1, 1, 1],
            ...     'timestamp': pd.date_range('2024-01-01', periods=4, freq='D'),
            ...     'amount': [100, 120, 150, 140]
            ... })
            >>> engine = BehavioralFeatureEngine()
            >>> velocity_df = engine.compute_velocity_features(df, windows=[7, 14])
        """
        self.validate_data(df)
        windows = windows or self.config.velocity_windows

        logger.info(f"Computing velocity features for windows: {windows}")

        df_sorted = df.sort_values([self.entity_col, self.timestamp_col])
        features = []

        for window in windows:
            # Group by entity and compute rolling velocity
            grouped = df_sorted.groupby(self.entity_col)

            # Calculate change over window
            value_diff = grouped[self.value_col].diff(periods=1)
            time_diff = grouped[self.timestamp_col].diff(periods=1).dt.days

            # Velocity = change / time
            velocity = value_diff / time_diff.replace(0, np.nan)

            # Rolling window statistics on velocity
            velocity_mean = (
                grouped.apply(
                    lambda x: x.assign(velocity_temp=velocity.loc[x.index])
                    .set_index(self.timestamp_col)["velocity_temp"]
                    .rolling(f"{window}D")
                    .mean()
                )
                .reset_index(level=0, drop=True)
            )

            velocity_std = (
                grouped.apply(
                    lambda x: x.assign(velocity_temp=velocity.loc[x.index])
                    .set_index(self.timestamp_col)["velocity_temp"]
                    .rolling(f"{window}D")
                    .std()
                )
                .reset_index(level=0, drop=True)
            )

            velocity_max = (
                grouped.apply(
                    lambda x: x.assign(velocity_temp=velocity.loc[x.index])
                    .set_index(self.timestamp_col)["velocity_temp"]
                    .rolling(f"{window}D")
                    .max()
                )
                .reset_index(level=0, drop=True)
            )

            velocity_min = (
                grouped.apply(
                    lambda x: x.assign(velocity_temp=velocity.loc[x.index])
                    .set_index(self.timestamp_col)["velocity_temp"]
                    .rolling(f"{window}D")
                    .min()
                )
                .reset_index(level=0, drop=True)
            )

            # Add features to result
            features.append(
                pd.DataFrame(
                    {
                        f"{prefix}_mean_{window}d": velocity_mean,
                        f"{prefix}_std_{window}d": velocity_std,
                        f"{prefix}_max_{window}d": velocity_max,
                        f"{prefix}_min_{window}d": velocity_min,
                        f"{prefix}_range_{window}d": velocity_max - velocity_min,
                    },
                    index=df_sorted.index,
                )
            )

        result_df = pd.concat([df_sorted.reset_index(drop=True)] + features, axis=1)
        logger.info(f"Generated {len(features) * 5} velocity features")

        return result_df

    def compute_momentum_features(
        self,
        df: pd.DataFrame,
        windows: Optional[List[int]] = None,
        prefix: str = "momentum",
    ) -> pd.DataFrame:
        """
        Compute momentum features (acceleration, second derivative).

        Momentum captures whether the rate of change is itself increasing or decreasing.

        Args:
            df: Input dataframe with entity, timestamp, and value columns
            windows: List of window sizes in days. If None, uses config.
            prefix: Prefix for feature names

        Returns:
            DataFrame with momentum features

        Example:
            >>> momentum_df = engine.compute_momentum_features(df, windows=[7, 14])
        """
        self.validate_data(df)
        windows = windows or self.config.momentum_windows

        logger.info(f"Computing momentum features for windows: {windows}")

        df_sorted = df.sort_values([self.entity_col, self.timestamp_col])
        features = []

        for window in windows:
            grouped = df_sorted.groupby(self.entity_col)

            # First derivative (velocity)
            first_diff = grouped[self.value_col].diff()

            # Second derivative (acceleration/momentum)
            second_diff = grouped[self.value_col].diff().diff()

            # Rolling statistics on momentum
            momentum_mean = (
                grouped.apply(
                    lambda x: x.assign(momentum_temp=second_diff.loc[x.index])
                    .set_index(self.timestamp_col)["momentum_temp"]
                    .rolling(f"{window}D")
                    .mean()
                )
                .reset_index(level=0, drop=True)
            )

            momentum_std = (
                grouped.apply(
                    lambda x: x.assign(momentum_temp=second_diff.loc[x.index])
                    .set_index(self.timestamp_col)["momentum_temp"]
                    .rolling(f"{window}D")
                    .std()
                )
                .reset_index(level=0, drop=True)
            )

            # Positive/negative momentum counts
            positive_momentum = (
                grouped.apply(
                    lambda x: x.assign(momentum_temp=second_diff.loc[x.index])
                    .set_index(self.timestamp_col)["momentum_temp"]
                    .rolling(f"{window}D")
                    .apply(lambda x: (x > 0).sum())
                )
                .reset_index(level=0, drop=True)
            )

            negative_momentum = (
                grouped.apply(
                    lambda x: x.assign(momentum_temp=second_diff.loc[x.index])
                    .set_index(self.timestamp_col)["momentum_temp"]
                    .rolling(f"{window}D")
                    .apply(lambda x: (x < 0).sum())
                )
                .reset_index(level=0, drop=True)
            )

            features.append(
                pd.DataFrame(
                    {
                        f"{prefix}_mean_{window}d": momentum_mean,
                        f"{prefix}_std_{window}d": momentum_std,
                        f"{prefix}_positive_count_{window}d": positive_momentum,
                        f"{prefix}_negative_count_{window}d": negative_momentum,
                        f"{prefix}_ratio_{window}d": positive_momentum
                        / (negative_momentum + 1),
                    },
                    index=df_sorted.index,
                )
            )

        result_df = pd.concat([df_sorted.reset_index(drop=True)] + features, axis=1)
        logger.info(f"Generated {len(features) * 5} momentum features")

        return result_df

    def compute_volatility_features(
        self,
        df: pd.DataFrame,
        windows: Optional[List[int]] = None,
        prefix: str = "volatility",
    ) -> pd.DataFrame:
        """
        Compute volatility features (std, coefficient of variation, range).

        Volatility measures the variability and unpredictability of behavior.

        Args:
            df: Input dataframe with entity, timestamp, and value columns
            windows: List of window sizes in days. If None, uses config.
            prefix: Prefix for feature names

        Returns:
            DataFrame with volatility features

        Example:
            >>> volatility_df = engine.compute_volatility_features(df, windows=[30, 60])
        """
        self.validate_data(df)
        windows = windows or self.config.volatility_windows

        logger.info(f"Computing volatility features for windows: {windows}")

        df_sorted = df.sort_values([self.entity_col, self.timestamp_col])
        features = []

        for window in windows:
            grouped = df_sorted.groupby(self.entity_col)

            # Rolling statistics
            rolling_mean = (
                grouped.apply(
                    lambda x: x.set_index(self.timestamp_col)[self.value_col]
                    .rolling(f"{window}D")
                    .mean()
                )
                .reset_index(level=0, drop=True)
            )

            rolling_std = (
                grouped.apply(
                    lambda x: x.set_index(self.timestamp_col)[self.value_col]
                    .rolling(f"{window}D")
                    .std()
                )
                .reset_index(level=0, drop=True)
            )

            rolling_min = (
                grouped.apply(
                    lambda x: x.set_index(self.timestamp_col)[self.value_col]
                    .rolling(f"{window}D")
                    .min()
                )
                .reset_index(level=0, drop=True)
            )

            rolling_max = (
                grouped.apply(
                    lambda x: x.set_index(self.timestamp_col)[self.value_col]
                    .rolling(f"{window}D")
                    .max()
                )
                .reset_index(level=0, drop=True)
            )

            # Coefficient of variation (CV = std / mean)
            cv = rolling_std / (rolling_mean.abs() + 1e-10)

            # Range
            value_range = rolling_max - rolling_min

            # Interquartile range
            iqr = (
                grouped.apply(
                    lambda x: x.set_index(self.timestamp_col)[self.value_col]
                    .rolling(f"{window}D")
                    .quantile(0.75)
                    - x.set_index(self.timestamp_col)[self.value_col]
                    .rolling(f"{window}D")
                    .quantile(0.25)
                )
                .reset_index(level=0, drop=True)
            )

            features.append(
                pd.DataFrame(
                    {
                        f"{prefix}_std_{window}d": rolling_std,
                        f"{prefix}_cv_{window}d": cv,
                        f"{prefix}_range_{window}d": value_range,
                        f"{prefix}_iqr_{window}d": iqr,
                        f"{prefix}_relative_range_{window}d": value_range
                        / (rolling_mean.abs() + 1e-10),
                    },
                    index=df_sorted.index,
                )
            )

        result_df = pd.concat([df_sorted.reset_index(drop=True)] + features, axis=1)
        logger.info(f"Generated {len(features) * 5} volatility features")

        return result_df

    def compute_elasticity_features(
        self,
        df: pd.DataFrame,
        reference_col: str,
        windows: Optional[List[int]] = None,
        prefix: str = "elasticity",
    ) -> pd.DataFrame:
        """
        Compute elasticity features (sensitivity to changes in reference variable).

        Elasticity = % change in value / % change in reference

        Args:
            df: Input dataframe
            reference_col: Column to measure elasticity against (e.g., 'limit', 'income')
            windows: List of window sizes in days
            prefix: Prefix for feature names

        Returns:
            DataFrame with elasticity features

        Example:
            >>> # Measure spending elasticity to credit limit changes
            >>> df['credit_limit'] = 10000
            >>> elasticity_df = engine.compute_elasticity_features(
            ...     df, reference_col='credit_limit', windows=[30]
            ... )
        """
        self.validate_data(df)

        if reference_col not in df.columns:
            raise ValueError(f"Reference column '{reference_col}' not found in dataframe")

        windows = windows or [30, 60, 90]

        logger.info(f"Computing elasticity features for windows: {windows}")

        df_sorted = df.sort_values([self.entity_col, self.timestamp_col])
        features = []

        for window in windows:
            grouped = df_sorted.groupby(self.entity_col)

            # Percentage changes
            value_pct_change = grouped[self.value_col].pct_change()
            reference_pct_change = grouped[reference_col].pct_change()

            # Elasticity = % change in value / % change in reference
            elasticity = value_pct_change / (reference_pct_change.abs() + 1e-10)

            # Rolling statistics on elasticity
            elasticity_mean = (
                grouped.apply(
                    lambda x: x.assign(elasticity_temp=elasticity.loc[x.index])
                    .set_index(self.timestamp_col)["elasticity_temp"]
                    .rolling(f"{window}D")
                    .mean()
                )
                .reset_index(level=0, drop=True)
            )

            elasticity_std = (
                grouped.apply(
                    lambda x: x.assign(elasticity_temp=elasticity.loc[x.index])
                    .set_index(self.timestamp_col)["elasticity_temp"]
                    .rolling(f"{window}D")
                    .std()
                )
                .reset_index(level=0, drop=True)
            )

            # High elasticity events (abs(elasticity) > 2)
            high_elasticity_count = (
                grouped.apply(
                    lambda x: x.assign(elasticity_temp=elasticity.loc[x.index])
                    .set_index(self.timestamp_col)["elasticity_temp"]
                    .rolling(f"{window}D")
                    .apply(lambda x: (x.abs() > 2).sum())
                )
                .reset_index(level=0, drop=True)
            )

            features.append(
                pd.DataFrame(
                    {
                        f"{prefix}_mean_{window}d": elasticity_mean,
                        f"{prefix}_std_{window}d": elasticity_std,
                        f"{prefix}_high_events_{window}d": high_elasticity_count,
                    },
                    index=df_sorted.index,
                )
            )

        result_df = pd.concat([df_sorted.reset_index(drop=True)] + features, axis=1)
        logger.info(f"Generated {len(features) * 3} elasticity features")

        return result_df

    def compute_stability_features(
        self,
        df: pd.DataFrame,
        windows: Optional[List[int]] = None,
        prefix: str = "stability",
    ) -> pd.DataFrame:
        """
        Compute stability indices (consistency over time, trend strength).

        Stability measures how consistent and predictable behavior is.

        Args:
            df: Input dataframe with entity, timestamp, and value columns
            windows: List of window sizes in days. If None, uses config.
            prefix: Prefix for feature names

        Returns:
            DataFrame with stability features

        Example:
            >>> stability_df = engine.compute_stability_features(df, windows=[60, 90])
        """
        self.validate_data(df)
        windows = windows or self.config.stability_windows

        logger.info(f"Computing stability features for windows: {windows}")

        df_sorted = df.sort_values([self.entity_col, self.timestamp_col])
        features = []

        for window in windows:
            grouped = df_sorted.groupby(self.entity_col)

            # Trend strength (R-squared from linear regression)
            def compute_trend_strength(series):
                """Compute R² for trend strength."""
                if len(series) < 2:
                    return np.nan
                x = np.arange(len(series))
                y = series.values
                if np.all(np.isnan(y)):
                    return np.nan
                mask = ~np.isnan(y)
                if mask.sum() < 2:
                    return np.nan
                correlation = np.corrcoef(x[mask], y[mask])[0, 1]
                return correlation ** 2 if not np.isnan(correlation) else np.nan

            trend_strength = (
                grouped.apply(
                    lambda x: x.set_index(self.timestamp_col)[self.value_col]
                    .rolling(f"{window}D")
                    .apply(compute_trend_strength, raw=False)
                )
                .reset_index(level=0, drop=True)
            )

            # Consistency index (1 - CV)
            rolling_mean = (
                grouped.apply(
                    lambda x: x.set_index(self.timestamp_col)[self.value_col]
                    .rolling(f"{window}D")
                    .mean()
                )
                .reset_index(level=0, drop=True)
            )

            rolling_std = (
                grouped.apply(
                    lambda x: x.set_index(self.timestamp_col)[self.value_col]
                    .rolling(f"{window}D")
                    .std()
                )
                .reset_index(level=0, drop=True)
            )

            consistency_index = 1 - (
                rolling_std / (rolling_mean.abs() + 1e-10)
            ).clip(upper=1)

            # Regularity (inverse of time between events)
            time_diff = grouped[self.timestamp_col].diff().dt.days

            regularity = (
                grouped.apply(
                    lambda x: x.assign(time_diff_temp=time_diff.loc[x.index])
                    .set_index(self.timestamp_col)["time_diff_temp"]
                    .rolling(f"{window}D")
                    .std()
                )
                .reset_index(level=0, drop=True)
            )

            regularity_index = 1 / (regularity + 1)

            # Predictability (autocorrelation at lag 1)
            autocorr = (
                grouped.apply(
                    lambda x: x.set_index(self.timestamp_col)[self.value_col]
                    .rolling(f"{window}D")
                    .apply(lambda s: s.autocorr(lag=1) if len(s) > 1 else np.nan, raw=False)
                )
                .reset_index(level=0, drop=True)
            )

            features.append(
                pd.DataFrame(
                    {
                        f"{prefix}_trend_strength_{window}d": trend_strength,
                        f"{prefix}_consistency_{window}d": consistency_index,
                        f"{prefix}_regularity_{window}d": regularity_index,
                        f"{prefix}_predictability_{window}d": autocorr,
                        f"{prefix}_composite_{window}d": (
                            trend_strength + consistency_index + autocorr.fillna(0)
                        )
                        / 3,
                    },
                    index=df_sorted.index,
                )
            )

        result_df = pd.concat([df_sorted.reset_index(drop=True)] + features, axis=1)
        logger.info(f"Generated {len(features) * 5} stability features")

        return result_df

    def compute_all_features(
        self,
        df: pd.DataFrame,
        include_velocity: bool = True,
        include_momentum: bool = True,
        include_volatility: bool = True,
        include_stability: bool = True,
    ) -> pd.DataFrame:
        """
        Compute all behavioral features at once.

        Args:
            df: Input dataframe with entity, timestamp, and value columns
            include_velocity: Whether to include velocity features
            include_momentum: Whether to include momentum features
            include_volatility: Whether to include volatility features
            include_stability: Whether to include stability features

        Returns:
            DataFrame with all requested behavioral features

        Example:
            >>> df = pd.DataFrame({
            ...     'user_id': [1, 1, 1, 1, 2, 2, 2],
            ...     'timestamp': pd.date_range('2024-01-01', periods=7, freq='D'),
            ...     'amount': [100, 120, 150, 140, 200, 180, 220]
            ... })
            >>> engine = BehavioralFeatureEngine()
            >>> all_features = engine.compute_all_features(df)
        """
        logger.info("Computing all behavioral features")
        self.validate_data(df)

        result_df = df.copy()

        if include_velocity:
            velocity_df = self.compute_velocity_features(df)
            velocity_cols = [
                col for col in velocity_df.columns if col.startswith("velocity_")
            ]
            result_df = result_df.merge(
                velocity_df[[self.entity_col, self.timestamp_col] + velocity_cols],
                on=[self.entity_col, self.timestamp_col],
                how="left",
            )

        if include_momentum:
            momentum_df = self.compute_momentum_features(df)
            momentum_cols = [
                col for col in momentum_df.columns if col.startswith("momentum_")
            ]
            result_df = result_df.merge(
                momentum_df[[self.entity_col, self.timestamp_col] + momentum_cols],
                on=[self.entity_col, self.timestamp_col],
                how="left",
            )

        if include_volatility:
            volatility_df = self.compute_volatility_features(df)
            volatility_cols = [
                col for col in volatility_df.columns if col.startswith("volatility_")
            ]
            result_df = result_df.merge(
                volatility_df[[self.entity_col, self.timestamp_col] + volatility_cols],
                on=[self.entity_col, self.timestamp_col],
                how="left",
            )

        if include_stability:
            stability_df = self.compute_stability_features(df)
            stability_cols = [
                col for col in stability_df.columns if col.startswith("stability_")
            ]
            result_df = result_df.merge(
                stability_df[[self.entity_col, self.timestamp_col] + stability_cols],
                on=[self.entity_col, self.timestamp_col],
                how="left",
            )

        logger.info(
            f"Generated {len(result_df.columns) - len(df.columns)} behavioral features"
        )

        return result_df


if __name__ == "__main__":
    # Example usage
    logger.info("Running behavioral features example")

    # Create sample data
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    df = pd.DataFrame(
        {
            "user_id": np.repeat([1, 2, 3], [34, 33, 33]),
            "timestamp": dates,
            "amount": np.random.gamma(2, 50, 100) + np.linspace(0, 100, 100),
        }
    )

    # Initialize engine
    engine = BehavioralFeatureEngine()

    # Compute all features
    features_df = engine.compute_all_features(df)

    logger.info(f"Generated features shape: {features_df.shape}")
    logger.info(f"Feature columns: {features_df.columns.tolist()}")
