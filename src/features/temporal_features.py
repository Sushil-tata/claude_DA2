"""
Temporal Features Engineering Module

Implements comprehensive temporal feature engineering including:
- Rolling window features (short/medium/long term)
- Lag features (historical values at different time points)
- Lead features (forward-looking features)
- Overlapping window features
- Multi-resolution signals (daily, weekly, monthly)
- Event-triggered windows
- Trend and seasonality extraction

Author: Principal Data Science Decision Agent
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from pathlib import Path
from scipy import signal, stats
from statsmodels.tsa.seasonal import seasonal_decompose


class TemporalFeatureConfig:
    """Configuration for temporal feature engineering."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize temporal feature configuration.

        Args:
            config_path: Path to feature_config.yaml. If None, uses defaults.

        Example:
            >>> config = TemporalFeatureConfig("config/feature_config.yaml")
            >>> config.rolling_windows
            {'short_term': [7, 14], 'medium_term': [30, 60], 'long_term': [90, 180]}
        """
        if config_path and Path(config_path).exists():
            with open(config_path, "r") as f:
                full_config = yaml.safe_load(f)
                temporal_config = full_config.get("temporal", {})
        else:
            temporal_config = {}

        self.rolling_windows = temporal_config.get(
            "rolling_windows",
            {
                "short_term": [7, 14],
                "medium_term": [30, 60],
                "long_term": [90, 180],
            },
        )
        self.lag_features = temporal_config.get("lag_features", [1, 7, 14, 30])
        self.lead_features = temporal_config.get("lead_features", [7, 14, 30])
        self.overlapping_windows = temporal_config.get(
            "overlapping_windows", {"enabled": True, "overlap_ratio": 0.5}
        )
        self.multi_resolution = temporal_config.get(
            "multi_resolution",
            {"enabled": True, "resolutions": ["daily", "weekly", "monthly"]},
        )


class TemporalFeatureEngine:
    """
    Engine for computing temporal features from time-series data.

    Provides comprehensive temporal feature extraction including rolling windows,
    lags, leads, multi-resolution aggregations, and trend/seasonality decomposition.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        entity_col: str = "user_id",
        timestamp_col: str = "timestamp",
        value_col: str = "amount",
    ):
        """
        Initialize temporal feature engine.

        Args:
            config_path: Path to feature_config.yaml
            entity_col: Column name for entity identifier
            timestamp_col: Column name for timestamp
            value_col: Column name for value to compute features on

        Example:
            >>> engine = TemporalFeatureEngine(
            ...     config_path="config/feature_config.yaml",
            ...     entity_col="customer_id",
            ...     value_col="transaction_amount"
            ... )
        """
        self.config = TemporalFeatureConfig(config_path)
        self.entity_col = entity_col
        self.timestamp_col = timestamp_col
        self.value_col = value_col
        logger.info("Initialized TemporalFeatureEngine")

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

    def compute_rolling_window_features(
        self,
        df: pd.DataFrame,
        windows: Optional[Dict[str, List[int]]] = None,
        aggregations: Optional[List[str]] = None,
        prefix: str = "rolling",
    ) -> pd.DataFrame:
        """
        Compute rolling window features across multiple time horizons.

        Args:
            df: Input dataframe with entity, timestamp, and value columns
            windows: Dict of window categories and sizes. If None, uses config.
            aggregations: List of aggregation functions to apply
            prefix: Prefix for feature names

        Returns:
            DataFrame with rolling window features

        Example:
            >>> df = pd.DataFrame({
            ...     'user_id': [1, 1, 1, 1],
            ...     'timestamp': pd.date_range('2024-01-01', periods=4, freq='D'),
            ...     'amount': [100, 120, 150, 140]
            ... })
            >>> engine = TemporalFeatureEngine()
            >>> rolling_df = engine.compute_rolling_window_features(df)
        """
        self.validate_data(df)
        windows = windows or self.config.rolling_windows
        aggregations = aggregations or ["mean", "std", "min", "max", "sum", "count"]

        logger.info(f"Computing rolling window features for windows: {windows}")

        df_sorted = df.sort_values([self.entity_col, self.timestamp_col])
        features = []

        # Flatten windows dict
        all_windows = []
        for category, window_list in windows.items():
            all_windows.extend([(w, category) for w in window_list])

        for window, category in all_windows:
            grouped = df_sorted.groupby(self.entity_col)

            for agg in aggregations:
                if agg == "mean":
                    feature = (
                        grouped.apply(
                            lambda x: x.set_index(self.timestamp_col)[self.value_col]
                            .rolling(f"{window}D")
                            .mean()
                        )
                        .reset_index(level=0, drop=True)
                    )
                elif agg == "std":
                    feature = (
                        grouped.apply(
                            lambda x: x.set_index(self.timestamp_col)[self.value_col]
                            .rolling(f"{window}D")
                            .std()
                        )
                        .reset_index(level=0, drop=True)
                    )
                elif agg == "min":
                    feature = (
                        grouped.apply(
                            lambda x: x.set_index(self.timestamp_col)[self.value_col]
                            .rolling(f"{window}D")
                            .min()
                        )
                        .reset_index(level=0, drop=True)
                    )
                elif agg == "max":
                    feature = (
                        grouped.apply(
                            lambda x: x.set_index(self.timestamp_col)[self.value_col]
                            .rolling(f"{window}D")
                            .max()
                        )
                        .reset_index(level=0, drop=True)
                    )
                elif agg == "sum":
                    feature = (
                        grouped.apply(
                            lambda x: x.set_index(self.timestamp_col)[self.value_col]
                            .rolling(f"{window}D")
                            .sum()
                        )
                        .reset_index(level=0, drop=True)
                    )
                elif agg == "count":
                    feature = (
                        grouped.apply(
                            lambda x: x.set_index(self.timestamp_col)[self.value_col]
                            .rolling(f"{window}D")
                            .count()
                        )
                        .reset_index(level=0, drop=True)
                    )
                else:
                    continue

                features.append(
                    pd.DataFrame(
                        {f"{prefix}_{agg}_{window}d_{category}": feature},
                        index=df_sorted.index,
                    )
                )

        result_df = pd.concat([df_sorted.reset_index(drop=True)] + features, axis=1)
        logger.info(
            f"Generated {len(features)} rolling window features across {len(all_windows)} windows"
        )

        return result_df

    def compute_lag_features(
        self,
        df: pd.DataFrame,
        lags: Optional[List[int]] = None,
        prefix: str = "lag",
    ) -> pd.DataFrame:
        """
        Compute lag features (historical values at different time points).

        Args:
            df: Input dataframe with entity, timestamp, and value columns
            lags: List of lag periods in days. If None, uses config.
            prefix: Prefix for feature names

        Returns:
            DataFrame with lag features

        Example:
            >>> lag_df = engine.compute_lag_features(df, lags=[1, 7, 14])
        """
        self.validate_data(df)
        lags = lags or self.config.lag_features

        logger.info(f"Computing lag features for lags: {lags}")

        df_sorted = df.sort_values([self.entity_col, self.timestamp_col])
        features = []

        for lag in lags:
            grouped = df_sorted.groupby(self.entity_col)

            # Simple lag by number of periods
            lag_feature = grouped[self.value_col].shift(lag)

            # Time-aware lag (value from approximately N days ago)
            df_sorted_copy = df_sorted.copy()
            df_sorted_copy["_lag_date"] = df_sorted_copy[self.timestamp_col] - pd.Timedelta(
                days=lag
            )

            # Merge to get closest value from lag days ago
            time_aware_lag = (
                df_sorted_copy.merge(
                    df_sorted[[self.entity_col, self.timestamp_col, self.value_col]],
                    left_on=[self.entity_col, "_lag_date"],
                    right_on=[self.entity_col, self.timestamp_col],
                    how="left",
                    suffixes=("", "_lag"),
                )
                .groupby([self.entity_col, self.timestamp_col])
                .first()[f"{self.value_col}_lag"]
            )

            features.append(
                pd.DataFrame(
                    {
                        f"{prefix}_{lag}d": lag_feature,
                        f"{prefix}_{lag}d_time_aware": time_aware_lag,
                    },
                    index=df_sorted.index,
                )
            )

        result_df = pd.concat([df_sorted.reset_index(drop=True)] + features, axis=1)
        logger.info(f"Generated {len(features) * 2} lag features")

        return result_df

    def compute_lead_features(
        self,
        df: pd.DataFrame,
        leads: Optional[List[int]] = None,
        prefix: str = "lead",
    ) -> pd.DataFrame:
        """
        Compute lead features (forward-looking features).

        Note: Use with caution to avoid data leakage. Only appropriate for
        certain use cases like nowcasting or when future values are known.

        Args:
            df: Input dataframe with entity, timestamp, and value columns
            leads: List of lead periods in days. If None, uses config.
            prefix: Prefix for feature names

        Returns:
            DataFrame with lead features

        Example:
            >>> lead_df = engine.compute_lead_features(df, leads=[7, 14])
        """
        self.validate_data(df)
        leads = leads or self.config.lead_features

        logger.warning("Computing lead features - ensure no data leakage!")
        logger.info(f"Computing lead features for leads: {leads}")

        df_sorted = df.sort_values([self.entity_col, self.timestamp_col])
        features = []

        for lead in leads:
            grouped = df_sorted.groupby(self.entity_col)

            # Simple lead by number of periods
            lead_feature = grouped[self.value_col].shift(-lead)

            features.append(
                pd.DataFrame(
                    {f"{prefix}_{lead}d": lead_feature},
                    index=df_sorted.index,
                )
            )

        result_df = pd.concat([df_sorted.reset_index(drop=True)] + features, axis=1)
        logger.info(f"Generated {len(features)} lead features")

        return result_df

    def compute_overlapping_window_features(
        self,
        df: pd.DataFrame,
        base_window: int = 30,
        overlap_ratio: Optional[float] = None,
        n_windows: int = 3,
        prefix: str = "overlap",
    ) -> pd.DataFrame:
        """
        Compute overlapping window features for capturing multi-scale patterns.

        Args:
            df: Input dataframe
            base_window: Base window size in days
            overlap_ratio: Ratio of overlap between windows. If None, uses config.
            n_windows: Number of overlapping windows
            prefix: Prefix for feature names

        Returns:
            DataFrame with overlapping window features

        Example:
            >>> overlap_df = engine.compute_overlapping_window_features(
            ...     df, base_window=30, n_windows=3
            ... )
        """
        self.validate_data(df)

        if overlap_ratio is None:
            overlap_ratio = self.config.overlapping_windows.get("overlap_ratio", 0.5)

        logger.info(
            f"Computing {n_windows} overlapping windows with "
            f"base={base_window}d, overlap={overlap_ratio}"
        )

        df_sorted = df.sort_values([self.entity_col, self.timestamp_col])
        features = []

        step = int(base_window * (1 - overlap_ratio))

        for i in range(n_windows):
            window_end = base_window + (i * step)
            window_start = i * step

            grouped = df_sorted.groupby(self.entity_col)

            # Compute mean for this overlapping window
            window_mean = (
                grouped.apply(
                    lambda x: x.set_index(self.timestamp_col)[self.value_col]
                    .rolling(f"{window_end}D")
                    .mean()
                )
                .reset_index(level=0, drop=True)
            )

            features.append(
                pd.DataFrame(
                    {f"{prefix}_mean_w{i}_{window_start}_{window_end}d": window_mean},
                    index=df_sorted.index,
                )
            )

        # Cross-window features
        if len(features) > 1:
            # Difference between consecutive windows
            for i in range(len(features) - 1):
                diff = features[i + 1].iloc[:, 0] - features[i].iloc[:, 0]
                features.append(
                    pd.DataFrame(
                        {f"{prefix}_diff_w{i}_w{i+1}": diff}, index=df_sorted.index
                    )
                )

        result_df = pd.concat([df_sorted.reset_index(drop=True)] + features, axis=1)
        logger.info(f"Generated {len(features)} overlapping window features")

        return result_df

    def compute_multi_resolution_features(
        self,
        df: pd.DataFrame,
        resolutions: Optional[List[str]] = None,
        aggregations: Optional[List[str]] = None,
        prefix: str = "resolution",
    ) -> pd.DataFrame:
        """
        Compute multi-resolution features (daily, weekly, monthly aggregations).

        Args:
            df: Input dataframe
            resolutions: List of resolutions. If None, uses config.
            aggregations: List of aggregation functions
            prefix: Prefix for feature names

        Returns:
            DataFrame with multi-resolution features

        Example:
            >>> multi_res_df = engine.compute_multi_resolution_features(
            ...     df, resolutions=['daily', 'weekly', 'monthly']
            ... )
        """
        self.validate_data(df)

        if resolutions is None:
            resolutions = self.config.multi_resolution.get(
                "resolutions", ["daily", "weekly", "monthly"]
            )

        aggregations = aggregations or ["mean", "std", "sum", "count"]

        logger.info(f"Computing multi-resolution features for: {resolutions}")

        df_sorted = df.sort_values([self.entity_col, self.timestamp_col])
        features = []

        for resolution in resolutions:
            # Determine resampling frequency
            if resolution == "daily":
                freq = "D"
            elif resolution == "weekly":
                freq = "W"
            elif resolution == "monthly":
                freq = "M"
            else:
                logger.warning(f"Unknown resolution: {resolution}, skipping")
                continue

            # Resample and aggregate
            resampled = (
                df_sorted.set_index(self.timestamp_col)
                .groupby(self.entity_col)[self.value_col]
                .resample(freq)
            )

            for agg in aggregations:
                if agg == "mean":
                    feature = resampled.mean()
                elif agg == "std":
                    feature = resampled.std()
                elif agg == "sum":
                    feature = resampled.sum()
                elif agg == "count":
                    feature = resampled.count()
                else:
                    continue

                # Merge back to original dataframe
                feature_df = feature.reset_index()
                feature_df.columns = [
                    self.entity_col,
                    self.timestamp_col,
                    f"{prefix}_{resolution}_{agg}",
                ]

                # Forward fill to match original timestamps
                df_with_feature = pd.merge_asof(
                    df_sorted.sort_values(self.timestamp_col),
                    feature_df.sort_values(self.timestamp_col),
                    on=self.timestamp_col,
                    by=self.entity_col,
                    direction="backward",
                )

                features.append(
                    pd.DataFrame(
                        {
                            f"{prefix}_{resolution}_{agg}": df_with_feature[
                                f"{prefix}_{resolution}_{agg}"
                            ]
                        },
                        index=df_sorted.index,
                    )
                )

        result_df = pd.concat([df_sorted.reset_index(drop=True)] + features, axis=1)
        logger.info(f"Generated {len(features)} multi-resolution features")

        return result_df

    def compute_event_triggered_features(
        self,
        df: pd.DataFrame,
        event_col: str,
        event_value: Union[str, int, float],
        prefix: str = "since_event",
    ) -> pd.DataFrame:
        """
        Compute features based on time since/until events.

        Args:
            df: Input dataframe
            event_col: Column containing event indicators
            event_value: Value that indicates an event occurred
            prefix: Prefix for feature names

        Returns:
            DataFrame with event-triggered features

        Example:
            >>> df['is_payment'] = [0, 1, 0, 0, 1, 0]
            >>> event_df = engine.compute_event_triggered_features(
            ...     df, event_col='is_payment', event_value=1
            ... )
        """
        self.validate_data(df)

        if event_col not in df.columns:
            raise ValueError(f"Event column '{event_col}' not found in dataframe")

        logger.info(f"Computing event-triggered features for {event_col}={event_value}")

        df_sorted = df.sort_values([self.entity_col, self.timestamp_col])
        grouped = df_sorted.groupby(self.entity_col)

        # Days since last event
        event_dates = df_sorted[df_sorted[event_col] == event_value][
            [self.entity_col, self.timestamp_col]
        ]
        event_dates = event_dates.rename(columns={self.timestamp_col: "event_date"})

        df_with_events = pd.merge_asof(
            df_sorted.sort_values(self.timestamp_col),
            event_dates.sort_values("event_date"),
            left_on=self.timestamp_col,
            right_on="event_date",
            by=self.entity_col,
            direction="backward",
        )

        days_since_event = (
            df_with_events[self.timestamp_col] - df_with_events["event_date"]
        ).dt.days

        # Count of events in last N days
        event_indicator = (df_sorted[event_col] == event_value).astype(int)

        count_last_30d = (
            grouped.apply(
                lambda x: x.assign(event_temp=event_indicator.loc[x.index])
                .set_index(self.timestamp_col)["event_temp"]
                .rolling("30D")
                .sum()
            )
            .reset_index(level=0, drop=True)
        )

        count_last_60d = (
            grouped.apply(
                lambda x: x.assign(event_temp=event_indicator.loc[x.index])
                .set_index(self.timestamp_col)["event_temp"]
                .rolling("60D")
                .sum()
            )
            .reset_index(level=0, drop=True)
        )

        # Average time between events
        time_between_events = grouped[self.timestamp_col].diff().dt.days
        time_between_events = time_between_events.where(event_indicator == 1)

        avg_time_between = (
            grouped.apply(
                lambda x: x.assign(time_temp=time_between_events.loc[x.index])
                .set_index(self.timestamp_col)["time_temp"]
                .rolling("90D")
                .mean()
            )
            .reset_index(level=0, drop=True)
        )

        features = pd.DataFrame(
            {
                f"{prefix}_days_since": days_since_event,
                f"{prefix}_count_30d": count_last_30d,
                f"{prefix}_count_60d": count_last_60d,
                f"{prefix}_avg_time_between": avg_time_between,
            },
            index=df_sorted.index,
        )

        result_df = pd.concat([df_sorted.reset_index(drop=True), features], axis=1)
        logger.info(f"Generated 4 event-triggered features")

        return result_df

    def compute_trend_seasonality_features(
        self,
        df: pd.DataFrame,
        decomposition_model: str = "additive",
        period: int = 7,
        prefix: str = "decomp",
    ) -> pd.DataFrame:
        """
        Extract trend and seasonality components from time series.

        Args:
            df: Input dataframe
            decomposition_model: 'additive' or 'multiplicative'
            period: Period for seasonal decomposition (e.g., 7 for weekly)
            prefix: Prefix for feature names

        Returns:
            DataFrame with trend and seasonality features

        Example:
            >>> trend_df = engine.compute_trend_seasonality_features(
            ...     df, period=7  # Weekly seasonality
            ... )
        """
        self.validate_data(df)

        logger.info(
            f"Computing trend/seasonality features with {decomposition_model} model"
        )

        df_sorted = df.sort_values([self.entity_col, self.timestamp_col])
        features = []

        for entity_id in df_sorted[self.entity_col].unique():
            entity_data = df_sorted[df_sorted[self.entity_col] == entity_id]

            if len(entity_data) < 2 * period:
                logger.warning(
                    f"Insufficient data for entity {entity_id} "
                    f"(need {2*period}, got {len(entity_data)})"
                )
                # Fill with NaN
                entity_features = pd.DataFrame(
                    {
                        f"{prefix}_trend": np.nan,
                        f"{prefix}_seasonal": np.nan,
                        f"{prefix}_residual": np.nan,
                    },
                    index=entity_data.index,
                )
            else:
                try:
                    # Perform decomposition
                    ts = entity_data.set_index(self.timestamp_col)[self.value_col]

                    # Resample to ensure regular frequency
                    ts = ts.resample("D").mean().interpolate(method="linear")

                    decomposition = seasonal_decompose(
                        ts, model=decomposition_model, period=period, extrapolate_trend="freq"
                    )

                    # Create feature dataframe
                    entity_features = pd.DataFrame(
                        {
                            f"{prefix}_trend": decomposition.trend,
                            f"{prefix}_seasonal": decomposition.seasonal,
                            f"{prefix}_residual": decomposition.resid,
                        }
                    )

                    # Merge back to original index
                    entity_features = entity_data.set_index(self.timestamp_col).join(
                        entity_features, how="left"
                    )
                    entity_features.index = entity_data.index

                except Exception as e:
                    logger.warning(
                        f"Failed to decompose for entity {entity_id}: {str(e)}"
                    )
                    entity_features = pd.DataFrame(
                        {
                            f"{prefix}_trend": np.nan,
                            f"{prefix}_seasonal": np.nan,
                            f"{prefix}_residual": np.nan,
                        },
                        index=entity_data.index,
                    )

            features.append(entity_features)

        features_df = pd.concat(features)

        result_df = pd.concat(
            [df_sorted.reset_index(drop=True), features_df.reset_index(drop=True)],
            axis=1,
        )
        logger.info("Generated 3 trend/seasonality features")

        return result_df

    def compute_all_features(
        self,
        df: pd.DataFrame,
        include_rolling: bool = True,
        include_lag: bool = True,
        include_multi_resolution: bool = True,
        include_trend: bool = False,  # Expensive, off by default
    ) -> pd.DataFrame:
        """
        Compute all temporal features at once.

        Args:
            df: Input dataframe
            include_rolling: Whether to include rolling window features
            include_lag: Whether to include lag features
            include_multi_resolution: Whether to include multi-resolution features
            include_trend: Whether to include trend/seasonality (computationally expensive)

        Returns:
            DataFrame with all requested temporal features

        Example:
            >>> df = pd.DataFrame({
            ...     'user_id': [1] * 100,
            ...     'timestamp': pd.date_range('2024-01-01', periods=100, freq='D'),
            ...     'amount': np.random.randn(100).cumsum() + 100
            ... })
            >>> engine = TemporalFeatureEngine()
            >>> all_features = engine.compute_all_features(df)
        """
        logger.info("Computing all temporal features")
        self.validate_data(df)

        result_df = df.copy()

        if include_rolling:
            rolling_df = self.compute_rolling_window_features(df)
            rolling_cols = [
                col for col in rolling_df.columns if col.startswith("rolling_")
            ]
            result_df = result_df.merge(
                rolling_df[[self.entity_col, self.timestamp_col] + rolling_cols],
                on=[self.entity_col, self.timestamp_col],
                how="left",
            )

        if include_lag:
            lag_df = self.compute_lag_features(df)
            lag_cols = [col for col in lag_df.columns if col.startswith("lag_")]
            result_df = result_df.merge(
                lag_df[[self.entity_col, self.timestamp_col] + lag_cols],
                on=[self.entity_col, self.timestamp_col],
                how="left",
            )

        if include_multi_resolution:
            multi_res_df = self.compute_multi_resolution_features(df)
            multi_res_cols = [
                col for col in multi_res_df.columns if col.startswith("resolution_")
            ]
            result_df = result_df.merge(
                multi_res_df[[self.entity_col, self.timestamp_col] + multi_res_cols],
                on=[self.entity_col, self.timestamp_col],
                how="left",
            )

        if include_trend:
            trend_df = self.compute_trend_seasonality_features(df)
            trend_cols = [col for col in trend_df.columns if col.startswith("decomp_")]
            result_df = result_df.merge(
                trend_df[[self.entity_col, self.timestamp_col] + trend_cols],
                on=[self.entity_col, self.timestamp_col],
                how="left",
            )

        logger.info(
            f"Generated {len(result_df.columns) - len(df.columns)} temporal features"
        )

        return result_df


if __name__ == "__main__":
    # Example usage
    logger.info("Running temporal features example")

    # Create sample data
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    df = pd.DataFrame(
        {
            "user_id": 1,
            "timestamp": dates,
            "amount": np.random.randn(100).cumsum() + 100,
        }
    )

    # Initialize engine
    engine = TemporalFeatureEngine()

    # Compute all features
    features_df = engine.compute_all_features(df)

    logger.info(f"Generated features shape: {features_df.shape}")
    logger.info(f"Sample features: {[c for c in features_df.columns if 'rolling' in c][:5]}")
