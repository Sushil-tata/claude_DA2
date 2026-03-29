"""
Rolling window aggregations for temporal features.
"""
import logging

logger = logging.getLogger(__name__)


def compute_rolling_windows(
    df,
    entity_key: str,
    timestamp_col: str,
    value_col: str,
    lookback_windows: list,
    aggregations: list = None
):
    """
    Compute rolling window aggregations.

    Args:
        df: Spark or pandas DataFrame
        entity_key: Entity identifier (e.g., "customer_id")
        timestamp_col: Timestamp column name
        value_col: Column to aggregate
        lookback_windows: List of window sizes in days (e.g., [7, 30, 90])
        aggregations: List of aggregation functions (default: ["sum", "avg", "count"])

    Returns:
        DataFrame with rolling window features

    Example:
        >>> df_with_features = compute_rolling_windows(
        ...     df,
        ...     entity_key="customer_id",
        ...     timestamp_col="transaction_timestamp",
        ...     value_col="transaction_amount",
        ...     lookback_windows=[7, 30, 90],
        ...     aggregations=["sum", "avg", "count"]
        ... )
    """
    if aggregations is None:
        aggregations = ["sum", "avg", "count"]

    # Check if PySpark DataFrame
    try:
        from pyspark.sql import DataFrame as SparkDataFrame
        is_spark = isinstance(df, SparkDataFrame)
    except ImportError:
        is_spark = False

    if is_spark:
        return _compute_rolling_windows_spark(
            df, entity_key, timestamp_col, value_col, lookback_windows, aggregations
        )
    else:
        return _compute_rolling_windows_pandas(
            df, entity_key, timestamp_col, value_col, lookback_windows, aggregations
        )


def _compute_rolling_windows_spark(
    df, entity_key, timestamp_col, value_col, lookback_windows, aggregations
):
    """Spark implementation of rolling windows"""
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    result_df = df

    for window_days in lookback_windows:
        # Convert days to seconds for Spark window
        window_seconds = window_days * 86400

        # Define window spec: range-based window looking back from current row
        window_spec = (
            Window.partitionBy(entity_key)
            .orderBy(F.col(timestamp_col).cast("long"))
            .rangeBetween(-window_seconds, 0)
        )

        for agg_func in aggregations:
            feature_name = f"{value_col}_{agg_func}_{window_days}d"

            if agg_func == "sum":
                result_df = result_df.withColumn(
                    feature_name,
                    F.sum(value_col).over(window_spec)
                )
            elif agg_func == "avg":
                result_df = result_df.withColumn(
                    feature_name,
                    F.avg(value_col).over(window_spec)
                )
            elif agg_func == "count":
                result_df = result_df.withColumn(
                    feature_name,
                    F.count(value_col).over(window_spec)
                )
            elif agg_func == "std":
                result_df = result_df.withColumn(
                    feature_name,
                    F.stddev(value_col).over(window_spec)
                )
            elif agg_func == "min":
                result_df = result_df.withColumn(
                    feature_name,
                    F.min(value_col).over(window_spec)
                )
            elif agg_func == "max":
                result_df = result_df.withColumn(
                    feature_name,
                    F.max(value_col).over(window_spec)
                )

    logger.info(f"Computed rolling windows: {lookback_windows} days, aggregations: {aggregations}")
    return result_df


def _compute_rolling_windows_pandas(
    df, entity_key, timestamp_col, value_col, lookback_windows, aggregations
):
    """Pandas implementation of rolling windows"""
    import pandas as pd

    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        df = df.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    # Sort by entity and timestamp
    df = df.sort_values([entity_key, timestamp_col])

    for window_days in lookback_windows:
        window_str = f"{window_days}D"

        for agg_func in aggregations:
            feature_name = f"{value_col}_{agg_func}_{window_days}d"

            if agg_func == "sum":
                df[feature_name] = df.groupby(entity_key)[value_col].transform(
                    lambda x: x.rolling(window_str, min_periods=1).sum()
                )
            elif agg_func == "avg":
                df[feature_name] = df.groupby(entity_key)[value_col].transform(
                    lambda x: x.rolling(window_str, min_periods=1).mean()
                )
            elif agg_func == "count":
                df[feature_name] = df.groupby(entity_key)[value_col].transform(
                    lambda x: x.rolling(window_str, min_periods=1).count()
                )
            elif agg_func == "std":
                df[feature_name] = df.groupby(entity_key)[value_col].transform(
                    lambda x: x.rolling(window_str, min_periods=1).std()
                )
            elif agg_func == "min":
                df[feature_name] = df.groupby(entity_key)[value_col].transform(
                    lambda x: x.rolling(window_str, min_periods=1).min()
                )
            elif agg_func == "max":
                df[feature_name] = df.groupby(entity_key)[value_col].transform(
                    lambda x: x.rolling(window_str, min_periods=1).max()
                )

    logger.info(f"Computed rolling windows (pandas): {lookback_windows} days")
    return df
