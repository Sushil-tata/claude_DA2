"""
Tag-based feature engineering for categorical transaction data.
"""
import logging

logger = logging.getLogger(__name__)


def compute_tag_features(
    df,
    entity_key: str,
    timestamp_col: str,
    tag_col: str,
    lookback_window: int = 30
):
    """
    Compute frequency-based features for transaction categories/tags.

    For each category, counts occurrences within the lookback window.

    Args:
        df: Spark or pandas DataFrame
        entity_key: Entity identifier (e.g., "customer_id")
        timestamp_col: Timestamp column
        tag_col: Tag/category column (e.g., "transaction_category")
        lookback_window: Lookback window in days

    Returns:
        DataFrame with tag frequency features

    Example:
        >>> df_with_tags = compute_tag_features(
        ...     df,
        ...     entity_key="customer_id",
        ...     timestamp_col="transaction_timestamp",
        ...     tag_col="transaction_category",
        ...     lookback_window=30
        ... )
    """
    # Check if PySpark DataFrame
    try:
        from pyspark.sql import DataFrame as SparkDataFrame
        is_spark = isinstance(df, SparkDataFrame)
    except ImportError:
        is_spark = False

    if is_spark:
        return _compute_tag_features_spark(df, entity_key, timestamp_col, tag_col, lookback_window)
    else:
        return _compute_tag_features_pandas(df, entity_key, timestamp_col, tag_col, lookback_window)


def _compute_tag_features_spark(df, entity_key, timestamp_col, tag_col, lookback_window):
    """Spark implementation of tag features"""
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    # Get unique tags
    tags = [row[tag_col] for row in df.select(tag_col).distinct().collect()]
    logger.info(f"Found {len(tags)} unique tags: {tags}")

    result_df = df

    # Create features for each tag
    window_seconds = lookback_window * 86400
    window_spec = (
        Window.partitionBy(entity_key)
        .orderBy(F.col(timestamp_col).cast("long"))
        .rangeBetween(-window_seconds, 0)
    )

    for tag in tags:
        feature_name = f"tag_{tag}_count_{lookback_window}d"

        # Create binary indicator for this tag
        result_df = result_df.withColumn(
            f"_is_{tag}",
            F.when(F.col(tag_col) == tag, 1).otherwise(0)
        )

        # Count occurrences in window
        result_df = result_df.withColumn(
            feature_name,
            F.sum(f"_is_{tag}").over(window_spec)
        )

        # Drop temporary column
        result_df = result_df.drop(f"_is_{tag}")

    logger.info(f"Computed tag features for {len(tags)} tags with {lookback_window}d window")
    return result_df


def _compute_tag_features_pandas(df, entity_key, timestamp_col, tag_col, lookback_window):
    """Pandas implementation of tag features"""
    import pandas as pd

    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        df = df.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    # Get unique tags
    tags = df[tag_col].unique()
    logger.info(f"Found {len(tags)} unique tags: {tags}")

    # Sort by entity and timestamp
    df = df.sort_values([entity_key, timestamp_col])

    for tag in tags:
        feature_name = f"tag_{tag}_count_{lookback_window}d"

        # Create binary indicator
        df[f"_is_{tag}"] = (df[tag_col] == tag).astype(int)

        # Rolling count
        df[feature_name] = df.groupby(entity_key)[f"_is_{tag}"].transform(
            lambda x: x.rolling(f"{lookback_window}D", min_periods=1).sum()
        )

        # Drop temporary column
        df = df.drop(columns=[f"_is_{tag}"])

    logger.info(f"Computed tag features (pandas): {len(tags)} tags")
    return df


def compute_category_aggregates(
    df,
    entity_key: str,
    timestamp_col: str,
    tag_col: str,
    value_col: str,
    lookback_window: int = 30
):
    """
    Compute aggregated spending by category.

    Args:
        df: DataFrame
        entity_key: Entity identifier
        timestamp_col: Timestamp column
        tag_col: Category column
        value_col: Value to aggregate (e.g., transaction_amount)
        lookback_window: Lookback window in days

    Returns:
        DataFrame with category aggregates

    Example:
        >>> df = compute_category_aggregates(
        ...     df,
        ...     entity_key="customer_id",
        ...     tag_col="transaction_category",
        ...     value_col="transaction_amount",
        ...     lookback_window=30
        ... )
    """
    # Check if PySpark
    try:
        from pyspark.sql import DataFrame as SparkDataFrame
        from pyspark.sql import Window
        from pyspark.sql import functions as F

        is_spark = isinstance(df, SparkDataFrame)
    except ImportError:
        is_spark = False

    if is_spark:
        # Get unique categories
        categories = [row[tag_col] for row in df.select(tag_col).distinct().collect()]

        result_df = df
        window_seconds = lookback_window * 86400
        window_spec = (
            Window.partitionBy(entity_key)
            .orderBy(F.col(timestamp_col).cast("long"))
            .rangeBetween(-window_seconds, 0)
        )

        for category in categories:
            feature_name = f"category_{category}_sum_{lookback_window}d"

            # Sum for this category
            result_df = result_df.withColumn(
                f"_cat_{category}_val",
                F.when(F.col(tag_col) == category, F.col(value_col)).otherwise(0)
            )

            result_df = result_df.withColumn(
                feature_name,
                F.sum(f"_cat_{category}_val").over(window_spec)
            )

            result_df = result_df.drop(f"_cat_{category}_val")

        return result_df
    else:
        # Pandas implementation
        import pandas as pd

        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            df = df.copy()
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])

        categories = df[tag_col].unique()
        df = df.sort_values([entity_key, timestamp_col])

        for category in categories:
            feature_name = f"category_{category}_sum_{lookback_window}d"

            # Create category-specific value
            df[f"_cat_{category}_val"] = df.apply(
                lambda row: row[value_col] if row[tag_col] == category else 0,
                axis=1
            )

            # Rolling sum
            df[feature_name] = df.groupby(entity_key)[f"_cat_{category}_val"].transform(
                lambda x: x.rolling(f"{lookback_window}D", min_periods=1).sum()
            )

            df = df.drop(columns=[f"_cat_{category}_val"])

        return df
