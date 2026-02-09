"""
Liquidity ratio features for income estimation.
"""
import logging

logger = logging.getLogger(__name__)


def compute_liquidity_features(
    df,
    entity_key: str,
    timestamp_col: str,
    amount_col: str,
    balance_col: str,
    lookback_window: int = 30
):
    """
    Compute liquidity-related features:
    - Income/expense ratio
    - Average balance
    - Balance volatility
    - Net cash flow

    Args:
        df: Spark or pandas DataFrame
        entity_key: Entity identifier (e.g., "customer_id")
        timestamp_col: Timestamp column
        amount_col: Transaction amount column (positive for income, negative for expense)
        balance_col: Account balance column
        lookback_window: Lookback window in days

    Returns:
        DataFrame with liquidity features

    Example:
        >>> df_with_liquidity = compute_liquidity_features(
        ...     df,
        ...     entity_key="customer_id",
        ...     timestamp_col="transaction_timestamp",
        ...     amount_col="transaction_amount",
        ...     balance_col="account_balance",
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
        return _compute_liquidity_features_spark(
            df, entity_key, timestamp_col, amount_col, balance_col, lookback_window
        )
    else:
        return _compute_liquidity_features_pandas(
            df, entity_key, timestamp_col, amount_col, balance_col, lookback_window
        )


def _compute_liquidity_features_spark(
    df, entity_key, timestamp_col, amount_col, balance_col, lookback_window
):
    """Spark implementation of liquidity features"""
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    window_seconds = lookback_window * 86400
    window_spec = (
        Window.partitionBy(entity_key)
        .orderBy(F.col(timestamp_col).cast("long"))
        .rangeBetween(-window_seconds, 0)
    )

    result_df = df

    # Separate income and expenses
    result_df = result_df.withColumn(
        "_income",
        F.when(F.col(amount_col) > 0, F.col(amount_col)).otherwise(0)
    )
    result_df = result_df.withColumn(
        "_expense",
        F.when(F.col(amount_col) < 0, F.abs(F.col(amount_col))).otherwise(0)
    )

    # Sum income and expenses in window
    result_df = result_df.withColumn(
        f"total_income_{lookback_window}d",
        F.sum("_income").over(window_spec)
    )
    result_df = result_df.withColumn(
        f"total_expense_{lookback_window}d",
        F.sum("_expense").over(window_spec)
    )

    # Income/expense ratio
    result_df = result_df.withColumn(
        f"income_expense_ratio_{lookback_window}d",
        F.when(
            F.col(f"total_expense_{lookback_window}d") > 0,
            F.col(f"total_income_{lookback_window}d") / F.col(f"total_expense_{lookback_window}d")
        ).otherwise(F.lit(None))
    )

    # Average balance
    result_df = result_df.withColumn(
        f"avg_balance_{lookback_window}d",
        F.avg(balance_col).over(window_spec)
    )

    # Balance volatility (standard deviation)
    result_df = result_df.withColumn(
        f"balance_volatility_{lookback_window}d",
        F.stddev(balance_col).over(window_spec)
    )

    # Net cash flow
    result_df = result_df.withColumn(
        f"net_cash_flow_{lookback_window}d",
        F.col(f"total_income_{lookback_window}d") - F.col(f"total_expense_{lookback_window}d")
    )

    # Clean up temporary columns
    result_df = result_df.drop("_income", "_expense")

    logger.info(f"Computed liquidity features with {lookback_window}d window (Spark)")
    return result_df


def _compute_liquidity_features_pandas(
    df, entity_key, timestamp_col, amount_col, balance_col, lookback_window
):
    """Pandas implementation of liquidity features"""
    import pandas as pd

    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        df = df.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    # Sort by entity and timestamp
    df = df.sort_values([entity_key, timestamp_col])

    # Separate income and expenses
    df["_income"] = df[amount_col].apply(lambda x: x if x > 0 else 0)
    df["_expense"] = df[amount_col].apply(lambda x: abs(x) if x < 0 else 0)

    window_str = f"{lookback_window}D"

    # Sum income and expenses in rolling window
    df[f"total_income_{lookback_window}d"] = df.groupby(entity_key)["_income"].transform(
        lambda x: x.rolling(window_str, min_periods=1).sum()
    )
    df[f"total_expense_{lookback_window}d"] = df.groupby(entity_key)["_expense"].transform(
        lambda x: x.rolling(window_str, min_periods=1).sum()
    )

    # Income/expense ratio
    df[f"income_expense_ratio_{lookback_window}d"] = (
        df[f"total_income_{lookback_window}d"] / df[f"total_expense_{lookback_window}d"]
    ).replace([float('inf'), -float('inf')], None)

    # Average balance
    df[f"avg_balance_{lookback_window}d"] = df.groupby(entity_key)[balance_col].transform(
        lambda x: x.rolling(window_str, min_periods=1).mean()
    )

    # Balance volatility
    df[f"balance_volatility_{lookback_window}d"] = df.groupby(entity_key)[balance_col].transform(
        lambda x: x.rolling(window_str, min_periods=1).std()
    )

    # Net cash flow
    df[f"net_cash_flow_{lookback_window}d"] = (
        df[f"total_income_{lookback_window}d"] - df[f"total_expense_{lookback_window}d"]
    )

    # Clean up temporary columns
    df = df.drop(columns=["_income", "_expense"])

    logger.info(f"Computed liquidity features with {lookback_window}d window (pandas)")
    return df
