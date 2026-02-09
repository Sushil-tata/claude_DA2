"""
Income estimation feature pipeline.

Orchestrates all feature engineering for income estimation use case.
"""
import logging
from decision_agent.features.windows import compute_rolling_windows
from decision_agent.features.tags import compute_tag_features, compute_category_aggregates
from decision_agent.features.tag_pca import compute_tag_pca, get_tag_feature_columns
from decision_agent.features.liquidity import compute_liquidity_features

logger = logging.getLogger(__name__)


def compute_income_features(df, feature_config: dict):
    """
    Compute all features for income estimation.

    Pipeline:
    1. Rolling window aggregations (7d, 30d, 90d)
    2. Tag frequency features
    3. Category aggregate features
    4. Tag PCA for dimensionality reduction
    5. Liquidity ratio features

    Args:
        df: Spark or pandas DataFrame with transaction data
        feature_config: Feature configuration from YAML

    Returns:
        DataFrame with all computed features

    Expected input columns:
        - customer_id
        - transaction_timestamp
        - transaction_amount
        - transaction_category
        - account_balance
        - income_level (label)
    """
    logger.info("Starting income feature pipeline...")

    lookback_windows = feature_config.get("lookback_windows", [7, 30, 90])
    entity_key = "customer_id"
    timestamp_col = "transaction_timestamp"
    amount_col = "transaction_amount"
    category_col = "transaction_category"
    balance_col = "account_balance"

    # Step 1: Rolling window features
    logger.info("Step 1: Computing rolling window aggregations...")
    df_with_windows = compute_rolling_windows(
        df,
        entity_key=entity_key,
        timestamp_col=timestamp_col,
        value_col=amount_col,
        lookback_windows=lookback_windows,
        aggregations=["sum", "avg", "count"]
    )

    # Step 2: Tag frequency features
    logger.info("Step 2: Computing tag frequency features...")
    df_with_tags = compute_tag_features(
        df_with_windows,
        entity_key=entity_key,
        timestamp_col=timestamp_col,
        tag_col=category_col,
        lookback_window=30  # Use 30-day window for tag features
    )

    # Step 3: Category aggregate features
    logger.info("Step 3: Computing category aggregate features...")
    df_with_categories = compute_category_aggregates(
        df_with_tags,
        entity_key=entity_key,
        timestamp_col=timestamp_col,
        tag_col=category_col,
        value_col=amount_col,
        lookback_window=30
    )

    # Step 4: Tag PCA (dimensionality reduction)
    logger.info("Step 4: Computing tag PCA features...")
    tag_cols = get_tag_feature_columns(df_with_categories, tag_prefix="tag_")

    if len(tag_cols) > 3:
        df_with_pca = compute_tag_pca(
            df_with_categories,
            tag_feature_cols=tag_cols,
            n_components=3,
            output_prefix="tag_pca_component"
        )
    else:
        logger.info("Skipping PCA: not enough tag features")
        df_with_pca = df_with_categories

    # Step 5: Liquidity features
    logger.info("Step 5: Computing liquidity features...")
    df_final = compute_liquidity_features(
        df_with_pca,
        entity_key=entity_key,
        timestamp_col=timestamp_col,
        amount_col=amount_col,
        balance_col=balance_col,
        lookback_window=30
    )

    # Aggregate to customer level (one row per customer)
    logger.info("Step 6: Aggregating to customer level...")
    df_aggregated = aggregate_to_customer_level(df_final, entity_key)

    logger.info("Income feature pipeline completed!")
    return df_aggregated


def aggregate_to_customer_level(df, entity_key: str):
    """
    Aggregate transaction-level features to customer level.

    Takes the most recent row for each customer (with all accumulated features).

    Args:
        df: DataFrame with transaction-level features
        entity_key: Entity identifier column

    Returns:
        DataFrame with one row per customer
    """
    # Check if PySpark DataFrame
    try:
        from pyspark.sql import DataFrame as SparkDataFrame
        from pyspark.sql import Window
        from pyspark.sql import functions as F

        is_spark = isinstance(df, SparkDataFrame)
    except ImportError:
        is_spark = False

    if is_spark:
        # Get most recent row per customer
        window_spec = Window.partitionBy(entity_key).orderBy(
            F.col("transaction_timestamp").desc()
        )

        df_with_rank = df.withColumn("_rank", F.row_number().over(window_spec))
        df_latest = df_with_rank.filter(F.col("_rank") == 1).drop("_rank")

        logger.info(f"Aggregated to customer level (Spark): {df_latest.count()} customers")
        return df_latest
    else:
        # Pandas: get most recent row per customer
        df_sorted = df.sort_values([entity_key, "transaction_timestamp"], ascending=[True, False])
        df_latest = df_sorted.groupby(entity_key).first().reset_index()

        logger.info(f"Aggregated to customer level (pandas): {len(df_latest)} customers")
        return df_latest
