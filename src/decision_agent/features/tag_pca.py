"""
PCA dimensionality reduction on tag/category embeddings using Spark ML.
"""
import logging

logger = logging.getLogger(__name__)


def compute_tag_pca(
    df,
    tag_feature_cols: list,
    n_components: int = 3,
    output_prefix: str = "tag_pca_component"
):
    """
    Apply PCA to tag frequency features for dimensionality reduction.

    Uses Spark ML PCA for distributed computation (Spark-native).
    Falls back to pandas+sklearn for pandas DataFrames.

    Args:
        df: DataFrame with tag features (Spark or pandas)
        tag_feature_cols: List of tag feature column names
        n_components: Number of PCA components to keep
        output_prefix: Prefix for output column names

    Returns:
        DataFrame with PCA components added

    Example:
        >>> tag_cols = ["tag_salary_count_30d", "tag_rent_count_30d", "tag_groceries_count_30d"]
        >>> df_with_pca = compute_tag_pca(df, tag_cols, n_components=3)
    """
    # Check if PySpark DataFrame
    try:
        from pyspark.sql import DataFrame as SparkDataFrame
        is_spark = isinstance(df, SparkDataFrame)
    except ImportError:
        is_spark = False

    if is_spark:
        return _compute_tag_pca_spark(df, tag_feature_cols, n_components, output_prefix)
    else:
        return _compute_tag_pca_pandas(df, tag_feature_cols, n_components, output_prefix)


def _compute_tag_pca_spark(df, tag_feature_cols, n_components, output_prefix):
    """
    Spark-native PCA implementation using Spark ML.

    Uses VectorAssembler + PCA from pyspark.ml for distributed computation.
    No toPandas() conversion.
    """
    from pyspark.ml.feature import VectorAssembler, PCA, StandardScaler
    from pyspark.sql import functions as F

    logger.info(f"Computing PCA with Spark ML on {len(tag_feature_cols)} tag features")

    # Step 1: Assemble tag features into a single vector column
    assembler = VectorAssembler(
        inputCols=tag_feature_cols,
        outputCol="_tag_features_vector",
        handleInvalid="keep"  # Keep rows with NaN values
    )

    # Fill nulls with 0 before assembling
    df_filled = df
    for col in tag_feature_cols:
        df_filled = df_filled.fillna({col: 0.0})

    df_with_vector = assembler.transform(df_filled)

    # Step 2: Optional - Standardize features (improves PCA quality)
    scaler = StandardScaler(
        inputCol="_tag_features_vector",
        outputCol="_tag_features_scaled",
        withStd=True,
        withMean=True
    )
    scaler_model = scaler.fit(df_with_vector)
    df_scaled = scaler_model.transform(df_with_vector)

    # Step 3: Apply PCA
    pca = PCA(
        k=n_components,
        inputCol="_tag_features_scaled",
        outputCol="_pca_features"
    )

    pca_model = pca.fit(df_scaled)
    df_with_pca = pca_model.transform(df_scaled)

    # Log explained variance
    explained_variance = pca_model.explainedVariance.toArray()
    logger.info(f"PCA explained variance ratio: {explained_variance}")
    logger.info(f"Total variance explained: {explained_variance.sum():.2%}")

    # Step 4: Extract PCA components as separate columns
    # The PCA output is a DenseVector, we need to extract individual components
    result_df = df_with_pca

    for i in range(n_components):
        component_name = f"{output_prefix}_{i + 1}"

        # Extract the i-th element from the PCA vector
        # Using UDF to extract vector elements
        def get_element(v, idx):
            """Extract element from DenseVector"""
            try:
                return float(v[idx])
            except (IndexError, TypeError):
                return 0.0

        from pyspark.sql.types import DoubleType
        from pyspark.sql.functions import udf

        get_element_udf = udf(lambda v: get_element(v, i), DoubleType())

        result_df = result_df.withColumn(
            component_name,
            get_element_udf(F.col("_pca_features"))
        )

    # Clean up intermediate columns
    result_df = result_df.drop(
        "_tag_features_vector",
        "_tag_features_scaled",
        "_pca_features"
    )

    logger.info(f"✓ Computed {n_components} PCA components using Spark ML (distributed)")
    return result_df


def _compute_tag_pca_pandas(df, tag_feature_cols, n_components, output_prefix):
    """Pandas implementation of PCA using sklearn"""
    from sklearn.decomposition import PCA
    import pandas as pd

    # Extract tag features
    tag_data = df[tag_feature_cols].fillna(0)

    # Fit PCA
    pca = PCA(n_components=n_components)
    pca_components = pca.fit_transform(tag_data)

    logger.info(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    logger.info(f"Total variance explained: {pca.explained_variance_ratio_.sum():.2%}")

    # Add PCA components to DataFrame
    result_df = df.copy()

    for i in range(n_components):
        component_name = f"{output_prefix}_{i + 1}"
        result_df[component_name] = pca_components[:, i]

    logger.info(f"Computed {n_components} PCA components (pandas)")
    return result_df


def get_tag_feature_columns(df, tag_prefix: str = "tag_") -> list:
    """
    Helper to extract tag feature columns from DataFrame.

    Args:
        df: DataFrame
        tag_prefix: Prefix for tag columns

    Returns:
        List of tag feature column names
    """
    # Check if Spark or pandas
    try:
        from pyspark.sql import DataFrame as SparkDataFrame
        is_spark = isinstance(df, SparkDataFrame)
    except ImportError:
        is_spark = False

    if is_spark:
        cols = df.columns
    else:
        cols = df.columns.tolist()

    tag_cols = [col for col in cols if col.startswith(tag_prefix)]

    logger.info(f"Found {len(tag_cols)} tag feature columns")
    return tag_cols
