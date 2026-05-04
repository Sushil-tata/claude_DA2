"""
Persona Features Engineering Module

Implements persona and segmentation features using ML clustering and NLP:
- Transaction description clustering using NLP (TF-IDF)
- Merchant segmentation (hierarchical, DBSCAN, GMM)
- Behavioral persona tagging (K-means, HDBSCAN)
- Category concentration indices
- Merchant diversity metrics
- Transaction pattern signatures

Author: Principal Data Science Decision Agent
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from pathlib import Path
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.cluster import (
    AgglomerativeClustering,
    DBSCAN,
    KMeans,
)
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    logger.warning("HDBSCAN not available. Install with: pip install hdbscan")


class PersonaFeatureConfig:
    """Configuration for persona feature engineering."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize persona feature configuration.

        Args:
            config_path: Path to feature_config.yaml. If None, uses defaults.

        Example:
            >>> config = PersonaFeatureConfig("config/feature_config.yaml")
            >>> config.clustering_method
            'kmeans'
        """
        if config_path and Path(config_path).exists():
            with open(config_path, "r") as f:
                full_config = yaml.safe_load(f)
                persona_config = full_config.get("persona", {})
        else:
            persona_config = {}

        clustering_config = persona_config.get("clustering", {})
        self.clustering_method = clustering_config.get("method", "kmeans")
        self.n_clusters = clustering_config.get("n_clusters", [5, 10, 15, 20])
        self.clustering_features = clustering_config.get(
            "features",
            [
                "transaction_frequency",
                "average_transaction_amount",
                "merchant_diversity",
                "category_concentration",
            ],
        )

        self.merchant_segmentation = persona_config.get(
            "merchant_segmentation", {"enabled": True, "methods": ["hierarchical", "dbscan", "gmm"]}
        )

        nlp_config = persona_config.get("nlp_features", {})
        self.nlp_enabled = nlp_config.get("enabled", True)
        self.vectorization = nlp_config.get("vectorization", "tfidf")
        self.max_features = nlp_config.get("max_features", 1000)


class PersonaFeatureEngine:
    """
    Engine for computing persona and segmentation features.

    Uses clustering algorithms and NLP to identify behavioral personas,
    merchant segments, and transaction patterns.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        entity_col: str = "user_id",
        timestamp_col: str = "timestamp",
    ):
        """
        Initialize persona feature engine.

        Args:
            config_path: Path to feature_config.yaml
            entity_col: Column name for entity identifier
            timestamp_col: Column name for timestamp

        Example:
            >>> engine = PersonaFeatureEngine(
            ...     config_path="config/feature_config.yaml",
            ...     entity_col="customer_id"
            ... )
        """
        self.config = PersonaFeatureConfig(config_path)
        self.entity_col = entity_col
        self.timestamp_col = timestamp_col
        logger.info("Initialized PersonaFeatureEngine")

    def validate_data(self, df: pd.DataFrame, required_cols: List[str]) -> None:
        """
        Validate input dataframe has required columns.

        Args:
            df: Input dataframe
            required_cols: List of required column names

        Raises:
            ValueError: If required columns are missing
        """
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

    def compute_transaction_description_features(
        self,
        df: pd.DataFrame,
        description_col: str = "description",
        n_components: int = 10,
        prefix: str = "desc",
    ) -> pd.DataFrame:
        """
        Extract features from transaction descriptions using TF-IDF and clustering.

        Args:
            df: Input dataframe
            description_col: Column containing transaction descriptions
            n_components: Number of PCA components to extract
            prefix: Prefix for feature names

        Returns:
            DataFrame with description-based features

        Example:
            >>> df = pd.DataFrame({
            ...     'user_id': [1, 1, 2, 2],
            ...     'timestamp': pd.date_range('2024-01-01', periods=4, freq='D'),
            ...     'description': ['Coffee at Starbucks', 'Groceries at Walmart',
            ...                     'Gas at Shell', 'Dinner at Restaurant']
            ... })
            >>> engine = PersonaFeatureEngine()
            >>> desc_df = engine.compute_transaction_description_features(df)
        """
        required_cols = [self.entity_col, description_col]
        self.validate_data(df, required_cols)

        if not self.config.nlp_enabled:
            logger.info("NLP features disabled in config")
            return df

        logger.info("Computing transaction description features using TF-IDF")

        df_sorted = df.copy()

        # Handle missing descriptions
        df_sorted[description_col] = df_sorted[description_col].fillna("unknown")

        # TF-IDF vectorization
        vectorizer = TfidfVectorizer(
            max_features=self.config.max_features,
            stop_words="english",
            lowercase=True,
            ngram_range=(1, 2),
        )

        try:
            tfidf_matrix = vectorizer.fit_transform(df_sorted[description_col])

            # Reduce dimensionality with PCA
            n_components = min(n_components, tfidf_matrix.shape[1], tfidf_matrix.shape[0])

            if n_components > 0:
                pca = PCA(n_components=n_components, random_state=42)
                tfidf_reduced = pca.fit_transform(tfidf_matrix.toarray())

                # Add PCA components as features
                for i in range(n_components):
                    df_sorted[f"{prefix}_tfidf_pca_{i}"] = tfidf_reduced[:, i]

                logger.info(
                    f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}"
                )

                # Cluster descriptions
                if len(df_sorted) >= 5:
                    n_clusters = min(5, len(df_sorted) // 2)
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    df_sorted[f"{prefix}_cluster"] = kmeans.fit_predict(tfidf_reduced)

                    # Distance to cluster center
                    distances = kmeans.transform(tfidf_reduced)
                    df_sorted[f"{prefix}_cluster_distance"] = distances.min(axis=1)

        except Exception as e:
            logger.warning(f"Failed to compute TF-IDF features: {str(e)}")

        logger.info(
            f"Generated {sum(col.startswith(prefix) for col in df_sorted.columns)} description features"
        )

        return df_sorted

    def compute_merchant_segmentation_features(
        self,
        df: pd.DataFrame,
        merchant_col: str = "merchant",
        amount_col: str = "amount",
        methods: Optional[List[str]] = None,
        prefix: str = "merchant_seg",
    ) -> pd.DataFrame:
        """
        Segment merchants using clustering algorithms.

        Args:
            df: Input dataframe
            merchant_col: Column containing merchant names
            amount_col: Column containing transaction amounts
            methods: Clustering methods to use. If None, uses config.
            prefix: Prefix for feature names

        Returns:
            DataFrame with merchant segmentation features

        Example:
            >>> df = pd.DataFrame({
            ...     'user_id': [1, 1, 2, 2, 3, 3],
            ...     'timestamp': pd.date_range('2024-01-01', periods=6, freq='D'),
            ...     'merchant': ['Starbucks', 'Walmart', 'Shell', 'Starbucks', 'Walmart', 'Target'],
            ...     'amount': [5.5, 120, 45, 6.0, 135, 89]
            ... })
            >>> engine = PersonaFeatureEngine()
            >>> merchant_df = engine.compute_merchant_segmentation_features(df)
        """
        required_cols = [self.entity_col, merchant_col, amount_col]
        self.validate_data(df, required_cols)

        if methods is None:
            methods = self.config.merchant_segmentation.get(
                "methods", ["hierarchical", "dbscan", "gmm"]
            )

        logger.info(f"Computing merchant segmentation using: {methods}")

        df_sorted = df.copy()

        # Aggregate merchant statistics
        merchant_stats = (
            df_sorted.groupby(merchant_col)
            .agg(
                {
                    amount_col: ["count", "mean", "std", "min", "max"],
                    self.entity_col: "nunique",
                }
            )
            .reset_index()
        )

        merchant_stats.columns = [
            "_".join(col).strip("_") if col[1] else col[0]
            for col in merchant_stats.columns.values
        ]

        # Prepare features for clustering
        feature_cols = [col for col in merchant_stats.columns if col != merchant_col]
        X = merchant_stats[feature_cols].fillna(0)

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Apply clustering methods
        if "hierarchical" in methods and len(merchant_stats) >= 2:
            try:
                n_clusters = min(5, max(2, len(merchant_stats) // 3))
                hierarchical = AgglomerativeClustering(
                    n_clusters=n_clusters, linkage="ward"
                )
                merchant_stats[f"{prefix}_hierarchical"] = hierarchical.fit_predict(
                    X_scaled
                )
            except Exception as e:
                logger.warning(f"Hierarchical clustering failed: {str(e)}")

        if "dbscan" in methods and len(merchant_stats) >= 2:
            try:
                dbscan = DBSCAN(eps=0.5, min_samples=2)
                merchant_stats[f"{prefix}_dbscan"] = dbscan.fit_predict(X_scaled)
            except Exception as e:
                logger.warning(f"DBSCAN clustering failed: {str(e)}")

        if "gmm" in methods and len(merchant_stats) >= 2:
            try:
                n_components = min(5, max(2, len(merchant_stats) // 3))
                gmm = GaussianMixture(n_components=n_components, random_state=42)
                merchant_stats[f"{prefix}_gmm"] = gmm.fit_predict(X_scaled)
            except Exception as e:
                logger.warning(f"GMM clustering failed: {str(e)}")

        # Merge back to original dataframe
        cluster_cols = [
            col for col in merchant_stats.columns if col.startswith(prefix)
        ]
        df_sorted = df_sorted.merge(
            merchant_stats[[merchant_col] + cluster_cols],
            on=merchant_col,
            how="left",
        )

        logger.info(f"Generated {len(cluster_cols)} merchant segmentation features")

        return df_sorted

    def compute_behavioral_persona_features(
        self,
        df: pd.DataFrame,
        amount_col: str = "amount",
        category_col: Optional[str] = None,
        merchant_col: Optional[str] = None,
        n_clusters_list: Optional[List[int]] = None,
        prefix: str = "persona",
    ) -> pd.DataFrame:
        """
        Tag behavioral personas using clustering on aggregated features.

        Args:
            df: Input dataframe
            amount_col: Column containing transaction amounts
            category_col: Optional category column
            merchant_col: Optional merchant column
            n_clusters_list: List of cluster counts to try. If None, uses config.
            prefix: Prefix for feature names

        Returns:
            DataFrame with persona features

        Example:
            >>> df = pd.DataFrame({
            ...     'user_id': [1, 1, 1, 2, 2, 2],
            ...     'timestamp': pd.date_range('2024-01-01', periods=6, freq='D'),
            ...     'amount': [50, 75, 100, 20, 25, 30],
            ...     'category': ['Food', 'Retail', 'Food', 'Food', 'Food', 'Retail']
            ... })
            >>> engine = PersonaFeatureEngine()
            >>> persona_df = engine.compute_behavioral_persona_features(df)
        """
        required_cols = [self.entity_col, self.timestamp_col, amount_col]
        self.validate_data(df, required_cols)

        if n_clusters_list is None:
            n_clusters_list = self.config.n_clusters

        logger.info(f"Computing behavioral personas for n_clusters: {n_clusters_list}")

        # Aggregate features per entity
        agg_dict = {
            amount_col: ["count", "mean", "std", "min", "max", "sum"],
        }

        if category_col and category_col in df.columns:
            agg_dict[category_col] = "nunique"

        if merchant_col and merchant_col in df.columns:
            agg_dict[merchant_col] = "nunique"

        entity_features = df.groupby(self.entity_col).agg(agg_dict)

        # Flatten column names
        entity_features.columns = [
            "_".join(col).strip("_") for col in entity_features.columns.values
        ]
        entity_features = entity_features.reset_index()

        # Add temporal features
        if pd.api.types.is_datetime64_any_dtype(df[self.timestamp_col]):
            time_features = (
                df.groupby(self.entity_col)[self.timestamp_col]
                .agg(["min", "max"])
                .reset_index()
            )
            time_features["days_active"] = (
                time_features["max"] - time_features["min"]
            ).dt.days + 1
            entity_features = entity_features.merge(
                time_features[[self.entity_col, "days_active"]],
                on=self.entity_col,
                how="left",
            )

            # Transaction frequency (txns per day)
            entity_features["txn_frequency"] = (
                entity_features[f"{amount_col}_count"]
                / entity_features["days_active"].replace(0, 1)
            )

        # Prepare features for clustering
        feature_cols = [
            col for col in entity_features.columns if col != self.entity_col
        ]
        X = entity_features[feature_cols].fillna(0)

        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Apply K-means clustering with different K values
        for n_clusters in n_clusters_list:
            if len(entity_features) < n_clusters:
                logger.warning(
                    f"Not enough entities ({len(entity_features)}) for {n_clusters} clusters"
                )
                continue

            try:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                entity_features[f"{prefix}_kmeans_{n_clusters}"] = kmeans.fit_predict(
                    X_scaled
                )

                # Silhouette score as feature quality
                distances = kmeans.transform(X_scaled)
                entity_features[f"{prefix}_cluster_distance_{n_clusters}"] = (
                    distances.min(axis=1)
                )

            except Exception as e:
                logger.warning(
                    f"K-means clustering with {n_clusters} clusters failed: {str(e)}"
                )

        # HDBSCAN (hierarchical density-based clustering)
        if HDBSCAN_AVAILABLE and len(entity_features) >= 5:
            try:
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=max(2, len(entity_features) // 10),
                    min_samples=1,
                )
                entity_features[f"{prefix}_hdbscan"] = clusterer.fit_predict(X_scaled)
                entity_features[f"{prefix}_hdbscan_probability"] = (
                    clusterer.probabilities_
                )
            except Exception as e:
                logger.warning(f"HDBSCAN clustering failed: {str(e)}")

        # Merge back to original dataframe
        persona_cols = [col for col in entity_features.columns if col.startswith(prefix)]
        df_with_personas = df.merge(
            entity_features[[self.entity_col] + persona_cols],
            on=self.entity_col,
            how="left",
        )

        logger.info(f"Generated {len(persona_cols)} behavioral persona features")

        return df_with_personas

    def compute_category_concentration_features(
        self,
        df: pd.DataFrame,
        category_col: str = "category",
        amount_col: str = "amount",
        windows: Optional[List[int]] = None,
        prefix: str = "category_conc",
    ) -> pd.DataFrame:
        """
        Compute category concentration indices (diversity metrics).

        Args:
            df: Input dataframe
            category_col: Column containing categories
            amount_col: Column containing amounts
            windows: Time windows in days
            prefix: Prefix for feature names

        Returns:
            DataFrame with category concentration features

        Example:
            >>> df = pd.DataFrame({
            ...     'user_id': [1, 1, 1, 1],
            ...     'timestamp': pd.date_range('2024-01-01', periods=4, freq='D'),
            ...     'category': ['Food', 'Retail', 'Food', 'Gas'],
            ...     'amount': [50, 100, 75, 45]
            ... })
            >>> engine = PersonaFeatureEngine()
            >>> conc_df = engine.compute_category_concentration_features(df)
        """
        required_cols = [self.entity_col, self.timestamp_col, category_col, amount_col]
        self.validate_data(df, required_cols)

        windows = windows or [30, 60, 90]

        logger.info("Computing category concentration features")

        df_sorted = df.sort_values([self.entity_col, self.timestamp_col])

        features = []

        for window in windows:
            # Rolling category counts
            df_sorted["_temp_date"] = df_sorted[self.timestamp_col]

            # For each row, calculate Herfindahl index over window
            def compute_herfindahl(group):
                """Compute Herfindahl-Hirschman Index (HHI) for concentration."""
                if len(group) == 0:
                    return np.nan

                # Category spending shares
                cat_amounts = group.groupby(category_col)[amount_col].sum()
                total = cat_amounts.sum()

                if total == 0:
                    return np.nan

                shares = cat_amounts / total
                hhi = (shares ** 2).sum()
                return hhi

            def compute_entropy(group):
                """Compute Shannon entropy for diversity."""
                if len(group) == 0:
                    return np.nan

                cat_amounts = group.groupby(category_col)[amount_col].sum()
                total = cat_amounts.sum()

                if total == 0:
                    return np.nan

                probs = cat_amounts / total
                entropy = -np.sum(probs * np.log(probs + 1e-10))
                return entropy

            # Apply rolling window calculations
            rolling_hhi = []
            rolling_entropy = []
            rolling_unique_cats = []
            rolling_top_cat_share = []

            for idx, row in df_sorted.iterrows():
                entity = row[self.entity_col]
                current_date = row[self.timestamp_col]
                start_date = current_date - pd.Timedelta(days=window)

                # Filter data in window
                window_data = df_sorted[
                    (df_sorted[self.entity_col] == entity)
                    & (df_sorted[self.timestamp_col] >= start_date)
                    & (df_sorted[self.timestamp_col] <= current_date)
                ]

                # Calculate metrics
                hhi = compute_herfindahl(window_data)
                entropy = compute_entropy(window_data)
                unique_cats = window_data[category_col].nunique()

                # Top category share
                if len(window_data) > 0:
                    cat_amounts = window_data.groupby(category_col)[amount_col].sum()
                    total = cat_amounts.sum()
                    top_share = cat_amounts.max() / total if total > 0 else np.nan
                else:
                    top_share = np.nan

                rolling_hhi.append(hhi)
                rolling_entropy.append(entropy)
                rolling_unique_cats.append(unique_cats)
                rolling_top_cat_share.append(top_share)

            features.append(
                pd.DataFrame(
                    {
                        f"{prefix}_hhi_{window}d": rolling_hhi,
                        f"{prefix}_entropy_{window}d": rolling_entropy,
                        f"{prefix}_unique_count_{window}d": rolling_unique_cats,
                        f"{prefix}_top_share_{window}d": rolling_top_cat_share,
                    },
                    index=df_sorted.index,
                )
            )

        result_df = pd.concat([df_sorted.reset_index(drop=True)] + features, axis=1)

        if "_temp_date" in result_df.columns:
            result_df = result_df.drop(columns=["_temp_date"])

        logger.info(f"Generated {len(features) * 4} category concentration features")

        return result_df

    def compute_merchant_diversity_features(
        self,
        df: pd.DataFrame,
        merchant_col: str = "merchant",
        amount_col: str = "amount",
        windows: Optional[List[int]] = None,
        prefix: str = "merchant_div",
    ) -> pd.DataFrame:
        """
        Compute merchant diversity metrics.

        Args:
            df: Input dataframe
            merchant_col: Column containing merchant names
            amount_col: Column containing amounts
            windows: Time windows in days
            prefix: Prefix for feature names

        Returns:
            DataFrame with merchant diversity features

        Example:
            >>> df = pd.DataFrame({
            ...     'user_id': [1, 1, 1, 1],
            ...     'timestamp': pd.date_range('2024-01-01', periods=4, freq='D'),
            ...     'merchant': ['Starbucks', 'Walmart', 'Starbucks', 'Target'],
            ...     'amount': [5, 100, 6, 75]
            ... })
            >>> engine = PersonaFeatureEngine()
            >>> div_df = engine.compute_merchant_diversity_features(df)
        """
        required_cols = [self.entity_col, self.timestamp_col, merchant_col, amount_col]
        self.validate_data(df, required_cols)

        windows = windows or [30, 60, 90]

        logger.info("Computing merchant diversity features")

        df_sorted = df.sort_values([self.entity_col, self.timestamp_col])
        grouped = df_sorted.groupby(self.entity_col)

        features = []

        for window in windows:
            # Unique merchant count
            unique_merchants = (
                grouped.apply(
                    lambda x: x.set_index(self.timestamp_col)[merchant_col]
                    .rolling(f"{window}D")
                    .apply(lambda s: s.nunique(), raw=False)
                )
                .reset_index(level=0, drop=True)
            )

            # Repeat merchant ratio (merchants visited >1 time)
            def repeat_ratio(series):
                if len(series) == 0:
                    return np.nan
                counts = series.value_counts()
                repeat_merchants = (counts > 1).sum()
                return repeat_merchants / len(counts) if len(counts) > 0 else 0

            repeat_merchant_ratio = (
                grouped.apply(
                    lambda x: x.set_index(self.timestamp_col)[merchant_col]
                    .rolling(f"{window}D")
                    .apply(repeat_ratio, raw=False)
                )
                .reset_index(level=0, drop=True)
            )

            # New merchant rate (% of transactions at new merchants)
            df_sorted["_is_new_merchant"] = ~df_sorted.duplicated(
                subset=[self.entity_col, merchant_col], keep="first"
            )

            new_merchant_rate = (
                grouped.apply(
                    lambda x: x.set_index(self.timestamp_col)["_is_new_merchant"]
                    .astype(int)
                    .rolling(f"{window}D")
                    .mean()
                )
                .reset_index(level=0, drop=True)
            )

            features.append(
                pd.DataFrame(
                    {
                        f"{prefix}_unique_count_{window}d": unique_merchants,
                        f"{prefix}_repeat_ratio_{window}d": repeat_merchant_ratio,
                        f"{prefix}_new_rate_{window}d": new_merchant_rate,
                    },
                    index=df_sorted.index,
                )
            )

        result_df = pd.concat([df_sorted.reset_index(drop=True)] + features, axis=1)

        if "_is_new_merchant" in result_df.columns:
            result_df = result_df.drop(columns=["_is_new_merchant"])

        logger.info(f"Generated {len(features) * 3} merchant diversity features")

        return result_df

    def compute_all_features(
        self,
        df: pd.DataFrame,
        amount_col: str = "amount",
        category_col: Optional[str] = None,
        merchant_col: Optional[str] = None,
        description_col: Optional[str] = None,
        include_nlp: bool = True,
        include_clustering: bool = True,
    ) -> pd.DataFrame:
        """
        Compute all persona and segmentation features at once.

        Args:
            df: Input dataframe
            amount_col: Column containing amounts
            category_col: Optional category column
            merchant_col: Optional merchant column
            description_col: Optional description column
            include_nlp: Whether to include NLP features
            include_clustering: Whether to include clustering features

        Returns:
            DataFrame with all persona features

        Example:
            >>> df = pd.DataFrame({
            ...     'user_id': [1, 1, 2, 2],
            ...     'timestamp': pd.date_range('2024-01-01', periods=4, freq='D'),
            ...     'amount': [50, 75, 20, 25],
            ...     'category': ['Food', 'Retail', 'Food', 'Food'],
            ...     'merchant': ['Starbucks', 'Walmart', 'McDonalds', 'Starbucks']
            ... })
            >>> engine = PersonaFeatureEngine()
            >>> all_features = engine.compute_all_features(df)
        """
        logger.info("Computing all persona features")
        result_df = df.copy()

        if include_nlp and description_col and description_col in df.columns:
            result_df = self.compute_transaction_description_features(
                result_df, description_col=description_col
            )

        if merchant_col and merchant_col in df.columns:
            result_df = self.compute_merchant_segmentation_features(
                result_df, merchant_col=merchant_col, amount_col=amount_col
            )
            result_df = self.compute_merchant_diversity_features(
                result_df, merchant_col=merchant_col, amount_col=amount_col
            )

        if category_col and category_col in df.columns:
            result_df = self.compute_category_concentration_features(
                result_df, category_col=category_col, amount_col=amount_col
            )

        if include_clustering:
            result_df = self.compute_behavioral_persona_features(
                result_df,
                amount_col=amount_col,
                category_col=category_col if category_col in df.columns else None,
                merchant_col=merchant_col if merchant_col in df.columns else None,
            )

        logger.info(
            f"Generated {len(result_df.columns) - len(df.columns)} persona features"
        )

        return result_df


if __name__ == "__main__":
    # Example usage
    logger.info("Running persona features example")

    # Create sample data
    np.random.seed(42)
    n_samples = 100
    df = pd.DataFrame(
        {
            "user_id": np.random.choice([1, 2, 3], n_samples),
            "timestamp": pd.date_range("2024-01-01", periods=n_samples, freq="D"),
            "amount": np.random.gamma(2, 20, n_samples),
            "category": np.random.choice(
                ["Food", "Retail", "Gas", "Entertainment"], n_samples
            ),
            "merchant": np.random.choice(
                ["Starbucks", "Walmart", "Shell", "Target", "McDonalds"], n_samples
            ),
        }
    )

    # Initialize engine
    engine = PersonaFeatureEngine()

    # Compute all features
    features_df = engine.compute_all_features(
        df, category_col="category", merchant_col="merchant"
    )

    logger.info(f"Generated features shape: {features_df.shape}")
    logger.info(f"Sample persona features: {[c for c in features_df.columns if 'persona' in c][:5]}")
