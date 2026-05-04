"""
Cohort Analysis and Customer Segmentation

Analyze customer behavior across cohorts and segments:
- Cohort retention analysis
- Customer segmentation (K-means, hierarchical)
- Segment profiling and characterization
- Cohort-based model performance
- Lifetime value by cohort

Use Cases:
- Understand customer retention patterns
- Identify high-value customer segments
- Tailor models/strategies by segment
- Track cohort-specific metrics

Usage:
    analyzer = CohortAnalyzer(
        transactions_df,
        customer_id_col="customer_id",
        date_col="event_date"
    )

    retention = analyzer.compute_retention_curve(cohort_period="month")
    segments = analyzer.segment_customers(n_clusters=5)
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class CohortAnalyzer:
    """
    Analyze customer cohorts and retention patterns.

    Cohort: Group of customers who share a common characteristic
    (e.g., acquisition month, first purchase month)
    """

    def __init__(
        self,
        transactions_df: pd.DataFrame,
        customer_id_col: str = "customer_id",
        date_col: str = "event_date",
        value_col: Optional[str] = "amount"
    ):
        """
        Initialize cohort analyzer.

        Args:
            transactions_df: Transaction history DataFrame
            customer_id_col: Customer ID column name
            date_col: Date column name
            value_col: Transaction value column (optional)
        """
        self.transactions = transactions_df.copy()
        self.customer_id_col = customer_id_col
        self.date_col = date_col
        self.value_col = value_col

        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(self.transactions[date_col]):
            self.transactions[date_col] = pd.to_datetime(self.transactions[date_col])

        logger.info(f"CohortAnalyzer initialized:")
        logger.info(f"  Transactions: {len(self.transactions):,}")
        logger.info(f"  Customers: {self.transactions[customer_id_col].nunique():,}")
        logger.info(f"  Date range: {self.transactions[date_col].min()} to {self.transactions[date_col].max()}")

    def compute_retention_curve(
        self,
        cohort_period: str = "month",
        retention_periods: int = 12
    ) -> pd.DataFrame:
        """
        Compute retention curve by cohort.

        Retention curve shows what % of customers from each cohort
        are still active N periods later.

        Args:
            cohort_period: "month", "quarter", or "year"
            retention_periods: Number of periods to track

        Returns:
            DataFrame with retention rates by cohort and period
        """
        logger.info(f"Computing retention curve (cohort_period: {cohort_period})")

        # Identify each customer's cohort (first transaction period)
        customer_cohorts = self.transactions.groupby(self.customer_id_col)[self.date_col].min().reset_index()
        customer_cohorts.columns = [self.customer_id_col, "cohort_date"]

        # Convert to cohort period
        if cohort_period == "month":
            customer_cohorts["cohort"] = customer_cohorts["cohort_date"].dt.to_period("M")
            self.transactions["period"] = self.transactions[self.date_col].dt.to_period("M")
        elif cohort_period == "quarter":
            customer_cohorts["cohort"] = customer_cohorts["cohort_date"].dt.to_period("Q")
            self.transactions["period"] = self.transactions[self.date_col].dt.to_period("Q")
        elif cohort_period == "year":
            customer_cohorts["cohort"] = customer_cohorts["cohort_date"].dt.to_period("Y")
            self.transactions["period"] = self.transactions[self.date_col].dt.to_period("Y")
        else:
            raise ValueError(f"Unknown cohort_period: {cohort_period}")

        # Merge cohort info with transactions
        cohort_data = self.transactions.merge(customer_cohorts, on=self.customer_id_col)

        # Compute period offset from cohort
        cohort_data["period_number"] = (
            cohort_data["period"].astype("int64") - cohort_data["cohort"].astype("int64")
        )

        # Count active customers by cohort and period
        retention_matrix = cohort_data.groupby(["cohort", "period_number"])[self.customer_id_col].nunique()
        retention_matrix = retention_matrix.reset_index()
        retention_matrix.columns = ["cohort", "period_number", "active_customers"]

        # Pivot to matrix format
        retention_pivot = retention_matrix.pivot(
            index="cohort",
            columns="period_number",
            values="active_customers"
        ).fillna(0)

        # Compute retention rates (% of period 0)
        retention_rates = retention_pivot.div(retention_pivot[0], axis=0) * 100

        logger.info(f"✓ Retention curve computed ({len(retention_rates)} cohorts)")

        return retention_rates

    def segment_customers(
        self,
        features: Optional[List[str]] = None,
        n_clusters: int = 5,
        method: str = "kmeans"
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Segment customers using clustering.

        Args:
            features: Feature columns to use for segmentation
                     (None = auto-generate RFM features)
            n_clusters: Number of segments
            method: "kmeans" or "hierarchical"

        Returns:
            Tuple of (customer_segments DataFrame, cluster_stats)
        """
        logger.info(f"Segmenting customers (n_clusters: {n_clusters}, method: {method})")

        # Generate features if not provided
        if features is None:
            customer_features = self._compute_rfm_features()
            feature_cols = ["recency", "frequency", "monetary"]
        else:
            customer_features = self.transactions.groupby(self.customer_id_col)[features].mean().reset_index()
            feature_cols = features

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(customer_features[feature_cols])

        # Cluster
        if method == "kmeans":
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        elif method == "hierarchical":
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
        else:
            raise ValueError(f"Unknown method: {method}")

        clusters = clusterer.fit_predict(X_scaled)

        # Add cluster labels
        customer_features["segment"] = clusters

        # Compute segment statistics
        segment_stats = self._compute_segment_stats(customer_features, feature_cols)

        logger.info(f"✓ Customers segmented into {n_clusters} segments")

        return customer_features, segment_stats

    def _compute_rfm_features(self) -> pd.DataFrame:
        """
        Compute RFM (Recency, Frequency, Monetary) features.

        RFM is a classic customer segmentation approach:
        - Recency: Days since last transaction
        - Frequency: Number of transactions
        - Monetary: Total transaction value
        """
        reference_date = self.transactions[self.date_col].max()

        rfm = self.transactions.groupby(self.customer_id_col).agg({
            self.date_col: lambda x: (reference_date - x.max()).days,  # Recency
            self.customer_id_col: "count",  # Frequency
            self.value_col: "sum" if self.value_col else "count"  # Monetary
        }).reset_index()

        rfm.columns = [self.customer_id_col, "recency", "frequency", "monetary"]

        return rfm

    def _compute_segment_stats(
        self,
        customer_features: pd.DataFrame,
        feature_cols: List[str]
    ) -> Dict[str, Any]:
        """Compute statistics for each segment"""
        segment_stats = {}

        for segment_id in customer_features["segment"].unique():
            segment_data = customer_features[customer_features["segment"] == segment_id]

            stats = {
                "size": len(segment_data),
                "size_pct": len(segment_data) / len(customer_features) * 100,
                "features": {}
            }

            for feature in feature_cols:
                stats["features"][feature] = {
                    "mean": segment_data[feature].mean(),
                    "median": segment_data[feature].median(),
                    "std": segment_data[feature].std()
                }

            segment_stats[f"segment_{segment_id}"] = stats

        return segment_stats

    def profile_segments(
        self,
        customer_segments: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create detailed segment profiles.

        Args:
            customer_segments: Output from segment_customers()

        Returns:
            Segment profile DataFrame
        """
        logger.info("Profiling segments...")

        # Merge with transaction data
        segment_transactions = self.transactions.merge(
            customer_segments[[self.customer_id_col, "segment"]],
            on=self.customer_id_col
        )

        # Aggregate by segment
        profiles = segment_transactions.groupby("segment").agg({
            self.customer_id_col: "nunique",
            self.date_col: ["min", "max"],
            self.value_col: ["mean", "median", "sum"] if self.value_col else "count"
        }).reset_index()

        # Flatten column names
        profiles.columns = ["_".join(col).strip("_") for col in profiles.columns.values]

        logger.info(f"✓ Segment profiles created for {len(profiles)} segments")

        return profiles

    def compute_lifetime_value_by_cohort(
        self,
        cohort_period: str = "month"
    ) -> pd.DataFrame:
        """
        Compute average customer lifetime value by cohort.

        Args:
            cohort_period: "month", "quarter", or "year"

        Returns:
            DataFrame with LTV by cohort
        """
        if not self.value_col:
            raise ValueError("value_col required for LTV computation")

        logger.info(f"Computing LTV by cohort (cohort_period: {cohort_period})")

        # Identify cohorts
        customer_cohorts = self.transactions.groupby(self.customer_id_col)[self.date_col].min().reset_index()
        customer_cohorts.columns = [self.customer_id_col, "cohort_date"]

        if cohort_period == "month":
            customer_cohorts["cohort"] = customer_cohorts["cohort_date"].dt.to_period("M")
        elif cohort_period == "quarter":
            customer_cohorts["cohort"] = customer_cohorts["cohort_date"].dt.to_period("Q")
        elif cohort_period == "year":
            customer_cohorts["cohort"] = customer_cohorts["cohort_date"].dt.to_period("Y")

        # Compute LTV per customer
        customer_ltv = self.transactions.groupby(self.customer_id_col)[self.value_col].sum().reset_index()
        customer_ltv.columns = [self.customer_id_col, "ltv"]

        # Merge with cohorts
        cohort_ltv = customer_cohorts.merge(customer_ltv, on=self.customer_id_col)

        # Aggregate by cohort
        ltv_by_cohort = cohort_ltv.groupby("cohort")["ltv"].agg(["mean", "median", "sum", "count"]).reset_index()
        ltv_by_cohort.columns = ["cohort", "avg_ltv", "median_ltv", "total_ltv", "customer_count"]

        logger.info(f"✓ LTV computed for {len(ltv_by_cohort)} cohorts")

        return ltv_by_cohort


def plot_retention_curve(retention_df: pd.DataFrame, title: str = "Cohort Retention Curve"):
    """
    Plot retention curve visualization.

    Args:
        retention_df: Retention rates DataFrame from compute_retention_curve()
        title: Plot title
    """
    plt.figure(figsize=(12, 6))

    for cohort in retention_df.index[-5:]:  # Plot last 5 cohorts
        plt.plot(
            retention_df.columns,
            retention_df.loc[cohort],
            marker='o',
            label=str(cohort)
        )

    plt.xlabel("Periods Since Cohort Start")
    plt.ylabel("Retention Rate (%)")
    plt.title(title)
    plt.legend(title="Cohort")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    return plt


def plot_segment_profiles(
    customer_segments: pd.DataFrame,
    feature1: str = "recency",
    feature2: str = "monetary"
):
    """
    Plot customer segments in 2D feature space.

    Args:
        customer_segments: Segmented customers DataFrame
        feature1: First feature for x-axis
        feature2: Second feature for y-axis
    """
    plt.figure(figsize=(10, 8))

    for segment in customer_segments["segment"].unique():
        segment_data = customer_segments[customer_segments["segment"] == segment]

        plt.scatter(
            segment_data[feature1],
            segment_data[feature2],
            label=f"Segment {segment}",
            alpha=0.6,
            s=50
        )

    plt.xlabel(feature1.capitalize())
    plt.ylabel(feature2.capitalize())
    plt.title("Customer Segments")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    return plt


# Example usage
if __name__ == "__main__":
    # Generate sample transaction data
    np.random.seed(42)

    dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")
    transactions = []

    for customer_id in range(1000):
        # Random first transaction date
        first_date = np.random.choice(dates[:180])

        # Random number of transactions
        n_transactions = np.random.randint(1, 20)

        for _ in range(n_transactions):
            # Random transaction date after first
            days_offset = np.random.randint(0, (dates[-1] - first_date).days)
            txn_date = first_date + timedelta(days=days_offset)

            # Random amount
            amount = np.random.lognormal(4, 1)

            transactions.append({
                "customer_id": f"CUST{customer_id:04d}",
                "event_date": txn_date,
                "amount": amount
            })

    transactions_df = pd.DataFrame(transactions)

    # Initialize analyzer
    analyzer = CohortAnalyzer(
        transactions_df,
        customer_id_col="customer_id",
        date_col="event_date",
        value_col="amount"
    )

    # Compute retention
    retention = analyzer.compute_retention_curve(cohort_period="month")
    print("\nRetention Curve:")
    print(retention.head())

    # Segment customers
    segments, stats = analyzer.segment_customers(n_clusters=5)
    print(f"\nSegments created: {segments['segment'].nunique()}")
    print("\nSegment sizes:")
    print(segments.groupby("segment").size())

    # Compute LTV by cohort
    ltv = analyzer.compute_lifetime_value_by_cohort(cohort_period="month")
    print("\nLTV by Cohort:")
    print(ltv.head())

    # Plot retention curve
    # plot_retention_curve(retention).show()
    print("\nRetention and segmentation analysis complete!")
