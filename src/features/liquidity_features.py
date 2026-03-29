"""
Liquidity Features Engineering Module

Implements liquidity-specific features for financial decision making:
- OTB (On-The-Book) behavioral optionality features
- Repayment buffer features (days to due, payment cushion)
- Installment lock impacts (EMI burden, payment regularity)
- Behavioral drag metrics
- Utilization patterns
- Credit availability metrics

Author: Principal Data Science Decision Agent
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from pathlib import Path
from scipy import stats


class LiquidityFeatureConfig:
    """Configuration for liquidity feature engineering."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize liquidity feature configuration.

        Args:
            config_path: Path to feature_config.yaml. If None, uses defaults.

        Example:
            >>> config = LiquidityFeatureConfig("config/feature_config.yaml")
            >>> config.otb_metrics
            ['otb_utilization', 'otb_velocity', 'otb_stability', 'otb_momentum']
        """
        if config_path and Path(config_path).exists():
            with open(config_path, "r") as f:
                full_config = yaml.safe_load(f)
                liquidity_config = full_config.get("liquidity", {})
        else:
            liquidity_config = {}

        self.otb_metrics = liquidity_config.get(
            "otb_metrics",
            ["otb_utilization", "otb_velocity", "otb_stability", "otb_momentum"],
        )
        self.repayment_buffers = liquidity_config.get(
            "repayment_buffers",
            ["days_to_due", "payment_cushion", "min_payment_ratio"],
        )
        self.installment_lock = liquidity_config.get(
            "installment_lock",
            ["emi_burden", "installment_volatility", "payment_regularity"],
        )


class LiquidityFeatureEngine:
    """
    Engine for computing liquidity-specific features from financial data.

    Focuses on credit availability, utilization patterns, repayment capacity,
    and behavioral optionality in credit products.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        entity_col: str = "user_id",
        timestamp_col: str = "timestamp",
    ):
        """
        Initialize liquidity feature engine.

        Args:
            config_path: Path to feature_config.yaml
            entity_col: Column name for entity identifier
            timestamp_col: Column name for timestamp

        Example:
            >>> engine = LiquidityFeatureEngine(
            ...     config_path="config/feature_config.yaml",
            ...     entity_col="customer_id"
            ... )
        """
        self.config = LiquidityFeatureConfig(config_path)
        self.entity_col = entity_col
        self.timestamp_col = timestamp_col
        logger.info("Initialized LiquidityFeatureEngine")

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

        if not pd.api.types.is_datetime64_any_dtype(df[self.timestamp_col]):
            raise ValueError(
                f"{self.timestamp_col} must be datetime type. "
                f"Got {df[self.timestamp_col].dtype}"
            )

    def compute_otb_features(
        self,
        df: pd.DataFrame,
        credit_limit_col: str = "credit_limit",
        outstanding_balance_col: str = "outstanding_balance",
        windows: Optional[List[int]] = None,
        prefix: str = "otb",
    ) -> pd.DataFrame:
        """
        Compute On-The-Book (OTB) behavioral optionality features.

        OTB = Credit Limit - Outstanding Balance (available credit)

        Args:
            df: Input dataframe
            credit_limit_col: Column name for credit limit
            outstanding_balance_col: Column name for outstanding balance
            windows: Time windows in days for rolling features
            prefix: Prefix for feature names

        Returns:
            DataFrame with OTB features

        Example:
            >>> df = pd.DataFrame({
            ...     'user_id': [1, 1, 1],
            ...     'timestamp': pd.date_range('2024-01-01', periods=3, freq='D'),
            ...     'credit_limit': [10000, 10000, 12000],
            ...     'outstanding_balance': [3000, 4000, 3500]
            ... })
            >>> engine = LiquidityFeatureEngine()
            >>> otb_df = engine.compute_otb_features(df)
        """
        required_cols = [
            self.entity_col,
            self.timestamp_col,
            credit_limit_col,
            outstanding_balance_col,
        ]
        self.validate_data(df, required_cols)

        windows = windows or [7, 14, 30, 60, 90]

        logger.info(f"Computing OTB features for windows: {windows}")

        df_sorted = df.sort_values([self.entity_col, self.timestamp_col])

        # Calculate OTB
        df_sorted[f"{prefix}_absolute"] = (
            df_sorted[credit_limit_col] - df_sorted[outstanding_balance_col]
        )

        # OTB Utilization (what % of available credit is being used)
        df_sorted[f"{prefix}_utilization"] = (
            df_sorted[outstanding_balance_col]
            / (df_sorted[credit_limit_col] + 1e-10)
        ).clip(0, 1)

        # Available credit ratio
        df_sorted[f"{prefix}_available_ratio"] = (
            df_sorted[f"{prefix}_absolute"] / (df_sorted[credit_limit_col] + 1e-10)
        ).clip(0, 1)

        # Features across time windows
        features = []
        grouped = df_sorted.groupby(self.entity_col)

        for window in windows:
            # OTB velocity (rate of change in available credit)
            otb_velocity = (
                grouped.apply(
                    lambda x: x.set_index(self.timestamp_col)[f"{prefix}_absolute"]
                    .diff()
                    .rolling(f"{window}D")
                    .mean()
                )
                .reset_index(level=0, drop=True)
            )

            # OTB volatility (stability of available credit)
            otb_std = (
                grouped.apply(
                    lambda x: x.set_index(self.timestamp_col)[f"{prefix}_absolute"]
                    .rolling(f"{window}D")
                    .std()
                )
                .reset_index(level=0, drop=True)
            )

            otb_mean = (
                grouped.apply(
                    lambda x: x.set_index(self.timestamp_col)[f"{prefix}_absolute"]
                    .rolling(f"{window}D")
                    .mean()
                )
                .reset_index(level=0, drop=True)
            )

            otb_cv = otb_std / (otb_mean.abs() + 1e-10)

            # Utilization trends
            util_mean = (
                grouped.apply(
                    lambda x: x.set_index(self.timestamp_col)[f"{prefix}_utilization"]
                    .rolling(f"{window}D")
                    .mean()
                )
                .reset_index(level=0, drop=True)
            )

            util_max = (
                grouped.apply(
                    lambda x: x.set_index(self.timestamp_col)[f"{prefix}_utilization"]
                    .rolling(f"{window}D")
                    .max()
                )
                .reset_index(level=0, drop=True)
            )

            # OTB momentum (second derivative)
            otb_momentum = (
                grouped.apply(
                    lambda x: x.set_index(self.timestamp_col)[f"{prefix}_absolute"]
                    .diff()
                    .diff()
                    .rolling(f"{window}D")
                    .mean()
                )
                .reset_index(level=0, drop=True)
            )

            features.append(
                pd.DataFrame(
                    {
                        f"{prefix}_velocity_{window}d": otb_velocity,
                        f"{prefix}_volatility_cv_{window}d": otb_cv,
                        f"{prefix}_utilization_mean_{window}d": util_mean,
                        f"{prefix}_utilization_max_{window}d": util_max,
                        f"{prefix}_momentum_{window}d": otb_momentum,
                    },
                    index=df_sorted.index,
                )
            )

        result_df = pd.concat([df_sorted.reset_index(drop=True)] + features, axis=1)
        logger.info(f"Generated {3 + len(features) * 5} OTB features")

        return result_df

    def compute_repayment_buffer_features(
        self,
        df: pd.DataFrame,
        due_date_col: str = "due_date",
        payment_amount_col: str = "payment_amount",
        balance_col: str = "outstanding_balance",
        income_col: Optional[str] = None,
        prefix: str = "repay_buffer",
    ) -> pd.DataFrame:
        """
        Compute repayment buffer features (capacity to meet obligations).

        Args:
            df: Input dataframe
            due_date_col: Column name for payment due date
            payment_amount_col: Column name for required payment amount
            balance_col: Column name for current balance
            income_col: Optional column name for income
            prefix: Prefix for feature names

        Returns:
            DataFrame with repayment buffer features

        Example:
            >>> df = pd.DataFrame({
            ...     'user_id': [1, 1, 1],
            ...     'timestamp': pd.date_range('2024-01-01', periods=3, freq='D'),
            ...     'due_date': pd.date_range('2024-01-15', periods=3, freq='M'),
            ...     'payment_amount': [500, 500, 550],
            ...     'outstanding_balance': [5000, 4800, 4600]
            ... })
            >>> engine = LiquidityFeatureEngine()
            >>> buffer_df = engine.compute_repayment_buffer_features(df)
        """
        required_cols = [
            self.entity_col,
            self.timestamp_col,
            due_date_col,
            payment_amount_col,
            balance_col,
        ]
        self.validate_data(df, required_cols)

        logger.info("Computing repayment buffer features")

        df_sorted = df.sort_values([self.entity_col, self.timestamp_col])

        # Ensure due_date is datetime
        if not pd.api.types.is_datetime64_any_dtype(df_sorted[due_date_col]):
            df_sorted[due_date_col] = pd.to_datetime(df_sorted[due_date_col])

        # Days to due (buffer time)
        df_sorted[f"{prefix}_days_to_due"] = (
            df_sorted[due_date_col] - df_sorted[self.timestamp_col]
        ).dt.days

        # Payment cushion (ratio of payment to balance)
        df_sorted[f"{prefix}_payment_ratio"] = (
            df_sorted[payment_amount_col] / (df_sorted[balance_col] + 1e-10)
        ).clip(0, 1)

        # Minimum payment coverage (can they afford minimum payment?)
        if income_col and income_col in df.columns:
            df_sorted[f"{prefix}_payment_to_income"] = (
                df_sorted[payment_amount_col] / (df_sorted[income_col] + 1e-10)
            ).clip(0, 1)

            df_sorted[f"{prefix}_debt_to_income"] = (
                df_sorted[balance_col] / (df_sorted[income_col] + 1e-10)
            )
        else:
            logger.info("Income column not provided, skipping income-based metrics")

        # Rolling window features
        grouped = df_sorted.groupby(self.entity_col)
        windows = [30, 60, 90]

        features = []
        for window in windows:
            # Average days to due
            avg_days_to_due = (
                grouped.apply(
                    lambda x: x.set_index(self.timestamp_col)[f"{prefix}_days_to_due"]
                    .rolling(f"{window}D")
                    .mean()
                )
                .reset_index(level=0, drop=True)
            )

            # Minimum days to due (tightest deadline)
            min_days_to_due = (
                grouped.apply(
                    lambda x: x.set_index(self.timestamp_col)[f"{prefix}_days_to_due"]
                    .rolling(f"{window}D")
                    .min()
                )
                .reset_index(level=0, drop=True)
            )

            # Payment ratio trends
            avg_payment_ratio = (
                grouped.apply(
                    lambda x: x.set_index(self.timestamp_col)[f"{prefix}_payment_ratio"]
                    .rolling(f"{window}D")
                    .mean()
                )
                .reset_index(level=0, drop=True)
            )

            features.append(
                pd.DataFrame(
                    {
                        f"{prefix}_avg_days_to_due_{window}d": avg_days_to_due,
                        f"{prefix}_min_days_to_due_{window}d": min_days_to_due,
                        f"{prefix}_avg_payment_ratio_{window}d": avg_payment_ratio,
                    },
                    index=df_sorted.index,
                )
            )

        result_df = pd.concat([df_sorted.reset_index(drop=True)] + features, axis=1)
        logger.info(f"Generated repayment buffer features")

        return result_df

    def compute_installment_lock_features(
        self,
        df: pd.DataFrame,
        emi_col: str = "emi_amount",
        income_col: Optional[str] = None,
        payment_date_col: Optional[str] = None,
        windows: Optional[List[int]] = None,
        prefix: str = "installment",
    ) -> pd.DataFrame:
        """
        Compute installment lock features (EMI burden, payment regularity).

        Args:
            df: Input dataframe
            emi_col: Column name for EMI amount
            income_col: Optional column name for income
            payment_date_col: Optional column for payment dates
            windows: Time windows in days
            prefix: Prefix for feature names

        Returns:
            DataFrame with installment lock features

        Example:
            >>> df = pd.DataFrame({
            ...     'user_id': [1, 1, 1],
            ...     'timestamp': pd.date_range('2024-01-01', periods=3, freq='D'),
            ...     'emi_amount': [1000, 1000, 1200],
            ...     'income': [5000, 5000, 5500]
            ... })
            >>> engine = LiquidityFeatureEngine()
            >>> install_df = engine.compute_installment_lock_features(df)
        """
        required_cols = [self.entity_col, self.timestamp_col, emi_col]
        self.validate_data(df, required_cols)

        windows = windows or [30, 60, 90]

        logger.info("Computing installment lock features")

        df_sorted = df.sort_values([self.entity_col, self.timestamp_col])

        # EMI burden (if income available)
        if income_col and income_col in df.columns:
            df_sorted[f"{prefix}_emi_burden"] = (
                df_sorted[emi_col] / (df_sorted[income_col] + 1e-10)
            ).clip(0, 1)
        else:
            logger.info("Income column not provided for EMI burden calculation")

        # EMI volatility
        grouped = df_sorted.groupby(self.entity_col)

        features = []
        for window in windows:
            # EMI stability (CV)
            emi_mean = (
                grouped.apply(
                    lambda x: x.set_index(self.timestamp_col)[emi_col]
                    .rolling(f"{window}D")
                    .mean()
                )
                .reset_index(level=0, drop=True)
            )

            emi_std = (
                grouped.apply(
                    lambda x: x.set_index(self.timestamp_col)[emi_col]
                    .rolling(f"{window}D")
                    .std()
                )
                .reset_index(level=0, drop=True)
            )

            emi_cv = emi_std / (emi_mean.abs() + 1e-10)

            # EMI increase frequency
            emi_diff = grouped[emi_col].diff()
            emi_increases = (
                grouped.apply(
                    lambda x: x.assign(emi_diff_temp=emi_diff.loc[x.index])
                    .set_index(self.timestamp_col)["emi_diff_temp"]
                    .rolling(f"{window}D")
                    .apply(lambda x: (x > 0).sum())
                )
                .reset_index(level=0, drop=True)
            )

            feature_dict = {
                f"{prefix}_mean_{window}d": emi_mean,
                f"{prefix}_volatility_cv_{window}d": emi_cv,
                f"{prefix}_increase_count_{window}d": emi_increases,
            }

            # Add burden statistics if available
            if income_col and income_col in df.columns:
                burden_mean = (
                    grouped.apply(
                        lambda x: x.set_index(self.timestamp_col)[
                            f"{prefix}_emi_burden"
                        ]
                        .rolling(f"{window}D")
                        .mean()
                    )
                    .reset_index(level=0, drop=True)
                )

                burden_max = (
                    grouped.apply(
                        lambda x: x.set_index(self.timestamp_col)[
                            f"{prefix}_emi_burden"
                        ]
                        .rolling(f"{window}D")
                        .max()
                    )
                    .reset_index(level=0, drop=True)
                )

                feature_dict[f"{prefix}_burden_mean_{window}d"] = burden_mean
                feature_dict[f"{prefix}_burden_max_{window}d"] = burden_max

            features.append(pd.DataFrame(feature_dict, index=df_sorted.index))

        # Payment regularity (if payment dates available)
        if payment_date_col and payment_date_col in df.columns:
            # Ensure datetime
            if not pd.api.types.is_datetime64_any_dtype(df_sorted[payment_date_col]):
                df_sorted[payment_date_col] = pd.to_datetime(
                    df_sorted[payment_date_col], errors="coerce"
                )

            # Days between payments
            days_between = grouped[payment_date_col].diff().dt.days

            # Regularity index (lower std = more regular)
            for window in windows:
                regularity = (
                    grouped.apply(
                        lambda x: x.assign(days_temp=days_between.loc[x.index])
                        .set_index(self.timestamp_col)["days_temp"]
                        .rolling(f"{window}D")
                        .std()
                    )
                    .reset_index(level=0, drop=True)
                )

                regularity_index = 1 / (regularity + 1)

                features.append(
                    pd.DataFrame(
                        {f"{prefix}_regularity_{window}d": regularity_index},
                        index=df_sorted.index,
                    )
                )

        result_df = pd.concat([df_sorted.reset_index(drop=True)] + features, axis=1)
        logger.info(f"Generated installment lock features")

        return result_df

    def compute_utilization_pattern_features(
        self,
        df: pd.DataFrame,
        balance_col: str = "outstanding_balance",
        limit_col: str = "credit_limit",
        transaction_amount_col: Optional[str] = None,
        windows: Optional[List[int]] = None,
        prefix: str = "utilization",
    ) -> pd.DataFrame:
        """
        Compute detailed utilization pattern features.

        Args:
            df: Input dataframe
            balance_col: Column name for balance
            limit_col: Column name for credit limit
            transaction_amount_col: Optional transaction amount column
            windows: Time windows in days
            prefix: Prefix for feature names

        Returns:
            DataFrame with utilization pattern features

        Example:
            >>> df = pd.DataFrame({
            ...     'user_id': [1, 1, 1],
            ...     'timestamp': pd.date_range('2024-01-01', periods=3, freq='D'),
            ...     'outstanding_balance': [3000, 4500, 2000],
            ...     'credit_limit': [10000, 10000, 10000]
            ... })
            >>> engine = LiquidityFeatureEngine()
            >>> util_df = engine.compute_utilization_pattern_features(df)
        """
        required_cols = [self.entity_col, self.timestamp_col, balance_col, limit_col]
        self.validate_data(df, required_cols)

        windows = windows or [7, 14, 30, 60, 90]

        logger.info("Computing utilization pattern features")

        df_sorted = df.sort_values([self.entity_col, self.timestamp_col])

        # Current utilization
        df_sorted[f"{prefix}_rate"] = (
            df_sorted[balance_col] / (df_sorted[limit_col] + 1e-10)
        ).clip(0, 1)

        grouped = df_sorted.groupby(self.entity_col)
        features = []

        for window in windows:
            # Utilization statistics
            util_mean = (
                grouped.apply(
                    lambda x: x.set_index(self.timestamp_col)[f"{prefix}_rate"]
                    .rolling(f"{window}D")
                    .mean()
                )
                .reset_index(level=0, drop=True)
            )

            util_max = (
                grouped.apply(
                    lambda x: x.set_index(self.timestamp_col)[f"{prefix}_rate"]
                    .rolling(f"{window}D")
                    .max()
                )
                .reset_index(level=0, drop=True)
            )

            util_min = (
                grouped.apply(
                    lambda x: x.set_index(self.timestamp_col)[f"{prefix}_rate"]
                    .rolling(f"{window}D")
                    .min()
                )
                .reset_index(level=0, drop=True)
            )

            util_std = (
                grouped.apply(
                    lambda x: x.set_index(self.timestamp_col)[f"{prefix}_rate"]
                    .rolling(f"{window}D")
                    .std()
                )
                .reset_index(level=0, drop=True)
            )

            # High utilization events (> 80%)
            high_util_count = (
                grouped.apply(
                    lambda x: x.set_index(self.timestamp_col)[f"{prefix}_rate"]
                    .rolling(f"{window}D")
                    .apply(lambda s: (s > 0.8).sum())
                )
                .reset_index(level=0, drop=True)
            )

            # Utilization velocity
            util_velocity = (
                grouped.apply(
                    lambda x: x.set_index(self.timestamp_col)[f"{prefix}_rate"]
                    .diff()
                    .rolling(f"{window}D")
                    .mean()
                )
                .reset_index(level=0, drop=True)
            )

            features.append(
                pd.DataFrame(
                    {
                        f"{prefix}_mean_{window}d": util_mean,
                        f"{prefix}_max_{window}d": util_max,
                        f"{prefix}_min_{window}d": util_min,
                        f"{prefix}_std_{window}d": util_std,
                        f"{prefix}_range_{window}d": util_max - util_min,
                        f"{prefix}_high_events_{window}d": high_util_count,
                        f"{prefix}_velocity_{window}d": util_velocity,
                    },
                    index=df_sorted.index,
                )
            )

        result_df = pd.concat([df_sorted.reset_index(drop=True)] + features, axis=1)
        logger.info(f"Generated {1 + len(features) * 7} utilization pattern features")

        return result_df

    def compute_all_features(
        self,
        df: pd.DataFrame,
        credit_limit_col: str = "credit_limit",
        outstanding_balance_col: str = "outstanding_balance",
        include_otb: bool = True,
        include_utilization: bool = True,
    ) -> pd.DataFrame:
        """
        Compute all liquidity features at once.

        Args:
            df: Input dataframe
            credit_limit_col: Column name for credit limit
            outstanding_balance_col: Column name for outstanding balance
            include_otb: Whether to include OTB features
            include_utilization: Whether to include utilization features

        Returns:
            DataFrame with all requested liquidity features

        Example:
            >>> df = pd.DataFrame({
            ...     'user_id': [1, 1, 1],
            ...     'timestamp': pd.date_range('2024-01-01', periods=3, freq='D'),
            ...     'credit_limit': [10000, 10000, 12000],
            ...     'outstanding_balance': [3000, 4000, 3500]
            ... })
            >>> engine = LiquidityFeatureEngine()
            >>> all_features = engine.compute_all_features(df)
        """
        logger.info("Computing all liquidity features")
        result_df = df.copy()

        if include_otb:
            otb_df = self.compute_otb_features(
                df,
                credit_limit_col=credit_limit_col,
                outstanding_balance_col=outstanding_balance_col,
            )
            otb_cols = [col for col in otb_df.columns if col.startswith("otb_")]
            result_df = result_df.merge(
                otb_df[[self.entity_col, self.timestamp_col] + otb_cols],
                on=[self.entity_col, self.timestamp_col],
                how="left",
            )

        if include_utilization:
            util_df = self.compute_utilization_pattern_features(
                df, balance_col=outstanding_balance_col, limit_col=credit_limit_col
            )
            util_cols = [
                col for col in util_df.columns if col.startswith("utilization_")
            ]
            result_df = result_df.merge(
                util_df[[self.entity_col, self.timestamp_col] + util_cols],
                on=[self.entity_col, self.timestamp_col],
                how="left",
            )

        logger.info(
            f"Generated {len(result_df.columns) - len(df.columns)} liquidity features"
        )

        return result_df


if __name__ == "__main__":
    # Example usage
    logger.info("Running liquidity features example")

    # Create sample data
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    df = pd.DataFrame(
        {
            "user_id": np.repeat([1, 2], [50, 50]),
            "timestamp": np.tile(dates[:50], 2),
            "credit_limit": 10000,
            "outstanding_balance": np.random.uniform(1000, 8000, 100),
        }
    )

    # Initialize engine
    engine = LiquidityFeatureEngine()

    # Compute all features
    features_df = engine.compute_all_features(df)

    logger.info(f"Generated features shape: {features_df.shape}")
    logger.info(f"Sample OTB features: {[c for c in features_df.columns if 'otb' in c][:5]}")
