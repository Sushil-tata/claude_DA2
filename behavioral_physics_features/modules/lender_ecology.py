"""
Lender Ecology Engine - Cross-Lender Dynamics & CardX Interactions
===================================================================

Thailand-specific lender classification and diffusion analysis.

Features:
- Lender type exposure (SFI/Commercial Bank/Personal Loan/Leasing/Fintech/CardX)
- Lender concentration (HHI)
- Cross-lender diffusion (synchronized delinquency)
- CardX vs Others analysis

Thai Financial Institution Categories:
- SFI: Specialized Financial Institutions (สถาบันการเงินเฉพาะกิจ)
- COMMERCIAL_BANK: Commercial Banks (ธนาคารพาณิชย์)
- PERSONAL_LOAN: Personal Loan Companies (บริษัทสินเชื่อส่วนบุคคล)
- LEASING: Leasing/Hire Purchase (บริษัทลิสซิ่ง)
- FINTECH: Fintech/Digital Lenders (ฟินเทค)
- CARDX: Internal CardX lender

Author: Behavioral Physics Team
Version: 1.0.0
"""

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from typing import List, Dict

from .config import get_config


class LenderEcologyEngine:
    """
    Lender ecology and cross-lender dynamics.

    Feature Families:
    1. Exposure (8 features): Balance/limit share by lender type
    2. Concentration (3 features): HHI, num lenders, dominance
    3. Diffusion (5 features): Synchronized delinquency, spillover
    4. CardX Analysis (9 features): CardX vs Others comparison
    """

    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.config = get_config()

    def compute_all_features(
        self,
        bureau_trade_df: DataFrame,
        state_df: DataFrame
    ) -> DataFrame:
        """
        Compute all lender ecology features.

        Args:
            bureau_trade_df: Bureau trade monthly with lender info
            state_df: State assignments from StateBuilder

        Returns:
            DataFrame with lender ecology features added
        """
        # 1. Map lender types
        typed_df = self._map_lender_types(bureau_trade_df)

        # 2. Exposure by lender type
        exposure_df = self._compute_exposure_shares(typed_df)

        # 3. Lender concentration
        concentration_df = self._compute_concentration(typed_df)

        # 4. Cross-lender diffusion
        diffusion_df = self._compute_diffusion(typed_df)

        # 5. CardX vs Others
        cardx_df = self._compute_cardx_features(typed_df, state_df)

        # Join all to state_df
        result = state_df

        for df in [exposure_df, concentration_df, diffusion_df, cardx_df]:
            result = result.join(df, on=["cust_id", "as_of_month"], how="left")

        return result

    def _map_lender_types(self, bureau_df: DataFrame) -> DataFrame:
        """
        Map raw lender names to standard types.

        Returns: PSU_BANK, PRIVATE_BANK, FINTECH, CONSUMER_FINANCE, CARDX, OTHER
        """
        # UDF for lender type mapping
        map_type_udf = F.udf(
            lambda name, id: self.config.lender_types.map_lender_type(
                name or "", id or ""
            )
        )

        return bureau_df.withColumn(
            "lender_type",
            map_type_udf(
                F.col("lender_type_raw"),
                F.col("lender_id")
            )
        )

    def _compute_exposure_shares(self, typed_df: DataFrame) -> DataFrame:
        """
        Exposure share by lender type (Thai financial institution categories).

        Features:
        - sfi_balance_share (Specialized Financial Institutions)
        - commercial_bank_balance_share (Commercial Banks)
        - personal_loan_balance_share (Personal Loan Companies)
        - leasing_balance_share (Leasing/Hire Purchase)
        - fintech_balance_share (Fintech/Digital Lenders)
        - cardx_balance_share (CardX)
        - sfi_limit_share
        - commercial_bank_limit_share
        - fintech_limit_share
        - cardx_limit_share
        """
        # Total exposure by type
        type_exposure = typed_df.groupBy(
            "cust_id", "as_of_month", "lender_type"
        ).agg(
            F.sum("balance").alias("type_balance"),
            F.sum("credit_limit").alias("type_limit")
        )

        # Total exposure overall
        total_exposure = typed_df.groupBy("cust_id", "as_of_month").agg(
            F.sum("balance").alias("total_balance"),
            F.sum("credit_limit").alias("total_limit")
        )

        # Pivot to get columns by lender type
        balance_pivot = type_exposure.groupBy("cust_id", "as_of_month").pivot(
            "lender_type"
        ).agg(
            F.first("type_balance")
        )

        limit_pivot = type_exposure.groupBy("cust_id", "as_of_month").pivot(
            "lender_type"
        ).agg(
            F.first("type_limit")
        )

        # Join with totals
        shares = balance_pivot.join(
            total_exposure, on=["cust_id", "as_of_month"]
        ).join(
            limit_pivot,
            on=["cust_id", "as_of_month"],
            how="left"
        )

        # Calculate balance shares
        # Thai financial institution types
        lender_types = ["SFI", "COMMERCIAL_BANK", "PERSONAL_LOAN", "LEASING", "FINTECH", "CARDX"]

        for ltype in lender_types:
            # Balance share
            shares = shares.withColumn(
                f"{ltype.lower()}_balance_share",
                F.when(
                    F.col("total_balance") > 0,
                    F.coalesce(F.col(ltype), F.lit(0.0)) / F.col("total_balance")
                ).otherwise(0.0)
            )

            # Limit share (for selected types)
            if ltype in ["PSU_BANK", "FINTECH", "CARDX"]:
                shares = shares.withColumn(
                    f"{ltype.lower()}_limit_share",
                    F.when(
                        F.col("total_limit") > 0,
                        F.coalesce(F.col(ltype), F.lit(0.0)) / F.col("total_limit")
                    ).otherwise(0.0)
                )

        # Select final columns
        share_cols = ["cust_id", "as_of_month"]
        for ltype in lender_types:
            share_cols.append(f"{ltype.lower()}_balance_share")
        for ltype in ["PSU_BANK", "FINTECH", "CARDX"]:
            share_cols.append(f"{ltype.lower()}_limit_share")

        return shares.select(*share_cols)

    def _compute_concentration(self, typed_df: DataFrame) -> DataFrame:
        """
        Lender concentration (Herfindahl-Hirschman Index).

        Features:
        - lender_hhi: Concentration index (0=diversified, 1=single lender)
        - num_lenders: Number of unique lenders
        - dominant_lender_flag: Single lender >50% exposure
        """
        # Balance by lender
        lender_balances = typed_df.groupBy(
            "cust_id", "as_of_month", "lender_id"
        ).agg(
            F.sum("balance").alias("lender_balance")
        )

        # Total balance
        total_balance = typed_df.groupBy("cust_id", "as_of_month").agg(
            F.sum("balance").alias("total_balance")
        )

        # Calculate shares
        lender_shares = lender_balances.join(
            total_balance,
            on=["cust_id", "as_of_month"]
        ).withColumn(
            "share",
            F.when(
                F.col("total_balance") > 0,
                F.col("lender_balance") / F.col("total_balance")
            ).otherwise(0.0)
        )

        # HHI = sum of squared shares
        hhi = lender_shares.groupBy("cust_id", "as_of_month").agg(
            F.sum(F.pow(F.col("share"), 2)).alias("lender_hhi"),
            F.count("lender_id").alias("num_lenders"),
            F.max("share").alias("max_lender_share")
        )

        # Dominant lender flag (>50% with single lender)
        hhi = hhi.withColumn(
            "dominant_lender_flag",
            (F.col("max_lender_share") > 0.50).cast("int")
        )

        return hhi.select(
            "cust_id", "as_of_month",
            "lender_hhi", "num_lenders", "dominant_lender_flag"
        )

    def _compute_diffusion(self, typed_df: DataFrame) -> DataFrame:
        """
        Cross-lender diffusion features.

        Features:
        - num_lenders_delinquent: Count of lenders with DPD > 0
        - synchronized_delinquency_flag: 2+ lenders delinquent
        - fintech_delinquent_flag: Fintech lender delinquent
        - psu_delinquent_flag: PSU bank delinquent
        - diffusion_score: num_delinquent / num_lenders
        """
        # Delinquent lenders
        delinq_lenders = typed_df.filter(
            F.col("dpd") > 0
        ).groupBy("cust_id", "as_of_month").agg(
            F.count("lender_id").alias("num_lenders_delinquent"),
            F.collect_set("lender_type").alias("delinquent_lender_types")
        )

        # Total lenders
        total_lenders = typed_df.groupBy("cust_id", "as_of_month").agg(
            F.countDistinct("lender_id").alias("total_lenders")
        )

        # Join
        diffusion = total_lenders.join(
            delinq_lenders,
            on=["cust_id", "as_of_month"],
            how="left"
        )

        # Fill nulls
        diffusion = diffusion.withColumn(
            "num_lenders_delinquent",
            F.coalesce(F.col("num_lenders_delinquent"), F.lit(0))
        )

        # Synchronized delinquency flag
        diffusion = diffusion.withColumn(
            "synchronized_delinquency_flag",
            (F.col("num_lenders_delinquent") >= 2).cast("int")
        )

        # Diffusion score (proportion of lenders delinquent)
        diffusion = diffusion.withColumn(
            "diffusion_score",
            F.when(
                F.col("total_lenders") > 0,
                F.col("num_lenders_delinquent") / F.col("total_lenders")
            ).otherwise(0.0)
        )

        # Lender type delinquency flags
        diffusion = diffusion.withColumn(
            "fintech_delinquent_flag",
            F.array_contains(
                F.coalesce(F.col("delinquent_lender_types"), F.array()),
                "FINTECH"
            ).cast("int")
        )

        diffusion = diffusion.withColumn(
            "psu_delinquent_flag",
            F.array_contains(
                F.coalesce(F.col("delinquent_lender_types"), F.array()),
                "PSU_BANK"
            ).cast("int")
        )

        return diffusion.select(
            "cust_id", "as_of_month",
            "num_lenders_delinquent",
            "synchronized_delinquency_flag",
            "diffusion_score",
            "fintech_delinquent_flag",
            "psu_delinquent_flag"
        )

    def _compute_cardx_features(
        self,
        typed_df: DataFrame,
        state_df: DataFrame
    ) -> DataFrame:
        """
        CardX vs Others analysis.

        Features:
        - cardx_balance_bureau: CardX balance from bureau
        - others_total_balance: Non-CardX balance
        - cardx_vs_others_balance_ratio: CardX / Others
        - cardx_delinquent_flag: CardX account delinquent
        - others_delinquent_flag: Others delinquent
        - cardx_first_delinquent: CardX delinquent before others
        - others_first_delinquent: Others delinquent before CardX
        - cross_trigger_flag: Delinquency in one triggers other
        - cardx_vs_others_dpd_diff: CardX DPD - Others max DPD
        """
        # CardX accounts
        cardx_agg = typed_df.filter(
            F.col("lender_type") == "CARDX"
        ).groupBy("cust_id", "as_of_month").agg(
            F.sum("balance").alias("cardx_balance_bureau"),
            F.max("dpd").alias("cardx_dpd_bureau"),
            F.sum("credit_limit").alias("cardx_limit_bureau")
        )

        # Others accounts
        others_agg = typed_df.filter(
            F.col("lender_type") != "CARDX"
        ).groupBy("cust_id", "as_of_month").agg(
            F.sum("balance").alias("others_total_balance"),
            F.max("dpd").alias("others_max_dpd"),
            F.sum("credit_limit").alias("others_total_limit")
        )

        # Join with state_df
        cardx_comp = state_df.select("cust_id", "as_of_month").join(
            cardx_agg, on=["cust_id", "as_of_month"], how="left"
        ).join(
            others_agg, on=["cust_id", "as_of_month"], how="left"
        )

        # Balance ratio
        cardx_comp = cardx_comp.withColumn(
            "cardx_vs_others_balance_ratio",
            F.when(
                F.col("others_total_balance") > 0,
                F.coalesce(F.col("cardx_balance_bureau"), F.lit(0.0)) /
                F.col("others_total_balance")
            ).otherwise(F.lit(None))
        )

        # Delinquency flags
        cardx_comp = cardx_comp.withColumn(
            "cardx_delinquent_flag",
            (F.coalesce(F.col("cardx_dpd_bureau"), F.lit(0)) > 0).cast("int")
        )

        cardx_comp = cardx_comp.withColumn(
            "others_delinquent_flag",
            (F.coalesce(F.col("others_max_dpd"), F.lit(0)) > 0).cast("int")
        )

        # First delinquent flags
        cardx_comp = cardx_comp.withColumn(
            "cardx_first_delinquent",
            (
                (F.col("cardx_delinquent_flag") == 1) &
                (F.col("others_delinquent_flag") == 0)
            ).cast("int")
        )

        cardx_comp = cardx_comp.withColumn(
            "others_first_delinquent",
            (
                (F.col("others_delinquent_flag") == 1) &
                (F.col("cardx_delinquent_flag") == 0)
            ).cast("int")
        )

        # DPD difference
        cardx_comp = cardx_comp.withColumn(
            "cardx_vs_others_dpd_diff",
            F.coalesce(F.col("cardx_dpd_bureau"), F.lit(0)) -
            F.coalesce(F.col("others_max_dpd"), F.lit(0))
        )

        # Cross-trigger flag (both delinquent)
        cardx_comp = cardx_comp.withColumn(
            "cross_trigger_flag",
            (
                (F.col("cardx_delinquent_flag") == 1) &
                (F.col("others_delinquent_flag") == 1)
            ).cast("int")
        )

        return cardx_comp.select(
            "cust_id", "as_of_month",
            "cardx_balance_bureau",
            "others_total_balance",
            "cardx_vs_others_balance_ratio",
            "cardx_delinquent_flag",
            "others_delinquent_flag",
            "cardx_first_delinquent",
            "others_first_delinquent",
            "cross_trigger_flag",
            "cardx_vs_others_dpd_diff"
        )


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    from pyspark.sql import SparkSession
    from datetime import date

    spark = SparkSession.builder \
        .appName("LenderEcologyTest") \
        .master("local[*]") \
        .getOrCreate()

    print("\n" + "="*70)
    print("LENDER ECOLOGY ENGINE - EXAMPLE")
    print("="*70)

    # Sample bureau trade data
    bureau_data = [
        ("CUST001", date(2024, 1, 31), "L001", "BANGKOK BANK", "PRIVATE_BANK", 10000, 50000, 0),
        ("CUST001", date(2024, 1, 31), "L002", "RABBIT FINANCE", "FINTECH", 5000, 20000, 30),
        ("CUST001", date(2024, 1, 31), "CARDX", "CARDX", "CARDX", 15000, 30000, 0),
        ("CUST002", date(2024, 1, 31), "L003", "SCB", "PRIVATE_BANK", 20000, 100000, 45),
        ("CUST002", date(2024, 1, 31), "L004", "AEON", "FINTECH", 8000, 10000, 60),
        ("CUST002", date(2024, 1, 31), "CARDX", "CARDX", "CARDX", 12000, 25000, 30),
    ]

    bureau_df = spark.createDataFrame(
        bureau_data,
        ["cust_id", "as_of_month", "lender_id", "lender_type_raw",
         "lender_type", "balance", "credit_limit", "dpd"]
    )

    # Sample state data
    state_data = [
        ("CUST001", date(2024, 1, 31)),
        ("CUST002", date(2024, 1, 31)),
    ]

    state_df = spark.createDataFrame(state_data, ["cust_id", "as_of_month"])

    # Compute features
    engine = LenderEcologyEngine(spark)
    features_df = engine.compute_all_features(bureau_df, state_df)

    print("\nLender Ecology Features:")
    features_df.select(
        "cust_id",
        "lender_hhi",
        "num_lenders",
        "fintech_balance_share",
        "cardx_balance_share",
        "synchronized_delinquency_flag",
        "cardx_first_delinquent"
    ).show(truncate=False)

    print("\n✅ Lender Ecology Engine test complete")

    spark.stop()
