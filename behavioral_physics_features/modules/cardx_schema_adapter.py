"""
CardX Schema Adapter - Maps CardX Internal Tables to Behavioral Physics Schema
==============================================================================

Handles complete join logic:
1. CardX Monthly (spl_acct_mthly) - ACCT_NUM, CUST_NUM
2. CardX Origin (spl_ln_orig) - CUST_NUM → CUST_ID (ID_NO)
3. Bureau ID Dummy (mnf_cra_rvw_id_dummy) - ID_NO → REF_NO
4. Bureau Tables - REF_NO

Author: Behavioral Physics Team
Version: 1.0.0
"""

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from typing import Optional


class CardXSchemaAdapter:
    """
    Adapts CardX internal tables to behavioral physics schema.

    Handles join logic to connect CardX CUST_NUM to Bureau REF_NO.
    """

    def __init__(self, spark: SparkSession):
        self.spark = spark

    def adapt_cardx_monthly_data(
        self,
        cardx_monthly_df: DataFrame,
        cardx_origin_df: DataFrame,
        bureau_id_dummy_df: DataFrame
    ) -> DataFrame:
        """
        Adapt CardX monthly data with proper joins to get REF_NO (cust_id).

        Args:
            cardx_monthly_df: spl_acct_mthly table (ACCT_NUM, CUST_NUM, monthly data)
            cardx_origin_df: spl_ln_orig table (CUST_NUM → CUST_ID)
            bureau_id_dummy_df: mnf_cra_rvw_id_dummy (ID_NO → REF_NO)

        Returns:
            DataFrame with schema: cust_id, as_of_month, cardx_dpd, cardx_balance, cardx_credit_limit
        """
        print("📊 Adapting CardX schema with join logic...")

        # Step 1: Join CardX monthly with origin to get ID_NO (CUST_ID)
        cardx_with_id = cardx_monthly_df.alias("monthly").join(
            cardx_origin_df.select("CUST_NUM", F.col("CUST_ID").alias("ID_NO")).alias("orig"),
            on="CUST_NUM",
            how="left"
        )

        print(f"✓ Joined CardX monthly with origin: {cardx_with_id.count():,} rows")

        # Step 2: Join with bureau ID dummy to get REF_NO
        cardx_with_ref = cardx_with_id.join(
            bureau_id_dummy_df.select("ID_NO", "REF_NO").alias("dummy"),
            on="ID_NO",
            how="left"
        )

        print(f"✓ Joined with bureau ID dummy: {cardx_with_ref.count():,} rows")

        # Step 3: Map to behavioral physics schema
        cardx_mapped = self._map_cardx_columns(cardx_with_ref)

        # Step 4: Validate join success
        total_rows = cardx_mapped.count()
        rows_with_ref = cardx_mapped.filter(F.col("cust_id").isNotNull()).count()
        join_success_rate = (rows_with_ref / total_rows * 100) if total_rows > 0 else 0

        print(f"✓ Join success rate: {join_success_rate:.1f}% ({rows_with_ref:,}/{total_rows:,})")

        if join_success_rate < 80:
            print(f"⚠️  Warning: Low join success rate. Check if ID_NO exists in bureau_id_dummy")

        return cardx_mapped

    def _map_cardx_columns(self, cardx_df: DataFrame) -> DataFrame:
        """
        Map CardX columns to expected behavioral physics schema.

        Expected CardX columns:
        - REF_NO → cust_id
        - ACCT_NUM → account_id
        - Monthly date column → as_of_month
        - DPD column → cardx_dpd
        - Balance column → cardx_balance
        - Limit column → cardx_credit_limit
        """
        # Map REF_NO to cust_id
        cardx_df = cardx_df.withColumn("cust_id", F.col("REF_NO"))

        # Map account number
        if "ACCT_NUM" in cardx_df.columns:
            cardx_df = cardx_df.withColumn("cardx_account_id", F.col("ACCT_NUM"))

        # Map as_of_month (adjust column name based on your actual CardX table)
        # Common options: ASOFDATE, AS_OF_DT, SNAPSHOT_DT, MONTH_END_DT
        month_cols = ["ASOFDATE", "AS_OF_DT", "SNAPSHOT_DT", "MONTH_END_DT", "RPT_DT"]
        as_of_col = None
        for col in month_cols:
            if col in cardx_df.columns:
                as_of_col = col
                break

        if as_of_col:
            cardx_df = cardx_df.withColumn(
                "as_of_month",
                F.to_date(F.col(as_of_col))
            )
        else:
            print(f"⚠️  Warning: No as_of_month column found. Looking for: {month_cols}")
            print(f"    Available columns: {cardx_df.columns}")

        # Map DPD (adjust column name based on your actual CardX table)
        dpd_cols = ["DPD", "DAYS_PAST_DUE", "DELINQUENCY_DAYS", "OVERDUE_DAYS"]
        dpd_col = None
        for col in dpd_cols:
            if col in cardx_df.columns:
                dpd_col = col
                break

        if dpd_col:
            cardx_df = cardx_df.withColumn("cardx_dpd", F.col(dpd_col))
        else:
            print(f"⚠️  Warning: No DPD column found. Looking for: {dpd_cols}")
            cardx_df = cardx_df.withColumn("cardx_dpd", F.lit(0))

        # Map balance (adjust column name based on your actual CardX table)
        balance_cols = ["BALANCE", "OUTSTANDING_BALANCE", "CURR_BAL", "PRINCIPAL_BALANCE"]
        balance_col = None
        for col in balance_cols:
            if col in cardx_df.columns:
                balance_col = col
                break

        if balance_col:
            cardx_df = cardx_df.withColumn("cardx_balance", F.col(balance_col))
        else:
            print(f"⚠️  Warning: No balance column found. Looking for: {balance_cols}")
            cardx_df = cardx_df.withColumn("cardx_balance", F.lit(0))

        # Map credit limit (adjust column name based on your actual CardX table)
        limit_cols = ["CREDIT_LIMIT", "LIMIT", "APPROVED_LIMIT", "SANCTION_LIMIT"]
        limit_col = None
        for col in limit_cols:
            if col in cardx_df.columns:
                limit_col = col
                break

        if limit_col:
            cardx_df = cardx_df.withColumn("cardx_credit_limit", F.col(limit_col))
        else:
            print(f"⚠️  Warning: No credit limit column found. Looking for: {limit_cols}")
            cardx_df = cardx_df.withColumn("cardx_credit_limit", F.lit(0))

        # Select final columns
        final_cols = ["cust_id", "as_of_month", "cardx_dpd", "cardx_balance", "cardx_credit_limit"]

        # Add optional columns if they exist
        if "cardx_account_id" in cardx_df.columns:
            final_cols.append("cardx_account_id")

        return cardx_df.select(*final_cols)

    def load_and_adapt_cardx(
        self,
        catalog: str = "cdx_mdz_prd"
    ) -> DataFrame:
        """
        Convenience method to load all CardX tables and perform complete adaptation.

        Args:
            catalog: Databricks catalog name

        Returns:
            Adapted CardX DataFrame ready for behavioral physics
        """
        print("Loading CardX tables...")

        # Load CardX monthly
        cardx_monthly = self.spark.table(f"{catalog}.cdx_curated_spl_acl_db.spl_acct_mthly")
        print(f"✓ Loaded spl_acct_mthly: {cardx_monthly.count():,} rows")

        # Load CardX origin
        cardx_origin = self.spark.table(f"{catalog}.cdx_curated_spl_acl_db.spl_ln_orig")
        print(f"✓ Loaded spl_ln_orig: {cardx_origin.count():,} rows")

        # Load bureau ID dummy
        bureau_id_dummy = self.spark.table(f"{catalog}.cdx_persist_mnf_res_db.mnf_cra_rvw_id_dummy")
        print(f"✓ Loaded mnf_cra_rvw_id_dummy: {bureau_id_dummy.count():,} rows")

        # Adapt
        cardx_adapted = self.adapt_cardx_monthly_data(
            cardx_monthly_df=cardx_monthly,
            cardx_origin_df=cardx_origin,
            bureau_id_dummy_df=bureau_id_dummy
        )

        return cardx_adapted


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    from pyspark.sql import SparkSession

    spark = SparkSession.builder \
        .appName("CardXSchemaAdapterTest") \
        .getOrCreate()

    print("\n" + "="*70)
    print("CARDX SCHEMA ADAPTER - TEST")
    print("="*70)

    # Initialize adapter
    adapter = CardXSchemaAdapter(spark)

    # Option 1: Load and adapt in one step
    try:
        cardx_adapted = adapter.load_and_adapt_cardx(catalog="cdx_mdz_prd")

        print("\n✓ CardX adaptation successful!")
        print(f"  Rows: {cardx_adapted.count():,}")
        print(f"  Customers: {cardx_adapted.select('cust_id').distinct().count():,}")

        # Show sample
        print("\nSample CardX data:")
        cardx_adapted.show(5)

        # Show schema
        print("\nCardX schema:")
        cardx_adapted.printSchema()

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease ensure:")
        print("  1. Running in Databricks")
        print("  2. Access to cdx_mdz_prd catalog")
        print("  3. Tables exist:")
        print("     - cdx_curated_spl_acl_db.spl_acct_mthly")
        print("     - cdx_curated_spl_acl_db.spl_ln_orig")
        print("     - cdx_persist_mnf_res_db.mnf_cra_rvw_id_dummy")

    spark.stop()
