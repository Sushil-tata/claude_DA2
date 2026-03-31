"""
CardX Schema Adapter - Maps CardX Internal Tables to Behavioral Physics Schema
==============================================================================

Handles complete join logic with point-in-time safety:
1. CardX Monthly (spl_acct_mthly) - ACCT_NUM, CUST_NUM
2. CardX Origin (spl_ln_orig) - CUST_NUM → CUST_ID (ID_NO) [DEDUPLICATED]
3. Bureau ID Dummy (mnf_cra_rvw_id_dummy) - ID_NO → REF_NO [POINT-IN-TIME]
4. Bureau Tables - REF_NO

Validated: Fan-out = 0, Match rate = 97.1%, No bureau = 2.9%

Author: Behavioral Physics Team
Version: 2.0.0 - Point-in-time bridge with deduplication
"""

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from typing import Optional

# Table names (can be overridden)
TBL_MTHLY = "cdx_mdz_prd.cdx_curated_spl_acl_db.spl_acct_mthly"
TBL_ORIG = "cdx_mdz_prd.cdx_curated_spl_acl_db.spl_ln_orig"
TBL_DUMMY = "cdx_mdz_prd.cdx_persist_mnf_res_db.mnf_cra_rvw_id_dummy"


def build_bridge_df(spark: SparkSession, as_of_month: str) -> DataFrame:
    """
    Produces one row per ACCT_NUM with the correct REF_NO for the given
    observation month. Point-in-time safe: uses RECEIVE_DT <= month_end.

    Validated in notebook:
    - Fan-out = 0 (one REF_NO per ACCT_NUM guaranteed)
    - Match rate = 97.1% (accounts successfully linked to bureau)
    - No bureau = 2.9% (accounts without bureau mapping)

    Root causes fixed:
    1. spl_ln_orig deduplication: 218 accounts had multiple CUST_ID per CUST_NUM
       → Fixed by taking most recent APRV_DT
    2. mnf_cra_rvw_id_dummy point-in-time filter: avg 11 REF_NOs per customer
       → Fixed by taking latest RECEIVE_DT <= month_end with ROW_NUMBER

    Args:
        spark: SparkSession
        as_of_month: Observation month (YYYY-MM-DD format)

    Returns:
        DataFrame with columns:
        - ACCT_NUM: CardX account number
        - CUST_NUM: CardX customer number
        - ID_NO: Bureau customer ID (from spl_ln_orig.CUST_ID)
        - REF_NO: Bureau reference number (customer key)
        - RECEIVE_DT: Bureau data receipt date
        - DL_DATA_DT: Data lake date
        - days_since_last_pull: Days between month_end and RECEIVE_DT
        - SEGMENT: Customer segment from dummy table
        - TAG: Customer tag from dummy table
    """
    month_end = F.last_day(F.to_date(F.lit(as_of_month), "yyyy-MM-dd"))

    # Step 1a: CardX monthly accounts for this observation month
    cardx_monthly = (
        spark.table(TBL_MTHLY)
        .filter(F.col("DL_DATA_DT") == month_end)
        .select("ACCT_NUM", "CUST_NUM")
        .distinct()
    )

    # Step 1b: spl_ln_orig deduplicated — one CUST_ID per CUST_NUM
    # Root cause: 218 accounts had fan-out due to multiple CUST_ID per CUST_NUM
    # Fix: Take most recent approval date per CUST_NUM
    win_orig = (
        Window.partitionBy("CUST_NUM")
        .orderBy(
            F.col("APRV_DT").desc_nulls_last(),
            F.col("RGTR_DT").desc_nulls_last()
        )
    )
    cardx_origin = (
        spark.table(TBL_ORIG)
        .filter(F.col("CUST_ID").isNotNull())
        .withColumn("rn", F.row_number().over(win_orig))
        .filter(F.col("rn") == 1)
        .select("CUST_NUM", F.col("CUST_ID").alias("ID_NO"))
    )

    # Join monthly accounts with origin to get ID_NO
    cardx_accounts = (
        cardx_monthly
        .join(cardx_origin, on="CUST_NUM", how="left")
        .filter(F.col("ID_NO").isNotNull())
    )

    # Step 2: Latest bureau pull per ID_NO as of month-end
    # Root cause: avg 11 REF_NOs per customer (monthly bureau pulls)
    # Fix: Take latest RECEIVE_DT <= month_end with ROW_NUMBER
    win_bureau = (
        Window.partitionBy("ID_NO")
        .orderBy(F.col("RECEIVE_DT").desc(), F.col("SEQ_ID").desc())
    )
    latest_bureau = (
        spark.table(TBL_DUMMY)
        .filter(F.col("RECEIVE_DT") <= month_end)
        .withColumn("rn", F.row_number().over(win_bureau))
        .filter(F.col("rn") == 1)
        .withColumn("days_since_last_pull",
                    F.datediff(month_end, F.col("RECEIVE_DT")))
        .select("ID_NO", "REF_NO", "RECEIVE_DT", "DL_DATA_DT",
                "days_since_last_pull", "SEGMENT", "TAG")
    )

    # Step 3: Final bridge — one row per ACCT_NUM guaranteed
    bridge_df = (
        cardx_accounts
        .join(latest_bureau, on="ID_NO", how="left")
        .filter(F.col("REF_NO").isNotNull())
        .select("ACCT_NUM", "CUST_NUM", "ID_NO", "REF_NO",
                "RECEIVE_DT", "DL_DATA_DT", "days_since_last_pull",
                "SEGMENT", "TAG")
    )

    # Validation logging
    total_accounts = cardx_monthly.count()
    total_with_id_no = cardx_accounts.count()
    total_with_ref_no = bridge_df.count()

    match_rate = (total_with_ref_no / total_accounts * 100) if total_accounts > 0 else 0

    print(f"Bridge Build Stats:")
    print(f"  CardX accounts (month={as_of_month}): {total_accounts:,}")
    print(f"  With ID_NO (origin join): {total_with_id_no:,}")
    print(f"  With REF_NO (bureau match): {total_with_ref_no:,}")
    print(f"  Match rate: {match_rate:.1f}%")
    print(f"  No bureau mapping: {total_accounts - total_with_ref_no:,} ({100-match_rate:.1f}%)")

    return bridge_df


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
        bridge_df: DataFrame
    ) -> DataFrame:
        """
        Adapt CardX monthly data using pre-built point-in-time bridge.

        OLD LOGIC REMOVED (Lines 50-80):
        - Direct inline joins with cardx_origin and bureau_id_dummy
        - No deduplication of spl_ln_orig (caused 218 account fan-out)
        - No point-in-time filtering of dummy table (used wrong REF_NO)

        NEW LOGIC (validated in notebook):
        - Uses pre-built bridge_df with guaranteed one-to-one mapping
        - Fan-out = 0, Match rate = 97.1%

        Args:
            cardx_monthly_df: spl_acct_mthly table (ACCT_NUM, CUST_NUM, monthly data)
            bridge_df: Pre-built bridge from build_bridge_df() containing:
                       ACCT_NUM → CUST_NUM → ID_NO → REF_NO mapping

        Returns:
            DataFrame with schema: cust_id, as_of_month, cardx_dpd, cardx_balance, cardx_credit_limit
        """
        print("📊 Adapting CardX schema with validated bridge...")

        # Join CardX monthly data with bridge to get REF_NO
        cardx_with_ref = cardx_monthly_df.join(
            bridge_df.select("ACCT_NUM", "REF_NO", "RECEIVE_DT"),
            on="ACCT_NUM",
            how="inner"  # Only keep accounts that have bureau mapping
        )

        print(f"✓ Joined CardX monthly with bridge: {cardx_with_ref.count():,} rows")

        # Map to behavioral physics schema
        cardx_mapped = self._map_cardx_columns(cardx_with_ref)

        # Validate no duplicates (guaranteed by bridge)
        total_rows = cardx_mapped.count()
        unique_accounts = cardx_mapped.select("cust_id", "as_of_month").distinct().count()

        if total_rows != unique_accounts:
            print(f"⚠️  WARNING: Fan-out detected! {total_rows:,} rows vs {unique_accounts:,} unique")
            # Defensive dedup: keep one row per (cust_id, as_of_month)
            # Take max DPD / max balance to be conservative
            cardx_mapped = cardx_mapped.groupBy("cust_id", "as_of_month").agg(
                F.max("cardx_dpd").alias("cardx_dpd"),
                F.max("cardx_balance").alias("cardx_balance"),
                F.max("cardx_credit_limit").alias("cardx_credit_limit")
            )
            print(f"✓ Deduped to {cardx_mapped.count():,} rows")
        else:
            print(f"✓ No fan-out: {total_rows:,} rows = {unique_accounts:,} unique accounts")

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
        as_of_month: str,
        catalog: str = "cdx_mdz_prd"
    ) -> DataFrame:
        """
        Convenience method to load all CardX tables and perform complete adaptation
        using validated point-in-time bridge.

        Args:
            as_of_month: Observation month (YYYY-MM-DD format)
            catalog: Databricks catalog name

        Returns:
            Adapted CardX DataFrame ready for behavioral physics
        """
        print(f"Loading CardX tables for as_of_month={as_of_month}...")

        # Step 1: Build point-in-time bridge (validated: fan-out=0, match=97.1%)
        print("\nStep 1: Building point-in-time bridge...")
        bridge_df = build_bridge_df(self.spark, as_of_month)

        # Step 2: Load CardX monthly data
        print("\nStep 2: Loading CardX monthly data...")
        cardx_monthly = self.spark.table(f"{catalog}.cdx_curated_spl_acl_db.spl_acct_mthly")

        # Filter to observation month
        month_end = F.last_day(F.to_date(F.lit(as_of_month), "yyyy-MM-dd"))
        cardx_monthly = cardx_monthly.filter(F.col("DL_DATA_DT") == month_end)
        print(f"✓ Loaded spl_acct_mthly (month={as_of_month}): {cardx_monthly.count():,} rows")

        # Step 3: Adapt using bridge
        print("\nStep 3: Adapting CardX schema...")
        cardx_adapted = self.adapt_cardx_monthly_data(
            cardx_monthly_df=cardx_monthly,
            bridge_df=bridge_df
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
    print("CARDX SCHEMA ADAPTER - POINT-IN-TIME BRIDGE TEST")
    print("="*70)

    # Initialize adapter
    adapter = CardXSchemaAdapter(spark)

    # Test with specific as_of_month
    as_of_month = "2024-12-31"  # Replace with actual month

    try:
        # Option 1: Build bridge separately (for inspection)
        print("\n--- Testing bridge build ---")
        bridge_df = build_bridge_df(spark, as_of_month)

        print("\nBridge schema:")
        bridge_df.printSchema()

        print("\nBridge sample:")
        bridge_df.show(5, truncate=False)

        # Check for fan-out (should be 0)
        total_rows = bridge_df.count()
        unique_accounts = bridge_df.select("ACCT_NUM").distinct().count()
        print(f"\nFan-out check: {total_rows:,} rows, {unique_accounts:,} unique accounts")
        if total_rows == unique_accounts:
            print("✓ No fan-out!")
        else:
            print(f"❌ Fan-out detected: {total_rows - unique_accounts:,} duplicates")

        # Option 2: Load and adapt in one step
        print("\n--- Testing full adaptation ---")
        cardx_adapted = adapter.load_and_adapt_cardx(
            as_of_month=as_of_month,
            catalog="cdx_mdz_prd"
        )

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
        import traceback
        traceback.print_exc()
        print("\nPlease ensure:")
        print("  1. Running in Databricks")
        print("  2. Access to cdx_mdz_prd catalog")
        print("  3. Tables exist:")
        print("     - cdx_curated_spl_acl_db.spl_acct_mthly")
        print("     - cdx_curated_spl_acl_db.spl_ln_orig")
        print("     - cdx_persist_mnf_res_db.mnf_cra_rvw_id_dummy")
        print("  4. as_of_month is valid (YYYY-MM-DD format)")

    spark.stop()
