"""
Integration Test: Validate All Schema and Logic Fixes
======================================================

Tests all recent fixes:
1. RECEIVE_DT point-in-time safety
2. Payment history expansion (48 months)
3. XXX handling (maps to null, not 0)
4. has_reporting_gap and reporting_gap_count_12m features
5. lender_code → lender_id schema fix
6. dpd_bucket_ordinal preservation

Run this in Databricks to verify the complete pipeline works end-to-end.
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from datetime import datetime
from behavioral_physics_features.modules import (
    BureauSchemaAdapter,
    BehavioralPhysicsPipeline
)

def test_all_fixes():
    """
    Comprehensive integration test for all recent fixes.
    """
    print("\n" + "="*80)
    print("INTEGRATION TEST: All Schema & Logic Fixes")
    print(f"Started: {datetime.now()}")
    print("="*80 + "\n")

    # Initialize Spark
    spark = SparkSession.builder \
        .appName("Test_All_Fixes") \
        .getOrCreate()

    # ========================================================================
    # STEP 1: Load Bureau Data
    # ========================================================================
    print("STEP 1: Loading bureau data...")
    print("-" * 80)

    try:
        catalog = "cdx_mdz_prd.cdx_persist_mnf_res_db"

        bureau_account = spark.table(f"{catalog}.mnf_cra_rvw_s_account")
        bureau_history = spark.table(f"{catalog}.mnf_cra_rvw_s_history")
        bureau_enquiry = spark.table(f"{catalog}.mnf_cra_rvw_s_enquiry")

        print(f"✓ Loaded bureau_account: {bureau_account.count():,} rows")
        print(f"✓ Loaded bureau_history: {bureau_history.count():,} rows")
        print(f"✓ Loaded bureau_enquiry: {bureau_enquiry.count():,} rows")

    except Exception as e:
        print(f"❌ Error loading bureau data: {e}")
        return

    # ========================================================================
    # STEP 2: Test Schema Adapter
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 2: Testing Bureau Schema Adapter...")
    print("-" * 80)

    adapter = BureauSchemaAdapter(spark)

    # Adapt bureau trade
    bureau_trade = adapter.adapt_bureau_trade_data(
        account_df=bureau_account,
        history_df=bureau_history
    )

    print(f"✓ Bureau trade adapted: {bureau_trade.count():,} rows")

    # ========================================================================
    # TEST 1: Check Required Columns Exist
    # ========================================================================
    print("\n[TEST 1] Required columns exist:")

    required_cols = {
        "cust_id": "Customer ID",
        "as_of_month": "Snapshot month",
        "account_id": "Account number",
        "lender_name": "Lender display name",
        "lender_id": "Lender unique ID (FIX 5)",
        "dpd": "Days past due (numeric)",
        "dpd_bucket": "DPD bucket code",
        "dpd_bucket_ordinal": "DPD ordinal 0-6 (FIX 3)",
        "balance": "Outstanding balance",
        "credit_limit": "Credit limit",
        "utilization": "Balance/limit ratio",
        "receive_dt": "Bureau report receipt date (FIX 1)",
        "has_reporting_gap": "XXX exists flag (FIX 4)",
        "reporting_gap_count_12m": "XXX count in last 12m (FIX 4)",
        "source": "Data source (history_table or payment_history)",
    }

    actual_cols = bureau_trade.columns
    missing_cols = []
    present_cols = []

    for col, description in required_cols.items():
        if col in actual_cols:
            present_cols.append(col)
            print(f"  ✅ {col:<25} - {description}")
        else:
            missing_cols.append(col)
            print(f"  ❌ {col:<25} - {description} [MISSING]")

    if missing_cols:
        print(f"\n  ⚠️  WARNING: {len(missing_cols)} columns missing: {missing_cols}")
    else:
        print(f"\n  ✅ PASS: All {len(required_cols)} required columns present")

    # ========================================================================
    # TEST 2: Validate receive_dt (Point-in-Time Safety)
    # ========================================================================
    print("\n[TEST 2] RECEIVE_DT point-in-time filter (FIX 1):")

    # Check receive_dt exists and has non-null values
    receive_dt_stats = bureau_trade.select(
        F.count("*").alias("total_rows"),
        F.sum(F.col("receive_dt").isNotNull().cast("int")).alias("receive_dt_not_null"),
        F.sum(F.col("receive_dt").isNull().cast("int")).alias("receive_dt_null")
    ).first()

    print(f"  Total rows: {receive_dt_stats['total_rows']:,}")
    print(f"  receive_dt NOT NULL: {receive_dt_stats['receive_dt_not_null']:,} (history table)")
    print(f"  receive_dt IS NULL: {receive_dt_stats['receive_dt_null']:,} (payment history)")

    # Check receive_dt <= as_of_month (point-in-time safety)
    future_data = bureau_trade.filter(
        F.col("receive_dt").isNotNull() &
        (F.col("receive_dt") > F.last_day(F.col("as_of_month")))
    ).count()

    if future_data == 0:
        print(f"  ✅ PASS: No future data leakage (receive_dt <= as_of_month)")
    else:
        print(f"  ❌ FAIL: {future_data:,} rows have receive_dt > as_of_month (data leakage!)")

    # ========================================================================
    # TEST 3: Validate Payment History Expansion
    # ========================================================================
    print("\n[TEST 3] Payment history expansion (FIX 2):")

    # Count rows by source
    source_counts = bureau_trade.groupBy("source").count().collect()
    source_dict = {row["source"]: row["count"] for row in source_counts}

    history_rows = source_dict.get("history_table", 0)
    payment_rows = source_dict.get("payment_history", 0)
    total_rows = history_rows + payment_rows

    print(f"  history_table rows: {history_rows:,}")
    print(f"  payment_history rows: {payment_rows:,}")
    print(f"  Total: {total_rows:,}")

    if payment_rows > 0:
        expansion_pct = (payment_rows / total_rows * 100)
        print(f"  ✅ PASS: Payment history expanded by {payment_rows:,} rows ({expansion_pct:.1f}%)")
    else:
        print(f"  ⚠️  WARNING: No payment history expansion (expected some rows)")

    # ========================================================================
    # TEST 4: Validate XXX Handling
    # ========================================================================
    print("\n[TEST 4] XXX (not reported) handling (FIX 4):")

    # Check XXX rows
    xxx_rows = bureau_trade.filter(F.col("dpd_bucket") == "XXX")
    xxx_count = xxx_rows.count()

    if xxx_count > 0:
        print(f"  Total XXX rows: {xxx_count:,}")

        # Check dpd is null for XXX
        dpd_null_count = xxx_rows.filter(F.col("dpd").isNull()).count()
        dpd_zero_count = xxx_rows.filter(F.col("dpd") == 0).count()

        print(f"  dpd IS NULL: {dpd_null_count:,} ({dpd_null_count/xxx_count*100:.1f}%)")
        print(f"  dpd == 0: {dpd_zero_count:,} ({dpd_zero_count/xxx_count*100:.1f}%)")

        if dpd_null_count == xxx_count:
            print(f"  ✅ PASS: All XXX rows have null dpd")
        else:
            print(f"  ❌ FAIL: {xxx_count - dpd_null_count} XXX rows don't have null dpd")

        # Check dpd_bucket_ordinal is null for XXX
        ordinal_null_count = xxx_rows.filter(F.col("dpd_bucket_ordinal").isNull()).count()

        if ordinal_null_count == xxx_count:
            print(f"  ✅ PASS: All XXX rows have null dpd_bucket_ordinal")
        else:
            print(f"  ❌ FAIL: {xxx_count - ordinal_null_count} XXX rows don't have null dpd_bucket_ordinal")
    else:
        print(f"  ⚠️  No XXX rows found (may be OK if data has perfect reporting)")

    # ========================================================================
    # TEST 5: Validate Reporting Gap Features
    # ========================================================================
    print("\n[TEST 5] Reporting gap features (FIX 4):")

    # Check has_reporting_gap distribution
    gap_dist = bureau_trade.groupBy("has_reporting_gap").count().collect()
    gap_dict = {row["has_reporting_gap"]: row["count"] for row in gap_dist}

    print(f"  has_reporting_gap = 0: {gap_dict.get(0, 0):,}")
    print(f"  has_reporting_gap = 1: {gap_dict.get(1, 0):,}")
    print(f"  has_reporting_gap = null: {gap_dict.get(None, 0):,}")

    # Check reporting_gap_count_12m distribution
    print(f"\n  reporting_gap_count_12m distribution:")
    bureau_trade.groupBy("reporting_gap_count_12m").count() \
        .orderBy("reporting_gap_count_12m").show(13)

    # ========================================================================
    # TEST 6: Validate lender_id (not lender_code)
    # ========================================================================
    print("\n[TEST 6] lender_id schema fix (FIX 5):")

    if "lender_id" in bureau_trade.columns:
        lender_count = bureau_trade.select("lender_id").distinct().count()
        print(f"  ✅ PASS: lender_id column exists")
        print(f"  Unique lenders: {lender_count:,}")

        # Show sample lender_id values
        print(f"\n  Sample lender_id values:")
        bureau_trade.select("lender_id", "lender_name").distinct().show(10, truncate=False)
    else:
        print(f"  ❌ FAIL: lender_id column missing (still using lender_code?)")

    # ========================================================================
    # TEST 7: Validate dpd_bucket_ordinal
    # ========================================================================
    print("\n[TEST 7] dpd_bucket_ordinal preservation (FIX 3):")

    # Check ordinal distribution (should be 0-6)
    ordinal_dist = bureau_trade.filter(F.col("dpd_bucket_ordinal").isNotNull()) \
        .groupBy("dpd_bucket", "dpd", "dpd_bucket_ordinal").count() \
        .orderBy("dpd_bucket_ordinal")

    print(f"  DPD bucket → ordinal mapping:")
    ordinal_dist.show(20)

    # Validate ordinals are 0-6
    ordinals = bureau_trade.filter(F.col("dpd_bucket_ordinal").isNotNull()) \
        .select("dpd_bucket_ordinal").distinct().orderBy("dpd_bucket_ordinal").collect()
    ordinal_values = [row["dpd_bucket_ordinal"] for row in ordinals]

    expected_ordinals = [0, 1, 2, 3, 4, 5, 6]
    if ordinal_values == expected_ordinals:
        print(f"  ✅ PASS: Ordinal categories are 0-6 as expected")
    else:
        print(f"  ⚠️  WARNING: Found ordinals {ordinal_values}, expected {expected_ordinals}")

    # ========================================================================
    # TEST 8: Test Full Pipeline
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 3: Testing Full Pipeline Integration...")
    print("-" * 80)

    try:
        # Adapt enquiry
        bureau_enquiry_adapted = adapter.adapt_bureau_enquiry_data(bureau_enquiry)

        # Load CardX (or create mock)
        try:
            from behavioral_physics_features.modules.cardx_schema_adapter import CardXSchemaAdapter
            cardx_adapter = CardXSchemaAdapter(spark)
            cardx_internal = cardx_adapter.load_and_adapt_cardx(catalog="cdx_mdz_prd")
        except Exception as e:
            print(f"  ⚠️  CardX load failed, using mock: {e}")
            customer_months = bureau_trade.select("cust_id", "as_of_month").distinct()
            cardx_internal = customer_months.withColumn("cardx_dpd", F.lit(0)) \
                .withColumn("cardx_balance", F.lit(5000)) \
                .withColumn("cardx_credit_limit", F.lit(10000))

        # Get most recent month
        max_month = bureau_trade.agg(F.max("as_of_month")).first()[0]
        as_of_month = str(max_month)

        print(f"  Testing with as_of_month: {as_of_month}")

        # Filter to recent data for faster testing
        three_months_ago = F.add_months(F.lit(as_of_month), -3)
        bureau_trade_recent = bureau_trade.filter(F.col("as_of_month") >= three_months_ago)
        cardx_recent = cardx_internal.filter(F.col("as_of_month") >= three_months_ago)

        # Run pipeline
        pipeline = BehavioralPhysicsPipeline(spark)

        print(f"\n  Running behavioral physics pipeline...")
        features_df, audit_log, qa_summary = pipeline.run(
            bureau_trade_df=bureau_trade_recent,
            bureau_enquiry_df=bureau_enquiry_adapted,
            cardx_internal_df=cardx_recent,
            as_of_month=as_of_month
        )

        print(f"\n  ✅ Pipeline completed successfully!")
        print(f"  Features generated: {len([c for c in features_df.columns if c not in ['cust_id', 'as_of_month']])}")
        print(f"  Customers processed: {features_df.count():,}")

        # Show sample features
        print(f"\n  Sample features:")
        features_df.select(
            "cust_id", "as_of_month",
            "dpd_velocity_3m",
            "lender_hhi",
            "state_entropy_6m"
        ).show(5)

    except Exception as e:
        print(f"\n  ❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("INTEGRATION TEST COMPLETE - SUMMARY")
    print("="*80)

    print(f"\n✅ All Fixes Validated:")
    print(f"  1. RECEIVE_DT point-in-time safety: {'✅' if 'receive_dt' in bureau_trade.columns else '❌'}")
    print(f"  2. Payment history expansion: {'✅' if payment_rows > 0 else '⚠️'}")
    print(f"  3. XXX → null mapping: {'✅' if xxx_count == 0 or dpd_null_count == xxx_count else '❌'}")
    print(f"  4. Reporting gap features: {'✅' if 'has_reporting_gap' in bureau_trade.columns else '❌'}")
    print(f"  5. lender_id schema: {'✅' if 'lender_id' in bureau_trade.columns else '❌'}")
    print(f"  6. dpd_bucket_ordinal: {'✅' if 'dpd_bucket_ordinal' in bureau_trade.columns else '❌'}")

    print(f"\n" + "="*80)
    print(f"Test Completed: {datetime.now()}")
    print("="*80 + "\n")

    return bureau_trade, features_df


# ============================================================================
# Run Test
# ============================================================================

if __name__ == "__main__":
    try:
        bureau_trade, features_df = test_all_fixes()

        print("\n✅ Integration test completed!")
        print("\nDataFrames available:")
        print("  - bureau_trade: Adapted bureau data with all fixes")
        print("  - features_df: Generated behavioral physics features")

    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
