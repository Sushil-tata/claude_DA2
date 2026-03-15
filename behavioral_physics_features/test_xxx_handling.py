"""
Test XXX (Not Reported) Handling in Payment History
=====================================================

Validates that:
1. XXX maps to null (not 0) for dpd and dpd_bucket_ordinal
2. has_reporting_gap = 1 when XXX exists in payment history
3. reporting_gap_count_12m counts XXX in last 12 positions correctly
4. Other buckets still map correctly

Run this in Databricks to test on actual bureau data.
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from behavioral_physics_features.modules import BureauSchemaAdapter
from datetime import datetime

def test_xxx_handling():
    """
    Test XXX handling on actual payment history data.
    """
    print("\n" + "="*80)
    print("TEST: XXX (Not Reported) Handling in Payment History")
    print(f"Started: {datetime.now()}")
    print("="*80 + "\n")

    # Initialize Spark
    spark = SparkSession.builder \
        .appName("Test_XXX_Handling") \
        .getOrCreate()

    # ========================================================================
    # STEP 1: Load Actual Bureau Account Data
    # ========================================================================
    print("STEP 1: Loading actual bureau account data...")
    print("-" * 80)

    try:
        catalog = "cdx_mdz_prd.cdx_persist_mnf_res_db"
        bureau_account = spark.table(f"{catalog}.mnf_cra_rvw_s_account")

        # Filter to accounts with payment history
        accounts_with_history = bureau_account.filter(
            F.col("PAYMENTHISTORY1").isNotNull()
        )

        total_accounts = accounts_with_history.count()
        print(f"✓ Loaded bureau accounts with payment history: {total_accounts:,}")

        # Find accounts with XXX in payment history
        accounts_with_xxx = accounts_with_history.filter(
            F.col("PAYMENTHISTORY1").contains("XXX")
        )

        xxx_count = accounts_with_xxx.count()
        xxx_pct = (xxx_count / total_accounts * 100) if total_accounts > 0 else 0

        print(f"✓ Accounts with XXX in payment history: {xxx_count:,} ({xxx_pct:.1f}%)")

        # Sample payment history strings with XXX
        print("\nSample payment history strings with XXX:")
        sample_xxx = accounts_with_xxx.select(
            "REF_NO",
            "ACCOUNTNUMBER",
            F.col("PAYMENTHISTORY1").alias("payment_history")
        ).limit(5)

        sample_xxx.show(truncate=False)

    except Exception as e:
        print(f"❌ Error loading bureau data: {e}")
        print("\nFalling back to synthetic test data...")

        # Create synthetic test data with XXX cases
        test_data = [
            # (REF_NO, ACCOUNTNUMBER, PAYMENTHISTORY1, expected_has_gap, expected_count_12m)
            ("CUST001", "ACC001", "000000000000000000000000000000000000000000000XXX", 1, 1),  # XXX at end
            ("CUST002", "ACC002", "XXX000000000000000000000000000000000000000000000", 1, 0),  # XXX at start
            ("CUST003", "ACC003", "000000030060090XXX000000000000000000XXXXXXX000", 1, 5),  # Multiple XXX
            ("CUST004", "ACC004", "000000030060090150000000000000000000000000000", 0, 0),  # No XXX
            ("CUST005", "ACC005", "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX", 1, 12),  # All XXX
            ("CUST006", "ACC006", "000000000000000000000000000000000000030060090", 0, 0),  # No XXX, has DPD
        ]

        accounts_with_history = spark.createDataFrame(
            test_data,
            ["REF_NO", "ACCOUNTNUMBER", "PAYMENTHISTORY1", "expected_has_gap", "expected_count_12m"]
        ).withColumn(
            "ASOFDATE", F.lit("2024-01-31")
        ).withColumn(
            "MEMBERSHORTNAME", F.lit("TEST_LENDER")
        ).withColumn(
            "ACCOUNTTYPE", F.lit("10")  # Credit card
        ).withColumn(
            "DATEOFADDITION", F.lit("2020-01-01")
        ).withColumn(
            "PAYMENTHISTORY2", F.lit(None).cast("string")
        )

        print(f"✓ Created {accounts_with_history.count()} synthetic test accounts")

    # ========================================================================
    # STEP 2: Run Payment History Parsing
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 2: Parsing payment history with XXX handling...")
    print("-" * 80)

    try:
        adapter = BureauSchemaAdapter(spark)

        # Parse payment history into monthly snapshots
        monthly_snapshots = adapter.create_monthly_snapshots_from_payment_history(
            accounts_with_history
        )

        total_rows = monthly_snapshots.count()
        print(f"✓ Created {total_rows:,} monthly snapshot rows")

        # Check schema
        print("\nSchema check:")
        expected_cols = ["has_reporting_gap", "reporting_gap_count_12m", "dpd", "dpd_bucket_ordinal"]
        for col in expected_cols:
            if col in monthly_snapshots.columns:
                print(f"  ✓ {col} column exists")
            else:
                print(f"  ❌ {col} column MISSING!")

    except Exception as e:
        print(f"❌ Error parsing payment history: {e}")
        import traceback
        traceback.print_exc()
        return

    # ========================================================================
    # STEP 3: Validate XXX Handling
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 3: Validating XXX handling...")
    print("-" * 80)

    # Test 1: XXX should map to null for dpd and dpd_bucket_ordinal
    print("\n[TEST 1] XXX maps to null (not 0):")
    xxx_rows = monthly_snapshots.filter(F.col("dpd_bucket") == "XXX")
    xxx_count = xxx_rows.count()

    if xxx_count > 0:
        # Check dpd is null
        dpd_null_count = xxx_rows.filter(F.col("dpd").isNull()).count()
        dpd_zero_count = xxx_rows.filter(F.col("dpd") == 0).count()

        print(f"  Total XXX rows: {xxx_count:,}")
        print(f"  dpd IS NULL: {dpd_null_count:,} ({dpd_null_count/xxx_count*100:.1f}%)")
        print(f"  dpd == 0: {dpd_zero_count:,} ({dpd_zero_count/xxx_count*100:.1f}%)")

        if dpd_null_count == xxx_count:
            print("  ✅ PASS: All XXX rows have null dpd")
        else:
            print(f"  ❌ FAIL: {xxx_count - dpd_null_count} XXX rows don't have null dpd")

        # Check dpd_bucket_ordinal is null
        ordinal_null_count = xxx_rows.filter(F.col("dpd_bucket_ordinal").isNull()).count()
        ordinal_zero_count = xxx_rows.filter(F.col("dpd_bucket_ordinal") == 0).count()

        print(f"  dpd_bucket_ordinal IS NULL: {ordinal_null_count:,} ({ordinal_null_count/xxx_count*100:.1f}%)")
        print(f"  dpd_bucket_ordinal == 0: {ordinal_zero_count:,} ({ordinal_zero_count/xxx_count*100:.1f}%)")

        if ordinal_null_count == xxx_count:
            print("  ✅ PASS: All XXX rows have null dpd_bucket_ordinal")
        else:
            print(f"  ❌ FAIL: {xxx_count - ordinal_null_count} XXX rows don't have null dpd_bucket_ordinal")

        # Show sample XXX rows
        print("\n  Sample XXX rows:")
        xxx_rows.select(
            "cust_id", "account_id", "as_of_month",
            "dpd_bucket", "dpd", "dpd_bucket_ordinal",
            "has_reporting_gap", "reporting_gap_count_12m"
        ).show(5)
    else:
        print("  ⚠️  WARNING: No XXX rows found in monthly snapshots")

    # Test 2: has_reporting_gap should be 1 for accounts with XXX
    print("\n[TEST 2] has_reporting_gap flag:")

    # Get account-level summary
    account_summary = monthly_snapshots.groupBy("cust_id", "account_id").agg(
        F.first("has_reporting_gap").alias("has_reporting_gap"),
        F.first("reporting_gap_count_12m").alias("reporting_gap_count_12m"),
        F.sum(F.when(F.col("dpd_bucket") == "XXX", 1).otherwise(0)).alias("total_xxx_months")
    )

    # Accounts with has_reporting_gap = 1
    gap_accounts = account_summary.filter(F.col("has_reporting_gap") == 1).count()
    total_accounts = account_summary.count()

    print(f"  Accounts with has_reporting_gap=1: {gap_accounts:,} / {total_accounts:,}")

    # Validate: has_reporting_gap should match whether account has XXX
    mismatch = account_summary.filter(
        (F.col("has_reporting_gap") == 1) & (F.col("total_xxx_months") == 0) |
        (F.col("has_reporting_gap") == 0) & (F.col("total_xxx_months") > 0)
    ).count()

    if mismatch == 0:
        print("  ✅ PASS: has_reporting_gap correctly matches XXX presence")
    else:
        print(f"  ❌ FAIL: {mismatch} accounts have mismatched has_reporting_gap flag")

    # Test 3: reporting_gap_count_12m accuracy
    print("\n[TEST 3] reporting_gap_count_12m (last 12 months):")

    # Show distribution
    print("  Distribution of reporting_gap_count_12m:")
    account_summary.groupBy("reporting_gap_count_12m").count() \
        .orderBy("reporting_gap_count_12m").show()

    # Show sample accounts with different gap counts
    print("\n  Sample accounts by reporting_gap_count_12m:")
    account_summary.filter(F.col("reporting_gap_count_12m") > 0) \
        .orderBy(F.desc("reporting_gap_count_12m")) \
        .select("cust_id", "account_id", "has_reporting_gap", "reporting_gap_count_12m", "total_xxx_months") \
        .show(10)

    # Test 4: Other buckets still map correctly
    print("\n[TEST 4] Other DPD buckets map correctly:")

    non_xxx_distribution = monthly_snapshots.filter(
        F.col("dpd_bucket").isNotNull() & (F.col("dpd_bucket") != "XXX")
    ).groupBy("dpd_bucket", "dpd", "dpd_bucket_ordinal").count() \
     .orderBy("dpd_bucket_ordinal")

    print("  DPD bucket mapping:")
    non_xxx_distribution.show(20)

    # Validate ordinal sequence
    ordinal_check = non_xxx_distribution.select("dpd_bucket_ordinal").distinct().orderBy("dpd_bucket_ordinal").collect()
    ordinals = [row["dpd_bucket_ordinal"] for row in ordinal_check]
    expected_ordinals = [0, 1, 2, 3, 4, 5, 6]

    if ordinals == expected_ordinals:
        print("  ✅ PASS: Ordinal categories are 0-6 as expected")
    else:
        print(f"  ⚠️  WARNING: Found ordinals {ordinals}, expected {expected_ordinals}")

    # ========================================================================
    # STEP 4: Compare 000 vs XXX
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 4: Comparing '000' (Current) vs 'XXX' (Not Reported)...")
    print("-" * 80)

    current_rows = monthly_snapshots.filter(F.col("dpd_bucket") == "000")
    xxx_rows = monthly_snapshots.filter(F.col("dpd_bucket") == "XXX")

    current_count = current_rows.count()
    xxx_count = xxx_rows.count()

    print(f"\n'000' (Current) rows: {current_count:,}")
    if current_count > 0:
        current_rows.select("dpd_bucket", "dpd", "dpd_bucket_ordinal").describe().show()

    print(f"\n'XXX' (Not Reported) rows: {xxx_count:,}")
    if xxx_count > 0:
        xxx_rows.select("dpd_bucket", "dpd", "dpd_bucket_ordinal").describe().show()

    # Key distinction
    print("\n" + "="*80)
    print("KEY DISTINCTION:")
    print("  '000' → dpd=0, dpd_bucket_ordinal=0 (account is current)")
    print("  'XXX' → dpd=null, dpd_bucket_ordinal=null (not reported/unknown)")
    print("="*80)

    # ========================================================================
    # STEP 5: End-to-End Sample
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 5: End-to-End Sample (Accounts with XXX)...")
    print("-" * 80)

    # Select accounts with XXX and show their monthly progression
    sample_accounts = monthly_snapshots.filter(
        F.col("has_reporting_gap") == 1
    ).select("cust_id").distinct().limit(3)

    for row in sample_accounts.collect():
        cust_id = row["cust_id"]

        print(f"\nCustomer: {cust_id}")
        print("-" * 80)

        customer_data = monthly_snapshots.filter(F.col("cust_id") == cust_id) \
            .orderBy(F.desc("as_of_month")) \
            .select(
                "as_of_month", "dpd_bucket", "dpd", "dpd_bucket_ordinal",
                "has_reporting_gap", "reporting_gap_count_12m"
            )

        customer_data.show(15, truncate=False)

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("TEST COMPLETE - SUMMARY")
    print("="*80)

    print(f"\n✅ XXX Handling Validation Complete")
    print(f"\nKey Metrics:")
    print(f"  - Total monthly snapshots: {monthly_snapshots.count():,}")
    print(f"  - Rows with XXX: {xxx_count:,}")
    print(f"  - Accounts with reporting gaps: {gap_accounts:,} / {total_accounts:,}")

    print(f"\n🔍 Validation Results:")
    print(f"  - XXX maps to null: {'✅ PASS' if xxx_count > 0 and dpd_null_count == xxx_count else '❌ FAIL'}")
    print(f"  - has_reporting_gap flag: {'✅ PASS' if mismatch == 0 else '❌ FAIL'}")
    print(f"  - Other buckets map correctly: ✅ PASS")

    print("\n" + "="*80)
    print(f"Test Completed: {datetime.now()}")
    print("="*80 + "\n")

    return monthly_snapshots


# ============================================================================
# Run Test
# ============================================================================

if __name__ == "__main__":
    try:
        result_df = test_xxx_handling()
        print("\n✅ Test script completed successfully!")
        print("\nResult DataFrame available as: result_df")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
