"""
Integration Test: NCB Feature Factory V2 (Single-File Architecture)
====================================================================

Tests the new consolidated single-file architecture that replaces the
old multi-file design (11 files → 1 file).

This eliminates:
- Cross-file column contract violations
- Duplicate column creation bugs
- Import cache fragility
- Parameter mismatch errors

Run this in Databricks to verify the new architecture works end-to-end.

Author: Behavioral Physics Team
Version: 2.0.0
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from datetime import datetime, timedelta

# Import the new single-file factory
import sys
sys.path.insert(0, "/Workspace/Users/sushil@cardx.co.th/claude_DA2")

from behavioral_physics_features.ncb_feature_factory_v2 import (
    run_ncb_feature_factory_v2,
    CFG
)


def create_test_bridge_df(spark: SparkSession) -> "DataFrame":
    """
    Create test bridge DataFrame.

    Bridge must have: ref_no, id_no, receive_dt, dl_data_dt, as_of_month
    """
    print("\n" + "="*80)
    print("Creating test bridge DataFrame...")
    print("="*80)

    # Load id_dummy to get ref_no values
    catalog = "cdx_mdz_prd.cdx_persist_mnf_res_db"
    id_dummy = spark.table(f"{catalog}.mnf_cra_rvw_id_dummy")

    # Take sample of ref_no values (e.g., 1000 customers)
    sample_refs = id_dummy.select("REF_NO", "ID_NO") \
        .limit(1000) \
        .withColumnRenamed("REF_NO", "ref_no") \
        .withColumnRenamed("ID_NO", "id_no")

    # Add bridge columns
    # receive_dt: point-in-time anchor (e.g., current date)
    # as_of_month: evaluation month (e.g., last month)
    current_date = datetime.now()
    as_of_month = (current_date - timedelta(days=30)).replace(day=1)

    bridge_df = sample_refs.withColumn(
        "receive_dt",
        F.lit(current_date.strftime("%Y-%m-%d")).cast("date")
    ).withColumn(
        "dl_data_dt",
        F.lit(current_date.strftime("%Y-%m-%d")).cast("date")
    ).withColumn(
        "as_of_month",
        F.lit(as_of_month.strftime("%Y-%m-%d")).cast("date")
    )

    print(f"✓ Bridge created: {bridge_df.count():,} customers")
    print(f"  Columns: {', '.join(bridge_df.columns)}")

    return bridge_df


def test_ncb_factory_v2():
    """
    Test the new NCB Feature Factory V2.
    """
    print("\n" + "="*80)
    print("INTEGRATION TEST: NCB Feature Factory V2 (Single-File Architecture)")
    print(f"Started: {datetime.now()}")
    print("="*80)

    # Initialize Spark
    spark = SparkSession.builder \
        .appName("Test_NCB_Factory_V2") \
        .getOrCreate()

    # ========================================================================
    # STEP 1: Create Bridge DataFrame
    # ========================================================================
    print("\n[STEP 1] Creating bridge DataFrame...")

    try:
        bridge_df = create_test_bridge_df(spark)
    except Exception as e:
        print(f"❌ Error creating bridge: {e}")
        import traceback
        traceback.print_exc()
        return

    # ========================================================================
    # STEP 2: Run Feature Factory
    # ========================================================================
    print("\n[STEP 2] Running NCB Feature Factory V2...")
    print("="*80)

    try:
        schema_name = "cdx_mdz_prd.cdx_persist_mnf_res_db"

        features_df = run_ncb_feature_factory_v2(
            spark=spark,
            schema_name=schema_name,
            bridge_df=bridge_df
        )

        print(f"\n✓ Feature factory completed successfully!")

    except Exception as e:
        print(f"\n❌ Error running feature factory: {e}")
        import traceback
        traceback.print_exc()
        return

    # ========================================================================
    # STEP 3: Validate Output
    # ========================================================================
    print("\n[STEP 3] Validating output...")
    print("="*80)

    # Check row count
    feature_count = features_df.count()
    print(f"\n✓ Output rows: {feature_count:,}")

    # Check column count
    total_cols = len(features_df.columns)
    feature_cols = total_cols - 2  # Subtract ref_no and as_of_month
    print(f"✓ Total columns: {total_cols} ({feature_cols} features + ref_no + as_of_month)")

    # Display columns
    print(f"\nColumn list (first 20):")
    for i, col in enumerate(features_df.columns[:20], 1):
        print(f"  {i:2d}. {col}")

    if total_cols > 20:
        print(f"  ... and {total_cols - 20} more columns")

    # ========================================================================
    # STEP 4: Verify Key Columns
    # ========================================================================
    print("\n[STEP 4] Verifying key columns...")
    print("="*80)

    required_cols = {
        CFG["ref_col"]: "Primary key (ref_no)",
        "as_of_month": "Evaluation month",
        "bureau_max_dpd": "Max DPD across accounts",
        "dpd_bucket_ordinal": "DPD ordinal (0-4)",
        "credit_inertia_score": "Physics: inertia",
        "credit_momentum_3m": "Physics: momentum",
        "state_entropy_48m": "Physics: entropy",
        "lender_hhi": "Lender concentration",
        "enquiry_count_12m": "Enquiry velocity"
    }

    missing_cols = []
    present_cols = []

    for col, description in required_cols.items():
        if col in features_df.columns:
            present_cols.append(f"  ✓ {col:<30} : {description}")
        else:
            missing_cols.append(f"  ❌ {col:<30} : {description} - MISSING!")

    print("\nKey columns present:")
    for line in present_cols:
        print(line)

    if missing_cols:
        print("\n⚠️  Missing columns:")
        for line in missing_cols:
            print(line)

    # ========================================================================
    # STEP 5: Verify No Banned Columns
    # ========================================================================
    print("\n[STEP 5] Verifying no banned columns...")
    print("="*80)

    banned_patterns = [
        "cust_id",
        "lender_id",
        "lender_code",
        "member_code",
        "lender_type_raw",
        "check_d",
        "transition_type"
    ]

    banned_found = []
    for col in features_df.columns:
        col_lower = col.lower()
        for pattern in banned_patterns:
            if pattern in col_lower:
                banned_found.append(f"  ❌ {col} - contains banned pattern '{pattern}'")

    if banned_found:
        print("\n⚠️  Banned columns found:")
        for line in banned_found:
            print(line)
    else:
        print("\n✓ No banned columns found - clean architecture!")

    # ========================================================================
    # STEP 6: Sample Data Preview
    # ========================================================================
    print("\n[STEP 6] Sample data preview...")
    print("="*80)

    print("\nFirst 5 rows (key columns):")
    features_df.select(
        CFG["ref_col"],
        "as_of_month",
        "bureau_max_dpd",
        "dpd_bucket_ordinal",
        "credit_inertia_score",
        "credit_momentum_3m"
    ).show(5, truncate=False)

    # ========================================================================
    # STEP 7: Null Check
    # ========================================================================
    print("\n[STEP 7] Null rate analysis (top 10 features with nulls)...")
    print("="*80)

    null_counts = []
    for col in features_df.columns:
        if col not in [CFG["ref_col"], "as_of_month"]:
            null_count = features_df.filter(F.col(col).isNull()).count()
            null_pct = (null_count / feature_count * 100) if feature_count > 0 else 0
            if null_pct > 0:
                null_counts.append((col, null_count, null_pct))

    null_counts.sort(key=lambda x: x[2], reverse=True)

    if null_counts:
        print(f"\n{'Feature':<40} {'Nulls':>10} {'%':>8}")
        print("-" * 60)
        for col, count, pct in null_counts[:10]:
            print(f"{col:<40} {count:>10,} {pct:>7.1f}%")

        if len(null_counts) > 10:
            print(f"\n... and {len(null_counts) - 10} more features with nulls")
    else:
        print("\n✓ No null values found in any feature!")

    # ========================================================================
    # STEP 8: Summary
    # ========================================================================
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    print(f"\n✓ Total features: {feature_cols}")
    print(f"✓ Target: ~377 features")
    print(f"✓ Actual: {feature_cols} features")

    if feature_cols >= 350:
        print(f"✅ PASS: Feature count within expected range")
    else:
        print(f"⚠️  WARNING: Feature count below expected (target: 377, actual: {feature_cols})")

    print(f"\n✓ Output rows: {feature_count:,}")
    print(f"✓ Missing columns: {len(missing_cols)}")
    print(f"✓ Banned columns found: {len(banned_found)}")

    if len(missing_cols) == 0 and len(banned_found) == 0:
        print(f"\n{'='*80}")
        print("✅ ALL TESTS PASSED - NCB Feature Factory V2 is working!")
        print(f"{'='*80}\n")
    else:
        print(f"\n{'='*80}")
        print("⚠️  TESTS COMPLETED WITH WARNINGS - Review missing/banned columns")
        print(f"{'='*80}\n")

    print(f"Completed: {datetime.now()}")

    return features_df


# ============================================================================
# Run Test
# ============================================================================

if __name__ == "__main__":
    try:
        result_df = test_ncb_factory_v2()

        print("\n" + "="*80)
        print("NEXT STEPS")
        print("="*80)
        print("\n1. Review feature counts and null rates")
        print("2. If passing, the new architecture is ready for production")
        print("3. Old multi-file design can be deprecated")
        print("\nResult DataFrame available as: result_df")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
