"""
Test Behavioral Physics System on Actual Bureau Data
====================================================

This script tests the full behavioral physics pipeline on your actual
bureau tables from cdx_mdz_prd.cdx_persist_mnf_res_db.

Run this in Databricks or local Spark environment.
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from datetime import datetime
import json

# Import behavioral physics modules
from behavioral_physics_features.modules import (
    BureauSchemaAdapter,
    BehavioralPhysicsPipeline
)

def test_actual_bureau_data():
    """
    Test behavioral physics on actual bureau data.
    """
    print("\n" + "="*80)
    print("BEHAVIORAL PHYSICS - ACTUAL DATA TEST")
    print(f"Test Started: {datetime.now()}")
    print("="*80 + "\n")

    # Initialize Spark
    spark = SparkSession.builder \
        .appName("BehavioralPhysics_ActualDataTest") \
        .getOrCreate()

    # ========================================================================
    # STEP 1: Load Actual Bureau Tables
    # ========================================================================
    print("STEP 1: Loading Actual Bureau Tables...")
    print("-" * 80)

    try:
        # Your actual bureau tables
        catalog = "cdx_mdz_prd.cdx_persist_mnf_res_db"

        bureau_account = spark.table(f"{catalog}.mnf_cra_rvw_s_account")
        bureau_history = spark.table(f"{catalog}.mnf_cra_rvw_s_history")
        bureau_enquiry = spark.table(f"{catalog}.mnf_cra_rvw_s_enquiry")

        print(f"✓ Loaded bureau_account: {bureau_account.count():,} rows")
        print(f"✓ Loaded bureau_history: {bureau_history.count():,} rows")
        print(f"✓ Loaded bureau_enquiry: {bureau_enquiry.count():,} rows")

        # Show sample data
        print("\nSample bureau_account:")
        bureau_account.select(
            "REF_NO", "ACCOUNTNUMBER", "ASOFDATE", "MEMBERSHORTNAME",
            "ACCOUNTTYPE", "CREDITLIMIT", "AMOUNTOWED"
        ).show(5, truncate=False)

        print("\nSample bureau_history:")
        bureau_history.select(
            "REF_NO", "ASOFDATE", "CREDITLIMIT", "AMOUNTOWED", "OVERDUEMONTHS"
        ).show(5, truncate=False)

    except Exception as e:
        print(f"❌ Error loading bureau tables: {e}")
        print("\nPlease ensure:")
        print("  1. You're running this in Databricks")
        print("  2. You have access to cdx_mdz_prd.cdx_persist_mnf_res_db")
        print("  3. Tables exist: mnf_cra_rvw_s_account, mnf_cra_rvw_s_history, mnf_cra_rvw_s_enquiry")
        return

    # ========================================================================
    # STEP 2: Schema Adaptation
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 2: Adapting Bureau Schema...")
    print("-" * 80)

    try:
        # Initialize schema adapter
        adapter = BureauSchemaAdapter(spark)

        # Adapt bureau trade data
        print("Adapting bureau trade data (account + history)...")
        bureau_trade = adapter.adapt_bureau_trade_data(
            account_df=bureau_account,
            history_df=bureau_history
        )

        print(f"✓ Bureau trade adapted: {bureau_trade.count():,} rows")
        print(f"✓ Customers: {bureau_trade.select('cust_id').distinct().count():,}")

        # Check date range
        date_range = bureau_trade.select(
            F.min("as_of_month").alias("min_date"),
            F.max("as_of_month").alias("max_date")
        ).first()
        print(f"✓ Date range: {date_range['min_date']} to {date_range['max_date']}")

        # Show adapted schema
        print("\nAdapted Bureau Trade Schema:")
        print("Expected columns: cust_id, as_of_month, dpd, balance, credit_limit, utilization, lender_name")
        bureau_trade.select(
            "cust_id", "as_of_month", "dpd", "balance", "credit_limit",
            "utilization", "lender_name"
        ).show(5)

        # Check DPD distribution
        print("\nDPD Distribution:")
        bureau_trade.groupBy("dpd").count().orderBy("dpd").show(10)

        # Adapt enquiry data
        print("\nAdapting bureau enquiry data...")
        bureau_enquiry_adapted = adapter.adapt_bureau_enquiry_data(
            enquiry_df=bureau_enquiry
        )

        print(f"✓ Bureau enquiry adapted: {bureau_enquiry_adapted.count():,} rows")

    except Exception as e:
        print(f"❌ Error adapting schema: {e}")
        import traceback
        traceback.print_exc()
        return

    # ========================================================================
    # STEP 3: Load/Mock CardX Internal Data
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 3: Loading CardX Internal Data...")
    print("-" * 80)

    try:
        # TODO: Replace with your actual CardX table
        # cardx_internal = spark.table("your_catalog.cardx_internal_monthly")

        # For testing: Create mock CardX data based on bureau customers
        print("Creating mock CardX data for testing...")
        print("(Replace this with actual CardX table in production)")

        # Get unique customers from bureau
        customer_months = bureau_trade.select("cust_id", "as_of_month").distinct()

        # Create mock CardX data
        cardx_internal = customer_months.withColumn(
            "cardx_dpd", F.lit(0)  # Mock: no delinquency
        ).withColumn(
            "cardx_balance", F.lit(5000)  # Mock: 5000 balance
        ).withColumn(
            "cardx_credit_limit", F.lit(10000)  # Mock: 10000 limit
        )

        print(f"✓ CardX internal (mock): {cardx_internal.count():,} rows")

        print("\n⚠️  NOTE: Using mock CardX data. Replace with actual table:")
        print("   cardx_internal = spark.table('your_catalog.cardx_internal_monthly')")

    except Exception as e:
        print(f"❌ Error loading CardX data: {e}")
        import traceback
        traceback.print_exc()
        return

    # ========================================================================
    # STEP 4: Select As-of Month for Testing
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 4: Selecting As-of Month...")
    print("-" * 80)

    # Use most recent month from data
    max_month = bureau_trade.agg(F.max("as_of_month")).first()[0]
    as_of_month = str(max_month)

    print(f"✓ Testing with as_of_month: {as_of_month}")

    # Filter to recent data for faster testing
    print("\nFiltering to last 3 months for faster testing...")
    three_months_ago = F.add_months(F.lit(as_of_month), -3)

    bureau_trade_recent = bureau_trade.filter(F.col("as_of_month") >= three_months_ago)
    cardx_internal_recent = cardx_internal.filter(F.col("as_of_month") >= three_months_ago)

    print(f"✓ Bureau trade (recent): {bureau_trade_recent.count():,} rows")
    print(f"✓ CardX internal (recent): {cardx_internal_recent.count():,} rows")

    # ========================================================================
    # STEP 5: Run Behavioral Physics Pipeline
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 5: Running Behavioral Physics Pipeline...")
    print("="*80 + "\n")

    try:
        # Initialize pipeline
        pipeline = BehavioralPhysicsPipeline(spark)

        # Run full pipeline
        print("Computing 200 behavioral physics features...")
        print("This may take a few minutes on large data...\n")

        features_df, audit_log, qa_summary = pipeline.run(
            bureau_trade_df=bureau_trade_recent,
            bureau_enquiry_df=bureau_enquiry_adapted,
            cardx_internal_df=cardx_internal_recent,
            as_of_month=as_of_month
        )

        print("\n" + "="*80)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*80)

    except Exception as e:
        print(f"\n❌ Error running pipeline: {e}")
        import traceback
        traceback.print_exc()
        return

    # ========================================================================
    # STEP 6: Validate Results
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 6: Validating Results...")
    print("-" * 80)

    # Count features
    feature_cols = [col for col in features_df.columns if col not in ["cust_id", "as_of_month"]]
    print(f"\n✓ Total features generated: {len(feature_cols)}")
    print(f"✓ Total customers: {features_df.count():,}")

    # Show feature families
    print("\nFeature Families:")
    velocity_features = [f for f in feature_cols if 'velocity' in f]
    acceleration_features = [f for f in feature_cols if 'acceleration' in f or 'shock' in f]
    entropy_features = [f for f in feature_cols if 'entropy' in f or 'oscillation' in f]
    lender_features = [f for f in feature_cols if 'lender' in f]
    cardx_features = [f for f in feature_cols if 'cardx' in f]
    legal_features = [f for f in feature_cols if 'legal' in f or 'settlement' in f or 'writeoff' in f]
    tdr_features = [f for f in feature_cols if 'tdr' in f]

    print(f"  - Velocity: {len(velocity_features)} features")
    print(f"  - Acceleration: {len(acceleration_features)} features")
    print(f"  - Entropy: {len(entropy_features)} features")
    print(f"  - Lender Ecology: {len(lender_features)} features")
    print(f"  - CardX-Bureau: {len(cardx_features)} features")
    print(f"  - Legal: {len(legal_features)} features")
    print(f"  - TDR: {len(tdr_features)} features")

    # Sample features
    print("\n" + "-"*80)
    print("Sample Features (Top 10 customers):")
    print("-"*80)

    features_df.select(
        "cust_id", "as_of_month",
        "dpd_velocity_3m",
        "dpd_acceleration_3m",
        "state_entropy_6m",
        "lender_hhi",
        "synchronized_delinquency_flag"
    ).show(10)

    # ========================================================================
    # STEP 7: Quality Checks
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 7: Quality Checks...")
    print("-" * 80)

    # Check for nulls
    print("\nNull counts by feature (showing features with >10% nulls):")
    null_summary = []
    total_rows = features_df.count()

    for col in feature_cols[:20]:  # Check first 20 features
        null_count = features_df.filter(F.col(col).isNull()).count()
        null_pct = (null_count / total_rows * 100) if total_rows > 0 else 0

        if null_pct > 10:
            null_summary.append((col, null_pct))

    if null_summary:
        for col, pct in sorted(null_summary, key=lambda x: -x[1])[:10]:
            print(f"  {col}: {pct:.1f}% null")
    else:
        print("  ✓ No features with >10% nulls")

    # Feature distributions
    print("\nFeature Distributions (Sample):")
    features_df.select(
        "dpd_velocity_3m",
        "state_entropy_6m",
        "lender_hhi"
    ).describe().show()

    # ========================================================================
    # STEP 8: QA Summary
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 8: QA Summary...")
    print("-" * 80)

    print(json.dumps(qa_summary, indent=2, default=str))

    # ========================================================================
    # STEP 9: Audit Log
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 9: Audit Log...")
    print("-" * 80)

    print("\nPipeline Execution Audit:")
    audit_log.show(truncate=False)

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("TEST COMPLETE - SUMMARY")
    print("="*80)

    print(f"\n✅ Successfully processed actual bureau data!")
    print(f"\nKey Metrics:")
    print(f"  - Bureau accounts: {bureau_account.count():,}")
    print(f"  - Bureau history: {bureau_history.count():,}")
    print(f"  - Features generated: {len(feature_cols)}")
    print(f"  - Customers processed: {features_df.count():,}")
    print(f"  - As-of month: {as_of_month}")

    print(f"\nNext Steps:")
    print(f"  1. Review feature distributions above")
    print(f"  2. Check QA summary for quality issues")
    print(f"  3. Replace mock CardX data with actual table")
    print(f"  4. Train model with these features")
    print(f"  5. Measure AUC lift vs baseline")

    print(f"\n💾 To save features:")
    print(f"  features_df.write.format('delta').mode('overwrite').saveAsTable('feature_store.behavioral_physics')")

    print("\n" + "="*80)
    print(f"Test Completed: {datetime.now()}")
    print("="*80 + "\n")

    # Return for interactive use
    return features_df, audit_log, qa_summary


# ============================================================================
# Run Test
# ============================================================================

if __name__ == "__main__":
    try:
        features_df, audit_log, qa_summary = test_actual_bureau_data()

        print("\n✅ Test script completed successfully!")
        print("\nFeatures DataFrame available as: features_df")
        print("Audit log available as: audit_log")
        print("QA summary available as: qa_summary")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
