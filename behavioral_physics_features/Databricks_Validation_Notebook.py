# Databricks notebook source
# MAGIC %md
# MAGIC # Behavioral Physics Pipeline - Complete Validation
# MAGIC
# MAGIC **Purpose**: Validate all 6 critical fixes in the behavioral physics feature pipeline
# MAGIC
# MAGIC **Branch**: `recovery_agent_practical`
# MAGIC
# MAGIC **Commit**: `0380aaf` - Fix 6 critical issues in behavioral physics pipeline + Thai lender classification
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎯 What This Notebook Does
# MAGIC
# MAGIC 1. **Validates Schema Fixes**: Tests all 6 critical fixes
# MAGIC 2. **Validates Lender Classification**: Tests Thai lender type mapping
# MAGIC 3. **Validates XXX Handling**: Tests "not reported" handling
# MAGIC 4. **Runs Integration Test**: End-to-end pipeline test
# MAGIC 5. **Provides Results Summary**: Clear pass/fail indicators
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📋 Fixes Being Validated
# MAGIC
# MAGIC | # | Fix | Expected Result |
# MAGIC |---|-----|-----------------|
# MAGIC | 1 | RECEIVE_DT point-in-time safety | No data leakage |
# MAGIC | 2 | Payment history expansion (48m) | 4x more DPD history |
# MAGIC | 3 | DPD bucket ordinal (0-6) | Proper ordinal categories |
# MAGIC | 4 | XXX → null mapping | Correct "not reported" |
# MAGIC | 5 | lender_code → lender_id | Column exists |
# MAGIC | 6 | Thai lender classification | Accurate categories |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **⏱ Estimated Runtime**: 15-20 minutes
# MAGIC
# MAGIC **📊 Success Criteria**: All tests should PASS ✅

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup: Import Libraries

# COMMAND ----------

from pyspark.sql import functions as F
from datetime import datetime
import json

print(f"✓ Validation started: {datetime.now()}")
print(f"✓ Spark version: {spark.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Test 1: Lender Classification Validation
# MAGIC
# MAGIC **What**: Validates Thai lender type classification
# MAGIC
# MAGIC **Expected**:
# MAGIC - All major lenders correctly classified
# MAGIC - OTHER category <5%
# MAGIC - At least 4 active categories

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run Lender Classification Validator

# COMMAND ----------

%run ./behavioral_physics_features/validate_lender_classification.py

# COMMAND ----------

# MAGIC %md
# MAGIC ### Review Lender Classification Results
# MAGIC
# MAGIC **Check**:
# MAGIC - ✅ COMMERCIAL_BANK should be largest (~40-60%)
# MAGIC - ✅ SFI should have ~15-20%
# MAGIC - ✅ OTHER should be <5%
# MAGIC - ⚠️ If OTHER >5%, note which lenders need to be added to config.py

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Test 2: XXX Handling Validation
# MAGIC
# MAGIC **What**: Validates "XXX" (not reported) handling
# MAGIC
# MAGIC **Expected**:
# MAGIC - XXX maps to null (not 0)
# MAGIC - has_reporting_gap = 1 when XXX exists
# MAGIC - reporting_gap_count_12m counts correctly

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run XXX Handler Test

# COMMAND ----------

%run ./behavioral_physics_features/test_xxx_handling.py

# COMMAND ----------

# MAGIC %md
# MAGIC ### Review XXX Handling Results
# MAGIC
# MAGIC **Check**:
# MAGIC - ✅ [TEST 1] XXX maps to null: Should see "dpd IS NULL: 100.0%"
# MAGIC - ✅ [TEST 2] has_reporting_gap: Should see "PASS: has_reporting_gap correctly matches XXX presence"
# MAGIC - ✅ [TEST 3] reporting_gap_count_12m: Should see distribution 0-12
# MAGIC - ✅ [TEST 4] Other buckets: Should see ordinals 0-6

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Test 3: Integration Test (All Fixes)
# MAGIC
# MAGIC **What**: Comprehensive test of all 6 fixes + full pipeline
# MAGIC
# MAGIC **Expected**:
# MAGIC - All required columns exist
# MAGIC - Point-in-time filter works
# MAGIC - Payment history expanded
# MAGIC - Full pipeline completes successfully
# MAGIC - 200 features generated

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run Integration Test

# COMMAND ----------

%run ./behavioral_physics_features/test_all_fixes.py

# COMMAND ----------

# MAGIC %md
# MAGIC ### Review Integration Test Results
# MAGIC
# MAGIC **Check**:
# MAGIC - ✅ [TEST 1] Required columns: All 15 columns should exist
# MAGIC - ✅ [TEST 2] RECEIVE_DT: "No future data leakage"
# MAGIC - ✅ [TEST 3] Payment history: "expanded by X rows"
# MAGIC - ✅ [TEST 4] XXX handling: "All XXX rows have null dpd"
# MAGIC - ✅ [TEST 5] Reporting gap: Distribution shows values
# MAGIC - ✅ [TEST 6] lender_id: Column exists with unique lenders
# MAGIC - ✅ [TEST 7] dpd_bucket_ordinal: Ordinals 0-6
# MAGIC - ✅ [STEP 3] Pipeline: "Pipeline completed successfully"

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Test 4: Quick Data Quality Checks
# MAGIC
# MAGIC Manual spot checks on the adapted bureau data

# COMMAND ----------

# MAGIC %md
# MAGIC ### Check 1: Schema Validation

# COMMAND ----------

from behavioral_physics_features.modules import BureauSchemaAdapter

# Load bureau data
catalog = "cdx_mdz_prd.cdx_persist_mnf_res_db"
bureau_account = spark.table(f"{catalog}.mnf_cra_rvw_s_account")
bureau_history = spark.table(f"{catalog}.mnf_cra_rvw_s_history")

# Initialize adapter
adapter = BureauSchemaAdapter(spark)

# Adapt bureau data
bureau_trade = adapter.adapt_bureau_trade_data(
    account_df=bureau_account,
    history_df=bureau_history
)

print(f"✓ Bureau trade adapted: {bureau_trade.count():,} rows")
print(f"✓ Customers: {bureau_trade.select('cust_id').distinct().count():,}")

# Check schema
print("\n📋 Schema Check:")
required_columns = [
    "cust_id", "as_of_month", "account_id",
    "lender_name", "lender_id",  # Fix 5
    "dpd", "dpd_bucket", "dpd_bucket_ordinal",  # Fix 3
    "receive_dt",  # Fix 1
    "has_reporting_gap", "reporting_gap_count_12m",  # Fix 4
    "source"  # Fix 2
]

for col in required_columns:
    status = "✅" if col in bureau_trade.columns else "❌"
    print(f"{status} {col}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Check 2: Payment History Expansion

# COMMAND ----------

# Check source distribution
print("📊 Data Source Distribution:")
bureau_trade.groupBy("source").count().orderBy(F.desc("count")).show()

# Calculate expansion rate
source_counts = bureau_trade.groupBy("source").count().collect()
source_dict = {row["source"]: row["count"] for row in source_counts}

history_rows = source_dict.get("history_table", 0)
payment_rows = source_dict.get("payment_history", 0)
total_rows = history_rows + payment_rows

if payment_rows > 0:
    expansion_pct = (payment_rows / total_rows * 100)
    print(f"\n✅ Payment history expanded by {payment_rows:,} rows ({expansion_pct:.1f}%)")
else:
    print(f"\n⚠️  WARNING: No payment history expansion")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Check 3: XXX Distribution

# COMMAND ----------

# Check XXX in dpd_bucket
print("📊 DPD Bucket Distribution (including XXX):")
bureau_trade.groupBy("dpd_bucket").count().orderBy(F.desc("count")).show(15)

# Check XXX handling
xxx_check = bureau_trade.filter(F.col("dpd_bucket") == "XXX")
xxx_count = xxx_check.count()

if xxx_count > 0:
    print(f"\n✓ Found {xxx_count:,} XXX rows")

    # Check if dpd is null for XXX
    null_dpd_count = xxx_check.filter(F.col("dpd").isNull()).count()
    null_ordinal_count = xxx_check.filter(F.col("dpd_bucket_ordinal").isNull()).count()

    print(f"✓ dpd IS NULL: {null_dpd_count:,} / {xxx_count:,} ({null_dpd_count/xxx_count*100:.1f}%)")
    print(f"✓ dpd_bucket_ordinal IS NULL: {null_ordinal_count:,} / {xxx_count:,} ({null_ordinal_count/xxx_count*100:.1f}%)")

    if null_dpd_count == xxx_count and null_ordinal_count == xxx_count:
        print("\n✅ XXX handling CORRECT: All XXX rows have null dpd and dpd_bucket_ordinal")
    else:
        print("\n❌ XXX handling INCORRECT: Some XXX rows don't have null values")
else:
    print("\n⚠️  No XXX rows found (may be acceptable if data has perfect reporting)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Check 4: Lender Type Distribution

# COMMAND ----------

# Check lender_id exists
if "lender_id" in bureau_trade.columns:
    print("✅ lender_id column exists (Fix 5)")

    # Sample lenders
    print("\n📊 Sample Lenders:")
    bureau_trade.select("lender_id", "lender_name") \
        .distinct() \
        .orderBy("lender_id") \
        .show(20, truncate=False)

    unique_lenders = bureau_trade.select("lender_id").distinct().count()
    print(f"\n✓ Unique lenders: {unique_lenders:,}")
else:
    print("❌ lender_id column MISSING (Fix 5 failed)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Check 5: DPD Bucket Ordinal

# COMMAND ----------

# Check dpd_bucket_ordinal distribution
if "dpd_bucket_ordinal" in bureau_trade.columns:
    print("✅ dpd_bucket_ordinal column exists (Fix 3)")

    print("\n📊 DPD Bucket → Ordinal Mapping:")
    bureau_trade.filter(F.col("dpd_bucket_ordinal").isNotNull()) \
        .groupBy("dpd_bucket", "dpd", "dpd_bucket_ordinal") \
        .count() \
        .orderBy("dpd_bucket_ordinal") \
        .show(10)

    # Check ordinal values are 0-6
    ordinals = bureau_trade.filter(F.col("dpd_bucket_ordinal").isNotNull()) \
        .select("dpd_bucket_ordinal") \
        .distinct() \
        .orderBy("dpd_bucket_ordinal") \
        .collect()

    ordinal_values = [row["dpd_bucket_ordinal"] for row in ordinals]
    expected_ordinals = [0, 1, 2, 3, 4, 5, 6]

    if ordinal_values == expected_ordinals:
        print(f"\n✅ Ordinal categories correct: {ordinal_values}")
    else:
        print(f"\n⚠️  Ordinal categories: {ordinal_values} (expected: {expected_ordinals})")
else:
    print("❌ dpd_bucket_ordinal column MISSING (Fix 3 failed)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Check 6: Reporting Gap Features

# COMMAND ----------

# Check has_reporting_gap
if "has_reporting_gap" in bureau_trade.columns:
    print("✅ has_reporting_gap column exists (Fix 4)")

    print("\n📊 has_reporting_gap Distribution:")
    bureau_trade.groupBy("has_reporting_gap").count().orderBy("has_reporting_gap").show()
else:
    print("❌ has_reporting_gap column MISSING (Fix 4 failed)")

# Check reporting_gap_count_12m
if "reporting_gap_count_12m" in bureau_trade.columns:
    print("\n✅ reporting_gap_count_12m column exists (Fix 4)")

    print("\n📊 reporting_gap_count_12m Distribution:")
    bureau_trade.groupBy("reporting_gap_count_12m").count() \
        .orderBy("reporting_gap_count_12m") \
        .show(13)
else:
    print("❌ reporting_gap_count_12m column MISSING (Fix 4 failed)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Check 7: Point-in-Time Safety

# COMMAND ----------

# Check receive_dt
if "receive_dt" in bureau_trade.columns:
    print("✅ receive_dt column exists (Fix 1)")

    # Check for data leakage
    # Note: We can't validate against as_of_month here since we don't have it filtered yet
    # This will be validated in the main pipeline

    print("\n📊 receive_dt Statistics:")
    bureau_trade.select(
        F.min("receive_dt").alias("min_receive_dt"),
        F.max("receive_dt").alias("max_receive_dt"),
        F.count("receive_dt").alias("non_null_count"),
        F.sum(F.col("receive_dt").isNull().cast("int")).alias("null_count")
    ).show()

    # Null receive_dt should be from payment_history
    null_receive = bureau_trade.filter(F.col("receive_dt").isNull())
    null_count = null_receive.count()

    if null_count > 0:
        print(f"\n✓ receive_dt NULL rows: {null_count:,}")
        # Check if these are payment_history rows
        payment_hist_null = null_receive.filter(F.col("source") == "payment_history").count()
        print(f"✓ From payment_history: {payment_hist_null:,} ({payment_hist_null/null_count*100:.1f}%)")

        if payment_hist_null == null_count:
            print("\n✅ All null receive_dt are from payment_history (correct)")
        else:
            print(f"\n⚠️  {null_count - payment_hist_null:,} null receive_dt NOT from payment_history")
else:
    print("❌ receive_dt column MISSING (Fix 1 failed)")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Final Summary

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Validation Results Summary
# MAGIC
# MAGIC Review the outputs above and verify:
# MAGIC
# MAGIC ### Fix 1: RECEIVE_DT Point-in-Time Safety
# MAGIC - [ ] receive_dt column exists
# MAGIC - [ ] Null receive_dt only from payment_history
# MAGIC - [ ] No future data leakage detected
# MAGIC
# MAGIC ### Fix 2: Payment History Expansion
# MAGIC - [ ] source column shows "history_table" and "payment_history"
# MAGIC - [ ] Payment history rows >20% of total
# MAGIC - [ ] Time series extended backwards
# MAGIC
# MAGIC ### Fix 3: DPD Bucket Ordinal
# MAGIC - [ ] dpd_bucket_ordinal column exists
# MAGIC - [ ] Ordinal values are 0-6
# MAGIC - [ ] Mapping to dpd_bucket is correct
# MAGIC
# MAGIC ### Fix 4: XXX Handling
# MAGIC - [ ] XXX rows have null dpd (100%)
# MAGIC - [ ] XXX rows have null dpd_bucket_ordinal (100%)
# MAGIC - [ ] has_reporting_gap exists and has values
# MAGIC - [ ] reporting_gap_count_12m exists and ranges 0-12
# MAGIC
# MAGIC ### Fix 5: lender_id Schema
# MAGIC - [ ] lender_id column exists
# MAGIC - [ ] lender_id has unique values
# MAGIC - [ ] No "lender_code" column (renamed)
# MAGIC
# MAGIC ### Fix 6: Thai Lender Classification
# MAGIC - [ ] COMMERCIAL_BANK is largest category (~40-60%)
# MAGIC - [ ] SFI has significant share (~15-20%)
# MAGIC - [ ] OTHER category <5%
# MAGIC - [ ] At least 4 active categories
# MAGIC
# MAGIC ### Integration
# MAGIC - [ ] Full pipeline completes successfully
# MAGIC - [ ] 200 features generated
# MAGIC - [ ] No errors in any module

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Next Steps
# MAGIC
# MAGIC ## If All Tests Passed ✅
# MAGIC
# MAGIC 1. **Create Pull Request**
# MAGIC    - Go to: https://github.com/Sushil-tata/claude_DA2/pull/new/recovery_agent_practical
# MAGIC    - Title: "Fix 6 critical issues in behavioral physics pipeline"
# MAGIC    - Description: See ALL_FIXES_SUMMARY.md
# MAGIC
# MAGIC 2. **After PR Merge**
# MAGIC    - Retrain models with corrected data
# MAGIC    - Measure AUC lift (expected +2-5%)
# MAGIC    - Monitor reporting gap distribution
# MAGIC    - Update feature documentation
# MAGIC
# MAGIC ## If Tests Failed ❌
# MAGIC
# MAGIC 1. **Review Error Messages**
# MAGIC    - Check which specific test failed
# MAGIC    - Review the error details in the output above
# MAGIC
# MAGIC 2. **Common Issues**
# MAGIC    - **Lender classification**: Add missing lenders to config.py
# MAGIC    - **XXX handling**: Check if XXX exists in your data
# MAGIC    - **Schema issues**: Verify column names in actual tables
# MAGIC
# MAGIC 3. **Get Help**
# MAGIC    - See: behavioral_physics_features/ALL_FIXES_SUMMARY.md
# MAGIC    - See: behavioral_physics_features/THAI_LENDER_CLASSIFICATION.md
# MAGIC    - See: behavioral_physics_features/XXX_TESTING_README.md

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC **Validation Complete!** 🎉
# MAGIC
# MAGIC Review the results above and proceed with creating a pull request if all tests passed.
