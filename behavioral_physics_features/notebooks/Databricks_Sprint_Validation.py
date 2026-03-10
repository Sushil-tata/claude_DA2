# Databricks notebook source
# MAGIC %md
# MAGIC # Sprint Validation: Thai DPD + Explicit Join Keys + MEMBERSHORTNAME
# MAGIC
# MAGIC **Purpose**: Validate all fixes from recovery_agent_practical branch
# MAGIC
# MAGIC **Tests**:
# MAGIC 1. Bridge join validation (fan-out = 0, match rate ~97%)
# MAGIC 2. Explicit ref_no join keys (no implicit cust_id)
# MAGIC 3. Thai DPD classification (CURRENT/SM/NPL/CHARGE_OFF)
# MAGIC 4. MEMBERSHORTNAME scope (no lender_id)
# MAGIC 5. Full pipeline test with run_from_raw_tables()
# MAGIC 6. Output schema validation

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

# DBTITLE 1,Configuration
# Test parameters
TEST_MONTH = "2024-12-31"
CATALOG = "cdx_mdz_prd"  # Update to your catalog
TEST_CUSTOMER_SAMPLE = 1000  # Sample size for quick validation

print(f"🧪 Test Configuration:")
print(f"   Month: {TEST_MONTH}")
print(f"   Catalog: {CATALOG}")
print(f"   Sample: {TEST_CUSTOMER_SAMPLE} customers")

# COMMAND ----------

# DBTITLE 1,Import Pipeline
import sys, importlib

# Clear any stale cached module versions
for mod_name in list(sys.modules.keys()):
    if mod_name == 'modules' or mod_name.startswith('modules.'):
        del sys.modules[mod_name]

# Set correct path
MODULES_BASE = "/Workspace/Users/sushil@cardx.co.th/claude_DA2/behavioral_physics_features"
if MODULES_BASE not in sys.path:
    sys.path.insert(0, MODULES_BASE)

# Import directly from individual modules (bypass __init__.py cache issues)
from modules.main_pipeline import BehavioralPhysicsPipeline
from modules.cardx_schema_adapter import CardXSchemaAdapter, build_bridge_df
from modules.bureau_schema_adapter import BureauSchemaAdapter
from modules.state_builder import StateBuilder
from modules.lender_ecology import LenderEcologyEngine
from modules.config import get_config
from pyspark.sql import functions as F

print("✅ Imports successful")

# COMMAND ----------

# MAGIC %md
# MAGIC ## TEST 1: Bridge Join Validation

# COMMAND ----------

# DBTITLE 1,Test 1.1 - Build Bridge
print("=" * 70)
print("TEST 1: BRIDGE JOIN VALIDATION")
print("=" * 70)

cardx_adapter = CardXSchemaAdapter(spark)

# Build bridge
bridge_df = cardx_adapter.build_bridge_df(
    spark=spark,
    as_of_month=TEST_MONTH,
    catalog=CATALOG
)

print(f"\n📊 Bridge Statistics:")
bridge_count = bridge_df.count()
print(f"   Total rows: {bridge_count:,}")

# Check columns
bridge_cols = bridge_df.columns
print(f"   Columns: {bridge_cols}")

# Display sample
print(f"\n🔍 Sample Bridge Data:")
display(bridge_df.limit(10))

# COMMAND ----------

# DBTITLE 1,Test 1.2 - Fan-out Check (CRITICAL)
print("\n🔍 FAN-OUT CHECK (Must be 0)")
print("-" * 50)

# Check for duplicate ACCT_NUM
fanout_check = bridge_df.groupBy("ACCT_NUM").count().filter("count > 1")
fanout_count = fanout_check.count()

if fanout_count == 0:
    print("✅ PASSED: No fan-out detected (each ACCT_NUM has exactly 1 REF_NO)")
else:
    print(f"❌ FAILED: {fanout_count} accounts have multiple REF_NO mappings")
    print("   Sample duplicates:")
    display(fanout_check.limit(20))
    raise ValueError("FAN-OUT DETECTED - BLOCKING ISSUE")

# COMMAND ----------

# DBTITLE 1,Test 1.3 - Match Rate Check
print("\n🔍 MATCH RATE CHECK (Expected ~97%)")
print("-" * 50)

# Load CardX accounts for the month
cardx_accounts = spark.table(f"{CATALOG}.spl_acct_mthly") \
    .filter(F.col("DL_DATA_DT") == TEST_MONTH)

cardx_count = cardx_accounts.count()
bridge_with_ref = bridge_df.filter(F.col("REF_NO").isNotNull()).count()
match_rate = (bridge_with_ref / cardx_count * 100) if cardx_count > 0 else 0

print(f"   CardX accounts: {cardx_count:,}")
print(f"   With REF_NO: {bridge_with_ref:,}")
print(f"   Match rate: {match_rate:.1f}%")

if match_rate >= 95.0:
    print(f"✅ PASSED: Match rate {match_rate:.1f}% >= 95%")
elif match_rate >= 90.0:
    print(f"⚠️  WARNING: Match rate {match_rate:.1f}% below expected 97%")
else:
    print(f"❌ FAILED: Match rate {match_rate:.1f}% too low")
    raise ValueError("MATCH RATE TOO LOW")

# COMMAND ----------

# MAGIC %md
# MAGIC ## TEST 2: Explicit ref_no Join Keys

# COMMAND ----------

# DBTITLE 1,Test 2.1 - Check ref_no Column Exists
print("=" * 70)
print("TEST 2: EXPLICIT REF_NO JOIN KEYS")
print("=" * 70)

bureau_adapter = BureauSchemaAdapter(spark)

# Load bureau tables
bureau_history = spark.table(f"{CATALOG}.mnf_cra_rvw_s_history") \
    .filter(F.col("DL_DATA_DT") == TEST_MONTH)

bureau_account = spark.table(f"{CATALOG}.mnf_cra_rvw_s_account") \
    .filter(F.col("DL_DATA_DT") == TEST_MONTH)

print(f"\n📊 Bureau Data Loaded:")
print(f"   History rows: {bureau_history.count():,}")
print(f"   Account rows: {bureau_account.count():,}")

# Adapt bureau trade data
bureau_trade_df = bureau_adapter.adapt_bureau_trade_data(
    bureau_account_df=bureau_account,
    bureau_history_df=bureau_history,
    bridge_df=bridge_df,
    as_of_month=TEST_MONTH
)

print(f"\n✅ Bureau trade adapted: {bureau_trade_df.count():,} rows")

# Check schema
print(f"\n🔍 Checking ref_no column...")
if "ref_no" in bureau_trade_df.columns:
    print("✅ PASSED: ref_no column exists")
else:
    print("❌ FAILED: ref_no column missing")
    print(f"   Available columns: {bureau_trade_df.columns}")
    raise ValueError("ref_no COLUMN MISSING")

# Check cust_id alias exists
if "cust_id" in bureau_trade_df.columns:
    print("✅ PASSED: cust_id alias exists (for compatibility)")
else:
    print("⚠️  WARNING: cust_id alias missing")

# COMMAND ----------

# DBTITLE 1,Test 2.2 - Verify ref_no Values
print("\n🔍 VERIFY REF_NO VALUES")
print("-" * 50)

# Check ref_no not null
null_ref_no = bureau_trade_df.filter(F.col("ref_no").isNull()).count()
total_rows = bureau_trade_df.count()
null_pct = (null_ref_no / total_rows * 100) if total_rows > 0 else 0

print(f"   Total rows: {total_rows:,}")
print(f"   Null ref_no: {null_ref_no:,} ({null_pct:.2f}%)")

if null_pct < 5.0:
    print(f"✅ PASSED: Null ref_no rate {null_pct:.2f}% < 5%")
else:
    print(f"⚠️  WARNING: High null ref_no rate {null_pct:.2f}%")

# Check ref_no = cust_id (should be identical)
mismatch_count = bureau_trade_df.filter(
    F.col("ref_no") != F.col("cust_id")
).count()

if mismatch_count == 0:
    print("✅ PASSED: cust_id correctly aliased from ref_no")
else:
    print(f"❌ FAILED: {mismatch_count:,} rows where ref_no != cust_id")
    raise ValueError("CUST_ID ALIAS MISMATCH")

# COMMAND ----------

# DBTITLE 1,Test 2.3 - Check Join Keys
print("\n🔍 VERIFY JOIN KEYS USED")
print("-" * 50)

# Check that (ref_no, seq_tl) is unique tradeline key
tradeline_check = bureau_trade_df.groupBy("ref_no", "seq_tl", "as_of_month").count()
duplicates = tradeline_check.filter("count > 1").count()

if duplicates == 0:
    print("✅ PASSED: (ref_no, seq_tl, as_of_month) uniquely identifies rows")
else:
    print(f"❌ FAILED: {duplicates} duplicate (ref_no, seq_tl, as_of_month) combinations")
    display(tradeline_check.filter("count > 1").limit(20))
    raise ValueError("DUPLICATE TRADELINES DETECTED")

# COMMAND ----------

# MAGIC %md
# MAGIC ## TEST 3: Thai DPD Classification

# COMMAND ----------

# DBTITLE 1,Test 3.1 - Check DPD Parsing
print("=" * 70)
print("TEST 3: THAI DPD CLASSIFICATION")
print("=" * 70)

# Check DPD values exist
dpd_stats = bureau_trade_df.select(
    F.min("dpd").alias("min_dpd"),
    F.max("dpd").alias("max_dpd"),
    F.avg("dpd").alias("avg_dpd"),
    F.count(F.when(F.col("dpd").isNull(), 1)).alias("null_count")
).collect()[0]

print(f"\n📊 DPD Statistics:")
print(f"   Min DPD: {dpd_stats['min_dpd']}")
print(f"   Max DPD: {dpd_stats['max_dpd']}")
print(f"   Avg DPD: {dpd_stats['avg_dpd']:.2f}")
print(f"   Null count: {dpd_stats['null_count']:,}")

if dpd_stats['min_dpd'] is not None and dpd_stats['min_dpd'] >= 0:
    print("✅ PASSED: DPD values parsed correctly")
else:
    print("❌ FAILED: DPD parsing issue")

# COMMAND ----------

# DBTITLE 1,Test 3.2 - Verify Thai States
print("\n🔍 VERIFY THAI DPD STATES")
print("-" * 50)

# Build states using state_builder
from modules.state_builder import StateBuilder

state_builder = StateBuilder(spark)
states_df = state_builder.build_states(
    bureau_trade_monthly_df=bureau_trade_df.limit(10000)  # Sample for quick test
)

# Check state distribution
state_dist = states_df.groupBy("bureau_state").count().orderBy("bureau_state")

print("\n📊 State Distribution:")
display(state_dist)

# Check expected states
expected_states = {"CURRENT", "SM", "NPL", "CHARGE_OFF"}
actual_states = set([row['bureau_state'] for row in state_dist.collect()])

if expected_states.issubset(actual_states):
    print("✅ PASSED: All Thai states present (CURRENT, SM, NPL, CHARGE_OFF)")
else:
    missing = expected_states - actual_states
    print(f"⚠️  WARNING: Missing states: {missing}")

# Check no old India states
old_states = {"S0", "S1", "S2", "S3", "S4", "S5"}
old_found = old_states.intersection(actual_states)

if len(old_found) == 0:
    print("✅ PASSED: No old India states found")
else:
    print(f"❌ FAILED: Old India states still present: {old_found}")
    raise ValueError("OLD INDIA STATES DETECTED")

# COMMAND ----------

# MAGIC %md
# MAGIC ## TEST 4: MEMBERSHORTNAME Scope

# COMMAND ----------

# DBTITLE 1,Test 4.1 - Check lender_name Column
print("=" * 70)
print("TEST 4: MEMBERSHORTNAME SCOPE")
print("=" * 70)

# Check lender_name exists
if "lender_name" in bureau_trade_df.columns:
    print("✅ PASSED: lender_name column exists")
else:
    print("❌ FAILED: lender_name column missing")
    raise ValueError("LENDER_NAME COLUMN MISSING")

# Check NO lender_id column
if "lender_id" not in bureau_trade_df.columns:
    print("✅ PASSED: No lender_id column (correctly removed)")
else:
    print("❌ FAILED: lender_id column still exists")
    raise ValueError("LENDER_ID COLUMN PRESENT")

# Check lender_name values
lender_stats = bureau_trade_df.select(
    F.count("lender_name").alias("non_null"),
    F.countDistinct("lender_name").alias("unique_lenders")
).collect()[0]

print(f"\n📊 Lender Statistics:")
print(f"   Non-null lender_name: {lender_stats['non_null']:,}")
print(f"   Unique lenders: {lender_stats['unique_lenders']:,}")

# Show top lenders
print("\n🔝 Top 10 Lenders:")
top_lenders = bureau_trade_df.groupBy("lender_name").count() \
    .orderBy(F.desc("count")).limit(10)
display(top_lenders)

# COMMAND ----------

# DBTITLE 1,Test 4.2 - Thai Lender Classification
print("\n🔍 VERIFY THAI LENDER CLASSIFICATION")
print("-" * 50)

from modules.lender_ecology import LenderEcology

lender_engine = LenderEcology(spark)

# Apply lender type mapping
lender_typed = lender_engine._map_lender_types(
    bureau_trade_df.select("lender_name").distinct()
)

# Check distribution
type_dist = lender_typed.groupBy("lender_type").count().orderBy(F.desc("count"))

print("\n📊 Lender Type Distribution:")
display(type_dist)

# Check Thai categories present
thai_categories = {"SFI", "COMMERCIAL_BANK", "PERSONAL_LOAN", "LEASING", "FINTECH", "CARDX"}
actual_types = set([row['lender_type'] for row in type_dist.collect()])

thai_found = thai_categories.intersection(actual_types)
print(f"\n✅ Thai categories found: {thai_found}")

# Check OTHER percentage
other_count = type_dist.filter("lender_type = 'OTHER'").collect()
if len(other_count) > 0:
    other_pct = (other_count[0]['count'] / lender_typed.count() * 100)
    print(f"\n⚠️  OTHER category: {other_pct:.1f}%")
    if other_pct < 10.0:
        print("✅ PASSED: OTHER category < 10%")
    else:
        print(f"⚠️  WARNING: HIGH OTHER category {other_pct:.1f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ## TEST 5: Full Pipeline Test

# COMMAND ----------

# DBTITLE 1,Test 5.1 - Run Full Pipeline
print("=" * 70)
print("TEST 5: FULL PIPELINE")
print("=" * 70)

print(f"\n🚀 Running full pipeline for {TEST_MONTH}...")
print(f"   This may take 5-10 minutes for full month")

pipeline = BehavioralPhysicsPipeline(spark)

# Run pipeline
features_df, audit_log_df, qa_summary = pipeline.run_from_raw_tables(
    as_of_month=TEST_MONTH,
    catalog=CATALOG
)

print(f"\n✅ Pipeline completed successfully!")

# COMMAND ----------

# DBTITLE 1,Test 5.2 - Validate Output Schema
print("\n🔍 OUTPUT SCHEMA VALIDATION")
print("-" * 50)

# Check key columns
key_cols = ["cust_id", "as_of_month"]
for col in key_cols:
    if col in features_df.columns:
        print(f"✅ {col} present")
    else:
        print(f"❌ {col} MISSING")
        raise ValueError(f"MISSING COLUMN: {col}")

# Check feature count
feature_cols = [c for c in features_df.columns if c not in ["cust_id", "as_of_month"]]
print(f"\n📊 Features Generated: {len(feature_cols)}")

if len(feature_cols) >= 150:
    print(f"✅ PASSED: {len(feature_cols)} features >= 150 (expected ~200)")
else:
    print(f"⚠️  WARNING: Only {len(feature_cols)} features (expected ~200)")

# Show sample features
print(f"\n🔍 Sample Features (first 20):")
print(feature_cols[:20])

# COMMAND ----------

# DBTITLE 1,Test 5.3 - Data Quality Checks
print("\n🔍 DATA QUALITY CHECKS")
print("-" * 50)

# Check row count
total_customers = features_df.count()
print(f"   Total customers: {total_customers:,}")

if total_customers > 0:
    print("✅ PASSED: Output has data")
else:
    print("❌ FAILED: Empty output")
    raise ValueError("EMPTY OUTPUT")

# Check for null cust_id
null_cust = features_df.filter(F.col("cust_id").isNull()).count()
if null_cust == 0:
    print("✅ PASSED: No null cust_id")
else:
    print(f"❌ FAILED: {null_cust} null cust_id rows")
    raise ValueError("NULL CUST_ID")

# Check as_of_month
month_check = features_df.select("as_of_month").distinct().collect()
if len(month_check) == 1 and str(month_check[0]['as_of_month']) == TEST_MONTH:
    print(f"✅ PASSED: as_of_month = {TEST_MONTH}")
else:
    print(f"⚠️  WARNING: Unexpected as_of_month values: {month_check}")

# COMMAND ----------

# DBTITLE 1,Test 5.4 - Sample Output
print("\n📋 SAMPLE OUTPUT (First 100 Rows)")
print("-" * 50)

# Select subset of interesting columns
display_cols = ["cust_id", "as_of_month", "bureau_max_dpd", "bureau_state",
                "bureau_regime", "bureau_account_count"]
display_cols = [c for c in display_cols if c in features_df.columns]

display(features_df.select(display_cols).limit(100))

# COMMAND ----------

# MAGIC %md
# MAGIC ## TEST 6: Audit Log Validation

# COMMAND ----------

# DBTITLE 1,Test 6.1 - Check Audit Log
print("=" * 70)
print("TEST 6: AUDIT LOG")
print("=" * 70)

print("\n📊 Audit Log:")
display(audit_log_df)

# Check required fields
required_fields = ["run_id", "as_of_month", "feature_count", "customer_count"]
for field in required_fields:
    if field in audit_log_df.columns:
        print(f"✅ {field} present")
    else:
        print(f"⚠️  {field} missing from audit log")

# COMMAND ----------

# MAGIC %md
# MAGIC ## FINAL SUMMARY

# COMMAND ----------

# DBTITLE 1,Test Summary
print("\n" + "=" * 70)
print("🎯 FINAL TEST SUMMARY")
print("=" * 70)

test_results = {
    "TEST 1: Bridge Join": "✅ PASSED" if fanout_count == 0 else "❌ FAILED",
    "TEST 2: Explicit ref_no": "✅ PASSED" if "ref_no" in bureau_trade_df.columns else "❌ FAILED",
    "TEST 3: Thai DPD States": "✅ PASSED" if len(old_found) == 0 else "❌ FAILED",
    "TEST 4: MEMBERSHORTNAME": "✅ PASSED" if "lender_id" not in bureau_trade_df.columns else "❌ FAILED",
    "TEST 5: Full Pipeline": "✅ PASSED" if total_customers > 0 else "❌ FAILED",
    "TEST 6: Audit Log": "✅ PASSED" if len(audit_log_df.columns) >= 4 else "⚠️  PARTIAL"
}

for test_name, result in test_results.items():
    print(f"\n{test_name}: {result}")

# Count passes
passed = sum(1 for r in test_results.values() if "✅" in r)
total = len(test_results)

print("\n" + "=" * 70)
print(f"OVERALL RESULT: {passed}/{total} tests passed")
print("=" * 70)

if passed == total:
    print("\n🎉 ALL TESTS PASSED - READY FOR PRODUCTION!")
else:
    print(f"\n⚠️  {total - passed} test(s) failed - review failures above")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC
# MAGIC **If all tests passed**:
# MAGIC 1. ✅ Merge PR #3 to main
# MAGIC 2. ✅ Tag release: `v1.0.0-thai-classification`
# MAGIC 3. ✅ Deploy to production
# MAGIC 4. ✅ Run backfill for historical months
# MAGIC
# MAGIC **If tests failed**:
# MAGIC 1. Review failure details above
# MAGIC 2. Fix issues in code
# MAGIC 3. Re-run validation
# MAGIC
# MAGIC **Performance Tuning** (if needed):
# MAGIC - Add `.repartition(200)` before heavy aggregations
# MAGIC - Cache intermediate DataFrames
# MAGIC - Optimize window operations
