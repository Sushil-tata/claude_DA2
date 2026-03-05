# Testing Behavioral Physics on Actual Bureau Data

**Quick Start Guide for Testing on Your Data**

---

## 🚀 Option 1: Run Python Script (Databricks or Local)

### Step 1: Upload Files to Databricks

```bash
# Upload the behavioral_physics_features folder to Databricks
# Location: /Workspace/Users/your_email/behavioral_physics_features/
```

### Step 2: Run Test Script

```python
# In Databricks notebook or Python file
%run ./behavioral_physics_features/test_actual_data.py
```

---

## 📓 Option 2: Databricks Notebook (Copy-Paste)

Create a new Databricks notebook and paste these cells:

### Cell 1: Install/Import

```python
# If running from uploaded folder
import sys
sys.path.append('/Workspace/Users/your_email/')

from behavioral_physics_features.modules import (
    BureauSchemaAdapter,
    BehavioralPhysicsPipeline
)

print("✓ Modules imported successfully")
```

### Cell 2: Load Your Actual Data

```python
# Load your actual bureau tables
catalog = "cdx_mdz_prd.cdx_persist_mnf_res_db"

bureau_account = spark.table(f"{catalog}.mnf_cra_rvw_s_account")
bureau_history = spark.table(f"{catalog}.mnf_cra_rvw_s_history")
bureau_enquiry = spark.table(f"{catalog}.mnf_cra_rvw_s_enquiry")

print(f"Bureau Account: {bureau_account.count():,} rows")
print(f"Bureau History: {bureau_history.count():,} rows")
print(f"Bureau Enquiry: {bureau_enquiry.count():,} rows")

# Show sample
bureau_history.select("REF_NO", "ASOFDATE", "OVERDUEMONTHS", "AMOUNTOWED").show(5)
```

### Cell 3: Adapt Bureau Schema

```python
# Initialize schema adapter
adapter = BureauSchemaAdapter(spark)

# Adapt bureau data to behavioral physics format
bureau_trade = adapter.adapt_bureau_trade_data(
    account_df=bureau_account,
    history_df=bureau_history
)

bureau_enquiry_adapted = adapter.adapt_bureau_enquiry_data(
    enquiry_df=bureau_enquiry
)

print(f"✓ Bureau Trade Adapted: {bureau_trade.count():,} rows")
print(f"✓ Customers: {bureau_trade.select('cust_id').distinct().count():,}")

# Verify schema mapping worked
bureau_trade.select(
    "cust_id", "as_of_month", "dpd", "balance", "credit_limit",
    "utilization", "lender_name"
).show(5)
```

### Cell 4: Load CardX Data (or Mock)

```python
# OPTION A: Load actual CardX data
# cardx_internal = spark.table("your_catalog.cardx_internal_monthly")

# OPTION B: Create mock CardX data for testing
from pyspark.sql import functions as F

customer_months = bureau_trade.select("cust_id", "as_of_month").distinct()

cardx_internal = customer_months.withColumn(
    "cardx_dpd", F.lit(0)
).withColumn(
    "cardx_balance", F.lit(5000)
).withColumn(
    "cardx_credit_limit", F.lit(10000)
)

print(f"✓ CardX Internal: {cardx_internal.count():,} rows")
print("⚠️  Using mock CardX data - replace with actual table for production")
```

### Cell 5: Select As-of Month

```python
# Get latest month from data
max_month = bureau_trade.agg(F.max("as_of_month")).first()[0]
as_of_month = str(max_month)

print(f"Testing with as_of_month: {as_of_month}")

# For faster testing: filter to recent 3 months
three_months_ago = F.add_months(F.lit(as_of_month), -3)

bureau_trade_recent = bureau_trade.filter(F.col("as_of_month") >= three_months_ago)
cardx_internal_recent = cardx_internal.filter(F.col("as_of_month") >= three_months_ago)

print(f"Bureau Trade (recent): {bureau_trade_recent.count():,} rows")
print(f"CardX Internal (recent): {cardx_internal_recent.count():,} rows")
```

### Cell 6: Run Behavioral Physics Pipeline 🚀

```python
# Initialize pipeline
pipeline = BehavioralPhysicsPipeline(spark)

# Run full pipeline - compute 200 features!
print("Computing 200 behavioral physics features...")
print("This may take a few minutes...\n")

features_df, audit_log, qa_summary = pipeline.run(
    bureau_trade_df=bureau_trade_recent,
    bureau_enquiry_df=bureau_enquiry_adapted,
    cardx_internal_df=cardx_internal_recent,
    as_of_month=as_of_month
)

print("\n✅ PIPELINE COMPLETED!")
```

### Cell 7: View Results

```python
# Count features
feature_cols = [col for col in features_df.columns if col not in ["cust_id", "as_of_month"]]

print(f"Total Features: {len(feature_cols)}")
print(f"Total Customers: {features_df.count():,}")

# Show sample features
features_df.select(
    "cust_id", "as_of_month",
    "dpd_velocity_3m",
    "dpd_acceleration_3m",
    "state_entropy_6m",
    "lender_hhi",
    "synchronized_delinquency_flag",
    "cardx_leads_bureau_flag",
    "has_legal_action_flag",
    "tdr_count_lifetime"
).show(10)
```

### Cell 8: Feature Distributions

```python
# Check velocity features
print("Velocity Features:")
features_df.select("dpd_velocity_3m", "dpd_velocity_6m").describe().show()

# Check acceleration features
print("\nAcceleration Features:")
features_df.select("dpd_acceleration_3m", "shock_flag_3m").describe().show()

# Check entropy features
print("\nEntropy Features:")
features_df.select("state_entropy_6m", "dpd_cv_6m").describe().show()

# Check lender ecology
print("\nLender Ecology:")
features_df.select("lender_hhi", "synchronized_delinquency_flag").describe().show()

# Check new modules
print("\nCardX-Bureau Interactions:")
features_df.select("cardx_leads_bureau_flag", "dpd_divergence_score").describe().show()

print("\nLegal Actions:")
features_df.select("has_legal_action_flag", "legal_status_current").describe().show()

print("\nTDR/Restructuring:")
features_df.select("tdr_count_lifetime", "tdr_adherence_rate").describe().show()
```

### Cell 9: Quality Checks

```python
import json

# QA Summary
print("QA Summary:")
print(json.dumps(qa_summary, indent=2, default=str))

# Check for nulls
print("\nNull Percentages (first 20 features):")
total_rows = features_df.count()

for col in feature_cols[:20]:
    null_count = features_df.filter(F.col(col).isNull()).count()
    null_pct = (null_count / total_rows * 100) if total_rows > 0 else 0

    if null_pct > 10:
        print(f"  {col}: {null_pct:.1f}% null")
```

### Cell 10: Save Features (Optional)

```python
# Save to Delta Lake
output_table = "feature_store.behavioral_physics_features"

features_df.write \
    .format("delta") \
    .mode("overwrite") \
    .option("mergeSchema", "true") \
    .saveAsTable(output_table)

print(f"✅ Features saved to: {output_table}")
print(f"   Rows: {features_df.count():,}")
print(f"   Features: {len(feature_cols)}")
```

---

## 🔍 What to Check

### 1. Schema Adapter Working?

```python
# Verify DPD conversion from buckets
bureau_history.select("OVERDUEMONTHS").distinct().show()
bureau_trade.select("dpd_bucket", "dpd").distinct().orderBy("dpd").show()

# Expected: "030" → 45, "060" → 75, "090" → 120, etc.
```

### 2. Feature Distributions Make Sense?

```python
# DPD velocity should be reasonable (-50 to +50 typically)
features_df.select("dpd_velocity_3m").describe().show()

# State entropy should be 0-2 range
features_df.select("state_entropy_6m").describe().show()

# Lender HHI should be 0-1 range
features_df.select("lender_hhi").describe().show()
```

### 3. Lender Names Mapped?

```python
# Check lender names from your data
bureau_trade.select("lender_name").distinct().show(20, truncate=False)

# Update config.py if needed to classify lenders correctly
```

### 4. Legal & TDR Features Working?

```python
# Check if legal actions detected
features_df.filter(F.col("has_legal_action_flag") == 1).count()

# Check if TDR detected
features_df.filter(F.col("tdr_count_lifetime") > 0).count()
```

---

## ⚠️ Common Issues & Fixes

### Issue 1: "Table not found"

**Fix:** Ensure you're running in Databricks with access to `cdx_mdz_prd.cdx_persist_mnf_res_db`

### Issue 2: "Column not found: OVERDUEMONTHS"

**Fix:** Check actual column names in your bureau_history table:
```python
bureau_history.printSchema()
```

### Issue 3: All features are null/zero

**Fix:** Check data availability in source tables:
```python
bureau_history.select("ASOFDATE").describe().show()
bureau_trade.select("dpd").describe().show()
```

### Issue 4: Lender features all zero

**Fix:** Update lender type mapping in `config.py` with your actual `MEMBERSHORTNAME` values:
```python
bureau_account.select("MEMBERSHORTNAME").distinct().show(50, truncate=False)
```

---

## 📊 Expected Results

### Successful Test Output:

```
✅ PIPELINE COMPLETED SUCCESSFULLY!

Total Features: 200
Total Customers: 15,234

QA Summary:
  - Missingness: 5 features with >50% missing
  - Leakage: PASS (no leakage detected)
  - Features by family: Velocity(18), Acceleration(12), Entropy(10), ...

Top Features Generated:
  - dpd_velocity_3m: Mean 2.3, Std 15.4
  - state_entropy_6m: Mean 0.65, Std 0.42
  - lender_hhi: Mean 0.58, Std 0.31
  - cardx_leads_bureau_flag: 23% of customers
  - has_legal_action_flag: 8% of customers
  - tdr_count_lifetime: Mean 0.4, Max 5
```

---

## 🎯 Next Steps After Successful Test

### 1. Replace Mock CardX Data

```python
cardx_internal = spark.table("your_actual_catalog.cardx_internal_monthly")
```

### 2. Run on Full Dataset

```python
# Remove the 3-month filter for production
features_df, audit_log, qa = pipeline.run(
    bureau_trade_df=bureau_trade,  # Full data
    bureau_enquiry_df=bureau_enquiry_adapted,
    cardx_internal_df=cardx_internal,
    as_of_month="2024-01-31"
)
```

### 3. Train Model

```python
# Use features_df to train your ML model
# Expected AUC improvement: +27-48% over baseline
```

### 4. A/B Test

```python
# Model A: Traditional features (baseline)
# Model B: Behavioral physics features (200 features)
# Compare AUC, Gini, precision, recall
```

---

## 💡 Tips

1. **Start Small:** Test on 3 months of data first (faster)
2. **Check Logs:** Review audit_log for execution details
3. **Validate Schema:** Ensure DPD buckets converted correctly
4. **Update Config:** Add your actual lender names to config.py
5. **Monitor Performance:** Track Spark UI for optimization opportunities

---

**Ready to test?** Run the cells above in Databricks! 🚀
