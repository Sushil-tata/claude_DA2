# Using Behavioral Physics with Your Actual Bureau Data

**Date:** 2026-02-09
**Schema:** CIBIL/NCB Thailand Bureau Tables

---

## 📊 Your Bureau Tables

You have **4 bureau tables** in `cdx_mdz_prd.cdx_persist_mnf_res_db`:

1. **mnf_cra_rvw_id_dummy** - Customer identifiers
2. **mnf_cra_rvw_s_account** - Account master (current status + payment history)
3. **mnf_cra_rvw_s_enquiry** - Enquiry data
4. **mnf_cra_rvw_s_history** - Monthly account snapshots ⭐ **KEY TABLE**

---

## 🚀 Quick Start with Your Data

### Step 1: Load Your Bureau Tables

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("BehavioralPhysics").getOrCreate()

# Load your actual bureau tables
bureau_account = spark.table("cdx_mdz_prd.cdx_persist_mnf_res_db.mnf_cra_rvw_s_account")
bureau_history = spark.table("cdx_mdz_prd.cdx_persist_mnf_res_db.mnf_cra_rvw_s_history")
bureau_enquiry = spark.table("cdx_mdz_prd.cdx_persist_mnf_res_db.mnf_cra_rvw_s_enquiry")
```

### Step 2: Use Schema Adapter

```python
from behavioral_physics_features.modules import BureauSchemaAdapter, BehavioralPhysicsPipeline

# Initialize adapter
adapter = BureauSchemaAdapter(spark)

# Adapt bureau schema to behavioral physics format
bureau_trade = adapter.adapt_bureau_trade_data(
    account_df=bureau_account,
    history_df=bureau_history
)

bureau_enquiry_adapted = adapter.adapt_bureau_enquiry_data(
    enquiry_df=bureau_enquiry
)
```

### Step 3: Load CardX Internal Data

```python
# Your CardX internal monthly data (replace with actual table name)
cardx_internal = spark.table("your_catalog.cardx_internal_monthly")

# Expected schema: cust_id, as_of_month, cardx_dpd, cardx_balance, cardx_credit_limit
```

### Step 4: Run Behavioral Physics Pipeline

```python
# Initialize pipeline
pipeline = BehavioralPhysicsPipeline(spark)

# Run feature computation
features_df, audit_log, qa_summary = pipeline.run(
    bureau_trade_df=bureau_trade,           # Adapted bureau data
    bureau_enquiry_df=bureau_enquiry_adapted,  # Adapted enquiry data
    cardx_internal_df=cardx_internal,       # CardX data
    as_of_month="2024-01-31",               # Snapshot date
    output_table="feature_store.behavioral_physics_monthly"  # Output table
)

# Features ready!
print(f"Generated {len(features_df.columns)-2} features for {features_df.count()} customers")
```

---

## 📋 Schema Mappings

### Bureau History Table → Behavioral Physics

| Your Column | → | Expected Column | Transformation |
|-------------|---|-----------------|----------------|
| REF_NO | → | cust_id | Direct mapping |
| ASOFDATE | → | as_of_month | Direct mapping |
| CREDITLIMIT | → | credit_limit | Direct mapping |
| AMOUNTOWED | → | balance | Direct mapping |
| OVERDUEMONTHS | → | dpd | **Bucket to numeric** |

**DPD Bucket Conversion:**
```
"000" → 0 DPD
"001" → 15 DPD (1-30 days, midpoint)
"030" → 45 DPD (31-60 days)
"060" → 75 DPD (61-90 days)
"090" → 120 DPD (91-150 days)
"180" → 270 DPD (181+ days)
```

### Bureau Account Table → Behavioral Physics

| Your Column | → | Expected Column | Notes |
|-------------|---|-----------------|-------|
| REF_NO | → | cust_id | |
| ACCOUNTNUMBER | → | account_id | |
| MEMBERSHORTNAME | → | lender_name | Used for lender ecology |
| ACCOUNTTYPE | → | account_type | |
| DATEACCOUNTOPENED | → | account_open_date | |
| DATEOFLASTDEBTRESTRUCTURE | → | last_tdr_date | **TDR tracking!** |
| PAYMENTHISTORY1/2 | → | (parsed) | Monthly DPD array |

### Bureau Enquiry Table → Behavioral Physics

| Your Column | → | Expected Column |
|-------------|---|-----------------|
| REF_NO | → | cust_id |
| DATEOFENQUIRY | → | enquiry_date |
| ENQUIRYPURPOSE | → | enquiry_purpose |
| MEMBERSHORTNAME | → | lender_name |

---

## 🔬 Advanced: Payment History Parsing

Your bureau data includes **PAYMENTHISTORY1** and **PAYMENTHISTORY2** strings:

**Format:** `"000000030060090"` (each 3 characters = 1 month's DPD bucket)

**Example:**
```
"000000030060" = [000, 000, 030, 060]
             = [0 DPD, 0 DPD, 31-60 DPD, 61-90 DPD]
             = Customer was current for 2 months, then delinquent
```

**To create monthly snapshots from payment history:**

```python
# This explodes the payment history string into monthly records
monthly_snapshots = adapter.create_monthly_snapshots_from_payment_history(
    account_df=bureau_account
)

# Result: One row per (customer, account, month) from payment history
monthly_snapshots.select("cust_id", "as_of_month", "dpd").show()
```

**Use this if:**
- You want to reconstruct historical monthly DPD from payment history strings
- You don't have/trust the mnf_cra_rvw_s_history table

**Don't use this if:**
- mnf_cra_rvw_s_history already provides monthly snapshots (recommended)

---

## 🎯 TDR (Restructuring) Features

Your bureau data includes **DATEOFLASTDEBTRESTRUCTURE** field!

This enables TDR-based features:
- `has_tdr` - Binary flag if customer has restructuring
- `months_since_tdr` - Time since last restructuring
- Can track TDR performance and relapse

**Automatically extracted by adapter** ✅

---

## 📊 Output Features

After running the pipeline with your adapted bureau data, you get **140 features**:

### Sample Output

```
cust_id | as_of_month | dpd_velocity_3m | dpd_acceleration_3m | state_entropy_6m | lender_hhi | ...
--------|-------------|-----------------|---------------------|------------------|------------|----
CUST001 | 2024-01-31  | 5.3            | 2.1                 | 0.85            | 0.42       | ...
CUST002 | 2024-01-31  | -1.2           | -0.5                | 0.42            | 0.68       | ...
```

**Features include:**
- ✅ Velocity/Acceleration from OVERDUEMONTHS history
- ✅ Lender ecology from MEMBERSHORTNAME
- ✅ TDR flags from DATEOFLASTDEBTRESTRUCTURE
- ✅ State transitions from monthly snapshots
- ✅ Enquiry patterns from DATEOFENQUIRY
- ✅ CardX interactions (if CardX data provided)

---

## 🔍 Data Quality Checks

The adapter performs automatic quality checks:

1. **DPD Bucket Validation** - Warns if unknown buckets found
2. **Missing Value Handling** - Null OVERDUEMONTHS → 0 DPD
3. **Date Parsing** - Converts ASOFDATE to proper date format
4. **Column Existence** - Only maps columns that exist in your data

**Check adapter output:**
```python
print("Adapted bureau trade data:")
bureau_trade.printSchema()
bureau_trade.select("cust_id", "as_of_month", "dpd", "balance").show(5)

print(f"Total rows: {bureau_trade.count()}")
print(f"Customers: {bureau_trade.select('cust_id').distinct().count()}")
print(f"Date range: {bureau_trade.select(min('as_of_month'), max('as_of_month')).first()}")
```

---

## ⚠️ Important Notes

### 1. Account Identification

Your history table has `SEQ_TL` (sequence) instead of `ACCOUNTNUMBER`.

**Workaround:**
- Adapter uses history table as primary source (monthly snapshots)
- Joins with account table by `cust_id` to enrich with account attributes
- If you need account-level features, ensure SEQ_TL can be linked to ACCOUNTNUMBER

### 2. Payment History Strings

**PAYMENTHISTORY1** and **PAYMENTHISTORY2** contain rich historical data.

**Current implementation:**
- Extracts max DPD from history string
- Flags if payment history exists

**Advanced implementation available:**
- `create_monthly_snapshots_from_payment_history()` explodes into monthly records
- Use if you need full historical trajectory

### 3. Lender Classification

**MEMBERSHORTNAME** values need to be mapped to lender types.

**Update in config.py:**
```python
# Add your actual lender names
LENDER_TYPE_MAPPING = {
    "PSU_BANK": ["GOVERNMENT SAVINGS BANK", "BANK FOR AGRICULTURE", ...],
    "PRIVATE_BANK": ["BANGKOK BANK", "KASIKORN", "SCB", "KRUNGSRI", ...],
    "FINTECH": ["RABBIT FINANCE", "AEON", "KREDIVO", ...],
    # Add your actual MEMBERSHORTNAME values here
}
```

---

## 🚦 Next Steps

### 1. Test Adapter with Sample Data

```python
# Test on small sample first
sample_account = bureau_account.limit(1000)
sample_history = bureau_history.filter(
    F.col("REF_NO").isin([row.REF_NO for row in sample_account.select("REF_NO").distinct().collect()])
)

test_trade = adapter.adapt_bureau_trade_data(sample_account, sample_history)
test_trade.show(10)
```

### 2. Validate Schema Mappings

```python
# Check if all expected columns exist
expected_cols = ["cust_id", "as_of_month", "dpd", "balance", "credit_limit"]
actual_cols = test_trade.columns

for col in expected_cols:
    if col in actual_cols:
        print(f"✓ {col} exists")
    else:
        print(f"✗ {col} MISSING")
```

### 3. Run Full Pipeline

```python
# Once adapter works, run full behavioral physics pipeline
features_df, audit_log, qa_summary = pipeline.run(
    bureau_trade,
    bureau_enquiry_adapted,
    cardx_internal,
    as_of_month="2024-01-31"
)
```

### 4. Validate Feature Quality

```python
# Check QA summary
print(json.dumps(qa_summary, indent=2, default=str))

# Check feature distributions
for col in ["dpd_velocity_3m", "state_entropy_6m", "lender_hhi"]:
    features_df.select(col).describe().show()
```

---

## 📞 Troubleshooting

### Issue: "Column not found: OVERDUEMONTHS"

**Solution:** Check if history table has this column. If not, update schema adapter.

### Issue: DPD values look wrong

**Solution:** Verify DPD bucket mapping matches your bureau provider's conventions.

### Issue: Too many nulls in features

**Solution:** Check data availability in source tables. Some accounts may not have monthly history.

### Issue: Lender ecology features all zero

**Solution:** Update lender type mapping in config.py with your actual MEMBERSHORTNAME values.

---

## ✅ Success Criteria

Your implementation is working correctly if:

1. ✅ Adapter runs without errors
2. ✅ DPD values are numeric (0-270 range)
3. ✅ Monthly snapshots exist for multiple months
4. ✅ Lender names are populated
5. ✅ TDR flags work (if restructuring data exists)
6. ✅ Feature pipeline produces 140 features
7. ✅ No data leakage (all features use data ≤ as_of_month)

---

**Ready to run on your actual bureau data!** 🚀

---

**Questions?** Review `bureau_schema_adapter.py` for implementation details or run the test example at the bottom of the file.
