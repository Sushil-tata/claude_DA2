# Databricks Testing Guide

## 🚀 Quick Start (5 Minutes)

### Step 1: Upload Notebook to Databricks

**Option A: Using Databricks CLI**
```bash
databricks workspace import \
  behavioral_physics_features/notebooks/Databricks_Sprint_Validation.py \
  /Workspace/behavioral_physics_features/notebooks/Databricks_Sprint_Validation \
  --language PYTHON --format SOURCE
```

**Option B: Manual Upload**
1. Go to Databricks workspace
2. Navigate to Workspace → Users → [your user]
3. Click "Import"
4. Upload `Databricks_Sprint_Validation.py`

### Step 2: Configure Cluster

**Recommended Cluster Configuration**:
- **Runtime**: DBR 13.3 LTS or higher
- **Node type**: Standard_DS3_v2 (or similar)
- **Workers**: 2-4 (autoscaling)
- **Spark Config**:
  ```
  spark.sql.adaptive.enabled true
  spark.sql.adaptive.coalescePartitions.enabled true
  ```

### Step 3: Update Configuration

In the notebook, update these variables (Cell 2):
```python
TEST_MONTH = "2024-12-31"  # Your test month
CATALOG = "cdx_mdz_prd"    # Your catalog name
```

### Step 4: Run Notebook

**Run All Cells**: Cmd/Ctrl + Shift + Enter

**Expected Runtime**: 5-10 minutes for full month

---

## 📋 What Gets Tested

### ✅ TEST 1: Bridge Join Validation
- **Check**: Fan-out = 0 (each ACCT_NUM → exactly 1 REF_NO)
- **Check**: Match rate ~97% (CardX → Bureau mapping)
- **Critical**: Blocks merge if fan-out detected

### ✅ TEST 2: Explicit ref_no Join Keys
- **Check**: `ref_no` column exists (not renamed to cust_id)
- **Check**: No null ref_no after joins
- **Check**: `cust_id` correctly aliased from `ref_no`
- **Check**: (ref_no, seq_tl) uniquely identifies tradelines

### ✅ TEST 3: Thai DPD Classification
- **Check**: DPD values parsed correctly (OVERDUEMONTHS × 30)
- **Check**: Thai states present: CURRENT, SM, NPL, CHARGE_OFF
- **Check**: No old India states (S0-S5)

### ✅ TEST 4: MEMBERSHORTNAME Scope
- **Check**: `lender_name` column exists
- **Check**: NO `lender_id` column (correctly removed)
- **Check**: Thai lender classification works
- **Check**: OTHER category < 10%

### ✅ TEST 5: Full Pipeline
- **Check**: run_from_raw_tables() completes successfully
- **Check**: 150-200 features generated
- **Check**: No null cust_id in output
- **Check**: Correct as_of_month

### ✅ TEST 6: Audit Log
- **Check**: Audit log created with required fields
- **Check**: Run metadata captured

---

## 🎯 Success Criteria

All tests must pass:
```
TEST 1: Bridge Join         ✅ PASSED
TEST 2: Explicit ref_no     ✅ PASSED
TEST 3: Thai DPD States     ✅ PASSED
TEST 4: MEMBERSHORTNAME     ✅ PASSED
TEST 5: Full Pipeline       ✅ PASSED
TEST 6: Audit Log           ✅ PASSED

OVERALL RESULT: 6/6 tests passed
```

**If you see**: `🎉 ALL TESTS PASSED - READY FOR PRODUCTION!`
→ **You can merge PR #3**

---

## ❌ Common Issues & Fixes

### Issue 1: Table Not Found
```
Error: Table 'cdx_mdz_prd.spl_acct_mthly' not found
```

**Fix**: Update `CATALOG` variable to your catalog name
```python
CATALOG = "your_catalog_name"
```

### Issue 2: Date Not Found
```
No data for as_of_month = '2024-12-31'
```

**Fix**: Use a month that exists in your data
```python
# Find available months first
spark.sql(f"SELECT DISTINCT DL_DATA_DT FROM {CATALOG}.spl_acct_mthly ORDER BY DL_DATA_DT DESC").show()
```

### Issue 3: Permission Denied
```
Error: User does not have SELECT privilege on table
```

**Fix**: Grant read access to tables
```sql
GRANT SELECT ON TABLE cdx_mdz_prd.spl_acct_mthly TO `your_user@company.com`;
GRANT SELECT ON TABLE cdx_mdz_prd.spl_ln_orig TO `your_user@company.com`;
GRANT SELECT ON TABLE cdx_mdz_prd.mnf_cra_rvw_id_dummy TO `your_user@company.com`;
GRANT SELECT ON TABLE cdx_mdz_prd.mnf_cra_rvw_s_account TO `your_user@company.com`;
GRANT SELECT ON TABLE cdx_mdz_prd.mnf_cra_rvw_s_history TO `your_user@company.com`;
GRANT SELECT ON TABLE cdx_mdz_prd.mnf_cra_rvw_s_enquiry TO `your_user@company.com`;
```

### Issue 4: Module Import Failed
```
ModuleNotFoundError: No module named 'modules'
```

**Fix**: Upload entire `behavioral_physics_features` folder to Databricks:
```bash
databricks workspace import_dir \
  behavioral_physics_features \
  /Workspace/behavioral_physics_features \
  --overwrite
```

### Issue 5: Out of Memory
```
Error: Container killed by YARN for exceeding memory limits
```

**Fix**: Use larger cluster or reduce sample size
```python
TEST_CUSTOMER_SAMPLE = 500  # Reduce sample
```

---

## 📊 Expected Output

### Console Output
```
🧪 Test Configuration:
   Month: 2024-12-31
   Catalog: cdx_mdz_prd
   Sample: 1000 customers

======================================================================
TEST 1: BRIDGE JOIN VALIDATION
======================================================================

📊 Bridge Statistics:
   Total rows: 3,282,638
   Columns: ['ACCT_NUM', 'REF_NO', 'RECEIVE_DT', 'DL_DATA_DT']

✅ PASSED: No fan-out detected
✅ PASSED: Match rate 97.1% >= 95%

======================================================================
TEST 2: EXPLICIT REF_NO JOIN KEYS
======================================================================

📊 Bureau Data Loaded:
   History rows: 15,234,567
   Account rows: 12,345,678

✅ Bureau trade adapted: 60,913,468 rows
✅ PASSED: ref_no column exists
✅ PASSED: cust_id alias exists
✅ PASSED: cust_id correctly aliased from ref_no

======================================================================
TEST 3: THAI DPD CLASSIFICATION
======================================================================

📊 DPD Statistics:
   Min DPD: 0
   Max DPD: 540
   Avg DPD: 45.23

✅ PASSED: All Thai states present (CURRENT, SM, NPL, CHARGE_OFF)
✅ PASSED: No old India states found

======================================================================
TEST 4: MEMBERSHORTNAME SCOPE
======================================================================

✅ PASSED: lender_name column exists
✅ PASSED: No lender_id column (correctly removed)

📊 Lender Statistics:
   Non-null lender_name: 60,913,468
   Unique lenders: 156

✅ PASSED: OTHER category < 10%

======================================================================
TEST 5: FULL PIPELINE
======================================================================

🚀 Running full pipeline for 2024-12-31...

✅ Pipeline completed successfully!

📊 Features Generated: 198

✅ PASSED: Output has data
✅ PASSED: No null cust_id

======================================================================
TEST 6: AUDIT LOG
======================================================================

✅ run_id present
✅ as_of_month present
✅ feature_count present
✅ customer_count present

======================================================================
🎯 FINAL TEST SUMMARY
======================================================================

TEST 1: Bridge Join: ✅ PASSED
TEST 2: Explicit ref_no: ✅ PASSED
TEST 3: Thai DPD States: ✅ PASSED
TEST 4: MEMBERSHORTNAME: ✅ PASSED
TEST 5: Full Pipeline: ✅ PASSED
TEST 6: Audit Log: ✅ PASSED

======================================================================
OVERALL RESULT: 6/6 tests passed
======================================================================

🎉 ALL TESTS PASSED - READY FOR PRODUCTION!
```

---

## 🔄 What to Do After Testing

### ✅ If All Tests Pass

1. **Merge PR**:
   ```bash
   gh pr merge 3 --squash
   ```

2. **Tag Release**:
   ```bash
   git checkout main
   git pull origin main
   git tag -a v1.0.0 -m "Thai DPD classification + explicit join keys"
   git push origin v1.0.0
   ```

3. **Deploy to Production**:
   - Upload to production Databricks workspace
   - Schedule monthly job
   - Configure alerts

4. **Run Backfill** (optional):
   ```python
   for month in ["2024-10-31", "2024-11-30", "2024-12-31"]:
       pipeline.run_from_raw_tables(
           as_of_month=month,
           catalog="cdx_mdz_prd",
           output_table="behavioral_physics.features_monthly"
       )
   ```

### ❌ If Tests Fail

1. **Review failure details** in notebook output
2. **Fix issues** in code locally
3. **Commit and push** fixes
4. **Re-run validation** notebook
5. **Repeat** until all tests pass

---

## 📞 Support

**Issues**: https://github.com/Sushil-tata/claude_DA2/issues
**PR**: https://github.com/Sushil-tata/claude_DA2/pull/3

---

## 📚 Additional Resources

- **Execution Order**: See `/tmp/execution_order.md` for complete pipeline flow
- **Thai Classification**: See `THAI_LENDER_CLASSIFICATION.md`
- **Bridge Implementation**: See `BRIDGE_JOIN_IMPLEMENTATION.md`
