# Databricks Notebook Import Guide

**Notebook**: `Databricks_Validation_Notebook.py`

**Purpose**: Complete validation of all 6 fixes in behavioral physics pipeline

---

## 📥 How to Import into Databricks

### **Method 1: Upload via Workspace UI** (Recommended)

1. **Open Databricks Workspace**
   - Go to your Databricks workspace
   - Navigate to "Workspace" in the left sidebar

2. **Navigate to Destination Folder**
   - Click on "Users" → Your user folder
   - Or navigate to your team's shared folder

3. **Import Notebook**
   - Click the dropdown arrow (⌄) next to your folder
   - Select "Import"
   - Click "Browse" and select: `Databricks_Validation_Notebook.py`
   - Click "Import"

4. **Verify Import**
   - The notebook should appear in your folder
   - Click to open it
   - You should see all cells with %md (Markdown) and Python code

### **Method 2: Import via Databricks CLI**

```bash
# Install Databricks CLI if not already installed
pip install databricks-cli

# Configure authentication
databricks configure --token

# Import notebook
databricks workspace import \
  /path/to/Databricks_Validation_Notebook.py \
  /Users/your.email@company.com/Databricks_Validation_Notebook \
  --language PYTHON --overwrite
```

### **Method 3: Import via Git Sync**

If you have Databricks Repos enabled:

1. **Clone Repository**
   - In Databricks, go to "Repos"
   - Click "Add Repo"
   - Enter: `https://github.com/Sushil-tata/claude_DA2.git`
   - Branch: `recovery_agent_practical`

2. **Navigate to Notebook**
   - Go to: `behavioral_physics_features/Databricks_Validation_Notebook.py`
   - The notebook will automatically sync with git changes

---

## 🚀 How to Run the Notebook

### **Prerequisites**

1. **Cluster Requirements**:
   - Databricks Runtime 13.0+ (or compatible version)
   - Python 3.10+
   - Access to: `cdx_mdz_prd.cdx_persist_mnf_res_db`

2. **Permissions**:
   - Read access to bureau tables (mnf_cra_rvw_s_account, mnf_cra_rvw_s_history, mnf_cra_rvw_s_enquiry)
   - Write access to temp directories (for test outputs)

### **Step-by-Step Execution**

#### **Option A: Run All Cells** (Recommended for first run)

1. **Attach to Cluster**
   - Open the notebook
   - Click "Connect" dropdown in top-right
   - Select your cluster (or create a new one)

2. **Run All**
   - Click "Run All" in the toolbar
   - This will execute all cells sequentially
   - **⏱ Estimated time**: 15-20 minutes

3. **Monitor Progress**
   - Watch each cell execute
   - Check for ✅ (pass) or ❌ (fail) indicators
   - Review output in each section

#### **Option B: Run Cell by Cell** (Recommended for debugging)

1. **Setup Section**
   - Run cells in "Setup: Import Libraries"
   - Verify Spark version displays

2. **Test 1: Lender Classification**
   - Run cells in this section
   - Review lender type distribution
   - Check if OTHER category <5%

3. **Test 2: XXX Handling**
   - Run cells in this section
   - Verify XXX maps to null (100%)
   - Check reporting gap features

4. **Test 3: Integration Test**
   - Run cells in this section
   - This runs the full pipeline
   - Verify all 6 fixes pass

5. **Test 4: Data Quality Checks**
   - Run manual spot checks
   - Review each fix individually

6. **Final Summary**
   - Review checklist
   - Mark items as complete

---

## 📊 What the Notebook Does

### **Section 1: Lender Classification Validation**

**Runs**: `validate_lender_classification.py`

**Validates**:
- Top lenders by customer count
- Lender type distribution (SFI, COMMERCIAL_BANK, etc.)
- OTHER category percentage

**Expected Output**:
```
Top 30 Lenders by Customer Count:
✅ ธนาคารกรุงเทพ จำกัด (มหาชน)     BBL        COMMERCIAL_BANK      50,000
✅ KASIKORNBANK PUBLIC CO...        KBANK      COMMERCIAL_BANK      45,000
...

Distribution by lender type:
✅ COMMERCIAL_BANK              45        250,000    55.0%
✅ SFI                          12         80,000    17.6%
⚠️  OTHER                        5          5,000     1.1%

Quality Checks:
  ✅ PASS         OTHER category <5%
```

### **Section 2: XXX Handling Validation**

**Runs**: `test_xxx_handling.py`

**Validates**:
- XXX maps to null (not 0)
- has_reporting_gap feature
- reporting_gap_count_12m feature

**Expected Output**:
```
[TEST 1] XXX maps to null (not 0):
  dpd IS NULL: 100.0%
  ✅ PASS: All XXX rows have null dpd

[TEST 2] has_reporting_gap flag:
  ✅ PASS: has_reporting_gap correctly matches XXX presence

[TEST 3] reporting_gap_count_12m:
  Distribution shows 0-12 range ✅
```

### **Section 3: Integration Test**

**Runs**: `test_all_fixes.py`

**Validates**:
- All 6 fixes together
- Full pipeline execution
- 200 features generated

**Expected Output**:
```
[TEST 1] Required columns exist:
  ✅ PASS: All 15 required columns present

[TEST 2] RECEIVE_DT point-in-time filter:
  ✅ PASS: No future data leakage

[TEST 3] Payment history expansion:
  ✅ PASS: Payment history expanded by 15,234,567 rows (33.3%)

[TEST 4-7] All other tests:
  ✅ PASS

STEP 3: Testing Full Pipeline Integration...
  ✅ Pipeline completed successfully!
  Features generated: 200
```

### **Section 4: Data Quality Checks**

**Manual spot checks**:
- Schema validation (columns exist)
- Payment history expansion (source distribution)
- XXX distribution (dpd_bucket counts)
- Lender type distribution (lender_id samples)
- DPD bucket ordinal (mapping verification)
- Reporting gap features (distribution checks)
- Point-in-time safety (receive_dt validation)

---

## ✅ Success Criteria

**All tests should PASS**:

| Test | What to Check | Expected Result |
|------|---------------|-----------------|
| **Lender Classification** | OTHER category % | <5% |
| **Lender Classification** | Active categories | ≥4 types |
| **XXX Handling** | dpd IS NULL for XXX | 100% |
| **XXX Handling** | dpd_bucket_ordinal IS NULL for XXX | 100% |
| **Integration** | Required columns | All 15 exist |
| **Integration** | Payment history expansion | >20% of rows |
| **Integration** | Pipeline completion | Success |
| **Integration** | Features generated | 200 |
| **Data Quality** | lender_id column | Exists |
| **Data Quality** | dpd_bucket_ordinal | Values 0-6 |

---

## ❌ Troubleshooting

### **Issue 1: Import Failed**

**Error**: "Could not import notebook"

**Solution**:
- Ensure file is `.py` format (not `.ipynb`)
- Check file isn't corrupted
- Try Method 2 (Databricks CLI)

### **Issue 2: Cluster Won't Start**

**Error**: Cluster startup failed

**Solution**:
- Check cluster has sufficient resources
- Verify Databricks Runtime version ≥13.0
- Try a different cluster

### **Issue 3: Table Not Found**

**Error**: "Table 'cdx_mdz_prd.cdx_persist_mnf_res_db.mnf_cra_rvw_s_account' not found"

**Solution**:
- Verify you have access to the catalog
- Check table names are correct
- Run: `SHOW TABLES IN cdx_mdz_prd.cdx_persist_mnf_res_db`

### **Issue 4: Module Not Found**

**Error**: "No module named 'behavioral_physics_features'"

**Solution**:
- Ensure all files are in the workspace
- Files should be in: `/Repos/your-repo/behavioral_physics_features/`
- Or upload all Python files to the same directory as notebook

### **Issue 5: Test Failures**

**Error**: Some tests show ❌ FAIL

**Solution**:
- **Lender classification fails**: Add missing lenders to `config.py`
- **XXX handling fails**: Check if XXX exists in your data
- **Schema fails**: Verify column names match actual tables
- Review detailed error messages in cell output

---

## 📝 After Running

### **If All Tests Pass** ✅

1. **Review Results**
   - Go through each section's output
   - Verify all ✅ checkmarks
   - Check "Final Summary" section

2. **Create Pull Request**
   ```
   Title: Fix 6 critical issues in behavioral physics pipeline
   Description: All validation tests passed in Databricks
   Link: Attach notebook results or screenshots
   ```

3. **Next Steps**
   - Merge PR
   - Retrain models
   - Measure AUC lift

### **If Tests Fail** ❌

1. **Identify Failed Tests**
   - Note which sections show ❌
   - Copy error messages

2. **Review Documentation**
   - `ALL_FIXES_SUMMARY.md` - Complete fix documentation
   - `THAI_LENDER_CLASSIFICATION.md` - Lender classification guide
   - `XXX_TESTING_README.md` - XXX handling guide

3. **Fix Issues**
   - Update `config.py` for lender classification
   - Verify data quality for other issues

4. **Re-run Tests**
   - Run failed sections again
   - Or run entire notebook again

---

## 📊 Estimated Resource Usage

| Metric | Estimate |
|--------|----------|
| **Runtime** | 15-20 minutes |
| **Cluster Size** | Standard (8GB+ driver, 2+ workers) |
| **Data Processed** | ~50M+ rows (bureau + history) |
| **Memory Peak** | ~10-15GB |
| **Output Size** | ~2-3MB (text logs) |

---

## 🔗 Related Files

- **Notebook**: `Databricks_Validation_Notebook.py`
- **Test Scripts**:
  - `test_all_fixes.py`
  - `test_xxx_handling.py`
  - `validate_lender_classification.py`
- **Documentation**:
  - `ALL_FIXES_SUMMARY.md`
  - `THAI_LENDER_CLASSIFICATION.md`
  - `XXX_TESTING_README.md`
  - `LOCAL_TEST_RESULTS.md`

---

**Ready to import and run!** 🚀

Import the notebook into Databricks and run all cells to validate the fixes.
