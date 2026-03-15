# Testing XXX (Not Reported) Handling in Payment History

This directory contains tests to validate that "XXX" (not reported) in payment history strings is handled correctly.

---

## 🎯 What Was Fixed

### Before (WRONG ❌):
- `"XXX"` mapped to ordinal `0` (treated as "000" - current)
- No way to detect reporting gaps
- No way to measure reporting quality

### After (CORRECT ✅):
- `"XXX"` maps to `null` for both `dpd` and `dpd_bucket_ordinal`
- Added `has_reporting_gap` = 1 if any "XXX" in payment history
- Added `reporting_gap_count_12m` = count of "XXX" in last 12 months
- Models can now filter/segment based on reporting quality

---

## 🧪 Testing Approaches

### Option 1: Quick SQL Validation (Fastest)

Run in Databricks SQL Editor:

```bash
# In Databricks SQL workspace
# Copy and run: validate_xxx_sql.sql
```

**What it does**:
- ✓ Counts accounts with XXX in payment history
- ✓ Shows sample payment history strings with XXX
- ✓ Checks XXX distribution (start/middle/end)
- ✓ Finds accounts with ALL XXX (complete reporting gap)

**Expected results**:
```sql
-- Should see something like:
total_accounts | accounts_with_xxx | pct_with_xxx
1,234,567     | 234,567          | 19.0

-- Sample strings:
REF001 | ACC001 | "000000030060XXX000000000000000000000000000"
REF002 | ACC002 | "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
```

---

### Option 2: Full Python Test (Comprehensive)

Run in Databricks notebook:

```python
# Upload test_xxx_handling.py to Databricks
# Or run in notebook:

%run ./test_xxx_handling.py

# Or import and run:
from test_xxx_handling import test_xxx_handling
result_df = test_xxx_handling()
```

**What it tests**:
1. ✓ XXX maps to null for `dpd` and `dpd_bucket_ordinal`
2. ✓ `has_reporting_gap` flag is 1 when XXX exists
3. ✓ `reporting_gap_count_12m` counts XXX in last 12 positions
4. ✓ Other buckets (000, 001, 030, 060, 090, 150, 180) still map correctly
5. ✓ Distinction between "000" (current, dpd=0) and "XXX" (not reported, dpd=null)

**Expected output**:
```
================================================================================
TEST: XXX (Not Reported) Handling in Payment History
================================================================================

STEP 1: Loading actual bureau account data...
✓ Loaded bureau accounts with payment history: 1,234,567
✓ Accounts with XXX in payment history: 234,567 (19.0%)

STEP 2: Parsing payment history with XXX handling...
✓ Created 45,678,901 monthly snapshot rows

STEP 3: Validating XXX handling...

[TEST 1] XXX maps to null (not 0):
  Total XXX rows: 3,456,789
  dpd IS NULL: 3,456,789 (100.0%)
  dpd == 0: 0 (0.0%)
  ✅ PASS: All XXX rows have null dpd

  dpd_bucket_ordinal IS NULL: 3,456,789 (100.0%)
  dpd_bucket_ordinal == 0: 0 (0.0%)
  ✅ PASS: All XXX rows have null dpd_bucket_ordinal

[TEST 2] has_reporting_gap flag:
  Accounts with has_reporting_gap=1: 234,567 / 1,234,567
  ✅ PASS: has_reporting_gap correctly matches XXX presence

[TEST 3] reporting_gap_count_12m (last 12 months):
  Distribution of reporting_gap_count_12m:
  +------------------------+------+
  |reporting_gap_count_12m |count |
  +------------------------+------+
  |0                       |900000|
  |1                       |150000|
  |2                       |100000|
  |3                       |50000 |
  |...                     |...   |
  +------------------------+------+

[TEST 4] Other DPD buckets map correctly:
  DPD bucket mapping:
  +----------+---+------------------+------+
  |dpd_bucket|dpd|dpd_bucket_ordinal|count |
  +----------+---+------------------+------+
  |000       |0  |0                 |30M   |
  |001       |15 |1                 |5M    |
  |030       |45 |2                 |3M    |
  |060       |75 |3                 |2M    |
  |090       |120|4                 |1M    |
  |150       |165|5                 |500K  |
  |180       |270|6                 |300K  |
  +----------+---+------------------+------+
  ✅ PASS: Ordinal categories are 0-6 as expected
```

---

## 🔍 What to Look For

### ✅ PASS Criteria

1. **XXX rows have null dpd**:
   ```python
   # Should see:
   dpd IS NULL: 100.0%
   dpd == 0: 0.0%
   ```

2. **XXX rows have null dpd_bucket_ordinal**:
   ```python
   # Should see:
   dpd_bucket_ordinal IS NULL: 100.0%
   dpd_bucket_ordinal == 0: 0.0%
   ```

3. **has_reporting_gap matches XXX presence**:
   ```python
   # If account has any "XXX" in payment history → has_reporting_gap = 1
   # If account has no "XXX" → has_reporting_gap = 0
   ```

4. **reporting_gap_count_12m is accurate**:
   ```python
   # Count "XXX" in last 36 characters (12 months * 3 chars)
   # Example: "...000000030060XXXXXXX000" → count = 3
   ```

5. **Other buckets unchanged**:
   ```python
   # "000" → dpd=0, ordinal=0
   # "030" → dpd=45, ordinal=2
   # "060" → dpd=75, ordinal=3
   # etc.
   ```

### ❌ FAIL Indicators

- XXX rows with `dpd = 0` (should be null)
- XXX rows with `dpd_bucket_ordinal = 0` (should be null)
- `has_reporting_gap = 0` but account has "XXX" in payment history
- `reporting_gap_count_12m` doesn't match actual count of XXX in last 12 positions

---

## 📊 Sample Output Interpretation

### Example: Good Account (No XXX)
```
cust_id  | as_of_month | dpd_bucket | dpd | dpd_bucket_ordinal | has_reporting_gap | reporting_gap_count_12m
---------|-------------|------------|-----|--------------------|--------------------|------------------------
CUST001  | 2024-01-31  | 000        | 0   | 0                  | 0                  | 0
CUST001  | 2023-12-31  | 000        | 0   | 0                  | 0                  | 0
CUST001  | 2023-11-30  | 030        | 45  | 2                  | 0                  | 0
```
✅ Clean reporting history, all months have valid DPD

### Example: Account with XXX (Reporting Gaps)
```
cust_id  | as_of_month | dpd_bucket | dpd  | dpd_bucket_ordinal | has_reporting_gap | reporting_gap_count_12m
---------|-------------|------------|------|--------------------|--------------------|------------------------
CUST002  | 2024-01-31  | 000        | 0    | 0                  | 1                  | 2
CUST002  | 2023-12-31  | XXX        | null | null               | 1                  | 2
CUST002  | 2023-11-30  | XXX        | null | null               | 1                  | 2
CUST002  | 2023-10-31  | 030        | 45   | 2                  | 1                  | 2
```
✅ Account has reporting gaps (XXX), correctly mapped to null

### Example: ALL XXX (Complete Reporting Gap)
```
cust_id  | as_of_month | dpd_bucket | dpd  | dpd_bucket_ordinal | has_reporting_gap | reporting_gap_count_12m
---------|-------------|------------|------|--------------------|--------------------|------------------------
CUST003  | 2024-01-31  | XXX        | null | null               | 1                  | 12
CUST003  | 2023-12-31  | XXX        | null | null               | 1                  | 12
CUST003  | 2023-11-30  | XXX        | null | null               | 1                  | 12
...      | ...         | XXX        | null | null               | 1                  | 12
```
✅ Account has complete reporting gap for last 12 months

---

## 🚨 Troubleshooting

### Issue: No XXX rows found

**Symptoms**: Test shows "No XXX rows found in monthly snapshots"

**Causes**:
1. Payment history parsing not enabled
2. XXX filtered out somewhere
3. Using wrong data source

**Fix**:
```python
# Check if payment history expansion is enabled in bureau_schema_adapter.py
# Line ~191-220 should have the payment history expansion code
```

### Issue: XXX maps to 0 instead of null

**Symptoms**: Test shows `dpd == 0` for XXX rows

**Causes**:
1. Using old version of `_convert_dpd_bucket()`
2. Old code cached

**Fix**:
```python
# Restart Spark session to clear cache
spark.catalog.clearCache()

# Re-import the adapter
from importlib import reload
import behavioral_physics_features.modules.bureau_schema_adapter as bsa
reload(bsa)
```

### Issue: has_reporting_gap always null

**Symptoms**: All accounts have `has_reporting_gap = null`

**Causes**:
1. Feature not added to history table side
2. Schema mismatch in union

**Fix**:
```python
# Check bureau_schema_adapter.py lines 191-193
# Should have:
# .withColumn("has_reporting_gap", F.lit(None).cast("int"))
# .withColumn("reporting_gap_count_12m", F.lit(None).cast("int"))
```

---

## 📈 Expected Performance Impact

### Before (XXX treated as 0):
- ❌ "Not reported" conflated with "Current (0 DPD)"
- ❌ Models couldn't distinguish good vs missing data
- ❌ Scorecard WOE binning distorted by XXX in "000" bin

### After (XXX treated as null):
- ✅ Models can filter low-quality reporting accounts
- ✅ Feature engineering can handle nulls appropriately
- ✅ Scorecard WOE binning uses clean categories
- ✅ `has_reporting_gap` enables reporting quality segmentation
- ✅ `reporting_gap_count_12m` measures recent reporting quality

---

## 🎯 Next Steps After Testing

1. **If tests PASS**:
   - Commit changes to git
   - Run full pipeline test with `test_actual_data.py`
   - Retrain models with corrected XXX handling
   - Compare model performance (expect slight lift from cleaner data)

2. **If tests FAIL**:
   - Check error messages carefully
   - Review the specific test that failed
   - Check code changes in `bureau_schema_adapter.py`
   - Run troubleshooting steps above

3. **Production Deployment**:
   - Update feature documentation to mention XXX handling
   - Add `has_reporting_gap` and `reporting_gap_count_12m` to feature registry
   - Consider filtering accounts with `reporting_gap_count_12m > 6` (>50% missing)
   - Monitor reporting gap distribution over time

---

## 📝 Files in This Test Suite

- **`test_xxx_handling.py`**: Comprehensive Python test (run in Databricks)
- **`validate_xxx_sql.sql`**: Quick SQL validation (run in SQL Editor)
- **`XXX_TESTING_README.md`**: This file

---

## 📞 Support

If tests fail or you see unexpected behavior:
1. Check the troubleshooting section above
2. Review changes in `bureau_schema_adapter.py` lines 93-104, 283-328, 409-438
3. Verify payment history expansion is enabled (line 191-220)
4. Check that history table has null reporting gap columns (line 191-193)

---

**Ready to test!** 🚀

Start with the quick SQL validation, then run the full Python test for comprehensive validation.
