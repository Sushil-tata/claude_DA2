# Three Fixes Applied - Ready for Execution

## Status: ✅ ALL FIXES COMPLETE

**Final Feature Count**: 377 features (335 base + 41 physics families + 1 new dpd_accel_1m)

---

## FIX 1: Family 1 Dependencies Updated ✅

**File**: `behavioral_physics_features/physics_families.py`

### Change 1.1: credit_momentum_3m (Line 51-56)
**Before**:
```python
# Use dpd_slope_3m (OLS slope from template)
panel = panel.withColumn(
    "credit_momentum_3m",
    F.coalesce(F.col("dpd_slope_3m"), F.lit(0.0)) * F.col("bureau_owed_sum")
)
```

**After**:
```python
# Use dpd_diff_velocity_3m (simple difference velocity from TrajectoryEngine)
panel = panel.withColumn(
    "credit_momentum_3m",
    F.coalesce(F.col("dpd_diff_velocity_3m"), F.lit(0.0)) * F.col("bureau_owed_sum")
)
```

### Change 1.2: momentum_sign_flip_6m (Lines 65-81)
**Before**: Used `dpd_slope_1m` (missing column)

**After**: Uses `dpd_jump` (existing 1-month DPD change)
```python
# Use dpd_jump (1-month DPD change) instead of missing dpd_slope_1m
w6 = w.rowsBetween(-5, 0)
panel = panel.withColumn(
    "dpd_jump_sign",
    F.when(F.col("dpd_jump") > 0, 1)
     .when(F.col("dpd_jump") < 0, -1)
     .otherwise(0)
)
panel = panel.withColumn(
    "momentum_sign_flip_6m",
    F.sum(
        F.when(
            F.col("dpd_jump_sign") != F.lag("dpd_jump_sign", 1).over(w),
            1
        ).otherwise(0)
    ).over(w6)
)
```

### Change 1.3: jerk_dpd_3m (Lines 105-113)
**Before**: Used `dpd_accel_3m`

**After**: Uses `dpd_accel_1m` (single month second difference)
```python
# Use dpd_accel_1m (single month second difference)
panel = panel.withColumn(
    "jerk_dpd_3m",
    F.avg(
        F.coalesce(F.col("dpd_accel_1m"), F.lit(0.0)) -
        F.lag(F.coalesce(F.col("dpd_accel_1m"), F.lit(0.0)), 1).over(w)
    ).over(w.rowsBetween(-2, 0))
)
```

---

## FIX 2: Global Rename dpd_slope → dpd_ols_slope ✅

### Files Modified:

#### 1. `behavioral_physics_features/production_pipeline.py`
**Renamed (replace_all)**:
- `dpd_slope_3m` → `dpd_ols_slope_3m`
- `dpd_slope_6m` → `dpd_ols_slope_6m`
- `dpd_slope_12m` → `dpd_ols_slope_12m`

**Rationale**: Distinguish OLS linear regression slope from simple difference velocity (dpd_diff_velocity)

#### 2. `behavioral_physics_features/modules/trajectory_engine.py`
**Renamed (replace_all)**:
- `dpd_velocity_3m` → `dpd_diff_velocity_3m`
- `dpd_velocity_6m` → `dpd_diff_velocity_6m`

**Rationale**: Clarify that this is simple difference rate, not OLS slope

**Note**: `cardx_dpd_velocity` and `bureau_dpd_velocity` in CardXBureauInteractions are internal computations, not affected by this rename.

---

## FIX 3: Add dpd_accel_1m to Template ✅

**File**: `behavioral_physics_features/production_pipeline.py`

**Location**: Lines 514-518 (after trajectory features loop, before w12 window)

**Code Added**:
```python
# Single-month acceleration (not averaged) for jerk computation
secdiff_1m = (F.col("bureau_max_dpd").cast("double")
              - 2*F.lag(F.col("bureau_max_dpd").cast("double"), 1).over(w)
              + F.lag(F.col("bureau_max_dpd").cast("double"), 2).over(w))
panel = panel.withColumn("dpd_accel_1m", secdiff_1m)
```

**Formula**: `dpd_accel_1m = DPD(t) - 2×DPD(t-1) + DPD(t-2)`

**Purpose**: Single-month second difference (not averaged over window) for jerk computation in Family 1

**Impact**: Adds 1 feature to Bucket B (120 → 121 features)

---

## Header Updates ✅

**File**: `behavioral_physics_features/production_pipeline.py`

### Line 2:
```python
# THREE-BUCKET + 7 PHYSICS FAMILIES ARCHITECTURE (377 features total)
```

### Line 18-22 (Bucket B):
```python
#   BUCKET B (Template): 121 features (+1 from dpd_accel_1m)
#     - Base state features (12)
#     - Bureau aggregations (14)
#     - Lender type counts (8 with Thai classification)
#     - Trajectory features (21: dpd_ols_slope, dpd_accel_1m, shock, deteriorate, improve, stress_frac)
```

---

## Final Feature Architecture

| Bucket | Features | Details |
|--------|----------|---------|
| **Bucket A** (Original Engines) | 128 | StateBuilder (7), TrajectoryEngine (60), LenderEcology (15), EnquiriesEngine (2), CardXBureauInteractions (15), LegalActions (18), TDRRestructuring (22) |
| **Bucket B** (Template) | 121 | Base (12), Aggregations (14), Lender types (8), **Trajectory (21)**, Enquiry (25), Repayment (22), CardX triggers (10), Exposure (15) |
| **Bucket C** (Advanced Physics) | 87 | Momentum & Inertia (15), Energy (14), Thermodynamics (7), Waves (4), Stress (14), Chaos (2), Network (15), Field (7), Phase (4), Relativity (0) |
| **7 Physics Families** | 41 | Inertia & Momentum (7), Critical Slowing (6), Phase Boundary (7), Hysteresis (6), Lender Ecology Topology (5), Enquiry Physics (5), Utilization Physics (5) |
| **TOTAL** | **377** | **128 + 121 + 87 + 41** |

---

## Dependency Verification ✅

All Family 1 dependencies now exist:

| Feature | Dependency | Source | Status |
|---------|------------|--------|--------|
| credit_inertia_score | streak_len, state_num, bureau_owed_sum | Template base | ✅ Exists |
| credit_momentum_3m | **dpd_diff_velocity_3m**, bureau_owed_sum | TrajectoryEngine, Template | ✅ Fixed |
| momentum_sign_flip_6m | **dpd_jump** | Template (line 452) | ✅ Fixed |
| dpd_upward_velocity_3m | dpd_jump | Template (line 452) | ✅ Exists |
| dpd_downward_velocity_3m | dpd_jump | Template (line 452) | ✅ Exists |
| velocity_asymmetry_6m | dpd_upward_velocity_3m, dpd_downward_velocity_3m | Family 1 | ✅ Exists |
| jerk_dpd_3m | **dpd_accel_1m** | Template (new line 514-518) | ✅ Fixed |

---

## No Null Dependencies Confirmed

### Columns Verified:
1. ✅ `dpd_jump` - created in production_pipeline.py line 452
2. ✅ `dpd_diff_velocity_3m` - renamed from dpd_velocity_3m in TrajectoryEngine
3. ✅ `dpd_accel_1m` - newly added in production_pipeline.py line 514-518
4. ✅ `dpd_ols_slope_3m/6m/12m` - renamed from dpd_slope_3m/6m/12m in production_pipeline.py
5. ✅ `bureau_owed_sum`, `streak_len`, `state_num`, `state` - all exist in template base features

### TrajectoryEngine Integration:
- ✅ TrajectoryEngine imported (line 82)
- ✅ TrajectoryEngine initialized (line 714)
- ✅ TrajectoryEngine computed and joined (lines 737-756)
- ✅ Produces dpd_diff_velocity_3m, dpd_diff_velocity_6m

---

## Next Steps

### STEP 5: Quality Report (After Running)
Run on Databricks and generate:
1. Null rate report (% null per feature)
2. Zero variance flags (constant features)
3. Correlation matrix (detect duplicates)
4. Feature count verification (should be exactly 377)

### Execution Command:
```python
# On Databricks
python behavioral_physics_features/production_pipeline.py \
  --dl_data_dt 2024-01-31 \
  --output_table decision_agent.ncb_behavioral_features \
  --cardx_membercodes 12345,67890
```

---

## Files Modified Summary

| File | Changes | Purpose |
|------|---------|---------|
| `behavioral_physics_features/physics_families.py` | Lines 51-56, 65-81, 105-113 | Fix Family 1 dependencies |
| `behavioral_physics_features/production_pipeline.py` | Global rename dpd_slope → dpd_ols_slope | Clarity: OLS vs difference |
| `behavioral_physics_features/production_pipeline.py` | Lines 514-518 (add dpd_accel_1m) | New 1-month acceleration |
| `behavioral_physics_features/production_pipeline.py` | Lines 2, 18-22 (header) | Update feature counts |
| `behavioral_physics_features/modules/trajectory_engine.py` | Global rename dpd_velocity → dpd_diff_velocity | Clarity: difference velocity |

---

**Status**: ✅ **READY FOR EXECUTION**

All 3 fixes applied. No null dependencies. Final count: **377 features**.
