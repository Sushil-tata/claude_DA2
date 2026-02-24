# Recovery Agent Practical - Production Audit

## 1) Repo Inventory

| file_path | category | operational_readiness | external dependencies | outputs produced |
|-----------|----------|---------------------|----------------------|------------------|
| `src/recovery_agent_practical/pipeline.py` | PIPELINE | **A runnable today** | features_df (numeric cols only), labels_df (account_id, recovery_amount_180d) | daily_scoring_df, audit_log_df, summary txt |
| `src/recovery_agent_practical/segmentation/persona_builder.py` | SEGMENTATION | **A runnable today** | payment_count_Xd, payment_amt_Xd, call_response_rate, bureau_total_outstanding, wrong_number_flag | persona_df (persona, 4 axis scores, confidence, flags) |
| `src/recovery_agent_practical/scoring/recovery_scorecard_6m.py` | SCORING | **A runnable today** | Numeric features only (GradientBoosting input) | score_df (p_recovery, expected_recovery, score_band) |
| `src/recovery_agent_practical/routing/action_overlay.py` | ROUTING | **A runnable today** | persona, stage, balance, score_band, months_since_chargeoff | action_df (recommended_action, priority_tier, offer_type, reasoning) |
| `src/recovery_agent_practical/outputs/schemas.py` | LOGGING | **A runnable today** | None (schema definitions only) | Dict schemas for daily_scoring, audit_log |
| `src/recovery_agent_practical/example_usage.py` | UTILS | **A runnable today** | None (generates synthetic data) | Console output, demonstrations |
| `decision_agent/features/bureau_features.py` | FEATURES | **A runnable today** | tuef_json (TUEF NCB), card_df (balance, days_past_due) | bureau_features_df (47 features) |
| `decision_agent/features/delinquency_features.py` | FEATURES | **A runnable today** | delinquency_history (DLNQ_HIST string), arrs_period_1-9 | dlnq+arrs features (25 features) |
| `decision_agent/features/collection_action_aggregator.py` | FEATURES | **A runnable today** | T4 actions (action_date, action_category, action_result), T5 settlements (ptp_date, promise_status) | action_features_df (35 features) |
| `decision_agent/labels/roll_rate_labeller.py` | LABELS | **A runnable today** | snapshot_t (T2 card), snapshot_t_plus (T2 card+90d), txn_df (T3 transactions) | labels_df (roll_bucket, outcome_pay_any, outcome_cured) |
| `decision_agent/tdr/data_contract.py` | MONITORING | **A runnable today** | account row (pd.Series) | DataQuality (completeness_pct, confidence_level, warnings) |

**NOT in repo** (referenced but symlinks not created):
- `billing_cycle_features.py` - symlink exists but content unknown
- `collections_feature_pipeline.py` - symlink exists but content unknown
- `pipeline_adapter.py` - symlink exists but content unknown
- `legal_queue_manager.py` - symlink exists but content unknown

---

## 2) Critical Data Contracts

### A. Feature Snapshot Input

**Required by PersonaBuilder + RecoveryScorecard:**

| Column | Type | Available at decision time? | Used by |
|--------|------|----------------------------|---------|
| `account_id` | str | ✅ Always | All |
| `stage` | str (SM/NPL/CHARGEOFF) | ✅ Always | PersonaBuilder, ActionRouter |
| `balance` | float | ✅ Always | ActionRouter (balance_band) |
| `days_past_due` | int | ✅ Always | BureauFeatures, Routing |
| `months_since_chargeoff` | int | ✅ Always | ActionRouter |
| `payment_count_30d` | int | ✅ Point-in-time safe | PersonaBuilder (payment axis) |
| `payment_count_90d` | int | ✅ Point-in-time safe | PersonaBuilder |
| `payment_count_180d` | int | ✅ Point-in-time safe | PersonaBuilder |
| `payment_count_365d` | int | ✅ Point-in-time safe | PersonaBuilder |
| `payment_amt_30d` | float | ✅ Point-in-time safe | PersonaBuilder |
| `payment_amt_90d` | float | ✅ Point-in-time safe | PersonaBuilder |
| `payment_amt_180d` | float | ✅ Point-in-time safe | PersonaBuilder |
| `payment_amt_365d` | float | ✅ Point-in-time safe | PersonaBuilder |
| `days_since_last_payment` | int | ✅ Point-in-time safe | PersonaBuilder |
| `last_payment_amount` | float | ✅ Point-in-time safe | PersonaBuilder |

### B. Actions/Contact Input (T4)

| Column | Type | Available at decision time? | Used by |
|--------|------|----------------------------|---------|
| `action_account_id` | str | ✅ Historical only | CollectionActionAggregator |
| `action_date` | date | ✅ Historical only | Window filters (30d/90d) |
| `action_category` | str (CALL/SMS/EMAIL) | ✅ Historical | Channel counting |
| `action_result` | str (CONNECTED/NO_ANSWER) | ✅ Historical | Response rates |
| `action_actor` | str | ✅ Historical | Agent continuity hash |

**⚠️ LEAKAGE RISK**: Must filter `action_date <= snapshot_date` before aggregation.

### C. Payments Input (T3)

| Column | Type | Available at decision time? | Used by |
|--------|------|----------------------------|---------|
| `account_id` | str | ✅ Historical | Payment aggregation |
| `payment_date` | date | ✅ Historical | Window filters |
| `payment_amount` | float | ✅ Historical | Sums by window |

**⚠️ LEAKAGE RISK**: Payment windows (30d/90d/180d/365d) must be computed as `payment_date < snapshot_date AND payment_date >= snapshot_date - window_days`.

### D. Bureau Input (NCB TUEF)

| Column | Type | Available at decision time? | Used by |
|--------|------|----------------------------|---------|
| `bureau_total_outstanding` | float | ⚠️ Stale (last bureau pull) | Capacity score |
| `bureau_monthly_instalment` | float | ⚠️ Stale | Capacity score |
| `bureau_secured_loan_flag` | bool | ⚠️ Stale | Capacity score |
| `bureau_delinquent_other` | bool | ⚠️ Stale | Strategic defaulter detection |
| `bureau_active_loan_count` | int | ⚠️ Stale | Capacity score |

**⚠️ STALENESS RISK**: Bureau data may be 30-90 days old. PersonaBuilder uses `bureau_available_flag=1` check.

### E. Labels/Outcomes Table

| Column | Type | Window | How computed |
|--------|------|--------|-------------|
| `account_id` | str | N/A | Join key |
| `recovery_amount_180d` | float | T+180 days | Sum of payments in (T, T+180] |
| `roll_bucket_t` | int (0-6) | At T | dpd_to_bucket(days_past_due, stage) |
| `roll_bucket_horizon` | int (0-6) | At T+90 | dpd_to_bucket from snapshot_t_plus |
| `outcome_pay_any_90d` | binary | T+90 | 1 if any payment_amount >= threshold |
| `outcome_pay_amount_90d` | float | T+90 | Sum of payments in (T, T+90] |
| `outcome_cured` | binary | T+90 | 1 if roll_bucket_horizon == 0 and roll_bucket_t > 0 |
| `outcome_chargeoff` | binary | T+90 | 1 if roll_bucket_horizon == 6 |

---

## 3) Windowing & Snapshot Audit

### PersonaBuilder Windowing

**File**: `src/recovery_agent_practical/segmentation/persona_builder.py`

**Payment Behavior Axis** (lines 78-95):
```python
def _calculate_payment_behavior(self, account: pd.Series) -> float:
    recent_payments  = account.get("payment_count_30d", 0)
    short_payments   = account.get("payment_count_90d", 0)
    medium_payments  = account.get("payment_count_180d", 0)
    long_payments    = account.get("payment_count_365d", 0)

    # Weighted average: recent counts more
    score = (
        recent_payments  * 10.0 +
        short_payments   * 5.0  +
        medium_payments  * 3.0  +
        long_payments    * 1.0
    ) / 19.0
```

**Lookback windows**:
- 0-30d, 31-90d, 91-180d, 181-365d
- **Assumed point-in-time safe**: Relies on upstream feature pipeline to compute these windows relative to `snapshot_date`.

**⚠️ LEAKAGE RISK**:
- **PersonaBuilder does NOT compute windows itself** - it expects `payment_count_Xd` columns from upstream.
- **No snapshot_date parameter passed to PersonaBuilder** - assumes features are already windowed correctly.

**Critical Missing**:
- PersonaBuilder has **no verification** that payment features are point-in-time safe.
- Recommendation: **Add snapshot_date parameter to assign_batch() for audit trail**.

---

### CollectionActionAggregator Windowing

**File**: `decision_agent/features/collection_action_aggregator.py`

**Window computation** (lines 299-310):
```python
def _aggregate_actions(self, df: pd.DataFrame, snap: date) -> Dict:
    cutoff_30d  = snap - timedelta(days=30)
    cutoff_90d  = snap - timedelta(days=90)
    cutoff_7d   = snap - timedelta(days=7)

    in_30d  = date_ser.apply(lambda d: d >= cutoff_30d and d <= snap)
    in_90d  = date_ser.apply(lambda d: d >= cutoff_90d and d <= snap)
    in_7d   = date_ser.apply(lambda d: d >= cutoff_7d and d <= snap)

    f["total_contacts_30d"] = int(in_30d.sum())
    f["total_contacts_90d"] = int(in_90d.sum())
```

**✅ Point-in-time safe**:
- `snapshot_date` explicitly passed as `snap` parameter.
- All windows bounded by `d <= snap` (no forward leakage).
- Aggregates **before** snapshot only.

**⚠️ Edge case**:
- If `action_date` > `snap`, row is **excluded** from all windows (correct behavior).

---

### DelinquencyFeatures Windowing

**File**: `decision_agent/features/delinquency_features.py`

**DLNQ_HIST parser** (lines 91-107):
```python
def parse(self, hist: Optional[str]) -> List[int]:
    buckets = []
    for ch in reversed(hist.strip()):   # reverse: rightmost = most recent
        if ch in NULL_CHARS:
            buckets.append(-1)
        else:
            buckets.append(BUCKET_MAP.get(ch.upper(), 0))
    return buckets  # index 0 = most recent month
```

**Window features** (lines 128-142):
```python
last12 = valid[:12]   # First 12 months from most recent
last24 = valid[:24]

feats["dlnq_max_bucket_12m"]   = float(max(last12))
feats["dlnq_times_30plus_12m"] = float(sum(1 for b in last12 if b >= 1))
```

**✅ Point-in-time safe**:
- DLNQ_HIST string is **historical by design** (each character = one month backward).
- Most recent = rightmost character = index 0 after reversal.
- No forward information possible.

**Arrears progression** (lines 210-284):
```python
periods: List[float] = []
for col in ARRS_COLS:  # arrs_period_1 ... arrs_period_9
    val = account.get(col)
    periods.append(float(val))

feats["arrs_trend_3m"] = p[0] - p[2]  # period_1 - period_3
```

**✅ Point-in-time safe**:
- `arrs_period_1` = **most recent** period (by convention).
- Trend = recent - older (no forward data).

**⚠️ ASSUMPTION RISK**:
- Assumes `arrs_period_1` is truly the **snapshot month** - not verified in code.
- If period columns are incorrectly aligned (e.g., period_1 = future), **leakage**.

---

### RollRateLabeller Windowing

**File**: `decision_agent/labels/roll_rate_labeller.py`

**Label generation** (lines 210-265):
```python
def compute_labels(
    self,
    snapshot_t:      pd.DataFrame,  # T2 at training snapshot T
    snapshot_t_plus: pd.DataFrame,  # T2 at T + outcome_window_days
    txn_df:          pd.DataFrame,  # T3 between T and T+N
):
    labels = self._compute_roll_buckets(snapshot_t, snapshot_t_plus)
    payment_labels = self._compute_payment_labels_from_txn(txn_df, snapshot_t)
```

**Payment labelling** (lines 506-563):
```python
def _compute_payment_labels_from_txn(self, txn_df, snapshot_t):
    # Determine snapshot date per account from T2
    snap_dates = snapshot_t[[account_id_col, data_date_col]]
    txn = txn.merge(snap_dates, on=account_id_col, how="left")

    for w in outcome_windows:  # [30, 60, 90]
        cutoff = txn[data_date_col] + pd.Timedelta(days=w)
        window_txn = txn[txn[txn_date_col] <= cutoff]

        agg = window_txn.groupby(account_id_col)[txn_amount_col].sum()
        results[f"outcome_pay_any_{w}d"] = (agg >= min_payment_threshold)
```

**✅ Point-in-time safe**:
- Uses **two snapshots** (T and T+N) to ensure labels computed forward from T.
- `txn_df` filtered to transactions **after** `snapshot_t.data_date`.
- Payment windows bounded by `txn_date <= snapshot_date + window_days`.

**⚠️ LEAKAGE RISK**:
- If `txn_df` passed contains transactions **before** snapshot_t, they are **NOT filtered out** (lines 509-529).
- Code assumes caller pre-filtered txn_df to `txn_date > snapshot_t.data_date`.
- **Recommendation**: Add explicit filter `txn_df = txn_df[txn_df[txn_date_col] > txn[data_date_col]]` before aggregation.

---

## 4) Label Generation Audit

### Anchor Date Per Account

**File**: `decision_agent/labels/roll_rate_labeller.py` (lines 514-519)

```python
if data_date_col in snapshot_t.columns:
    snap_dates = snapshot_t[[account_id_col, data_date_col]]
    snap_dates[data_date_col] = pd.to_datetime(snap_dates[data_date_col])
    txn = txn.merge(snap_dates, on=account_id_col, how="left")
```

**Anchor date**: `snapshot_t.data_date` per account.

**Per-account variability**:
- If `data_date` differs across accounts in same snapshot (e.g., accounts added on different days), each has different anchor.
- **This is CORRECT** for point-in-time safety.

---

### Forward 180-Day Window Calculation

**Pseudocode**:
```
FOR each account_id:
    anchor_date = snapshot_t[account_id].data_date

    FOR window in [30, 60, 90, 180]:
        horizon_date = anchor_date + window

        payments_in_window = SUM(
            txn_df WHERE account_id=account_id
                     AND txn_date > anchor_date
                     AND txn_date <= horizon_date
        ).transaction_amount

        outcome_pay_any_{window}d = 1 IF payments_in_window >= min_payment_threshold ELSE 0
        outcome_pay_amount_{window}d = payments_in_window
```

**Actual implementation** (lines 536-560):
```python
for w in outcome_windows:
    cutoff = txn[data_date_col] + pd.Timedelta(days=w)
    window_txn = txn[txn[txn_date_col] <= cutoff]

    agg = window_txn.groupby(account_id_col)[txn_amount_col].agg(total_paid="sum")

    results[f"outcome_pay_any_{w}d"] = (agg["total_paid"] >= min_payment_to_count)
    results[f"outcome_pay_amount_{w}d"] = agg["total_paid"]
```

**✅ Correct**: Computes per-account windows relative to each account's anchor date.

---

### Partial Payments Treatment

**Partial payments**: Treated as **sum of all payments** in window (line 548).

**NOT treated as event**: Binary flag `outcome_pay_any` set to 1 if **total sum** >= threshold (line 557).

**Implication**:
- Multiple small payments (e.g., 3 × 500 THB) count as **one "paid" event** if sum >= 500 THB.
- Single large payment (e.g., 5000 THB) also counts as one "paid" event.
- **Model sees amount, not frequency** - this is correct for recovery amount prediction.

---

### Exits Treatment (Sold/Legal/Closed)

**Roll bucket logic** (lines 466-499):
```python
def _compute_roll_buckets(snap_t, snap_t_plus):
    h = snap_t_plus[[account_id_col, dpd_col, stage_col]]
    h["roll_bucket_horizon"] = [dpd_to_bucket(d, s) for d, s in zip(...)]

    merged = t.merge(h, on=account_id_col, how="left")

    # Accounts that disappeared from T+N snapshot are assumed charged off
    merged["roll_bucket_horizon"] = merged["roll_bucket_horizon"].fillna(CHARGEOFF_BUCKET)
```

**Exit treatment**:
- Accounts **missing** from `snapshot_t_plus` → assigned `roll_bucket_horizon = 6` (CHARGEOFF).
- This is **correct** for debt sold / written off / closed accounts.

**⚠️ Implication**:
- If account closed due to **full payment**, it still gets `roll_bucket_horizon = 6` (CHARGEOFF bucket).
- **This is WRONG**.

**Red Flag**:
- Need to check `stage` at T+N:
  - If `stage == "CLOSED"` and `balance == 0` → `roll_bucket_horizon = 0` (CURED).
  - If `stage == "CHARGEOFF"` or missing → `roll_bucket_horizon = 6`.

**Current code conflates**:
- Paid-in-full closures (should = cured)
- Debt sales / writeoffs (should = chargeoff)

**Recommendation**: Add stage="CLOSED" check before assigning CHARGEOFF_BUCKET.

---

### Label Definitions (Plain English)

1. **outcome_pay_any_{window}d**: Did customer make **any payment >= 500 THB** in the {window} days after snapshot?
2. **outcome_pay_amount_{window}d**: **Total amount paid** in the {window} days after snapshot.
3. **roll_bucket_t**: DPD bucket (0-6) at snapshot date T.
4. **roll_bucket_horizon**: DPD bucket (0-6) at T + primary_window_days (default 90d).
5. **roll_buckets**: Signed delta = `roll_bucket_horizon - roll_bucket_t` (negative = improved).
6. **roll_direction**: CURE (-2 or less), IMPROVE (-1), STABLE (0), WORSEN (+1), CHARGEOFF (+2+).
7. **outcome_cured**: 1 if account went from DPD > 0 to DPD = 0 within primary window.
8. **outcome_chargeoff**: 1 if account reached chargeoff bucket within primary window.

---

### Red Flags

1. **⚠️ Closed accounts assumed chargeoff**: Accounts missing from snapshot_t_plus assigned bucket=6 without checking if paid-in-full.
2. **⚠️ No explicit txn_date filter**: Assumes caller pre-filtered txn_df to dates > snapshot_t. Should add defensive filter.
3. **⚠️ No verification of data_date uniqueness**: If multiple rows per account in snapshot_t with different data_dates, merges will duplicate.

---

## 5) Persona / Segmentation Implementation Audit

### PersonaBuilder

**File**: `src/recovery_agent_practical/segmentation/persona_builder.py`

### Axis Scores with Formulas

**1. Payment Behavior Axis** (lines 78-95):
```python
recent  = payment_count_30d  * 10.0
short   = payment_count_90d  * 5.0
medium  = payment_count_180d * 3.0
long    = payment_count_365d * 1.0

weighted_score = (recent + short + medium + long) / 19.0  # Max = 19

recent_amt  = payment_amt_30d  * 10.0
short_amt   = payment_amt_90d  * 5.0
medium_amt  = payment_amt_180d * 3.0
long_amt    = payment_amt_365d * 1.0

weighted_amt = (recent_amt + short_amt + medium_amt + long_amt) / 19.0

payment_behavior_score = (weighted_score + weighted_amt) / 2
```

**Required columns**: `payment_count_Xd`, `payment_amt_Xd`, `days_since_last_payment`, `last_payment_amount`

**Scaling**: 0-100 (normalized by max observed in batch).

---

**2. Engagement Axis** (lines 97-126):
```python
call_response_rate = calls_connected / outbound_calls_made  # 0-1
sms_response_rate  = sms_responded / sms_sent

ptp_kept_rate = ptp_kept / ptp_made

recency_score = max(0, 1 - days_since_last_contact / 365)

engagement_score = (
    call_response_rate * 40 +
    sms_response_rate  * 30 +
    ptp_kept_rate      * 20 +
    recency_score      * 10
)
```

**Required columns**: `outbound_calls_made`, `calls_connected`, `sms_sent`, `sms_responded`, `ptp_made`, `ptp_kept`, `days_since_last_contact`

**Scaling**: 0-100.

---

**3. Capacity Axis** (lines 128-157):
```python
if bureau_total_outstanding > 0:
    bureau_score = min(
        100,
        (balance / (bureau_total_outstanding + balance)) * 100
    )
else:
    bureau_score = 0

payment_history_score = min(100, (total_paid_12m / (balance * 1.2)) * 100)

capacity_score = (
    bureau_score * 0.6 +
    payment_history_score * 0.4
)
```

**Required columns**: `bureau_total_outstanding`, `balance`, `total_paid_12m`

**Scaling**: 0-100.

---

**4. Avoidance Axis** (lines 159-185):
```python
wrong_number_flag   = 1 if wrong_number else 0
dispute_flag        = 1 if dispute_raised else 0
lawyer_mentioned    = 1 if lawyer_contact else 0
sms_opt_out         = 1 if opted_out else 0

avoidance_score = (
    wrong_number_flag * 40 +
    dispute_flag      * 30 +
    lawyer_mentioned  * 20 +
    sms_opt_out       * 10
)
```

**Required columns**: `wrong_number_flag`, `dispute_flag`, `lawyer_mentioned`, `sms_opt_out`

**Scaling**: 0-100.

---

### Decision Rules / Tree

**File**: `src/recovery_agent_practical/segmentation/persona_builder.py` (lines 187-244)

```python
def _apply_decision_tree(payment, engagement, capacity, avoidance):
    # Rule 1: Active Payer
    if payment >= 60 and engagement >= 50:
        return "ACTIVE_PAYER", "HIGH"

    # Rule 2: Selective Defaulter
    if capacity >= 60 and payment < 40 and avoidance < 30:
        if bureau_delinquent_other == False:  # Not delinquent elsewhere
            return "SELECTIVE_DEFAULTER", "HIGH"

    # Rule 3: Liquidity Constrained
    if capacity < 40 and payment >= 20 and engagement >= 30:
        return "LIQUIDITY_CONSTRAINED", "MEDIUM"

    # Rule 4: Strategic
    if avoidance >= 60:
        return "STRATEGIC", "HIGH"

    # Rule 5: Dormant (fallback)
    return "DORMANT", "MEDIUM"
```

**Decision tree**:
```
                        ROOT
                          |
            ┌─────────────┴──────────────┐
      payment >= 60?                      NO
            |                              |
         YES + engagement>=50?       capacity >= 60?
            |                              |
       ACTIVE_PAYER              YES + payment<40 + avoidance<30 + !delinq_other
                                         |
                                  SELECTIVE_DEFAULTER
                                         |
                                   NO → capacity < 40?
                                         |
                                   YES + payment>=20 + engagement>=30
                                         |
                                  LIQUIDITY_CONSTRAINED
                                         |
                                   NO → avoidance >= 60?
                                         |
                                      STRATEGIC
                                         |
                                   NO → DORMANT
```

---

### Confidence Computation

**File**: `src/recovery_agent_practical/segmentation/persona_builder.py` (lines 246-267)

```python
def _determine_confidence(account: pd.Series) -> str:
    required_fields = [
        "payment_count_365d", "payment_amt_365d",
        "outbound_calls_made", "calls_connected",
        "bureau_total_outstanding", "ptp_made"
    ]

    missing = sum(1 for f in required_fields if account.get(f, 0) == 0)

    if missing == 0:
        return "HIGH"
    elif missing <= 2:
        return "MEDIUM"
    else:
        return "LOW"
```

**Confidence levels**:
- **HIGH**: All 6 key fields present (no zeros).
- **MEDIUM**: 1-2 missing fields.
- **LOW**: 3+ missing fields.

---

### Missing Value Handling Strategy

**Strategy**: **Zero-fill with flagging**.

**Implementation** (lines 70-76):
```python
def assign_persona(account: pd.Series) -> PersonaAssignment:
    # Fill missing numerics with 0
    account = account.fillna(0)

    # Compute axis scores (zeros reduce scores naturally)
    payment_score = self._calculate_payment_behavior(account)
    ...
```

**Flags** (lines 269-285):
```python
flags = []
if account.get("bureau_total_outstanding", 0) == 0:
    flags.append("NO_BUREAU_DATA")
if account.get("outbound_calls_made", 0) == 0:
    flags.append("NO_CONTACT_HISTORY")
if account.get("payment_count_365d", 0) == 0:
    flags.append("NO_PAYMENT_HISTORY")
```

**⚠️ Limitation**:
- Cannot distinguish **truly zero** (e.g., no payments made) from **missing data**.
- Zero-fill biases scores downward for accounts with incomplete data.
- **Confidence level** partially compensates but doesn't adjust scores.

---

### TODOs / Placeholders

**Search results**: None found (`TODO`, `FIXME`, `NotImplementedError`, `pass` in decision logic).

**Status**: ✅ No placeholders in production path.

---

## 6) Propensity / Recovery Scorecard Audit

### RecoveryScorecard6M

**File**: `src/recovery_agent_practical/scoring/recovery_scorecard_6m.py`

### Target Definition

**Target** (line 180):
```python
y_binary = (y_recovery_amount > self.payment_threshold).astype(int)
```

**Definition**:
- Binary: Did customer pay **>= 500 THB** (configurable) in next 180 days?
- Continuous: Total amount paid in next 180 days.

**Payment threshold**: Default 500 THB (line 51).

**Horizon**: 180 days (6 months) - hardcoded in class name, not configurable.

---

### Two-Part Model Implementation

**File**: `src/recovery_agent_practical/scoring/recovery_scorecard_6m.py` (lines 170-252)

**Part 1: P(pay) - GradientBoostingClassifier**:
```python
def fit(X: pd.DataFrame, y_recovery_amount: pd.Series):
    y_binary = (y_recovery_amount > threshold).astype(int)

    self.classifier = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )

    self.classifier.fit(X, y_binary, sample_weight=None)
```

**Part 2: E(amount | paid) - GradientBoostingRegressor**:
```python
payer_mask = (y_binary == 1)
X_payers = X[payer_mask]
y_amounts = y_recovery_amount[payer_mask]

self.regressor = GradientBoostingRegressor(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)

self.regressor.fit(X_payers, y_amounts)
```

**Prediction**:
```python
def predict(X: pd.DataFrame):
    p_pay = self.classifier.predict_proba(X)[:, 1]
    e_amount = self.regressor.predict(X)

    expected_recovery = p_pay * e_amount
    return p_pay, e_amount, expected_recovery
```

**✅ Correctly implemented**:
- Two stages trained independently.
- Regressor trained **only on payers** (y_binary == 1).
- Final prediction multiplies probabilities.

**⚠️ Missing**:
- No **sample weighting** by balance or recency (commented out line 189).
- No **cross-validation** within fit() - assumes caller does train/val split.

---

### How X and y Are Joined

**File**: `src/recovery_agent_practical/pipeline.py` (lines 102-114)

```python
def train_scorecard(features_df, labels_df, val_split=0.2):
    # Merge features with labels
    train_data = features_df.merge(labels_df, on="account_id", how="inner")

    if len(train_data) == 0:
        raise ValueError("No matching accounts between features and labels")

    # Extract numeric features only
    numeric_cols = train_data.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ["account_id", "recovery_amount_180d"]]

    X = train_data[numeric_cols]
    y = train_data["recovery_amount_180d"]
```

**Join key**: `account_id` (inner join).

**⚠️ Leakage risk**:
- If `features_df` contains **any columns** computed using data after snapshot_date, they will be included in numeric_cols.
- **No explicit check** that features are point-in-time safe.

**✅ Safe guard**:
- Only **numeric columns** used (line 109) - strings like "stage" excluded.
- This accidentally prevents some leakage (e.g., string columns from future).

---

### Validation Split Method

**File**: `src/recovery_agent_practical/pipeline.py` (lines 118-121)

```python
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=val_split, random_state=42
)
```

**Split method**: **Random split** (NOT time-based).

**⚠️ CRITICAL RISK**:
- Random split **violates temporal ordering**.
- Training data may contain accounts from **later dates** than validation data.
- If features change over time (e.g., DPD distribution shifts), model overfits to later data.

**Recommendation**: Use **time-based split** on `snapshot_date`:
```python
# Instead of random split:
train_mask = features_df["snapshot_date"] < cutoff_date
X_train = features_df[train_mask][numeric_cols]
X_val   = features_df[~train_mask][numeric_cols]
```

---

### Calibration Utilities

**Search results**: None found (no `calibration`, `isotonic`, `platt`, `CalibratedClassifierCV`).

**Status**: ❌ **No calibration implemented**.

**Implication**:
- P(pay) from GradientBoostingClassifier **may not be well-calibrated**.
- Expected recovery amounts may be systematically biased.

**Recommendation**: Add `CalibratedClassifierCV` wrapper around classifier.

---

### Gaps Summary

| Gap | Severity | Impact |
|-----|----------|--------|
| Random train/val split (not temporal) | **CRITICAL** | Overfitting to future data distribution |
| No calibration of P(pay) | **HIGH** | Biased recovery predictions |
| No sample weighting by balance/recency | **MEDIUM** | Treats all accounts equally (small/large) |
| No explicit leakage check on X columns | **MEDIUM** | Could use future features if present in features_df |
| Hardcoded 180d horizon | **LOW** | Not flexible for different outcome windows |

---

## 7) Routing / Overlay Audit

### ActionOverlayRouter

**File**: `src/recovery_agent_practical/routing/action_overlay.py`

### Routing Inputs Required

**File**: `src/recovery_agent_practical/routing/action_overlay.py` (lines 86-103)

```python
def route_action(
    persona: str,
    stage: str,
    balance: float,
    score_band: str,
    months_since_chargeoff: int,
    has_secured_assets: bool = False,
    bureau_delinquent_other: bool = False
) -> ActionRecommendation
```

**Required inputs**:
| Input | Type | Source |
|-------|------|--------|
| `persona` | str (ACTIVE_PAYER/SELECTIVE_DEFAULTER/LIQUIDITY_CONSTRAINED/STRATEGIC/DORMANT) | PersonaBuilder |
| `stage` | str (SM/NPL/CHARGEOFF) | Features |
| `balance` | float | Features |
| `score_band` | str (HOT/WARM/COLD/FROZEN) | RecoveryScorecard6M |
| `months_since_chargeoff` | int | Features (computed from chargeoff_date) |
| `has_secured_assets` | bool | BureauFeatures (bureau_secured_loan_flag) |
| `bureau_delinquent_other` | bool | BureauFeatures |

---

### Rule Table / Mapping Logic

**Balance band classification** (lines 54-63):
```python
def _classify_balance_band(balance: float) -> str:
    if balance < 50_000:
        return "SMALL"
    elif balance < 200_000:
        return "MEDIUM"
    else:
        return "LARGE"
```

**Routing rules** (lines 105-231, condensed):
```python
# ACTIVE_PAYER
if persona == "ACTIVE_PAYER":
    if score_band in ["HOT", "WARM"]:
        return SETTLEMENT_PLAN, TIER_1
    else:
        return AGENCY, TIER_2

# SELECTIVE_DEFAULTER
if persona == "SELECTIVE_DEFAULTER":
    if balance_band == "LARGE" and has_secured_assets:
        return LEGAL_REVIEW, TIER_1
    elif score_band == "HOT":
        return SETTLEMENT_LUMP, TIER_1
    else:
        return AGENCY, TIER_2

# LIQUIDITY_CONSTRAINED
if persona == "LIQUIDITY_CONSTRAINED":
    if score_band in ["HOT", "WARM"]:
        return SETTLEMENT_PLAN, TIER_2
    else:
        return HOLD, TIER_3

# STRATEGIC
if persona == "STRATEGIC":
    if has_secured_assets:
        return LEGAL_REVIEW, TIER_1
    else:
        return AGENCY, TIER_2

# DORMANT
if persona == "DORMANT":
    if score_band == "HOT":
        return SETTLEMENT_LUMP, TIER_2
    else:
        return HOLD, TIER_3

# Stage overrides
if stage == "CHARGEOFF" and months_since_chargeoff > 24:
    return HOLD, TIER_3  # Too stale

if stage == "CHARGEOFF" and bureau_delinquent_other:
    return HOLD, TIER_3  # Systemic distress
```

**Priority tier logic**:
- **TIER_1**: High-value, high-recovery, or legal path.
- **TIER_2**: Medium recovery, standard operations.
- **TIER_3**: Low recovery, hold or minimal action.

---

### Overrides for Legal/Agency/Hold

**Legal overrides** (lines 197-210):
```python
# Legal Review triggers:
# 1. SELECTIVE_DEFAULTER + LARGE balance + has_secured_assets
# 2. STRATEGIC + has_secured_assets

if persona == "SELECTIVE_DEFAULTER":
    if balance_band == "LARGE" and has_secured_assets:
        return ActionRecommendation(
            recommended_action="LEGAL_REVIEW",
            priority_tier="TIER_1",
            reasoning="Selective defaulter with large balance and secured assets - legal leverage available"
        )
```

**Hold overrides** (lines 212-231):
```python
# Hold triggers:
# 1. CHARGEOFF + months_since_chargeoff > 24 (too stale)
# 2. CHARGEOFF + bureau_delinquent_other (systemic distress)
# 3. LIQUIDITY_CONSTRAINED + score_band in [COLD, FROZEN]
# 4. DORMANT + score_band in [COLD, FROZEN]

if stage == "CHARGEOFF":
    if months_since_chargeoff > 24:
        return ActionRecommendation(
            recommended_action="HOLD",
            priority_tier="TIER_3",
            reasoning="Chargeoff too stale (>24 months) - uneconomic to pursue"
        )

    if bureau_delinquent_other:
        return ActionRecommendation(
            recommended_action="HOLD",
            priority_tier="TIER_3",
            reasoning="Systemic distress (delinquent at multiple lenders) - deep discount or debt sale only"
        )
```

**Agency overrides**:
- Catch-all for personas with WARM/COLD scores that don't qualify for settlement or legal.
- No explicit override logic - emerges from rule fallbacks.

---

### Weak Points / Missing Rules

| Weakness | Impact | Recommendation |
|----------|--------|----------------|
| **No balance-band consideration for LIQUIDITY_CONSTRAINED** | Small balance customers get same treatment as large | Add balance_band to LIQUIDITY_CONSTRAINED rules |
| **No recency of contact check** | May recommend PHONE contact for customers contacted yesterday | Add `days_since_last_contact` override for suppression |
| **No PTP behavior check** | ACTIVE_PAYER with broken PTPs gets SETTLEMENT_PLAN | Add `ptp_kept_rate` check to downgrade to AGENCY if low |
| **STRATEGIC always routed to LEGAL or AGENCY** | No hold option even if balance tiny | Add `balance < 10_000 → HOLD` for STRATEGIC |
| **No offer cap by balance** | SETTLEMENT_LUMP on 5000 THB balance uneconomic | Add `balance < threshold → HOLD` guard |
| **Stage=SM routing unclear** | SM stage never explicitly mentioned in rules | Clarify SM stage routing (currently falls through to persona) |

---

## 8) "What I Need from Databricks" List

### A) Run Daily Scoring End-to-End

**Required tables/columns**:

**Table 1: Account Snapshot** (`collections.account_snapshot_daily`)
```sql
SELECT
    account_id,                 -- str
    snapshot_date,              -- date (partitioning key)
    stage,                      -- str (SM/NPL/CHARGEOFF)
    balance,                    -- float
    days_past_due,              -- int
    months_since_chargeoff,     -- int
    principal_outstanding,      -- float
    accrued_interest,           -- float
    penalty_charges             -- float
FROM collections.account_snapshot_daily
WHERE snapshot_date = '2024-01-15'
```

**Table 2: Payment History** (`collections.payments_history`)
```sql
-- Pre-aggregate by windows (0-30, 31-90, 91-180, 181-365 days)
SELECT
    account_id,
    snapshot_date,
    payment_count_30d,          -- int
    payment_count_90d,
    payment_count_180d,
    payment_count_365d,
    payment_amt_30d,            -- float
    payment_amt_90d,
    payment_amt_180d,
    payment_amt_365d,
    days_since_last_payment,    -- int
    last_payment_amount         -- float
FROM collections.payment_features_daily
WHERE snapshot_date = '2024-01-15'
```

**Table 3: Contact Actions** (`efs.contact_actions`)
```sql
SELECT
    account_id,
    action_date,                -- date
    action_category,            -- str (CALL/SMS/EMAIL)
    action_result,              -- str (CONNECTED/NO_ANSWER/REFUSED)
    action_actor                -- str (agent ID)
FROM efs.contact_actions
WHERE action_date <= '2024-01-15'
  AND action_date >= DATE_SUB('2024-01-15', 90)  -- Last 90 days
```

**Table 4: PTP Records** (`efs.ptp_settlements`)
```sql
SELECT
    account_id,
    ptp_date,                   -- date
    promise_status,             -- str (KEPT/BROKEN/OPEN)
    ptp_amount                  -- float
FROM efs.ptp_settlements
WHERE ptp_date <= '2024-01-15'
```

**Table 5: Bureau Data** (`bureau.ncb_tuef_latest`)
```sql
SELECT
    account_id,
    bureau_pull_date,
    bureau_total_outstanding,   -- float
    bureau_monthly_instalment,  -- float
    bureau_secured_loan_flag,   -- bool
    bureau_delinquent_other     -- bool
FROM bureau.ncb_tuef_latest
WHERE account_id IN (SELECT account_id FROM collections.account_snapshot_daily WHERE snapshot_date = '2024-01-15')
```

---

### B) Train Personas (Unsupervised and Semi-Supervised Options)

**Option 1: Unsupervised (PersonaBuilder rule-based)**
- **No training needed** - rules are hardcoded.
- Just needs same tables as (A) above.

**Option 2: Semi-supervised (if clustering added)**

**Table needed**: Same as (A) but **multiple snapshots** for stability:
```sql
SELECT * FROM collections.payment_features_daily
WHERE snapshot_date BETWEEN '2024-01-01' AND '2024-12-31'
```

**Cluster on**: 4 axis scores (payment_behavior, engagement, capacity, avoidance).

**K-selection**: Elbow method on SSE (not implemented, would need to add).

---

### C) Train Propensity Model with Temporal Validation

**Training data** (6-12 months of historical snapshots):

**Table 1: Features** (`collections.feature_snapshot_historical`)
```sql
SELECT
    account_id,
    snapshot_date,
    -- All 50+ numeric features from PersonaBuilder axes + bureau + delinquency
    payment_count_30d, payment_amt_30d, ...,
    call_response_rate, sms_response_rate, ...,
    bureau_total_outstanding, ...,
    dlnq_max_bucket_12m, ...
FROM collections.feature_snapshot_historical
WHERE snapshot_date BETWEEN '2023-06-30' AND '2024-06-30'
```

**Table 2: Labels** (`collections.recovery_outcomes`)
```sql
SELECT
    account_id,
    snapshot_date,
    -- Forward-looking outcomes computed from T3 transactions
    recovery_amount_180d,       -- float
    outcome_pay_any_90d,        -- binary
    outcome_pay_amount_90d,     -- float
    outcome_cured,              -- binary
    roll_bucket_t,              -- int (0-6)
    roll_bucket_horizon         -- int (0-6)
FROM collections.recovery_outcomes
WHERE snapshot_date BETWEEN '2023-06-30' AND '2024-06-30'
```

**Join**:
```python
train_df = features.merge(labels, on=["account_id", "snapshot_date"], how="inner")

# Temporal validation split:
train_mask = train_df["snapshot_date"] < "2024-03-31"
val_mask   = train_df["snapshot_date"] >= "2024-03-31"

X_train = train_df[train_mask][numeric_cols]
X_val   = train_df[val_mask][numeric_cols]
```

---

**End of Audit**
