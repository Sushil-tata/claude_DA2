# BFE Repository - Quick Reference Card

**Version:** BFE_v1.0 | **Status:** Production-Ready

---

## 🚀 Installation (One Command)

```bash
pip install -r decision_engine/feature_store/behavioral_feature_engine/requirements.txt
```

---

## 📖 Basic Usage

### Get Features (Single Account)

```python
from decision_engine.feature_store.behavioral_feature_engine import get_features

features = get_features(
    account_id="ACC123456",
    account_history={
        "delinquency": dpd_df,      # Must have: account_id, date, dpd
        "payment": payment_df        # Must have: account_id, date, payment_amount, amount_due
    },
    as_of_date="2024-01-31",
    feature_sets=["delinquency", "payment"]  # or ["all"]
)

# Access features
current_dpd = features["delinquency.dpd_current"]
regime = features["delinquency.delinquency_regime"]
payment_ratio = features["payment.payment_ratio_6M_mean"]
```

### Batch Processing

```python
from decision_engine.feature_store.behavioral_feature_engine import get_features_batch

features_df = get_features_batch(
    accounts_history={
        "ACC001": {"delinquency": df1, "payment": df2},
        "ACC002": {"delinquency": df3, "payment": df4},
        # ...
    },
    as_of_date="2024-01-31",
    feature_sets=["delinquency", "payment"]
)

# Result: DataFrame with one row per account
```

---

## 🗂️ Data Requirements

### Delinquency Module

**Required columns:**
- `account_id` - Unique identifier
- `date` - Observation date (monthly)
- `dpd` - Days Past Due (0 = current)

**Optional columns:**
- `account_open_date` - Account opening date

### Payment Module

**Required columns:**
- `account_id` - Unique identifier
- `date` - Payment/statement date (monthly)
- `payment_amount` - Amount paid
- `amount_due` - Amount due

**Optional columns:**
- `minimum_due` - Minimum payment required
- `statement_balance` - Statement balance
- `due_date` - Payment due date
- `payment_date` - Actual payment date
- `balance` - Outstanding balance

---

## 🎯 Top 10 Features (Copy-Paste Ready)

```python
# Delinquency Features
features["delinquency.dpd_current"]                  # Current DPD
features["delinquency.delinquency_regime"]           # STABLE/DETERIORATING/etc.
features["delinquency.dpd_6M_slope"]                 # Trajectory
features["delinquency.cure_flag_6M"]                 # Recovery indicator
features["delinquency.bucket_velocity_6M"]           # Deterioration speed

# Payment Features
features["payment.payment_ratio_6M_mean"]            # Payment consistency
features["payment.payment_regime"]                   # CONSISTENT_FULL_PAYER/etc.
features["payment.consecutive_missed_current"]       # Payment crisis indicator
features["payment.payment_ratio_6M_slope"]           # Payment trend
features["payment.payment_regularity_score_6M"]      # Timing consistency
```

---

## 🔧 Advanced Usage

### List All Features

```python
from decision_engine.feature_store.behavioral_feature_engine import list_features

# All features
catalog = list_features()

# Specific module
delinq_features = list_features(module="delinquency")

# Inspect metadata
print(catalog["delinquency.dpd_current"])
# {
#   "tier": 1,
#   "definition": "Current Days Past Due (spot value)",
#   "feature_type": "continuous",
#   "signal_direction": "higher = riskier",
#   ...
# }
```

### Export Feature Registry

```python
from decision_engine.feature_store.behavioral_feature_engine import save_feature_registry

save_feature_registry(output_path="my_features.json")
```

### Reset Schema Mappings

```python
from decision_engine.feature_store.behavioral_feature_engine import reset_schema_mappings

reset_schema_mappings()  # Forces re-prompting for schema
```

---

## 🛡️ Safety Features

### Point-in-Time Safety (Automatic)

```python
# BFE automatically filters data to as_of_date
features = get_features(
    account_id="ACC123",
    account_history={"delinquency": df},
    as_of_date="2024-01-31"
)
# Only data <= 2024-01-31 is used (NO look-ahead bias)
```

### Graceful Degradation

```python
# Missing data returns NaN (never crashes)
features["payment.payment_ratio_6M_mean"]  # NaN if insufficient data

# New accounts flagged
features["new_account_flag"]  # True if < 3 months history

# Staleness detected
features["staleness_days"]  # Days since last update
```

---

## 📊 Feature Families (for Correlation Analysis)

```python
DELINQUENCY_LEVEL = [
    "delinquency.dpd_current",
    "delinquency.bucket_current",
    "delinquency.dpd_6M_mean"
]

DELINQUENCY_TRAJECTORY = [
    "delinquency.dpd_6M_slope",
    "delinquency.dpd_6M_momentum",
    "delinquency.bucket_velocity_6M"
]

PAYMENT_RATIO = [
    "payment.payment_ratio_6M_mean",
    "payment.payment_ratio_6M_slope",
    "payment.payment_ratio_6M_std"
]

PAYMENT_BEHAVIOR = [
    "payment.consecutive_missed_current",
    "payment.missed_payment_6M_count",
    "payment.full_payment_flag_6M_count"
]
```

**Pruning Guidance:** Remove one feature from pairs with correlation > 0.85

---

## 🎛️ Regime Classifications

### Delinquency Regimes

| Regime | Meaning | Risk Level |
|--------|---------|------------|
| STABLE | Low DPD, low volatility | Low |
| DETERIORATING | Increasing DPD | High |
| RECOVERING | Decreasing DPD | Medium |
| VOLATILE | Oscillating DPD | High |
| SEVERELY_DELINQUENT | NPL/CO | Very High |
| ELEVATED | Above 30 DPD | Medium |

### Payment Regimes

| Regime | Meaning | Risk Level |
|--------|---------|------------|
| CONSISTENT_FULL_PAYER | Pays 90%+ consistently | Low |
| CONSISTENT_PARTIAL_PAYER | Pays 40-90% consistently | Medium |
| ERRATIC_PAYER | High volatility | High |
| MINIMAL_PAYER | Pays <40% | High |
| NON_PAYER | Missed 3+ in 6M | Very High |
| IMPROVING | Increasing payment ratio | Medium |
| DETERIORATING | Decreasing payment ratio | High |

---

## ⚡ Performance Tips

```python
# Batch processing is MUCH faster than looping
# DON'T DO THIS:
for account_id in accounts:
    features = get_features(account_id, ...)  # SLOW

# DO THIS:
features_df = get_features_batch(accounts_history, ...)  # FAST
```

---

## 🧪 Testing

```bash
# Run tests
pytest decision_engine/feature_store/behavioral_feature_engine/tests/ -v

# Run examples
python decision_engine/feature_store/behavioral_feature_engine/example_usage.py
```

---

## 📞 Help

```python
# Comprehensive docs
decision_engine/feature_store/behavioral_feature_engine/README.md

# Implementation details
decision_engine/feature_store/behavioral_feature_engine/IMPLEMENTATION_SUMMARY.md

# Delivery summary
decision_engine/feature_store/behavioral_feature_engine/DELIVERY_SUMMARY.md
```

---

## 🚨 Common Issues

### Issue: "Schema mapping prompt appears"

**Solution:** This is expected on first run. Answer prompts to map your data fields.

Mappings are saved to `~/.bfe_schema_mapping.json` for future use.

### Issue: "Features return NaN"

**Causes:**
1. Insufficient history (need 6M+ for most features)
2. Missing required columns
3. New account (<3 months)

**Check:**
```python
features["months_of_history"]         # Should be >= 6
features["insufficient_history_flag"] # Should be False
```

### Issue: "Point-in-time error"

**Cause:** Data contains dates > as_of_date

**Solution:** Data is automatically filtered. If you see an error, check your data timestamps.

---

## 🔢 Quick Stats

- **Total Features:** 180+
- **Modules:** 2 (Delinquency, Payment)
- **Lines of Code:** 9,496
- **Documentation:** 1,400+ lines
- **Test Coverage:** Point-in-time safety validated

---

## 💡 Integration Pattern

```python
def score_account_with_bfe(account_id, as_of_date):
    """Integrate BFE into existing scoring"""

    # 1. Your existing features (unchanged)
    existing = your_feature_extractor(account_id)

    # 2. BFE features (additive)
    bfe = get_features(
        account_id=account_id,
        account_history={
            "delinquency": load_dpd_history(account_id),
            "payment": load_payment_history(account_id)
        },
        as_of_date=as_of_date,
        feature_sets=["delinquency", "payment"]
    )

    # 3. Merge (no collision - BFE features are namespaced)
    all_features = {**existing, **bfe}

    # 4. Score
    return your_model.predict(all_features)
```

---

**Version:** BFE_v1.0 | **Status:** Production-Ready | **Features:** 180+
