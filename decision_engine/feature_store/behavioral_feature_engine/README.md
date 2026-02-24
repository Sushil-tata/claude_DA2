# Behavioral Feature Engineering (BFE) Repository

**Version:** BFE_v1.0
**Status:** Production-Ready MVP (Delinquency + Payment Modules)

---

## 🎯 Overview

The BFE Repository is a self-contained, versioned module that enriches Decision Engine capabilities with behavioral features. It operates as a **feature provider**, not a scorer — augmenting existing Decision Engine features without modifying core logic.

### Key Principles

✅ **ADDITIVE** — Augments, never overwrites existing Decision Engine features
✅ **QUERYABLE** — Any agent can call `get_features(account_id, feature_set, as_of_date)`
✅ **VERSIONED** — Features tagged as `BFE_v{major}.{minor}`
✅ **AUDITABLE** — Every feature computation is traceable to source data
✅ **POINT-IN-TIME SAFE** — No look-ahead bias; respects `as_of_date`
✅ **INTERACTIVE SCHEMA** — Maps to your data schema at runtime (no upfront config)

---

## 📦 What's Included

### Feature Modules (v1.0)

| Module | Status | Features | Description |
|--------|--------|----------|-------------|
| **Delinquency** | ✅ Complete | 100+ | DPD statistics, bucket transitions, trajectories, regimes |
| **Payment** | ✅ Complete | 80+ | Payment ratios, timing, consistency, elasticity, regimes |
| **Repayment (Term Loan)** | 🔜 Planned | 50+ | EMI adherence, prepayments, restructuring |
| **Utilization (Revolving)** | 🔜 Planned | 40+ | Credit utilization, cash advances, transactor/revolver flags |
| **Balance Exposure** | 🔜 Planned | 30+ | Balance trajectory, overdue interest, penalties |
| **Bureau** | 🔜 Planned | 60+ | Bureau score trends, trade analysis, inquiries |
| **Cross-Product** | 🔜 Planned | 20+ | Multi-product exposure and delinquency |
| **Composite Indices** | 🔜 Planned | 10+ | Payment stress, credit hunger, early warning scores |

### Core Utilities

- **SchemaMapper**: Interactive runtime schema mapping
- **WindowCalculator**: Rolling statistics, slopes, momentum
- **TemporalValidator**: Point-in-time safety enforcement
- **NullHandler**: Graceful handling of missing data

---

## 🚀 Quick Start

### Installation

```bash
# Navigate to project root
cd /path/to/decision_engine

# Install dependencies
pip install -r feature_store/behavioral_feature_engine/requirements.txt

# Verify installation
python -c "from decision_engine.feature_store.behavioral_feature_engine import get_features; print('✓ BFE installed')"
```

### Basic Usage

```python
from decision_engine.feature_store.behavioral_feature_engine import get_features
import pandas as pd

# Prepare your data
account_history = {
    "delinquency": pd.DataFrame({
        "account_id": ["ACC123"] * 24,
        "date": pd.date_range("2022-02-01", periods=24, freq="MS"),
        "dpd": [0, 5, 15, 30, 45, 60, 90, 120, 95, 70, 45, 20, 0, 0, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
    }),
    "payment": pd.DataFrame({
        "account_id": ["ACC123"] * 24,
        "date": pd.date_range("2022-02-01", periods=24, freq="MS"),
        "payment_amount": [5000, 4500, 4000, 3500, 3000, 2500, 2000, 1500, 2000, 2500, 3000, 4000, 5000, 5000, 5000, 4800, 4600, 4400, 4200, 4000, 3800, 3600, 3400, 3200],
        "amount_due": [5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000]
    })
}

# Get features (FIRST RUN WILL PROMPT FOR SCHEMA MAPPING)
features = get_features(
    account_id="ACC123",
    account_history=account_history,
    as_of_date="2024-01-31",
    feature_sets=["delinquency", "payment"]
)

# Access features
print(f"Current DPD: {features['delinquency.dpd_current']}")
print(f"Delinquency Regime: {features['delinquency.delinquency_regime']}")
print(f"Payment Ratio (6M): {features['payment.payment_ratio_6M_mean']:.2%}")
print(f"Payment Regime: {features['payment.payment_regime']}")
```

### Interactive Schema Mapping

**On first run**, the BFE repository will prompt you to map its field requirements to your data schema:

```
══════════════════════════════════════════════════════════════════
SCHEMA MAPPING REQUIRED: DELINQUENCY Domain
══════════════════════════════════════════════════════════════════

REQUIRED FIELDS:
──────────────────────────────────────────────────────────────────

  BFE Field: account_id
  Description: Unique account identifier
  Type: string
  → Your column name: account_number

  BFE Field: date
  Description: Observation date (monthly)
  Type: date
  → Your column name: obs_date

  BFE Field: dpd
  Description: Days Past Due (0 = current)
  Type: int
  → Your column name: days_past_due

✓ Schema mapping complete for 'delinquency' domain
```

Mappings are saved to `~/.bfe_schema_mapping.json` and reused on subsequent runs.

---

## 📖 Feature Catalog

### Delinquency Module Features

#### Tier 1: Core Risk Drivers
- `dpd_current` - Current Days Past Due (spot value)
- `dpd_{W}_mean` - Average DPD over window W (3M, 6M, 12M, 24M)
- `dpd_{W}_max` - Maximum DPD over window
- `dpd_{W}_slope` - Linear regression slope of DPD
- `dpd_{W}_momentum` - Change in DPD slope (acceleration)
- `bucket_current` - Current bucket (REGULAR/SM/NPL/CHARGE_OFF)
- `bucket_velocity_{W}` - Rate of bucket deterioration
- `cure_flag_{W}` - Binary: moved from SM/NPL to REGULAR
- `re_delinquency_flag_{W}` - Binary: cured then re-entered delinquency
- **`delinquency_regime`** - STABLE/DETERIORATING/VOLATILE/RECOVERING/SEVERELY_DELINQUENT

#### Tier 2: Volatility & Stability
- `dpd_{W}_std` - Standard deviation of DPD
- `dpd_{W}_cv` - Coefficient of variation
- `bucket_oscillation_{W}` - Number of bucket changes
- `bucket_deterioration_streak` - Consecutive months worsening
- `bucket_improvement_streak` - Consecutive months stable/improving

### Payment Module Features

#### Tier 1: Core Payment Drivers
- `payment_ratio_{W}_mean` - Average payment ratio (paid/due)
- `payment_ratio_{W}_slope` - Velocity of payment ratio
- `payment_ratio_{W}_momentum` - Acceleration of payment ratio
- `payment_ratio_mom_change` - Month-over-month change
- `missed_payment_{W}_count` - Count of months with zero payment
- `consecutive_missed_current` - Current streak of missed payments
- `full_payment_flag_{W}_count` - Count of full payments
- `min_payment_flag_{W}_count` - Count of minimum payments
- **`payment_regime`** - CONSISTENT_FULL_PAYER/CONSISTENT_PARTIAL_PAYER/ERRATIC_PAYER/MINIMAL_PAYER/NON_PAYER/IMPROVING/DETERIORATING

#### Tier 2: Timing & Regularity
- `payment_timing_avg_{W}` - Average days early/late
- `payment_timing_std_{W}` - Consistency of timing
- `payment_regularity_score_{W}` - Regularity score (0-100)
- `early_payment_flag_{W}_count` - Count of pre-due-date payments

#### Tier 3: Elasticity
- `payment_ratio_elasticity` - Response to balance changes (with lag to prevent simultaneity bias)

---

## 🔧 Advanced Usage

### Batch Processing

```python
from decision_engine.feature_store.behavioral_feature_engine import get_features_batch

# Prepare data for multiple accounts
accounts_history = {
    "ACC001": {"delinquency": df1, "payment": df2},
    "ACC002": {"delinquency": df3, "payment": df4},
    # ... more accounts
}

# Batch compute
features_df = get_features_batch(
    accounts_history=accounts_history,
    as_of_date="2024-01-31",
    feature_sets=["delinquency", "payment"]
)

# Result: DataFrame with one row per account
print(features_df.head())
```

### Feature Metadata

```python
from decision_engine.feature_store.behavioral_feature_engine import list_features

# Get metadata for all features
feature_catalog = list_features()

# Filter to specific module
delinquency_features = list_features(module="delinquency")

# Inspect metadata
print(feature_catalog["delinquency.dpd_current"])
# {
#   "tier": 1,
#   "definition": "Current Days Past Due (spot value)",
#   "feature_type": "continuous",
#   "signal_direction": "higher = riskier",
#   "min_history_months": 1,
#   "use_cases": ["PD", "EWS", "Collections"]
# }
```

### Save Feature Registry

```python
from decision_engine.feature_store.behavioral_feature_engine import save_feature_registry

# Export to JSON
save_feature_registry(output_path="bfe_feature_registry.json")
```

### Reset Schema Mappings

```python
from decision_engine.feature_store.behavioral_feature_engine import reset_schema_mappings

# Clear all mappings (forces re-prompting)
reset_schema_mappings()
```

---

## 🏗️ Architecture

```
decision_engine/
└── feature_store/
    └── behavioral_feature_engine/
        ├── __init__.py                 # Unified API
        ├── modules/
        │   ├── delinquency.py          # ✅ Delinquency features
        │   ├── payment.py              # ✅ Payment features
        │   └── [future modules]        # 🔜 Bureau, Utilization, etc.
        ├── utils/
        │   ├── schema_mapper.py        # Interactive schema mapping
        │   ├── window_calculator.py    # Rolling stats, slopes
        │   ├── temporal_validator.py   # Point-in-time safety
        │   └── null_handler.py         # Missing data handling
        ├── tests/
        │   ├── test_bfe_point_in_time.py
        │   ├── test_bfe_integration.py
        │   └── test_modules/
        ├── example_usage.py            # 📘 Comprehensive examples
        ├── README.md                   # This file
        └── requirements.txt            # Dependencies
```

---

## 🧪 Running Examples

```bash
# Run comprehensive example (with synthetic data)
cd decision_engine/feature_store/behavioral_feature_engine
python example_usage.py
```

**Example Output:**
```
EXAMPLE 1: Single Account Feature Extraction
===============================================================================
Generating synthetic data for account: ACC123456
  ✓ DPD history: 24 months
  ✓ Payment history: 24 months

Computing features as of 2024-01-31...

FEATURES COMPUTED:
────────────────────────────────────────────────────────────────────────────
Delinquency Features (92):
  delinquency.dpd_current                        :      45.00
  delinquency.bucket_current                     : SM
  delinquency.delinquency_regime                 : DETERIORATING
  delinquency.dpd_6M_mean                        :      38.50
  delinquency.dpd_6M_slope                       :       5.20
  ...

Payment Features (78):
  payment.payment_ratio_6M_mean                  :       0.65
  payment.payment_regime                         : CONSISTENT_PARTIAL_PAYER
  payment.payment_ratio_6M_slope                 :      -0.03
  ...
```

---

## 📊 Feature Tiering

### Tier 1: Core Risk Drivers
**HIGH business value, direct predictive power**
- DPD levels, bucket states, payment ratios
- Regime classifications
- Transition flags (cure, redefault)

**Usage:** PD models, EWS, Collections prioritization

### Tier 2: Stability & Volatility Enhancers
**MEDIUM business value, context/stability indicators**
- Standard deviations, CVs, oscillations
- Timing consistency, regularity scores

**Usage:** Model ensembles, segmentation, stress testing

### Tier 3: Interaction & Experimental Features
**EXPLORATORY, may require pruning**
- Elasticities, cross-domain interactions
- Highly correlated feature families

**Usage:** Feature selection, advanced modeling

---

## 🛡️ Safety Features

### Point-in-Time Safety
- ✅ All features respect `as_of_date`
- ✅ No look-ahead bias (enforced by `TemporalValidator`)
- ✅ Data filtered to `<= as_of_date` before computation

### Graceful Degradation
- ✅ Missing data returns `NaN` with metadata (no errors)
- ✅ New accounts (<3 months) flagged appropriately
- ✅ Insufficient history handled gracefully

### Auditability
- ✅ Every feature tagged with `module_version`, `as_of_date`
- ✅ Schema mappings persisted for reproducibility
- ✅ Feature metadata includes definitions, signal directions

---

## 🔮 Roadmap

### v1.1 (Next Release)
- [ ] **Bureau Module** - Bureau score trends, trade analysis, inquiries
- [ ] **Utilization Module** - Credit utilization, cash advances
- [ ] **Feature Correlation Analysis** - Auto-detect redundant features (>0.85 correlation)
- [ ] **Pruning Recommendations** - Suggest features to remove

### v1.2 (Future)
- [ ] **Composite Indices** - Payment stress, credit hunger, early warning
- [ ] **Interactions Module** - Cross-domain feature interactions
- [ ] **Cross-Product Module** - Multi-product exposure features
- [ ] **Streaming API** - Real-time feature computation

---

## 🤝 Contributing

### Adding a New Module

1. Create `modules/your_module.py`:
   ```python
   class YourModuleFeatureEngine:
       VERSION = "BFE_v1.0"
       MODULE_NAME = "bfe.your_module"

       def compute_features(self, account_history, account_id, as_of_date):
           # Your logic here
           pass
   ```

2. Register in `__init__.py`:
   ```python
   MODULES = {
       "delinquency": DelinquencyFeatureEngine,
       "payment": PaymentFeatureEngine,
       "your_module": YourModuleFeatureEngine  # Add here
   }
   ```

3. Add metadata function:
   ```python
   def get_feature_metadata() -> Dict[str, Dict]:
       return {
           "feature_name": {
               "tier": 1,
               "definition": "...",
               ...
           }
       }
   ```

---

## 📞 Support

**Questions?** Contact the Decision Engine Feature Engineering team.

**Issues?** File an issue in the repository with:
- BFE version (`BFE_v1.0`)
- Error message / unexpected behavior
- Sample data (anonymized)

---

## 📄 License

Internal use only - Decision Engine Team

---

**Built with ❤️ for robust, production-grade behavioral analytics**
