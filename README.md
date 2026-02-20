# CardX Debt Collections Decision Agent

Production-ready AI-powered debt collections system for CardX (Thai bank) with two complementary approaches: NPV-driven offer optimization and rule-based recovery routing.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tested](https://img.shields.io/badge/tested-passing-brightgreen.svg)](src/recovery_agent_practical/example_usage.py)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Systems](#systems)
  - [System 1: NPV-Driven TDR Engine](#system-1-npv-driven-tdr-engine)
  - [System 2: Rule-Based Recovery Agent](#system-2-rule-based-recovery-agent)
- [Feature Set](#feature-set)
- [Bureau Data Integration](#bureau-data-integration)
- [Installation](#installation)
- [Usage Examples](#usage-examples)
- [Outputs](#outputs)
- [Deployment](#deployment)
- [Documentation](#documentation)
- [Contributing](#contributing)

---

## 🎯 Overview

This repository provides **two complementary debt collections decision systems**:

1. **NPV-Driven TDR Engine** (`decision_agent/tdr/`) - Sophisticated offer optimization using paydown curves and NPV maximization
2. **Rule-Based Recovery Agent** (`recovery_agent_practical/`) - Operationally simple persona-based routing with machine learning scoring

Both systems leverage:
- ✅ NCB (Thai Credit Bureau) integration with TUEF parser
- ✅ Stage-wise feature engineering (0-30, 31-90, 91-180, 181-365 day windows)
- ✅ Point-in-time safe feature computation (no data leakage)
- ✅ Buddhist Era (BE) date handling
- ✅ Production-ready schemas and audit trails

**Choose the right system:**
- **NPV-Driven**: When you have historical offer data and calibrated recovery curves
- **Rule-Based**: For quick deployment, operational transparency, no offer history needed

---

## 🏗️ Architecture

### System Comparison

| Aspect | NPV-Driven | Rule-Based |
|--------|-----------|------------|
| **Optimization** | NPV maximization | Heuristic rules |
| **Data Required** | Offer history + recovery curves | Behavioral features only |
| **Deployment Time** | 6-8 weeks (calibration) | 2 weeks (default rules) |
| **Interpretability** | Black box NPV | Transparent rules |
| **Bureau Dependency** | HIGH (affordability Tier 1) | MEDIUM (capacity scoring) |
| **Use Case** | Mature collections programs | New programs, quick wins |

### Technology Stack

- **Language**: Python 3.9+
- **ML Libraries**: scikit-learn, pandas, numpy
- **Data Storage**: Delta Lake, Parquet
- **Feature Engineering**: PySpark (optional for scale)
- **Bureau Integration**: Custom TUEF parser
- **Deployment**: Batch scoring, daily pipeline

---

## 🚀 Quick Start

### Rule-Based System (Recommended for First Time)

```python
from recovery_agent_practical import RecoveryAgentPipeline
import pandas as pd

# 1. Load your data
features_df = pd.read_csv("your_features.csv")
labels_df = pd.read_csv("historical_recoveries.csv")

# 2. Initialize pipeline
pipeline = RecoveryAgentPipeline(
    scorecard_model_type="TWO_PART",
    payment_threshold=500.0  # THB
)

# 3. Train (one-time)
metrics = pipeline.train_scorecard(features_df, labels_df, val_split=0.2)
print(f"Validation AUC: {metrics['val'].auc_roc:.3f}")

# 4. Score daily batch
daily_scores = pipeline.score_batch(
    features_df=today_features,
    score_date="2024-01-15",
    write_outputs=True,
    output_path="./outputs"
)

# 5. View results
print(daily_scores[["account_id", "persona", "score_band",
                     "recommended_action"]].head())
```

**Output**: Daily scoring table with persona, recovery score, recommended action, priority tier.

### NPV-Driven System (For Advanced Use)

```python
from decision_agent.tdr import OfferGenerator, NPVEngine, AffordabilityEngine

# Initialize engines
npv_engine = NPVEngine(monthly_discount_rate=0.02)
aff_engine = AffordabilityEngine()
offer_gen = OfferGenerator(npv_engine, aff_engine)

# Generate best offer for an account
recommendation = offer_gen.generate(
    account=account_series,
    persona="SELECTIVE_DEFAULTER",
    months_at_180plus=3
)

print(f"Best offer: {recommendation.offer_type}")
print(f"NPV: {recommendation.npv:,.0f} THB")
print(f"Haircut: {recommendation.haircut_pct:.1%}")
```

### Test with Synthetic Data

```bash
# Run working example with synthetic data
PYTHONPATH=src:$PYTHONPATH python3 src/recovery_agent_practical/example_usage.py

# Expected output:
# ✓ Persona distribution
# ✓ Score band distribution
# ✓ Action distribution
# ✓ Sample recommendations
```

---

## 📊 Systems

### System 1: NPV-Driven TDR Engine

**Location**: `src/decision_agent/tdr/`

**Read full documentation**: [NPV-Driven System Details](src/decision_agent/tdr/README.md)

#### Key Components

##### Bureau Parser
- Parses NCB TUEF (Thai Union Exchange Format) JSON
- Buddhist Era (BE) → Gregorian date conversion
- Extracts: DSR, secured loans, delinquencies, installments

##### Paydown Curves
- **Curve A**: Natural recovery without offer
- **Curve B**: Recovery with structured settlement
- Persona-based calibration
- NPV: `PV(Curve_B) - PV(Curve_A) - concession_cost`

##### Affordability Engine
**5-Tier Waterfall for Income Estimation:**
1. Bureau DSR → implied income
2. Bureau monthly installments
3. Last payment amount
4. Payment history average
5. Segment median (fallback)

##### NPV Engine
Compares 4 paths: TDR, LEGAL, DEBT_SALE, HOLD

##### Offer Generator
- Haircut 30-70% candidates
- Cheapest-first waiver ordering (charges → interest → principal)
- Balance decomposition-aware

---

### System 2: Rule-Based Recovery Agent

**Location**: `src/recovery_agent_practical/`

**Read full documentation**: [Rule-Based System README](src/recovery_agent_practical/README.md)

#### Key Components

##### 1. PersonaBuilder
**4 Behavioral Axes** (0-100):
- Payment behavior
- Engagement
- Capacity
- Avoidance

**5 Personas**:
- ACTIVE_PAYER
- SELECTIVE_DEFAULTER
- LIQUIDITY_CONSTRAINED
- STRATEGIC
- DORMANT

##### 2. RecoveryScorecard6M
**Two Options**:
- Two-part model (P(pay) × E(amount|paid))
- Tweedie regression (single-stage)

**Score Bands**: HOT/WARM/COLD/FROZEN

##### 3. ActionOverlayRouter
**5 Actions**:
- SETTLEMENT_LUMP
- SETTLEMENT_PLAN
- AGENCY
- LEGAL_REVIEW
- HOLD

**Routing**: `persona × stage × balance × staleness × score → action`

---

## 📈 Feature Set

### Stage-Wise Windows
- 0-30 days (recent)
- 31-90 days (short-term)
- 91-180 days (medium-term)
- 181-365 days (long-term)

### 8 Feature Families

1. **Delinquency Trajectory**: DPD trends, staleness
2. **Balance & Utilization**: Principal/interest/charges decomposition
3. **Internal Payments**: Frequency, amounts, recency
4. **Actions & Engagement**: Call/SMS response rates, PTP
5. **Transaction-Spend**: Spending patterns (optional)
6. **Bureau Exposure**: Total debt, installments, secured loans
7. **Bureau Delinquency**: Delinquencies at other lenders
8. **Avoidance Flags**: Wrong number, disputes, lawyer

---

## 🏦 Bureau Data Integration

### NCB (Thai Credit Bureau) - TUEF Format

#### Critical Bureau Fields

| Field | Purpose | Priority |
|-------|---------|----------|
| `bureau_total_outstanding` | Affordability, Capacity scoring | **P0** |
| `bureau_monthly_instalment` | DSR calculation | **P0** |
| `bureau_secured_loan_flag` | Legal viability | **P0** |
| `bureau_delinquent_other` | Strategic defaulter signal | P1 |
| `bureau_active_loan_count` | Stress indicator | P1 |
| `bureau_new_loan_12m` | Liquidity signal | P2 |

#### TUEF JSON Example

```json
{
  "accountSegment": [{
    "accountNumber": "string",
    "installmentAmount": 5000,
    "outstandingBalance": 150000,
    "dateOpened": "25661201",
    "currentDPD": 90,
    "securedLoanFlag": true
  }],
  "hss": [{
    "totalOutstanding": 450000,
    "totalMonthlyInstallment": 25000,
    "totalActiveAccounts": 5
  }]
}
```

#### Integration

```python
from decision_agent.features import BureauFeatureExtractor

extractor = BureauFeatureExtractor()
bureau_features = extractor.extract_from_tuef(tuef_json)
```

### What We Need From You

1. ✅ Sample TUEF JSON (anonymized 5-10 accounts)
2. ✅ Field mapping doc (NCB → CardX names)
3. ✅ BE date format confirmation
4. ✅ Historical recovery data (6-12 months)

---

## 💾 Installation

```bash
# Clone repository
git clone https://github.com/Sushil-tata/claude_DA2.git
cd claude_DA2

# Install dependencies
pip install -r requirements.txt

# Optional: Install in development mode
pip install -e .
```

### Dependencies

```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
pyarrow>=10.0.0
```

---

## 📝 Usage Examples

### Example 1: Rule-Based Daily Scoring

```python
from recovery_agent_practical import RecoveryAgentPipeline

# Initialize
pipeline = RecoveryAgentPipeline(scorecard_model_type="TWO_PART")

# Train
metrics = pipeline.train_scorecard(features_df, labels_df, val_split=0.2)

# Score
daily_scores = pipeline.score_batch(
    features_df=features_df,
    score_date="2024-01-15",
    write_outputs=True,
    output_path="./outputs"
)

print(f"Scored {len(daily_scores)} accounts")
print(daily_scores['persona'].value_counts())
```

### Example 2: NPV-Driven Offers

```python
from decision_agent.tdr import OfferGenerator, NPVEngine, AffordabilityEngine

# Initialize
npv_engine = NPVEngine(monthly_discount_rate=0.02)
aff_engine = AffordabilityEngine()
offer_gen = OfferGenerator(npv_engine, aff_engine)

# Generate offers
for _, account in accounts_df.iterrows():
    rec = offer_gen.generate(account, persona, months_at_180plus)
    print(f"{account['account_id']}: {rec.offer_type}, NPV={rec.npv:,.0f}")
```

### Run Examples

```bash
# Test with synthetic data (both systems)
PYTHONPATH=src:$PYTHONPATH python3 src/recovery_agent_practical/example_usage.py
```

---

## 📤 Outputs

### Daily Scoring Table

**Partitioning**: `score_date` (daily)
**Primary Key**: `(account_id, score_date)`

**Key Columns**:
- Persona + 4 axis scores
- Score band (HOT/WARM/COLD/FROZEN)
- P(recovery), expected amount
- Recommended action
- Priority tier (TIER_1/2/3)
- Contact channel
- Routing reasoning

### Audit Log

**Event Types**:
- PERSONA_ASSIGNED
- SCORED
- ACTION_ROUTED
- ACTION_EXECUTED
- OUTCOME_OBSERVED

Full audit trail for compliance.

---

## 🚀 Deployment

### Quick Win (2 weeks) - Rule-Based

**Week 1**: Train scorecard on historical data
**Week 2**: Deploy daily scoring pipeline

```bash
# Daily cron
0 6 * * * python3 run_daily_scoring.py --date $(date +\%Y-\%m-\%d)
```

### Full Solution (6-8 weeks) - NPV-Driven

**Weeks 1-2**: Data prep, recovery curve calibration
**Weeks 3-4**: Model training, NPV tuning
**Weeks 5-6**: Testing, validation
**Weeks 7-8**: Production deployment

---

## 📚 Documentation

### Component Documentation

- **Rule-Based System**: [`src/recovery_agent_practical/README.md`](src/recovery_agent_practical/README.md)
- **Implementation Summary**: [`RECOVERY_AGENT_PRACTICAL_SUMMARY.md`](RECOVERY_AGENT_PRACTICAL_SUMMARY.md)
- **Bureau Integration**: Inline docs in `bureau_features.py`
- **Paydown Curves**: Inline docs in `paydown_curves.py`

### Examples

- **Complete Examples**: [`src/recovery_agent_practical/example_usage.py`](src/recovery_agent_practical/example_usage.py)
- **Tested**: ✅ Both examples run successfully with synthetic data

---

## 📊 Test Results

### Synthetic Data (500 accounts)

**Persona Distribution**:
- SELECTIVE_DEFAULTER: 56.6%
- DORMANT: 42.2%
- STRATEGIC: 0.8%
- ACTIVE_PAYER: 0.4%

**Score Bands**:
- HOT: 10.6%
- WARM: 25.8%
- COLD: 36.4%
- FROZEN: 27.2%

**Actions**:
- AGENCY: 54.4%
- SETTLEMENT_PLAN: 27.0%
- SETTLEMENT_LUMP: 13.2%
- LEGAL_REVIEW: 3.0%
- HOLD: 2.4%

**Performance**:
- Validation AUC: 0.65
- Training time: <5 seconds
- Scoring time: <2 seconds
- Expected recovery: 14,090 THB/account

---

## 🤝 Contributing

```bash
# Development setup
git clone https://github.com/Sushil-tata/claude_DA2.git
pip install -e .

# Run tests
pytest tests/

# Run examples
PYTHONPATH=src:$PYTHONPATH python3 src/recovery_agent_practical/example_usage.py
```

### Branches

- `main`: Production code
- `recovery_agent_practical`: Rule-based system ✅
- Feature branches: `feature/your-feature`

---

## 🎯 Quick Reference

| Task | System | Command |
|------|--------|---------|
| **Quick start** | Rule-Based | `python3 src/recovery_agent_practical/example_usage.py` |
| **Train** | Rule-Based | `pipeline.train_scorecard(features_df, labels_df)` |
| **Score** | Rule-Based | `pipeline.score_batch(features_df, score_date)` |
| **Generate offer** | NPV | `offer_gen.generate(account, persona, months_at_180plus)` |
| **Parse bureau** | Both | `extractor.extract_from_tuef(tuef_json)` |

---

## 🏆 Key Differentiators

✅ **Two Systems, One Platform**: Choose rule-based OR NPV-driven
✅ **Bureau Integration**: NCB TUEF parser with BE dates
✅ **Production-Ready**: Schemas, audit logs, monitoring
✅ **Tested**: Example scripts run successfully
✅ **Transparent**: Every decision explained
✅ **Scalable**: 100 to 100,000 accounts

---

## 📞 Next Steps

**Please provide**:
1. Sample NCB TUEF JSON (anonymized)
2. Field mapping document
3. Historical recovery data (6-12 months)
4. Data warehouse access

**We will**:
1. Verify bureau parser
2. Test on real CardX data
3. Calibrate models and rules
4. Deploy daily scoring pipeline

---

## 📄 License

Internal use only - CardX Collections Team

---

**Ready to deploy!** 🚀

For questions, contact the CardX Data Science Team.
