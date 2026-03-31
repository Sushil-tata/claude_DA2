# Recovery Agent Practical

Production-ready debt collections decision agent for CardX (Thai bank).

## Architecture

**Frozen Methodology**: Segmentation → Recovery Curves → Relative Recovery Risk → 6M Scorecard → Action Overlay

**NO NPV optimization in V1** - pure rule-based approach for operational simplicity.

## Components

### 1. Persona Builder (`segmentation/persona_builder.py`)

Rule-based customer segmentation using historical behavior only.

**5 Personas:**
- `ACTIVE_PAYER`: Regular payment activity, responsive to contact
- `SELECTIVE_DEFAULTER`: Has capacity but chooses not to pay
- `LIQUIDITY_CONSTRAINED`: Wants to pay but lacks funds
- `STRATEGIC`: Sophisticated avoidance behavior
- `DORMANT`: No engagement despite contact attempts

**Methodology:**
- Calculates 4 behavioral axes from historical data:
  - Payment behavior score (0-100)
  - Engagement score (0-100)
  - Capacity score (0-100)
  - Avoidance score (0-100)
- Applies decision tree rules to assign persona
- Returns persona + underlying axis scores

**Usage:**
```python
from recovery_agent_practical.segmentation import PersonaBuilder

builder = PersonaBuilder(payment_threshold=500.0)

# Single account
assignment = builder.assign_persona(account_series)
print(f"Persona: {assignment.persona}")
print(f"Payment score: {assignment.payment_behavior_score}")
print(f"Confidence: {assignment.confidence_level}")

# Batch
persona_df = builder.assign_batch(features_df)
```

### 2. Recovery Scorecard 6M (`scoring/recovery_scorecard_6m.py`)

Predicts recovery potential over 180-day forward window.

**Target:** Will customer pay > 500 THB (configurable) within next 180 days?

**Two Implementation Options:**

**Option A (Default): Two-Part Model**
- Part 1: `GradientBoostingClassifier` for P(any payment > threshold)
- Part 2: `GradientBoostingRegressor` for E(amount | paid)
- Expected recovery = P(pay) × E(amount | paid)
- **Advantage**: Interpretable components, handles zero-inflation explicitly

**Option B: Tweedie Regression**
- Single-stage `TweedieRegressor`
- Handles zero-inflation natively via compound Poisson-Gamma
- **Advantage**: Simpler, fewer hyperparameters

**Score Bands:**
- `HOT` (≥70%): High recovery probability
- `WARM` (40-70%): Moderate recovery probability
- `COLD` (15-40%): Low recovery probability
- `FROZEN` (<15%): Minimal recovery probability

**Usage:**
```python
from recovery_agent_practical.scoring import RecoveryScorecard6M

# Initialize
scorecard = RecoveryScorecard6M(
    model_type="TWO_PART",  # or "TWEEDIE"
    payment_threshold=500.0
)

# Train
metrics = scorecard.train(X_train, y_train, X_val, y_val)
print(f"Validation AUC: {metrics['val'].auc_roc}")

# Score
scores_df = scorecard.score_batch(features_df, score_date="2024-01-15")
```

### 3. Action Overlay Router (`routing/action_overlay.py`)

Rule-based action recommendation using:
- `persona × stage × balance_band × months_since_chargeoff × score_band`

**Action Families:**
- `SETTLEMENT_LUMP`: One-time lump sum settlement offer
- `SETTLEMENT_PLAN`: Structured payment plan over 6-12 months
- `AGENCY`: Refer to external collection agency
- `LEGAL_REVIEW`: Escalate to legal team
- `HOLD`: No action - monitor only

**Routing Logic:**
1. Stage filter: NPL vs. CHARGEOFF
2. Persona mapping: Willingness + capacity signals
3. Balance band: SMALL (<50K), MEDIUM (50-200K), LARGE (>200K)
4. Staleness: months_since_chargeoff (fresh vs. stale)
5. Score band: HOT/WARM/COLD/FROZEN

**Example Rules:**
- `ACTIVE_PAYER + HOT → SETTLEMENT_PLAN` (likely to comply)
- `SELECTIVE_DEFAULTER + LARGE + has_secured_assets → LEGAL_REVIEW`
- `DORMANT + FROZEN + stale → HOLD` (not economic)

**Usage:**
```python
from recovery_agent_practical.routing import ActionOverlayRouter

router = ActionOverlayRouter()

# Single account
action = router.route_action(
    account_id="12345",
    persona="SELECTIVE_DEFAULTER",
    stage="NPL",
    balance=150_000,
    months_since_chargeoff=0,
    score_band="WARM",
    has_secured_assets=True
)

print(f"Action: {action.recommended_action}")
print(f"Reasoning: {action.reasoning}")

# Batch
actions_df = router.route_batch(accounts_df)
```

### 4. Pipeline Orchestrator (`pipeline.py`)

End-to-end workflow:
1. Input: Feature DataFrame (from collections_feature_pipeline)
2. Persona assignment (PersonaBuilder)
3. Recovery scoring (RecoveryScorecard6M)
4. Action routing (ActionOverlayRouter)
5. Output: Daily scoring table + audit log

**Usage:**
```python
from recovery_agent_practical import RecoveryAgentPipeline

# Initialize
pipeline = RecoveryAgentPipeline(
    scorecard_model_type="TWO_PART",
    payment_threshold=500.0
)

# Train scorecard (one-time)
metrics = pipeline.train_scorecard(
    features_df=training_features,
    labels_df=training_labels,
    val_split=0.2
)

# Daily scoring
daily_scores = pipeline.score_batch(
    features_df=today_features,
    score_date="2024-01-15",
    write_outputs=True,
    output_path="./outputs"
)
```

## Output Schemas

### Daily Scoring Table

**Partitioning:** `score_date` (daily partitions)
**Primary Key:** `(account_id, score_date)`
**Sort By:** `priority_tier`, `score_band`, `balance DESC`

**Columns:**
```python
{
    # Primary keys
    "account_id": "string",
    "score_date": "date",

    # Persona assignment
    "persona": "string",
    "payment_behavior_score": "float",
    "engagement_score": "float",
    "capacity_score": "float",
    "avoidance_score": "float",
    "persona_confidence": "string",

    # Recovery score (6M)
    "score_band": "string",
    "p_recovery_6m": "float",
    "expected_recovery_amount": "float",
    "model_version": "string",

    # Action recommendation
    "recommended_action": "string",
    "priority_tier": "string",
    "contact_channel": "string",
    "offer_type": "string",

    # Context
    "stage": "string",
    "balance": "float",
    "balance_band": "string",
    "days_past_due": "int",
    "months_since_chargeoff": "int",

    # Metadata
    "pipeline_run_id": "string",
    "created_at": "timestamp",
}
```

### Audit Log

**Partitioning:** `event_timestamp` (daily)
**Primary Key:** `audit_id`
**Indexes:** `account_id`, `event_type`, `pipeline_run_id`

**Event Types:**
- `PERSONA_ASSIGNED`
- `SCORED`
- `ACTION_ROUTED`
- `ACTION_EXECUTED`
- `OUTCOME_OBSERVED`

## Feature Set V1

**Stage-wise windows:** 0-30, 31-90, 91-180, 181-365 days + since_event features

**Feature Families:**

1. **Delinquency Trajectory**
   - `dpd_current`, `dpd_30d_ago`, `dpd_90d_ago`
   - `dpd_trend` (IMPROVING | STABLE | DETERIORATING)
   - `months_at_180plus`

2. **Balance & Utilization**
   - `balance`, `principal_outstanding`, `accrued_interest`, `penalty_charges`
   - `credit_limit`, `utilization_pct`

3. **Internal Payments**
   - `payment_count_{30d|90d|180d|365d}`
   - `payment_amt_{30d|90d|180d|365d}`
   - `days_since_last_payment`, `last_payment_amount`

4. **Actions & Engagement**
   - `contacts_made_{30d|90d}`, `calls_connected`, `outbound_calls_made`
   - `call_response_rate`, `sms_response_rate`
   - `ptp_made`, `ptp_kept`, `ptp_kept_rate`
   - `days_since_last_contact`, `last_contact_outcome`

5. **Transaction-Spend (Optional)**
   - `transaction_count_{30d|90d}`, `spend_amt_{30d|90d}`
   - If available from transactional data

6. **Bureau Exposure & Delinquency**
   - `bureau_total_outstanding`, `bureau_monthly_instalment`
   - `bureau_secured_loan_flag`, `bureau_secured_outstanding`
   - `bureau_delinquent_other`, `bureau_active_loan_count`
   - `bureau_new_loan_12m`

7. **Bureau Velocity**
   - `bureau_new_loan_12m`, `bureau_enquiries_3m`

8. **Avoidance Flags**
   - `wrong_number_flag`, `dispute_flag`, `lawyer_mentioned`, `sms_opt_out`

## Reused Modules

The following modules are reused unchanged from the existing decision_agent system:

1. `bureau_features.py` - NCB/TUEF bureau feature extraction
2. `delinquency_features.py` - DPD trajectory and roll rate features
3. `billing_cycle_features.py` - Billing cycle alignment features
4. `collection_action_aggregator.py` - Contact action aggregation
5. `collections_feature_pipeline.py` - Main feature pipeline orchestrator
6. `pipeline_adapter.py` - Feature pipeline adapter
7. `roll_rate_labeller.py` - Label generation for model training
8. `data_contract.py` - Input data validation
9. `legal_queue_manager.py` - Legal case routing (optional)

These are symlinked into `recovery_agent_practical/features/` and can be used directly.

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Quick Start

```python
from recovery_agent_practical import RecoveryAgentPipeline
import pandas as pd

# 1. Load features (from your feature pipeline)
features_df = pd.read_parquet("features_snapshot_2024-01-15.parquet")

# 2. Load training labels (historical recovery outcomes)
labels_df = pd.read_parquet("training_labels_6m.parquet")
# columns: account_id, recovery_amount_180d

# 3. Initialize pipeline
pipeline = RecoveryAgentPipeline(scorecard_model_type="TWO_PART")

# 4. Train scorecard (one-time)
metrics = pipeline.train_scorecard(features_df, labels_df, val_split=0.2)
print(f"Validation AUC: {metrics['val'].auc_roc}")

# 5. Score daily batch
daily_scores = pipeline.score_batch(
    features_df=features_df,
    score_date="2024-01-15",
    write_outputs=True,
    output_path="./outputs"
)

# 6. View results
print(daily_scores[["account_id", "persona", "score_band", "recommended_action"]].head(10))
```

## Operational Workflow

### Daily Scoring Process

```bash
# 1. Generate features for today's snapshot
python -m recovery_agent_practical.features.collections_feature_pipeline \
    --snapshot_date 2024-01-15 \
    --output features_2024-01-15.parquet

# 2. Run scoring pipeline
python -m recovery_agent_practical.pipeline \
    --features features_2024-01-15.parquet \
    --score_date 2024-01-15 \
    --output ./outputs

# 3. Load daily scoring table to database
python load_to_db.py \
    --file ./outputs/daily_scoring_2024-01-15.parquet \
    --table collections.daily_scores
```

### Model Retraining

```bash
# Monthly retraining with updated labels
python train_scorecard.py \
    --features features_train.parquet \
    --labels labels_6m.parquet \
    --model_type TWO_PART \
    --output ./models/scorecard_2024-02.pkl
```

## Monitoring & Model Performance

### Key Metrics to Track

1. **Score Band Stability**
   - Track % of accounts in each band (HOT/WARM/COLD/FROZEN) over time
   - Alert if distribution shifts significantly

2. **Persona Stability**
   - Monitor persona distribution trends
   - Identify emerging behavioral patterns

3. **Action Distribution**
   - Track recommended action mix
   - Ensure routing rules are balanced

4. **Recovery Performance**
   - Compare predicted vs. actual recovery at 6 months
   - Calculate MAE, RMSE, AUC for model performance
   - Segment performance by persona, score band, action

5. **Calibration**
   - Plot predicted P(recovery) vs. observed rate
   - Ensure model is well-calibrated across score bands

### Monthly Model Review

```python
from recovery_agent_practical.monitoring import ModelPerformanceReport

# Generate monthly performance report
report = ModelPerformanceReport(
    scoring_table="collections.daily_scores",
    outcome_table="collections.recovery_outcomes",
    month="2024-01"
)

report.generate_html("./reports/performance_2024-01.html")
```

## Configuration

### Model Hyperparameters

```python
# Two-part model
classifier_params = {
    "n_estimators": 100,
    "max_depth": 5,
    "learning_rate": 0.1,
    "min_samples_leaf": 50,
}

regressor_params = {
    "n_estimators": 100,
    "max_depth": 5,
    "learning_rate": 0.1,
    "min_samples_leaf": 50,
}

scorecard = RecoveryScorecard6M(
    model_type="TWO_PART",
    model_params={
        "classifier_params": classifier_params,
        "regressor_params": regressor_params,
    }
)
```

### Business Rules

```python
# Action routing thresholds
router = ActionOverlayRouter(
    small_balance_threshold=50_000,     # THB
    medium_balance_threshold=200_000,   # THB
    stale_chargeoff_months=18,          # months
)

# Payment threshold for recovery
pipeline = RecoveryAgentPipeline(payment_threshold=500.0)  # THB
```

## Comparison: NPV-Driven vs. Rule-Based

| Aspect | NPV-Driven (decision_agent) | Rule-Based (recovery_agent_practical) |
|--------|----------------------------|--------------------------------------|
| **Optimization** | NPV maximization | Rule-based heuristics |
| **Data Requirements** | Historical offer outcomes, recovery curves | Historical behavior only |
| **Complexity** | High (paydown curves, affordability, NPV calc) | Low (persona → score → action) |
| **Interpretability** | Black box NPV | Fully transparent rules |
| **Deployment** | Requires calibrated data | Works with defaults |
| **Maintenance** | Continuous recalibration | Periodic rule review |
| **Best For** | Mature collections with history | New programs, operational simplicity |

## Roadmap

### V1.0 (Current)
- ✅ Rule-based persona builder
- ✅ Two-part + Tweedie recovery scorecard
- ✅ Rule-based action overlay
- ✅ Daily scoring pipeline
- ✅ Output schemas

### V1.1 (Planned)
- [ ] Model monitoring dashboard
- [ ] Calibration analysis tools
- [ ] A/B test framework for rule changes
- [ ] Integration with CRM systems

### V2.0 (Future)
- [ ] NPV optimization layer (optional)
- [ ] Real-time scoring API
- [ ] Automated model retraining
- [ ] Champion/challenger model framework

## License

Internal use only - CardX Collections Team

## Contact

For questions or issues, contact the Data Science team.
