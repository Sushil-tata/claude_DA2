# Behavioral Physics Feature Factory - Architecture

**Version:** 1.0.0
**Technology:** PySpark 3.x
**Paradigm:** Behavioral Physics Modeling

---

## Philosophy

Traditional bureau features are **static aggregates** (sum, count, avg). This factory creates **dynamic features** modeling borrower behavior using physics concepts:

- **Velocity**: Rate of change (DPD slope, payment decline)
- **Acceleration**: Rate of velocity change (deterioration shock)
- **Inertia/Stickiness**: Resistance to state change (bad-state trap)
- **Friction**: Effort to cure (payment/cure ratio)
- **Shocks**: Sudden regime shifts (clean→stressed)
- **Diffusion**: Spread across lenders (synchronized delinquency)
- **Phase Transitions**: State boundary crossings (S0→S2 jump)
- **Entropy**: Behavioral volatility/unpredictability

---

## State Framework

**5-State Model** (by DPD):
- **S0**: 0 DPD (CLEAN)
- **S1**: 1-30 DPD (EARLY STRESS)
- **S2**: 31-90 DPD (SUB-STANDARD)
- **S3**: 91-180 DPD (DOUBTFUL)
- **S4**: 181+ DPD (LOSS)

**Regimes**:
- **NORMAL**: S0, S1 (healthy repayment dynamics)
- **STRESSED**: S2, S3, S4 (cure attempts, fatigue)

---

## Module Architecture

```
behavioral_physics_features/
├── modules/
│   ├── config.py                  # Windows, thresholds, mappings
│   ├── state_builder.py           # Monthly state assignment (S0-S4)
│   ├── trajectory_engine.py       # Velocity, acceleration, transitions, entropy
│   ├── lender_ecology.py          # PSU/FINTECH/CARDX, diffusion, HHI
│   ├── repayment_dynamics.py      # NORMAL vs STRESSED regime behavior
│   ├── enquiries_engine.py        # Enquiry velocity, conversion
│   ├── feature_registry.py        # Feature catalog + metadata
│   └── main_pipeline.py           # Orchestration + QA
├── tests/
│   ├── test_state_builder.py
│   ├── test_trajectory.py
│   └── test_integration.py
└── docs/
    ├── ARCHITECTURE.md (this file)
    ├── FEATURE_CATALOG.md
    └── DEPLOYMENT_GUIDE.md
```

---

## Data Flow

```
Input DataFrames:
  ├── bureau_trade_monthly_df     (trade lines)
  ├── bureau_enquiry_df           (inquiries)
  └── cardx_internal_monthly_df   (internal CardX)

         ↓

  StateBuilder
    └── Assigns S0-S4 per (cust_id, as_of_month)

         ↓

  TrajectoryEngine
    └── Velocity, acceleration, transitions, entropy

         ↓

  LenderEcology
    └── Lender type mapping, diffusion, HHI

         ↓

  RepaymentDynamics
    └── NORMAL vs STRESSED regime split

         ↓

  EnquiriesEngine
    └── Velocity, conversion tracking

         ↓

  FeatureRegistry.compute_all()
    └── Executes all feature functions

         ↓

Output:
  ├── monthly_feature_df     (cust_id, as_of_month, 120+ features)
  ├── audit_log_df           (QA metrics)
  └── QA Summary             (printed)
```

---

## Key Innovations

### 1. **State Transition Physics**
- Not just "current DPD" but **transition speed**: S0→S2 in 2 months vs 6 months
- **Cure half-life**: Time to reduce DPD by 50%
- **Bad-state trap probability**: P(stuck in S3/S4 | entered S3)

### 2. **Regime-Dependent Repayment**
- **NORMAL**: Payment consistency, effort score
- **STRESSED**: Cure attempts, payment fatigue (declining payments despite staying stressed)
- **Delta features**: STRESSED_behavior - NORMAL_behavior

### 3. **Lender Diffusion**
- **Synchronized delinquency**: Multiple lenders in S2+ simultaneously
- **Cross-lender spillover**: Fintech stress → CardX stress (lag analysis)
- **Lender migration**: PSU → Fintech shift signals distress

### 4. **Behavioral Entropy**
- **State entropy**: Shannon entropy of state distribution
- **Oscillation volatility**: Frequency of state changes
- **Unpredictability index**: Regime shift randomness

### 5. **Enquiry Intelligence**
- **Velocity**: d(enquiries)/dt (1m, 3m, 6m)
- **Acceleration**: d²(enquiries)/dt² (burst detection)
- **Conversion tracking**: Enquiry → New trade within 30/60 days

---

## Feature Families (120+ Total)

| Family | Count | Examples |
|--------|-------|----------|
| **State Transition** | 20 | transition_speed_s0_s2, cure_halflife, bad_state_trap_prob |
| **Velocity/Acceleration** | 18 | dpd_velocity_3m, dpd_acceleration_6m, shock_flag |
| **Stickiness/Inertia** | 12 | avg_consecutive_months_s3, max_streak_s4, regime_persistence |
| **Entropy/Volatility** | 10 | state_entropy_6m, oscillation_count, volatility_index |
| **Lender Ecology** | 25 | lender_hhi, fintech_share, cardx_first_delinq_flag, diffusion_score |
| **Repayment NORMAL** | 12 | payment_consistency_s0, effort_score_normal, payment_cv_normal |
| **Repayment STRESSED** | 15 | cure_attempt_count, payment_fatigue_slope, last_minute_proxy |
| **Regime Deltas** | 8 | delta_effort_stressed_normal, delta_consistency |
| **Enquiry Dynamics** | 12 | enquiry_velocity_3m, enquiry_acceleration, conversion_rate_30d |
| **CardX Interactions** | 8 | cardx_vs_others_dpd_diff, cross_trigger_flag |

---

## Scalability Features

### Performance
- **Repartitioning**: By (cust_id % 100) for balanced processing
- **Caching**: State assignments cached (reused across modules)
- **Window Functions**: Optimized for Spark (avoid shuffles where possible)
- **Broadcast Joins**: Config tables broadcast

### Data Volume
- Designed for **millions of customers × 24-36 months**
- Supports **incremental processing** (monthly batches)
- **Checkpoint-ready** for fault tolerance

### Validation
- **Train/Test/OOT splits**: Temporal validation (no leakage)
- **Point-in-time filtering**: as_of_month enforced
- **Drift monitoring**: Feature stability checks built-in

---

## Quality Guards

### Data Quality
- **Row count validation**: One row per (cust_id, as_of_month)
- **Missingness reporting**: Top 20 features by null %
- **Outlier winsorization**: 99th percentile caps
- **Schema validation**: Required columns checked

### Leakage Prevention
- **As-of-month filtering**: All data ≤ as_of_month
- **No future information**: Enquiries, trades filtered strictly
- **Audit trail**: Every feature logs data lineage

### Unit Testing
- **Embedded asserts**: Lightweight checks in code
- **Edge case coverage**: Zero trades, single month, etc.
- **Regression tests**: Known cases with expected outputs

---

## Deployment

### Databricks Workflow
```python
# Job definition
{
  "name": "behavioral_physics_monthly",
  "tasks": [
    {
      "task_key": "feature_generation",
      "spark_python_task": {
        "python_file": "main_pipeline.py",
        "parameters": ["--as_of_month", "2024-01-31"]
      }
    }
  ]
}
```

### Output Tables
- **Feature Store**: `feature_store.behavioral_physics_monthly`
- **Audit Log**: `feature_store.behavioral_physics_audit`
- **Metadata**: `feature_store.behavioral_physics_registry`

---

## Comparison: Traditional vs Behavioral Physics

| Aspect | Traditional Bureau | Behavioral Physics |
|--------|-------------------|-------------------|
| **DPD** | Current DPD value | Velocity, acceleration, shock flags |
| **Utilization** | Current ratio | Trajectory, regime shifts |
| **Accounts** | Count by status | Lender diffusion, HHI, migration |
| **Payments** | Sum/avg | Regime-dependent dynamics, fatigue |
| **Enquiries** | Count in window | Velocity, acceleration, conversion |
| **Risk Signal** | Static threshold | Phase transitions, entropy |

**Expected Lift**: 15-30% improvement in model AUC/Gini vs traditional features

---

## Next Steps (Post-Deployment)

1. **A/B Test**: Compare models with/without physics features
2. **Feature Importance**: Identify top 20 drivers
3. **Monitoring**: Track feature drift monthly
4. **Iteration**: Add product-specific physics (credit card vs loan)
5. **Integration**: Merge with pandas BFE module (hybrid approach)

---

**Built for scale, designed for insight, optimized for lift.**
