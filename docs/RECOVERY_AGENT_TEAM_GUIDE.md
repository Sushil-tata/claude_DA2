# CardX Recovery NBA — Agent Pipeline: Team Guide

**Branch:** `feature/agent-layer`
**Audience:** Data Science team (DS engineers + DS leads)
**Last updated:** 2026-03-30

---

## 1. What is this system?

A 9-agent pipeline that runs daily and produces a **Next Best Action (NBA)** decision for every delinquent credit card account in the recovery portfolio. Each account gets:

- A recommended action (DIGITAL_NUDGE / AGENT_CALL / AGENCY / LEGAL / HOLD)
- An optimal discount offer (0–60%)
- A contact channel and timing
- A human-readable explanation for the collector dashboard

The pipeline does **not** create features or re-run propensity models. Those are handled by your existing enterprise feature notebook and recovery-engine-v2 (which run upstream). The agents read pre-computed scores, derive segmentation, compute ERV, apply business rules, and write final decisions.

---

## 2. Architecture at a glance

```
Your enterprise feature notebook (runs separately)
  └─→ cdx_mdz_prd.feature_store (100+ features)

recovery-engine-v2 (runs at 02:00 daily)
  └─→ recovery.model_scores
        (propensity_30d, propensity_180d, account_id, score_date, ...)

Agent pipeline (runs at 03:00 daily)
  └─→ [1] DataQualityAgent    validates model_scores
  └─→ [2] FeatureAgent        joins feature store + derives segments
  └─→ [3] HoldoutAgent        tags 5% control group per segment
  └─→ [4] ModelAgent    ─┐    computes ERV per action × discount
  └─→ [5] ConstraintAgent─┘   applies business rules (parallel with [4])
  └─→ [6] DecisionAgent       merges model + constraints, calls Claude for edge cases
  └─→ [7] ValidationAgent     checks decisions before any action is taken
  └─→ [8] ExplainAgent        writes collector-facing explanations
  └─→ recovery.nba_decisions  ← collector dashboard reads this
```

---

## 3. Key concepts your team needs to know

### SIGNAL_SEGMENT (A/B/C/D)
Describes **how much data we have** on each account. Determined by FeatureAgent.

| Segment | CardX signal | Bureau (NCB) | Strategy |
|---------|-------------|--------------|----------|
| A | Available | Available (pulled within 60 days) | Maximise recovery amount, minimal discount |
| B | Available | Not available | Balanced ERV optimisation |
| C | Not available | Available | Reactivation focus, accept higher discount |
| D | Neither available | Neither available | Exploration only — low-cost actions |

"CardX signal available" = `propensity_30d` is not null.
"Bureau available" = `bureau_pull_date` within 60 days OR `ncb_tradeline_count > 0`.

The 60-day bureau lag is configurable: set `BUREAU_SIGNAL_LAG_DAYS` env var.

### BEHAVIOURAL_PERSONA (Cooperative / Stressed / Sporadic / Disconnected)
Describes **how the customer behaves**. Derived by clustering within each SIGNAL_SEGMENT.

| Persona | willingness | capacity |
|---------|------------|---------|
| Cooperative | High (≥0.5) | High (≥0.5) |
| Stressed | High | Low |
| Sporadic | Low | High |
| Disconnected | Low | Low |

Personas are derived from KMeans clustering on `propensity_30d`, `propensity_180d`, `willingness_score`, `capacity_score`. When no trained cluster model exists, rule-based thresholds are used as fallback.

### ERV (Expected Recovery Value)
The core metric for action selection. Computed per action × discount level:

```
ERV(action, d) = P(pay | d) × E(amount) × (1 − d) − cost(action)
```

- `P(pay | d)` = propensity × elasticity adjustment for discount
- `E(amount)` = predicted recovery amount (from AmountModelTrainer, or outstanding_balance as fallback)
- `d` = discount fraction (0 to 0.60)
- `cost(action)` = DIGITAL_NUDGE: 10 THB, AGENT_CALL: 150, AGENCY: 500, LEGAL: 2,000

The action + discount with **highest net ERV** is selected. LEGAL is only selected over AGENCY if ERV(LEGAL) > ERV(AGENCY) + 5,000 THB.

### Propensity signals
- `propensity_30d` (P_1M) — primary signal for DIGITAL_NUDGE and AGENT_CALL decisions
- `propensity_180d` (P_6M) — used for AGENCY and LEGAL decisions (longer horizon)

Both come **pre-computed** from recovery-engine-v2. Agents never re-train or re-score propensity.

---

## 4. Delta tables — what goes where

| Table | Written by | Read by | Contents |
|-------|-----------|---------|---------|
| `recovery.model_scores` | recovery-engine-v2 | DataQualityAgent | Propensity scores + raw features |
| `recovery.feature_output` | FeatureAgent | ModelAgent, ConstraintAgent | Enriched features + segments |
| `recovery.model_agent_output` | ModelAgent | DecisionAgent | ERV results + recommended action |
| `recovery.constraint_overrides` | ConstraintAgent | DecisionAgent | Business rule overrides |
| `recovery.nba_decisions` | DecisionAgent | Collector dashboard | Final NBA per account |
| `recovery.validation_results` | ValidationAgent | Monitoring | Decision audit |
| `recovery.nba_explanations` | ExplainAgent | Collector dashboard | Human-readable text |
| `recovery.treatment_log` | ModelAgent + HoldoutAgent | Uplift model training | Every action decision, append-only |
| `recovery.holdout_assignments` | HoldoutAgent | ValidationAgent | Control group assignments |
| `recovery.agent_audit_log` | All agents | Debugging | Agent status signals |

---

## 5. How to run the pipeline

### One-time setup (do this once per environment)

**Step 1 — Confirm the branch and install the wheel**
```bash
git checkout feature/agent-layer
pip wheel . -w dist/
databricks fs cp dist/decision_agent*.whl dbfs:/FileStore/wheels/decision_agent-latest.whl
```

**Step 2 — Create the recovery database**
Run this once in a Databricks notebook:
```sql
CREATE DATABASE IF NOT EXISTS recovery;
-- Tables are created automatically on first write by AgentMemory
```

**Step 3 — Connect your enterprise feature store**
In your Databricks cluster configuration (Environment Variables):
```
ENTERPRISE_FEATURE_TABLE = cdx_mdz_prd.feature_store
ENTERPRISE_FEATURE_JOIN_KEYS = account_id,score_date
```
This is the only configuration needed to connect your existing 100+ feature pipeline.

**Step 4 — Set model directories** (once models are trained — see Section 7)
```
PERSONA_MODEL_DIR     = /dbfs/FileStore/models/persona_clusters/
WILLINGNESS_MODEL_DIR = /dbfs/FileStore/models/willingness_model/
CAPACITY_MODEL_DIR    = /dbfs/FileStore/models/capacity_model/
AMOUNT_MODEL_DIR      = /dbfs/FileStore/models/amount_model/
ELASTICITY_MODEL_DIR  = /dbfs/FileStore/models/elasticity_model/
UPLIFT_MODEL_DIR      = /dbfs/FileStore/models/uplift_models/
```

---

### Running a single agent (testing / debugging)

The entry point is `jobs/run_agent.py`. Every agent uses the same command pattern:

```bash
python jobs/run_agent.py \
  --agent <agent_name> \
  --execution-date 2026-03-30 \
  [--dry-run]   # skips all Delta writes — safe to run anytime
  [--local]     # uses /tmp/agent_data instead of Delta Lake (no Spark needed)
```

**Agent names:** `data_quality`, `feature`, `holdout`, `model`, `capacity`, `constraint`, `decision`, `validation`, `explain`

**Examples:**
```bash
# Test DataQualityAgent without writing anything
python jobs/run_agent.py --agent data_quality --execution-date 2026-03-30 --dry-run

# Run FeatureAgent in local mode (no Spark, writes to /tmp/agent_data)
python jobs/run_agent.py --agent feature --execution-date 2026-03-30 --local

# Run the full sequence manually (in order)
for agent in data_quality feature holdout model constraint decision validation explain; do
  echo "Running $agent..."
  python jobs/run_agent.py --agent $agent --execution-date 2026-03-30
done
```

---

### Running in Databricks Workflows (production)

The workflow DAG is defined in `databricks/workflows/agentic_nba_workflow.yml`.

**Schedule:** 03:00 Bangkok time daily (after recovery-engine-v2 finishes at 02:00)

**To trigger manually in Databricks UI:**
1. Go to Workflows → `cardx_agentic_nba_pipeline`
2. Click **Run now**
3. Override `execution_date` if testing a past date
4. Set `dry_run=true` to test without writing final decisions

**Key workflow parameters:**

| Parameter | Default | What it does |
|-----------|---------|-------------|
| `execution_date` | today | Scoring date (YYYY-MM-DD) |
| `dry_run` | false | Skip final Delta writes |
| `champion_challenger_enabled` | true | Activate holdout control group |
| `holdout_pct` | 5 | % of each segment assigned to control group |

---

## 6. Reading results

### Check if the pipeline ran successfully
```sql
-- See agent status for today's run
SELECT agent_name, status, message, timestamp
FROM recovery.agent_audit_log
WHERE execution_date = '2026-03-30'
ORDER BY timestamp;
```

### Check final decisions
```sql
-- Action distribution today
SELECT recommended_action, signal_segment, COUNT(*) as n_accounts
FROM recovery.nba_decisions
WHERE execution_date = '2026-03-30'
GROUP BY 1, 2
ORDER BY signal_segment, n_accounts DESC;
```

### Check ERV and discount levels
```sql
-- Avg ERV and discount by segment + persona
SELECT
  signal_segment,
  behavioural_persona,
  ROUND(AVG(erv_at_d_optimal), 0)  AS avg_erv_thb,
  ROUND(AVG(d_optimal) * 100, 1)   AS avg_discount_pct,
  COUNT(*)                          AS n_accounts
FROM recovery.nba_decisions
WHERE execution_date = '2026-03-30'
GROUP BY 1, 2
ORDER BY 1, 2;
```

### Check constraint overrides
```sql
-- Which accounts were overridden by business rules and why
SELECT override_reason, COUNT(*) AS n
FROM recovery.constraint_overrides
WHERE execution_date = '2026-03-30'
  AND constraint_override_flag = true
GROUP BY 1
ORDER BY 2 DESC;
```

### Read the collector explanations
```sql
-- What the dashboard shows for a specific account
SELECT account_id, explanation_text, recommended_action, erv_at_d_optimal
FROM recovery.nba_explanations
WHERE execution_date = '2026-03-30'
  AND account_id = 'ACC_12345';
```

---

## 7. Model training sequence

All models are **optional at cold start** — the pipeline runs with rule-based fallbacks until each model is trained. Train them in this order as data becomes available.

### What runs without any trained models (cold start)
- SIGNAL_SEGMENT — fully working (pure rule-based)
- BEHAVIOURAL_PERSONA — rule-based thresholds (willingness/capacity ≥ 0.5)
- ERV computation — uses `outstanding_balance` as amount proxy
- Propensity — already pre-computed by recovery-engine-v2 (no training needed here)

---

### Week 1–2: Willingness and Capacity models

**Train WillingnessModelTrainer** — needs CRM contact log

```python
from src.models.willingness_model_trainer import WillingnessModelTrainer

# Required columns in contact_log_df:
#   contact_responded (0/1) — label: did customer respond within 14 days?
#   contact_attempts_30d, contact_success_rate_3m, days_since_last_response,
#   broken_promise_count_3m, partial_payment_count_3m, digital_open_rate_3m
# Filter: only rows where attempted_contact = 1

trainer = WillingnessModelTrainer(response_window_days=14)
result = trainer.fit(contact_log_df)
print(f"AUC: {result['auc']} | Response rate: {result['response_rate']:.1%}")
print(f"Top features: {result['top_features'][:5]}")
```

**Train CapacityModelTrainer** — needs bureau + card payment data

```python
from src.models.capacity_model_trainer import CapacityModelTrainer

# Required columns in bureau_card_df:
#   made_any_payment_30d (0/1) — label: any payment within 30 days?
#   ncb_other_accounts_current, ncb_total_revolving_util, ncb_active_tradelines,
#   ncb_enquiry_count_3m, card_payment_pct_minimum_3m,
#   card_months_since_last_payment, card_balance_to_limit_ratio, dpd_current
# Filter: active_contact_period = 1 only (no HOLD/no-contact accounts)
# NOTE: No deposit/CASA signals — CardX does not have access to these

trainer = CapacityModelTrainer(outcome_window_days=30)
result = trainer.fit(bureau_card_df)
print(f"AUC: {result['auc']} | Payment rate: {result['payment_rate']:.1%}")
```

---

### Week 3: Persona clustering (requires outcome labels)

```python
from src.models.persona_cluster_trainer import PersonaClusterTrainer

# feature_output_df must include recovery_30d, recovery_90d, recovery_180d
# These are joined back from the historical outcomes table
# Clustering is done WITHIN each SIGNAL_SEGMENT separately

trainer = PersonaClusterTrainer(
    n_clusters=4,                    # 4 personas per segment
    min_pairwise_recovery_diff=0.05, # clusters must differ by ≥5pp on recovery_180d
)
result = trainer.fit(feature_output_with_outcomes_df)

# Check which segments passed business validation
for seg, r in result.items():
    status = r.get("business_validation", {}).get("recommendation", "SKIPPED")
    saved  = r.get("saved", False)
    print(f"Segment {seg}: {status} | model_saved={saved}")
```

**Business validation rules** — a cluster model is only saved if:
1. Max pairwise diff on `recovery_180d` ≥ 5pp between best and worst cluster
2. Kruskal-Wallis test p < 0.05 on `recovery_180d`
3. Cohen's d ≥ 0.2 between best and worst cluster

If a segment fails, the rule-based persona fallback is used for that segment. Do not force a failing model — reduce `n_clusters` and refit instead.

---

### Month 1: Amount models

```python
from src.models.amount_model_trainer import AmountModelTrainer

# recovered_df = accounts where recovery_amount > 0
# Must include ALL action types — do NOT filter to SETTLE only
# Required: signal_segment, behavioural_persona, discount_offered,
#           propensity_30d, propensity_180d, willingness_score,
#           capacity_score, recovery_amount

for window in [30, 180]:
    trainer = AmountModelTrainer(outcome_window_days=window)
    result = trainer.fit(recovered_df)
    print(f"{window}d | RMSE (THB): {result['rmse_thb']:,.0f} | n_train: {result['n_train']:,}")
```

---

### Month 2–3: Elasticity model

```python
from src.models.elasticity_model_trainer import ElasticityModelTrainer

# offer_df must have: accepted (0/1), discount_offered (0.0–0.60),
#   signal_segment, propensity_30d, willingness_score, capacity_score
# NOTE: Do NOT include erv_at_d_optimal as a feature — circular dependency

trainer = ElasticityModelTrainer()
result = trainer.fit(offer_df)
print(f"AUC: {result['auc']}")
```

---

### Month 3+: Uplift model (requires treatment + control data)

This activates **causal ERV** — accounts where τ(x) ≤ 0 receive HOLD regardless of propensity.

```python
from src.models.uplift_model_trainer import UpliftModelTrainer

# treatment_log_with_outcomes_df:
#   - Pull from recovery.treatment_log (append-only, written by ModelAgent daily)
#   - Join recovery outcomes (paid_30d, paid_90d, paid_180d) after 90+ days
#   - Must have holdout_flag column: 1 = control group, 0 = treated
#   - Need at least 90 days of data (~10,000+ accounts per segment)

trainer = UpliftModelTrainer()
result = trainer.fit(treatment_log_with_outcomes_df)
print(f"Segments trained: {result['segments_trained']}")
```

---

## 8. Monitoring the pipeline health

### Daily checks (automated — DataQualityAgent raises BLOCKED if these fail)

| Check | Threshold | Action if breached |
|-------|----------|-------------------|
| Null rate on `propensity_30d` | > 5% | Pipeline blocked — check recovery-engine-v2 |
| Segment distribution shift | > 10% WoW | Pipeline blocked — investigate upstream data |
| `recovery.model_scores` row count | < 80% of prior week | Pipeline blocked |

### Weekly model health checks (run manually)

```python
from src.models.cluster_business_validator import ClusterBusinessValidator

# Run after joining outcomes (takes 30 days for 30d outcomes)
validator = ClusterBusinessValidator()
result = validator.validate_all_segments(
    df=feature_output_with_outcomes_df,
    cluster_col="behavioural_persona",
    segment_col="signal_segment",
)
print(result["summary"])
# If any segment FAILS → refit PersonaClusterTrainer for that segment only
```

### Check for uplift model degradation (after Phase 2 is active)

```python
from src.models.uplift_evaluator import UpliftEvaluator

evaluator = UpliftEvaluator()
report = evaluator.evaluate(
    treatment_log_with_outcomes_df,
    segment_col="signal_segment",
)
print(report["qini_coefficient"])   # Should be > 0
print(report["negative_uplift_summary"])  # Accounts where model says HOLD
```

---

## 9. Common issues and fixes

### Pipeline BLOCKED — what it means

A BLOCKED signal means the agent detected a data quality problem that makes it unsafe to continue. The Workflow stops and sends an alert to `ds-oncall@cardx.co.th`.

**Check the audit log first:**
```sql
SELECT agent_name, status, message
FROM recovery.agent_audit_log
WHERE execution_date = '2026-03-30'
  AND status IN ('BLOCKED', 'FAILED');
```

| Error message | Root cause | Fix |
|---------------|-----------|-----|
| `feature_output is empty` | FeatureAgent failed or didn't run | Re-run FeatureAgent |
| `null_rate > 5% on propensity_30d` | recovery-engine-v2 scoring issue | Check upstream job |
| `PIT violation: N rows have score_date > execution_date` | Future-dated data in model_scores | Check upstream data pipeline |
| `No fitted model for segment X` | Persona model not trained yet | Expected — rule-based fallback is used |
| `Enterprise feature join failed` | Wrong table name or join key mismatch | Check ENTERPRISE_FEATURE_TABLE env var |

---

### Persona clusters failing business validation

```
Segment B: business validation FAILED — recommendation: REDUCE_CLUSTERS
Model NOT saved for this segment. Rule-based persona fallback will be used.
```

**This is expected and correct behaviour.** Clusters that don't differ on recovery_180d by ≥5pp are rejected. The rule-based fallback runs instead.

**Fix options:**
1. Reduce `n_clusters` for that segment: `PersonaClusterTrainer(n_clusters=3)`
2. Collect more labelled outcome data and refit
3. For Segment D (no signal), use `n_clusters=2` — less signal = less resolution

---

### Running in local mode (no Databricks)

```bash
# Run with local parquet instead of Delta Lake
python jobs/run_agent.py \
  --agent feature \
  --execution-date 2026-03-30 \
  --local

# Data is written to /tmp/agent_data/ by default
# Override: AGENT_LOCAL_DATA_DIR=/your/path
```

This is useful for unit testing and CI/CD. All agents support local mode.

---

## 10. Configuration reference

All key settings can be set as environment variables in Databricks cluster config. No code changes needed.

| Env variable | Default | What it controls |
|-------------|---------|-----------------|
| `ENTERPRISE_FEATURE_TABLE` | (none) | Your DS team's feature store Delta table |
| `ENTERPRISE_FEATURE_JOIN_KEYS` | `account_id,score_date` | Join keys for feature store |
| `BUREAU_SIGNAL_LAG_DAYS` | `60` | Days before bureau signal is considered stale |
| `PERSONA_MODEL_DIR` | `models/persona_clusters` | Path to trained persona cluster models |
| `WILLINGNESS_MODEL_DIR` | `models/willingness_model` | Path to willingness model |
| `CAPACITY_MODEL_DIR` | `models/capacity_model` | Path to capacity model |
| `AMOUNT_MODEL_DIR` | `models/amount_model` | Path to 30d + 180d amount models |
| `ELASTICITY_MODEL_DIR` | `models/elasticity_model` | Path to elasticity model |
| `UPLIFT_MODEL_DIR` | `models/uplift_models` | Path to T-learner uplift models |
| `AGENT_LOCAL_DATA_DIR` | `/tmp/agent_data` | Local parquet base path (non-Databricks) |

---

## 11. Who owns what

| Component | Owner | When to change |
|-----------|-------|---------------|
| Enterprise feature notebook | DS engineer | When new features added for propensity model |
| recovery-engine-v2 | DS lead | When propensity model retrained |
| Agent pipeline code (`src/agents/`) | DS lead | Pipeline logic changes |
| Model training (`src/models/`) | DS engineer | Model retraining |
| Business rules (`constraint_agent.py`) | DS lead + Credit Risk | When credit policy changes |
| Workflow DAG (`databricks/workflows/`) | DS lead | When new agent added or schedule changes |
| `recovery.nba_decisions` | Agent pipeline | Read-only for downstream consumers |

---

## 12. Maturity roadmap — what activates automatically

Each model activates automatically when its `.pkl` file is present in the configured directory. No code changes needed.

| Model | Status at cold start | Activates when |
|-------|---------------------|---------------|
| Propensity (30d, 180d) | ACTIVE — from recovery-engine-v2 | Already running |
| SIGNAL_SEGMENT | ACTIVE — rule-based | Always |
| BEHAVIOURAL_PERSONA | Rule-based fallback | PersonaClusterTrainer trained + business validated |
| Willingness score | Default 0.5 | WillingnessModelTrainer fitted |
| Capacity score | Default 0.5 | CapacityModelTrainer fitted |
| E(amount) | Uses outstanding_balance | AmountModelTrainer 30d + 180d fitted |
| Elasticity | Alpha curve fallback | ElasticityModelTrainer fitted |
| Causal ERV (uplift) | Predictive ERV mode | UpliftModelTrainer fitted (90+ days of data) |

---

*Questions or issues: raise in `#ds-recovery-agent` Slack channel or open a GitHub issue on `claude_DA2`.*
