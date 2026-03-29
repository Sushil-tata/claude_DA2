# CLAUDE.md — Decision Agent V2 / Advanced Platform (Desktop/claude)

## Project Overview

This is the **V2 advanced platform** — the more evolved, later-generation successor to DeepLearning/claude. It is built on a **Databricks-native stack** (PySpark, Delta Lake, MLflow, Databricks Workflows) and contains several unique components that do **not** exist in the GitHub baseline repo.

- **GitHub repo**: https://github.com/Sushil-tata/claude_DA2 (this codebase)
- **Domain**: Retail credit risk decision intelligence, SCB/CardX Thailand
- **Platform**: Databricks (PySpark + Delta Lake + MLflow)
- **Status**: Phase 1–3 complete + operationalization complete as of **2026-02-10**
- **Active branch for new work**: `recovery_agent_practical`

## What Makes V2 Different from V1 (DeepLearning/claude)

| Dimension | V1 (DeepLearning/claude) | V2 (Desktop/claude — THIS REPO) |
|---|---|---|
| Stack | Python/Pandas | Databricks-native PySpark + Delta Lake |
| Phases | Single build | 3 phases + operationalization |
| Behavioral Physics | Not present | Novel physics-based feature factory (partial) |
| Graph Fraud | Not present | POC brief written, code not yet built |
| Recovery Agent | Basic | Full 3,500-line operational V1 |
| BFE | Not present | BFE v1.3 (317 features, production-ready) |
| Streaming | Not present | Phase 3 includes streaming ingestion |

---

## Unique Components in This Repo

### 1. Behavioral Physics Feature Factory (`behavioral_physics_features/`)

**Novel research component.** Applies physics concepts to credit risk feature engineering — treating customer financial behavior as a physical system with measurable dynamics.

**Core Idea**: Instead of static aggregates (avg balance last 3M), compute dynamic state descriptors (acceleration of delinquency trajectory, entropy of payment timing, diffusion rate of utilization spread).

#### State Framework
| State | DPD Range | Meaning |
|---|---|---|
| S0 | 0 DPD | Current |
| S1 | 1–30 DPD | Early delinquency |
| S2 | 31–90 DPD | Moderate delinquency |
| S3 | 91–180 DPD | Serious delinquency |
| S4 | 181+ DPD | Chronic / write-off risk |

#### Physics Concepts Applied
- **Velocity**: rate of state change over time (DPD velocity, utilization velocity)
- **Acceleration**: second derivative — is delinquency worsening faster or slowing?
- **Entropy**: disorder in payment timing, amount variability
- **Diffusion**: spread of credit utilization across products/facilities
- **State Traps**: probability of getting stuck in S2/S3 (absorbing-state-like behavior)

#### Implementation Status
- **3/8 modules complete** (code exists):
  - `config.py` — 600 lines, all physics parameters and state definitions
  - `state_builder.py` — 450 lines, DPD state assignment and transition matrix construction
  - `trajectory_engine.py` — 500 lines, trajectory feature computation (velocity, acceleration)
- **5 modules fully specified** in `behavioral_physics_features/COMPLETE_IMPLEMENTATION.md` (ready to code):
  - `lender_ecology.py` — multi-lender interaction patterns
  - `repayment_dynamics.py` — payment rhythm and cycle analysis
  - `enquiries_engine.py` — enquiry velocity and clustering
  - `feature_registry.py` — feature catalog and metadata
  - `main_pipeline.py` — orchestration pipeline
- **Expected lift**: +15–30% AUC over static aggregate baseline
- **Branch**: `recovery_agent_practical`

### 2. CARDX_GRAPH_FRAUD_BRIEF.md — Graph-Based Fraud Detection POC

**Status: Brief written, NO CODE built yet.** This is the next major component to build.

**Concept**: Model fraud as a graph problem where entities and their relationships reveal fraud rings, ATO, and synthetic identity patterns that are invisible to per-account scoring.

#### Node Types
| Node | Description |
|---|---|
| CUSTOMER | Account holder |
| CARD | Physical/virtual card |
| MERCHANT | Transaction counterparty |
| DEVICE | Device fingerprint |
| IP_ADDRESS | Network identity |

#### Target Fraud Patterns
- **Fraud rings**: coordinated groups sharing devices/IPs/merchants
- **Card testing**: rapid low-value transactions at known test merchants
- **Account Takeover (ATO)**: device/IP change + behavioral shift
- **Merchant compromise**: spike in chargebacks from a single merchant subgraph
- **Mule accounts**: fund-flow patterns from compromised to mule nodes

#### Expected Impact
- +5–8% AUC improvement over standalone transaction scoring
- 70–85% fraud ring detection rate
- Read `CARDX_GRAPH_FRAUD_BRIEF.md` in repo root for full spec before building

### 3. Recovery Agent Practical (`src/recovery_agent_practical/`)

**Status: ~3,500 lines, fully operational V1 (rule-based, no NPV optimization yet).**

A complete collections decision agent for managing delinquent credit card accounts.

#### Customer Personas (5)
| Persona | Profile |
|---|---|
| ACTIVE_PAYER | Recently active, likely to self-cure |
| SELECTIVE_DEFAULTER | Can pay but chooses not to on some accounts |
| LIQUIDITY_CONSTRAINED | Willing to pay, genuinely unable right now |
| STRATEGIC | Gaming the system, delay tactics |
| DORMANT | No contact, no payment, extended inactivity |

#### Two-Part Recovery Scorecard
- **P(pay)**: probability of any payment in next 30 days
- **E(amount | paid)**: expected recovery amount given payment occurs
- Combined score: `Recovery_Score = P(pay) × E(amount|paid)` → drives action prioritization

#### Action Set
| Action | Trigger Condition |
|---|---|
| SETTLEMENT_LUMP | High P(pay), high capacity, S3/S4 |
| SETTLEMENT_PLAN | Moderate P(pay), liquidity-constrained persona |
| AGENCY | Low P(pay), extended dormancy |
| LEGAL_REVIEW | S4 + strategic persona + high balance |
| HOLD | Active payer, monitoring phase |

#### Key Files
```
src/recovery_agent_practical/
  pipeline.py           — main orchestration
  segmentation/         — persona classifier
  scoring/              — two-part recovery scorecard
  routing/              — action assignment logic
  features/             — collections-specific features
  legal_queue_manager.py — legal case management
  roll_rate_labeller.py  — DPD state labeling
  data_contract.py      — schema validation
  example_usage.py      — runnable demo
```

### 4. Bureau Feature Engine (BFE) v1.3

- **Location**: `behavioral_physics_features/` (NCB/bureau module)
- **Implementation**: `ncb_feature_factory_v2.py` (Pandas-based, production-ready)
- **317 features** derived from NCB (National Credit Bureau) Thailand bureau data
- Covers: tradeline counts, DPD history, enquiry patterns, limit utilization, installment/revolving split, XXX-handling (masked/missing bureau data), Thai lender classification
- See `BUREAU_MODULE_AUDIT.md` and `BFE_v1.3_DELIVERY.md` for full feature list and validation results

---

## Platform Architecture (3 Phases + Operationalization)

### Phase 1 — 16 Core Modules
Foundation: data ingestion, feature store, model registry, decision engine core, explainability, monitoring scaffolding.

### Phase 2 — NBA Debt Collection
Full Next-Best-Action engine for collections: persona detection, action scoring, contact strategy, channel optimization.

### Phase 3 — Advanced ML + Streaming
Streaming feature computation (real-time DPD updates), online learning hooks, advanced ensemble logic, A/B testing framework.

### Operationalization (Complete)
MLflow experiment tracking, Databricks Workflows DAG, Delta Lake feature store, deployment checklist. See `OPERATIONALIZATION_COMPLETE.md` and `DEPLOYMENT_CHECKLIST.md`.

---

## Directory Structure

```
src/
  agent/                  — NBA engine
  decision_agent/         — orchestration
  features/               — feature engineering modules
  models/                 — model classes
  recovery_agent_practical/ — collections agent (see above)
  validation/             — OOT, PSI, stress testing
  privacy/                — compliance, PII, audit
  production/             — serving, monitoring
behavioral_physics_features/  — physics feature factory (partial, see above)
decision_engine/
  feature_store/          — Delta Lake feature store interface
config/, conf/            — Databricks + environment configs
databricks/               — Databricks-specific notebooks and configs
databricks_notebooks/     — Notebook exports
jobs/                     — Workflow DAG definitions
deployment/               — CI/CD, deployment scripts
docs/                     — Architecture docs
tests/                    — Unit + integration tests
```

---

## Key Design Decisions

1. **Databricks-native**: All production paths use PySpark + Delta Lake. Pandas only in local dev/BFE.
2. **MLflow for all experiments**: every model training run tracked; no untracked experiments.
3. **Delta Lake as feature store**: time-travel queries for PIT-safe feature retrieval.
4. **Rule-based Recovery Agent V1**: NPV optimization (dynamic programming / RL) is intentionally deferred to V2 of the recovery agent — V1 establishes the data contract and action set first.
5. **Physics features isolated**: behavioral_physics_features/ is a self-contained module — it can be added to any pipeline without modifying core agent logic.

---

## Current Status

| Component | Status |
|---|---|
| Phase 1 (16 core modules) | Complete |
| Phase 2 (NBA debt collection) | Complete |
| Phase 3 (advanced ML + streaming) | Complete |
| Operationalization | Complete |
| BFE v1.3 | Complete, production-ready |
| Recovery Agent Practical | Complete (~3,500 lines) |
| Behavioral Physics (3/8 modules) | Partial — 5 modules specified, not coded |
| Graph Fraud POC | Brief only — NO CODE |

---

## What's Next (Picking Up Where Left Off)

### Priority 1: Build Graph Fraud POC Code
- Read `CARDX_GRAPH_FRAUD_BRIEF.md` first — full spec is there
- Recommended library: NetworkX for POC, then GraphFrames (PySpark) for scale
- Start with node/edge schema, then fraud ring detection (connected components + Louvain clustering)
- Add GNN layer (Graph SAGE or GCN) if POC graph features show lift

### Priority 2: Complete Remaining 5 Behavioral Physics Modules
- Spec is in `behavioral_physics_features/COMPLETE_IMPLEMENTATION.md`
- Implement in order: `feature_registry.py` → `repayment_dynamics.py` → `enquiries_engine.py` → `lender_ecology.py` → `main_pipeline.py`
- All 5 must integrate with `trajectory_engine.py` (already built) as the upstream dependency

### Priority 3: Test with Real CardX Data
- Connect to `cdx_mdz_prd` tables in Databricks
- Validate BFE v1.3 feature distributions on live NCB data
- Run OOT validation on Recovery Agent using real labeled collections data

---

## Important Context for Next Session

- **Branch for active development**: `recovery_agent_practical` (all new components are on this branch)
- **XXX handling**: Thai NCB data uses "XXX" as a placeholder for masked/missing values — BFE has special handling for this; never treat XXX as a string or null without going through the BFE null-handling logic
- **Thai lender classification**: Thai NCB encodes lender types differently from international bureaus — `THAI_LENDER_CLASSIFICATION.md` has the mapping
- **GDZ constraint**: Some CardX data lives in GDZ (Governed Data Zone) where compute is limited — use LightGBM/Pandas, not heavy Spark, for GDZ-bound workflows
- **V1 vs V2 relationship**: DeepLearning/claude (GitHub: claude) is the clean V1 baseline. This repo (Desktop/claude, GitHub: claude_DA2) is V2 with more advanced components. When merging features, V2 takes precedence.
