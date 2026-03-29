# Phase 2: Production Deployment & Monitoring

**Status:** 🚀 STARTING
**Date:** 2026-02-10
**Goal:** Deploy Phase 1 modules to production with safe rollout and monitoring

---

## Phase 2 Objectives

Transform Phase 1's production-ready modules into a **live production system** with:
1. Smart routing for production inference
2. Safe Challenger rollout with shadow mode
3. Explainability for regulatory compliance
4. Production monitoring with automated alerting
5. Databricks deployment automation

---

## Phase 2 Scope

### ✅ Phase 1 Complete (What We Have)
- 16 production modules implemented
- 3 P0 quality gates (data quality, fair lending, champion/challenger)
- Income signals (deposit periodicity, stability)
- Integration complete and tested
- Configuration-driven architecture

### 🚀 Phase 2 Deliverables (What We're Building)

#### 1. Production Inference Pipeline
**Goal:** Route customers intelligently between Champion and Challenger models

**Components:**
- Inference endpoint wrapper
- Model Router integration (OOD, data sufficiency, A/B testing)
- Prediction pipeline with explainability
- Batch scoring job for production

**Files to Create:**
- `src/decision_agent/inference/batch_scoring.py` - Production batch scoring
- `src/decision_agent/inference/inference_pipeline.py` - Inference orchestrator
- `databricks/notebooks/inference/batch_inference.py` - Databricks batch job
- `conf/inference/production_config.yaml` - Inference configuration

#### 2. Safe Model Deployment
**Goal:** Deploy Challenger safely without disrupting production

**Components:**
- Shadow deployment implementation
- Gradual rollout manager (10% → 50% → 100%)
- Traffic splitting logic
- Rollback mechanism

**Files to Create:**
- `src/decision_agent/deployment/rollout_manager.py` - Gradual rollout control
- `src/decision_agent/deployment/traffic_splitter.py` - A/B traffic routing
- `databricks/workflows/shadow_deployment_workflow.yml` - Shadow mode DAG
- `conf/deployment/rollout_config.yaml` - Rollout configuration

#### 3. Explainability Integration
**Goal:** SHAP explanations for every production prediction (FCRA compliance)

**Components:**
- Explainer integration into inference pipeline
- Reason code generation for adverse actions
- Explanation storage in audit log

**Files to Modify:**
- `src/decision_agent/inference/inference_pipeline.py` - Add SHAP step
- `src/decision_agent/decisions/prediction_logger.py` - Log reason codes

#### 4. Production Monitoring System
**Goal:** Daily monitoring job that detects drift and triggers alerts

**Components:**
- Databricks scheduled job for model_monitor.py
- Slack webhook integration
- PagerDuty integration
- Monitoring dashboard

**Files to Create:**
- `databricks/jobs/daily_monitoring_job.yml` - Scheduled monitoring job
- `src/decision_agent/monitoring/alerting.py` - Alert manager
- `src/decision_agent/monitoring/dashboard_data.py` - Dashboard data prep
- `conf/monitoring/monitoring_config.yaml` - Monitoring configuration

#### 5. Databricks Deployment Automation
**Goal:** Automated deployment to Databricks via CI/CD

**Components:**
- Deployment scripts
- GitHub Actions workflow for Databricks deployment
- Cluster configurations
- Job configurations

**Files to Create:**
- `.github/workflows/deploy-databricks.yml` - Deployment automation
- `deployment/deploy_to_databricks.sh` - Deployment script
- `deployment/cluster_configs/production_cluster.json` - Production cluster
- `databricks/workflows/production_pipeline.yml` - Production workflow

---

## Phase 2 Implementation Plan

### Sprint 1: Production Inference (Days 1-5)

#### Day 1: Batch Scoring Pipeline
**Goal:** Production batch inference job

**Tasks:**
1. Create `inference/batch_scoring.py`:
   - Load Champion and Challenger models from MLflow
   - Score batch of customers
   - Route predictions based on ModelRouter logic
   - Apply business rules (thresholds, overrides)

2. Create `inference/inference_pipeline.py`:
   - Orchestrate: load data → features → routing → prediction → explainability → logging
   - Handle errors gracefully
   - Support both Champion-only and Champion/Challenger modes

3. Create `conf/inference/production_config.yaml`:
   - Model URIs (Champion, Challenger)
   - Routing configuration (OOD thresholds, data sufficiency, A/B %)
   - Batch size, parallelism

**Deliverables:**
- Working batch scoring pipeline
- Configuration-driven inference
- Supports both single-model and A/B modes

#### Day 2: Model Router Integration
**Goal:** Integrate ModelRouter into production inference

**Tasks:**
1. Update `inference_pipeline.py`:
   - Load models (Champion, Challenger) from MLflow
   - Initialize ModelRouter with OODDetector
   - Route each customer → selected_model, routing_reason
   - Log routing decisions

2. Test routing logic:
   - OOD customers → Champion
   - Insufficient data → Champion
   - New customers (< 90 days) → Champion
   - A/B test (hash-based) → Champion/Challenger split

**Deliverables:**
- ModelRouter fully integrated
- Routing logic tested
- All routing decisions logged

#### Day 3: Explainability Integration
**Goal:** SHAP explanations for every prediction

**Tasks:**
1. Update `inference_pipeline.py`:
   - Initialize Explainer with models
   - Generate SHAP values for each prediction
   - Extract top-K reason codes
   - Format for adverse action notices (FCRA)

2. Update `prediction_logger.py`:
   - Add reason_codes column to audit log
   - Log SHAP values (optional, for debugging)

**Deliverables:**
- SHAP integrated into inference
- Reason codes generated for all predictions
- Adverse action notices ready

#### Day 4: Databricks Batch Job
**Goal:** Databricks notebook for production batch scoring

**Tasks:**
1. Create `databricks/notebooks/inference/batch_inference.py`:
   - Databricks-native notebook
   - Reads from production Delta tables
   - Calls inference_pipeline
   - Writes to decisions table
   - Sends summary metrics to Slack

2. Create `databricks/workflows/production_inference.yml`:
   - Daily batch job (scheduled)
   - Task dependencies: load → score → validate → write
   - Retry logic, timeout, alerting

**Deliverables:**
- Databricks batch job working
- Daily scheduling configured
- Error handling and retries

#### Day 5: Integration Testing
**Goal:** End-to-end inference pipeline test

**Tasks:**
1. Test with synthetic data:
   - Generate test customers
   - Run inference pipeline
   - Verify routing decisions
   - Check SHAP values
   - Validate audit log

2. Performance testing:
   - Batch size optimization
   - Parallelism tuning
   - Latency measurement

**Deliverables:**
- E2E inference test passing
- Performance benchmarks documented
- Ready for production deployment

---

### Sprint 2: Safe Deployment (Days 6-10)

#### Day 6: Shadow Deployment
**Goal:** Run Challenger in shadow mode

**Tasks:**
1. Create `deployment/shadow_mode.py`:
   - Wrap ShadowDeployment module
   - Run Champion (production) + Challenger (shadow)
   - Log both predictions
   - Compare offline

2. Create `databricks/workflows/shadow_deployment_workflow.yml`:
   - Parallel tasks: Champion scoring, Challenger shadow scoring
   - Join results, compute comparison metrics
   - Alert if Challenger behaves unexpectedly

**Deliverables:**
- Shadow mode working
- Challenger predictions logged separately
- Offline comparison running

#### Day 7: Gradual Rollout Manager
**Goal:** Controlled Challenger rollout (10% → 50% → 100%)

**Tasks:**
1. Create `deployment/rollout_manager.py`:
   - Current rollout percentage stored in config/state
   - Hash-based traffic splitting (deterministic)
   - Manual or automated rollout progression
   - Rollback mechanism

2. Create `conf/deployment/rollout_config.yaml`:
   - rollout_percentage: 0 (start with 0% = Champion only)
   - rollout_schedule: [10, 25, 50, 75, 100]
   - stability_threshold: minimum correlation
   - rollback_conditions: performance degradation triggers

**Deliverables:**
- Gradual rollout working
- Rollback tested
- Configuration-driven progression

#### Day 8: Traffic Splitter
**Goal:** A/B test traffic splitting logic

**Tasks:**
1. Create `deployment/traffic_splitter.py`:
   - Hash customer_id consistently
   - Split based on rollout_percentage
   - Track which customers see which model
   - Support sticky assignments (same customer always sees same model)

2. Test traffic splitting:
   - Verify deterministic (same customer → same model)
   - Verify distribution (50% split → ~50% to each model)
   - Verify sticky assignments

**Deliverables:**
- Traffic splitter working
- Deterministic and fair splitting
- Sticky assignments supported

#### Day 9: Rollback Mechanism
**Goal:** Automatic rollback if Challenger degrades

**Tasks:**
1. Add rollback logic to `rollout_manager.py`:
   - Monitor Challenger performance
   - Compare to Champion (stability_monitor)
   - If correlation < threshold OR MAE increases > 10% → rollback
   - Rollback sets rollout_percentage = 0 (Champion only)

2. Create alerting:
   - Slack notification on rollback
   - PagerDuty alert for manual review

**Deliverables:**
- Automatic rollback working
- Alerts sent on rollback
- Manual override supported

#### Day 10: Deployment Integration Test
**Goal:** Full deployment workflow test

**Tasks:**
1. Test deployment progression:
   - Start: 0% (Champion only)
   - Shadow mode: Challenger logs predictions
   - Ramp: 10% → 25% → 50%
   - Monitor: Check stability, performance
   - Rollback test: Inject bad Challenger → verify rollback

**Deliverables:**
- Full deployment workflow tested
- Shadow → Gradual → Full production path validated
- Rollback tested

---

### Sprint 3: Production Monitoring (Days 11-15)

#### Day 11: Monitoring Job Setup
**Goal:** Daily Databricks monitoring job

**Tasks:**
1. Create `databricks/jobs/daily_monitoring_job.yml`:
   - Scheduled job (daily at 9 AM)
   - Runs model_monitor.py
   - Outputs to monitoring_metrics table
   - Sends alerts if drift detected

2. Create `conf/monitoring/monitoring_config.yaml`:
   - Feature drift: PSI threshold (0.25)
   - Prediction drift: mean/std shift thresholds
   - Data quality: null rate, freshness
   - Alert webhooks: Slack, PagerDuty

**Deliverables:**
- Daily monitoring job deployed
- Configuration-driven thresholds
- Metrics logged to Delta table

#### Day 12: Alerting Integration
**Goal:** Slack and PagerDuty alerts

**Tasks:**
1. Create `monitoring/alerting.py`:
   - SlackAlerter class (webhook integration)
   - PagerDutyAlerter class (API integration)
   - AlertManager orchestrates multiple channels
   - Alert severity levels: INFO, WARNING, CRITICAL

2. Update `model_monitor.py`:
   - Use AlertManager for all alerts
   - Send INFO alerts for daily summary
   - Send WARNING for moderate drift (PSI 0.1-0.25)
   - Send CRITICAL for severe drift (PSI > 0.25) or data quality failure

**Deliverables:**
- Slack alerts working
- PagerDuty integration working
- Alert severity properly routed

#### Day 13: Monitoring Dashboard
**Goal:** Databricks SQL dashboard for monitoring

**Tasks:**
1. Create `monitoring/dashboard_data.py`:
   - Prepare aggregated metrics for dashboard
   - PSI trends over time
   - Prediction distribution evolution
   - Model performance (if labels available)

2. Create SQL queries for dashboard:
   - Feature drift trends (30-day window)
   - Prediction drift trends
   - Alert history
   - Model version tracking

3. Create Databricks SQL dashboard:
   - Chart: PSI by feature over time
   - Chart: Prediction mean/std over time
   - Chart: Alert frequency
   - Table: Recent alerts

**Deliverables:**
- Dashboard data pipeline
- SQL queries optimized
- Databricks dashboard deployed

#### Day 14: Automated Retraining Trigger
**Goal:** Trigger retraining when drift detected

**Tasks:**
1. Update `model_monitor.py`:
   - If PSI > 0.25 for multiple features → trigger retraining
   - Create retraining ticket/notification
   - Option: Automatically trigger training job (advanced)

2. Create retraining workflow:
   - Notification to ML team
   - Optional: Auto-trigger training job via Databricks API
   - Track retraining history

**Deliverables:**
- Retraining trigger logic
- Notifications working
- Retraining history tracked

#### Day 15: Monitoring Integration Test
**Goal:** Full monitoring workflow test

**Tasks:**
1. Test monitoring pipeline:
   - Inject drifted data → verify drift detection
   - Check alerts sent (Slack, PagerDuty)
   - Verify dashboard updates
   - Test retraining trigger

**Deliverables:**
- Full monitoring workflow tested
- Drift detection validated
- Alerts working end-to-end

---

### Sprint 4: Databricks Deployment Automation (Days 16-20)

#### Day 16: Deployment Scripts
**Goal:** Automated deployment to Databricks

**Tasks:**
1. Create `deployment/deploy_to_databricks.sh`:
   - Upload code to DBFS/Workspace
   - Create/update clusters
   - Deploy notebooks
   - Create/update jobs
   - Configure permissions

2. Create cluster configs:
   - `production_cluster.json` - Production inference cluster
   - `monitoring_cluster.json` - Monitoring job cluster
   - `training_cluster.json` - Model training cluster (from Phase 1)

**Deliverables:**
- Deployment script working
- Cluster configs defined
- One-command deployment

#### Day 17: GitHub Actions CI/CD
**Goal:** Automated deployment on merge to main

**Tasks:**
1. Create `.github/workflows/deploy-databricks.yml`:
   - Trigger: on push to main
   - Steps: lint → test → deploy
   - Use Databricks CLI for deployment
   - Environment: production (requires approval)

2. Configure GitHub secrets:
   - DATABRICKS_HOST
   - DATABRICKS_TOKEN
   - SLACK_WEBHOOK
   - PAGERDUTY_API_KEY

**Deliverables:**
- CI/CD pipeline working
- Automated deployment on merge
- Manual approval for production

#### Day 18: Job Configurations
**Goal:** Production Databricks jobs configured

**Tasks:**
1. Create `databricks/workflows/production_pipeline.yml`:
   - Training job (weekly)
   - Validation job (after training)
   - Inference job (daily)
   - Monitoring job (daily)

2. Configure job dependencies:
   - Training → Validation → Deployment decision
   - Inference (independent, daily)
   - Monitoring (independent, daily)

**Deliverables:**
- All production jobs configured
- Dependencies set up
- Scheduling configured

#### Day 19: Permissions & Security
**Goal:** Secure production deployment

**Tasks:**
1. Configure access control:
   - Production cluster: limited access
   - Delta tables: read/write permissions
   - Secrets: Databricks Secrets scope
   - Jobs: execution permissions

2. Security hardening:
   - Network isolation for clusters
   - Encryption at rest (Delta Lake)
   - Encryption in transit (TLS)
   - Audit logging enabled

**Deliverables:**
- Access control configured
- Security hardened
- Audit trail enabled

#### Day 20: End-to-End Deployment Test
**Goal:** Full production deployment test

**Tasks:**
1. Deploy to Databricks:
   - Run deployment script
   - Verify all jobs created
   - Check clusters configured
   - Validate permissions

2. Run production workflow:
   - Trigger training job
   - Run inference job
   - Run monitoring job
   - Check all outputs

**Deliverables:**
- Full production deployment tested
- All jobs running
- Ready for production use

---

## Phase 2 Success Criteria

### Must Have ✅
- [ ] Production inference pipeline deployed and running daily
- [ ] Model Router integrated (smart routing based on OOD, data sufficiency, A/B)
- [ ] SHAP explanations generated for all predictions
- [ ] Shadow deployment working (Challenger runs in shadow mode)
- [ ] Gradual rollout working (10% → 50% → 100% progression)
- [ ] Daily monitoring job running (drift detection, alerting)
- [ ] Slack alerts working (drift, data quality, performance)
- [ ] Automated deployment via GitHub Actions

### Nice to Have 🎯
- [ ] PagerDuty integration
- [ ] Databricks SQL dashboard deployed
- [ ] Automated retraining trigger
- [ ] Rollback mechanism tested
- [ ] Performance benchmarks documented

---

## Phase 2 Deliverables Summary

### New Files Created (~3,000 lines)

**Inference (5 files):**
- `src/decision_agent/inference/batch_scoring.py`
- `src/decision_agent/inference/inference_pipeline.py`
- `databricks/notebooks/inference/batch_inference.py`
- `databricks/workflows/production_inference.yml`
- `conf/inference/production_config.yaml`

**Deployment (5 files):**
- `src/decision_agent/deployment/rollout_manager.py`
- `src/decision_agent/deployment/traffic_splitter.py`
- `databricks/workflows/shadow_deployment_workflow.yml`
- `conf/deployment/rollout_config.yaml`
- `deployment/deploy_to_databricks.sh`

**Monitoring (5 files):**
- `src/decision_agent/monitoring/alerting.py`
- `src/decision_agent/monitoring/dashboard_data.py`
- `databricks/jobs/daily_monitoring_job.yml`
- `conf/monitoring/monitoring_config.yaml`
- `deployment/cluster_configs/production_cluster.json`

**CI/CD (2 files):**
- `.github/workflows/deploy-databricks.yml`
- `databricks/workflows/production_pipeline.yml`

**Total:** 17 new files, ~3,000 lines of production code

### Files Modified
- `src/decision_agent/decisions/prediction_logger.py` (add reason codes)
- `databricks/workflows/decision_agent_workflow.yml` (update for production)

---

## Phase 2 Architecture

```
Production Flow:
--------------

Daily Batch Inference:
  Customer Data (Delta)
    ↓
  Feature Engineering
    ↓
  Model Router Decision:
    - OOD Check → Champion/Challenger
    - Data Sufficiency → Champion if insufficient
    - A/B Hash → Champion/Challenger split
    ↓
  Prediction:
    - Champion Model (MLflow)
    - Challenger Model (MLflow, if A/B)
    ↓
  SHAP Explanation:
    - Top-K reason codes
    - Adverse action notices
    ↓
  Prediction Logger:
    - Audit trail to Delta
    - Reason codes stored
    ↓
  Decisions Table (Delta)

Daily Monitoring:
  Current Data (last 24h)
    ↓
  Baseline Data (training)
    ↓
  Feature Drift (PSI):
    - PSI > 0.25 → CRITICAL alert
    ↓
  Prediction Drift:
    - Mean/std shift > threshold → WARNING
    ↓
  Data Quality:
    - Null rates, freshness → WARNING/CRITICAL
    ↓
  Alert Manager:
    - Slack (all alerts)
    - PagerDuty (CRITICAL only)
    ↓
  Monitoring Metrics (Delta)
    ↓
  Dashboard Update

Shadow Deployment:
  Production Traffic
    ↓
  Parallel Scoring:
    - Champion (production)
    - Challenger (shadow)
    ↓
  Comparison Metrics:
    - Correlation, MAE difference
    ↓
  Decision:
    - If stable → increase rollout %
    - If unstable → rollback

Gradual Rollout:
  0% (Champion only)
    ↓
  Shadow mode (monitor Challenger)
    ↓
  10% → 25% → 50% (gradual increase)
    ↓ (monitor stability at each step)
  100% (Challenger becomes new Champion)
```

---

## Phase 2 Timeline

**Total Duration:** 20 days (4 weeks)

- **Sprint 1 (Days 1-5):** Production Inference
- **Sprint 2 (Days 6-10):** Safe Deployment
- **Sprint 3 (Days 11-15):** Production Monitoring
- **Sprint 4 (Days 16-20):** Databricks Deployment Automation

**Critical Path:**
Inference Pipeline → Model Router → Explainability → Databricks Job → Monitoring → Alerting → Deployment Automation

---

## Phase 2 → Phase 3 Handoff

**Prerequisites for Phase 3:**
- [ ] Phase 2 production system deployed
- [ ] Daily inference running successfully
- [ ] Monitoring detecting drift
- [ ] Challenger successfully promoted via gradual rollout
- [ ] All P0 gates functioning in production

**Phase 3 Preview (Advanced Features):**
1. Real-time Streaming Features
2. AutoML Integration
3. Deep Learning Models
4. Online Feature Computation
5. Real-time Inference API

---

**Phase 2 Status:** 🚀 READY TO START
**Next:** Implement Sprint 1 (Production Inference)
