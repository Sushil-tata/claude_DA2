# Deployment Checklist

## Pre-Deployment Verification

### ✅ Local Testing
- [x] Dry run successful: `python3 jobs/run_usecase.py --config conf/use_cases/income_estimation.yaml --dry-run`
- [x] All imports working
- [x] Configuration validated
- [ ] Full local run with PySpark (optional - requires PySpark installation)

### ✅ Code Quality
- [x] All modules have working implementations (no NotImplementedError)
- [x] Docstrings present
- [x] Import checks pass
- [ ] Unit tests pass (requires pytest installation)
- [ ] Linting passes (requires black, isort installation)

### ✅ Documentation
- [x] README.md complete with quickstart
- [x] Configuration examples provided
- [x] Architecture documented
- [x] Troubleshooting guide included

## Databricks Deployment Steps

### Step 1: Prerequisites
- [ ] Databricks workspace accessible
- [ ] DATABRICKS_HOST environment variable set
- [ ] DATABRICKS_TOKEN environment variable set
- [ ] Databricks CLI installed: `pip install databricks-cli`

### Step 2: Upload Code
```bash
# Upload Python modules
databricks workspace import_dir src/decision_agent /Workspace/decision_agent --overwrite

# Upload configuration
databricks workspace import_dir conf /Workspace/conf --overwrite

# Upload entry point
databricks workspace import jobs/run_usecase.py /Workspace/jobs/run_usecase --overwrite
```

### Step 3: Create Cluster
```bash
# Create feature engineering cluster
databricks clusters create --json-file deployment/cluster_configs/feature_cluster.json

# Note cluster ID for workflow configuration
```

### Step 4: Create Workflow
```bash
# Create Databricks Workflow
databricks jobs create --json-file databricks/workflows/decision_agent_workflow.yml

# Note job ID
```

### Step 5: Test Run
```bash
# Trigger manual run
databricks jobs run-now --job-id <job-id>

# Monitor execution
databricks runs list --job-id <job-id>

# Get run details
databricks runs get --run-id <run-id>
```

### Step 6: Verify Output
```sql
-- In Databricks SQL editor
SELECT * FROM decision_agent.income_decisions LIMIT 10;

-- Check decision count
SELECT COUNT(*) FROM decision_agent.income_decisions;

-- View MLflow experiments
-- Navigate to Machine Learning -> Experiments -> /decision_agent/experiments
```

## GitHub Actions Setup

### Step 1: Configure Secrets
In GitHub repository settings, add:
- [ ] `DATABRICKS_HOST`
- [ ] `DATABRICKS_TOKEN`

### Step 2: Verify CI Workflow
- [ ] Push to main branch triggers CI
- [ ] Unit tests pass
- [ ] Linting passes
- [ ] Import validation succeeds

### Step 3: Branch Protection (Optional)
- [ ] Require PR reviews
- [ ] Require CI checks to pass
- [ ] Prevent force pushes to main

## Production Checklist

### Monitoring
- [ ] MLflow experiment tracking enabled
- [ ] Databricks Workflow email notifications configured
- [ ] Delta Lake table monitoring setup
- [ ] Error alerting configured

### Data Quality
- [ ] Synthetic data generation verified
- [ ] Real data sources identified (if applicable)
- [ ] Data validation rules defined
- [ ] Feature quality checks in place

### Model Governance
- [ ] Model registry configured
- [ ] Model promotion workflow defined
- [ ] Model versioning strategy documented
- [ ] Rollback procedure documented

### Security
- [ ] Service principal authentication configured
- [ ] Databricks secrets for credentials
- [ ] Table-level permissions configured
- [ ] Cluster access controls set

### Performance
- [ ] Cluster autoscaling configured
- [ ] Delta Lake optimization enabled
- [ ] Feature computation performance tested
- [ ] Training time benchmarked

## Post-Deployment

### Validation
- [ ] End-to-end pipeline runs successfully
- [ ] Decisions written to Delta Lake
- [ ] MLflow experiments tracked
- [ ] No errors in workflow logs

### Documentation
- [ ] Update README with production details
- [ ] Document Databricks workspace location
- [ ] Document table schemas
- [ ] Document monitoring procedures

### Handoff
- [ ] Team trained on platform
- [ ] Runbooks created
- [ ] On-call procedures documented
- [ ] Knowledge transfer completed

## Rollback Plan

If deployment fails:

1. **Stop the workflow**:
   ```bash
   databricks jobs reset --job-id <job-id> --pause-status PAUSED
   ```

2. **Revert code changes** (if needed):
   ```bash
   git revert <commit-hash>
   git push origin main
   ```

3. **Restore previous Delta table version**:
   ```sql
   RESTORE TABLE decision_agent.income_decisions TO VERSION AS OF <version>;
   ```

4. **Investigate logs**:
   - Check Databricks Workflow UI
   - Review MLflow experiment logs
   - Check Delta Lake transaction log

## Support Contacts

- **Platform Owner**: Data Science Team
- **Databricks Admin**: [Add contact]
- **On-Call**: [Add rotation]

---

## Quick Reference Commands

### Local Development
```bash
# Dry run
python3 jobs/run_usecase.py --config conf/use_cases/income_estimation.yaml --dry-run

# List use cases
python3 jobs/run_usecase.py --list

# Run locally
python3 jobs/run_usecase.py --config conf/use_cases/income_estimation.yaml --local
```

### Databricks CLI
```bash
# List jobs
databricks jobs list

# Run job
databricks jobs run-now --job-id <job-id>

# Get run status
databricks runs get --run-id <run-id>

# List clusters
databricks clusters list
```

### SQL Queries
```sql
-- View recent decisions
SELECT * FROM decision_agent.income_decisions
WHERE as_of_dt = CURRENT_DATE()
LIMIT 100;

-- Decision count by date
SELECT as_of_dt, COUNT(*) as num_decisions
FROM decision_agent.income_decisions
GROUP BY as_of_dt
ORDER BY as_of_dt DESC;

-- Model version distribution
SELECT model_version, COUNT(*) as count
FROM decision_agent.income_decisions
GROUP BY model_version;
```

---

**Last Updated**: 2024-02-09
**Platform Version**: v0.1.0
