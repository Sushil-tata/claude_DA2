#!/bin/bash
################################################################################
# NBA Infrastructure Setup - Week 1
#
# Sets up:
# 1. Delta Lake databases and tables
# 2. MLflow experiment tracking
# 3. Git version control
# 4. Databricks secrets
# 5. Monitoring dashboards
################################################################################

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}NBA Infrastructure Setup${NC}"
echo -e "${BLUE}========================================${NC}"

# ========================================
# 1. Create Delta Lake Databases
# ========================================

echo -e "\n${BLUE}[1/6] Creating Delta Lake databases...${NC}"

databricks sql query --query "
CREATE DATABASE IF NOT EXISTS debt_collection
COMMENT 'NBA system for debt collection optimization'
LOCATION 'dbfs:/debt_collection/';

CREATE DATABASE IF NOT EXISTS debt_collection_dev
COMMENT 'Development environment for NBA'
LOCATION 'dbfs:/debt_collection_dev/';

CREATE DATABASE IF NOT EXISTS debt_collection_staging
COMMENT 'Staging environment for NBA'
LOCATION 'dbfs:/debt_collection_staging/';
"

echo -e "${GREEN}✓ Databases created${NC}"

# ========================================
# 2. Create Core Tables
# ========================================

echo -e "\n${BLUE}[2/6] Creating core tables...${NC}"

# Decision table
databricks sql query --file schemas/decision_table_schema.sql

# Model predictions log
databricks sql query --query "
CREATE TABLE IF NOT EXISTS debt_collection.model_predictions_log (
    prediction_id STRING,
    account_id STRING,
    business_date DATE,
    model_version STRING,
    action STRING,

    -- Predictions
    pay_any_score DECIMAL(5,4),
    expected_amount DECIMAL(10,2),
    uplift_score DECIMAL(5,4),

    -- Top-3 Actions
    action_rank_1 STRING,
    action_rank_1_score DECIMAL(5,4),
    action_rank_2 STRING,
    action_rank_2_score DECIMAL(5,4),
    action_rank_3 STRING,
    action_rank_3_score DECIMAL(5,4),

    -- Metadata
    prediction_timestamp TIMESTAMP,
    PRIMARY KEY (prediction_id)
)
USING DELTA
PARTITIONED BY (business_date);
"

# Decision execution log
databricks sql query --query "
CREATE TABLE IF NOT EXISTS debt_collection.decision_execution_log (
    execution_id STRING,
    account_id STRING,
    business_date DATE,

    -- Recommended
    recommended_action STRING,
    recommended_score DECIMAL(5,4),

    -- Constraints
    feasible_actions ARRAY<STRING>,
    blocked_actions ARRAY<STRING>,
    block_reasons ARRAY<STRING>,

    -- Executed
    executed_action STRING,
    execution_timestamp TIMESTAMP,
    execution_status STRING,  -- SUCCESS, FAILED, CAPACITY_LIMITED

    -- Outcomes (joined later)
    outcome_pay_any_7d BOOLEAN,
    outcome_pay_amount_7d DECIMAL(10,2),

    PRIMARY KEY (execution_id)
)
USING DELTA
PARTITIONED BY (business_date);
"

# Why cards (explainability)
databricks sql query --query "
CREATE TABLE IF NOT EXISTS debt_collection.why_cards (
    why_card_id STRING,
    account_id STRING,
    business_date DATE,

    -- State Summary
    bucket INT,
    delay INT,
    balance DECIMAL(10,2),
    persona_segment STRING,
    fatigue_score DECIMAL(5,4),

    -- Top-3 Actions
    top_actions ARRAY<STRUCT<
        action: STRING,
        score: DECIMAL(5,4),
        expected_amount: DECIMAL(10,2),
        reason: STRING
    >>,

    -- Top SHAP Drivers
    top_drivers ARRAY<STRUCT<
        feature: STRING,
        value: STRING,
        shap_value: DECIMAL(8,6)
    >>,

    -- Constraints Applied
    constraints_applied ARRAY<STRING>,

    created_timestamp TIMESTAMP,
    PRIMARY KEY (why_card_id)
)
USING DELTA;
"

echo -e "${GREEN}✓ Core tables created${NC}"

# ========================================
# 3. Setup MLflow Experiments
# ========================================

echo -e "\n${BLUE}[3/6] Setting up MLflow experiments...${NC}"

# Create experiments
databricks mlflow experiments create \
    --experiment-name "/Users/nba_team/pay_any_model" \
    --artifact-location "dbfs:/mlflow/pay_any_model"

databricks mlflow experiments create \
    --experiment-name "/Users/nba_team/amount_model" \
    --artifact-location "dbfs:/mlflow/amount_model"

databricks mlflow experiments create \
    --experiment-name "/Users/nba_team/uplift_models" \
    --artifact-location "dbfs:/mlflow/uplift_models"

databricks mlflow experiments create \
    --experiment-name "/Users/nba_team/model_comparison" \
    --artifact-location "dbfs:/mlflow/model_comparison"

# Create model registry entries
python -c "
import mlflow

mlflow.set_tracking_uri('databricks')

# Register model names
mlflow.create_registered_model(
    name='nba_pay_any_model',
    description='Binary classifier for payment probability (action-conditional)'
)

mlflow.create_registered_model(
    name='nba_amount_model',
    description='Tweedie regression for payment amount (action-conditional)'
)

mlflow.create_registered_model(
    name='nba_ensemble',
    description='Complete NBA ensemble (pay_any + amount + constraints)'
)
"

echo -e "${GREEN}✓ MLflow experiments created${NC}"

# ========================================
# 4. Setup Secrets
# ========================================

echo -e "\n${BLUE}[4/6] Setting up secrets...${NC}"

# Create secret scope
databricks secrets create-scope --scope nba_secrets

# Add placeholder secrets (replace with actual values)
echo "placeholder" | databricks secrets put --scope nba_secrets --key mlflow_tracking_uri
echo "placeholder" | databricks secrets put --scope nba_secrets --key model_registry_uri

echo -e "${GREEN}✓ Secret scope created${NC}"
echo -e "${YELLOW}⚠ Remember to update secrets with actual values${NC}"

# ========================================
# 5. Create Git Integration
# ========================================

echo -e "\n${BLUE}[5/6] Setting up Git integration...${NC}"

# Create .gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/

# MLflow
mlruns/
mlflow_artifacts/

# Databricks
.databricks/

# IDE
.vscode/
.idea/
*.swp

# Data
*.csv
*.parquet
*.delta

# Secrets
*.env
secrets/
EOF

# Initialize git if not already
if [ ! -d .git ]; then
    git init
    git add .gitignore
    git commit -m "Initial commit: NBA infrastructure setup"
fi

echo -e "${GREEN}✓ Git initialized${NC}"

# ========================================
# 6. Create Monitoring Dashboards
# ========================================

echo -e "\n${BLUE}[6/6] Creating monitoring dashboards...${NC}"

# Create dashboard directory
mkdir -p deployment/monitoring/dashboards/week1

# Create basic monitoring SQL
cat > deployment/monitoring/dashboards/week1/decision_table_health.sql << 'EOF'
-- Decision Table Health Dashboard
-- Run this daily to monitor data quality

-- 1. Row Count by Date
SELECT
    business_date,
    COUNT(*) as account_count,
    COUNT(DISTINCT account_id) as unique_accounts
FROM debt_collection.decision_table
WHERE business_date >= CURRENT_DATE() - INTERVAL 7 DAYS
GROUP BY business_date
ORDER BY business_date DESC;

-- 2. Delay Distribution
SELECT
    CASE
        WHEN delay <= 0 THEN '0_current'
        WHEN delay <= 30 THEN '1-30_dpd'
        WHEN delay <= 60 THEN '31-60_dpd'
        WHEN delay <= 90 THEN '61-90_dpd'
        ELSE '90+_dpd'
    END as delay_bucket,
    COUNT(*) as cnt,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) as pct,
    ROUND(AVG(balance), 2) as avg_balance
FROM debt_collection.decision_table
WHERE business_date = CURRENT_DATE()
GROUP BY 1
ORDER BY 1;

-- 3. Persona Distribution
SELECT
    persona_segment,
    COUNT(*) as cnt,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) as pct,
    ROUND(AVG(balance), 2) as avg_balance,
    ROUND(AVG(delay), 1) as avg_delay
FROM debt_collection.decision_table
WHERE business_date = CURRENT_DATE()
  AND persona_segment IS NOT NULL
GROUP BY persona_segment
ORDER BY cnt DESC;

-- 4. Operability Flags
SELECT
    SUM(CASE WHEN line_optin THEN 1 ELSE 0 END) as line_optin_count,
    SUM(CASE WHEN sms_optin THEN 1 ELSE 0 END) as sms_optin_count,
    SUM(CASE WHEN mobile_present THEN 1 ELSE 0 END) as mobile_present_count,
    SUM(CASE WHEN dnc THEN 1 ELSE 0 END) as dnc_count,
    SUM(CASE WHEN cease_and_desist THEN 1 ELSE 0 END) as cease_and_desist_count,
    SUM(CASE WHEN active_dispute THEN 1 ELSE 0 END) as active_dispute_count,
    COUNT(*) as total_accounts
FROM debt_collection.decision_table
WHERE business_date = CURRENT_DATE();

-- 5. Data Quality Issues
SELECT
    'Missing mobile' as issue,
    COUNT(*) as cnt
FROM debt_collection.decision_table
WHERE business_date = CURRENT_DATE()
  AND mobile_present = FALSE
  AND bucket >= 1

UNION ALL

SELECT
    'Negative delay' as issue,
    COUNT(*) as cnt
FROM debt_collection.decision_table
WHERE business_date = CURRENT_DATE()
  AND delay < 0

UNION ALL

SELECT
    'NULL persona' as issue,
    COUNT(*) as cnt
FROM debt_collection.decision_table
WHERE business_date = CURRENT_DATE()
  AND persona_segment IS NULL;
EOF

echo -e "${GREEN}✓ Monitoring dashboards created${NC}"

# ========================================
# Summary
# ========================================

echo -e "\n${BLUE}========================================${NC}"
echo -e "${GREEN}Infrastructure Setup Complete!${NC}"
echo -e "${BLUE}========================================${NC}"

echo -e "\nCreated:"
echo -e "  ✓ 3 Delta Lake databases"
echo -e "  ✓ 4 core tables (decision_table, predictions_log, execution_log, why_cards)"
echo -e "  ✓ 4 MLflow experiments"
echo -e "  ✓ 3 model registry entries"
echo -e "  ✓ Secret scope"
echo -e "  ✓ Git integration"
echo -e "  ✓ Monitoring dashboards"

echo -e "\nNext Steps:"
echo -e "  1. Update secrets with actual values"
echo -e "  2. Run decision table builder (Week 1)"
echo -e "  3. Validate data quality with monitoring queries"
echo -e "  4. Begin persona segmentation (Week 2)"

echo -e "\n${YELLOW}Important:${NC}"
echo -e "  - Review action catalog: conf/actions/action_catalog.yaml"
echo -e "  - Update business hours for your market"
echo -e "  - Confirm delay definition matches business logic"
