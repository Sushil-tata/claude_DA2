#!/bin/bash
#
# Deploy Decision Agent to Databricks
#
# Usage:
#   ./deployment/deploy_to_databricks.sh [environment]
#
# Arguments:
#   environment: dev, staging, or production (default: dev)
#
# Prerequisites:
#   - Databricks CLI installed and configured
#   - DATABRICKS_HOST and DATABRICKS_TOKEN environment variables set
#
# What this script does:
#   1. Upload Python code to DBFS/Workspace
#   2. Upload configuration files
#   3. Upload notebooks
#   4. Create/update clusters
#   5. Create/update jobs
#   6. Validate deployment

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Environment (default: dev)
ENVIRONMENT=${1:-dev}

echo "=========================================="
echo "Deploying Decision Agent to Databricks"
echo "Environment: $ENVIRONMENT"
echo "=========================================="

# Validate environment
if [[ ! "$ENVIRONMENT" =~ ^(dev|staging|production)$ ]]; then
    echo -e "${RED}Error: Invalid environment '$ENVIRONMENT'. Must be dev, staging, or production${NC}"
    exit 1
fi

# Check prerequisites
echo ""
echo "Checking prerequisites..."

if ! command -v databricks &> /dev/null; then
    echo -e "${RED}Error: Databricks CLI not found. Install with: pip install databricks-cli${NC}"
    exit 1
fi

if [ -z "$DATABRICKS_HOST" ]; then
    echo -e "${RED}Error: DATABRICKS_HOST environment variable not set${NC}"
    exit 1
fi

if [ -z "$DATABRICKS_TOKEN" ]; then
    echo -e "${RED}Error: DATABRICKS_TOKEN environment variable not set${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Prerequisites OK${NC}"

# Configuration based on environment
WORKSPACE_PATH="/Workspace/decision_agent/$ENVIRONMENT"
DBFS_PATH="/dbfs/decision_agent/$ENVIRONMENT"

echo ""
echo "Deployment paths:"
echo "  Workspace: $WORKSPACE_PATH"
echo "  DBFS: $DBFS_PATH"

# Step 1: Upload Python code
echo ""
echo "Step 1: Uploading Python code..."

databricks workspace mkdirs "$WORKSPACE_PATH/src"
databricks workspace import_dir src/decision_agent "$WORKSPACE_PATH/src/decision_agent" --overwrite
echo -e "${GREEN}✓ Python code uploaded${NC}"

# Step 2: Upload configuration files
echo ""
echo "Step 2: Uploading configuration files..."

databricks workspace mkdirs "$WORKSPACE_PATH/conf"
databricks workspace import_dir conf "$WORKSPACE_PATH/conf" --overwrite
echo -e "${GREEN}✓ Configuration files uploaded${NC}"

# Step 3: Upload notebooks
echo ""
echo "Step 3: Uploading notebooks..."

databricks workspace mkdirs "$WORKSPACE_PATH/notebooks"
databricks workspace import_dir databricks/notebooks "$WORKSPACE_PATH/notebooks" --overwrite
echo -e "${GREEN}✓ Notebooks uploaded${NC}"

# Step 4: Create/update clusters
echo ""
echo "Step 4: Creating/updating clusters..."

# Training cluster
TRAINING_CLUSTER_NAME="decision_agent_training_$ENVIRONMENT"
if databricks clusters get --cluster-name "$TRAINING_CLUSTER_NAME" &> /dev/null; then
    echo "  Training cluster exists: $TRAINING_CLUSTER_NAME"
else
    echo "  Creating training cluster: $TRAINING_CLUSTER_NAME"
    databricks clusters create --json-file deployment/cluster_configs/training_cluster.json
    echo -e "${GREEN}✓ Training cluster created${NC}"
fi

# Inference cluster
INFERENCE_CLUSTER_NAME="decision_agent_inference_$ENVIRONMENT"
if databricks clusters get --cluster-name "$INFERENCE_CLUSTER_NAME" &> /dev/null; then
    echo "  Inference cluster exists: $INFERENCE_CLUSTER_NAME"
else
    echo "  Creating inference cluster: $INFERENCE_CLUSTER_NAME"
    databricks clusters create --json-file deployment/cluster_configs/production_cluster.json
    echo -e "${GREEN}✓ Inference cluster created${NC}"
fi

# Step 5: Create/update jobs
echo ""
echo "Step 5: Creating/updating jobs..."

# Production inference job
INFERENCE_JOB_NAME="production_inference_$ENVIRONMENT"
echo "  Creating/updating job: $INFERENCE_JOB_NAME"

# Check if job exists
JOB_ID=$(databricks jobs list --output JSON | jq -r ".jobs[] | select(.settings.name == \"$INFERENCE_JOB_NAME\") | .job_id")

if [ -z "$JOB_ID" ]; then
    echo "    Job does not exist. Creating..."
    # Update workflow YAML with environment-specific paths
    sed "s|/Workspace/|$WORKSPACE_PATH/|g" databricks/workflows/production_inference.yml > /tmp/production_inference_$ENVIRONMENT.yml
    databricks jobs create --json-file /tmp/production_inference_$ENVIRONMENT.yml
    echo -e "${GREEN}✓ Inference job created${NC}"
else
    echo "    Job exists (ID: $JOB_ID). Updating..."
    sed "s|/Workspace/|$WORKSPACE_PATH/|g" databricks/workflows/production_inference.yml > /tmp/production_inference_$ENVIRONMENT.yml
    databricks jobs reset --job-id "$JOB_ID" --json-file /tmp/production_inference_$ENVIRONMENT.yml
    echo -e "${GREEN}✓ Inference job updated${NC}"
fi

# Daily monitoring job
MONITORING_JOB_NAME="daily_monitoring_$ENVIRONMENT"
echo "  Creating/updating job: $MONITORING_JOB_NAME"

JOB_ID=$(databricks jobs list --output JSON | jq -r ".jobs[] | select(.settings.name == \"$MONITORING_JOB_NAME\") | .job_id")

if [ -z "$JOB_ID" ]; then
    echo "    Job does not exist. Creating..."
    sed "s|/Workspace/|$WORKSPACE_PATH/|g" databricks/jobs/daily_monitoring_job.yml > /tmp/daily_monitoring_$ENVIRONMENT.yml
    databricks jobs create --json-file /tmp/daily_monitoring_$ENVIRONMENT.yml
    echo -e "${GREEN}✓ Monitoring job created${NC}"
else
    echo "    Job exists (ID: $JOB_ID). Updating..."
    sed "s|/Workspace/|$WORKSPACE_PATH/|g" databricks/jobs/daily_monitoring_job.yml > /tmp/daily_monitoring_$ENVIRONMENT.yml
    databricks jobs reset --job-id "$JOB_ID" --json-file /tmp/daily_monitoring_$ENVIRONMENT.yml
    echo -e "${GREEN}✓ Monitoring job updated${NC}"
fi

# Step 6: Validation
echo ""
echo "Step 6: Validating deployment..."

# Check workspace files
if databricks workspace ls "$WORKSPACE_PATH/src/decision_agent" &> /dev/null; then
    echo -e "${GREEN}✓ Workspace files validated${NC}"
else
    echo -e "${RED}✗ Workspace validation failed${NC}"
    exit 1
fi

# Check jobs
if databricks jobs list | grep -q "$INFERENCE_JOB_NAME"; then
    echo -e "${GREEN}✓ Inference job validated${NC}"
else
    echo -e "${RED}✗ Inference job validation failed${NC}"
    exit 1
fi

if databricks jobs list | grep -q "$MONITORING_JOB_NAME"; then
    echo -e "${GREEN}✓ Monitoring job validated${NC}"
else
    echo -e "${RED}✗ Monitoring job validation failed${NC}"
    exit 1
fi

# Deployment complete
echo ""
echo "=========================================="
echo -e "${GREEN}✓ Deployment Complete!${NC}"
echo "=========================================="
echo ""
echo "Deployed to environment: $ENVIRONMENT"
echo "Workspace path: $WORKSPACE_PATH"
echo ""
echo "Next steps:"
echo "  1. Verify jobs in Databricks UI"
echo "  2. Trigger test run: databricks jobs run-now --job-name $INFERENCE_JOB_NAME"
echo "  3. Monitor job execution in Databricks"
echo ""
