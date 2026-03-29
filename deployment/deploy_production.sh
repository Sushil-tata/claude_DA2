#!/bin/bash
################################################################################
# Production Deployment Script for Decision Agent Platform
#
# This script deploys the complete Decision Agent platform to production:
# - Databricks workspace setup
# - Cluster creation
# - Job deployment
# - Model deployment
# - Monitoring setup
#
# Usage:
#   ./deploy_production.sh --env production --validate
#
# Prerequisites:
#   - Databricks CLI configured
#   - AWS/Azure CLI configured (for cloud resources)
#   - Terraform installed (for infrastructure)
#   - kubectl configured (for K8s deployments)
################################################################################

set -e  # Exit on error
set -u  # Exit on undefined variable

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
ENVIRONMENT="production"
VALIDATE_ONLY=false
SKIP_INFRASTRUCTURE=false
SKIP_DATABRICKS=false
SKIP_MODELS=false
DRY_RUN=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --env)
      ENVIRONMENT="$2"
      shift 2
      ;;
    --validate)
      VALIDATE_ONLY=true
      shift
      ;;
    --skip-infrastructure)
      SKIP_INFRASTRUCTURE=true
      shift
      ;;
    --skip-databricks)
      SKIP_DATABRICKS=true
      shift
      ;;
    --skip-models)
      SKIP_MODELS=true
      shift
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Decision Agent Production Deployment${NC}"
echo -e "${BLUE}========================================${NC}"
echo "Environment: ${ENVIRONMENT}"
echo "Validate Only: ${VALIDATE_ONLY}"
echo "Dry Run: ${DRY_RUN}"
echo ""

# Function to print colored messages
log_info() {
  echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
  echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
  echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
  echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check prerequisites
check_prerequisites() {
  log_info "Checking prerequisites..."

  # Check Databricks CLI
  if ! command -v databricks &> /dev/null; then
    log_error "Databricks CLI not found. Install: pip install databricks-cli"
    exit 1
  fi

  # Check if Databricks is configured
  if ! databricks workspace ls / &> /dev/null; then
    log_error "Databricks CLI not configured. Run: databricks configure --token"
    exit 1
  fi

  # Check Terraform (if not skipping infrastructure)
  if [[ "${SKIP_INFRASTRUCTURE}" == false ]]; then
    if ! command -v terraform &> /dev/null; then
      log_warning "Terraform not found. Skipping infrastructure deployment."
      SKIP_INFRASTRUCTURE=true
    fi
  fi

  # Check kubectl (for K8s deployments)
  if ! command -v kubectl &> /dev/null; then
    log_warning "kubectl not found. Skipping Kubernetes deployments."
  fi

  log_success "Prerequisites check complete"
}

# Function to deploy infrastructure
deploy_infrastructure() {
  if [[ "${SKIP_INFRASTRUCTURE}" == true ]]; then
    log_info "Skipping infrastructure deployment"
    return
  fi

  log_info "Deploying infrastructure with Terraform..."

  cd deployment/terraform

  # Initialize Terraform
  terraform init

  # Create workspace
  terraform workspace select ${ENVIRONMENT} || terraform workspace new ${ENVIRONMENT}

  # Plan
  log_info "Running terraform plan..."
  terraform plan -var="environment=${ENVIRONMENT}" -out=tfplan

  if [[ "${DRY_RUN}" == true ]]; then
    log_warning "Dry run mode - skipping terraform apply"
    cd ../..
    return
  fi

  # Apply
  log_info "Applying terraform plan..."
  terraform apply tfplan

  cd ../..
  log_success "Infrastructure deployed"
}

# Function to upload code to Databricks
upload_code() {
  log_info "Uploading code to Databricks workspace..."

  # Upload Python modules
  log_info "Uploading Python modules..."
  databricks workspace import_dir src/decision_agent \
    /Workspace/decision_agent/${ENVIRONMENT}/src \
    --overwrite

  # Upload notebooks
  log_info "Uploading notebooks..."
  databricks workspace import_dir databricks/notebooks \
    /Workspace/decision_agent/${ENVIRONMENT}/notebooks \
    --overwrite

  # Upload configs
  log_info "Uploading configurations..."
  databricks workspace import_dir conf \
    /Workspace/decision_agent/${ENVIRONMENT}/conf \
    --overwrite

  log_success "Code uploaded to Databricks"
}

# Function to create clusters
create_clusters() {
  log_info "Creating Databricks clusters..."

  # Feature engineering cluster
  log_info "Creating feature engineering cluster..."
  if databricks clusters get --cluster-name "decision-agent-features-${ENVIRONMENT}" &> /dev/null; then
    log_warning "Feature cluster already exists, skipping creation"
  else
    databricks clusters create --json-file deployment/cluster_configs/feature_cluster.json
    log_success "Feature engineering cluster created"
  fi

  # Training cluster
  log_info "Creating training cluster..."
  if databricks clusters get --cluster-name "decision-agent-training-${ENVIRONMENT}" &> /dev/null; then
    log_warning "Training cluster already exists, skipping creation"
  else
    databricks clusters create --json-file deployment/cluster_configs/training_cluster.json
    log_success "Training cluster created"
  fi

  # Production inference cluster
  log_info "Creating production cluster..."
  if databricks clusters get --cluster-name "decision-agent-production-${ENVIRONMENT}" &> /dev/null; then
    log_warning "Production cluster already exists, skipping creation"
  else
    databricks clusters create --json-file deployment/cluster_configs/production_cluster.json
    log_success "Production cluster created"
  fi

  log_success "All clusters created"
}

# Function to deploy jobs
deploy_jobs() {
  log_info "Deploying Databricks jobs..."

  # Feature engineering job
  log_info "Deploying feature engineering job..."
  databricks jobs create --json '{
    "name": "decision-agent-features-'${ENVIRONMENT}'",
    "tasks": [
      {
        "task_key": "compute_features",
        "existing_cluster_id": "'$(get_cluster_id decision-agent-features-${ENVIRONMENT})'",
        "python_file": "dbfs:/Workspace/decision_agent/'${ENVIRONMENT}'/notebooks/features/feature_pipeline.py"
      }
    ],
    "schedule": {
      "quartz_cron_expression": "0 0 2 * * ?",
      "timezone_id": "UTC"
    }
  }' || log_warning "Feature job may already exist"

  # Training job
  log_info "Deploying training job..."
  databricks jobs create --json '{
    "name": "decision-agent-training-'${ENVIRONMENT}'",
    "tasks": [
      {
        "task_key": "train_model",
        "existing_cluster_id": "'$(get_cluster_id decision-agent-training-${ENVIRONMENT})'",
        "python_file": "dbfs:/Workspace/decision_agent/'${ENVIRONMENT}'/notebooks/training/train_model.py"
      }
    ],
    "schedule": {
      "quartz_cron_expression": "0 0 3 * * ?",
      "timezone_id": "UTC"
    }
  }' || log_warning "Training job may already exist"

  # Batch inference job
  log_info "Deploying batch inference job..."
  databricks jobs create --json '{
    "name": "decision-agent-inference-'${ENVIRONMENT}'",
    "tasks": [
      {
        "task_key": "batch_inference",
        "existing_cluster_id": "'$(get_cluster_id decision-agent-production-${ENVIRONMENT})'",
        "python_file": "dbfs:/Workspace/decision_agent/'${ENVIRONMENT}'/notebooks/inference/batch_inference.py"
      }
    ],
    "schedule": {
      "quartz_cron_expression": "0 0 4 * * ?",
      "timezone_id": "UTC"
    }
  }' || log_warning "Inference job may already exist"

  # Monitoring job
  log_info "Deploying monitoring job..."
  databricks jobs create --json '{
    "name": "decision-agent-monitoring-'${ENVIRONMENT}'",
    "tasks": [
      {
        "task_key": "model_monitoring",
        "existing_cluster_id": "'$(get_cluster_id decision-agent-production-${ENVIRONMENT})'",
        "python_file": "dbfs:/Workspace/decision_agent/'${ENVIRONMENT}'/notebooks/monitoring/model_monitor.py"
      }
    ],
    "schedule": {
      "quartz_cron_expression": "0 0 9 * * ?",
      "timezone_id": "UTC"
    }
  }' || log_warning "Monitoring job may already exist"

  log_success "All jobs deployed"
}

# Function to get cluster ID by name
get_cluster_id() {
  local cluster_name=$1
  databricks clusters list --output JSON | jq -r ".clusters[] | select(.cluster_name==\"${cluster_name}\") | .cluster_id"
}

# Function to deploy models
deploy_models() {
  if [[ "${SKIP_MODELS}" == true ]]; then
    log_info "Skipping model deployment"
    return
  fi

  log_info "Deploying models to Model Registry..."

  # Register champion model
  log_info "Registering champion model..."
  python -c "
import mlflow
mlflow.set_tracking_uri('databricks')

# Get latest champion model
client = mlflow.tracking.MlflowClient()
versions = client.search_model_versions('name=\"income_estimation_champion\"')

if versions:
    latest = max(versions, key=lambda v: int(v.version))
    print(f'Latest champion model: v{latest.version}')

    # Transition to Production
    client.transition_model_version_stage(
        name='income_estimation_champion',
        version=latest.version,
        stage='Production'
    )
    print('✓ Model transitioned to Production')
else:
    print('⚠ No champion model found')
"

  log_success "Models deployed"
}

# Function to set up monitoring
setup_monitoring() {
  log_info "Setting up monitoring..."

  # Create monitoring tables (if they don't exist)
  databricks sql execute --sql "
    CREATE TABLE IF NOT EXISTS decision_agent.model_predictions_log (
      prediction_id STRING,
      customer_id STRING,
      model_version STRING,
      prediction DOUBLE,
      confidence STRING,
      prediction_timestamp TIMESTAMP,
      features MAP<STRING, DOUBLE>
    )
    USING DELTA
    PARTITIONED BY (DATE(prediction_timestamp));

    CREATE TABLE IF NOT EXISTS decision_agent.model_monitoring_metrics (
      metric_id STRING,
      metric_name STRING,
      metric_value DOUBLE,
      metric_timestamp TIMESTAMP,
      model_version STRING,
      segment STRING
    )
    USING DELTA;

    CREATE TABLE IF NOT EXISTS decision_agent.alerts_log (
      alert_id STRING,
      alert_type STRING,
      severity STRING,
      message STRING,
      alert_timestamp TIMESTAMP,
      acknowledged BOOLEAN
    )
    USING DELTA;
  "

  log_success "Monitoring setup complete"
}

# Function to validate deployment
validate_deployment() {
  log_info "Validating deployment..."

  local errors=0

  # Check clusters
  log_info "Checking clusters..."
  for cluster in "decision-agent-features-${ENVIRONMENT}" \
                 "decision-agent-training-${ENVIRONMENT}" \
                 "decision-agent-production-${ENVIRONMENT}"; do
    if databricks clusters get --cluster-name "${cluster}" &> /dev/null; then
      log_success "✓ Cluster exists: ${cluster}"
    else
      log_error "✗ Cluster not found: ${cluster}"
      ((errors++))
    fi
  done

  # Check jobs
  log_info "Checking jobs..."
  for job in "decision-agent-features-${ENVIRONMENT}" \
             "decision-agent-training-${ENVIRONMENT}" \
             "decision-agent-inference-${ENVIRONMENT}" \
             "decision-agent-monitoring-${ENVIRONMENT}"; do
    if databricks jobs list --output JSON | jq -e ".jobs[] | select(.settings.name==\"${job}\")" > /dev/null; then
      log_success "✓ Job exists: ${job}"
    else
      log_error "✗ Job not found: ${job}"
      ((errors++))
    fi
  done

  # Check code uploaded
  log_info "Checking uploaded code..."
  if databricks workspace ls /Workspace/decision_agent/${ENVIRONMENT}/src &> /dev/null; then
    log_success "✓ Source code uploaded"
  else
    log_error "✗ Source code not found"
    ((errors++))
  fi

  # Check tables exist
  log_info "Checking Delta tables..."
  for table in "decision_agent.model_predictions_log" \
               "decision_agent.model_monitoring_metrics" \
               "decision_agent.alerts_log"; do
    if databricks sql execute --sql "DESCRIBE TABLE ${table}" &> /dev/null; then
      log_success "✓ Table exists: ${table}"
    else
      log_warning "⚠ Table not found: ${table}"
    fi
  done

  if [[ ${errors} -eq 0 ]]; then
    log_success "Deployment validation passed!"
    return 0
  else
    log_error "Deployment validation failed with ${errors} errors"
    return 1
  fi
}

# Function to print deployment summary
print_summary() {
  echo ""
  echo -e "${BLUE}========================================${NC}"
  echo -e "${BLUE}Deployment Summary${NC}"
  echo -e "${BLUE}========================================${NC}"
  echo "Environment: ${ENVIRONMENT}"
  echo "Databricks Workspace: $(databricks workspace ls / | head -1)"
  echo ""
  echo "Deployed Components:"
  echo "  ✓ Infrastructure (Terraform)"
  echo "  ✓ Databricks Clusters (3)"
  echo "  ✓ Databricks Jobs (4)"
  echo "  ✓ Source Code & Notebooks"
  echo "  ✓ Model Registry"
  echo "  ✓ Monitoring Tables"
  echo ""
  echo "Next Steps:"
  echo "  1. Run load tests: ./deployment/load_testing/run_load_tests.sh"
  echo "  2. Verify monitoring: databricks jobs run-now --job-name decision-agent-monitoring-${ENVIRONMENT}"
  echo "  3. Test inference: curl -X POST <api-endpoint>/predict -d '{\"customer_id\": \"CUST001\"}'"
  echo ""
  echo -e "${GREEN}Deployment complete!${NC}"
}

# Main execution
main() {
  check_prerequisites

  if [[ "${VALIDATE_ONLY}" == true ]]; then
    validate_deployment
    exit $?
  fi

  # Deploy infrastructure
  deploy_infrastructure

  if [[ "${SKIP_DATABRICKS}" == false ]]; then
    # Upload code
    upload_code

    # Create clusters
    create_clusters

    # Deploy jobs
    deploy_jobs

    # Deploy models
    deploy_models

    # Setup monitoring
    setup_monitoring
  fi

  # Validate deployment
  validate_deployment

  # Print summary
  print_summary
}

# Run main function
main
