#!/bin/bash
################################################################################
# Enable Performance Optimizations
#
# Activates all performance optimizations for the Decision Agent platform:
# 1. Feature caching with incremental updates
# 2. Inference optimization (quantization, batching)
# 3. Distributed training setup (optional)
# 4. Model compilation to ONNX
#
# Usage:
#   ./enable_optimizations.sh --all
#   ./enable_optimizations.sh --cache --inference
################################################################################

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Flags
ENABLE_CACHE=false
ENABLE_INFERENCE=false
ENABLE_DISTRIBUTED=false
ENABLE_ALL=false

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --all)
      ENABLE_ALL=true
      shift
      ;;
    --cache)
      ENABLE_CACHE=true
      shift
      ;;
    --inference)
      ENABLE_INFERENCE=true
      shift
      ;;
    --distributed)
      ENABLE_DISTRIBUTED=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

if [[ "${ENABLE_ALL}" == true ]]; then
  ENABLE_CACHE=true
  ENABLE_INFERENCE=true
  ENABLE_DISTRIBUTED=true
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Enabling Performance Optimizations${NC}"
echo -e "${BLUE}========================================${NC}"
echo "Feature Cache: ${ENABLE_CACHE}"
echo "Inference Optimization: ${ENABLE_INFERENCE}"
echo "Distributed Training: ${ENABLE_DISTRIBUTED}"
echo ""

# 1. Enable Feature Caching
if [[ "${ENABLE_CACHE}" == true ]]; then
  echo -e "${BLUE}[1/3] Enabling Feature Cache...${NC}"

  # Update configuration
  python -c "
import yaml

# Load config
with open('conf/optimization/performance_config.yaml') as f:
    config = yaml.safe_load(f)

# Enable feature caching
config['feature_caching']['enabled'] = True
config['feature_caching']['incremental_update']['enabled'] = True

# Save
with open('conf/optimization/performance_config.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)

print('✓ Feature caching enabled in configuration')
"

  # Create cache warming job
  echo "Creating cache warming job..."
  databricks jobs create --json '{
    "name": "decision-agent-cache-warming",
    "tasks": [{
      "task_key": "warm_cache",
      "python_file": "dbfs:/Workspace/decision_agent/production/jobs/warm_feature_cache.py"
    }],
    "schedule": {
      "quartz_cron_expression": "0 0 1 * * ?",
      "timezone_id": "UTC"
    }
  }' || echo "Job may already exist"

  # Run initial cache warming
  echo "Running initial cache warming..."
  python jobs/warm_feature_cache.py --top-n 100000

  echo -e "${GREEN}✓ Feature cache enabled${NC}"
  echo ""
fi

# 2. Enable Inference Optimization
if [[ "${ENABLE_INFERENCE}" == true ]]; then
  echo -e "${BLUE}[2/3] Enabling Inference Optimization...${NC}"

  # Update configuration
  python -c "
import yaml

with open('conf/optimization/performance_config.yaml') as f:
    config = yaml.safe_load(f)

# Enable inference optimizations
config['inference_optimization']['quantization']['enabled'] = True
config['inference_optimization']['prediction_caching']['enabled'] = True
config['inference_optimization']['batch_prediction']['adaptive_batching']['enabled'] = True

# Save
with open('conf/optimization/performance_config.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)

print('✓ Inference optimization enabled in configuration')
"

  # Quantize champion model
  echo "Quantizing champion model..."
  python -c "
import mlflow
from decision_agent.optimization.inference_optimizer import InferenceOptimizer

mlflow.set_tracking_uri('databricks')

# Load champion model
model = mlflow.pyfunc.load_model('models:/income_estimation_champion/Production')

# Quantize
optimizer = InferenceOptimizer(model._model_impl.python_model, model_type='sklearn')
quantized_model = optimizer.quantize(method='dynamic')

# Log quantized model
with mlflow.start_run(run_name='champion_quantized'):
    mlflow.sklearn.log_model(quantized_model, 'model')
    model_uri = f'runs:/{mlflow.active_run().info.run_id}/model'

    # Register
    model_details = mlflow.register_model(
        model_uri=model_uri,
        name='income_estimation_champion_quantized'
    )

    print(f'✓ Quantized model registered: v{model_details.version}')
"

  # Compile to ONNX
  echo "Compiling model to ONNX..."
  python -c "
import mlflow
from decision_agent.optimization.inference_optimizer import InferenceOptimizer

# Load and compile
model = mlflow.pyfunc.load_model('models:/income_estimation_champion/Production')
optimizer = InferenceOptimizer(model._model_impl.python_model, model_type='sklearn')

onnx_model = optimizer.compile_to_onnx(
    input_shape=(None, 20),
    output_path='models/champion.onnx'
)

print('✓ Model compiled to ONNX')
"

  echo -e "${GREEN}✓ Inference optimization enabled${NC}"
  echo ""
fi

# 3. Enable Distributed Training
if [[ "${ENABLE_DISTRIBUTED}" == true ]]; then
  echo -e "${BLUE}[3/3] Enabling Distributed Training...${NC}"

  # Update configuration
  python -c "
import yaml

with open('conf/optimization/performance_config.yaml') as f:
    config = yaml.safe_load(f)

# Enable distributed training
config['distributed_training']['enabled'] = True
config['distributed_training']['horovod']['num_workers'] = 4  # 4 GPUs

# Save
with open('conf/optimization/performance_config.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)

print('✓ Distributed training enabled in configuration')
"

  echo "Note: Distributed training requires GPU cluster."
  echo "Update cluster configuration to use GPU instances."

  echo -e "${GREEN}✓ Distributed training configuration updated${NC}"
  echo ""
fi

# Verify optimizations
echo -e "${BLUE}Verifying Optimizations...${NC}"

python -c "
import yaml

with open('conf/optimization/performance_config.yaml') as f:
    config = yaml.safe_load(f)

print('Current Optimization Status:')
print(f'  Feature Caching: {\"✓ Enabled\" if config[\"feature_caching\"][\"enabled\"] else \"✗ Disabled\"}')
print(f'  Quantization: {\"✓ Enabled\" if config[\"inference_optimization\"][\"quantization\"][\"enabled\"] else \"✗ Disabled\"}')
print(f'  Prediction Caching: {\"✓ Enabled\" if config[\"inference_optimization\"][\"prediction_caching\"][\"enabled\"] else \"✗ Disabled\"}')
print(f'  Distributed Training: {\"✓ Enabled\" if config[\"distributed_training\"][\"enabled\"] else \"✗ Disabled\"}')
"

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Optimizations Enabled Successfully!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Next Steps:"
echo "  1. Run benchmark tests to measure improvements"
echo "  2. Monitor cache hit rates"
echo "  3. Compare inference latency before/after"
echo "  4. Check model size reduction"
