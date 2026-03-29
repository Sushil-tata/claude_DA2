# Databricks notebook source
# MAGIC %md
# MAGIC # AutoML Model Optimization Pipeline
# MAGIC
# MAGIC This notebook runs comprehensive AutoML optimization:
# MAGIC 1. Databricks AutoML for baseline
# MAGIC 2. Hyperopt tuning for existing models
# MAGIC 3. Model comparison across families
# MAGIC 4. Best model selection and deployment
# MAGIC
# MAGIC **Goal:** Automatically find and deploy the best model

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

# Install dependencies
%pip install hyperopt mlflow

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import yaml
import mlflow
from pyspark.sql import SparkSession
from decision_agent.automl.databricks_automl import DatabricksAutoMLWrapper
from decision_agent.automl.hyperopt_tuner import HyperoptTuner, create_search_space_regression
from decision_agent.automl.model_comparison import ModelComparator
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Configuration

# COMMAND ----------

# Load AutoML config
with open("/dbfs/Workspace/decision_agent/production/conf/automl/automl_config.yaml") as f:
    config = yaml.safe_load(f)

print("AutoML Configuration:")
print(f"  Problem type: {config['databricks_automl']['problem_type']}")
print(f"  Target: {config['databricks_automl']['target_col']}")
print(f"  Max trials: {config['databricks_automl']['max_trials']}")
print(f"  Timeout: {config['databricks_automl']['timeout_minutes']} minutes")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Training Data

# COMMAND ----------

# Load training data
training_df = spark.table("decision_agent.training_data")

# Split into train/validation
train_df, val_df = training_df.randomSplit([0.8, 0.2], seed=42)

print(f"Training samples: {train_df.count():,}")
print(f"Validation samples: {val_df.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Databricks AutoML Baseline

# COMMAND ----------

print("=" * 80)
print("STEP 1: Running Databricks AutoML")
print("=" * 80)

# Initialize AutoML wrapper
automl = DatabricksAutoMLWrapper(
    target_col=config['databricks_automl']['target_col'],
    problem_type=config['databricks_automl']['problem_type']
)

# Run AutoML
best_trial = automl.train(
    dataset=train_df,
    timeout_minutes=config['databricks_automl']['timeout_minutes'],
    max_trials=config['databricks_automl']['max_trials'],
    metric=config['databricks_automl']['primary_metric'],
    exclude_cols=config['databricks_automl']['exclude_cols']
)

# Get results
print(f"✓ AutoML complete")
print(f"  Best metric ({config['databricks_automl']['primary_metric']}): {best_trial.metrics[automl.summary.primary_metric]:.4f}")
print(f"  Model URI: {best_trial.model_path}")

# Get feature importance
feature_importance = automl.get_feature_importance(top_n=10)
print(f"\nTop 10 features:")
for feature, importance in feature_importance.items():
    print(f"  {feature}: {importance:.4f}")

# Register AutoML model
automl_version = automl.register_best_model(
    model_name="income_estimation_automl",
    stage="Staging"
)

print(f"✓ AutoML model registered: v{automl_version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Hyperopt Tuning for Multiple Models

# COMMAND ----------

print("=" * 80)
print("STEP 2: Hyperparameter Tuning with Hyperopt")
print("=" * 80)

# Convert to pandas for sklearn models
train_pdf = train_df.toPandas()
val_pdf = val_df.toPandas()

feature_cols = [col for col in train_pdf.columns if col not in [
    config['databricks_automl']['target_col'],
    'customer_id',
    'as_of_date'
]]

X_train = train_pdf[feature_cols]
y_train = train_pdf[config['databricks_automl']['target_col']]
X_val = val_pdf[feature_cols]
y_val = val_pdf[config['databricks_automl']['target_col']]

# Dictionary to store best models
best_models = {}
best_params = {}

# Tune Gradient Boosting
print("\nTuning Gradient Boosting...")
gbm_search_space = create_search_space_regression("gradient_boosting")
gbm_tuner = HyperoptTuner(
    model_class=GradientBoostingRegressor,
    search_space=gbm_search_space,
    metric="rmse",
    max_evals=config['hyperopt']['max_evals'],
    use_spark=config['hyperopt']['use_spark']
)

gbm_best_params = gbm_tuner.tune(X_train, y_train, X_val, y_val,
                                  experiment_name="/Users/decision_agent/automl/gbm_tuning")
best_params['GradientBoosting'] = gbm_best_params
best_models['GradientBoosting'] = gbm_tuner.get_best_model(X_train, y_train)

print(f"✓ GBM tuned. Best RMSE: {gbm_tuner.best_score:.4f}")

# Tune Random Forest
print("\nTuning Random Forest...")
rf_search_space = create_search_space_regression("random_forest")
rf_tuner = HyperoptTuner(
    model_class=RandomForestRegressor,
    search_space=rf_search_space,
    metric="rmse",
    max_evals=config['hyperopt']['max_evals'],
    use_spark=False  # RF doesn't support SparkTrials well
)

rf_best_params = rf_tuner.tune(X_train, y_train, X_val, y_val,
                                experiment_name="/Users/decision_agent/automl/rf_tuning")
best_params['RandomForest'] = rf_best_params
best_models['RandomForest'] = rf_tuner.get_best_model(X_train, y_train)

print(f"✓ RF tuned. Best RMSE: {rf_tuner.best_score:.4f}")

# Tune XGBoost
print("\nTuning XGBoost...")
xgb_search_space = create_search_space_regression("xgboost")
xgb_tuner = HyperoptTuner(
    model_class=xgb.XGBRegressor,
    search_space=xgb_search_space,
    metric="rmse",
    max_evals=config['hyperopt']['max_evals']
)

xgb_best_params = xgb_tuner.tune(X_train, y_train, X_val, y_val,
                                  experiment_name="/Users/decision_agent/automl/xgb_tuning")
best_params['XGBoost'] = xgb_best_params
best_models['XGBoost'] = xgb_tuner.get_best_model(X_train, y_train)

print(f"✓ XGBoost tuned. Best RMSE: {xgb_tuner.best_score:.4f}")

print("\n✓ All models tuned with Hyperopt")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Model Comparison

# COMMAND ----------

print("=" * 80)
print("STEP 3: Model Comparison")
print("=" * 80)

# Prepare models for comparison
models_to_compare = [
    ("GradientBoosting_Tuned", best_models['GradientBoosting']),
    ("RandomForest_Tuned", best_models['RandomForest']),
    ("XGBoost_Tuned", best_models['XGBoost'])
]

# Add baseline models
models_to_compare.extend([
    ("GradientBoosting_Baseline", GradientBoostingRegressor(**config['model_comparison']['baseline_models'][0]['params'])),
    ("RandomForest_Baseline", RandomForestRegressor(**config['model_comparison']['baseline_models'][1]['params']))
])

# Create comparator
comparator = ModelComparator(
    models=models_to_compare,
    metrics=config['model_comparison']['metrics'],
    cv_folds=config['model_comparison']['cv_folds'],
    problem_type="regression"
)

# Run comparison
comparison_df = comparator.compare(
    X_train, y_train, X_val, y_val,
    experiment_name="/Users/decision_agent/automl/model_comparison"
)

# Display results
print("\nModel Comparison Results:")
print(comparison_df.to_string())

# Get recommendation
recommendation = comparator.get_recommendation(
    primary_metric=config['model_comparison']['primary_metric'],
    max_training_time=config['model_comparison']['constraints']['max_training_time_seconds'],
    max_prediction_time=config['model_comparison']['constraints']['max_prediction_time_seconds']
)

print(f"\n{'='*80}")
print("DEPLOYMENT RECOMMENDATION")
print(f"{'='*80}")
print(f"Recommended Model: {recommendation['recommended_model']}")
print(f"{config['model_comparison']['primary_metric'].upper()}: {recommendation['primary_metric_value']:.4f}")
print(f"Training Time: {recommendation['training_time_sec']:.2f}s")
print(f"Prediction Time: {recommendation['prediction_time_sec']:.4f}s")
print(f"\nReasoning:\n{recommendation['reasoning']}")
print(f"{'='*80}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Deploy Best Model

# COMMAND ----------

print("=" * 80)
print("STEP 4: Deploying Best Model")
print("=" * 80)

# Get best model
best_model_name = recommendation['recommended_model']
best_model = dict(models_to_compare)[best_model_name]

# Register with MLflow
with mlflow.start_run(run_name=f"champion_{best_model_name}"):
    # Log model
    mlflow.sklearn.log_model(best_model, "model")

    # Log metrics
    mlflow.log_metric("test_rmse", recommendation['primary_metric_value'])
    mlflow.log_metric("training_time_sec", recommendation['training_time_sec'])

    # Log params
    mlflow.log_params(best_params.get(best_model_name.replace('_Tuned', ''), {}))

    # Register model
    model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
    model_details = mlflow.register_model(
        model_uri=model_uri,
        name="income_estimation_champion"
    )

print(f"✓ Model registered: income_estimation_champion v{model_details.version}")

# Transition to Production
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name="income_estimation_champion",
    version=model_details.version,
    stage="Production"
)

print(f"✓ Model transitioned to Production")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Compare with Previous Champion

# COMMAND ----------

print("=" * 80)
print("STEP 5: Comparing with Previous Champion")
print("=" * 80)

# Get previous champion model
client = mlflow.tracking.MlflowClient()
production_versions = client.get_latest_versions("income_estimation_champion", stages=["Production"])

if len(production_versions) > 1:
    # There's a previous version
    previous_version = [v for v in production_versions if v.version != model_details.version][0]

    print(f"Comparing:")
    print(f"  New champion: v{model_details.version}")
    print(f"  Previous champion: v{previous_version.version}")

    # Get previous metrics
    previous_run = client.get_run(previous_version.run_id)
    previous_rmse = previous_run.data.metrics.get("test_rmse")

    if previous_rmse:
        improvement_pct = ((previous_rmse - recommendation['primary_metric_value']) / previous_rmse) * 100

        print(f"\nMetrics:")
        print(f"  Previous RMSE: {previous_rmse:.4f}")
        print(f"  New RMSE: {recommendation['primary_metric_value']:.4f}")
        print(f"  Improvement: {improvement_pct:+.2f}%")

        if improvement_pct > 0:
            print(f"  ✓ New model is better!")
        else:
            print(f"  ⚠ New model is worse - consider rollback")
else:
    print("No previous champion found - this is the first Production model")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

print("\n" + "=" * 80)
print("AUTOML OPTIMIZATION COMPLETE")
print("=" * 80)
print("\nSummary:")
print(f"  ✓ Databricks AutoML completed")
print(f"  ✓ Hyperopt tuning completed (3 models)")
print(f"  ✓ Model comparison completed (5 models)")
print(f"  ✓ Best model deployed to Production")
print("")
print(f"Champion Model: {best_model_name}")
print(f"Version: v{model_details.version}")
print(f"RMSE: {recommendation['primary_metric_value']:.4f}")
print("")
print("Next Steps:")
print("  1. Monitor model performance in production")
print("  2. Run batch inference with new model")
print("  3. Compare predictions with previous version")
print("=" * 80)
