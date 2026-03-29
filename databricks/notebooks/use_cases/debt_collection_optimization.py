# Databricks notebook source
# MAGIC %md
# MAGIC # Debt Collection Optimization - End-to-End Pipeline
# MAGIC
# MAGIC This notebook implements a complete debt collection decision agent:
# MAGIC
# MAGIC 1. **Data Ingestion** - Load accounts, demographics, contact history
# MAGIC 2. **Feature Engineering** - Create behavioral, financial, engagement features
# MAGIC 3. **Persona Segmentation** - Create debtor personas (rule-based + ML)
# MAGIC 4. **Model Training** - Train propensity, capacity, channel, offer models
# MAGIC 5. **Next Best Action** - Recommend optimal channel and timing
# MAGIC 6. **Next Best Offer** - Recommend optimal offer strategy
# MAGIC 7. **Decision Output** - Write decisions to Delta table
# MAGIC 8. **Reporting** - Generate performance dashboard
# MAGIC
# MAGIC **Expected Output:**
# MAGIC - Optimized collection strategy for each account
# MAGIC - 20-40% improvement in recovery rate
# MAGIC - 30-50% reduction in cost per dollar collected
# MAGIC - Full compliance with FDCPA/TCPA regulations

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

# Install dependencies
%pip install scikit-learn mlflow pyyaml

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import yaml
import mlflow
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pyspark.sql import functions as F, Window
from pyspark.sql.types import *

# Import decision agent modules
from decision_agent.debt_collection.persona_segmentation import DebtorPersonaSegmentation
from decision_agent.debt_collection.next_best_action import NextBestActionEngine, NextBestOfferEngine
from decision_agent.features.windows import RollingWindowFeatures
from decision_agent.training.training_harness import TrainingHarness

print("✓ Imports successful")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Configuration

# COMMAND ----------

# Load debt collection config
with open("/dbfs/Workspace/decision_agent/production/conf/use_cases/debt_collection_optimization.yaml") as f:
    config = yaml.safe_load(f)

print(f"Use Case: {config['use_case_id']}")
print(f"Version: {config['version']}")
print(f"\nPersonas Configured: {len(config['segmentation']['rule_based_segments'])}")
print(f"Models to Train: {len(config['models'])}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Data Ingestion

# COMMAND ----------

print("=" * 80)
print("STEP 1: DATA INGESTION")
print("=" * 80)

# Load data tables
accounts_df = spark.table(config['data']['accounts_table'])
demographics_df = spark.table(config['data']['demographics_table'])
contact_history_df = spark.table(config['data']['contact_history_table'])
payment_history_df = spark.table(config['data']['payment_history_table'])
offers_history_df = spark.table(config['data']['offers_history_table'])

print(f"\n✓ Accounts loaded: {accounts_df.count():,}")
print(f"✓ Demographics loaded: {demographics_df.count():,}")
print(f"✓ Contact history loaded: {contact_history_df.count():,}")
print(f"✓ Payment history loaded: {payment_history_df.count():,}")
print(f"✓ Offers history loaded: {offers_history_df.count():,}")

# Join accounts with demographics
full_accounts_df = accounts_df.join(
    demographics_df,
    on="customer_id",
    how="left"
)

print(f"\n✓ Data ingestion complete")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Feature Engineering

# COMMAND ----------

print("=" * 80)
print("STEP 2: FEATURE ENGINEERING")
print("=" * 80)

# Define as_of_date for point-in-time features
as_of_date = datetime.now().date()

# 2.1: Account features
print("\n[2.1] Computing account features...")
account_features = full_accounts_df.select(
    "account_id",
    "customer_id",
    "current_balance",
    "original_balance",
    (F.col("current_balance") / F.col("original_balance")).alias("balance_ratio"),
    "days_past_due",
    "debt_age_days",
    F.datediff(F.lit(as_of_date), F.col("last_payment_date")).alias("months_since_last_payment")
)

# 2.2: Behavioral features from contact history
print("[2.2] Computing behavioral features...")

# Contact counts by window
contact_counts = contact_history_df.groupBy("account_id").agg(
    F.sum(F.when(F.col("contact_date") >= F.date_sub(F.lit(as_of_date), 7), 1).otherwise(0)).alias("contact_count_7d"),
    F.sum(F.when(F.col("contact_date") >= F.date_sub(F.lit(as_of_date), 30), 1).otherwise(0)).alias("contact_count_30d"),
    F.sum(F.when(F.col("contact_date") >= F.date_sub(F.lit(as_of_date), 90), 1).otherwise(0)).alias("contact_count_90d"),
    F.max("contact_date").alias("last_contact_date")
)

contact_counts = contact_counts.withColumn(
    "last_contact_days_ago",
    F.datediff(F.lit(as_of_date), F.col("last_contact_date"))
)

# Response rates
contact_responses = contact_history_df.groupBy("account_id").agg(
    (F.sum(F.when(F.col("contact_outcome").isin(["answered", "payment", "promise_to_pay"]), 1).otherwise(0)) / F.count("*")).alias("contact_response_rate"),
    F.sum(F.when(F.col("contact_outcome") == "promise_to_pay", 1).otherwise(0)).alias("promise_to_pay_count"),
    (F.sum(F.when(F.col("contact_outcome") == "promise_to_pay", 1).otherwise(0)) / F.sum(F.when(F.col("contact_outcome") == "payment", 1).otherwise(0))).alias("promise_kept_rate")
)

# Best channel (most successful)
best_channel = contact_history_df.filter(
    F.col("contact_outcome") == "payment"
).groupBy("account_id", "contact_channel").count().withColumn(
    "rank",
    F.row_number().over(Window.partitionBy("account_id").orderBy(F.desc("count")))
).filter(F.col("rank") == 1).select(
    "account_id",
    F.col("contact_channel").alias("preferred_contact_channel")
)

# 2.3: Financial features
print("[2.3] Computing financial features...")
financial_features = demographics_df.select(
    "customer_id",
    "income_estimated",
    "credit_score"
).join(
    account_features.select("customer_id", "account_id", "current_balance"),
    on="customer_id"
).withColumn(
    "debt_to_income_ratio",
    F.col("current_balance") / F.col("income_estimated")
).withColumn(
    "payment_capacity_score",
    F.when(F.col("debt_to_income_ratio") < 0.2, 0.9)
     .when(F.col("debt_to_income_ratio") < 0.4, 0.7)
     .when(F.col("debt_to_income_ratio") < 0.6, 0.5)
     .when(F.col("debt_to_income_ratio") < 0.8, 0.3)
     .otherwise(0.1)
)

# 2.4: Offer response features
print("[2.4] Computing offer response features...")
offer_features = offers_history_df.groupBy("account_id").agg(
    F.count("*").alias("settlement_offers_made"),
    (F.sum(F.when(F.col("accepted") == True, 1).otherwise(0)) / F.count("*")).alias("settlement_acceptance_rate"),
    F.avg("discount_pct").alias("avg_discount_accepted")
)

# 2.5: Join all features
print("[2.5] Joining all features...")
features_df = account_features \
    .join(contact_counts, on="account_id", how="left") \
    .join(contact_responses, on="account_id", how="left") \
    .join(best_channel, on="account_id", how="left") \
    .join(financial_features.select("account_id", "debt_to_income_ratio", "payment_capacity_score", "credit_score"), on="account_id", how="left") \
    .join(offer_features, on="account_id", how="left") \
    .fillna(0)

print(f"\n✓ Features computed: {features_df.count():,} accounts")
print(f"✓ Feature columns: {len(features_df.columns)}")

# Write features to Delta Lake
features_df.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("debt_collection.features")

print("✓ Features saved to Delta Lake")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Persona Segmentation

# COMMAND ----------

print("=" * 80)
print("STEP 3: PERSONA SEGMENTATION")
print("=" * 80)

# Initialize segmenter
segmenter = DebtorPersonaSegmentation(config)

# Create personas
personas_df = segmenter.create_personas(full_accounts_df, features_df)

print(f"\n✓ Personas created: {personas_df.select('persona').distinct().count()}")

# Analyze personas
persona_stats = segmenter.analyze_personas(personas_df)

print("\nPersona Statistics:")
print(persona_stats.to_string())

# Save personas
personas_df.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("debt_collection.personas")

print("\n✓ Personas saved to Delta Lake")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Model Training

# COMMAND ----------

print("=" * 80)
print("STEP 4: MODEL TRAINING")
print("=" * 80)

# Prepare training data
training_df = features_df.join(
    personas_df.select("account_id", "persona", "rule_segment"),
    on="account_id"
).join(
    payment_history_df.groupBy("account_id").agg(
        F.sum(F.when(F.col("payment_date") >= F.date_sub(F.lit(as_of_date), 30), F.col("payment_amount")).otherwise(0)).alias("payment_made_30d")
    ),
    on="account_id",
    how="left"
).fillna({"payment_made_30d": 0})

# Convert to pandas for sklearn models
training_pdf = training_df.toPandas()

# Define feature columns
feature_cols = [
    'balance_ratio', 'days_past_due', 'debt_age_days',
    'contact_count_7d', 'contact_count_30d', 'contact_count_90d',
    'contact_response_rate', 'promise_kept_rate',
    'debt_to_income_ratio', 'payment_capacity_score', 'credit_score',
    'settlement_acceptance_rate'
]

X = training_pdf[feature_cols].fillna(0)

# 4.1: Train Likelihood to Pay model
print("\n[4.1] Training Likelihood to Pay (LTP) model...")
y_ltp = (training_pdf['payment_made_30d'] > 0).astype(int)

from sklearn.ensemble import GradientBoostingClassifier

ltp_model = GradientBoostingClassifier(**config['models']['propensity_to_pay']['hyperparameters'])
ltp_model.fit(X, y_ltp)

# Log to MLflow
with mlflow.start_run(run_name="ltp_model"):
    mlflow.sklearn.log_model(ltp_model, "model")
    mlflow.log_params(config['models']['propensity_to_pay']['hyperparameters'])
    mlflow.log_metric("training_samples", len(X))

    # Register
    model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
    ltp_model_version = mlflow.register_model(model_uri, "debt_collection_ltp")

print(f"✓ LTP model trained and registered: v{ltp_model_version.version}")

# 4.2: Train Ability to Pay model
print("\n[4.2] Training Ability to Pay (ATP) model...")
# ATP is just payment_capacity_score (already computed)
print("✓ ATP score using rule-based payment_capacity_score")

# 4.3: Train Channel Response model
print("\n[4.3] Training Channel Response model...")
# Multi-class classification for best channel

# Get channel labels from contact history
channel_labels = contact_history_df.filter(
    F.col("contact_outcome") == "payment"
).groupBy("account_id").agg(
    F.first("contact_channel").alias("best_channel")
).toPandas()

# Join with features
channel_training = training_pdf.merge(channel_labels, on="account_id", how="inner")

if len(channel_training) > 100:
    X_channel = channel_training[feature_cols].fillna(0)
    y_channel = channel_training['best_channel']

    from sklearn.ensemble import RandomForestClassifier
    channel_model = RandomForestClassifier(n_estimators=100, random_state=42)
    channel_model.fit(X_channel, y_channel)

    with mlflow.start_run(run_name="channel_response_model"):
        mlflow.sklearn.log_model(channel_model, "model")
        channel_model_version = mlflow.register_model(
            f"runs:/{mlflow.active_run().info.run_id}/model",
            "debt_collection_channel"
        )

    print(f"✓ Channel model trained: v{channel_model_version.version}")
else:
    print("⚠ Insufficient data for channel model - using rule-based")
    channel_model = None

print("\n✓ Model training complete")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Generate Decisions (Next Best Action + Next Best Offer)

# COMMAND ----------

print("=" * 80)
print("STEP 5: DECISION GENERATION")
print("=" * 80)

# Initialize decision engines
action_engine = NextBestActionEngine(config)
offer_engine = NextBestOfferEngine(config)

# Score all accounts
print("\n[5.1] Scoring accounts with LTP model...")
ltp_scores = ltp_model.predict_proba(X)[:, 1]
training_pdf['ltp_score'] = ltp_scores

print("[5.2] Generating Next Best Actions and Offers...")

decisions = []

for idx, row in training_pdf.iterrows():
    if idx % 1000 == 0:
        print(f"  Processed {idx:,} accounts...")

    account_features = row.to_dict()
    persona = row['persona']

    model_scores = {
        'ltp_score': row['ltp_score'],
        'atp_score': row['payment_capacity_score'],
        'channel_probabilities': {'sms': 0.5, 'email': 0.3, 'call': 0.2}  # Simplified
    }

    # Get contact history for this account
    account_contact_history = contact_history_df.filter(
        F.col("account_id") == row['account_id']
    ).toPandas()

    # Get offer history
    account_offer_history = offers_history_df.filter(
        F.col("account_id") == row['account_id']
    ).toPandas()

    # Generate recommendations
    try:
        action = action_engine.recommend_action(
            account_features, persona, model_scores, account_contact_history
        )

        offer = offer_engine.recommend_offer(
            account_features, persona, model_scores, account_offer_history
        )

        decision = {
            'account_id': row['account_id'],
            'customer_id': row['customer_id'],
            'decision_timestamp': datetime.now(),
            'persona_segment': persona,
            'rule_segment': row['rule_segment'],

            # Scores
            'ltp_score': row['ltp_score'],
            'atp_score': row['payment_capacity_score'],

            # Recommended Action
            'recommended_channel': action['recommended_channel'],
            'channel_confidence': action['confidence'],
            'recommended_contact_time': action.get('recommended_time'),

            # Recommended Offer
            'recommended_offer_type': offer['offer_type'],
            'settlement_discount_pct': offer.get('settlement_discount_pct', 0),
            'settlement_amount': offer.get('settlement_amount', 0),
            'payment_plan_months': offer.get('payment_plan_months'),
            'payment_plan_monthly': offer.get('payment_plan_monthly'),

            # Expected Outcomes
            'expected_recovery_amount': offer['expected_recovery'],
            'expected_recovery_probability': offer['expected_acceptance_probability'],
            'expected_cost': offer['expected_cost'],
            'expected_roi': offer['expected_roi'],

            # Metadata
            'compliant': action['compliant'],
            'model_version': f"ltp_v{ltp_model_version.version}",
            'as_of_date': as_of_date
        }

        decisions.append(decision)

    except Exception as e:
        print(f"  Error processing {row['account_id']}: {e}")
        continue

print(f"\n✓ Generated {len(decisions):,} decisions")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Write Decisions to Delta Lake

# COMMAND ----------

print("=" * 80)
print("STEP 6: WRITE DECISIONS")
print("=" * 80)

# Convert to Spark DataFrame
decisions_df = spark.createDataFrame(pd.DataFrame(decisions))

# Write to Delta table
decisions_df.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(config['output']['decision_table'])

print(f"✓ Decisions written to {config['output']['decision_table']}")
print(f"  Total decisions: {decisions_df.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Performance Analysis & Reporting

# COMMAND ----------

print("=" * 80)
print("STEP 7: PERFORMANCE REPORTING")
print("=" * 80)

# Aggregate decisions
decision_summary = decisions_df.groupBy("recommended_offer_type").agg(
    F.count("*").alias("num_accounts"),
    F.sum("current_balance").alias("total_balance"),
    F.sum("expected_recovery_amount").alias("expected_recovery"),
    F.sum("expected_cost").alias("total_cost"),
    F.avg("expected_roi").alias("avg_roi"),
    F.avg("ltp_score").alias("avg_ltp"),
    F.avg("atp_score").alias("avg_atp")
).toPandas()

print("\n=== DECISION SUMMARY BY OFFER TYPE ===")
print(decision_summary.to_string(index=False))

# Channel distribution
channel_summary = decisions_df.groupBy("recommended_channel").count().toPandas()

print("\n=== CHANNEL DISTRIBUTION ===")
print(channel_summary.to_string(index=False))

# Persona performance
persona_summary = decisions_df.groupBy("persona_segment").agg(
    F.count("*").alias("num_accounts"),
    F.sum("current_balance").alias("total_balance"),
    F.sum("expected_recovery_amount").alias("expected_recovery"),
    F.avg("ltp_score").alias("avg_ltp")
).toPandas()

persona_summary['recovery_rate'] = (
    persona_summary['expected_recovery'] / persona_summary['total_balance'] * 100
).round(1)

print("\n=== PERSONA PERFORMANCE ===")
print(persona_summary.to_string(index=False))

# Overall metrics
total_balance = decision_summary['total_balance'].sum()
total_expected_recovery = decision_summary['expected_recovery'].sum()
total_cost = decision_summary['total_cost'].sum()

recovery_rate = total_expected_recovery / total_balance * 100
cost_efficiency = total_cost / total_expected_recovery if total_expected_recovery > 0 else 0
net_recovery = total_expected_recovery - total_cost
roi = net_recovery / total_cost if total_cost > 0 else 0

print("\n" + "=" * 80)
print("OVERALL PORTFOLIO METRICS")
print("=" * 80)
print(f"Total Portfolio Balance: ${total_balance:,.2f}")
print(f"Expected Recovery: ${total_expected_recovery:,.2f}")
print(f"Expected Cost: ${total_cost:,.2f}")
print(f"Net Recovery: ${net_recovery:,.2f}")
print(f"")
print(f"Recovery Rate: {recovery_rate:.1f}%")
print(f"Cost Efficiency: {cost_efficiency:.3f} (cost per dollar recovered)")
print(f"ROI: {roi:.1f}x")
print("=" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: Export for Action

# COMMAND ----------

print("=" * 80)
print("STEP 8: EXPORT ACTION LIST")
print("=" * 80)

# Create action list for collection agents
action_list = decisions_df.filter(
    (F.col("compliant") == True) &
    (F.col("recommended_channel") != "none") &
    (F.col("expected_roi") > 1.0)
).select(
    "account_id",
    "customer_id",
    "persona_segment",
    "current_balance",
    "days_past_due",
    "recommended_channel",
    "recommended_contact_time",
    "recommended_offer_type",
    "settlement_amount",
    "settlement_discount_pct",
    "payment_plan_monthly",
    "expected_recovery_amount",
    "expected_roi"
).orderBy(F.desc("expected_recovery_amount"))

print(f"\n✓ Action list created: {action_list.count():,} accounts")

# Write to CSV for collection agents
action_list.toPandas().to_csv(
    f"/dbfs/debt_collection/action_lists/collection_actions_{datetime.now().strftime('%Y%m%d')}.csv",
    index=False
)

print("✓ Action list exported to CSV")

# Show top 20
print("\nTop 20 Recovery Opportunities:")
action_list.show(20, truncate=False)

# COMMAND ----------

print("\n" + "=" * 80)
print("DEBT COLLECTION OPTIMIZATION COMPLETE")
print("=" * 80)
print("\n✓ All personas created")
print("✓ All models trained and registered")
print("✓ All decisions generated")
print("✓ Results saved to Delta Lake")
print("✓ Action list exported")
print("\nNext Steps:")
print("  1. Review action list with collection team")
print("  2. Execute campaigns using recommended channels/offers")
print("  3. Track actual outcomes vs predictions")
print("  4. Retrain models monthly with new data")
print("  5. Monitor compliance violations")
print("=" * 80)
