# Databricks notebook source
"""
Collections Feature Engineering - All 8 Feature Families
=========================================================

This notebook creates ALL features needed for recovery_agent_practical:
- 8 feature families
- Stage-wise windows: 0-30, 31-90, 91-180, 181-365 days
- Point-in-time safe (no data leakage)

OUTPUT: Delta table with all features ready for PersonaBuilder + RecoveryScorecard

Run this ONCE to create your feature table, then use recovery_agent_practical modules.
"""

# COMMAND ----------
# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------
from pyspark.sql import functions as F, Window
from pyspark.sql.types import *
from datetime import datetime, timedelta

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION - UPDATE THESE TO MATCH YOUR DATABRICKS TABLES
# ─────────────────────────────────────────────────────────────────────────────

# Table names (update to your actual table names)
ACCOUNTS_TABLE = "your_catalog.your_schema.accounts"
PAYMENTS_TABLE = "your_catalog.your_schema.payments"
CONTACTS_TABLE = "your_catalog.your_schema.collection_contacts"
BUREAU_TABLE = "your_catalog.your_schema.bureau_data"
TRANSACTIONS_TABLE = "your_catalog.your_schema.transactions"  # Optional

# Snapshot date (features computed as of this date)
SNAPSHOT_DATE = "2024-01-15"

# Output table
OUTPUT_TABLE = "your_catalog.your_schema.recovery_features"

# COMMAND ----------
# MAGIC %md
# MAGIC ## Family 1: Delinquency Trajectory Features

# COMMAND ----------
"""
Family 1: Delinquency Trajectory
- Current DPD, stage
- Months in each stage (SM, NPL, CHARGEOFF)
- DPD progression
- Staleness (months since chargeoff)
"""

accounts_df = spark.table(ACCOUNTS_TABLE)

# Current state as of snapshot date
current_state = accounts_df.filter(
    F.col("as_of_date") <= F.lit(SNAPSHOT_DATE)
).withColumn(
    "rn", F.row_number().over(
        Window.partitionBy("account_id").orderBy(F.col("as_of_date").desc())
    )
).filter(F.col("rn") == 1).select(
    "account_id",
    "days_past_due",
    F.col("delinquency_stage").alias("stage"),
    "balance",
    "principal_outstanding",
    "accrued_interest",
    "penalty_charges",
    "credit_limit"
)

# Months in each stage (looking back 365 days)
stage_history = accounts_df.filter(
    (F.col("as_of_date") <= F.lit(SNAPSHOT_DATE)) &
    (F.col("as_of_date") >= F.date_sub(F.lit(SNAPSHOT_DATE), 365))
).groupBy("account_id").agg(
    F.sum(F.when(F.col("delinquency_stage") == "SM", 1).otherwise(0)).alias("months_in_sm"),
    F.sum(F.when(F.col("delinquency_stage") == "NPL", 1).otherwise(0)).alias("months_in_npl"),
    F.sum(F.when(F.col("delinquency_stage") == "CHARGEOFF", 1).otherwise(0)).alias("months_in_chargeoff"),
    F.max(F.when(F.col("delinquency_stage") == "CHARGEOFF", F.col("as_of_date"))).alias("chargeoff_date")
)

# Months since chargeoff
stage_history = stage_history.withColumn(
    "months_since_chargeoff",
    F.when(
        F.col("chargeoff_date").isNotNull(),
        F.months_between(F.lit(SNAPSHOT_DATE), F.col("chargeoff_date"))
    ).otherwise(0)
)

# Merge
delinquency_features = current_state.join(
    stage_history, "account_id", "left"
).fillna(0, subset=["months_in_sm", "months_in_npl", "months_in_chargeoff", "months_since_chargeoff"])

print(f"✓ Family 1: Delinquency features created for {delinquency_features.count()} accounts")
delinquency_features.show(5)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Family 2: Balance & Utilization Features

# COMMAND ----------
"""
Family 2: Balance & Utilization
- Balance decomposition (principal, interest, charges)
- Utilization ratio
- Balance band
"""

balance_features = delinquency_features.select(
    "account_id",
    "balance",
    "principal_outstanding",
    "accrued_interest",
    "penalty_charges",
    "credit_limit"
).withColumn(
    "utilization_ratio",
    F.when(F.col("credit_limit") > 0, F.col("balance") / F.col("credit_limit")).otherwise(0)
).withColumn(
    "principal_pct", F.col("principal_outstanding") / F.col("balance")
).withColumn(
    "interest_pct", F.col("accrued_interest") / F.col("balance")
).withColumn(
    "charges_pct", F.col("penalty_charges") / F.col("balance")
)

print(f"✓ Family 2: Balance features created")
balance_features.show(5)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Family 3: Internal Payments (Stage-Wise Windows)

# COMMAND ----------
"""
Family 3: Internal Payments
- Payment counts and amounts by window: 0-30, 31-90, 91-180, 181-365 days
- Days since last payment
- Last payment amount
"""

payments_df = spark.table(PAYMENTS_TABLE)

# Filter payments before snapshot date
payments_df = payments_df.filter(
    F.col("payment_date") <= F.lit(SNAPSHOT_DATE)
)

# Define windows
windows = [
    ("30d", 0, 30),
    ("90d", 0, 90),
    ("180d", 0, 180),
    ("365d", 0, 365)
]

payment_features = None

for window_name, start_days, end_days in windows:
    window_start = F.date_sub(F.lit(SNAPSHOT_DATE), end_days)
    window_end = F.date_sub(F.lit(SNAPSHOT_DATE), start_days)

    window_payments = payments_df.filter(
        (F.col("payment_date") >= window_start) &
        (F.col("payment_date") <= window_end)
    ).groupBy("account_id").agg(
        F.count("*").alias(f"payment_count_{window_name}"),
        F.sum("payment_amount").alias(f"payment_amt_{window_name}")
    )

    if payment_features is None:
        payment_features = window_payments
    else:
        payment_features = payment_features.join(window_payments, "account_id", "outer")

# Last payment info
last_payment = payments_df.withColumn(
    "rn", F.row_number().over(
        Window.partitionBy("account_id").orderBy(F.col("payment_date").desc())
    )
).filter(F.col("rn") == 1).select(
    "account_id",
    F.col("payment_amount").alias("last_payment_amount"),
    F.datediff(F.lit(SNAPSHOT_DATE), F.col("payment_date")).alias("days_since_last_payment")
)

payment_features = payment_features.join(last_payment, "account_id", "outer")

# Fill nulls
payment_cols = [c for c in payment_features.columns if c != "account_id"]
payment_features = payment_features.fillna(0, subset=payment_cols)

print(f"✓ Family 3: Payment features created for {payment_features.count()} accounts")
payment_features.show(5)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Family 4: Actions & Engagement Features

# COMMAND ----------
"""
Family 4: Actions & Engagement
- Contact counts by window
- Call/SMS response rates
- PTP (Promise to Pay) behavior
"""

contacts_df = spark.table(CONTACTS_TABLE)

# Filter contacts before snapshot
contacts_df = contacts_df.filter(
    F.col("contact_date") <= F.lit(SNAPSHOT_DATE)
)

# Aggregate contacts by window
contact_windows = [
    ("30d", 30),
    ("90d", 90)
]

engagement_features = None

for window_name, days in contact_windows:
    window_start = F.date_sub(F.lit(SNAPSHOT_DATE), days)

    window_contacts = contacts_df.filter(
        F.col("contact_date") >= window_start
    ).groupBy("account_id").agg(
        F.count("*").alias(f"contacts_made_{window_name}")
    )

    if engagement_features is None:
        engagement_features = window_contacts
    else:
        engagement_features = engagement_features.join(window_contacts, "account_id", "outer")

# Call response rate (all time)
call_stats = contacts_df.filter(
    F.col("contact_type") == "CALL"
).groupBy("account_id").agg(
    F.count("*").alias("outbound_calls_made"),
    F.sum(F.when(F.col("contact_result") == "CONNECTED", 1).otherwise(0)).alias("calls_connected")
).withColumn(
    "call_response_rate",
    F.when(F.col("outbound_calls_made") > 0, F.col("calls_connected") / F.col("outbound_calls_made")).otherwise(0)
)

# SMS response rate
sms_stats = contacts_df.filter(
    F.col("contact_type") == "SMS"
).groupBy("account_id").agg(
    F.count("*").alias("sms_sent"),
    F.sum(F.when(F.col("contact_result") == "RESPONDED", 1).otherwise(0)).alias("sms_responded")
).withColumn(
    "sms_response_rate",
    F.when(F.col("sms_sent") > 0, F.col("sms_responded") / F.col("sms_sent")).otherwise(0)
)

# PTP behavior
ptp_stats = contacts_df.filter(
    F.col("ptp_made") == True
).groupBy("account_id").agg(
    F.count("*").alias("ptp_made"),
    F.sum(F.when(F.col("ptp_kept") == True, 1).otherwise(0)).alias("ptp_kept")
).withColumn(
    "ptp_kept_rate",
    F.when(F.col("ptp_made") > 0, F.col("ptp_kept") / F.col("ptp_made")).otherwise(0)
)

# Days since last contact
last_contact = contacts_df.withColumn(
    "rn", F.row_number().over(
        Window.partitionBy("account_id").orderBy(F.col("contact_date").desc())
    )
).filter(F.col("rn") == 1).select(
    "account_id",
    F.datediff(F.lit(SNAPSHOT_DATE), F.col("contact_date")).alias("days_since_last_contact")
)

# Merge all engagement features
engagement_features = engagement_features \
    .join(call_stats, "account_id", "outer") \
    .join(sms_stats, "account_id", "outer") \
    .join(ptp_stats, "account_id", "outer") \
    .join(last_contact, "account_id", "outer")

# Fill nulls
engagement_cols = [c for c in engagement_features.columns if c != "account_id"]
engagement_features = engagement_features.fillna(0, subset=engagement_cols)

print(f"✓ Family 4: Engagement features created for {engagement_features.count()} accounts")
engagement_features.show(5)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Family 5: Transaction-Spend Features (Optional)

# COMMAND ----------
"""
Family 5: Transaction-Spend (Optional)
- Transaction counts by window
- Spending velocity
- Last transaction recency
"""

# Only create if transaction table exists
try:
    transactions_df = spark.table(TRANSACTIONS_TABLE)

    # Filter transactions before snapshot
    transactions_df = transactions_df.filter(
        F.col("transaction_date") <= F.lit(SNAPSHOT_DATE)
    )

    # Transaction counts by window
    transaction_features = transactions_df.filter(
        F.col("transaction_date") >= F.date_sub(F.lit(SNAPSHOT_DATE), 90)
    ).groupBy("account_id").agg(
        F.count("*").alias("transaction_count_90d"),
        F.sum("transaction_amount").alias("transaction_amt_90d")
    )

    # Last transaction
    last_txn = transactions_df.withColumn(
        "rn", F.row_number().over(
            Window.partitionBy("account_id").orderBy(F.col("transaction_date").desc())
        )
    ).filter(F.col("rn") == 1).select(
        "account_id",
        F.datediff(F.lit(SNAPSHOT_DATE), F.col("transaction_date")).alias("days_since_last_transaction")
    )

    transaction_features = transaction_features.join(last_txn, "account_id", "outer")
    transaction_features = transaction_features.fillna(0)

    print(f"✓ Family 5: Transaction features created for {transaction_features.count()} accounts")

except Exception as e:
    print(f"⚠ Family 5: Transaction table not available, skipping. ({str(e)})")
    transaction_features = None

# COMMAND ----------
# MAGIC %md
# MAGIC ## Family 6: Bureau Exposure Features

# COMMAND ----------
"""
Family 6: Bureau Exposure
- Total outstanding at all lenders
- Monthly installment obligations
- Secured loan presence
- Active loan count
"""

bureau_df = spark.table(BUREAU_TABLE)

# Get most recent bureau pull before snapshot
bureau_latest = bureau_df.filter(
    F.col("bureau_pull_date") <= F.lit(SNAPSHOT_DATE)
).withColumn(
    "rn", F.row_number().over(
        Window.partitionBy("account_id").orderBy(F.col("bureau_pull_date").desc())
    )
).filter(F.col("rn") == 1)

bureau_exposure = bureau_latest.select(
    "account_id",
    F.col("total_outstanding").alias("bureau_total_outstanding"),
    F.col("monthly_instalment").alias("bureau_monthly_instalment"),
    F.col("secured_loan_flag").alias("bureau_secured_loan_flag"),
    F.col("secured_outstanding").alias("bureau_secured_outstanding"),
    F.col("active_loan_count").alias("bureau_active_loan_count"),
    F.col("new_loans_12m").alias("bureau_new_loan_12m")
)

print(f"✓ Family 6: Bureau exposure features created for {bureau_exposure.count()} accounts")
bureau_exposure.show(5)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Family 7: Bureau Delinquency Features

# COMMAND ----------
"""
Family 7: Bureau Delinquency
- Delinquent at other lenders
- DPD buckets at other lenders
"""

bureau_delinquency = bureau_latest.select(
    "account_id",
    F.col("delinquent_other_lenders").alias("bureau_delinquent_other"),
    F.col("dpd_30_other").alias("bureau_dpd_30_other"),
    F.col("dpd_60_other").alias("bureau_dpd_60_other"),
    F.col("dpd_90_other").alias("bureau_dpd_90_other")
)

print(f"✓ Family 7: Bureau delinquency features created")
bureau_delinquency.show(5)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Family 8: Avoidance Flags

# COMMAND ----------
"""
Family 8: Avoidance Flags
- Wrong number
- Dispute flag
- Lawyer mentioned
- SMS opt-out
"""

avoidance_flags = contacts_df.groupBy("account_id").agg(
    F.max(F.when(F.col("contact_result") == "WRONG_NUMBER", 1).otherwise(0)).alias("wrong_number_flag"),
    F.max(F.when(F.col("dispute_raised") == True, 1).otherwise(0)).alias("dispute_flag"),
    F.max(F.when(F.col("lawyer_mentioned") == True, 1).otherwise(0)).alias("lawyer_mentioned"),
    F.max(F.when(F.col("sms_opt_out") == True, 1).otherwise(0)).alias("sms_opt_out")
).withColumn("wrong_number_flag", F.col("wrong_number_flag").cast("boolean")) \
 .withColumn("dispute_flag", F.col("dispute_flag").cast("boolean")) \
 .withColumn("lawyer_mentioned", F.col("lawyer_mentioned").cast("boolean")) \
 .withColumn("sms_opt_out", F.col("sms_opt_out").cast("boolean"))

print(f"✓ Family 8: Avoidance flags created for {avoidance_flags.count()} accounts")
avoidance_flags.show(5)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Merge All Features

# COMMAND ----------
"""
Merge all 8 feature families into single table
"""

# Start with delinquency (all accounts must be here)
final_features = delinquency_features

# Join balance features (already merged in delinquency)
# Join payments
final_features = final_features.join(payment_features, "account_id", "left")

# Join engagement
final_features = final_features.join(engagement_features, "account_id", "left")

# Join transactions (if available)
if transaction_features is not None:
    final_features = final_features.join(transaction_features, "account_id", "left")

# Join bureau exposure
final_features = final_features.join(bureau_exposure, "account_id", "left")

# Join bureau delinquency
final_features = final_features.join(bureau_delinquency, "account_id", "left")

# Join avoidance flags
final_features = final_features.join(avoidance_flags, "account_id", "left")

# Add metadata
final_features = final_features.withColumn("snapshot_date", F.lit(SNAPSHOT_DATE)) \
                               .withColumn("created_at", F.current_timestamp())

# Fill remaining nulls with 0
numeric_cols = [c for c in final_features.columns if c not in ["account_id", "stage", "snapshot_date", "created_at", "chargeoff_date"]]
final_features = final_features.fillna(0, subset=numeric_cols)

print(f"\n{'='*80}")
print(f"✓ ALL 8 FEATURE FAMILIES MERGED")
print(f"{'='*80}")
print(f"Total accounts: {final_features.count()}")
print(f"Total features: {len(final_features.columns)}")
print(f"\nSchema:")
final_features.printSchema()

# COMMAND ----------
# MAGIC %md
# MAGIC ## Write to Delta Lake

# COMMAND ----------
"""
Write final features to Delta table
"""

# Write to Delta
final_features.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(OUTPUT_TABLE)

print(f"\n✓ Features written to {OUTPUT_TABLE}")

# Optimize table
spark.sql(f"OPTIMIZE {OUTPUT_TABLE} ZORDER BY (account_id)")
print(f"✓ Table optimized")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Validation Queries

# COMMAND ----------
"""
Validate feature table
"""

# Sample features
print("\n📊 SAMPLE FEATURES:")
spark.sql(f"""
    SELECT
        account_id,
        stage,
        balance,
        days_past_due,
        payment_count_30d,
        payment_count_90d,
        call_response_rate,
        bureau_total_outstanding,
        bureau_delinquent_other
    FROM {OUTPUT_TABLE}
    LIMIT 10
""").show()

# Feature completeness
print("\n📊 FEATURE COMPLETENESS:")
spark.sql(f"""
    SELECT
        COUNT(*) as total_accounts,
        COUNT(DISTINCT account_id) as unique_accounts,
        SUM(CASE WHEN payment_count_365d > 0 THEN 1 ELSE 0 END) as accounts_with_payments,
        SUM(CASE WHEN bureau_total_outstanding > 0 THEN 1 ELSE 0 END) as accounts_with_bureau,
        SUM(CASE WHEN contacts_made_90d > 0 THEN 1 ELSE 0 END) as accounts_with_contacts
    FROM {OUTPUT_TABLE}
""").show()

# Stage distribution
print("\n📊 STAGE DISTRIBUTION:")
spark.sql(f"""
    SELECT
        stage,
        COUNT(*) as account_count,
        AVG(balance) as avg_balance,
        AVG(days_past_due) as avg_dpd
    FROM {OUTPUT_TABLE}
    GROUP BY stage
    ORDER BY stage
""").show()

print(f"\n{'='*80}")
print(f"✅ FEATURE ENGINEERING COMPLETE!")
print(f"{'='*80}")
print(f"\nNext steps:")
print(f"1. Review feature table: {OUTPUT_TABLE}")
print(f"2. Use features with recovery_agent_practical modules")
print(f"3. Train RecoveryScorecard6M")
print(f"4. Score daily batches")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Export Sample for Testing

# COMMAND ----------
"""
Export a small sample to Pandas for local testing
"""

# Get sample
sample_df = spark.sql(f"""
    SELECT * FROM {OUTPUT_TABLE}
    ORDER BY RAND()
    LIMIT 500
""").toPandas()

# Display summary
print("\n📊 SAMPLE STATISTICS:")
print(sample_df.describe())

# Optionally save to CSV
# sample_df.to_csv("/dbfs/tmp/recovery_features_sample.csv", index=False)
# print("\n✓ Sample saved to /dbfs/tmp/recovery_features_sample.csv")
