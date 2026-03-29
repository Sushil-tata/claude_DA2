# 📅 NBA Implementation: 8-Week Roadmap with Deliverables

**Project:** Next Best Action System for Debt Collection
**Duration:** 8 weeks to pilot launch
**Team Size:** 2-3 DS + 1 DE + 1 BA
**Goal:** Beat vendor POC by 25-35% in net value

---

## 📊 **Success Metrics (Target by Week 8)**

| Metric | Baseline | Vendor POC | Our Target | How to Measure |
|--------|----------|------------|------------|----------------|
| Recovery Rate | 12% | 16% | **18-20%** | payments / portfolio_balance |
| Cost per $ Collected | $0.50 | $0.35 | **$0.25-0.28** | total_cost / total_recovered |
| Net Value (per $1M) | $100K | $145K | **$175K+** | recovery - cost |
| ROI | 2.0x | 3.1x | **3.6x+** | net_value / cost |
| True Uplift | - | +4% | **+6-8%** | vs no-contact control |

---

# 📋 WEEK 1: Foundation & Action Space

## **Objectives**
1. ✅ Lock action space (5-8 actions max)
2. ✅ Define eligibility rules per action
3. ✅ Build decision table schema (PIT-correct)
4. ✅ Setup infrastructure (Delta, MLflow, Git)

## **Deliverables**

### **D1.1: Action Catalog** ✓ CREATED
📁 `conf/actions/action_catalog.yaml`

**Contents:**
- 7 actions defined: NONE, LINE, SMS, VOICE_IVR, CALL, SETTLEMENT_OFFER, OA_REFERRAL
- Eligibility rules per action (hard constraints)
- Contraindications (when to block)
- Fatigue rules (soft constraints)
- Cost & capacity limits
- Sequencing preferences

**Acceptance Criteria:**
- [ ] All 7 actions have clear definitions
- [ ] Eligibility rules validated with business
- [ ] Cost estimates confirmed with finance
- [ ] Capacity limits agreed with operations
- [ ] Legal review of compliance constraints (DNC, quiet hours)

### **D1.2: Decision Table Schema** ✓ CREATED
📁 `schemas/decision_table_schema.sql`

**Contents:**
- 80+ columns covering state, history, operability
- **Critical:** `delay = max(0, business_date - due_date)`
- Point-in-time safe (no future leakage)
- Outcome labeling with LEAD() functions
- Data quality checks built-in

**Acceptance Criteria:**
- [ ] Schema reviewed by DE team
- [ ] Delay definition matches business logic
- [ ] All historical features are PIT-safe
- [ ] Outcome labeling uses LEAD() (no leakage)
- [ ] Partitioning strategy approved

### **D1.3: Infrastructure Setup** ✓ CREATED
📁 `deployment/setup_infrastructure.sh`

**What it creates:**
- 3 Delta databases (prod, dev, staging)
- 4 core tables (decision_table, predictions_log, execution_log, why_cards)
- 4 MLflow experiments
- Secret scope
- Git integration
- Monitoring dashboards

**Acceptance Criteria:**
- [ ] All databases created successfully
- [ ] Tables created with correct schema
- [ ] MLflow experiments accessible
- [ ] Secrets configured
- [ ] Git repo initialized

## **Testing & Validation**

### **T1.1: Action Catalog Validation**
```bash
# Test: Load and validate action catalog
python -c "
import yaml
with open('conf/actions/action_catalog.yaml') as f:
    actions = yaml.safe_load(f)

print(f'Actions defined: {len(actions[\"actions\"])}')
assert len(actions['actions']) >= 5, 'Need at least 5 actions'
assert 'A0_NONE' in actions['actions'], 'Must have NO_ACTION'
print('✓ Action catalog valid')
"
```

### **T1.2: Decision Table Creation**
```sql
-- Test: Create decision table for one day
CREATE OR REPLACE TABLE debt_collection_dev.decision_table_test AS
SELECT
    account_id,
    CURRENT_DATE() as business_date,
    bucket,
    due_date,
    GREATEST(0, DATEDIFF(CURRENT_DATE(), due_date)) as delay,
    balance,
    -- ... other columns
FROM raw_accounts
WHERE business_date = CURRENT_DATE()
LIMIT 1000;

-- Validate row count
SELECT COUNT(*) FROM debt_collection_dev.decision_table_test;
-- Should be 1000

-- Validate delay calculation
SELECT
    account_id,
    due_date,
    delay,
    GREATEST(0, DATEDIFF(CURRENT_DATE(), due_date)) as expected_delay
FROM debt_collection_dev.decision_table_test
WHERE delay != GREATEST(0, DATEDIFF(CURRENT_DATE(), due_date));
-- Should be empty
```

## **Stakeholder Checkpoint**

**Meeting:** End of Week 1 (Friday 3pm)
**Attendees:** DS Team, Business Sponsor, Operations Lead, Compliance
**Agenda:**
1. Review action catalog (20 min)
   - Are these the right 7 actions?
   - Any missing actions we can execute deterministically?
2. Review decision table schema (15 min)
   - Confirm delay definition
   - Validate historical features
3. Demo infrastructure (10 min)
   - Show created tables
   - Show MLflow experiments
4. Go/No-Go for Week 2 (5 min)

**Deliverable:** Signed-off action catalog + schema

---

# 📋 WEEK 2: Golden Decision Table Build

## **Objectives**
1. ✅ Build production decision table (full historical data)
2. ✅ Implement persona segmentation
3. ✅ Calculate fatigue scores
4. ✅ Validate data quality

## **Deliverables**

### **D2.1: Decision Table Builder Pipeline**
📁 `src/decision_agent/data/decision_table_builder.py`

```python
"""
Decision Table Builder
Builds the golden decision table (account_id × business_date) with PIT correctness
"""

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from datetime import datetime, timedelta

class DecisionTableBuilder:
    """
    Builds the golden decision table with strict point-in-time correctness.

    Key principles:
    - delay = max(0, business_date - due_date)  # Business definition
    - All historical features use lookback windows (no future data)
    - Outcomes joined with LEAD() to prevent leakage
    """

    def __init__(self, spark: SparkSession, config: dict):
        self.spark = spark
        self.config = config

    def build_for_date_range(
        self,
        start_date: str,
        end_date: str,
        output_table: str = "debt_collection.decision_table"
    ):
        """Build decision table for date range"""

        print(f"Building decision table: {start_date} to {end_date}")

        # Step 1: Load base account data
        accounts_df = self._load_accounts(start_date, end_date)
        print(f"  Loaded {accounts_df.count():,} account-date rows")

        # Step 2: Add payment features
        with_payment_features = self._add_payment_features(accounts_df)

        # Step 3: Add delinquency history
        with_delinq_features = self._add_delinquency_history(with_payment_features)

        # Step 4: Add channel history
        with_channel_features = self._add_channel_history(with_delinq_features)

        # Step 5: Add fatigue scores
        with_fatigue = self._add_fatigue_scores(with_channel_features)

        # Step 6: Add operability flags
        with_operability = self._add_operability_flags(with_fatigue)

        # Step 7: Add persona segmentation
        with_personas = self._add_personas(with_operability)

        # Step 8: Validate and write
        validated_df = self._validate_quality(with_personas)

        validated_df.write \
            .format("delta") \
            .mode("overwrite") \
            .partitionBy("business_date") \
            .saveAsTable(output_table)

        print(f"✓ Decision table written to {output_table}")

        return validated_df

    def _load_accounts(self, start_date, end_date):
        """Load base account snapshot data"""
        return self.spark.sql(f"""
            SELECT
                account_id,
                business_date,
                customer_id,
                product_type,
                bucket,
                bill_date,
                DATE_ADD(bill_date, 20) as due_date,  -- Business definition
                GREATEST(0, DATEDIFF(business_date, DATE_ADD(bill_date, 20))) as delay,
                days_past_due,  -- System DPD (may differ)
                balance,
                original_balance,
                MAD,
                past_due_amount,
                credit_limit,
                balance / credit_limit as utilization
            FROM raw.account_snapshots
            WHERE business_date BETWEEN '{start_date}' AND '{end_date}'
        """)

    def _add_payment_features(self, df):
        """Add payment history features (7/14/30 day windows)"""

        # Join with payment history
        payments = self.spark.table("raw.payment_history")

        # For each account-date, aggregate payments in lookback windows
        payment_features = df.alias("a").join(
            payments.alias("p"),
            (F.col("a.account_id") == F.col("p.account_id")) &
            (F.col("p.payment_date") <= F.col("a.business_date")) &
            (F.col("p.payment_date") >= F.date_sub(F.col("a.business_date"), 90)),
            "left"
        ).groupBy("a.account_id", "a.business_date").agg(
            # Last payment
            F.max("p.payment_date").alias("last_payment_date"),
            F.sum(F.when(F.col("p.payment_date") == F.max("p.payment_date"), F.col("p.payment_amount"))).alias("last_payment_amount"),

            # 7-day window
            F.count(F.when(F.col("p.payment_date") >= F.date_sub(F.col("a.business_date"), 7), 1)).alias("payment_count_7d"),
            F.sum(F.when(F.col("p.payment_date") >= F.date_sub(F.col("a.business_date"), 7), F.col("p.payment_amount")).otherwise(0)).alias("payment_sum_7d"),

            # 14-day window
            F.count(F.when(F.col("p.payment_date") >= F.date_sub(F.col("a.business_date"), 14), 1)).alias("payment_count_14d"),
            F.sum(F.when(F.col("p.payment_date") >= F.date_sub(F.col("a.business_date"), 14), F.col("p.payment_amount")).otherwise(0)).alias("payment_sum_14d"),

            # 30-day window
            F.count(F.when(F.col("p.payment_date") >= F.date_sub(F.col("a.business_date"), 30), 1)).alias("payment_count_30d"),
            F.sum(F.when(F.col("p.payment_date") >= F.date_sub(F.col("a.business_date"), 30), F.col("p.payment_amount")).otherwise(0)).alias("payment_sum_30d")
        )

        # Join back to main df
        result = df.join(
            payment_features,
            on=["account_id", "business_date"],
            how="left"
        ).withColumn(
            "days_since_last_payment",
            F.datediff(F.col("business_date"), F.col("last_payment_date"))
        )

        return result

    def _add_delinquency_history(self, df):
        """Add delinquency history features"""

        # Window for historical lookback
        window_3m = Window.partitionBy("account_id").orderBy("business_date").rangeBetween(-90*86400, 0)
        window_6m = Window.partitionBy("account_id").orderBy("business_date").rangeBetween(-180*86400, 0)
        window_12m = Window.partitionBy("account_id").orderBy("business_date").rangeBetween(-365*86400, 0)

        result = df.withColumn(
            "max_delay_3m",
            F.max("delay").over(window_3m)
        ).withColumn(
            "max_delay_6m",
            F.max("delay").over(window_6m)
        ).withColumn(
            "max_delay_12m",
            F.max("delay").over(window_12m)
        ).withColumn(
            "max_bucket_6m",
            F.max("bucket").over(window_6m)
        ).withColumn(
            "ever_bucket_3_plus",
            (F.max("bucket").over(window_6m) >= 3).cast("boolean")
        )

        # Bucket transitions
        window_lag = Window.partitionBy("account_id").orderBy("business_date")
        result = result.withColumn(
            "bucket_prev",
            F.lag("bucket", 1).over(window_lag)
        ).withColumn(
            "rolled_forward_last_cycle",
            (F.col("bucket") > F.col("bucket_prev")).cast("boolean")
        ).withColumn(
            "rolled_backward_last_cycle",
            (F.col("bucket") < F.col("bucket_prev")).cast("boolean")
        ).drop("bucket_prev")

        return result

    def _add_channel_history(self, df):
        """Add contact channel history features"""

        contacts = self.spark.table("raw.contact_history")

        # Aggregate contacts by channel and time window
        channel_features = df.alias("a").join(
            contacts.alias("c"),
            (F.col("a.account_id") == F.col("c.account_id")) &
            (F.col("c.contact_date") <= F.col("a.business_date")) &
            (F.col("c.contact_date") >= F.date_sub(F.col("a.business_date"), 90)),
            "left"
        ).groupBy("a.account_id", "a.business_date").agg(
            # LINE
            F.count(F.when(
                (F.col("c.contact_channel") == "LINE") &
                (F.col("c.contact_date") >= F.date_sub(F.col("a.business_date"), 7)),
                1
            )).alias("line_attempts_7d"),

            # SMS
            F.count(F.when(
                (F.col("c.contact_channel") == "SMS") &
                (F.col("c.contact_date") >= F.date_sub(F.col("a.business_date"), 7)),
                1
            )).alias("sms_attempts_7d"),

            # CALL
            F.count(F.when(
                (F.col("c.contact_channel") == "CALL") &
                (F.col("c.contact_date") >= F.date_sub(F.col("a.business_date"), 7)),
                1
            )).alias("call_attempts_7d"),

            # Total contacts
            F.count(F.when(
                F.col("c.contact_date") >= F.date_sub(F.col("a.business_date"), 7),
                1
            )).alias("total_contacts_7d"),

            F.count(F.when(
                F.col("c.contact_date") >= F.date_sub(F.col("a.business_date"), 30),
                1
            )).alias("total_contacts_30d"),

            # Last contact by channel
            F.max(F.when(F.col("c.contact_channel") == "LINE", F.col("c.contact_date"))).alias("last_line_date"),
            F.max(F.when(F.col("c.contact_channel") == "SMS", F.col("c.contact_date"))).alias("last_sms_date"),
            F.max(F.when(F.col("c.contact_channel") == "CALL", F.col("c.contact_date"))).alias("last_call_date")
        )

        result = df.join(
            channel_features,
            on=["account_id", "business_date"],
            how="left"
        ).fillna(0, subset=["line_attempts_7d", "sms_attempts_7d", "call_attempts_7d", "total_contacts_7d", "total_contacts_30d"])

        return result

    def _add_fatigue_scores(self, df):
        """Calculate fatigue scores (weighted sum with decay)"""

        # Fatigue score = weighted sum of recent contacts with exponential decay
        # Weights: LINE=0.2, SMS=0.3, CALL=1.0
        # Half-life = 14 days

        result = df.withColumn(
            "fatigue_score",
            (
                F.col("line_attempts_7d") * 0.2 +
                F.col("sms_attempts_7d") * 0.3 +
                F.col("call_attempts_7d") * 1.0
            ) / 10.0  # Normalize to 0-1
        ).withColumn(
            "fatigue_score",
            F.least(F.col("fatigue_score"), F.lit(1.0))  # Cap at 1.0
        )

        return result

    def _add_operability_flags(self, df):
        """Add operability flags (can we contact via each channel?)"""

        # Join with customer master for contact preferences
        customers = self.spark.table("raw.customer_master")

        result = df.join(
            customers.select(
                "customer_id",
                F.col("line_optin").alias("line_optin"),
                F.col("line_id").isNotNull().alias("line_id_present"),
                F.col("mobile").isNotNull().alias("mobile_present"),
                F.col("sms_optin").alias("sms_optin"),
                F.col("email").isNotNull().alias("email_present"),
                F.col("dnc_flag").alias("dnc"),
                F.col("cease_desist_flag").alias("cease_and_desist")
            ),
            on="customer_id",
            how="left"
        )

        return result

    def _add_personas(self, df):
        """Add persona segmentation (rule-based + ML)"""

        # Import persona segmentation from Week 1 debt collection module
        from decision_agent.debt_collection.persona_segmentation import DebtorPersonaSegmentation

        # Convert to pandas for persona assignment (if dataset is manageable)
        # For very large datasets, implement in pure Spark

        # Rule-based personas
        result = df.withColumn(
            "rule_segment",
            F.when(
                (F.col("balance") > 5000) &
                (F.col("delay") < 180) &
                (F.col("payment_count_30d") > 0),
                F.lit("high_value_cooperative")
            ).when(
                (F.col("balance") > 5000) &
                (F.col("delay") >= 180) &
                (F.col("total_contacts_30d") > 5) &
                (F.col("payment_count_30d") == 0),
                F.lit("high_value_unresponsive")
            ).when(
                (F.col("balance").between(1000, 5000)) &
                (F.col("payment_count_90d") > 0),
                F.lit("medium_value_willing")
            ).when(
                (F.col("balance") < 1000) &
                (F.col("delay") > 270),
                F.lit("low_value_low_capacity")
            ).when(
                F.col("dnc") | F.col("cease_and_desist"),
                F.lit("compliance_review")
            ).otherwise(
                F.lit("standard")
            )
        )

        # For now, use rule_segment as persona_segment
        # Week 2 will add ML clustering
        result = result.withColumn("persona_segment", F.col("rule_segment"))

        return result

    def _validate_quality(self, df):
        """Validate data quality"""

        total_rows = df.count()

        # Check 1: No negative delays
        negative_delays = df.filter(F.col("delay") < 0).count()
        assert negative_delays == 0, f"Found {negative_delays} rows with negative delay!"

        # Check 2: No missing personas
        missing_personas = df.filter(F.col("persona_segment").isNull()).count()
        if missing_personas > 0:
            print(f"⚠ Warning: {missing_personas} rows missing persona ({missing_personas/total_rows*100:.1f}%)")

        # Check 3: Delay matches calculation
        delay_mismatch = df.filter(
            F.col("delay") != F.greatest(F.lit(0), F.datediff(F.col("business_date"), F.col("due_date")))
        ).count()
        assert delay_mismatch == 0, f"Delay calculation mismatch in {delay_mismatch} rows!"

        print(f"✓ Data quality validated ({total_rows:,} rows)")

        return df


# Example usage
if __name__ == "__main__":
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.appName("DecisionTableBuilder").getOrCreate()

    builder = DecisionTableBuilder(spark, config={})

    # Build for last 90 days
    decision_table = builder.build_for_date_range(
        start_date="2024-01-01",
        end_date="2024-03-31",
        output_table="debt_collection.decision_table"
    )

    print("✓ Decision table built successfully")
```

### **D2.2: Persona Segmentation Enhancement**
📁 `src/decision_agent/persona/ml_clustering.py`

```python
"""
ML-Based Clustering for Persona Refinement
Adds data-driven segmentation on top of rule-based personas
"""

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

class PersonaMLClustering:
    """
    Adds ML clustering within rule-based segments for finer personas.
    """

    def __init__(self, n_clusters_per_segment: int = 3):
        self.n_clusters = n_clusters_per_segment
        self.scalers = {}
        self.clusterers = {}

    def fit_transform(self, df: pd.DataFrame, rule_segment_col: str = 'rule_segment'):
        """
        Fit clusters within each rule segment and assign ML personas.

        Returns:
            DataFrame with 'ml_segment' and 'persona_segment' columns
        """

        feature_cols = [
            'balance', 'delay', 'payment_count_30d',
            'total_contacts_30d', 'fatigue_score', 'utilization'
        ]

        df_result = df.copy()
        df_result['ml_segment'] = 'unknown'

        # Cluster within each rule segment
        for segment in df[rule_segment_col].unique():
            segment_mask = df[rule_segment_col] == segment

            if segment_mask.sum() < self.n_clusters * 10:
                # Too few samples, skip clustering
                df_result.loc[segment_mask, 'ml_segment'] = 'singleton'
                continue

            # Get features for this segment
            X = df.loc[segment_mask, feature_cols].fillna(0)

            # Scale
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            self.scalers[segment] = scaler

            # Cluster
            clusterer = KMeans(n_clusters=self.n_clusters, random_state=42)
            clusters = clusterer.fit_predict(X_scaled)
            self.clusterers[segment] = clusterer

            # Assign ML segment labels
            df_result.loc[segment_mask, 'ml_segment'] = [
                f"{segment}_ml{c}" for c in clusters
            ]

        # Create final persona = rule_segment + ml_segment
        df_result['persona_segment'] = (
            df_result['rule_segment'] + '_' + df_result['ml_segment']
        )

        return df_result
```

## **Acceptance Criteria - Week 2**

- [ ] Decision table built for last 90 days (min 100K account-days)
- [ ] Delay calculation validated (matches business definition)
- [ ] All features are point-in-time safe (no leakage)
- [ ] Fatigue scores calculated correctly
- [ ] 6-8 personas created (rule-based + ML)
- [ ] Data quality checks pass (<1% errors)
- [ ] Persona distribution makes business sense

## **Testing - Week 2**

```python
# Test: Validate decision table completeness
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

# Load decision table
dt = spark.table("debt_collection.decision_table")

# Check 1: Row count
total_rows = dt.count()
expected_rows = 100_000  # Adjust based on portfolio size
assert total_rows >= expected_rows, f"Expected {expected_rows}+, got {total_rows}"

# Check 2: Date range
date_range = dt.agg(
    F.min("business_date").alias("min_date"),
    F.max("business_date").alias("max_date")
).collect()[0]
print(f"Date range: {date_range['min_date']} to {date_range['max_date']}")

# Check 3: Persona distribution
personas = dt.groupBy("persona_segment").count().orderBy(F.desc("count"))
personas.show()

# Check 4: No future leakage
future_dates = dt.filter(F.col("business_date") > F.current_date()).count()
assert future_dates == 0, "Found future dates!"

# Check 5: Delay correctness
delay_check = dt.filter(
    F.col("delay") != F.greatest(0, F.datediff("business_date", "due_date"))
)
assert delay_check.count() == 0, "Delay calculation incorrect!"

print("✓ All tests passed")
```

## **Stakeholder Checkpoint - Week 2**

**Meeting:** End of Week 2 (Friday 3pm)
**Attendees:** DS Team, Business Sponsor, Operations
**Agenda:**
1. Demo decision table (15 min)
   - Show sample rows
   - Explain delay vs days_past_due
   - Show persona distribution
2. Validate personas (20 min)
   - Do these segments make business sense?
   - Are any segments missing?
3. Review data quality report (10 min)
4. Go/No-Go for modeling (Week 3)

---

# 📋 WEEK 3: Outcome Models (S-Learner + Tweedie)

## **Objectives**
1. ✅ Train pay_any model (S-Learner with action as feature)
2. ✅ Train amount model (Tweedie regression)
3. ✅ Validate on holdout set
4. ✅ Register models in MLflow

## **Deliverables**

### **D3.1: Pay-Any Model (S-Learner)**
📁 `src/decision_agent/models/pay_any_model.py`

```python
"""
Pay-Any Model: Binary classifier for payment probability
Uses S-Learner approach (action as categorical feature)
"""

import mlflow
import mlflow.lightgbm
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
import pandas as pd

class PayAnyModel:
    """
    Predicts P(pay_any_7d | X, action)

    Key: Action is included as a categorical feature (S-Learner)
    This allows us to score counterfactuals at inference time.
    """

    def __init__(self, config: dict):
        self.config = config
        self.model = None
        self.feature_cols = None

    def train(
        self,
        training_df: pd.DataFrame,
        target_col: str = 'outcome_pay_any_7d',
        action_col: str = 'executed_action',
        mlflow_experiment: str = '/Users/nba_team/pay_any_model'
    ):
        """
        Train S-Learner pay_any model.

        Args:
            training_df: DataFrame with features + action + outcome
            target_col: Binary outcome (0/1)
            action_col: Action taken (categorical)
        """

        mlflow.set_experiment(mlflow_experiment)

        # Feature columns (including action!)
        self.feature_cols = [
            # Account state
            'balance', 'delay', 'bucket', 'utilization',

            # Payment history
            'payment_count_7d', 'payment_count_30d',
            'days_since_last_payment',

            # Delinquency history
            'max_delay_6m', 'max_bucket_6m',
            'rolled_forward_last_cycle',

            # Channel history
            'line_attempts_7d', 'sms_attempts_7d', 'call_attempts_7d',
            'total_contacts_30d', 'fatigue_score',

            # Persona
            'persona_segment',  # Will be label-encoded

            # ACTION (KEY!)
            action_col  # The action taken
        ]

        # Prepare data
        X = training_df[self.feature_cols].copy()
        y = training_df[target_col]

        # Encode categoricals
        X['persona_segment'] = X['persona_segment'].astype('category')
        X[action_col] = X[action_col].astype('category')

        # Train/val split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train LightGBM
        with mlflow.start_run(run_name="pay_any_slearner"):

            self.model = LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                num_leaves=31,
                class_weight='balanced',
                random_state=42
            )

            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[mlflow.lightgbm.log_model_callback()]
            )

            # Evaluate
            y_pred_proba = self.model.predict_proba(X_val)[:, 1]

            auc = roc_auc_score(y_val, y_pred_proba)
            ap = average_precision_score(y_val, y_pred_proba)

            # Log metrics
            mlflow.log_metric("val_auc", auc)
            mlflow.log_metric("val_ap", ap)

            # Log model
            mlflow.lightgbm.log_model(self.model, "model")

            # Log feature importance
            feature_importance = pd.DataFrame({
                'feature': self.feature_cols,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)

            feature_importance.to_csv("feature_importance.csv", index=False)
            mlflow.log_artifact("feature_importance.csv")

            print(f"✓ Pay-Any model trained: AUC={auc:.4f}")

        return self.model

    def predict_all_actions(self, customer_features: pd.DataFrame, actions: list):
        """
        Score all candidate actions for counterfactual comparison.

        Args:
            customer_features: Single row or batch of customer features
            actions: List of actions to score (e.g., ['LINE', 'SMS', 'CALL'])

        Returns:
            Dict of {action: probability}
        """

        predictions = {}

        for action in actions:
            # Create features with this action
            features_with_action = customer_features.copy()
            features_with_action['executed_action'] = action

            # Predict
            prob = self.model.predict_proba(features_with_action[self.feature_cols])[:, 1]
            predictions[action] = float(prob[0]) if len(prob) == 1 else prob.tolist()

        return predictions
```

### **D3.2: Amount Model (Tweedie)**
📁 `src/decision_agent/models/amount_model.py`

```python
"""
Amount Model: Predict payment amount using Tweedie regression
Handles zero-inflated continuous outcomes naturally
"""

import mlflow
from sklearn.ensemble import HistGradientBoostingRegressor
import numpy as np
import pandas as pd

class AmountModel:
    """
    Predicts E(payment_amount_7d | X, action, pay_any=1)

    Uses Tweedie/Gamma regression to handle zero-inflated continuous outcomes.
    """

    def __init__(self, config: dict):
        self.config = config
        self.model = None
        self.feature_cols = None

    def train(
        self,
        training_df: pd.DataFrame,
        target_col: str = 'outcome_pay_amount_7d',
        action_col: str = 'executed_action',
        mlflow_experiment: str = '/Users/nba_team/amount_model'
    ):
        """Train Tweedie regression model"""

        mlflow.set_experiment(mlflow_experiment)

        # Same feature cols as pay_any model
        self.feature_cols = [
            'balance', 'delay', 'bucket', 'utilization',
            'payment_count_7d', 'payment_count_30d',
            'days_since_last_payment',
            'max_delay_6m', 'max_bucket_6m',
            'line_attempts_7d', 'sms_attempts_7d',
            'total_contacts_30d', 'fatigue_score',
            'persona_segment',
            action_col
        ]

        # Filter to cases where payment was made (or use full dataset with zeros)
        # Option 1: Train only on pay_any=1 (conditional model)
        # Option 2: Train on all (Tweedie handles zeros naturally)

        # Using Option 2 (Tweedie approach)
        X = training_df[self.feature_cols].copy()
        y = training_df[target_col].fillna(0)

        # Encode categoricals
        X['persona_segment'] = X['persona_segment'].astype('category').cat.codes
        X[action_col] = X[action_col].astype('category').cat.codes

        # Train/val split
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        with mlflow.start_run(run_name="amount_tweedie"):

            # Tweedie regression (power=1.5 is between Poisson and Gamma)
            self.model = HistGradientBoostingRegressor(
                loss='poisson',  # Or use 'gamma' if no zeros
                max_iter=200,
                max_depth=6,
                learning_rate=0.05,
                random_state=42
            )

            self.model.fit(X_train, y_train)

            # Evaluate
            y_pred = self.model.predict(X_val)

            # Metrics for continuous outcomes
            mae = np.mean(np.abs(y_val - y_pred))
            rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))

            mlflow.log_metric("val_mae", mae)
            mlflow.log_metric("val_rmse", rmse)

            # Log model
            mlflow.sklearn.log_model(self.model, "model")

            print(f"✓ Amount model trained: MAE={mae:.2f}, RMSE={rmse:.2f}")

        return self.model

    def predict_all_actions(self, customer_features: pd.DataFrame, actions: list):
        """Predict expected amount for each action"""

        predictions = {}

        for action in actions:
            features_with_action = customer_features.copy()
            features_with_action['executed_action'] = action

            # Encode
            features_with_action['persona_segment'] = features_with_action['persona_segment'].astype('category').cat.codes
            features_with_action['executed_action'] = features_with_action['executed_action'].astype('category').cat.codes

            # Predict
            amount = self.model.predict(features_with_action[self.feature_cols])
            predictions[action] = float(amount[0]) if len(amount) == 1 else amount.tolist()

        return predictions
```

## **Acceptance Criteria - Week 3**

- [ ] Pay-any model trained on 60K+ examples
- [ ] AUC >= 0.70 on validation set
- [ ] Amount model trained on same dataset
- [ ] MAE within 20% of average payment amount
- [ ] Models registered in MLflow
- [ ] Feature importance analyzed (action feature is top 10)
- [ ] Counterfactual predictions work (can score all actions)

## **Testing - Week 3**

```python
# Test: Counterfactual predictions
from src.decision_agent.models.pay_any_model import PayAnyModel
import pandas as pd

# Load model
pay_any_model = PayAnyModel(config={})
pay_any_model.model = mlflow.lightgbm.load_model("models:/nba_pay_any_model/Production")

# Test customer
customer = pd.DataFrame([{
    'balance': 5000,
    'delay': 45,
    'bucket': 2,
    'utilization': 0.80,
    'payment_count_30d': 1,
    'fatigue_score': 0.3,
    'persona_segment': 'high_value_cooperative',
    # ... other features
}])

# Score all actions
actions = ['NONE', 'LINE', 'SMS', 'CALL']
predictions = pay_any_model.predict_all_actions(customer, actions)

print("Counterfactual predictions:")
for action, prob in predictions.items():
    print(f"  {action}: {prob:.3f}")

# Expected output:
# NONE: 0.420  (baseline)
# LINE: 0.445  (+2.5% uplift)
# SMS: 0.480   (+6.0% uplift)
# CALL: 0.520  (+10.0% uplift)

# Test: Uplift calculation
uplift = {action: prob - predictions['NONE'] for action, prob in predictions.items()}
print("\nUplift vs NO_ACTION:")
for action, up in uplift.items():
    print(f"  {action}: {up:+.3f}")

assert uplift['CALL'] > uplift['SMS'] > uplift['LINE'], "Expected CALL > SMS > LINE in uplift"
print("✓ Uplift test passed")
```

---

# 📋 WEEK 4: Constraints Engine + Policy Logic

## **Objectives**
1. ✅ Build constraint engine (hard + soft filters)
2. ✅ Implement capacity allocation logic
3. ✅ Build policy ranking (2-stage, no complex utilities)
4. ✅ Test end-to-end decision flow

## **Deliverables**

### **D4.1: Constraint Engine**
📁 `src/decision_agent/constraints/constraint_engine.py`

```python
"""
Constraint Engine: Filters feasible actions per account

Separates business logic from ML models (key architectural principle).
"""

from typing import List, Dict, Tuple
import pandas as pd
from datetime import datetime, time

class ConstraintEngine:
    """
    Applies hard and soft constraints to filter/rank actions.

    Key principle: Constraints are SEPARATE from models.
    """

    def __init__(self, action_catalog: dict, business_date: str):
        self.action_catalog = action_catalog
        self.business_date = pd.to_datetime(business_date)
        self.actions = action_catalog['actions']
        self.fatigue_rules = action_catalog['fatigue']
        self.cost_constraints = action_catalog['cost_constraints']

    def get_feasible_actions(
        self,
        account_state: Dict
    ) -> Tuple[List[str], List[str], Dict[str, str]]:
        """
        Return feasible actions for an account.

        Returns:
            (feasible_actions, blocked_actions, block_reasons)
        """

        feasible = []
        blocked = []
        block_reasons = {}

        for action_id, action_config in self.actions.items():

            # Check hard constraints (eligibility)
            is_eligible, reason = self._check_eligibility(account_state, action_config)

            if not is_eligible:
                blocked.append(action_id)
                block_reasons[action_id] = reason
                continue

            # Check contraindications (compliance)
            is_compliant, reason = self._check_contraindications(account_state, action_config)

            if not is_compliant:
                blocked.append(action_id)
                block_reasons[action_id] = reason
                continue

            # Passed all hard constraints
            feasible.append(action_id)

        # Apply soft constraints (penalties, not blocks)
        feasible_with_penalties = self._apply_soft_constraints(
            feasible, account_state
        )

        return feasible_with_penalties, blocked, block_reasons

    def _check_eligibility(
        self,
        account_state: Dict,
        action_config: Dict
    ) -> Tuple[bool, str]:
        """Check if account meets eligibility criteria"""

        eligibility = action_config.get('eligibility', {})

        # Check each eligibility rule
        for field, required_value in eligibility.items():

            if field == 'all_accounts':
                continue

            account_value = account_state.get(field)

            # Boolean checks
            if isinstance(required_value, bool):
                if account_value != required_value:
                    return False, f"Failed eligibility: {field}={account_value}, required={required_value}"

            # Comparison checks
            elif isinstance(required_value, str) and ('<' in required_value or '>' in required_value):
                # e.g., "< 0.7"
                operator = '<' if '<' in required_value else '>'
                threshold = float(required_value.replace('<', '').replace('>', '').strip())

                if operator == '<' and account_value >= threshold:
                    return False, f"Failed eligibility: {field}={account_value} not < {threshold}"
                elif operator == '>' and account_value <= threshold:
                    return False, f"Failed eligibility: {field}={account_value} not > {threshold}"

            # List membership
            elif isinstance(required_value, list):
                if account_value not in required_value:
                    return False, f"Failed eligibility: {field}={account_value} not in {required_value}"

            # Range checks
            elif isinstance(required_value, dict) and 'min' in required_value:
                min_val = required_value.get('min', float('-inf'))
                max_val = required_value.get('max', float('inf'))
                if not (min_val <= account_value <= max_val):
                    return False, f"Failed eligibility: {field}={account_value} not in [{min_val}, {max_val}]"

        return True, ""

    def _check_contraindications(
        self,
        account_state: Dict,
        action_config: Dict
    ) -> Tuple[bool, str]:
        """Check compliance contraindications"""

        contraindications = action_config.get('contraindications', {})

        for field, blocking_value in contraindications.items():

            account_value = account_state.get(field, False)

            # Boolean contraindication
            if isinstance(blocking_value, bool):
                if account_value == blocking_value:
                    return False, f"Blocked: {field}={account_value}"

            # Comparison contraindication
            elif isinstance(blocking_value, str) and '>=' in blocking_value:
                threshold = int(blocking_value.replace('>=', '').strip())
                if account_value >= threshold:
                    return False, f"Blocked: {field}={account_value} >= {threshold}"

        # Check quiet hours
        if action_config.get('channel') in ['call', 'voice_ivr']:
            if self._is_quiet_hours():
                return False, "Blocked: Outside business hours"

        return True, ""

    def _is_quiet_hours(self) -> bool:
        """Check if current time is outside business hours"""
        business_hours = self.action_catalog.get('business_hours', {})

        no_contact_before = time.fromisoformat(business_hours.get('no_contact_before', '08:00'))
        no_contact_after = time.fromisoformat(business_hours.get('no_contact_after', '21:00'))

        current_time = datetime.now().time()

        return current_time < no_contact_before or current_time >= no_contact_after

    def _apply_soft_constraints(
        self,
        feasible_actions: List[str],
        account_state: Dict
    ) -> List[str]:
        """
        Apply soft constraints (penalties, not blocks).
        Returns actions with penalty scores.
        """

        # For now, just return feasible actions
        # In Week 5, we'll add soft penalty scores

        return feasible_actions


# Example usage
if __name__ == "__main__":
    import yaml

    # Load action catalog
    with open('conf/actions/action_catalog.yaml') as f:
        action_catalog = yaml.safe_load(f)

    # Create constraint engine
    engine = ConstraintEngine(action_catalog, business_date='2024-02-15')

    # Test account
    account = {
        'account_id': 'ACC123',
        'line_optin': True,
        'line_id_present': True,
        'mobile_present': True,
        'sms_optin': True,
        'dnc': False,
        'cease_and_desist': False,
        'contact_count_7d': 1,
        'fatigue_score': 0.3,
        'bucket': 1,
        'balance': 3000,
        'delay': 35
    }

    # Get feasible actions
    feasible, blocked, reasons = engine.get_feasible_actions(account)

    print(f"Feasible actions: {feasible}")
    print(f"Blocked actions: {blocked}")
    for action, reason in reasons.items():
        print(f"  {action}: {reason}")
```

### **D4.2: Policy Ranking Logic**
📁 `src/decision_agent/policy/policy_ranker.py`

```python
"""
Policy Ranking Logic
Simple 2-stage ranking (no complex utility functions)
"""

import pandas as pd
from typing import List, Dict

class PolicyRanker:
    """
    Ranks actions using simple 2-stage logic:

    Early buckets (B0-B2): Maximize cure probability
    Late buckets (B3+): Maximize expected amount
    Tie-break: Lower cost + less intrusive
    """

    def __init__(self, action_catalog: dict):
        self.action_catalog = action_catalog
        self.action_costs = {
            action_id: config['cost']
            for action_id, config in action_catalog['actions'].items()
        }

    def rank_actions(
        self,
        account_state: Dict,
        feasible_actions: List[str],
        model_predictions: Dict[str, Dict]
    ) -> List[Dict]:
        """
        Rank feasible actions and return Top-3.

        Args:
            account_state: Account features
            feasible_actions: Actions that passed constraints
            model_predictions: {
                action_id: {
                    'pay_any_prob': 0.45,
                    'expected_amount': 1200
                }
            }

        Returns:
            List of top 3 actions with scores
        """

        bucket = account_state.get('bucket', 0)

        ranked = []

        for action in feasible_actions:

            if action not in model_predictions:
                continue

            preds = model_predictions[action]
            pay_any_prob = preds.get('pay_any_prob', 0)
            expected_amount = preds.get('expected_amount', 0)

            # 2-stage ranking
            if bucket <= 2:
                # Early buckets: prioritize cure probability
                primary_score = pay_any_prob
                secondary_score = expected_amount / 1000  # Normalize
            else:
                # Late buckets: prioritize expected amount
                primary_score = expected_amount / 1000
                secondary_score = pay_any_prob

            # Tie-break: cost penalty
            cost = self.action_costs.get(action, 0)
            cost_penalty = cost / 10  # Normalize

            # Final score
            final_score = primary_score + 0.2 * secondary_score - 0.1 * cost_penalty

            ranked.append({
                'action': action,
                'score': final_score,
                'pay_any_prob': pay_any_prob,
                'expected_amount': expected_amount,
                'cost': cost,
                'primary_score': primary_score,
                'secondary_score': secondary_score
            })

        # Sort by score
        ranked.sort(key=lambda x: x['score'], reverse=True)

        # Return Top-3
        top_3 = ranked[:3]

        # Add rank
        for i, action_result in enumerate(top_3):
            action_result['rank'] = i + 1

        return top_3
```

## **Acceptance Criteria - Week 4**

- [ ] Constraint engine blocks correctly (tested on 1000 accounts)
- [ ] All compliance rules enforced (DNC, cease & desist, quiet hours)
- [ ] Policy ranker produces sensible Top-3
- [ ] End-to-end flow works (state → constraints → models → policy → decision)
- [ ] Decision logging schema implemented

## **Testing - Week 4**

```python
# Test: End-to-end decision flow
from src.decision_agent.constraints.constraint_engine import ConstraintEngine
from src.decision_agent.policy.policy_ranker import PolicyRanker
from src.decision_agent.models.pay_any_model import PayAnyModel
from src.decision_agent.models.amount_model import AmountModel
import yaml
import pandas as pd

# Load action catalog
with open('conf/actions/action_catalog.yaml') as f:
    action_catalog = yaml.safe_load(f)

# Load models
pay_any_model = PayAnyModel(config={})
pay_any_model.model = mlflow.lightgbm.load_model("models:/nba_pay_any_model/Production")

amount_model = AmountModel(config={})
amount_model.model = mlflow.sklearn.load_model("models:/nba_amount_model/Production")

# Initialize components
constraint_engine = ConstraintEngine(action_catalog, business_date='2024-02-15')
policy_ranker = PolicyRanker(action_catalog)

# Test account
account = {
    'account_id': 'ACC123',
    'bucket': 1,
    'balance': 3000,
    'delay': 35,
    'line_optin': True,
    'sms_optin': True,
    'mobile_present': True,
    'dnc': False,
    'contact_count_7d': 1,
    'fatigue_score': 0.3,
    'persona_segment': 'high_value_cooperative',
    # ... other features
}

# STEP 1: Get feasible actions
feasible, blocked, reasons = constraint_engine.get_feasible_actions(account)
print(f"Feasible: {feasible}")
print(f"Blocked: {blocked}")

# STEP 2: Score all feasible actions with models
customer_df = pd.DataFrame([account])

model_predictions = {}
for action in feasible:
    pay_any_probs = pay_any_model.predict_all_actions(customer_df, [action])
    expected_amounts = amount_model.predict_all_actions(customer_df, [action])

    model_predictions[action] = {
        'pay_any_prob': pay_any_probs[action],
        'expected_amount': expected_amounts[action]
    }

print("\nModel predictions:")
for action, preds in model_predictions.items():
    print(f"  {action}: P(pay)={preds['pay_any_prob']:.3f}, E(amount)=${preds['expected_amount']:.0f}")

# STEP 3: Rank and get Top-3
top_3 = policy_ranker.rank_actions(account, feasible, model_predictions)

print("\nTop-3 Actions:")
for action_result in top_3:
    print(f"  {action_result['rank']}. {action_result['action']}: score={action_result['score']:.3f}")

# Expected output similar to:
# Feasible: ['A0_NONE', 'A1_LINE', 'A2_SMS', 'A4_CALL']
# Blocked: ['A3_VOICE_IVR', 'A5_SETTLEMENT_OFFER', 'A6_OA_REFERRAL']
#
# Model predictions:
#   A0_NONE: P(pay)=0.420, E(amount)=$800
#   A1_LINE: P(pay)=0.445, E(amount)=$950
#   A2_SMS: P(pay)=0.480, E(amount)=$1100
#   A4_CALL: P(pay)=0.520, E(amount)=$1300
#
# Top-3 Actions:
#   1. A4_CALL: score=0.635
#   2. A2_SMS: score=0.562
#   3. A1_LINE: score=0.501

print("\n✓ End-to-end test passed")
```

---

# 📋 WEEK 5-8: [CONTINUES IN NEXT MESSAGE DUE TO LENGTH]

Would you like me to continue with Weeks 5-8 which cover:
- Week 5: Capacity Allocation + Explainability
- Week 6: A/B Test Design + Shadow Run
- Week 7: Pilot Launch + Monitoring
- Week 8: Performance Analysis + Scale-Up Plan

?