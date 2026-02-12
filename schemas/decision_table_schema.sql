-- Golden Decision Table Schema
-- Grain: account_id × business_date
-- Point-in-time correct (no future leakage)

CREATE TABLE IF NOT EXISTS debt_collection.decision_table (

    -- Primary Keys
    account_id STRING NOT NULL,
    business_date DATE NOT NULL,

    -- Account Identifiers
    customer_id STRING,
    product_type STRING,  -- CC, PL, Auto, etc.

    -- ========================================
    -- CORE STATE (Business-Defined)
    -- ========================================

    -- Delinquency State
    bucket INT,  -- 0=current, 1=1-30 DPD, 2=31-60, etc.
    due_date DATE,  -- bill_date + 20 days (business definition)
    delay INT,  -- CRITICAL: max(0, business_date - due_date)
    days_past_due INT,  -- System DPD (may differ from delay)
    cycle_day INT,  -- Day within billing cycle (1-30)

    -- Balance & Payment Info
    balance DECIMAL(10,2),
    original_balance DECIMAL(10,2),
    MAD DECIMAL(10,2),  -- Minimum amount due
    past_due_amount DECIMAL(10,2),
    credit_limit DECIMAL(10,2),
    utilization DECIMAL(5,4),  -- balance / limit

    -- ========================================
    -- PAYMENT FEATURES (Historical)
    -- ========================================

    -- Last Payment
    days_since_last_payment INT,
    last_payment_amount DECIMAL(10,2),
    last_payment_date DATE,

    -- Payment Windows (7/14/30 days)
    payment_count_7d INT,
    payment_count_14d INT,
    payment_count_30d INT,
    payment_sum_7d DECIMAL(10,2),
    payment_sum_14d DECIMAL(10,2),
    payment_sum_30d DECIMAL(10,2),

    -- MAD Coverage
    mad_coverage_last_cycle DECIMAL(5,4),  -- paid / MAD
    mad_coverage_last_3_cycles DECIMAL(5,4),
    mad_paid_on_time_count_6m INT,

    -- Payment Patterns
    avg_payment_amount_6m DECIMAL(10,2),
    payment_regularity_score DECIMAL(5,4),  -- CV of payment intervals

    -- ========================================
    -- DELINQUENCY HISTORY
    -- ========================================

    max_delay_3m INT,
    max_delay_6m INT,
    max_delay_12m INT,
    max_bucket_3m INT,
    max_bucket_6m INT,
    max_bucket_12m INT,

    -- Bucket Transitions
    bucket_transitions_6m INT,  -- How many times changed buckets
    ever_bucket_3_plus BOOLEAN,  -- Ever been in B3+
    time_in_current_bucket_days INT,

    -- Roll Patterns
    rolled_forward_last_cycle BOOLEAN,  -- Got worse
    rolled_backward_last_cycle BOOLEAN,  -- Got better
    self_cured_last_3m BOOLEAN,  -- Went from delinquent to current

    -- ========================================
    -- CHANNEL HISTORY (Action-Conditional Features)
    -- ========================================

    -- Contact Attempts by Channel (last 7/14/30 days)
    line_attempts_7d INT,
    line_attempts_14d INT,
    line_attempts_30d INT,
    sms_attempts_7d INT,
    sms_attempts_14d INT,
    sms_attempts_30d INT,
    voice_attempts_7d INT,
    voice_attempts_14d INT,
    voice_attempts_30d INT,
    call_attempts_7d INT,
    call_attempts_14d INT,
    call_attempts_30d INT,

    -- Total Contact Fatigue
    total_contacts_7d INT,
    total_contacts_30d INT,
    total_contacts_90d INT,

    -- Last Contact per Channel
    days_since_last_line INT,
    days_since_last_sms INT,
    days_since_last_call INT,

    -- Response Rates by Channel (historical)
    line_response_rate_90d DECIMAL(5,4),  -- % of LINE that got response
    sms_response_rate_90d DECIMAL(5,4),
    call_connect_rate_90d DECIMAL(5,4),

    -- Best Historical Channel
    best_channel_historical STRING,  -- Channel with most payments after contact
    best_time_of_day_historical INT,  -- Hour (0-23)
    best_day_of_week_historical INT,  -- 1=Mon, 7=Sun

    -- Fatigue Score (weighted sum with decay)
    fatigue_score DECIMAL(5,4),

    -- Promise to Pay
    active_ptp BOOLEAN,
    ptp_amount DECIMAL(10,2),
    ptp_date DATE,
    ptp_kept_rate_6m DECIMAL(5,4),

    -- ========================================
    -- OPERABILITY FLAGS
    -- ========================================

    line_optin BOOLEAN,
    line_id_present BOOLEAN,
    mobile_present BOOLEAN,
    mobile_valid BOOLEAN,
    sms_optin BOOLEAN,
    email_present BOOLEAN,
    email_valid BOOLEAN,
    address_valid BOOLEAN,

    -- Compliance Flags
    dnc BOOLEAN,  -- Do not call
    cease_and_desist BOOLEAN,
    active_dispute BOOLEAN,
    complaint_filed BOOLEAN,
    fraud_flag BOOLEAN,
    bankruptcy_flag BOOLEAN,
    deceased_flag BOOLEAN,

    -- ========================================
    -- OFFER HISTORY
    -- ========================================

    settlement_offers_made_6m INT,
    settlement_offers_accepted_6m INT,
    settlement_acceptance_rate DECIMAL(5,4),
    avg_discount_accepted DECIMAL(5,2),

    payment_plans_offered_6m INT,
    payment_plans_accepted_6m INT,
    payment_plan_completion_rate DECIMAL(5,4),

    last_settlement_offer_date DATE,
    days_since_last_settlement_offer INT,

    active_settlement BOOLEAN,
    active_payment_plan BOOLEAN,

    -- ========================================
    -- DEMOGRAPHIC & FINANCIAL (if available)
    -- ========================================

    age INT,
    income_estimated DECIMAL(10,2),
    employment_status STRING,
    credit_score INT,
    debt_to_income_ratio DECIMAL(5,4),

    -- Derived Financial Capacity
    payment_capacity_score DECIMAL(5,4),  -- Rule-based or modeled

    -- ========================================
    -- PERSONA SEGMENTATION (Derived)
    -- ========================================

    persona_segment STRING,  -- 'high_value_cooperative', etc.
    rule_segment STRING,
    ml_segment STRING,

    -- ========================================
    -- SYSTEM-DEFINED ACTION (for uplift)
    -- ========================================

    sd_recommended_action STRING,  -- System default recommendation (if exists)
    efs_executed_action STRING,  -- What was actually executed

    -- ========================================
    -- OUTCOMES (JOINED LATER - NO LEAKAGE!)
    -- ========================================
    -- These are joined with LEAD() to ensure no leakage
    -- Horizon H = 7 or 14 days for early buckets, 30 for late

    -- Binary Outcomes (joined as separate step)
    -- outcome_pay_any_7d BOOLEAN,
    -- outcome_pay_any_14d BOOLEAN,
    -- outcome_pay_any_30d BOOLEAN,

    -- Amount Outcomes
    -- outcome_pay_amount_7d DECIMAL(10,2),
    -- outcome_pay_amount_14d DECIMAL(10,2),
    -- outcome_pay_amount_30d DECIMAL(10,2),

    -- Bucket Change
    -- outcome_bucket_7d INT,
    -- outcome_bucket_14d INT,
    -- outcome_bucket_change_7d INT,  -- negative = better

    -- Roll Worse Flag
    -- outcome_roll_worse_7d BOOLEAN,
    -- outcome_roll_worse_14d BOOLEAN,

    -- ========================================
    -- METADATA
    -- ========================================

    created_timestamp TIMESTAMP,
    pipeline_version STRING,

    PRIMARY KEY (account_id, business_date)
)
USING DELTA
PARTITIONED BY (business_date)
LOCATION 'dbfs:/debt_collection/decision_table/'
TBLPROPERTIES (
    'delta.autoOptimize.optimizeWrite' = 'true',
    'delta.autoOptimize.autoCompact' = 'true'
);

-- Create indexes for common queries
CREATE INDEX idx_bucket_date ON debt_collection.decision_table (bucket, business_date);
CREATE INDEX idx_persona_date ON debt_collection.decision_table (persona_segment, business_date);

-- ========================================
-- CRITICAL: DELAY DEFINITION
-- ========================================

-- COMMENT: delay = max(0, business_date - due_date)
-- This is the PRIMARY delinquency metric, NOT days_past_due from system
-- days_past_due may be calculated differently by different systems
-- delay is the business-defined metric used for ALL modeling

-- ========================================
-- OUTCOME LABELING (Separate Process)
-- ========================================

-- Run this AFTER decision table is built to join outcomes
-- This ensures no leakage (outcomes are FUTURE relative to decision_date)

CREATE OR REPLACE TEMP VIEW decision_table_with_outcomes AS
SELECT
    dt.*,

    -- 7-day outcomes
    LEAD(pay_any_7d, 1) OVER (PARTITION BY account_id ORDER BY business_date) as outcome_pay_any_7d,
    LEAD(pay_amount_7d, 1) OVER (PARTITION BY account_id ORDER BY business_date) as outcome_pay_amount_7d,
    LEAD(bucket, 7) OVER (PARTITION BY account_id ORDER BY business_date) as outcome_bucket_7d,

    -- 14-day outcomes
    LEAD(pay_any_14d, 1) OVER (PARTITION BY account_id ORDER BY business_date) as outcome_pay_any_14d,
    LEAD(pay_amount_14d, 1) OVER (PARTITION BY account_id ORDER BY business_date) as outcome_pay_amount_14d,

    -- 30-day outcomes
    LEAD(pay_any_30d, 1) OVER (PARTITION BY account_id ORDER BY business_date) as outcome_pay_any_30d,
    LEAD(pay_amount_30d, 1) OVER (PARTITION BY account_id ORDER BY business_date) as outcome_pay_amount_30d,

    -- Bucket change (negative = improvement)
    LEAD(bucket, 7) OVER (PARTITION BY account_id ORDER BY business_date) - bucket as outcome_bucket_change_7d,

    -- Roll worse flag (moved to higher bucket)
    CASE WHEN LEAD(bucket, 7) OVER (PARTITION BY account_id ORDER BY business_date) > bucket
         THEN TRUE ELSE FALSE END as outcome_roll_worse_7d

FROM debt_collection.decision_table dt;

-- ========================================
-- DATA QUALITY CHECKS
-- ========================================

-- Check 1: No future leakage
SELECT
    COUNT(*) as leakage_violations
FROM decision_table_with_outcomes
WHERE outcome_pay_any_7d IS NOT NULL
  AND business_date >= CURRENT_DATE();
-- Should be 0

-- Check 2: Delay calculation is correct
SELECT
    account_id,
    business_date,
    due_date,
    delay,
    DATEDIFF(business_date, due_date) as calculated_delay
FROM decision_table
WHERE delay != GREATEST(0, DATEDIFF(business_date, due_date))
LIMIT 10;
-- Should be empty

-- Check 3: Persona distribution
SELECT
    persona_segment,
    COUNT(*) as cnt,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) as pct
FROM decision_table
WHERE business_date = CURRENT_DATE()
GROUP BY persona_segment
ORDER BY cnt DESC;
