"""
Output Schemas for Recovery Agent Practical
===========================================

Daily Scoring Table + Audit Log schemas for production deployment.
"""

from dataclasses import dataclass
from typing import Optional
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# DAILY SCORING TABLE
# ─────────────────────────────────────────────────────────────────────────────

DAILY_SCORING_SCHEMA = {
    # Primary keys
    "account_id": "string",
    "score_date": "date",  # Snapshot date

    # Persona assignment
    "persona": "string",  # ACTIVE_PAYER | SELECTIVE_DEFAULTER | LIQUIDITY_CONSTRAINED | STRATEGIC | DORMANT
    "payment_behavior_score": "float",
    "engagement_score": "float",
    "capacity_score": "float",
    "avoidance_score": "float",
    "persona_confidence": "string",  # HIGH | MEDIUM | LOW
    "persona_flags": "string",  # Semicolon-separated flags

    # Recovery score (6M)
    "score_band": "string",  # HOT | WARM | COLD | FROZEN
    "p_recovery_6m": "float",  # Probability of recovery in 180d
    "expected_recovery_amount": "float",  # Expected THB recovered
    "model_version": "string",
    "model_type": "string",  # TWO_PART | TWEEDIE

    # Action recommendation
    "recommended_action": "string",  # SETTLEMENT_LUMP | SETTLEMENT_PLAN | AGENCY | LEGAL_REVIEW | HOLD
    "priority_tier": "string",  # TIER_1 | TIER_2 | TIER_3
    "contact_channel": "string",  # PHONE | SMS | EMAIL | LEGAL_NOTICE | NONE
    "offer_type": "string",  # HAIRCUT_30 | HAIRCUT_50 | PLAN_6M | PLAN_12M | null
    "routing_reasoning": "string",

    # Account context (for filtering/reporting)
    "stage": "string",  # SM | NPL | CHARGEOFF
    "balance": "float",
    "balance_band": "string",  # SMALL | MEDIUM | LARGE
    "days_past_due": "int",
    "months_since_chargeoff": "int",
    "has_secured_assets": "boolean",
    "bureau_delinquent_other": "boolean",

    # Metadata
    "pipeline_run_id": "string",
    "created_at": "timestamp",
}

# Partitioning: score_date (daily partitions)
# Primary key: (account_id, score_date)
# Sort by: priority_tier, score_band, balance DESC


# ─────────────────────────────────────────────────────────────────────────────
# AUDIT LOG
# ─────────────────────────────────────────────────────────────────────────────

AUDIT_LOG_SCHEMA = {
    # Event identification
    "audit_id": "string",  # UUID
    "account_id": "string",
    "event_timestamp": "timestamp",
    "event_type": "string",  # PERSONA_ASSIGNED | SCORED | ACTION_ROUTED | ACTION_EXECUTED | OUTCOME_OBSERVED

    # Event details
    "event_detail": "string",  # JSON string with event-specific data
    "previous_value": "string",  # For state changes
    "new_value": "string",

    # Context
    "pipeline_run_id": "string",
    "model_version": "string",
    "user_id": "string",  # If manual override
    "source_system": "string",  # RECOVERY_AGENT_PRACTICAL | MANUAL | API

    # Metadata
    "created_at": "timestamp",
}

# Partitioning: event_timestamp (daily)
# Primary key: audit_id
# Indexes: account_id, event_type, pipeline_run_id


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE SNAPSHOT TABLE (OPTIONAL - FOR DEBUGGING)
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_SNAPSHOT_SCHEMA = {
    # Primary keys
    "account_id": "string",
    "snapshot_date": "date",

    # Delinquency trajectory
    "dpd_current": "int",
    "dpd_30d_ago": "int",
    "dpd_90d_ago": "int",
    "dpd_trend": "string",  # IMPROVING | STABLE | DETERIORATING
    "months_at_180plus": "int",

    # Balance & utilization
    "balance": "float",
    "principal_outstanding": "float",
    "accrued_interest": "float",
    "penalty_charges": "float",
    "credit_limit": "float",
    "utilization_pct": "float",

    # Payment behavior (stage-wise windows)
    "payment_count_30d": "int",
    "payment_count_90d": "int",
    "payment_count_180d": "int",
    "payment_count_365d": "int",
    "payment_amt_30d": "float",
    "payment_amt_90d": "float",
    "payment_amt_180d": "float",
    "payment_amt_365d": "float",
    "days_since_last_payment": "int",
    "last_payment_amount": "float",

    # Engagement
    "contacts_made_30d": "int",
    "contacts_made_90d": "int",
    "calls_connected": "int",
    "outbound_calls_made": "int",
    "call_response_rate": "float",
    "sms_sent": "int",
    "sms_responded": "int",
    "sms_response_rate": "float",
    "days_since_last_contact": "int",
    "last_contact_outcome": "string",

    # PTP
    "ptp_made": "int",
    "ptp_kept": "int",
    "ptp_kept_rate": "float",

    # Bureau
    "bureau_total_outstanding": "float",
    "bureau_monthly_instalment": "float",
    "bureau_secured_loan_flag": "boolean",
    "bureau_secured_outstanding": "float",
    "bureau_delinquent_other": "boolean",
    "bureau_active_loan_count": "int",
    "bureau_new_loan_12m": "int",

    # Avoidance flags
    "wrong_number_flag": "boolean",
    "dispute_flag": "boolean",
    "lawyer_mentioned": "boolean",
    "sms_opt_out": "boolean",

    # Metadata
    "feature_pipeline_version": "string",
    "created_at": "timestamp",
}

# Partitioning: snapshot_date
# Primary key: (account_id, snapshot_date)


# ─────────────────────────────────────────────────────────────────────────────
# OUTCOME TRACKING TABLE (FOR MODEL MONITORING)
# ─────────────────────────────────────────────────────────────────────────────

OUTCOME_TRACKING_SCHEMA = {
    # Link to scoring
    "account_id": "string",
    "score_date": "date",  # Original score date
    "observation_date": "date",  # 180 days after score_date

    # Predicted vs. actual
    "predicted_p_recovery": "float",
    "predicted_amount": "float",
    "actual_recovery_amount": "float",
    "actual_recovery_flag": "boolean",  # paid > threshold

    # Action taken
    "recommended_action": "string",
    "action_executed": "string",  # May differ from recommendation
    "action_execution_date": "date",
    "action_execution_channel": "string",

    # Outcome metrics
    "error_absolute": "float",  # |predicted - actual|
    "error_squared": "float",
    "within_confidence_band": "boolean",

    # Metadata
    "model_version": "string",
    "created_at": "timestamp",
}

# Partitioning: observation_date
# Primary key: (account_id, score_date)
# Used for model performance monitoring and recalibration


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def create_daily_scoring_row(
    account_id: str,
    score_date: str,
    persona_result,  # PersonaAssignment
    recovery_score,  # RecoveryScore
    action_rec,  # ActionRecommendation
    account_data: pd.Series,
    pipeline_run_id: str,
) -> dict:
    """
    Create daily scoring table row from pipeline outputs.
    """
    return {
        # Primary keys
        "account_id": account_id,
        "score_date": score_date,

        # Persona
        "persona": persona_result.persona,
        "payment_behavior_score": persona_result.payment_behavior_score,
        "engagement_score": persona_result.engagement_score,
        "capacity_score": persona_result.capacity_score,
        "avoidance_score": persona_result.avoidance_score,
        "persona_confidence": persona_result.confidence_level,
        "persona_flags": "; ".join(persona_result.flags),

        # Recovery score
        "score_band": recovery_score.score_band,
        "p_recovery_6m": recovery_score.p_recovery,
        "expected_recovery_amount": recovery_score.expected_recovery,
        "model_version": recovery_score.model_version,
        "model_type": recovery_score.model_type,

        # Action
        "recommended_action": action_rec.recommended_action,
        "priority_tier": action_rec.priority_tier,
        "contact_channel": action_rec.contact_channel,
        "offer_type": action_rec.offer_type,
        "routing_reasoning": action_rec.reasoning,

        # Context
        "stage": account_data.get("stage"),
        "balance": account_data.get("balance"),
        "balance_band": action_rec.balance_band,
        "days_past_due": account_data.get("days_past_due"),
        "months_since_chargeoff": action_rec.months_since_chargeoff,
        "has_secured_assets": account_data.get("has_secured_assets", False),
        "bureau_delinquent_other": account_data.get("bureau_delinquent_other", False),

        # Metadata
        "pipeline_run_id": pipeline_run_id,
        "created_at": pd.Timestamp.now(),
    }


def create_audit_event(
    account_id: str,
    event_type: str,
    event_detail: str,
    pipeline_run_id: str,
    model_version: str = "v1.0",
    previous_value: Optional[str] = None,
    new_value: Optional[str] = None,
) -> dict:
    """
    Create audit log event.
    """
    import uuid

    return {
        "audit_id": str(uuid.uuid4()),
        "account_id": account_id,
        "event_timestamp": pd.Timestamp.now(),
        "event_type": event_type,
        "event_detail": event_detail,
        "previous_value": previous_value,
        "new_value": new_value,
        "pipeline_run_id": pipeline_run_id,
        "model_version": model_version,
        "user_id": "SYSTEM",
        "source_system": "RECOVERY_AGENT_PRACTICAL",
        "created_at": pd.Timestamp.now(),
    }
