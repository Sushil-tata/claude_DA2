"""
Example Usage - Recovery Agent Practical
========================================

Demonstrates end-to-end usage of the recovery agent system.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from recovery_agent_practical import RecoveryAgentPipeline
from recovery_agent_practical.segmentation import PersonaBuilder
from recovery_agent_practical.scoring import RecoveryScorecard6M
from recovery_agent_practical.routing import ActionOverlayRouter


# ─────────────────────────────────────────────────────────────────────────────
# GENERATE SYNTHETIC DATA
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic_features(n_accounts: int = 1000) -> pd.DataFrame:
    """Generate synthetic feature data for demonstration"""
    np.random.seed(42)

    # Account identifiers
    account_ids = [f"ACC{i:06d}" for i in range(n_accounts)]

    # Stage distribution
    stages = np.random.choice(["NPL", "CHARGEOFF"], n_accounts, p=[0.6, 0.4])

    # Balance
    balances = np.random.lognormal(mean=11.5, sigma=0.8, size=n_accounts)  # ~50K-500K THB

    # DPD
    dpds = []
    for stage in stages:
        if stage == "NPL":
            dpd = np.random.randint(90, 180)
        else:
            dpd = np.random.randint(180, 365)
        dpds.append(dpd)

    # Months since chargeoff
    months_since_chargeoff = []
    for stage in stages:
        if stage == "CHARGEOFF":
            months = np.random.randint(1, 36)
        else:
            months = 0
        months_since_chargeoff.append(months)

    # Payment behavior features
    payment_count_30d = np.random.poisson(lam=0.5, size=n_accounts)
    payment_count_90d = np.random.poisson(lam=1.5, size=n_accounts)
    payment_count_180d = np.random.poisson(lam=3.0, size=n_accounts)
    payment_count_365d = np.random.poisson(lam=6.0, size=n_accounts)

    payment_amt_30d = payment_count_30d * np.random.uniform(500, 5000, n_accounts)
    payment_amt_90d = payment_count_90d * np.random.uniform(500, 5000, n_accounts)
    payment_amt_180d = payment_count_180d * np.random.uniform(500, 5000, n_accounts)
    payment_amt_365d = payment_count_365d * np.random.uniform(500, 5000, n_accounts)

    days_since_last_payment = np.random.randint(1, 180, size=n_accounts)
    last_payment_amount = np.random.uniform(500, 10000, size=n_accounts)

    # Engagement features
    outbound_calls_made = np.random.poisson(lam=5, size=n_accounts)
    calls_connected = np.random.binomial(outbound_calls_made, 0.3)
    call_response_rate = calls_connected / np.maximum(1, outbound_calls_made)

    contacts_made_30d = np.random.poisson(lam=3, size=n_accounts)
    contacts_made_90d = np.random.poisson(lam=8, size=n_accounts)

    sms_sent = np.random.poisson(lam=10, size=n_accounts)
    sms_responded = np.random.binomial(sms_sent, 0.2)
    sms_response_rate = sms_responded / np.maximum(1, sms_sent)

    days_since_last_contact = np.random.randint(1, 90, size=n_accounts)

    ptp_made = np.random.poisson(lam=2, size=n_accounts)
    ptp_kept = np.random.binomial(ptp_made, 0.4)
    ptp_kept_rate = ptp_kept / np.maximum(1, ptp_made)

    # Bureau features
    bureau_total_outstanding = balances * np.random.uniform(2, 5, size=n_accounts)
    bureau_monthly_instalment = bureau_total_outstanding * np.random.uniform(0.02, 0.05, size=n_accounts)
    bureau_secured_loan_flag = np.random.choice([True, False], n_accounts, p=[0.3, 0.7])
    bureau_secured_outstanding = np.where(
        bureau_secured_loan_flag,
        np.random.uniform(100_000, 2_000_000, n_accounts),
        0
    )
    bureau_delinquent_other = np.random.choice([True, False], n_accounts, p=[0.4, 0.6])
    bureau_active_loan_count = np.random.poisson(lam=2, size=n_accounts)
    bureau_new_loan_12m = np.random.poisson(lam=0.5, size=n_accounts)

    # Avoidance flags
    wrong_number_flag = np.random.choice([True, False], n_accounts, p=[0.1, 0.9])
    dispute_flag = np.random.choice([True, False], n_accounts, p=[0.15, 0.85])
    lawyer_mentioned = np.random.choice([True, False], n_accounts, p=[0.05, 0.95])
    sms_opt_out = np.random.choice([True, False], n_accounts, p=[0.2, 0.8])

    # Balance decomposition
    principal_outstanding = balances * 0.7
    accrued_interest = balances * 0.2
    penalty_charges = balances * 0.1

    # Credit limit
    credit_limit = balances * np.random.uniform(1.5, 3.0, size=n_accounts)

    # Build DataFrame
    df = pd.DataFrame({
        "account_id": account_ids,
        "stage": stages,
        "balance": balances,
        "days_past_due": dpds,
        "months_since_chargeoff": months_since_chargeoff,

        # Payment features
        "payment_count_30d": payment_count_30d,
        "payment_count_90d": payment_count_90d,
        "payment_count_180d": payment_count_180d,
        "payment_count_365d": payment_count_365d,
        "payment_amt_30d": payment_amt_30d,
        "payment_amt_90d": payment_amt_90d,
        "payment_amt_180d": payment_amt_180d,
        "payment_amt_365d": payment_amt_365d,
        "days_since_last_payment": days_since_last_payment,
        "last_payment_amount": last_payment_amount,

        # Engagement features
        "outbound_calls_made": outbound_calls_made,
        "calls_connected": calls_connected,
        "call_response_rate": call_response_rate,
        "contacts_made_30d": contacts_made_30d,
        "contacts_made_90d": contacts_made_90d,
        "sms_sent": sms_sent,
        "sms_responded": sms_responded,
        "sms_response_rate": sms_response_rate,
        "days_since_last_contact": days_since_last_contact,
        "ptp_made": ptp_made,
        "ptp_kept": ptp_kept,
        "ptp_kept_rate": ptp_kept_rate,

        # Bureau features
        "bureau_total_outstanding": bureau_total_outstanding,
        "bureau_monthly_instalment": bureau_monthly_instalment,
        "bureau_secured_loan_flag": bureau_secured_loan_flag,
        "bureau_secured_outstanding": bureau_secured_outstanding,
        "bureau_delinquent_other": bureau_delinquent_other,
        "bureau_active_loan_count": bureau_active_loan_count,
        "bureau_new_loan_12m": bureau_new_loan_12m,

        # Avoidance flags
        "wrong_number_flag": wrong_number_flag,
        "dispute_flag": dispute_flag,
        "lawyer_mentioned": lawyer_mentioned,
        "sms_opt_out": sms_opt_out,

        # Balance decomposition
        "principal_outstanding": principal_outstanding,
        "accrued_interest": accrued_interest,
        "penalty_charges": penalty_charges,
        "credit_limit": credit_limit,

        # Context
        "has_secured_assets": bureau_secured_loan_flag,
    })

    return df


def generate_synthetic_labels(features_df: pd.DataFrame) -> pd.DataFrame:
    """Generate synthetic labels for training"""
    np.random.seed(42)

    labels = []
    for _, row in features_df.iterrows():
        # Recovery amount depends on:
        # - Payment history (higher payment_count → higher recovery)
        # - Engagement (higher call_response_rate → higher recovery)
        # - Balance (larger balance → potentially larger recovery but lower probability)

        payment_score = (
            row["payment_count_180d"] * 0.3 +
            row["call_response_rate"] * 100 * 0.3 +
            row["ptp_kept_rate"] * 100 * 0.2 +
            (100 - row["days_since_last_payment"] / 3.65) * 0.2
        )

        # Probability of any payment
        p_payment = 1 / (1 + np.exp(-(payment_score - 50) / 20))

        # If pays, amount is proportional to balance and capacity
        if np.random.random() < p_payment:
            recovery_pct = np.random.beta(2, 5)  # Typically 10-30% recovery
            recovery_amount = row["balance"] * recovery_pct
        else:
            recovery_amount = 0.0

        labels.append({
            "account_id": row["account_id"],
            "recovery_amount_180d": recovery_amount,
        })

    return pd.DataFrame(labels)


# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE 1: COMPONENT-BY-COMPONENT USAGE
# ─────────────────────────────────────────────────────────────────────────────

def example_component_by_component():
    """Demonstrate using each component individually"""
    print("=" * 80)
    print("EXAMPLE 1: Component-by-Component Usage")
    print("=" * 80)
    print()

    # Generate synthetic data
    print("Generating synthetic data...")
    features_df = generate_synthetic_features(n_accounts=100)
    print(f"  ✓ Generated {len(features_df)} accounts")
    print()

    # ─────────────────────────────────────────────────────────────────────────
    # Step 1: Persona Assignment
    # ─────────────────────────────────────────────────────────────────────────
    print("Step 1: Persona Assignment")
    print("-" * 80)

    persona_builder = PersonaBuilder(payment_threshold=500.0)
    persona_df = persona_builder.assign_batch(features_df)

    print(f"Assigned personas for {len(persona_df)} accounts")
    print("\nPersona distribution:")
    print(persona_df["persona"].value_counts())
    print("\nSample assignments:")
    print(persona_df[["account_id", "persona", "payment_behavior_score", "engagement_score", "confidence_level"]].head(5))
    print()

    # ─────────────────────────────────────────────────────────────────────────
    # Step 2: Recovery Scoring (requires training first)
    # ─────────────────────────────────────────────────────────────────────────
    print("Step 2: Recovery Scoring (Training)")
    print("-" * 80)

    # Generate labels
    labels_df = generate_synthetic_labels(features_df)

    # Prepare features for scoring
    X = features_df.drop(columns=["account_id"], errors="ignore")

    # Train scorecard
    scorecard = RecoveryScorecard6M(model_type="TWO_PART", payment_threshold=500.0)

    # Merge features with labels
    train_data = features_df.merge(labels_df, on="account_id")
    X_train = train_data.drop(columns=["account_id", "recovery_amount_180d"])
    y_train = train_data["recovery_amount_180d"]

    print(f"Training on {len(X_train)} accounts...")
    metrics = scorecard.train(X_train, y_train)

    print(f"\n  ✓ Training complete")
    print(f"  - AUC: {metrics['train'].auc_roc:.3f}")
    print(f"  - Precision: {metrics['train'].precision:.3f}")
    print(f"  - MAE: {metrics['train'].mae:.0f} THB")
    print()

    # Score
    print("Scoring accounts...")
    scores_df = scorecard.score_batch(features_df, score_date="2024-01-15")

    print("\nScore band distribution:")
    print(scores_df["score_band"].value_counts())
    print("\nSample scores:")
    print(scores_df[["account_id", "score_band", "p_recovery", "expected_recovery"]].head(5))
    print()

    # ─────────────────────────────────────────────────────────────────────────
    # Step 3: Action Routing
    # ─────────────────────────────────────────────────────────────────────────
    print("Step 3: Action Routing")
    print("-" * 80)

    # Merge persona and scores
    enriched_df = features_df.merge(persona_df, on="account_id").merge(scores_df, on="account_id")

    # Route actions
    router = ActionOverlayRouter()
    actions_df = router.route_batch(enriched_df)

    print(f"Routed actions for {len(actions_df)} accounts")
    print("\nAction distribution:")
    print(actions_df["recommended_action"].value_counts())
    print("\nPriority tier distribution:")
    print(actions_df["priority_tier"].value_counts())
    print("\nSample recommendations:")
    print(actions_df[["account_id", "persona", "score_band", "recommended_action", "priority_tier"]].head(10))
    print()


# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE 2: END-TO-END PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def example_end_to_end_pipeline():
    """Demonstrate using the full pipeline orchestrator"""
    print("=" * 80)
    print("EXAMPLE 2: End-to-End Pipeline")
    print("=" * 80)
    print()

    # Generate data
    print("Generating synthetic data...")
    features_df = generate_synthetic_features(n_accounts=500)
    labels_df = generate_synthetic_labels(features_df)
    print(f"  ✓ Generated {len(features_df)} accounts with labels")
    print()

    # Initialize pipeline
    print("Initializing pipeline...")
    pipeline = RecoveryAgentPipeline(
        scorecard_model_type="TWO_PART",
        payment_threshold=500.0
    )
    print("  ✓ Pipeline initialized")
    print()

    # Train scorecard
    print("Training scorecard...")
    metrics = pipeline.train_scorecard(
        features_df=features_df,
        labels_df=labels_df,
        val_split=0.2
    )

    print(f"\n  ✓ Scorecard training complete")
    if "val" in metrics:
        print(f"  - Validation AUC: {metrics['val'].auc_roc:.3f}")
        print(f"  - Validation MAE: {metrics['val'].mae:.0f} THB")
    print()

    # Score batch
    print("Scoring batch for 2024-01-15...")
    daily_scores = pipeline.score_batch(
        features_df=features_df,
        score_date="2024-01-15",
        write_outputs=False,  # Don't write files in example
    )

    print(f"\n  ✓ Scoring complete for {len(daily_scores)} accounts")
    print()

    # Summary
    print("=" * 80)
    print("SCORING SUMMARY")
    print("=" * 80)
    print()

    print("Persona Distribution:")
    persona_counts = daily_scores["persona"].value_counts()
    for persona, count in persona_counts.items():
        pct = count / len(daily_scores) * 100
        print(f"  {persona:25s}: {count:4d} ({pct:5.1f}%)")
    print()

    print("Score Band Distribution:")
    for band in ["HOT", "WARM", "COLD", "FROZEN"]:
        count = (daily_scores["score_band"] == band).sum()
        pct = count / len(daily_scores) * 100
        print(f"  {band:10s}: {count:4d} ({pct:5.1f}%)")
    print()

    print("Action Distribution:")
    action_counts = daily_scores["recommended_action"].value_counts()
    for action, count in action_counts.items():
        pct = count / len(daily_scores) * 100
        print(f"  {action:20s}: {count:4d} ({pct:5.1f}%)")
    print()

    print("Expected Recovery:")
    total_expected = daily_scores["expected_recovery_amount"].sum()
    mean_expected = daily_scores["expected_recovery_amount"].mean()
    print(f"  Total: {total_expected:,.0f} THB")
    print(f"  Mean:  {mean_expected:,.0f} THB per account")
    print()

    # Sample output
    print("Sample Daily Scores:")
    print(daily_scores[[
        "account_id", "persona", "score_band", "p_recovery_6m",
        "recommended_action", "priority_tier"
    ]].head(10).to_string(index=False))
    print()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Run examples
    example_component_by_component()
    print("\n\n")
    example_end_to_end_pipeline()

    print()
    print("=" * 80)
    print("Examples complete!")
    print("=" * 80)
