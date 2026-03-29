"""
BFE Repository - Example Usage
===============================

Demonstrates how to use the Behavioral Feature Engineering repository
with synthetic data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date

# Import BFE repository
from decision_engine.feature_store.behavioral_feature_engine import (
    get_features,
    get_features_batch,
    list_features,
    save_feature_registry
)


def generate_synthetic_delinquency_data(account_id: str, months: int = 24) -> pd.DataFrame:
    """Generate synthetic DPD history for demonstration"""
    np.random.seed(hash(account_id) % 2**32)

    dates = []
    dpd_values = []

    end_date = date(2024, 1, 31)
    current_date = end_date - timedelta(days=30 * months)

    # Simulate DPD progression
    current_dpd = 0

    for i in range(months):
        dates.append(current_date)

        # Random walk with drift
        drift = np.random.choice([-5, 0, 5, 10], p=[0.1, 0.5, 0.3, 0.1])
        noise = np.random.randint(-10, 10)

        current_dpd = max(0, current_dpd + drift + noise)

        # Cap at 365
        current_dpd = min(current_dpd, 365)

        dpd_values.append(current_dpd)

        # Move to next month
        current_date += timedelta(days=30)

    df = pd.DataFrame({
        "account_id": account_id,
        "date": dates,
        "dpd": dpd_values,
        "account_open_date": dates[0]
    })

    return df


def generate_synthetic_payment_data(account_id: str, months: int = 24) -> pd.DataFrame:
    """Generate synthetic payment history for demonstration"""
    np.random.seed(hash(account_id) % 2**32)

    dates = []
    payment_amounts = []
    amounts_due = []
    minimum_dues = []
    statement_balances = []
    balances = []

    end_date = date(2024, 1, 31)
    current_date = end_date - timedelta(days=30 * months)

    current_balance = np.random.uniform(50000, 200000)

    for i in range(months):
        dates.append(current_date)

        # Amount due (typically 5% of balance + interest)
        amount_due = current_balance * 0.05 + np.random.uniform(500, 2000)
        amounts_due.append(amount_due)

        # Minimum due (10% of amount due)
        minimum_due = amount_due * 0.10
        minimum_dues.append(minimum_due)

        # Statement balance
        statement_balance = current_balance
        statement_balances.append(statement_balance)

        # Payment amount (with variability)
        payment_behavior = np.random.choice(["full", "partial", "minimum", "none"], p=[0.3, 0.4, 0.2, 0.1])

        if payment_behavior == "full":
            payment = amount_due * np.random.uniform(0.95, 1.05)
        elif payment_behavior == "partial":
            payment = amount_due * np.random.uniform(0.40, 0.90)
        elif payment_behavior == "minimum":
            payment = minimum_due * np.random.uniform(0.90, 1.10)
        else:  # none
            payment = 0

        payment_amounts.append(payment)

        # Update balance
        current_balance = max(0, current_balance + amount_due - payment)
        balances.append(current_balance)

        # Move to next month
        current_date += timedelta(days=30)

    df = pd.DataFrame({
        "account_id": account_id,
        "date": dates,
        "payment_amount": payment_amounts,
        "amount_due": amounts_due,
        "minimum_due": minimum_dues,
        "statement_balance": statement_balances,
        "balance": balances,
        "due_date": [d + timedelta(days=15) for d in dates],
        "payment_date": [d + timedelta(days=np.random.randint(0, 25)) for d in dates]
    })

    return df


def example_single_account():
    """Example: Get features for a single account"""
    print("=" * 80)
    print("EXAMPLE 1: Single Account Feature Extraction")
    print("=" * 80)
    print()

    # Generate synthetic data
    account_id = "ACC123456"

    print(f"Generating synthetic data for account: {account_id}")
    dpd_history = generate_synthetic_delinquency_data(account_id, months=24)
    payment_history = generate_synthetic_payment_data(account_id, months=24)

    print(f"  ✓ DPD history: {len(dpd_history)} months")
    print(f"  ✓ Payment history: {len(payment_history)} months")
    print()

    # Prepare account history
    account_history = {
        "delinquency": dpd_history,
        "payment": payment_history
    }

    # Get features as of 2024-01-31
    as_of_date = "2024-01-31"

    print(f"Computing features as of {as_of_date}...")
    print()

    # ─────────────────────────────────────────────────────────────────────
    # NOTE: First run will prompt for schema mapping!
    # You'll be asked to map BFE field names to your data column names
    # ─────────────────────────────────────────────────────────────────────

    features = get_features(
        account_id=account_id,
        account_history=account_history,
        as_of_date=as_of_date,
        feature_sets=["delinquency", "payment"]
    )

    print()
    print("-" * 80)
    print("FEATURES COMPUTED:")
    print("-" * 80)

    # Display features by category
    delinquency_features = {k: v for k, v in features.items() if k.startswith("delinquency.")}
    payment_features = {k: v for k, v in features.items() if k.startswith("payment.")}

    print(f"\nDelinquency Features ({len(delinquency_features)}):")
    for key in sorted(delinquency_features.keys())[:10]:  # Show first 10
        value = delinquency_features[key]
        if isinstance(value, float):
            print(f"  {key:50s}: {value:10.2f}")
        else:
            print(f"  {key:50s}: {value}")

    print(f"\n... and {len(delinquency_features) - 10} more delinquency features")

    print(f"\nPayment Features ({len(payment_features)}):")
    for key in sorted(payment_features.keys())[:10]:  # Show first 10
        value = payment_features[key]
        if isinstance(value, float):
            print(f"  {key:50s}: {value:10.2f}")
        else:
            print(f"  {key:50s}: {value}")

    print(f"\n... and {len(payment_features) - 10} more payment features")

    # Show key insights
    print()
    print("-" * 80)
    print("KEY INSIGHTS:")
    print("-" * 80)
    print(f"  Current DPD: {features.get('delinquency.dpd_current', 'N/A')}")
    print(f"  Current Bucket: {features.get('delinquency.bucket_current', 'N/A')}")
    print(f"  Delinquency Regime: {features.get('delinquency.delinquency_regime', 'N/A')}")
    print(f"  Payment Regime: {features.get('payment.payment_regime', 'N/A')}")
    print(f"  Payment Ratio (6M avg): {features.get('payment.payment_ratio_6M_mean', 'N/A'):.2%}")
    print()


def example_batch_processing():
    """Example: Get features for multiple accounts (batch)"""
    print()
    print("=" * 80)
    print("EXAMPLE 2: Batch Processing (Multiple Accounts)")
    print("=" * 80)
    print()

    # Generate data for 5 accounts
    account_ids = [f"ACC{i:06d}" for i in range(1, 6)]

    accounts_history = {}

    print("Generating synthetic data for accounts:")
    for account_id in account_ids:
        dpd_history = generate_synthetic_delinquency_data(account_id, months=18)
        payment_history = generate_synthetic_payment_data(account_id, months=18)

        accounts_history[account_id] = {
            "delinquency": dpd_history,
            "payment": payment_history
        }

        print(f"  ✓ {account_id}")

    print()
    print("Computing features for all accounts...")
    print()

    # Batch processing
    features_df = get_features_batch(
        accounts_history=accounts_history,
        as_of_date="2024-01-31",
        feature_sets=["delinquency", "payment"]
    )

    print("-" * 80)
    print("BATCH RESULTS:")
    print("-" * 80)
    print(f"Total accounts processed: {len(features_df)}")
    print(f"Total features per account: {len(features_df.columns)}")
    print()

    # Show summary
    print("Summary of Key Features:")
    print()
    print(features_df[[
        "account_id",
        "delinquency.dpd_current",
        "delinquency.bucket_current",
        "delinquency.delinquency_regime",
        "payment.payment_regime",
        "payment.payment_ratio_6M_mean"
    ]].to_string(index=False))
    print()


def example_feature_catalog():
    """Example: List available features"""
    print()
    print("=" * 80)
    print("EXAMPLE 3: Feature Catalog")
    print("=" * 80)
    print()

    # List all features
    feature_catalog = list_features()

    print(f"Total features available: {len(feature_catalog)}")
    print()

    # Show features by module
    delinq_features = {k: v for k, v in feature_catalog.items() if k.startswith("delinquency.")}
    payment_features = {k: v for k, v in feature_catalog.items() if k.startswith("payment.")}

    print(f"Delinquency module: {len(delinq_features)} features")
    print(f"Payment module: {len(payment_features)} features")
    print()

    # Show sample feature metadata
    print("Sample Feature Metadata:")
    print("-" * 80)

    sample_features = [
        "delinquency.dpd_current",
        "delinquency.delinquency_regime",
        "payment.payment_ratio_6M_mean",
        "payment.payment_regime"
    ]

    for feature_name in sample_features:
        if feature_name in feature_catalog:
            meta = feature_catalog[feature_name]
            print(f"\n{feature_name}:")
            print(f"  Definition: {meta.get('definition', 'N/A')}")
            print(f"  Tier: {meta.get('tier', 'N/A')}")
            print(f"  Type: {meta.get('feature_type', 'N/A')}")
            print(f"  Signal: {meta.get('signal_direction', 'N/A')}")
            print(f"  Min History: {meta.get('min_history_months', 'N/A')} months")

    print()


def example_save_registry():
    """Example: Save feature registry to JSON"""
    print()
    print("=" * 80)
    print("EXAMPLE 4: Save Feature Registry")
    print("=" * 80)
    print()

    output_path = "bfe_feature_registry.json"

    print(f"Saving feature registry to {output_path}...")

    save_feature_registry(output_path=output_path)

    print()
    print(f"✓ Feature registry saved successfully!")
    print(f"  You can now inspect the registry at: {output_path}")
    print()


def main():
    """Run all examples"""
    print()
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "BFE REPOSITORY - EXAMPLE USAGE" + " " * 28 + "║")
    print("╚" + "═" * 78 + "╝")
    print()

    # Run examples
    example_single_account()
    example_batch_processing()
    example_feature_catalog()
    example_save_registry()

    print()
    print("=" * 80)
    print("All examples completed successfully!")
    print("=" * 80)
    print()
    print("NEXT STEPS:")
    print("  1. Review the generated feature_registry.json")
    print("  2. Integrate get_features() into your Decision Engine")
    print("  3. Configure schema mappings for your production data")
    print("  4. Extend with additional modules (bureau, utilization, etc.)")
    print()


if __name__ == "__main__":
    main()
