"""
Test Bureau Module Integration
================================

Quick test to verify bureau module works with BFE v1.3
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

import pandas as pd
from datetime import date
from decision_engine.feature_store.behavioral_feature_engine import get_features


def test_bureau_module():
    """Test bureau feature extraction with synthetic data"""

    print("\n" + "="*80)
    print("BFE v1.3 - BUREAU MODULE INTEGRATION TEST")
    print("="*80)

    # Create synthetic data matching user's schema
    account_history = {
        # Internal delinquency data
        "delinquency": pd.DataFrame({
            "account_id": ["CUST001"] * 12,
            "date": pd.date_range("2023-02-01", periods=12, freq="MS"),
            "dpd": [0, 0, 0, 0, 0, 30, 45, 60, 30, 15, 0, 0]
        }),

        # Internal payment data
        "payment": pd.DataFrame({
            "account_id": ["CUST001"] * 12,
            "date": pd.date_range("2023-02-01", periods=12, freq="MS"),
            "payment_amount": [5000, 5000, 5000, 5000, 5000, 3000, 2500, 2000, 3500, 4500, 5000, 5000],
            "amount_due": [5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000]
        }),

        # Bureau account data (matching user's schema)
        "bureau_accounts": pd.DataFrame({
            "REF_NO": ["CUST001"] * 5,
            "ASOFDATE": ["2024-01-31"] * 5,
            "ACCOUNTTYPE": ["CREDIT_CARD", "PERSONAL_LOAN", "HOME_LOAN", "CREDIT_CARD", "AUTO_LOAN"],
            "ACCOUNTSTATUS": ["ACTIVE", "ACTIVE", "ACTIVE", "CLOSED", "ACTIVE"],
            "CREDITLIMIT": [50000, 200000, 5000000, 30000, 800000],
            "AMOUNTOWED": [25000, 150000, 4500000, 0, 600000],
            "OVERDUEMONTHS": ["0", "0", "0", "0", "2"],
            "AMOUNTPASTDUE": ["0", "0", "0", "0", "15000"],
            "DATEACCOUNTOPENED": ["2020-01-15", "2021-06-01", "2019-03-20", "2022-01-01", "2023-05-15"],
            "DATEACCOUNTCLOSED": [None, None, None, "2023-12-31", None],
            "PAYMENTHISTORY1": ["000000000000", "000000000000", "000000000000", "000000", "000000120000"],
            "DATEOFLASTPAYMENT": ["2024-01-25", "2024-01-28", "2024-01-30", "2023-11-30", "2024-01-20"]
        }),

        # Bureau enquiry data
        "bureau_enquiries": pd.DataFrame({
            "REF_NO": ["CUST001"] * 4,
            "DATEOFENQUIRY": ["2023-10-15", "2023-11-20", "2023-12-10", "2024-01-10"],
            "ENQUIRYPURPOSE": ["CREDIT_CARD", "PERSONAL_LOAN", "AUTO_LOAN", "CREDIT_CARD"],
            "ENQUIRYAMOUNT": [50000, 200000, 800000, 100000]
        })
    }

    # Compute features
    print("\nComputing features (delinquency + payment + bureau + vintage + interactions)...")
    features = get_features(
        account_id="CUST001",
        account_history=account_history,
        as_of_date="2024-01-31",
        feature_sets=["delinquency", "payment", "bureau", "vintage", "interactions"]
    )

    print("\n" + "="*80)
    print("BUREAU FEATURES EXTRACTED")
    print("="*80)

    # Show bureau features
    bureau_features = {k: v for k, v in features.items() if k.startswith("bureau.")}
    print(f"\nTotal Bureau Features: {len(bureau_features)}")
    print("\nKey Bureau Features:")
    print("-" * 80)
    print(f"  bureau.bureau_total_accounts           : {features.get('bureau.bureau_total_accounts')}")
    print(f"  bureau.bureau_active_accounts          : {features.get('bureau.bureau_active_accounts')}")
    print(f"  bureau.bureau_overdue_accounts         : {features.get('bureau.bureau_overdue_accounts')}")
    print(f"  bureau.bureau_utilization_ratio        : {features.get('bureau.bureau_utilization_ratio'):.2%}")
    print(f"  bureau.bureau_maxed_out_accounts       : {features.get('bureau.bureau_maxed_out_accounts')}")
    print(f"  bureau.bureau_enquiries_6m             : {features.get('bureau.bureau_enquiries_6m')}")
    print(f"  bureau.bureau_enquiries_3m             : {features.get('bureau.bureau_enquiries_3m')}")
    print(f"  bureau.bureau_oldest_account_age_months: {features.get('bureau.bureau_oldest_account_age_months'):.1f}")

    # Show interaction features
    print("\n" + "="*80)
    print("BUREAU INTERACTION FEATURES")
    print("="*80)

    interaction_features = {k: v for k, v in features.items() if k.startswith("interactions.interaction_bureau") or k.startswith("interactions.interaction_rfm_bureau") or k.startswith("interactions.interaction_payment_bureau")}
    print(f"\nTotal Bureau Interaction Features: {len(interaction_features)}")
    print("\nKey Bureau Interactions:")
    print("-" * 80)

    for key, value in sorted(interaction_features.items()):
        feature_name = key.split(".")[-1]
        print(f"  {feature_name:50s} : {value}")

    # Summary
    print("\n" + "="*80)
    print("INTEGRATION TEST SUMMARY")
    print("="*80)

    total_features = len([k for k in features.keys() if not k.startswith(("bfe_version", "account_id", "as_of_date", "computed_at"))])
    delinquency_count = len([k for k in features.keys() if k.startswith("delinquency.")])
    payment_count = len([k for k in features.keys() if k.startswith("payment.")])
    bureau_count = len([k for k in features.keys() if k.startswith("bureau.")])
    vintage_count = len([k for k in features.keys() if k.startswith("vintage.")])
    interaction_count = len([k for k in features.keys() if k.startswith("interactions.")])

    print(f"\nModule Feature Counts:")
    print(f"  Delinquency : {delinquency_count:3d} features")
    print(f"  Payment     : {payment_count:3d} features")
    print(f"  Bureau      : {bureau_count:3d} features ← NEW in v1.3")
    print(f"  Vintage     : {vintage_count:3d} features")
    print(f"  Interactions: {interaction_count:3d} features ← Enhanced in v1.3")
    print(f"  " + "-" * 40)
    print(f"  TOTAL       : {total_features:3d} features")

    print(f"\nBFE Version: {features.get('bfe_version')}")
    print(f"As of Date : {features.get('as_of_date')}")

    print("\n✅ Bureau module integration successful!")
    print("\n" + "="*80)

    return features


if __name__ == "__main__":
    try:
        features = test_bureau_module()
    except Exception as e:
        print(f"\n❌ Error during test: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
