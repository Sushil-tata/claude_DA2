"""
BFE Module 5: Bureau Features
==============================

Bureau credit report features including:
- Credit score trends and volatility
- Trade line analysis (total accounts, active, delinquent)
- Payment history across all accounts
- Credit utilization metrics
- Inquiry patterns (credit seeking behavior)
- Account mix and diversity
- Public records and collections
- Time-based bureau trends

Version: BFE_v1.3
Author: Behavioral Feature Engineering Team
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, date, timedelta
import logging

from ..utils import WindowCalculator, TemporalValidator, NullHandler

logger = logging.getLogger(__name__)


class BureauFeatureEngine:
    """
    Computes bureau credit report features with point-in-time safety.

    Data sources:
    - mnf_cra_rvw_s_account: Trade line/account data
    - mnf_cra_rvw_s_enquiry: Credit inquiry data
    - mnf_cra_rvw_s_history: Payment history data
    - mnf_cra_rvw_id_dummy: Customer identification data

    Features:
    - Credit score features (trends, volatility, momentum)
    - Trade line features (account counts, mix, status)
    - Delinquency features (overdue accounts, default status)
    - Utilization features (credit used, available credit)
    - Inquiry features (hard inquiries, credit seeking)
    - Payment history features (on-time payments, missed payments)
    - Account age features (oldest account, average age)
    - Risk indicators (public records, collections, restructures)
    """

    VERSION = "BFE_v1.3"
    MODULE_NAME = "bfe.bureau"

    def __init__(self):
        """Initialize bureau feature engine"""
        self.window_calc = WindowCalculator()
        self.temporal_validator = TemporalValidator()
        self.null_handler = NullHandler()
        self.logger = logging.getLogger(__name__)

    def compute_features(
        self,
        account_history: Dict[str, pd.DataFrame],
        account_id: str,
        as_of_date: date,
        schema_mappings: Optional[Dict[str, Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Compute bureau features from credit report data.

        Args:
            account_history: Dictionary with bureau data:
                - "bureau_accounts": Trade line/account data
                - "bureau_enquiries": Inquiry data
                - "bureau_history": Payment history data (optional)
            account_id: Customer identifier (REF_NO)
            as_of_date: Point-in-time date for feature computation
            schema_mappings: Field mappings (optional, uses defaults)

        Returns:
            Dictionary of bureau features
        """
        features = {}

        # Get bureau data
        accounts_df = account_history.get("bureau_accounts")
        enquiries_df = account_history.get("bureau_enquiries")
        history_df = account_history.get("bureau_history")

        if accounts_df is None or len(accounts_df) == 0:
            self.logger.warning(f"No bureau account data for {account_id}")
            return self._get_null_features()

        # Filter to customer and as_of_date
        accounts_df = accounts_df[accounts_df["REF_NO"] == account_id].copy()

        if len(accounts_df) == 0:
            return self._get_null_features()

        # Convert dates
        if "ASOFDATE" in accounts_df.columns:
            accounts_df["ASOFDATE"] = pd.to_datetime(accounts_df["ASOFDATE"])
            # Filter to as_of_date (point-in-time safety)
            accounts_df = accounts_df[accounts_df["ASOFDATE"] <= pd.to_datetime(as_of_date)]

        if len(accounts_df) == 0:
            return self._get_null_features()

        # Compute feature groups
        features.update(self._compute_account_count_features(accounts_df))
        features.update(self._compute_account_mix_features(accounts_df))
        features.update(self._compute_delinquency_features(accounts_df))
        features.update(self._compute_utilization_features(accounts_df))
        features.update(self._compute_account_age_features(accounts_df, as_of_date))
        features.update(self._compute_payment_history_features(accounts_df))
        features.update(self._compute_risk_indicator_features(accounts_df))

        # Enquiry features (if available)
        if enquiries_df is not None and len(enquiries_df) > 0:
            enquiries_df = enquiries_df[enquiries_df["REF_NO"] == account_id].copy()
            if len(enquiries_df) > 0:
                features.update(self._compute_enquiry_features(enquiries_df, as_of_date))

        # Add metadata
        features["module_version"] = self.VERSION
        features["module_name"] = self.MODULE_NAME
        features["as_of_date"] = as_of_date

        return features

    def _compute_account_count_features(self, accounts_df: pd.DataFrame) -> Dict[str, Any]:
        """Compute account count and status features"""
        features = {}

        # Total accounts
        features["bureau_total_accounts"] = len(accounts_df)

        # Account status counts
        if "ACCOUNTSTATUS" in accounts_df.columns:
            status_counts = accounts_df["ACCOUNTSTATUS"].value_counts()

            # Active accounts
            active_statuses = ["ACTIVE", "CURRENT", "OPEN"]
            features["bureau_active_accounts"] = sum(
                status_counts.get(status, 0) for status in active_statuses
            )

            # Closed accounts
            closed_statuses = ["CLOSED", "SETTLED", "PAID"]
            features["bureau_closed_accounts"] = sum(
                status_counts.get(status, 0) for status in closed_statuses
            )

            # Active ratio
            if features["bureau_total_accounts"] > 0:
                features["bureau_active_ratio"] = (
                    features["bureau_active_accounts"] / features["bureau_total_accounts"]
                )
            else:
                features["bureau_active_ratio"] = np.nan

        # Credit type flag counts
        if "CREDITTYPEFLAG" in accounts_df.columns:
            credit_type_counts = accounts_df["CREDITTYPEFLAG"].value_counts()
            features["bureau_secured_accounts"] = credit_type_counts.get("SECURED", 0)
            features["bureau_unsecured_accounts"] = credit_type_counts.get("UNSECURED", 0)

        return features

    def _compute_account_mix_features(self, accounts_df: pd.DataFrame) -> Dict[str, Any]:
        """Compute account type mix and diversity features"""
        features = {}

        if "ACCOUNTTYPE" in accounts_df.columns:
            account_types = accounts_df["ACCOUNTTYPE"].value_counts()

            # Common account types
            features["bureau_credit_card_accounts"] = account_types.get("CREDIT_CARD", 0)
            features["bureau_personal_loan_accounts"] = account_types.get("PERSONAL_LOAN", 0)
            features["bureau_home_loan_accounts"] = account_types.get("HOME_LOAN", 0)
            features["bureau_auto_loan_accounts"] = account_types.get("AUTO_LOAN", 0)

            # Account type diversity (number of unique types)
            features["bureau_account_type_diversity"] = accounts_df["ACCOUNTTYPE"].nunique()

            # Credit card flag
            features["bureau_has_credit_card"] = (
                features["bureau_credit_card_accounts"] > 0
            )

        # Loan class mix
        if "LOANCLASS" in accounts_df.columns:
            features["bureau_loan_class_diversity"] = accounts_df["LOANCLASS"].nunique()

        return features

    def _compute_delinquency_features(self, accounts_df: pd.DataFrame) -> Dict[str, Any]:
        """Compute bureau delinquency features"""
        features = {}

        # Overdue accounts
        if "OVERDUEMONTHS" in accounts_df.columns:
            # Convert to numeric if string
            accounts_df["overdue_numeric"] = pd.to_numeric(
                accounts_df["OVERDUEMONTHS"], errors="coerce"
            )

            # Count overdue accounts
            features["bureau_overdue_accounts"] = (
                accounts_df["overdue_numeric"] > 0
            ).sum()

            # Maximum overdue months
            features["bureau_max_overdue_months"] = (
                accounts_df["overdue_numeric"].max()
                if not accounts_df["overdue_numeric"].isna().all()
                else 0
            )

            # Total overdue months across all accounts
            features["bureau_total_overdue_months"] = (
                accounts_df["overdue_numeric"].sum()
                if not accounts_df["overdue_numeric"].isna().all()
                else 0
            )

        # Past due amounts
        if "AMOUNTPASTDUE" in accounts_df.columns:
            # Convert to numeric
            accounts_df["past_due_numeric"] = pd.to_numeric(
                accounts_df["AMOUNTPASTDUE"], errors="coerce"
            )

            # Total past due
            features["bureau_total_past_due_amount"] = (
                accounts_df["past_due_numeric"].sum()
                if not accounts_df["past_due_numeric"].isna().all()
                else 0.0
            )

            # Accounts with past due amounts
            features["bureau_accounts_with_past_due"] = (
                accounts_df["past_due_numeric"] > 0
            ).sum()

        # Default status
        if "DEFAULTDATE" in accounts_df.columns:
            features["bureau_defaulted_accounts"] = (
                accounts_df["DEFAULTDATE"].notna()
            ).sum()

        # DQ indicators
        if "DQ1" in accounts_df.columns or "DQ2" in accounts_df.columns:
            dq_cols = [col for col in ["DQ1", "DQ2"] if col in accounts_df.columns]
            for col in dq_cols:
                features[f"bureau_{col.lower()}_count"] = (
                    accounts_df[col].notna()
                ).sum()

        return features

    def _compute_utilization_features(self, accounts_df: pd.DataFrame) -> Dict[str, Any]:
        """Compute credit utilization features"""
        features = {}

        if "CREDITLIMIT" in accounts_df.columns and "AMOUNTOWED" in accounts_df.columns:
            # Convert to numeric
            accounts_df["credit_limit_numeric"] = pd.to_numeric(
                accounts_df["CREDITLIMIT"], errors="coerce"
            )
            accounts_df["amount_owed_numeric"] = pd.to_numeric(
                accounts_df["AMOUNTOWED"], errors="coerce"
            )

            # Total credit limit
            features["bureau_total_credit_limit"] = (
                accounts_df["credit_limit_numeric"].sum()
                if not accounts_df["credit_limit_numeric"].isna().all()
                else 0.0
            )

            # Total amount owed
            features["bureau_total_amount_owed"] = (
                accounts_df["amount_owed_numeric"].sum()
                if not accounts_df["amount_owed_numeric"].isna().all()
                else 0.0
            )

            # Overall utilization ratio
            if features["bureau_total_credit_limit"] > 0:
                features["bureau_utilization_ratio"] = (
                    features["bureau_total_amount_owed"] /
                    features["bureau_total_credit_limit"]
                )
            else:
                features["bureau_utilization_ratio"] = np.nan

            # Maxed out accounts (utilization > 90%)
            account_utilization = (
                accounts_df["amount_owed_numeric"] /
                accounts_df["credit_limit_numeric"].replace(0, np.nan)
            )
            features["bureau_maxed_out_accounts"] = (
                account_utilization > 0.90
            ).sum()

            # High utilization accounts (> 70%)
            features["bureau_high_utilization_accounts"] = (
                account_utilization > 0.70
            ).sum()

            # Average utilization per account
            features["bureau_avg_utilization_per_account"] = (
                account_utilization.mean()
                if not account_utilization.isna().all()
                else np.nan
            )

        return features

    def _compute_account_age_features(
        self,
        accounts_df: pd.DataFrame,
        as_of_date: date
    ) -> Dict[str, Any]:
        """Compute account age and vintage features"""
        features = {}

        if "DATEACCOUNTOPENED" in accounts_df.columns:
            accounts_df["opened_date"] = pd.to_datetime(
                accounts_df["DATEACCOUNTOPENED"], errors="coerce"
            )

            # Filter valid dates
            valid_opened = accounts_df["opened_date"].dropna()

            if len(valid_opened) > 0:
                # Oldest account age (months)
                oldest_date = valid_opened.min()
                features["bureau_oldest_account_age_months"] = (
                    (pd.to_datetime(as_of_date) - oldest_date).days / 30.0
                )

                # Average account age
                account_ages = (
                    pd.to_datetime(as_of_date) - valid_opened
                ).dt.days / 30.0

                features["bureau_avg_account_age_months"] = account_ages.mean()
                features["bureau_newest_account_age_months"] = account_ages.min()

                # Recently opened accounts (last 6 months)
                recent_cutoff = pd.to_datetime(as_of_date) - pd.DateOffset(months=6)
                features["bureau_accounts_opened_6m"] = (
                    valid_opened >= recent_cutoff
                ).sum()

                # Recently opened accounts (last 12 months)
                recent_cutoff_12m = pd.to_datetime(as_of_date) - pd.DateOffset(months=12)
                features["bureau_accounts_opened_12m"] = (
                    valid_opened >= recent_cutoff_12m
                ).sum()

        # Closed account analysis
        if "DATEACCOUNTCLOSED" in accounts_df.columns:
            accounts_df["closed_date"] = pd.to_datetime(
                accounts_df["DATEACCOUNTCLOSED"], errors="coerce"
            )

            valid_closed = accounts_df["closed_date"].dropna()

            if len(valid_closed) > 0:
                # Recently closed accounts (last 6 months)
                recent_cutoff = pd.to_datetime(as_of_date) - pd.DateOffset(months=6)
                features["bureau_accounts_closed_6m"] = (
                    valid_closed >= recent_cutoff
                ).sum()

        return features

    def _compute_payment_history_features(self, accounts_df: pd.DataFrame) -> Dict[str, Any]:
        """Compute payment history features"""
        features = {}

        # Payment history strings (PAYMENTHISTORY1, PAYMENTHISTORY2)
        if "PAYMENTHISTORY1" in accounts_df.columns:
            # Each character represents a month: 0=current, 1=30dpd, 2=60dpd, etc.
            payment_histories = accounts_df["PAYMENTHISTORY1"].fillna("")

            if len(payment_histories) > 0:
                # Count of on-time payments (0s in history)
                on_time_counts = payment_histories.apply(lambda x: x.count("0") if x else 0)
                features["bureau_on_time_payment_count"] = on_time_counts.sum()

                # Count of missed payments (non-0s in history)
                missed_counts = payment_histories.apply(
                    lambda x: len([c for c in x if c not in ["0", ""] ]) if x else 0
                )
                features["bureau_missed_payment_count"] = missed_counts.sum()

                # On-time payment ratio
                total_payment_records = features["bureau_on_time_payment_count"] + features["bureau_missed_payment_count"]
                if total_payment_records > 0:
                    features["bureau_on_time_payment_ratio"] = (
                        features["bureau_on_time_payment_count"] / total_payment_records
                    )
                else:
                    features["bureau_on_time_payment_ratio"] = np.nan

        # Last payment date analysis
        if "DATEOFLASTPAYMENT" in accounts_df.columns:
            accounts_df["last_payment_date"] = pd.to_datetime(
                accounts_df["DATEOFLASTPAYMENT"], errors="coerce"
            )

            valid_last_payments = accounts_df["last_payment_date"].dropna()

            if len(valid_last_payments) > 0:
                features["bureau_most_recent_payment_date"] = valid_last_payments.max()

        return features

    def _compute_risk_indicator_features(self, accounts_df: pd.DataFrame) -> Dict[str, Any]:
        """Compute risk indicator features"""
        features = {}

        # Debt restructure flag
        if "DATEOFLASTDEBTRESTRUCTURE" in accounts_df.columns:
            accounts_df["restructure_date"] = pd.to_datetime(
                accounts_df["DATEOFLASTDEBTRESTRUCTURE"], errors="coerce"
            )

            features["bureau_accounts_restructured"] = (
                accounts_df["restructure_date"].notna()
            ).sum()

            features["bureau_has_restructured_debt"] = (
                features["bureau_accounts_restructured"] > 0
            )

        # Number of co-borrowers (joint accounts)
        if "NUMBEROFCOBORROWERS" in accounts_df.columns:
            accounts_df["coborrowers_numeric"] = pd.to_numeric(
                accounts_df["NUMBEROFCOBORROWERS"], errors="coerce"
            )

            features["bureau_total_coborrowers"] = (
                accounts_df["coborrowers_numeric"].sum()
                if not accounts_df["coborrowers_numeric"].isna().all()
                else 0
            )

            features["bureau_joint_accounts"] = (
                accounts_df["coborrowers_numeric"] > 0
            ).sum()

        # Collateral presence
        collateral_cols = [col for col in accounts_df.columns if col.startswith("COLLATERAL")]
        if collateral_cols:
            features["bureau_collateralized_accounts"] = 0
            for col in collateral_cols:
                features["bureau_collateralized_accounts"] += (
                    accounts_df[col].notna()
                ).sum()

        return features

    def _compute_enquiry_features(
        self,
        enquiries_df: pd.DataFrame,
        as_of_date: date
    ) -> Dict[str, Any]:
        """Compute credit inquiry features"""
        features = {}

        if "DATEOFENQUIRY" in enquiries_df.columns:
            enquiries_df["enquiry_date"] = pd.to_datetime(
                enquiries_df["DATEOFENQUIRY"], errors="coerce"
            )

            # Filter to as_of_date (point-in-time safety)
            enquiries_df = enquiries_df[
                enquiries_df["enquiry_date"] <= pd.to_datetime(as_of_date)
            ]

            if len(enquiries_df) > 0:
                # Total enquiries
                features["bureau_total_enquiries"] = len(enquiries_df)

                # Enquiries in last 3 months
                cutoff_3m = pd.to_datetime(as_of_date) - pd.DateOffset(months=3)
                features["bureau_enquiries_3m"] = (
                    enquiries_df["enquiry_date"] >= cutoff_3m
                ).sum()

                # Enquiries in last 6 months
                cutoff_6m = pd.to_datetime(as_of_date) - pd.DateOffset(months=6)
                features["bureau_enquiries_6m"] = (
                    enquiries_df["enquiry_date"] >= cutoff_6m
                ).sum()

                # Enquiries in last 12 months
                cutoff_12m = pd.to_datetime(as_of_date) - pd.DateOffset(months=12)
                features["bureau_enquiries_12m"] = (
                    enquiries_df["enquiry_date"] >= cutoff_12m
                ).sum()

                # Days since last enquiry
                most_recent_enquiry = enquiries_df["enquiry_date"].max()
                features["bureau_days_since_last_enquiry"] = (
                    (pd.to_datetime(as_of_date) - most_recent_enquiry).days
                )

                # Enquiry purpose breakdown
                if "ENQUIRYPURPOSE" in enquiries_df.columns:
                    purpose_counts = enquiries_df["ENQUIRYPURPOSE"].value_counts()

                    # Common purposes
                    features["bureau_enquiries_credit_card"] = purpose_counts.get("CREDIT_CARD", 0)
                    features["bureau_enquiries_personal_loan"] = purpose_counts.get("PERSONAL_LOAN", 0)
                    features["bureau_enquiries_home_loan"] = purpose_counts.get("HOME_LOAN", 0)

                # Total enquiry amount
                if "ENQUIRYAMOUNT" in enquiries_df.columns:
                    enquiries_df["enquiry_amount_numeric"] = pd.to_numeric(
                        enquiries_df["ENQUIRYAMOUNT"], errors="coerce"
                    )

                    features["bureau_total_enquiry_amount"] = (
                        enquiries_df["enquiry_amount_numeric"].sum()
                        if not enquiries_df["enquiry_amount_numeric"].isna().all()
                        else 0.0
                    )

                    # Average enquiry amount
                    features["bureau_avg_enquiry_amount"] = (
                        enquiries_df["enquiry_amount_numeric"].mean()
                        if not enquiries_df["enquiry_amount_numeric"].isna().all()
                        else np.nan
                    )

        return features

    def _get_null_features(self) -> Dict[str, Any]:
        """Return null/default values for all bureau features"""
        return {
            # Account counts
            "bureau_total_accounts": 0,
            "bureau_active_accounts": 0,
            "bureau_closed_accounts": 0,
            "bureau_active_ratio": np.nan,
            "bureau_secured_accounts": 0,
            "bureau_unsecured_accounts": 0,

            # Account mix
            "bureau_credit_card_accounts": 0,
            "bureau_personal_loan_accounts": 0,
            "bureau_home_loan_accounts": 0,
            "bureau_auto_loan_accounts": 0,
            "bureau_account_type_diversity": 0,
            "bureau_has_credit_card": False,
            "bureau_loan_class_diversity": 0,

            # Delinquency
            "bureau_overdue_accounts": 0,
            "bureau_max_overdue_months": 0,
            "bureau_total_overdue_months": 0,
            "bureau_total_past_due_amount": 0.0,
            "bureau_accounts_with_past_due": 0,
            "bureau_defaulted_accounts": 0,

            # Utilization
            "bureau_total_credit_limit": 0.0,
            "bureau_total_amount_owed": 0.0,
            "bureau_utilization_ratio": np.nan,
            "bureau_maxed_out_accounts": 0,
            "bureau_high_utilization_accounts": 0,
            "bureau_avg_utilization_per_account": np.nan,

            # Account age
            "bureau_oldest_account_age_months": np.nan,
            "bureau_avg_account_age_months": np.nan,
            "bureau_newest_account_age_months": np.nan,
            "bureau_accounts_opened_6m": 0,
            "bureau_accounts_opened_12m": 0,
            "bureau_accounts_closed_6m": 0,

            # Payment history
            "bureau_on_time_payment_count": 0,
            "bureau_missed_payment_count": 0,
            "bureau_on_time_payment_ratio": np.nan,

            # Risk indicators
            "bureau_accounts_restructured": 0,
            "bureau_has_restructured_debt": False,
            "bureau_total_coborrowers": 0,
            "bureau_joint_accounts": 0,
            "bureau_collateralized_accounts": 0,

            # Enquiries
            "bureau_total_enquiries": 0,
            "bureau_enquiries_3m": 0,
            "bureau_enquiries_6m": 0,
            "bureau_enquiries_12m": 0,
            "bureau_days_since_last_enquiry": np.nan,
            "bureau_total_enquiry_amount": 0.0,
            "bureau_avg_enquiry_amount": np.nan,
        }

    @staticmethod
    def get_feature_metadata() -> Dict[str, Dict[str, Any]]:
        """
        Get metadata for all bureau features.

        Returns:
            Dictionary mapping feature names to metadata
        """
        return {
            # Account count features
            "bureau_total_accounts": {
                "tier": 1,
                "definition": "Total number of credit accounts in bureau",
                "feature_type": "continuous",
                "signal_direction": "higher = more credit history",
                "use_cases": ["Credit scoring", "Application decisioning"]
            },
            "bureau_active_accounts": {
                "tier": 1,
                "definition": "Number of currently active accounts",
                "feature_type": "continuous",
                "signal_direction": "varies",
                "use_cases": ["Credit scoring", "Debt capacity assessment"]
            },
            "bureau_active_ratio": {
                "tier": 1,
                "definition": "Ratio of active to total accounts",
                "feature_type": "continuous",
                "signal_direction": "higher = more active credit usage",
                "use_cases": ["Credit scoring"]
            },

            # Delinquency features
            "bureau_overdue_accounts": {
                "tier": 1,
                "definition": "Number of accounts currently overdue",
                "feature_type": "continuous",
                "signal_direction": "higher = higher risk",
                "use_cases": ["PD models", "Collections", "Early warning"]
            },
            "bureau_max_overdue_months": {
                "tier": 1,
                "definition": "Maximum overdue months across all accounts",
                "feature_type": "continuous",
                "signal_direction": "higher = higher risk",
                "use_cases": ["PD models", "Severity assessment"]
            },
            "bureau_defaulted_accounts": {
                "tier": 1,
                "definition": "Number of accounts with default status",
                "feature_type": "continuous",
                "signal_direction": "higher = very high risk",
                "use_cases": ["PD models", "Collections strategy"]
            },

            # Utilization features
            "bureau_utilization_ratio": {
                "tier": 1,
                "definition": "Total amount owed / Total credit limit",
                "feature_type": "continuous",
                "signal_direction": "higher = higher risk (debt stress)",
                "use_cases": ["PD models", "Limit management"]
            },
            "bureau_maxed_out_accounts": {
                "tier": 1,
                "definition": "Number of accounts with >90% utilization",
                "feature_type": "continuous",
                "signal_direction": "higher = higher risk",
                "use_cases": ["Early warning", "Collections"]
            },

            # Account age features
            "bureau_oldest_account_age_months": {
                "tier": 2,
                "definition": "Age of oldest credit account in months",
                "feature_type": "continuous",
                "signal_direction": "higher = more established credit history",
                "use_cases": ["Credit scoring", "New-to-credit detection"]
            },
            "bureau_accounts_opened_6m": {
                "tier": 2,
                "definition": "Number of accounts opened in last 6 months",
                "feature_type": "continuous",
                "signal_direction": "higher = credit seeking behavior",
                "use_cases": ["Early warning", "Fraud detection"]
            },

            # Payment history features
            "bureau_on_time_payment_ratio": {
                "tier": 1,
                "definition": "Ratio of on-time payments to total payments",
                "feature_type": "continuous",
                "signal_direction": "higher = better payment behavior",
                "use_cases": ["PD models", "Credit scoring"]
            },

            # Enquiry features
            "bureau_enquiries_6m": {
                "tier": 2,
                "definition": "Number of hard inquiries in last 6 months",
                "feature_type": "continuous",
                "signal_direction": "higher = credit hungry (risk signal)",
                "use_cases": ["Early warning", "Application fraud"]
            },

            # Risk indicators
            "bureau_has_restructured_debt": {
                "tier": 1,
                "definition": "Has any restructured debt accounts",
                "feature_type": "binary",
                "signal_direction": "True = higher risk",
                "use_cases": ["PD models", "Collections"]
            }
        }


# Example usage and testing
if __name__ == "__main__":
    print("\n" + "="*80)
    print("BUREAU FEATURE ENGINE - EXAMPLE")
    print("="*80)

    # Create sample bureau data
    sample_accounts = pd.DataFrame({
        "REF_NO": ["CUST001"] * 5,
        "ASOFDATE": ["2024-01-31"] * 5,
        "ACCOUNTTYPE": ["CREDIT_CARD", "PERSONAL_LOAN", "HOME_LOAN", "CREDIT_CARD", "AUTO_LOAN"],
        "ACCOUNTSTATUS": ["ACTIVE", "ACTIVE", "ACTIVE", "CLOSED", "ACTIVE"],
        "CREDITLIMIT": [50000, 200000, 5000000, 30000, 800000],
        "AMOUNTOWED": [25000, 150000, 4500000, 0, 600000],
        "OVERDUEMONTHS": ["0", "0", "0", "0", "2"],
        "DATEACCOUNTOPENED": ["2020-01-15", "2021-06-01", "2019-03-20", "2022-01-01", "2023-05-15"],
        "PAYMENTHISTORY1": ["000000000000", "000000000000", "000000000000", "000000", "000000120000"]
    })

    sample_enquiries = pd.DataFrame({
        "REF_NO": ["CUST001"] * 3,
        "DATEOFENQUIRY": ["2023-10-15", "2023-11-20", "2024-01-10"],
        "ENQUIRYPURPOSE": ["CREDIT_CARD", "PERSONAL_LOAN", "AUTO_LOAN"],
        "ENQUIRYAMOUNT": [50000, 200000, 800000]
    })

    # Initialize engine
    engine = BureauFeatureEngine()

    # Compute features
    features = engine.compute_features(
        account_history={
            "bureau_accounts": sample_accounts,
            "bureau_enquiries": sample_enquiries
        },
        account_id="CUST001",
        as_of_date=date(2024, 1, 31)
    )

    print("\nBureau Features Computed:")
    print("-" * 80)
    for key, value in sorted(features.items()):
        if key not in ["module_version", "module_name", "as_of_date"]:
            print(f"  {key:50s} : {value}")

    print(f"\nTotal bureau features: {len([k for k in features.keys() if not k.startswith('module')])}")

    print("\n" + "="*80)
    print("BUREAU FEATURE SUMMARY:")
    print("="*80)
    print(f"Total Accounts: {features['bureau_total_accounts']}")
    print(f"Active Accounts: {features['bureau_active_accounts']}")
    print(f"Overdue Accounts: {features['bureau_overdue_accounts']}")
    print(f"Utilization Ratio: {features['bureau_utilization_ratio']:.2%}")
    print(f"Enquiries (6M): {features['bureau_enquiries_6m']}")
    print(f"On-Time Payment Ratio: {features.get('bureau_on_time_payment_ratio', 'N/A')}")
