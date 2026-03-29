"""
Data Contract
=============
Defines exactly what fields the Offer Engine needs,
checks completeness, and scores confidence.

Design principle: engine never breaks on missing data.
  - REQUIRED fields (4): engine returns error if missing
  - STANDARD fields (11): confirmed Q1-D available
  - BUREAU fields (7): available but sometimes null
  - TDR_HISTORY fields (4): accepted offers only

Completeness score drives confidence level:
  90-100%  → HIGH
  70-89%   → MEDIUM
  <70%     → LOW
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# FIELD REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

# (field_name, weight_in_completeness_score, description)
REQUIRED_FIELDS: List[Tuple[str, float, str]] = [
    ("account_id",      0.0,  "Account identifier"),
    ("days_past_due",   0.0,  "Current DPD"),
    ("balance",         0.0,  "Current outstanding balance"),
    ("stage",           0.0,  "SM | NPL | CHARGEOFF"),
]

STANDARD_FIELDS: List[Tuple[str, float, str]] = [
    ("last_payment_date",    10.0, "Date of last payment"),
    ("last_payment_amount",  10.0, "Amount of last payment"),
    ("payment_count_12m",    10.0, "Number of payments in last 12 months"),
    ("total_paid_12m",       10.0, "Total amount paid in last 12 months"),
    ("outbound_calls_made",   8.0, "Outbound calls made to customer"),
    ("calls_connected",       8.0, "Calls where customer answered"),
    ("sms_sent",              6.0, "SMS messages sent"),
    ("sms_responded",         6.0, "SMS responses received"),
    ("last_contact_date",     8.0, "Date of last contact attempt"),
    ("last_contact_outcome",  7.0, "Last contact result (answered/no_answer/promise/refused)"),
    ("ptp_made",              7.0, "Promise-to-pay commitments made"),
    ("ptp_kept",              7.0, "Promise-to-pay commitments honoured"),
    ("total_contacts_90d",    3.0, "Total contact attempts in 90 days"),
]

BUREAU_FIELDS: List[Tuple[str, float, str]] = [
    ("bureau_total_outstanding",  5.0, "Total outstanding across all lenders"),
    ("bureau_monthly_instalment", 5.0, "Total monthly instalment all lenders"),
    ("bureau_secured_loan_flag",  5.0, "Has secured loan (mortgage/auto)"),
    ("bureau_secured_outstanding",3.0, "Outstanding on secured loans"),
    ("bureau_new_loan_12m",       3.0, "New loan taken in last 12 months"),
    ("bureau_active_loan_count",  2.0, "Number of active loans"),
    ("bureau_delinquent_other",   2.0, "Delinquent at other lenders"),
]

TDR_HISTORY_FIELDS: List[Tuple[str, float, str]] = [
    ("tdr_date",           3.0, "Date of TDR (accepted offers only)"),
    ("tdr_type",           3.0, "Offer type accepted"),
    ("tdr_new_loan_amount",3.0, "New loan amount after TDR"),
    ("tdr_tenor_months",   3.0, "Tenor of accepted TDR"),
]

ALL_OPTIONAL_FIELDS = STANDARD_FIELDS + BUREAU_FIELDS + TDR_HISTORY_FIELDS
TOTAL_OPTIONAL_WEIGHT = sum(w for _, w, _ in ALL_OPTIONAL_FIELDS)


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DataQuality:
    """Completeness assessment for one account row."""
    account_id: str
    completeness_pct: float
    confidence_level: str           # HIGH | MEDIUM | LOW
    missing_fields: List[str]
    missing_standard: List[str]     # standard fields missing
    missing_bureau: List[str]       # bureau fields missing
    has_tdr_history: bool
    derived_fields: List[str]       # fields we computed (e.g. haircut_pct)
    warnings: List[str]


# ─────────────────────────────────────────────────────────────────────────────
# CONTRACT CHECKER
# ─────────────────────────────────────────────────────────────────────────────

class DataContract:
    """
    Validates input, computes completeness, derives available fields.
    Never breaks — always returns best available data with quality flag.
    """

    # ── PUBLIC ────────────────────────────────────────────────────────────────

    def validate(self, account: pd.Series) -> DataQuality:
        """
        Check completeness and return DataQuality for one account.
        Raises ValueError only if REQUIRED fields are missing.
        """
        aid = str(account.get("account_id", "unknown"))

        # 1. Check required fields — hard stop
        missing_required = [
            f for f, _, _ in REQUIRED_FIELDS
            if f not in account.index or pd.isna(account[f])
        ]
        if missing_required:
            raise ValueError(
                f"Account {aid} missing required fields: {missing_required}"
            )

        # 2. Score optional fields
        missing_standard = []
        missing_bureau   = []
        missing_tdr      = []
        score            = 0.0

        for f, w, _ in STANDARD_FIELDS:
            if f in account.index and not _is_null(account[f]):
                score += w
            else:
                missing_standard.append(f)

        for f, w, _ in BUREAU_FIELDS:
            if f in account.index and not _is_null(account[f]):
                score += w
            else:
                missing_bureau.append(f)

        for f, w, _ in TDR_HISTORY_FIELDS:
            if f in account.index and not _is_null(account[f]):
                score += w
            else:
                missing_tdr.append(f)

        completeness = round(score / TOTAL_OPTIONAL_WEIGHT * 100, 1)

        confidence = (
            "HIGH"   if completeness >= 90 else
            "MEDIUM" if completeness >= 70 else
            "LOW"
        )

        has_tdr = len(missing_tdr) < len(TDR_HISTORY_FIELDS)

        # 3. Derived fields
        derived, warnings = self._derive_fields(account, has_tdr)

        return DataQuality(
            account_id=aid,
            completeness_pct=completeness,
            confidence_level=confidence,
            missing_fields=missing_standard + missing_bureau + missing_tdr,
            missing_standard=missing_standard,
            missing_bureau=missing_bureau,
            has_tdr_history=has_tdr,
            derived_fields=derived,
            warnings=warnings,
        )

    def enrich(self, account: pd.Series) -> pd.Series:
        """
        Add derived fields to account row.
        Returns enriched Series — original unchanged.
        """
        account = account.copy()

        # Derive TDR haircut if we have balance history and TDR data
        if (not _is_null(account.get("balance_at_tdr_date"))
                and not _is_null(account.get("tdr_new_loan_amount"))):
            bal = float(account["balance_at_tdr_date"])
            new = float(account["tdr_new_loan_amount"])
            account["tdr_haircut_pct"] = round(
                max(bal - new, 0) / max(bal, 1), 4
            )
        elif (not _is_null(account.get("balance"))
                and not _is_null(account.get("tdr_new_loan_amount"))):
            # Approximate with current balance — lower confidence
            bal = float(account["balance"])
            new = float(account["tdr_new_loan_amount"])
            account["tdr_haircut_pct"] = round(
                max(bal - new, 0) / max(bal, 1), 4
            )
            account["tdr_haircut_approximate"] = True

        # Derive TDR outcome — presence of TDR record = accepted
        if not _is_null(account.get("tdr_date")):
            account["tdr_outcome"] = 1
        else:
            # Check if ever contacted — if yes, implicit rejection
            if not _is_null(account.get("outbound_calls_made")):
                account["tdr_outcome"] = 0
            # else: never contacted — unknown

        # Derive contact response rate
        calls_made = account.get("outbound_calls_made", 0)
        calls_conn = account.get("calls_connected", 0)
        if not _is_null(calls_made) and float(calls_made) > 0:
            account["call_response_rate"] = round(
                float(calls_conn) / float(calls_made), 4
            )
        else:
            account["call_response_rate"] = 0.0

        # Derive SMS response rate
        sms_sent = account.get("sms_sent", 0)
        sms_resp = account.get("sms_responded", 0)
        if not _is_null(sms_sent) and float(sms_sent) > 0:
            account["sms_response_rate"] = round(
                float(sms_resp) / float(sms_sent), 4
            )
        else:
            account["sms_response_rate"] = 0.0

        # Derive PTP kept rate
        ptp_made = account.get("ptp_made", 0)
        ptp_kept = account.get("ptp_kept", 0)
        if not _is_null(ptp_made) and float(ptp_made) > 0:
            account["ptp_kept_rate"] = round(
                float(ptp_kept) / float(ptp_made), 4
            )
        else:
            account["ptp_kept_rate"] = 0.0

        # Derive days since last payment
        if not _is_null(account.get("last_payment_date")):
            try:
                last_pay = pd.to_datetime(account["last_payment_date"])
                account["days_since_last_payment"] = (
                    pd.Timestamp.today() - last_pay
                ).days
            except Exception:
                account["days_since_last_payment"] = 999

        # Derive days since last contact
        if not _is_null(account.get("last_contact_date")):
            try:
                last_contact = pd.to_datetime(account["last_contact_date"])
                account["days_since_last_contact"] = (
                    pd.Timestamp.today() - last_contact
                ).days
            except Exception:
                account["days_since_last_contact"] = 999

        # Derive payment regularity score (0-1)
        p12 = float(account.get("payment_count_12m", 0) or 0)
        account["payment_regularity"] = round(min(p12 / 12.0, 1.0), 4)

        # Derive willingness score (0-1) from available signals
        account["willingness_score"] = round(
            _compute_willingness(account), 4
        )

        return account

    def validate_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate all accounts. Returns quality report DataFrame."""
        rows = []
        for _, row in df.iterrows():
            try:
                q = self.validate(row)
                rows.append({
                    "account_id":        q.account_id,
                    "completeness_pct":  q.completeness_pct,
                    "confidence_level":  q.confidence_level,
                    "has_tdr_history":   q.has_tdr_history,
                    "missing_count":     len(q.missing_fields),
                    "missing_standard":  ", ".join(q.missing_standard),
                    "missing_bureau":    ", ".join(q.missing_bureau),
                    "warnings":          "; ".join(q.warnings),
                })
            except ValueError as e:
                rows.append({
                    "account_id":       str(row.get("account_id", "unknown")),
                    "completeness_pct": 0.0,
                    "confidence_level": "INVALID",
                    "missing_count":    99,
                    "warnings":         str(e),
                })
        return pd.DataFrame(rows)

    # ── PRIVATE ───────────────────────────────────────────────────────────────

    def _derive_fields(
        self, account: pd.Series, has_tdr: bool
    ) -> Tuple[List[str], List[str]]:
        derived  = []
        warnings = []

        if has_tdr:
            if not _is_null(account.get("tdr_new_loan_amount")):
                derived.append("tdr_haircut_pct")
                if _is_null(account.get("balance_at_tdr_date")):
                    warnings.append(
                        "tdr_haircut_pct approximated using current balance "
                        "(balance_at_tdr_date not provided) — use snapshot join for accuracy"
                    )

        if not _is_null(account.get("outbound_calls_made")):
            derived.append("call_response_rate")

        if not _is_null(account.get("sms_sent")):
            derived.append("sms_response_rate")

        if not _is_null(account.get("ptp_made")):
            derived.append("ptp_kept_rate")

        if _is_null(account.get("bureau_monthly_instalment")):
            warnings.append(
                "bureau_monthly_instalment missing — "
                "affordability will use payment history proxy (lower confidence)"
            )

        if _is_null(account.get("bureau_secured_loan_flag")):
            warnings.append(
                "bureau_secured_loan_flag missing — legal path cannot be assessed"
            )

        return derived, warnings


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _is_null(val) -> bool:
    if val is None:
        return True
    try:
        return pd.isna(val)
    except Exception:
        return False


def _compute_willingness(account: pd.Series) -> float:
    """
    Willingness score 0-1 from available contact + payment signals.
    Higher = more likely to engage with offer.
    """
    score = 0.0
    weight_total = 0.0

    # Payment regularity (strongest signal)
    p12 = float(account.get("payment_count_12m", 0) or 0)
    score        += (p12 / 12.0) * 30
    weight_total += 30

    # PTP kept rate
    ptp_rate = float(account.get("ptp_kept_rate", 0) or 0)
    score        += ptp_rate * 25
    weight_total += 25

    # Call response rate
    call_rate = float(account.get("call_response_rate", 0) or 0)
    score        += call_rate * 20
    weight_total += 20

    # SMS response rate
    sms_rate = float(account.get("sms_response_rate", 0) or 0)
    score        += sms_rate * 15
    weight_total += 15

    # Recency of last payment (paid recently = willing)
    days_pay = float(account.get("days_since_last_payment", 999) or 999)
    recency  = max(0, 1 - days_pay / 365)
    score        += recency * 10
    weight_total += 10

    return round(score / weight_total, 4) if weight_total > 0 else 0.0
