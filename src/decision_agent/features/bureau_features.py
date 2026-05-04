"""
Bureau Feature Builder — TUEF / NCB Schema
==========================================
Parses Thai Union Exchange Format (TUEF) NCB bureau JSON responses and
aggregates tradeline-level data to customer-level features for use in the
RecoveryScorecard, NBA, and AffordabilityEngine.

TUEF Bureau JSON Structure
--------------------------
The NCB API returns a nested dict per customer:

    {
      "nameSegment":    { name / DOB / gender (Buddhist Era dates) },
      "idSegment":      { Thai national ID },
      "addressSegment": [ ... ],
      "accountSegment": [
          {
              "memberShortName":          "SCB",
              "accountType":              "27",        # see codes below
              "accountStatus":            "10",        # see codes below
              "restructureCode":          "02",        # "02" = TDR
              "amountOwed":               "1810000",   # current outstanding (THB)
              "creditLimit":              "2000000",
              "installmentAmount":        "15000",     # contractual monthly payment
              "dateOfLastPayment":        "25681101",  # Buddhist Era YYYYMMDD
              "defaultDate":              "",
              "dateOfLastDebtRestructure":"",
              "paymentHistory1":          "000000001000...",  # 3-char codes × 24 months
              "paymentHistory2":          "000000000000...",  # next 24 months
              "hss": [                                 # monthly balance snapshots
                  { "asOfDate": "25681130",
                    "overDueMonths": "000",
                    "amountOwed":    "1810000" },
                  ...
              ]
          },
          ...
      ],
      "enquirySegment": [
          { "dateOfEnquiry": "25681115",
            "memberShortName": "KTC",
            "enquiryPurpose": "01" },
          ...
      ]
    }

Buddhist Era (BE) dates
-----------------------
All dates in TUEF use BE calendar.  CE year = BE year − 543.
Format is YYYYMMDD.  "00000000" / "" / "99999999" treated as null.

accountType codes
-----------------
    "01", "05" → Personal Loan      (unsecured)
    "21"       → Motorcycle Loan    (secured)
    "22"       → Credit Card        (unsecured)
    "27"       → Car Loan           (secured)
    "36"       → Cooperative Loan   (unsecured — co-operative/saving group)

accountStatus codes
-------------------
    "10"  → Normal / Active
    "11"  → Closed
    "15"  → Restructured (TDR variant)
    "16"  → Restructured (TDR variant)
    "20"  → Default / NPL (treated as chargeoff equivalent)
    "26"  → Special Mention TDR
    "36"  → Restructured Current
    "42"  → Closed (paid off)

restructureCode
---------------
    "02"  → Debt restructure (TDR flag)

paymentHistory codes (each 3 chars)
------------------------------------
    "000"  → Current (no overdue)
    "001"–"005" → 1–5 months past due
    "F  "  → Account closed/fully paid
    "   "  → No data for that month

hss (Historical Summary Segment)
---------------------------------
Monthly balance snapshots, most recent first.
Balance dip = amountOwed[t-1] − amountOwed[t] > 0 → inferred payment.
Excludes TDR / chargeoff accounts (balance movements = restructure adj, not payment).

External payment proxy
-----------------------
    implied_payment ≈ max(amountOwed_prior − amountOwed_current, 0)

    - Uses 2 most-recent hss entries
    - Excludes TDR and chargeoff accounts
    - Minimum threshold: 100 THB to filter rounding noise

Key derived features (47 total)
---------------------------------
Group A — Portfolio composition      (10 features)
Group B — External payment proxy     ( 8 features)
Group C — Stress signals             ( 5 features)
Group D — Selective defaulter        ( 1 feature)
Group E — ATP / affordability        ( 4 features)
Group F — Credit-seeking signals     ( 3 features)  ← new (enquiry-based)
Group G — Payment behaviour history  ( 8 features)  ← new (paymentHistory strings)
Group H — Balance & obligation       ( 8 features)

Usage
------
    from decision_agent.features.bureau_features import (
        TUEFParser, BureauFeatureBuilder, BureauConfig,
    )

    parser  = TUEFParser()
    builder = BureauFeatureBuilder()

    # Option A — parse raw TUEF JSON map (preferred)
    tuef_map = { "TOK001": tuef_json_dict, "TOK002": ... }
    bureau_features = builder.build_from_tuef(tuef_map, card_df, as_of_date="2025-11-30")

    # Option B — pre-parsed tradeline DataFrame (backward compat)
    bureau_features = builder.build(tradeline_df, card_df)
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS — TUEF CODE TABLES
# ─────────────────────────────────────────────────────────────────────────────

# accountType → (is_secured, label)
_ACCOUNT_TYPE_MAP: Dict[str, Tuple[bool, str]] = {
    "01": (False, "personal_loan"),
    "05": (False, "personal_loan_installment"),
    "21": (True,  "motorcycle_loan"),
    "22": (False, "credit_card"),
    "27": (True,  "car_loan"),
    "36": (False, "cooperative_loan"),
}

# accountStatus sets
_STATUS_CLOSED:     frozenset = frozenset({"11", "42"})
_STATUS_TDR:        frozenset = frozenset({"15", "16", "26", "36"})
_STATUS_DEFAULT:    frozenset = frozenset({"20"})
_STATUS_NORMAL:     frozenset = frozenset({"10"})

# paymentHistory code → numeric DPD bucket (0 = current)
def _ph_code_to_dpd(code3: str) -> Optional[int]:
    """Convert 3-char paymentHistory code to integer DPD months (None = no data)."""
    code3 = code3.strip()
    if not code3 or code3 in ("", "F", "f"):
        return None       # closed / no data — don't count in DPD stats
    try:
        return int(code3)
    except ValueError:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# TUEF PARSER
# ─────────────────────────────────────────────────────────────────────────────

class TUEFParser:
    """
    Converts raw TUEF NCB bureau JSON dicts to flat DataFrames.

    All TUEF dates are Buddhist Era (BE).  This parser converts them to CE.
    BE year − 543 = CE year.  "00000000", "99999999", "" → pd.NaT.
    """

    # ── Date parsing ──────────────────────────────────────────────────────────

    @staticmethod
    def parse_be_date(date_str: Optional[str]) -> Optional[pd.Timestamp]:
        """
        Parse Buddhist Era YYYYMMDD string to CE pd.Timestamp.
        Returns None for null / invalid / sentinel values.
        """
        if not date_str or not isinstance(date_str, str):
            return None
        date_str = date_str.strip()
        if len(date_str) != 8 or date_str in ("00000000", "99999999"):
            return None
        try:
            be_year  = int(date_str[:4])
            month    = int(date_str[4:6])
            day      = int(date_str[6:8])
            ce_year  = be_year - 543
            if ce_year < 1900 or month < 1 or month > 12 or day < 1 or day > 31:
                return None
            return pd.Timestamp(year=ce_year, month=month, day=day)
        except (ValueError, OverflowError):
            return None

    # ── Account classification ────────────────────────────────────────────────

    @staticmethod
    def classify_account(acct: Dict[str, Any]) -> Dict[str, Any]:
        """
        Derive boolean classification flags from TUEF accountSegment fields.

        Returns dict with keys:
            is_secured, is_unsecured, is_tdr, is_chargeoff, is_closed, is_active,
            account_type_label
        """
        acc_type   = str(acct.get("accountType",   "")).strip()
        acc_status = str(acct.get("accountStatus", "")).strip()
        restruct   = str(acct.get("restructureCode", "")).strip()

        type_info  = _ACCOUNT_TYPE_MAP.get(acc_type, (False, "unknown"))
        is_secured   = type_info[0]
        is_unsecured = not is_secured

        is_tdr       = (restruct == "02") or (acc_status in _STATUS_TDR)
        is_default   = acc_status in _STATUS_DEFAULT
        has_default_date = bool(
            TUEFParser.parse_be_date(acct.get("defaultDate")) is not None
        )
        is_chargeoff = is_default or has_default_date
        is_closed    = acc_status in _STATUS_CLOSED

        # Active = not closed AND not chargeoff
        is_active    = not is_closed and not is_chargeoff

        return {
            "is_secured":       is_secured,
            "is_unsecured":     is_unsecured,
            "is_tdr":           is_tdr,
            "is_chargeoff":     is_chargeoff,
            "is_closed":        is_closed,
            "is_active":        is_active,
            "account_type_code": acc_type,
            "account_type_label": type_info[1],
        }

    # ── HSS payment inference ─────────────────────────────────────────────────

    @staticmethod
    def extract_hss_payment(
        hss: List[Dict[str, Any]],
        is_tdr: bool,
        is_chargeoff: bool,
        min_threshold: float = 100.0,
    ) -> Dict[str, float]:
        """
        Infer monthly payment from the 2 most-recent HSS entries (balance dip).

        Logic:
        - Sort hss by asOfDate descending (most recent first)
        - implied_payment = max(amountOwed[older] − amountOwed[newer], 0)
        - Exclude TDR / chargeoff accounts (balance dips = restructure adj)
        - If fewer than 2 hss entries, implied_payment = 0

        Returns dict with: implied_payment, is_paying,
                           hss_current_balance, hss_prior_balance, hss_months_available
        """
        result = {
            "implied_payment":    0.0,
            "is_paying":          False,
            "hss_current_balance": 0.0,
            "hss_prior_balance":   0.0,
            "hss_months_available": len(hss),
        }

        if not hss or is_tdr or is_chargeoff:
            return result

        # Parse and sort hss entries by asOfDate (CE) descending
        parsed = []
        for entry in hss:
            dt = TUEFParser.parse_be_date(entry.get("asOfDate", ""))
            try:
                owed = float(str(entry.get("amountOwed", "0")).replace(",", ""))
            except ValueError:
                owed = 0.0
            if dt is not None:
                parsed.append((dt, owed))

        if len(parsed) < 1:
            return result

        parsed.sort(key=lambda x: x[0], reverse=True)   # newest first
        result["hss_current_balance"] = parsed[0][1]

        if len(parsed) >= 2:
            result["hss_prior_balance"] = parsed[1][1]
            dip = parsed[1][1] - parsed[0][1]          # prior − current
            payment = max(dip, 0.0)
            result["implied_payment"] = payment
            result["is_paying"]       = payment >= min_threshold

        return result

    # ── paymentHistory string parsing ─────────────────────────────────────────

    @staticmethod
    def parse_payment_history(
        ph1: str,
        ph2: str,
        lookback_months: int = 24,
    ) -> Dict[str, Any]:
        """
        Parse paymentHistory1 + paymentHistory2 strings (3-char codes each).

        paymentHistory1: most recent 24 months (most recent = first 3 chars)
        paymentHistory2: next 24 months (older)

        Returns stats over the last `lookback_months`:
            max_dpd_months, months_delinquent, months_current,
            months_no_data, streak_current (from most recent)
        """
        combined = (ph1 or "") + (ph2 or "")
        codes = [combined[i:i+3] for i in range(0, len(combined), 3)]
        codes = codes[:lookback_months]          # cap at lookback window

        dpd_values = [_ph_code_to_dpd(c) for c in codes]
        valid      = [v for v in dpd_values if v is not None]

        max_dpd       = max(valid) if valid else 0
        months_dlinq  = sum(1 for v in valid if v > 0)
        months_current = sum(1 for v in valid if v == 0)
        months_no_data = sum(1 for v in dpd_values if v is None)

        # Streak of consecutive current months from most recent
        streak = 0
        for v in dpd_values:
            if v == 0:
                streak += 1
            elif v is None:
                break           # gap = end of observable streak
            else:
                break

        return {
            "max_dpd_months":     max_dpd,
            "months_delinquent":  months_dlinq,
            "months_current":     months_current,
            "months_no_data":     months_no_data,
            "streak_current":     streak,
        }

    # ── Tradeline parser ──────────────────────────────────────────────────────

    def parse_tradelines(
        self,
        tuef_json:  Dict[str, Any],
        m_token:    str,
        as_of_date: Optional[pd.Timestamp] = None,
        cardx_lender_keywords: List[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Parse accountSegment array into a list of flat tradeline dicts.

        Excludes CardX's own accounts (identified by memberShortName keyword match).
        Each dict represents one external tradeline with all derived flags.

        Args:
            tuef_json:              Raw TUEF JSON dict for one customer
            m_token:                Customer identifier (for join)
            as_of_date:             Bureau pull date (CE) for recency calculations
            cardx_lender_keywords:  Keywords identifying CardX in memberShortName
                                    Default: ["CARDX", "CDX"]
        """
        if cardx_lender_keywords is None:
            cardx_lender_keywords = ["CARDX", "CDX"]

        accounts  = tuef_json.get("accountSegment", []) or []
        rows: List[Dict[str, Any]] = []

        for acct in accounts:
            lender = str(acct.get("memberShortName", "")).upper().strip()

            # Skip CardX's own tradeline — we only want external lenders
            if any(kw in lender for kw in cardx_lender_keywords):
                continue

            flags = self.classify_account(acct)

            # Outstanding balance
            try:
                amount_owed = float(str(acct.get("amountOwed", "0")).replace(",", ""))
            except ValueError:
                amount_owed = 0.0

            # Contractual installment (secured loan monthly obligation)
            try:
                installment = float(str(acct.get("installmentAmount", "0")).replace(",", ""))
            except ValueError:
                installment = 0.0

            # HSS payment inference
            hss  = acct.get("hss", []) or []
            pay  = self.extract_hss_payment(
                hss,
                is_tdr      = flags["is_tdr"],
                is_chargeoff = flags["is_chargeoff"],
            )

            # Payment history patterns
            ph_stats = self.parse_payment_history(
                acct.get("paymentHistory1", ""),
                acct.get("paymentHistory2", ""),
                lookback_months=24,
            )

            # Date fields (CE)
            last_pmt_dt = self.parse_be_date(acct.get("dateOfLastPayment", ""))
            default_dt  = self.parse_be_date(acct.get("defaultDate", ""))
            tdr_dt      = self.parse_be_date(acct.get("dateOfLastDebtRestructure", ""))

            # Days since last payment (relative to as_of_date if provided)
            ref_dt = as_of_date if as_of_date else pd.Timestamp.now()
            days_since_last_payment = (
                (ref_dt - last_pmt_dt).days
                if last_pmt_dt and pd.notna(last_pmt_dt) else np.nan
            )

            rows.append({
                "m_token":           m_token,
                "lender":            lender,
                # Classification
                "is_secured":        flags["is_secured"],
                "is_unsecured":      flags["is_unsecured"],
                "is_tdr":            flags["is_tdr"],
                "is_chargeoff":      flags["is_chargeoff"],
                "is_closed":         flags["is_closed"],
                "is_active":         flags["is_active"],
                "account_type_code": flags["account_type_code"],
                "account_type_label": flags["account_type_label"],
                # Balances
                "amount_owed":       amount_owed,
                "installment_amount": installment,
                # HSS payment proxy
                "implied_payment":   pay["implied_payment"],
                "is_paying":         pay["is_paying"],
                "hss_current_balance": pay["hss_current_balance"],
                "hss_prior_balance":   pay["hss_prior_balance"],
                "hss_months_available": pay["hss_months_available"],
                # Payment history stats (last 24 months)
                "ph_max_dpd_months":   ph_stats["max_dpd_months"],
                "ph_months_delinquent": ph_stats["months_delinquent"],
                "ph_months_current":   ph_stats["months_current"],
                "ph_streak_current":   ph_stats["streak_current"],
                # Dates
                "days_since_last_payment": days_since_last_payment,
                "has_default_date":  int(default_dt is not None),
                "has_tdr_date":      int(tdr_dt    is not None),
            })

        return rows

    # ── Enquiry parser ─────────────────────────────────────────────────────────

    def parse_enquiries(
        self,
        tuef_json:  Dict[str, Any],
        as_of_date: Optional[pd.Timestamp] = None,
    ) -> Dict[str, float]:
        """
        Summarise enquirySegment into credit-seeking stress signals.

        Returns:
            enquiry_count_90d:  Bureau enquiries in last 90 days
            enquiry_count_180d: Bureau enquiries in last 180 days
            enquiry_count_total: All enquiries in record
            days_since_last_enquiry: Recency of most recent enquiry
        """
        enquiries = tuef_json.get("enquirySegment", []) or []
        ref_dt    = as_of_date if as_of_date else pd.Timestamp.now()

        parsed_dates = []
        for enq in enquiries:
            dt = self.parse_be_date(enq.get("dateOfEnquiry", ""))
            if dt is not None:
                parsed_dates.append(dt)

        if not parsed_dates:
            return {
                "enquiry_count_90d":        0,
                "enquiry_count_180d":       0,
                "enquiry_count_total":      0,
                "days_since_last_enquiry":  np.nan,
            }

        parsed_dates.sort(reverse=True)

        def count_within(days: int) -> int:
            cutoff = ref_dt - pd.Timedelta(days=days)
            return sum(1 for d in parsed_dates if d >= cutoff)

        days_since = (ref_dt - parsed_dates[0]).days

        return {
            "enquiry_count_90d":       count_within(90),
            "enquiry_count_180d":      count_within(180),
            "enquiry_count_total":     len(parsed_dates),
            "days_since_last_enquiry": float(days_since),
        }


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BureauConfig:
    """
    Thresholds and business rules for bureau feature computation.
    Column-name mappings are no longer needed — TUEFParser uses internal
    standardised column names after parsing.
    """
    # ── Identity ──────────────────────────────────────────────────────────────
    customer_id_col: str = "m_token"

    # ── Payment proxy ─────────────────────────────────────────────────────────
    # Minimum balance dip (THB) to count as genuine payment
    min_payment_threshold: float = 100.0

    # ── Selective defaulter ───────────────────────────────────────────────────
    # Min paying external accounts (any type) while CardX DPD >= 90
    selective_defaulter_min_paying_accounts: int = 2

    # ── CardX lender keyword filter ───────────────────────────────────────────
    cardx_lender_keywords: List[str] = field(
        default_factory=lambda: ["CARDX", "CDX", "CARD X"]
    )

    # ── Distress score weights ────────────────────────────────────────────────
    distress_chargeoff_weight: float = 2.0
    distress_tdr_weight:       float = 1.0

    # ── DSR assumptions for income back-calculation ───────────────────────────
    # Secured loan (mortgage / car): Debt Service Ratio assumed to be 35%
    secured_dsr_assumption:          float = 0.35
    # Total unsecured DSR cap used to estimate remaining capacity
    total_unsecured_dsr_assumption:  float = 0.40

    # ── Secured monthly payment floor ─────────────────────────────────────────
    # If balance dip underestimates (e.g. slow-amortising mortgage),
    # use max(balance_dip, outstanding × floor_rate) as monthly obligation.
    secured_monthly_floor_pct: float = 0.005   # 0.5% of outstanding


# ─────────────────────────────────────────────────────────────────────────────
# BUREAU FEATURE BUILDER
# ─────────────────────────────────────────────────────────────────────────────

class BureauFeatureBuilder:
    """
    Aggregates NCB TUEF bureau data to customer-level features.

    Two entry points:
        build_from_tuef(tuef_map, card_df, as_of_date)  ← preferred, raw JSON
        build(tradeline_df, card_df)                    ← pre-parsed DataFrame (legacy)

    Handles:
    - Missing bureau: zero-fill with bureau_available_flag = 0
    - Accounts without hss: implied_payment = 0
    - TDR / chargeoff exclusion from payment inference
    - CardX account exclusion (external lenders only)
    """

    def __init__(self, config: Optional[BureauConfig] = None):
        self.cfg    = config or BureauConfig()
        self.parser = TUEFParser()

    # ── PUBLIC: preferred entry point ──────────────────────────────────────────

    def build_from_tuef(
        self,
        tuef_map:   Dict[str, Dict[str, Any]],  # {m_token: tuef_json_dict}
        card_df:    pd.DataFrame,                # T2 CardX snapshot
        as_of_date: Optional[str] = None,        # "YYYY-MM-DD"
    ) -> pd.DataFrame:
        """
        Parse TUEF JSON map and compute bureau features.

        Args:
            tuef_map:   Dict mapping m_token → raw TUEF JSON dict from NCB API
            card_df:    T2 CardX account snapshot with balance + days_past_due
            as_of_date: Bureau snapshot date ("YYYY-MM-DD") for recency calculations

        Returns:
            DataFrame: one row per customer in card_df, 47 bureau features
        """
        as_of_ts = pd.Timestamp(as_of_date) if as_of_date else None
        rows: List[Dict[str, Any]] = []

        for m_token, tuef_json in tuef_map.items():
            tradelines = self.parser.parse_tradelines(
                tuef_json, m_token, as_of_ts,
                cardx_lender_keywords=self.cfg.cardx_lender_keywords,
            )
            enq_stats  = self.parser.parse_enquiries(tuef_json, as_of_ts)

            if tradelines:
                tl_df = pd.DataFrame(tradelines)
                feat  = self._aggregate_tradelines(m_token, tl_df, enq_stats)
            else:
                feat  = self._zero_row(m_token, enq_stats)

            rows.append(feat)

        if not rows:
            return self._empty_bureau(card_df)

        bureau_feat = pd.DataFrame(rows)
        return self._finalize(bureau_feat, card_df)

    # ── PUBLIC: legacy entry point (pre-parsed DataFrame) ─────────────────────

    def build(
        self,
        bureau_df:       pd.DataFrame,
        card_df:         pd.DataFrame,
        customer_id_col: str = "m_token",
    ) -> pd.DataFrame:
        """
        Build bureau features from a pre-parsed flat tradeline DataFrame.

        Expected columns in bureau_df (internal names post-parse):
            m_token, is_secured, is_unsecured, is_tdr, is_chargeoff, is_active,
            amount_owed, implied_payment, is_paying

        This path is retained for backward-compatibility and testing with
        synthetic flat DataFrames.
        """
        if bureau_df is None or len(bureau_df) == 0:
            logger.warning("BureauFeatureBuilder.build: empty bureau — zero-fill")
            return self._empty_bureau(card_df)

        bureau = bureau_df.copy()
        cid    = customer_id_col

        # Coerce implied_payment / is_paying if not present
        if "implied_payment" not in bureau.columns:
            # Legacy flat DataFrames may have old placeholder col names
            bureau = self._coerce_legacy_cols(bureau)

        customers = bureau[cid].unique()
        rows: List[Dict[str, Any]] = []

        for cust_id in customers:
            tl_df = bureau[bureau[cid] == cust_id].copy()
            feat  = self._aggregate_tradelines(cust_id, tl_df, enq_stats={})
            rows.append(feat)

        bureau_feat = pd.DataFrame(rows)
        return self._finalize(bureau_feat, card_df, customer_id_col=customer_id_col)

    # ── PRIVATE: per-customer aggregation ────────────────────────────────────

    def _aggregate_tradelines(
        self,
        cust_id:   str,
        tl:        pd.DataFrame,
        enq_stats: Dict[str, float],
    ) -> Dict[str, Any]:
        """Aggregate all tradelines for one customer into a flat feature dict."""
        c = self.cfg

        def col_sum(col: str, df: pd.DataFrame = None) -> float:
            df = df if df is not None else tl
            return float(df[col].sum()) if col in df.columns else 0.0

        def col_any(col: str, val=True) -> int:
            return int((tl[col] == val).any()) if col in tl.columns else 0

        def safe_mean(series) -> float:
            return float(series.mean()) if len(series) > 0 else 0.0

        # Subsets
        secured_tl   = tl[tl["is_secured"]   == True]  if "is_secured"   in tl.columns else tl.iloc[0:0]
        unsecured_tl = tl[tl["is_unsecured"] == True]  if "is_unsecured" in tl.columns else tl.iloc[0:0]
        tdr_tl       = tl[tl["is_tdr"]       == True]  if "is_tdr"       in tl.columns else tl.iloc[0:0]
        co_tl        = tl[tl["is_chargeoff"] == True]  if "is_chargeoff" in tl.columns else tl.iloc[0:0]
        active_tl    = tl[tl["is_active"]    == True]  if "is_active"    in tl.columns else tl.iloc[0:0]
        paying_tl    = tl[tl["is_paying"]    == True]  if "is_paying"    in tl.columns else tl.iloc[0:0]

        active_unsecured = active_tl[active_tl["is_unsecured"] == True] if "is_unsecured" in active_tl.columns else active_tl.iloc[0:0]

        # Balances
        total_outstanding    = col_sum("amount_owed")
        secured_outstanding  = col_sum("amount_owed", secured_tl)
        unsecured_outstanding = col_sum("amount_owed", unsecured_tl)
        tdr_outstanding      = col_sum("amount_owed", tdr_tl)
        co_outstanding       = col_sum("amount_owed", co_tl)

        # Payment proxy
        total_implied_payment  = col_sum("implied_payment")
        paying_accounts_count  = int((tl["is_paying"] == True).sum()) if "is_paying" in tl.columns else 0
        active_unsec_paying    = int((active_unsecured["is_paying"] == True).sum()) if "is_paying" in active_unsecured.columns else 0

        # Secured monthly obligation (balance dip or floor)
        secured_dip = col_sum("implied_payment", secured_tl)
        secured_monthly = max(
            secured_dip,
            secured_outstanding * c.secured_monthly_floor_pct,
        )

        # Contractual installment total (from installmentAmount field)
        installment_total = col_sum("installment_amount", active_tl)

        # Payment consistency (std dev of implied payments across accounts)
        pay_series = tl["implied_payment"] if "implied_payment" in tl.columns else pd.Series(dtype=float)
        payment_std = float(pay_series.std()) if len(pay_series) > 1 else 0.0

        # Distress score
        n_co  = len(co_tl)
        n_tdr = len(tdr_tl)
        distress_score = n_co * c.distress_chargeoff_weight + n_tdr * c.distress_tdr_weight

        # Payment history stats (aggregate across tradelines — worst-case)
        ph_max_dpd       = int(tl["ph_max_dpd_months"].max())   if "ph_max_dpd_months"   in tl.columns else 0
        ph_avg_delinquent = safe_mean(tl["ph_months_delinquent"]) if "ph_months_delinquent" in tl.columns else 0.0
        ph_avg_current    = safe_mean(tl["ph_months_current"])     if "ph_months_current"   in tl.columns else 0.0
        ph_min_streak     = int(tl["ph_streak_current"].min())     if "ph_streak_current"   in tl.columns else 0

        # Days since last payment (best / most recent across accounts)
        dslp_series = tl["days_since_last_payment"].dropna() if "days_since_last_payment" in tl.columns else pd.Series(dtype=float)
        days_since_last_payment = float(dslp_series.min()) if len(dslp_series) > 0 else np.nan

        # Income proxy from secured DSR back-calculation
        income_proxy_secured = (
            secured_monthly / c.secured_dsr_assumption
            if secured_monthly > 0 else 0.0
        )

        return {
            c.customer_id_col: cust_id,
            "bureau_available_flag": 1,

            # ── Group A: Portfolio composition ────────────────────────────────
            "bureau_total_accounts":            len(tl),
            "bureau_secured_accounts":          len(secured_tl),
            "bureau_unsecured_accounts":        len(unsecured_tl),
            "bureau_tdr_accounts":              len(tdr_tl),
            "bureau_chargeoff_accounts":        len(co_tl),
            "bureau_active_accounts":           len(active_tl),
            "bureau_active_unsecured_accounts": len(active_unsecured),
            "bureau_secured_flag":              int(len(secured_tl) > 0),
            "bureau_has_external_tdr":          int(len(tdr_tl) > 0),
            "bureau_has_external_chargeoff":    int(len(co_tl) > 0),

            # ── Group B: Balances ─────────────────────────────────────────────
            "bureau_total_outstanding":         round(total_outstanding, 2),
            "bureau_secured_outstanding":       round(secured_outstanding, 2),
            "bureau_unsecured_outstanding":     round(unsecured_outstanding, 2),
            "bureau_tdr_outstanding":           round(tdr_outstanding, 2),
            "bureau_chargeoff_outstanding":     round(co_outstanding, 2),

            # ── Group C: External payment proxy (balance-dip method) ──────────
            "bureau_implied_payment_total":     round(total_implied_payment, 2),
            "bureau_paying_accounts_count":     paying_accounts_count,
            "bureau_active_unsecured_paying":   active_unsec_paying,
            "bureau_payment_consistency":       round(payment_std, 2),
            "bureau_secured_implied_monthly":   round(secured_monthly, 2),
            "bureau_installment_monthly_total": round(installment_total, 2),

            # ── Group D: Stress signals ───────────────────────────────────────
            "bureau_distress_score":            round(distress_score, 2),
            "bureau_tdr_count":                 n_tdr,
            "bureau_chargeoff_count":           n_co,
            "bureau_multi_lender_stress":       int(n_co + n_tdr >= 3),
            "bureau_pct_accounts_stressed":     round((n_co + n_tdr) / max(len(tl), 1), 4),

            # ── Group E: Selective defaulter signals ──────────────────────────
            # (completed in _finalize after CardX join)
            "bureau_paying_active_unsecured":   active_unsec_paying,

            # ── Group F: ATP / affordability signals ──────────────────────────
            "bureau_income_proxy_from_secured": round(income_proxy_secured, 2),
            # cardx_share_of_wallet and atp_proxy_monthly computed in _finalize

            # ── Group G: Credit-seeking / enquiry signals ─────────────────────
            "bureau_enquiry_count_90d":         int(enq_stats.get("enquiry_count_90d", 0)),
            "bureau_enquiry_count_180d":        int(enq_stats.get("enquiry_count_180d", 0)),
            "bureau_enquiry_count_total":       int(enq_stats.get("enquiry_count_total", 0)),
            "bureau_days_since_last_enquiry":   float(enq_stats.get("days_since_last_enquiry", np.nan)),

            # ── Group H: Payment behaviour history (paymentHistory strings) ───
            "bureau_ph_max_dpd_months":         ph_max_dpd,
            "bureau_ph_avg_months_delinquent":  round(ph_avg_delinquent, 2),
            "bureau_ph_avg_months_current":     round(ph_avg_current, 2),
            "bureau_ph_min_streak_current":     ph_min_streak,
            "bureau_days_since_last_payment":   round(days_since_last_payment, 1) if not np.isnan(days_since_last_payment) else np.nan,
        }

    # ── PRIVATE: finalize — join CardX, compute derived signals ──────────────

    def _finalize(
        self,
        bureau_feat:     pd.DataFrame,
        card_df:         pd.DataFrame,
        customer_id_col: str = "m_token",
    ) -> pd.DataFrame:
        """
        1. Left-join CardX balance + DPD onto bureau features
        2. Compute selective_defaulter_flag
        3. Compute ATP signals (share-of-wallet, atp_proxy)
        4. Fill missing customers (no bureau) with zeros
        """
        c   = self.cfg
        cid = customer_id_col

        # Join CardX data
        if cid in card_df.columns and {"balance", "days_past_due"}.issubset(card_df.columns):
            cardx = card_df[[cid, "balance", "days_past_due"]].copy()
            cardx = cardx.groupby(cid).agg(
                balance=("balance", "sum"),
                days_past_due=("days_past_due", "max"),
            ).reset_index()
            cardx = cardx.rename(columns={cid: c.customer_id_col})
            bureau_feat = bureau_feat.merge(cardx, on=c.customer_id_col, how="left")

            bureau_feat = self._compute_selective_defaulter(bureau_feat)
            bureau_feat = self._compute_atp_signals(bureau_feat)
            bureau_feat = bureau_feat.drop(columns=["balance", "days_past_due"], errors="ignore")

        # Add zero rows for customers not in bureau
        all_customers = (
            card_df[[cid]]
            .drop_duplicates()
            .rename(columns={cid: c.customer_id_col})
        )
        bureau_feat = all_customers.merge(bureau_feat, on=c.customer_id_col, how="left")

        # Fill missing bureau_available_flag
        bureau_feat["bureau_available_flag"] = bureau_feat["bureau_available_flag"].fillna(0).astype(int)

        # Zero-fill numeric columns for customers without bureau
        numeric_cols = bureau_feat.select_dtypes(include="number").columns
        bureau_feat[numeric_cols] = bureau_feat[numeric_cols].fillna(0)

        logger.info(
            "BureauFeatureBuilder: %d customers, %d features",
            len(bureau_feat),
            len(bureau_feat.columns) - 1,
        )
        return bureau_feat.reset_index(drop=True)

    # ── PRIVATE: selective defaulter ─────────────────────────────────────────

    def _compute_selective_defaulter(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Flag selective defaulters: paying >= N external accounts while CardX DPD >= 90.

        Uses bureau_paying_accounts_count (ALL non-TDR/CO paying accounts —
        secured and unsecured) so that a customer paying a mortgage + 1 card
        while ignoring CardX is correctly flagged.
        """
        c = self.cfg
        has_dpd    = "days_past_due" in df.columns
        has_paying = "bureau_paying_accounts_count" in df.columns

        if has_dpd and has_paying:
            df["selective_defaulter_flag"] = (
                (df["days_past_due"] >= 90) &
                (df["bureau_paying_accounts_count"] >= c.selective_defaulter_min_paying_accounts)
            ).astype(int)
        else:
            df["selective_defaulter_flag"] = 0

        return df

    # ── PRIVATE: ATP signals ──────────────────────────────────────────────────

    def _compute_atp_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute bureau-derived Ability-To-Pay (ATP) inputs for the
        AffordabilityEngine Tier 2 and Tier 3.
        """
        c = self.cfg

        # CardX share of total wallet (external + CardX debt)
        if "balance" in df.columns and "bureau_total_outstanding" in df.columns:
            total_with_cardx = df["bureau_total_outstanding"] + df["balance"].fillna(0)
            df["bureau_cardx_share_of_wallet"] = (
                df["balance"].fillna(0) / total_with_cardx.replace(0, np.nan)
            ).fillna(0).clip(0, 1).round(4)
        else:
            df["bureau_cardx_share_of_wallet"] = 0.0

        # ATP proxy: external implied payments × CardX share
        if "bureau_implied_payment_total" in df.columns:
            df["bureau_atp_proxy_monthly"] = (
                df["bureau_implied_payment_total"] *
                df.get("bureau_cardx_share_of_wallet", 0)
            ).round(2)
        else:
            df["bureau_atp_proxy_monthly"] = 0.0

        # Implied income from secured DSR back-calculation
        if "bureau_secured_implied_monthly" in df.columns:
            df["bureau_income_from_secured_dsr"] = (
                df["bureau_secured_implied_monthly"] / c.secured_dsr_assumption
            ).round(2)
        else:
            df["bureau_income_from_secured_dsr"] = 0.0

        # Total external obligation proxy
        df["bureau_total_external_obligation_proxy"] = (
            df.get("bureau_implied_payment_total", 0) +
            df.get("bureau_secured_implied_monthly", 0)
        ).round(2)

        return df

    # ── PRIVATE: zero-row for customer with no tradelines ────────────────────

    def _zero_row(
        self, cust_id: str, enq_stats: Dict[str, float]
    ) -> Dict[str, Any]:
        """Return a feature dict with all zeros (customer in tuef_map but no tradelines)."""
        row: Dict[str, Any] = {self.cfg.customer_id_col: cust_id, "bureau_available_flag": 1}
        for col in BUREAU_FEATURE_COLS:
            if col not in row:
                row[col] = 0
        # Populate enquiry stats even if no tradelines
        row["bureau_enquiry_count_90d"]       = int(enq_stats.get("enquiry_count_90d", 0))
        row["bureau_enquiry_count_180d"]      = int(enq_stats.get("enquiry_count_180d", 0))
        row["bureau_enquiry_count_total"]     = int(enq_stats.get("enquiry_count_total", 0))
        row["bureau_days_since_last_enquiry"] = float(enq_stats.get("days_since_last_enquiry", np.nan))
        return row

    # ── PRIVATE: empty bureau fallback ───────────────────────────────────────

    def _empty_bureau(self, card_df: pd.DataFrame) -> pd.DataFrame:
        """Return zero-filled bureau features when no bureau data available at all."""
        cid = self.cfg.customer_id_col
        customers = card_df[["m_token"]].drop_duplicates().rename(
            columns={"m_token": cid}
        ) if "m_token" in card_df.columns else pd.DataFrame({cid: []})

        for col in BUREAU_FEATURE_COLS:
            customers[col] = 0.0
        customers["bureau_days_since_last_payment"] = np.nan
        customers["bureau_days_since_last_enquiry"] = np.nan
        return customers

    # ── PRIVATE: legacy column coercion ──────────────────────────────────────

    @staticmethod
    def _coerce_legacy_cols(bureau: pd.DataFrame) -> pd.DataFrame:
        """
        Map old placeholder column names to the new internal schema
        so pre-TUEF synthetic DataFrames can still use build().

        Old names (from placeholder design)      New names (TUEF-native)
        ──────────────────────────────────────   ────────────────────────
        bureau_secured_flag                  →   is_secured
        bureau_unsecured_flag                →   is_unsecured
        bureau_tdr_flag                      →   is_tdr
        bureau_chargeoff_flag                →   is_chargeoff
        bureau_outstanding_current           →   amount_owed
        bureau_outstanding_prior             →   hss_prior_balance
        """
        rename_map = {
            "bureau_secured_flag":       "is_secured",
            "bureau_unsecured_flag":     "is_unsecured",
            "bureau_tdr_flag":           "is_tdr",
            "bureau_chargeoff_flag":     "is_chargeoff",
            "bureau_outstanding_current":"amount_owed",
            "bureau_outstanding_prior":  "hss_prior_balance",
        }
        bureau = bureau.rename(columns={k: v for k, v in rename_map.items() if k in bureau.columns})

        # Derive implied_payment from balance dip if not present
        if "implied_payment" not in bureau.columns:
            prior   = bureau.get("hss_prior_balance", pd.Series(0.0, index=bureau.index))
            current = bureau.get("amount_owed",       pd.Series(0.0, index=bureau.index))
            dip = (prior - current).clip(lower=0)

            # Exclude TDR and chargeoff from payment inference
            exclude = pd.Series(False, index=bureau.index)
            if "is_tdr" in bureau.columns:
                exclude |= (bureau["is_tdr"] == 1) | (bureau["is_tdr"] == True)
            if "is_chargeoff" in bureau.columns:
                exclude |= (bureau["is_chargeoff"] == 1) | (bureau["is_chargeoff"] == True)

            bureau["implied_payment"] = np.where(exclude, 0.0, dip)
            bureau["is_paying"]       = (bureau["implied_payment"] >= 100.0)

        # Derive is_active if not present
        if "is_active" not in bureau.columns:
            is_tdr = bureau.get("is_tdr", False)
            is_co  = bureau.get("is_chargeoff", False)
            bureau["is_active"] = ~(is_tdr.astype(bool) | is_co.astype(bool))

        # Add expected columns with defaults
        for col in ["ph_max_dpd_months", "ph_months_delinquent", "ph_months_current",
                    "ph_streak_current", "installment_amount"]:
            if col not in bureau.columns:
                bureau[col] = 0

        return bureau


# ─────────────────────────────────────────────────────────────────────────────
# AFFORDABILITY ENGINE INTEGRATION
# ─────────────────────────────────────────────────────────────────────────────

def enrich_affordability_with_bureau(
    affordability_profile,
    bureau_row: pd.Series,
    config=None,
) -> object:
    """
    Update an AffordabilityProfile with bureau-derived signals.

    Replaces Tier 2 and Tier 3 of the affordability waterfall when
    bureau data is available.  Called by AffordabilityEngine after
    standard waterfall computation.

    Priority:
      If bureau available → use bureau Tier 2/3 instead of internal proxies
      If bureau_income_from_secured_dsr > 0 → replace Tier 2
      If bureau_atp_proxy_monthly > 0 → augment Tier 3

    Returns the updated AffordabilityProfile (in-place modification).
    """
    if bureau_row.get("bureau_available_flag", 0) == 0:
        return affordability_profile   # no bureau — keep existing waterfall result

    notes = list(affordability_profile.notes)

    # ── Tier 2 replacement: secured DSR back-calculation ─────────────────────
    income_from_secured = float(bureau_row.get("bureau_income_from_secured_dsr", 0))
    if income_from_secured > 0 and affordability_profile.income_tier_used > 2:
        affordability_profile.estimated_monthly_income = income_from_secured
        affordability_profile.income_tier_used = 2
        affordability_profile.income_confidence = "MEDIUM"
        notes.append(
            f"Bureau Tier 2: income estimated from secured loan DSR back-calc "
            f"({income_from_secured:,.0f} THB/month)"
        )

    # ── Total external obligations from bureau ────────────────────────────────
    bureau_obligations = float(bureau_row.get("bureau_total_external_obligation_proxy", 0))
    if bureau_obligations > 0:
        affordability_profile.total_monthly_obligations = bureau_obligations
        notes.append(
            f"Bureau: external obligations proxy = {bureau_obligations:,.0f} THB/month "
            f"(TUEF balance-dip method)"
        )

    # ── Recompute disposable income and payment capacity ─────────────────────
    income  = affordability_profile.estimated_monthly_income
    oblig   = affordability_profile.total_monthly_obligations
    living  = income * (config.living_expense_pct if config else 0.35)

    affordability_profile.estimated_living_expenses = living
    affordability_profile.disposable_income = max(income - oblig - living, 0)
    affordability_profile.payment_capacity  = max(
        affordability_profile.disposable_income,
        float(bureau_row.get("bureau_atp_proxy_monthly", 0)),
    )

    # ── Selective defaulter flag ──────────────────────────────────────────────
    if bureau_row.get("selective_defaulter_flag", 0) == 1:
        notes.append(
            "SELECTIVE DEFAULTER: paying >= 2 external accounts "
            "while CardX DPD >= 90. Legal lever before settlement offer."
        )
        affordability_profile.income_confidence = "HIGH (selective defaulter — capacity exists)"

    # ── External TDR / chargeoff stress ──────────────────────────────────────
    if bureau_row.get("bureau_has_external_tdr", 0) == 1:
        n_tdr = int(bureau_row.get("bureau_tdr_count", 1))
        notes.append(
            f"Bureau: {n_tdr} external account(s) in TDR — "
            f"customer already restructuring elsewhere. Capacity constrained."
        )
    if bureau_row.get("bureau_has_external_chargeoff", 0) == 1:
        n_co = int(bureau_row.get("bureau_chargeoff_count", 1))
        notes.append(
            f"Bureau: {n_co} external chargeoff(s) — "
            f"systemic distress. Deep discount or debt sale path."
        )

    # ── Credit-seeking stress signal (recent enquiries) ───────────────────────
    enq_90d = int(bureau_row.get("bureau_enquiry_count_90d", 0))
    if enq_90d >= 3:
        notes.append(
            f"Bureau: {enq_90d} credit enquiries in 90 days — "
            f"credit-seeking behaviour, may indicate financial stress."
        )

    affordability_profile.notes = notes
    return affordability_profile


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE COLUMN REFERENCE
# ─────────────────────────────────────────────────────────────────────────────

BUREAU_FEATURE_COLS = [
    # Availability
    "bureau_available_flag",

    # Group A: Portfolio composition
    "bureau_total_accounts",
    "bureau_secured_accounts",
    "bureau_unsecured_accounts",
    "bureau_tdr_accounts",
    "bureau_chargeoff_accounts",
    "bureau_active_accounts",
    "bureau_active_unsecured_accounts",
    "bureau_secured_flag",
    "bureau_has_external_tdr",
    "bureau_has_external_chargeoff",

    # Group B: Balances
    "bureau_total_outstanding",
    "bureau_secured_outstanding",
    "bureau_unsecured_outstanding",
    "bureau_tdr_outstanding",
    "bureau_chargeoff_outstanding",

    # Group C: External payment proxy (TUEF balance-dip method)
    "bureau_implied_payment_total",
    "bureau_paying_accounts_count",
    "bureau_active_unsecured_paying",
    "bureau_payment_consistency",
    "bureau_secured_implied_monthly",
    "bureau_installment_monthly_total",

    # Group D: Stress signals
    "bureau_distress_score",
    "bureau_tdr_count",
    "bureau_chargeoff_count",
    "bureau_multi_lender_stress",
    "bureau_pct_accounts_stressed",

    # Group E: Selective defaulter
    "selective_defaulter_flag",
    "bureau_paying_active_unsecured",

    # Group F: ATP / affordability
    "bureau_income_proxy_from_secured",
    "bureau_cardx_share_of_wallet",
    "bureau_atp_proxy_monthly",
    "bureau_income_from_secured_dsr",
    "bureau_total_external_obligation_proxy",

    # Group G: Credit-seeking / enquiry signals  (TUEF enquirySegment)
    "bureau_enquiry_count_90d",
    "bureau_enquiry_count_180d",
    "bureau_enquiry_count_total",
    "bureau_days_since_last_enquiry",

    # Group H: Payment behaviour history  (TUEF paymentHistory strings)
    "bureau_ph_max_dpd_months",
    "bureau_ph_avg_months_delinquent",
    "bureau_ph_avg_months_current",
    "bureau_ph_min_streak_current",
    "bureau_days_since_last_payment",
]


# ─────────────────────────────────────────────────────────────────────────────
# SMOKE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # ── Synthetic TUEF JSON (realistic structure, Buddhist Era dates) ─────────
    # BE date 25681130 = 30 Nov 2025 CE   (2568 - 543 = 2025)
    # BE date 25681031 = 31 Oct 2025 CE
    # BE date 25680930 = 30 Sep 2025 CE

    tuef_map = {

        # TOK001 — "Selective Defaulter"
        # Paying SCB mortgage + KBank card while ignoring CardX
        "TOK001": {
            "nameSegment":  {"name": "SOMCHAI K", "dateOfBirth": "25421201"},
            "idSegment":    {"idNumber": "1234567890123"},
            "accountSegment": [
                {
                    "memberShortName":  "SCB",
                    "accountType":      "27",        # Car Loan (secured)
                    "accountStatus":    "10",        # Normal
                    "restructureCode":  "",
                    "amountOwed":       "180000",
                    "creditLimit":      "500000",
                    "installmentAmount": "8500",
                    "dateOfLastPayment": "25681101",  # Nov 2025 BE
                    "defaultDate":       "",
                    "dateOfLastDebtRestructure": "",
                    "paymentHistory1":  "000000000000000000000000",  # 8×3=24 chars all current
                    "paymentHistory2":  "000000000000000000000000",
                    "hss": [
                        {"asOfDate": "25681130", "overDueMonths": "000", "amountOwed": "180000"},
                        {"asOfDate": "25681031", "overDueMonths": "000", "amountOwed": "188500"},  # dip = 8500
                        {"asOfDate": "25680930", "overDueMonths": "000", "amountOwed": "197000"},
                    ],
                },
                {
                    "memberShortName":  "KBANK",
                    "accountType":      "22",        # Credit Card (unsecured)
                    "accountStatus":    "10",        # Normal
                    "restructureCode":  "",
                    "amountOwed":       "28000",
                    "creditLimit":      "50000",
                    "installmentAmount": "2500",
                    "dateOfLastPayment": "25681105",
                    "defaultDate":       "",
                    "dateOfLastDebtRestructure": "",
                    "paymentHistory1":  "000000000000000000000000",
                    "paymentHistory2":  "000000000000000000000000",
                    "hss": [
                        {"asOfDate": "25681130", "overDueMonths": "000", "amountOwed": "28000"},
                        {"asOfDate": "25681031", "overDueMonths": "000", "amountOwed": "30500"},  # dip = 2500
                    ],
                },
            ],
            "enquirySegment": [
                {"dateOfEnquiry": "25681101", "memberShortName": "BBL",    "enquiryPurpose": "01"},
                {"dateOfEnquiry": "25681015", "memberShortName": "KBANK",  "enquiryPurpose": "01"},
            ],
        },

        # TOK002 — "Truly Insolvent"
        # SCB chargeoff, KBANK TDR, GSB chargeoff — systemic distress
        "TOK002": {
            "nameSegment":  {"name": "PRIYA W", "dateOfBirth": "25450615"},
            "idSegment":    {"idNumber": "9876543210123"},
            "accountSegment": [
                {
                    "memberShortName":  "SCB",
                    "accountType":      "01",        # Personal Loan (unsecured)
                    "accountStatus":    "20",        # Default / NPL  ← chargeoff
                    "restructureCode":  "",
                    "amountOwed":       "45000",
                    "creditLimit":      "50000",
                    "installmentAmount": "0",
                    "dateOfLastPayment": "25660101",
                    "defaultDate":       "25671201",  # defaulted Dec 2024
                    "dateOfLastDebtRestructure": "",
                    "paymentHistory1":  "005005005004003002001000000000000000000000000000",
                    "paymentHistory2":  "000000000000000000000000",
                    "hss": [
                        {"asOfDate": "25681130", "overDueMonths": "005", "amountOwed": "45000"},
                        {"asOfDate": "25681031", "overDueMonths": "005", "amountOwed": "45000"},
                    ],
                },
                {
                    "memberShortName":  "KBANK",
                    "accountType":      "05",        # Personal Loan Installment (unsecured)
                    "accountStatus":    "15",        # Restructured TDR  ← TDR
                    "restructureCode":  "02",
                    "amountOwed":       "80000",
                    "creditLimit":      "100000",
                    "installmentAmount": "1500",     # TDR restructured payment
                    "dateOfLastPayment": "25680901",
                    "defaultDate":       "",
                    "dateOfLastDebtRestructure": "25680601",
                    "paymentHistory1":  "000000000000003003002001000000000000000000000000",
                    "paymentHistory2":  "000000000000000000000000",
                    "hss": [
                        {"asOfDate": "25681130", "overDueMonths": "000", "amountOwed": "80000"},
                        {"asOfDate": "25681031", "overDueMonths": "000", "amountOwed": "80500"},  # tiny dip (TDR)
                    ],
                },
                {
                    "memberShortName":  "GSB",
                    "accountType":      "22",        # Credit Card
                    "accountStatus":    "20",        # Default  ← chargeoff
                    "restructureCode":  "",
                    "amountOwed":       "30000",
                    "creditLimit":      "30000",
                    "installmentAmount": "0",
                    "dateOfLastPayment": "25651201",
                    "defaultDate":       "25671001",
                    "dateOfLastDebtRestructure": "",
                    "paymentHistory1":  "005005005005005005004003002001000000000000000000",
                    "paymentHistory2":  "000000000000000000000000",
                    "hss": [
                        {"asOfDate": "25681130", "overDueMonths": "005", "amountOwed": "30000"},
                        {"asOfDate": "25681031", "overDueMonths": "005", "amountOwed": "30000"},
                    ],
                },
            ],
            "enquirySegment": [],
        },

        # TOK003 — "Life Event / Capacity Limited"
        # Only one small personal loan at BBL, paying normally
        "TOK003": {
            "nameSegment":  {"name": "NAPHA T", "dateOfBirth": "25510320"},
            "idSegment":    {"idNumber": "1122334455667"},
            "accountSegment": [
                {
                    "memberShortName":  "BBL",
                    "accountType":      "01",        # Personal Loan (unsecured)
                    "accountStatus":    "10",        # Normal
                    "restructureCode":  "",
                    "amountOwed":       "15000",
                    "creditLimit":      "20000",
                    "installmentAmount": "800",
                    "dateOfLastPayment": "25681110",
                    "defaultDate":       "",
                    "dateOfLastDebtRestructure": "",
                    "paymentHistory1":  "000000000000000000000000",
                    "paymentHistory2":  "000000000000000000000000",
                    "hss": [
                        {"asOfDate": "25681130", "overDueMonths": "000", "amountOwed": "15000"},
                        {"asOfDate": "25681031", "overDueMonths": "000", "amountOwed": "15800"},  # dip = 800
                    ],
                },
            ],
            "enquirySegment": [
                {"dateOfEnquiry": "25681120", "memberShortName": "SCB", "enquiryPurpose": "01"},
            ],
        },
    }

    # ── Synthetic CardX T2 snapshot ───────────────────────────────────────────
    card_df = pd.DataFrame([
        {"account_id": "ACC001", "m_token": "TOK001", "balance": 120_000, "days_past_due": 180},
        {"account_id": "ACC002", "m_token": "TOK002", "balance": 45_000,  "days_past_due": 210},
        {"account_id": "ACC003", "m_token": "TOK003", "balance": 8_000,   "days_past_due": 60},
    ])

    builder  = BureauFeatureBuilder(BureauConfig())
    features = builder.build_from_tuef(tuef_map, card_df, as_of_date="2025-11-30")

    sep = "=" * 70
    print(f"\n{sep}")
    print("  Bureau Features — TUEF Smoke Test")
    print(sep)

    display_cols = [
        "m_token",
        "bureau_total_accounts", "bureau_secured_accounts",
        "bureau_chargeoff_accounts", "bureau_tdr_accounts",
        "bureau_implied_payment_total",
        "bureau_paying_accounts_count",
        "selective_defaulter_flag",
        "bureau_distress_score",
        "bureau_cardx_share_of_wallet",
        "bureau_atp_proxy_monthly",
        "bureau_income_from_secured_dsr",
        "bureau_enquiry_count_90d",
        "bureau_ph_max_dpd_months",
    ]
    avail = [c for c in display_cols if c in features.columns]
    print(features[avail].to_string(index=False))

    print(f"\n{sep}")
    print("  Persona Classification")
    print(sep)
    for _, row in features.iterrows():
        if row.get("selective_defaulter_flag") == 1:
            persona = "SELECTIVE DEFAULTER — legal lever first"
        elif row.get("bureau_distress_score", 0) >= 3:
            persona = "TRULY INSOLVENT — debt sale / deep discount"
        else:
            persona = "LIFE EVENT / CAPACITY LIMITED — instalment plan"
        print(f"  {row['m_token']}: {persona}")

    # ── Backward-compat check: legacy flat DataFrame via build() ──────────────
    print(f"\n{sep}")
    print("  Backward-Compat: legacy build() with flat tradeline DataFrame")
    print(sep)

    legacy_bureau = pd.DataFrame([
        {"m_token": "TOK001", "bureau_lender_id": "SCB",
         "bureau_secured_flag": 1, "bureau_unsecured_flag": 0,
         "bureau_tdr_flag": 0, "bureau_chargeoff_flag": 0,
         "bureau_outstanding_current": 180_000, "bureau_outstanding_prior": 188_500},
        {"m_token": "TOK001", "bureau_lender_id": "KBANK",
         "bureau_secured_flag": 0, "bureau_unsecured_flag": 1,
         "bureau_tdr_flag": 0, "bureau_chargeoff_flag": 0,
         "bureau_outstanding_current": 28_000, "bureau_outstanding_prior": 30_500},
    ])

    feat_legacy = builder.build(legacy_bureau, card_df[card_df["m_token"] == "TOK001"].copy())
    print(feat_legacy[["m_token", "bureau_implied_payment_total", "bureau_paying_accounts_count"]].to_string(index=False))

    print(f"\n✓ TUEF smoke test passed — {len(features.columns)} columns generated")
