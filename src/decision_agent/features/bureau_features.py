"""
Bureau Feature Builder
======================
Aggregates tradeline-level NCB bureau data to customer-level features
for use in the RecoveryScorecard, NBA, and AffordabilityEngine.

Bureau data structure (tradeline-level)
----------------------------------------
One row per (customer, external_account).  Fields confirmed available:

  - secured_flag        : 1 if account is secured (mortgage, car loan)
  - unsecured_flag      : 1 if unsecured (credit card, personal loan)
  - tdr_flag            : 1 if account is in debt restructure at that lender
  - chargeoff_flag      : 1 if account charged off at that lender
  - outstanding_current : outstanding balance at current bureau pull date
  - outstanding_prior   : outstanding balance at prior bureau pull date

External payment proxy
-----------------------
NCB does NOT provide payment amounts directly.
Payment is INFERRED from balance dips between two consecutive bureau pulls:

    implied_payment = max(outstanding_prior - outstanding_current, 0)

This is directionally correct but overstates payments slightly
(does not remove interest accrual from balance).  Known approximation.
Accounts in TDR or chargeoff are excluded from payment inference
(balance movements on those are restructure adjustments, not payments).

Column names
-------------
All column names below are placeholder internal names.
Actual CDX bureau table column names will be confirmed when schema arrives.
Map via schema_mapper using T6_BUREAU table alias.

Key derived features (39 total)
---------------------------------
Group A — Portfolio composition (what debts does the customer carry?)
Group B — External payment proxy (are they paying other lenders?)
Group C — Stress signals (TDR / chargeoff externally)
Group D — Selective defaulter detection (paying others, not CardX)
Group E — ATP signals (bureau-derived capacity inputs)

Usage
------
    from decision_agent.features.bureau_features import BureauFeatureBuilder, BureauConfig

    builder = BureauFeatureBuilder(BureauConfig())
    bureau_features = builder.build(
        bureau_df=ncb_tradeline_df,      # tradeline-level, one row per external account
        card_df=card_df,                 # T2 — for CardX balance comparison
        customer_id_col="m_token",       # join key between bureau and card
    )
    # bureau_features: one row per customer, 39 features

Integration
-----------
    CollectionsFeaturePipeline Stage 7: bureau features
    AffordabilityEngine: replaces Tier 2/3 with bureau-derived ATP
    RecoveryScorecard: selective_defaulter_flag → persona routing
    NPV engine: bureau_distress_score → legal path viability
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BureauConfig:
    """
    Column name mappings for bureau tradeline DataFrame.
    These are internal names post schema_mapper rename.
    Update when CDX bureau schema is confirmed.
    """

    # ── Identity ──────────────────────────────────────────────────────────────
    customer_id_col: str        = "m_token"         # join key to T1/T2
    lender_id_col: str          = "bureau_lender_id"
    account_type_col: str       = "bureau_account_type"

    # ── Account classification flags ──────────────────────────────────────────
    secured_flag_col: str       = "bureau_secured_flag"
    unsecured_flag_col: str     = "bureau_unsecured_flag"
    tdr_flag_col: str           = "bureau_tdr_flag"
    chargeoff_flag_col: str     = "bureau_chargeoff_flag"

    # ── Balance fields (two consecutive bureau pulls) ─────────────────────────
    outstanding_current_col: str = "bureau_outstanding_current"
    outstanding_prior_col: str   = "bureau_outstanding_prior"

    # ── Bureau pull date ──────────────────────────────────────────────────────
    bureau_date_col: str        = "bureau_data_date"

    # ── Thresholds ────────────────────────────────────────────────────────────
    # Minimum balance dip to count as a payment (THB)
    min_payment_threshold: float = 100.0

    # Selective defaulter: how many external accounts must be paying
    # while CardX is 90+ DPD to flag as selective defaulter
    selective_defaulter_min_paying_accounts: int = 2

    # Distress score weights
    distress_chargeoff_weight: float    = 2.0   # chargeoff is worse than TDR
    distress_tdr_weight: float          = 1.0

    # DSR assumption for secured obligations (to back-calculate implied income)
    # e.g. if secured loan = 10k/month and mortgage DSR = 35%, implied income = 28.5k
    secured_dsr_assumption: float       = 0.35
    total_unsecured_dsr_assumption: float = 0.40


# ─────────────────────────────────────────────────────────────────────────────
# BUREAU FEATURE BUILDER
# ─────────────────────────────────────────────────────────────────────────────

class BureauFeatureBuilder:
    """
    Aggregates NCB tradeline data to customer-level features.

    Handles:
    - Missing bureau (graceful zero-fill with bureau_available_flag=0)
    - Accounts with no prior balance (cannot compute payment proxy)
    - TDR / chargeoff exclusion from payment inference
    - CardX account exclusion (we only want EXTERNAL lender data)
    """

    def __init__(self, config: Optional[BureauConfig] = None):
        self.cfg = config or BureauConfig()

    def build(
        self,
        bureau_df:       pd.DataFrame,
        card_df:         pd.DataFrame,          # T2 — CardX balance for comparison
        customer_id_col: str = "m_token",       # join key on card_df side
    ) -> pd.DataFrame:
        """
        Build bureau features aggregated to one row per customer.

        Args:
            bureau_df:       Tradeline-level NCB data (one row per external account)
            card_df:         T2 card DataFrame (account_id + m_token + balance + days_past_due)
            customer_id_col: Column in card_df that maps to bureau customer identifier

        Returns:
            DataFrame: one row per customer (m_token), 39 bureau features
        """
        c = self.cfg

        if bureau_df is None or len(bureau_df) == 0:
            logger.warning("BureauFeatureBuilder: no bureau data — returning zero-filled features")
            return self._empty_bureau(card_df, customer_id_col)

        bureau = bureau_df.copy()

        # ── Numeric coercion ──────────────────────────────────────────────────
        for col in [c.outstanding_current_col, c.outstanding_prior_col]:
            if col in bureau.columns:
                bureau[col] = pd.to_numeric(bureau[col], errors="coerce").fillna(0).clip(lower=0)

        for flag_col in [c.secured_flag_col, c.unsecured_flag_col,
                         c.tdr_flag_col, c.chargeoff_flag_col]:
            if flag_col in bureau.columns:
                bureau[flag_col] = pd.to_numeric(bureau[flag_col], errors="coerce").fillna(0)

        # ── Derive implied payment from balance dip ───────────────────────────
        bureau = self._compute_implied_payment(bureau)

        # ── Aggregate to customer level ───────────────────────────────────────
        customers = bureau[c.customer_id_col].unique()
        feat_rows = []

        for cust_id in customers:
            tl = bureau[bureau[c.customer_id_col] == cust_id]
            feat_rows.append(self._aggregate_tradelines(cust_id, tl))

        bureau_feat = pd.DataFrame(feat_rows)

        # ── Join CardX balance for selective defaulter detection ──────────────
        if customer_id_col in card_df.columns:
            cardx = card_df[[customer_id_col, "balance", "days_past_due"]].copy()
            cardx = cardx.rename(columns={
                customer_id_col: c.customer_id_col,
                "balance": "cardx_balance",
                "days_past_due": "cardx_dpd",
            })
            bureau_feat = bureau_feat.merge(cardx, on=c.customer_id_col, how="left")
            bureau_feat = self._compute_selective_defaulter(bureau_feat)
            bureau_feat = self._compute_atp_signals(bureau_feat)
            bureau_feat = bureau_feat.drop(columns=["cardx_balance", "cardx_dpd"], errors="ignore")

        # ── Accounts with no bureau — add zero rows ───────────────────────────
        if customer_id_col in card_df.columns:
            all_customers = card_df[[customer_id_col]].rename(
                columns={customer_id_col: c.customer_id_col}
            ).drop_duplicates()
            bureau_feat = all_customers.merge(bureau_feat, on=c.customer_id_col, how="left")
            bureau_feat["bureau_available_flag"] = bureau_feat["bureau_available_flag"].fillna(0)
            numeric_cols = bureau_feat.select_dtypes(include="number").columns
            bureau_feat[numeric_cols] = bureau_feat[numeric_cols].fillna(0)

        logger.info(
            "BureauFeatureBuilder: %d customers, %d features",
            len(bureau_feat), len(bureau_feat.columns) - 1,
        )
        return bureau_feat.reset_index(drop=True)

    # ── PRIVATE: payment inference ────────────────────────────────────────────

    def _compute_implied_payment(self, bureau: pd.DataFrame) -> pd.DataFrame:
        """
        Infer payment from balance dip between two bureau pulls.
        Exclude TDR and chargeoff accounts — balance moves on those
        are restructure adjustments, not genuine payments.
        """
        c = self.cfg

        has_prior   = c.outstanding_prior_col   in bureau.columns
        has_current = c.outstanding_current_col in bureau.columns
        has_tdr     = c.tdr_flag_col            in bureau.columns
        has_co      = c.chargeoff_flag_col      in bureau.columns

        if has_prior and has_current:
            dip = (
                bureau[c.outstanding_prior_col] - bureau[c.outstanding_current_col]
            ).clip(lower=0)

            # Exclude TDR and chargeoff accounts from payment inference
            exclude = pd.Series(False, index=bureau.index)
            if has_tdr:
                exclude |= (bureau[c.tdr_flag_col] == 1)
            if has_co:
                exclude |= (bureau[c.chargeoff_flag_col] == 1)

            bureau["_implied_payment"] = np.where(exclude, 0.0, dip)
            bureau["_is_paying"]       = (
                bureau["_implied_payment"] >= c.min_payment_threshold
            ).astype(int)
        else:
            bureau["_implied_payment"] = 0.0
            bureau["_is_paying"]       = 0

        return bureau

    # ── PRIVATE: per-customer aggregation ─────────────────────────────────────

    def _aggregate_tradelines(self, cust_id: str, tl: pd.DataFrame) -> Dict:
        """Aggregate all tradelines for one customer into a flat feature dict."""
        c = self.cfg

        def flag_sum(col):
            return int(tl[col].sum()) if col in tl.columns else 0

        def flag_any(col):
            return int((tl[col] == 1).any()) if col in tl.columns else 0

        def bal_sum(col):
            return float(tl[col].sum()) if col in tl.columns else 0.0

        secured_tl    = tl[tl[c.secured_flag_col] == 1]   if c.secured_flag_col   in tl.columns else tl.iloc[0:0]
        unsecured_tl  = tl[tl[c.unsecured_flag_col] == 1] if c.unsecured_flag_col in tl.columns else tl.iloc[0:0]
        tdr_tl        = tl[tl[c.tdr_flag_col] == 1]       if c.tdr_flag_col       in tl.columns else tl.iloc[0:0]
        co_tl         = tl[tl[c.chargeoff_flag_col] == 1] if c.chargeoff_flag_col in tl.columns else tl.iloc[0:0]

        # Active (non-CO, non-TDR) unsecured accounts — most comparable to CardX
        active_unsecured = unsecured_tl[
            (unsecured_tl.get(c.tdr_flag_col, pd.Series(0, index=unsecured_tl.index)) == 0) &
            (unsecured_tl.get(c.chargeoff_flag_col, pd.Series(0, index=unsecured_tl.index)) == 0)
        ] if len(unsecured_tl) > 0 else tl.iloc[0:0]

        total_outstanding  = bal_sum(c.outstanding_current_col)
        secured_outstanding = bal_sum(c.outstanding_current_col) if len(secured_tl) == 0 else float(
            secured_tl[c.outstanding_current_col].sum() if c.outstanding_current_col in secured_tl.columns else 0
        )
        unsecured_outstanding = float(
            unsecured_tl[c.outstanding_current_col].sum()
            if c.outstanding_current_col in unsecured_tl.columns else 0
        )

        # Payment proxy aggregation (active accounts only)
        paying_tl = tl[tl["_is_paying"] == 1]
        total_implied_payment     = float(tl["_implied_payment"].sum())
        paying_accounts_count     = int(tl["_is_paying"].sum())
        active_unsecured_paying   = int(
            active_unsecured["_is_paying"].sum() if "_is_paying" in active_unsecured.columns else 0
        )

        # Secured implied monthly obligation (from outstanding / assumed term)
        # For secured loans, balance dip underestimates monthly payment
        # (mortgage amortises slowly). Use a floor: secured_outstanding × 0.5% / month
        # as a minimum floor for income back-calculation.
        secured_implied_monthly = max(
            float(secured_tl["_implied_payment"].sum()) if "_implied_payment" in secured_tl.columns else 0,
            secured_outstanding * 0.005,  # 0.5% of secured outstanding as floor
        )

        # Distress score: weighted sum of external stress signals
        n_chargeoff  = len(co_tl)
        n_tdr        = len(tdr_tl)
        distress_score = (
            n_chargeoff * c.distress_chargeoff_weight +
            n_tdr       * c.distress_tdr_weight
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
            "bureau_active_unsecured_accounts": len(active_unsecured),

            "bureau_total_outstanding":         round(total_outstanding, 2),
            "bureau_secured_outstanding":       round(secured_outstanding, 2),
            "bureau_unsecured_outstanding":     round(unsecured_outstanding, 2),
            "bureau_tdr_outstanding":           round(
                float(tdr_tl[c.outstanding_current_col].sum())
                if c.outstanding_current_col in tdr_tl.columns else 0, 2
            ),
            "bureau_chargeoff_outstanding":     round(
                float(co_tl[c.outstanding_current_col].sum())
                if c.outstanding_current_col in co_tl.columns else 0, 2
            ),

            "bureau_secured_flag":              int(len(secured_tl) > 0),
            "bureau_has_external_tdr":          int(len(tdr_tl) > 0),
            "bureau_has_external_chargeoff":    int(len(co_tl) > 0),
            "bureau_tdr_count":                 n_tdr,
            "bureau_chargeoff_count":           n_chargeoff,

            # ── Group B: External payment proxy ───────────────────────────────
            # (balance dip method — excludes TDR/CO accounts)
            "bureau_implied_payment_total":     round(total_implied_payment, 2),
            "bureau_paying_accounts_count":     paying_accounts_count,
            "bureau_active_unsecured_paying":   active_unsecured_paying,

            # Consistency: are they paying the same accounts consistently?
            # High std dev = sporadic. Low = habitual payer.
            "bureau_payment_consistency":       round(
                float(tl["_implied_payment"].std()) if len(tl) > 1 else 0.0, 2
            ),

            # Secured monthly obligation proxy (for ATP / DSR calc)
            "bureau_secured_implied_monthly":   round(secured_implied_monthly, 2),

            # ── Group C: Stress signals ───────────────────────────────────────
            "bureau_distress_score":            round(distress_score, 2),
            "bureau_multi_lender_stress":       int(n_chargeoff + n_tdr >= 3),
            "bureau_pct_accounts_stressed":     round(
                (n_chargeoff + n_tdr) / max(len(tl), 1), 4
            ),

            # ── Group D: Selective defaulter signals ──────────────────────────
            # (completed in _compute_selective_defaulter after CardX join)
            "bureau_paying_active_unsecured":   active_unsecured_paying,

            # ── Group E: ATP signals ──────────────────────────────────────────
            # Implied income from secured DSR back-calculation
            "bureau_income_proxy_from_secured": round(
                secured_implied_monthly / c.secured_dsr_assumption
                if secured_implied_monthly > 0 else 0.0, 2
            ),
        }

    # ── PRIVATE: selective defaulter logic ────────────────────────────────────

    def _compute_selective_defaulter(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Flag accounts where customer is paying external lenders
        but NOT paying CardX (180+ DPD).

        selective_defaulter_flag = 1 means:
          - CardX DPD >= 90
          - AND customer is paying at least 1 external account (secured OR unsecured)
          - Paying a mortgage while ignoring CardX is a clear capacity signal

        bureau_paying_accounts_count covers ALL non-TDR/non-CO accounts with
        a balance dip — secured and unsecured.
        """
        c = self.cfg

        has_dpd    = "cardx_dpd" in df.columns
        has_paying = "bureau_paying_accounts_count" in df.columns

        if has_dpd and has_paying:
            df["selective_defaulter_flag"] = (
                (df["cardx_dpd"] >= 90) &
                (df["bureau_paying_accounts_count"] >= c.selective_defaulter_min_paying_accounts)
            ).astype(int)
        else:
            df["selective_defaulter_flag"] = 0

        return df

    # ── PRIVATE: ATP signals ──────────────────────────────────────────────────

    def _compute_atp_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute bureau-derived ATP (Ability To Pay) inputs.
        These feed into AffordabilityEngine Tier 2 and 3.
        """
        c = self.cfg

        # CardX share of total external debt
        if "cardx_balance" in df.columns and "bureau_total_outstanding" in df.columns:
            total_with_cardx = df["bureau_total_outstanding"] + df["cardx_balance"].fillna(0)
            df["bureau_cardx_share_of_wallet"] = (
                df["cardx_balance"].fillna(0) / total_with_cardx.replace(0, np.nan)
            ).fillna(0).clip(0, 1).round(4)
        else:
            df["bureau_cardx_share_of_wallet"] = 0.0

        # Available capacity for CardX: implied payment on active external accounts
        # scaled by CardX share of wallet
        if "bureau_implied_payment_total" in df.columns and "bureau_cardx_share_of_wallet" in df.columns:
            df["bureau_atp_proxy_monthly"] = (
                df["bureau_implied_payment_total"] * df["bureau_cardx_share_of_wallet"]
            ).round(2)
        else:
            df["bureau_atp_proxy_monthly"] = 0.0

        # Implied income from secured back-calculation
        # Income ≈ secured_monthly_payment / secured_DSR_assumption
        # This only works if customer has a secured loan (mortgage/car)
        if "bureau_secured_implied_monthly" in df.columns:
            df["bureau_income_from_secured_dsr"] = (
                df["bureau_secured_implied_monthly"] / c.secured_dsr_assumption
            ).round(2)
        else:
            df["bureau_income_from_secured_dsr"] = 0.0

        # Total external obligation (TDR accounts may have restructured obligations)
        # Used to estimate remaining capacity
        if "bureau_implied_payment_total" in df.columns and "bureau_secured_implied_monthly" in df.columns:
            df["bureau_total_external_obligation_proxy"] = (
                df["bureau_implied_payment_total"] + df["bureau_secured_implied_monthly"]
            ).round(2)
        else:
            df["bureau_total_external_obligation_proxy"] = 0.0

        return df

    # ── PRIVATE: empty bureau fallback ────────────────────────────────────────

    def _empty_bureau(
        self, card_df: pd.DataFrame, customer_id_col: str
    ) -> pd.DataFrame:
        """Return zero-filled bureau features when no bureau data available."""
        c = self.cfg

        customers = card_df[[customer_id_col]].drop_duplicates().rename(
            columns={customer_id_col: c.customer_id_col}
        )

        zero_cols = [
            "bureau_available_flag",
            "bureau_total_accounts", "bureau_secured_accounts",
            "bureau_unsecured_accounts", "bureau_tdr_accounts",
            "bureau_chargeoff_accounts", "bureau_active_unsecured_accounts",
            "bureau_total_outstanding", "bureau_secured_outstanding",
            "bureau_unsecured_outstanding", "bureau_tdr_outstanding",
            "bureau_chargeoff_outstanding",
            "bureau_secured_flag", "bureau_has_external_tdr",
            "bureau_has_external_chargeoff", "bureau_tdr_count",
            "bureau_chargeoff_count",
            "bureau_implied_payment_total", "bureau_paying_accounts_count",
            "bureau_active_unsecured_paying", "bureau_payment_consistency",
            "bureau_secured_implied_monthly",
            "bureau_distress_score", "bureau_multi_lender_stress",
            "bureau_pct_accounts_stressed",
            "bureau_paying_active_unsecured",
            "bureau_income_proxy_from_secured",
            "selective_defaulter_flag",
            "bureau_cardx_share_of_wallet",
            "bureau_atp_proxy_monthly",
            "bureau_income_from_secured_dsr",
            "bureau_total_external_obligation_proxy",
        ]

        for col in zero_cols:
            customers[col] = 0.0

        return customers


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
      If bureau_atp_proxy_monthly > 0 → replace/augment Tier 3

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
        # Replace internal obligation estimate with bureau-observed obligations
        affordability_profile.total_monthly_obligations = bureau_obligations
        notes.append(
            f"Bureau: external obligations proxy = {bureau_obligations:,.0f} THB/month "
            f"(balance-dip method)"
        )

    # ── Recompute disposable income and payment capacity ─────────────────────
    income  = affordability_profile.estimated_monthly_income
    oblig   = affordability_profile.total_monthly_obligations
    living  = income * (config.living_expense_pct if config else 0.35)

    affordability_profile.estimated_living_expenses = living
    affordability_profile.disposable_income = max(income - oblig - living, 0)
    affordability_profile.payment_capacity  = max(
        affordability_profile.disposable_income,
        float(bureau_row.get("bureau_atp_proxy_monthly", 0)),   # floor from bureau payment proxy
    )

    # ── Selective defaulter flag ──────────────────────────────────────────────
    if bureau_row.get("selective_defaulter_flag", 0) == 1:
        notes.append(
            "SELECTIVE DEFAULTER: paying >= 2 external active unsecured accounts "
            "while CardX DPD >= 90. Legal lever before settlement offer."
        )
        # Their true capacity is likely higher than our estimate
        # — they're choosing not to pay, not unable to pay
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

    affordability_profile.notes = notes
    return affordability_profile


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE COLUMN REFERENCE
# ─────────────────────────────────────────────────────────────────────────────

BUREAU_FEATURE_COLS = [
    # Availability
    "bureau_available_flag",
    # Portfolio composition (Group A)
    "bureau_total_accounts", "bureau_secured_accounts",
    "bureau_unsecured_accounts", "bureau_tdr_accounts",
    "bureau_chargeoff_accounts", "bureau_active_unsecured_accounts",
    "bureau_total_outstanding", "bureau_secured_outstanding",
    "bureau_unsecured_outstanding", "bureau_tdr_outstanding",
    "bureau_chargeoff_outstanding",
    "bureau_secured_flag", "bureau_has_external_tdr",
    "bureau_has_external_chargeoff", "bureau_tdr_count", "bureau_chargeoff_count",
    # External payment proxy (Group B)
    "bureau_implied_payment_total", "bureau_paying_accounts_count",
    "bureau_active_unsecured_paying", "bureau_payment_consistency",
    "bureau_secured_implied_monthly",
    # Stress signals (Group C)
    "bureau_distress_score", "bureau_multi_lender_stress",
    "bureau_pct_accounts_stressed",
    # Selective defaulter (Group D)
    "selective_defaulter_flag",
    # ATP signals (Group E)
    "bureau_cardx_share_of_wallet",
    "bureau_atp_proxy_monthly",
    "bureau_income_from_secured_dsr",
    "bureau_total_external_obligation_proxy",
]


# ─────────────────────────────────────────────────────────────────────────────
# SMOKE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src")
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # ── Synthetic bureau tradeline data ───────────────────────────────────────
    # One row per (customer, external_account)
    bureau_df = pd.DataFrame([
        # C001 — "Selective Defaulter": paying SCB and KBank but not CardX
        {"m_token": "TOK001", "bureau_lender_id": "SCB",
         "bureau_secured_flag": 1, "bureau_unsecured_flag": 0,
         "bureau_tdr_flag": 0, "bureau_chargeoff_flag": 0,
         "bureau_outstanding_current": 1_800_000, "bureau_outstanding_prior": 1_810_000},  # mortgage paying
        {"m_token": "TOK001", "bureau_lender_id": "KBANK",
         "bureau_secured_flag": 0, "bureau_unsecured_flag": 1,
         "bureau_tdr_flag": 0, "bureau_chargeoff_flag": 0,
         "bureau_outstanding_current": 28_000, "bureau_outstanding_prior": 30_500},        # card paying
        {"m_token": "TOK001", "bureau_lender_id": "BBL",
         "bureau_secured_flag": 0, "bureau_unsecured_flag": 1,
         "bureau_tdr_flag": 0, "bureau_chargeoff_flag": 0,
         "bureau_outstanding_current": 12_000, "bureau_outstanding_prior": 13_200},        # card paying

        # C002 — "Truly Insolvent": everything charged off or TDR everywhere
        {"m_token": "TOK002", "bureau_lender_id": "SCB",
         "bureau_secured_flag": 0, "bureau_unsecured_flag": 1,
         "bureau_tdr_flag": 0, "bureau_chargeoff_flag": 1,
         "bureau_outstanding_current": 45_000, "bureau_outstanding_prior": 45_000},        # chargeoff
        {"m_token": "TOK002", "bureau_lender_id": "KBANK",
         "bureau_secured_flag": 0, "bureau_unsecured_flag": 1,
         "bureau_tdr_flag": 1, "bureau_chargeoff_flag": 0,
         "bureau_outstanding_current": 80_000, "bureau_outstanding_prior": 80_500},        # TDR
        {"m_token": "TOK002", "bureau_lender_id": "GSB",
         "bureau_secured_flag": 0, "bureau_unsecured_flag": 1,
         "bureau_tdr_flag": 0, "bureau_chargeoff_flag": 1,
         "bureau_outstanding_current": 30_000, "bureau_outstanding_prior": 30_000},        # chargeoff

        # C003 — "Life Event": only CardX struggling, small personal loan elsewhere
        {"m_token": "TOK003", "bureau_lender_id": "BBL",
         "bureau_secured_flag": 0, "bureau_unsecured_flag": 1,
         "bureau_tdr_flag": 0, "bureau_chargeoff_flag": 0,
         "bureau_outstanding_current": 15_000, "bureau_outstanding_prior": 15_800},        # personal loan paying
    ])

    # ── Synthetic T2 (CardX) ──────────────────────────────────────────────────
    card_df = pd.DataFrame([
        {"account_id": "ACC001", "m_token": "TOK001", "balance": 120_000, "days_past_due": 180},
        {"account_id": "ACC002", "m_token": "TOK002", "balance": 45_000,  "days_past_due": 180},
        {"account_id": "ACC003", "m_token": "TOK003", "balance": 8_000,   "days_past_due": 60},
    ])

    builder = BureauFeatureBuilder(BureauConfig())
    features = builder.build(bureau_df, card_df, customer_id_col="m_token")

    sep = "=" * 65
    print(f"\n{sep}")
    print("  Bureau Features — Smoke Test")
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
    ]
    avail = [c for c in display_cols if c in features.columns]
    print(features[avail].to_string())

    print(f"\n{sep}")
    print("  Persona Classification (from bureau signals)")
    print(sep)
    for _, row in features.iterrows():
        persona = (
            "SELECTIVE DEFAULTER — legal lever first"
            if row.get("selective_defaulter_flag") == 1
            else "TRULY INSOLVENT — debt sale / deep discount"
            if row.get("bureau_distress_score", 0) >= 3
            else "LIFE EVENT / CAPACITY LIMITED — instalment plan"
        )
        print(f"  {row['m_token']}: {persona}")

    print(f"\n✓ smoke test passed")
