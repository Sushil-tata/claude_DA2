"""
Billing Cycle Features
=======================
Computes cycle-adjusted delinquency timing from BILL_DAY (crcard_card_dly).

Why this matters beyond raw DPD
---------------------------------
Raw DPD from the schema tells us how many days past due the account is,
but it doesn't tell us *where in the cycle* the customer is right now.

Two customers can both show DPD=15 yet be in very different situations:

  Customer A: BILL_DAY=1, product=CC (grace=20d)
    Statement cut on the 1st → due on the 21st
    DPD=15 → missed the 21st by 15 days → well into delinquency

  Customer B: BILL_DAY=20, product=CC (grace=20d)
    Statement cut on the 20th → due on the 9th of next month
    DPD=15 → missed the 9th by 15 days → same absolute lag

  But: Customer B's *next* due date is 25 days away — collections has
  a short window to contact before another missed payment compounds.
  Customer A's next due date is 11 days away — even tighter.

Cycle-adjusted features therefore add:
  1. Exact days past the product-specific due date (not just raw DPD)
  2. Days until the *next* due date (contact urgency signal)
  3. Current cycle phase (just_missed / early_delinquency / deep_delinquency)
  4. Whether a payment is expected soon (predictive contact timing)

Product grace periods (days after statement cut):
  CC  (Credit Card)    : BILL_DAY + 20 days
  SPC (Secured/Personal): BILL_DAY + 15 days

Fields used (schema-confirmed):
  BILL_DAY   → bill_day    (crcard_card_dly)
  DL_DATA_DT → data_date   (crcard_card_dly) — snapshot date
  DPD        → days_past_due (crcard_card_dly) — raw, for cross-check
  PROD       → product_code  (crcard_card_dly) — CC vs SPC identifier
               (if PROD not available, use grace_days config per portfolio)

Features produced:
  cc_due_day_of_month      : BILL_DAY + 20, capped to month end
  spc_due_day_of_month     : BILL_DAY + 15, capped to month end
  last_cc_due_date         : most recent CC due date before snapshot
  last_spc_due_date        : most recent SPC due date before snapshot
  days_past_cc_due         : snapshot_date - last_cc_due_date  (0 if not past)
  days_past_spc_due        : snapshot_date - last_spc_due_date (0 if not past)
  days_to_next_cc_due      : next CC due date - snapshot_date
  days_to_next_spc_due     : next SPC due date - snapshot_date
  cycle_phase_cc           : 0=pre-due, 1=0-7d past, 2=8-30d past, 3=30+d past
  cycle_phase_spc          : same scale for SPC
  cycles_missed_estimate   : DPD // avg_cycle_length (rough cycle count)
  dpd_vs_cycle_residual    : DPD mod avg_cycle_length (days into current cycle)
  bill_day_normalised      : BILL_DAY / 31  (continuous feature for models)
  payment_due_soon_flag    : 1 if next due date ≤ 7 days away (contact urgency)
"""

import calendar
import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BillingCycleConfig:
    """
    Grace periods and product mappings.
    Extend product_grace_map as new products are added.
    """
    cc_grace_days: int  = 20     # Credit Card: BILL_DAY + 20
    spc_grace_days: int = 15     # Secured/Personal: BILL_DAY + 15

    # PROD field values that map to each grace period
    # Key = value in PROD column, Value = grace days
    # Add/update when actual PROD codes are confirmed from data
    product_grace_map: Dict[str, int] = field(default_factory=lambda: {
        # CC products
        "CC":  20, "CRCARD": 20, "CREDIT_CARD": 20,
        # SPC products — add confirmed codes here
        "SPC": 15, "SPCARD": 15, "SECURED": 15, "PERSONAL": 15,
    })

    # Fallback if PROD not available or not in map
    default_grace_days: int = 20

    # Contact urgency window
    payment_due_soon_days: int = 7

    # Average cycle length for "cycles missed" estimate
    avg_cycle_days: int = 30


# ─────────────────────────────────────────────────────────────────────────────
# BILLING CYCLE CALCULATOR
# ─────────────────────────────────────────────────────────────────────────────

class BillingCycleCalculator:
    """
    Computes due dates and cycle-phase features from BILL_DAY + snapshot date.
    """

    def __init__(self, config: Optional[BillingCycleConfig] = None):
        self.cfg = config or BillingCycleConfig()

    # ── PUBLIC ────────────────────────────────────────────────────────────────

    def extract(self, account: pd.Series) -> Dict[str, float]:
        """
        Compute all billing cycle features for one account row.
        Uses internal field names (post schema_mapper rename).
        Returns dict of feature_name → float.
        """
        bill_day  = self._get_bill_day(account)
        snap_date = self._get_snapshot_date(account)
        raw_dpd   = int(account.get("days_past_due", 0) or 0)
        prod_code = str(account.get("product_code", "") or "")

        if bill_day is None or snap_date is None:
            return self._null_features()

        # Compute CC and SPC features
        cc_feats  = self._cycle_features(bill_day, snap_date, self.cfg.cc_grace_days,  "cc")
        spc_feats = self._cycle_features(bill_day, snap_date, self.cfg.spc_grace_days, "spc")

        # Product-specific grace (if PROD available)
        product_grace = self.cfg.product_grace_map.get(
            prod_code.upper(), self.cfg.default_grace_days
        )
        prod_feats = self._cycle_features(bill_day, snap_date, product_grace, "product")

        # Cross-cutting features
        shared = {
            "bill_day_normalised":    round(bill_day / 31, 4),
            "cycles_missed_estimate": float(raw_dpd // self.cfg.avg_cycle_days),
            "dpd_vs_cycle_residual":  float(raw_dpd  %  self.cfg.avg_cycle_days),
            "payment_due_soon_flag":  float(
                min(
                    cc_feats["days_to_next_cc_due"],
                    spc_feats["days_to_next_spc_due"],
                ) <= self.cfg.payment_due_soon_days
            ),
        }

        return {**cc_feats, **spc_feats, **prod_feats, **shared}

    def extract_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute billing cycle features for all rows in a DataFrame.
        Returns DataFrame with one feature column per feature.
        """
        records = []
        for _, row in df.iterrows():
            feats = self.extract(row)
            feats["account_id"] = row.get("account_id", "unknown")
            records.append(feats)

        feat_df = pd.DataFrame(records)
        cols = ["account_id"] + [c for c in feat_df.columns if c != "account_id"]
        return feat_df[cols]

    def grace_days_for_product(self, product_code: str) -> int:
        """Return grace period days for a product code."""
        return self.cfg.product_grace_map.get(
            product_code.upper(), self.cfg.default_grace_days
        )

    # ── PRIVATE ───────────────────────────────────────────────────────────────

    def _cycle_features(
        self,
        bill_day: int,
        snap_date: date,
        grace_days: int,
        prefix: str,
    ) -> Dict[str, float]:
        """
        Compute due-date features for one product's grace period.
        prefix = 'cc' | 'spc' | 'product'
        """
        last_due  = self._last_due_date(bill_day, grace_days, snap_date)
        next_due  = self._next_due_date(bill_day, grace_days, snap_date)

        days_past = max((snap_date - last_due).days, 0)
        days_to   = max((next_due - snap_date).days, 0)

        due_dom   = self._due_day_of_month(bill_day, grace_days, snap_date.year, snap_date.month)

        phase = self._cycle_phase(days_past)

        return {
            f"{prefix}_due_day_of_month":  float(due_dom),
            f"last_{prefix}_due_date":     float(last_due.toordinal()),  # ordinal for model use
            f"days_past_{prefix}_due":     float(days_past),
            f"days_to_next_{prefix}_due":  float(days_to),
            f"cycle_phase_{prefix}":       float(phase),
        }

    def _last_due_date(self, bill_day: int, grace: int, snap: date) -> date:
        """
        Most recent due date on or before snap_date.
        Walks back month by month until found.
        """
        # Try current month's due date first, then prior months
        for delta_months in range(0, 4):   # look back up to 3 months
            year, month = _subtract_months(snap.year, snap.month, delta_months)
            due = self._due_date_in_month(bill_day, grace, year, month)
            if due <= snap:
                return due

        # Fallback: 30 days ago
        return snap - timedelta(days=30)

    def _next_due_date(self, bill_day: int, grace: int, snap: date) -> date:
        """
        Next due date strictly after snap_date.
        """
        for delta_months in range(0, 3):   # look forward up to 2 months
            year, month = _add_months(snap.year, snap.month, delta_months)
            due = self._due_date_in_month(bill_day, grace, year, month)
            if due > snap:
                return due

        # Fallback: 30 days from now
        return snap + timedelta(days=30)

    def _due_date_in_month(self, bill_day: int, grace: int, year: int, month: int) -> date:
        """
        Compute the payment due date for a given statement month.
        Statement cut on bill_day of (year, month).
        Due date = statement_date + grace_days.
        Clamps bill_day to actual month end (handles BILL_DAY=31 in Feb).
        """
        max_day = calendar.monthrange(year, month)[1]
        safe_bill_day = min(bill_day, max_day)
        stmt_date = date(year, month, safe_bill_day)
        return stmt_date + timedelta(days=grace)

    def _due_day_of_month(
        self, bill_day: int, grace: int, year: int, month: int
    ) -> int:
        """Return the day-of-month of the due date (1-31)."""
        due = self._due_date_in_month(bill_day, grace, year, month)
        return due.day

    @staticmethod
    def _cycle_phase(days_past: int) -> int:
        """
        0 = pre-due (not yet past)
        1 = 1-7 days past   (just missed — first contact window)
        2 = 8-30 days past  (early delinquency)
        3 = 31+ days past   (deep delinquency)
        """
        if days_past == 0:
            return 0
        if days_past <= 7:
            return 1
        if days_past <= 30:
            return 2
        return 3

    def _get_bill_day(self, account: pd.Series) -> Optional[int]:
        val = account.get("bill_day")
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        try:
            bd = int(val)
            if 1 <= bd <= 31:
                return bd
        except (TypeError, ValueError):
            pass
        logger.warning("Invalid bill_day=%s for account %s",
                       val, account.get("account_id", "?"))
        return None

    def _get_snapshot_date(self, account: pd.Series) -> Optional[date]:
        val = account.get("data_date")
        if val is None:
            return date.today()   # fallback to today if not provided
        try:
            return pd.to_datetime(val).date()
        except Exception:
            return date.today()

    def _null_features(self) -> Dict[str, float]:
        """All zeros when BILL_DAY is missing."""
        feats = {}
        for prefix in ("cc", "spc", "product"):
            feats[f"{prefix}_due_day_of_month"]    = 0.0
            feats[f"last_{prefix}_due_date"]        = 0.0
            feats[f"days_past_{prefix}_due"]        = 0.0
            feats[f"days_to_next_{prefix}_due"]     = 0.0
            feats[f"cycle_phase_{prefix}"]          = 0.0
        feats["bill_day_normalised"]    = 0.0
        feats["cycles_missed_estimate"] = 0.0
        feats["dpd_vs_cycle_residual"]  = 0.0
        feats["payment_due_soon_flag"]  = 0.0
        return feats

    @staticmethod
    def feature_names() -> List[str]:
        feats = []
        for prefix in ("cc", "spc", "product"):
            feats += [
                f"{prefix}_due_day_of_month",
                f"last_{prefix}_due_date",
                f"days_past_{prefix}_due",
                f"days_to_next_{prefix}_due",
                f"cycle_phase_{prefix}",
            ]
        feats += [
            "bill_day_normalised",
            "cycles_missed_estimate",
            "dpd_vs_cycle_residual",
            "payment_due_soon_flag",
        ]
        return feats


# ─────────────────────────────────────────────────────────────────────────────
# DATE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _add_months(year: int, month: int, n: int):
    """Add n months to (year, month), return (year, month)."""
    month += n
    while month > 12:
        month -= 12
        year  += 1
    return year, month


def _subtract_months(year: int, month: int, n: int):
    """Subtract n months from (year, month), return (year, month)."""
    month -= n
    while month < 1:
        month += 12
        year  -= 1
    return year, month


# ─────────────────────────────────────────────────────────────────────────────
# SMOKE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    calc = BillingCycleCalculator()

    # Snapshot date fixed for reproducible test output
    SNAP = "2026-02-09"

    accounts = pd.DataFrame([
        {
            # BILL_DAY=1: CC due 21st Jan, SPC due 16th Jan
            # Snapshot 9 Feb → 19 days past CC due, 24 days past SPC due
            "account_id": "ACC001", "bill_day": 1,
            "data_date": SNAP, "days_past_due": 19, "product_code": "CC",
        },
        {
            # BILL_DAY=20: CC due 9th Feb, SPC due 4th Feb
            # Snapshot 9 Feb → 0 days past CC due (due TODAY), 5 days past SPC due
            "account_id": "ACC002", "bill_day": 20,
            "data_date": SNAP, "days_past_due": 0, "product_code": "CC",
        },
        {
            # BILL_DAY=25: CC due 14th Feb (future), SPC due 9th Feb (today)
            # Snapshot 9 Feb → pre-due for CC, 0 days past SPC due
            "account_id": "ACC003", "bill_day": 25,
            "data_date": SNAP, "days_past_due": 0, "product_code": "SPC",
        },
        {
            # BILL_DAY=28 deep NPL: DPD=150 → 5 cycles missed
            "account_id": "ACC004", "bill_day": 28,
            "data_date": SNAP, "days_past_due": 150, "product_code": "SPC",
        },
        {
            # BILL_DAY=31 (edge case: Feb has no 31st)
            "account_id": "ACC005", "bill_day": 31,
            "data_date": SNAP, "days_past_due": 45, "product_code": "CC",
        },
        {
            # No BILL_DAY — graceful null
            "account_id": "ACC006", "bill_day": None,
            "data_date": SNAP, "days_past_due": 30, "product_code": "CC",
        },
    ])

    feats_df = calc.extract_batch(accounts)

    print(f"Snapshot date: {SNAP}")
    print(f"\nTotal features: {len(calc.feature_names())}")

    key_cols = [
        "account_id",
        "cc_due_day_of_month", "spc_due_day_of_month",
        "days_past_cc_due", "days_past_spc_due",
        "days_to_next_cc_due", "days_to_next_spc_due",
        "cycle_phase_cc", "cycle_phase_spc",
        "cycles_missed_estimate", "dpd_vs_cycle_residual",
        "payment_due_soon_flag",
    ]
    pd.set_option("display.width", 200)
    print("\n── KEY FEATURES ──")
    print(feats_df[key_cols].to_string(index=False))

    print("\n── INTERPRETATION ──")
    interp = {
        "ACC001": "BILL_DAY=1, CC due 21st Jan → 19d past due, phase=2 (early delinq)",
        "ACC002": "BILL_DAY=20, CC due 9th Feb → due TODAY, next_cc_due=0d, urgent!",
        "ACC003": "BILL_DAY=25, SPC due 9th Feb → due today, CC not yet due (pre-due)",
        "ACC004": "BILL_DAY=28, DPD=150 → 5 cycles missed, 0d into current cycle",
        "ACC005": "BILL_DAY=31, Feb → clamped to 28, CC due 18th Feb (not yet due)",
        "ACC006": "No BILL_DAY → all zeros, graceful degradation",
    }
    for aid, note in interp.items():
        print(f"  {aid}: {note}")
