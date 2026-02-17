"""
Paydown Curves
==============
Empirical cumulative recovery curves by persona × months_since_180dpd.

Two curve sets:
  Curve A — Natural recovery WITHOUT structured offer.
             Accounts in collections with no TDR / plan agreed.
             Source: historical cohort of 180+ DPD accounts with EFS = N.

  Curve B — Recovery WITH structured plan agreed.
             Accounts where TDR / settlement was agreed (EFS flag = Y or
             TDR bureau restructureCode = "02").
             NOTE: Curve B is conditional on a plan being AGREED — it does
             NOT embed a take-up probability.  Take-up probability is
             not used in this NPV framework.

NPV formula (replaces FICO take-up-weighted formula):
  PV_A         = Σ_t  outstanding × ΔCurve_A[t]  × discount(t)
  PV_B         = Σ_t  outstanding × ΔCurve_B[t]  × discount(t)
  concession   = principal_waived + interest_waived + charges_waived + processing
  NPV_offer    = PV_B − concession          (absolute NPV of making offer)
  NPV_baseline = PV_A                        (NPV of doing nothing)
  Incremental  = NPV_offer − NPV_baseline   (gain from offer over baseline)

Path selection:
  recommended = argmax(NPV_offer, debt_sale_npv, legal_npv, NPV_baseline)

Data requirements for live system:
  - T2/T3 collection cohort data, 180+ DPD accounts
  - EFS action code = Y flag to split Curve A vs Curve B
  - At least 12 months of outcome data per cohort
  - Cohort definition: first_180dpd_date as index, grouped monthly

MVP status:
  Curves below are INDUSTRY-CALIBRATED DEFAULTS.
  Replace with PaydownCurveEngine.fit_from_history(cohort_df) when
  historical CardX collection outcome data is available.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# Persona labels — must match RecoveryScorecard output
PERSONAS = [
    "SELECTIVE_DEFAULTER",
    "LIFE_EVENT",
    "GRADUAL_DETERIORATOR",
    "STRATEGIC_NON_PAYER",
    "TRULY_INSOLVENT",
    "UNKNOWN",
]

# months_at_180plus bucket boundaries (0 = just crossed 180 DPD)
_STALENESS_BUCKETS = [0, 6, 12, 24]   # lower bound of each bracket

# Curve checkpoints: cumulative % of outstanding recovered BY month t
_CHECKPOINTS = [3, 6, 9, 12, 18, 24]  # months of collection effort


# ─────────────────────────────────────────────────────────────────────────────
# DEFAULT CURVE DATA  (industry-calibrated, not fitted to CardX)
# ─────────────────────────────────────────────────────────────────────────────
# Structure: persona → staleness_bucket → {month: cumulative_%_recovered}
# Staleness bucket = months already spent at 180+ DPD at time of decision

# Curve A: natural recovery WITHOUT structured offer
_DEFAULT_CURVE_A: Dict[str, Dict[int, Dict[int, float]]] = {
    "SELECTIVE_DEFAULTER": {
        0:  {3: 0.050, 6: 0.120, 9: 0.180, 12: 0.240, 18: 0.320, 24: 0.380},
        6:  {3: 0.040, 6: 0.090, 9: 0.140, 12: 0.185, 18: 0.255, 24: 0.305},
        12: {3: 0.025, 6: 0.065, 9: 0.100, 12: 0.135, 18: 0.190, 24: 0.240},
        24: {3: 0.015, 6: 0.040, 9: 0.060, 12: 0.080, 18: 0.115, 24: 0.145},
    },
    "LIFE_EVENT": {
        0:  {3: 0.040, 6: 0.100, 9: 0.155, 12: 0.205, 18: 0.275, 24: 0.330},
        6:  {3: 0.030, 6: 0.075, 9: 0.115, 12: 0.155, 18: 0.210, 24: 0.260},
        12: {3: 0.020, 6: 0.050, 9: 0.080, 12: 0.110, 18: 0.155, 24: 0.200},
        24: {3: 0.012, 6: 0.030, 9: 0.048, 12: 0.065, 18: 0.093, 24: 0.120},
    },
    "GRADUAL_DETERIORATOR": {
        0:  {3: 0.020, 6: 0.055, 9: 0.085, 12: 0.115, 18: 0.165, 24: 0.215},
        6:  {3: 0.015, 6: 0.040, 9: 0.065, 12: 0.090, 18: 0.130, 24: 0.170},
        12: {3: 0.010, 6: 0.028, 9: 0.045, 12: 0.062, 18: 0.090, 24: 0.120},
        24: {3: 0.006, 6: 0.016, 9: 0.026, 12: 0.037, 18: 0.054, 24: 0.072},
    },
    "STRATEGIC_NON_PAYER": {
        0:  {3: 0.010, 6: 0.028, 9: 0.045, 12: 0.062, 18: 0.090, 24: 0.118},
        6:  {3: 0.008, 6: 0.020, 9: 0.033, 12: 0.046, 18: 0.067, 24: 0.088},
        12: {3: 0.005, 6: 0.014, 9: 0.022, 12: 0.031, 18: 0.045, 24: 0.060},
        24: {3: 0.003, 6: 0.008, 9: 0.013, 12: 0.018, 18: 0.027, 24: 0.036},
    },
    "TRULY_INSOLVENT": {
        0:  {3: 0.008, 6: 0.018, 9: 0.028, 12: 0.038, 18: 0.054, 24: 0.068},
        6:  {3: 0.006, 6: 0.013, 9: 0.020, 12: 0.028, 18: 0.040, 24: 0.051},
        12: {3: 0.004, 6: 0.009, 9: 0.014, 12: 0.019, 18: 0.028, 24: 0.036},
        24: {3: 0.002, 6: 0.005, 9: 0.008, 12: 0.011, 18: 0.016, 24: 0.021},
    },
    "UNKNOWN": {
        # Use GRADUAL_DETERIORATOR as conservative default
        0:  {3: 0.020, 6: 0.055, 9: 0.085, 12: 0.115, 18: 0.165, 24: 0.215},
        6:  {3: 0.015, 6: 0.040, 9: 0.065, 12: 0.090, 18: 0.130, 24: 0.170},
        12: {3: 0.010, 6: 0.028, 9: 0.045, 12: 0.062, 18: 0.090, 24: 0.120},
        24: {3: 0.006, 6: 0.016, 9: 0.026, 12: 0.037, 18: 0.054, 24: 0.072},
    },
}

# Curve B: recovery WITH structured plan agreed (EFS = Y or TDR agreed)
_DEFAULT_CURVE_B: Dict[str, Dict[int, Dict[int, float]]] = {
    "SELECTIVE_DEFAULTER": {
        # High willingness — structured plan gives big uplift
        0:  {3: 0.080, 6: 0.200, 9: 0.320, 12: 0.430, 18: 0.570, 24: 0.670},
        6:  {3: 0.065, 6: 0.165, 9: 0.265, 12: 0.355, 18: 0.470, 24: 0.555},
        12: {3: 0.048, 6: 0.125, 9: 0.200, 12: 0.270, 18: 0.360, 24: 0.430},
        24: {3: 0.030, 6: 0.080, 9: 0.128, 12: 0.173, 18: 0.230, 24: 0.275},
    },
    "LIFE_EVENT": {
        # Willing but constrained — moderate uplift
        0:  {3: 0.060, 6: 0.160, 9: 0.255, 12: 0.345, 18: 0.465, 24: 0.560},
        6:  {3: 0.048, 6: 0.130, 9: 0.210, 12: 0.285, 18: 0.385, 24: 0.465},
        12: {3: 0.035, 6: 0.095, 9: 0.155, 12: 0.210, 18: 0.285, 24: 0.348},
        24: {3: 0.022, 6: 0.060, 9: 0.097, 12: 0.132, 18: 0.180, 24: 0.220},
    },
    "GRADUAL_DETERIORATOR": {
        0:  {3: 0.038, 6: 0.100, 9: 0.162, 12: 0.222, 18: 0.308, 24: 0.385},
        6:  {3: 0.030, 6: 0.080, 9: 0.132, 12: 0.182, 18: 0.252, 24: 0.315},
        12: {3: 0.022, 6: 0.058, 9: 0.096, 12: 0.133, 18: 0.186, 24: 0.235},
        24: {3: 0.013, 6: 0.035, 9: 0.057, 12: 0.080, 18: 0.113, 24: 0.143},
    },
    "STRATEGIC_NON_PAYER": {
        # Low willingness — offer helps little unless very attractive
        0:  {3: 0.018, 6: 0.048, 9: 0.078, 12: 0.108, 18: 0.155, 24: 0.200},
        6:  {3: 0.014, 6: 0.038, 9: 0.062, 12: 0.086, 18: 0.123, 24: 0.160},
        12: {3: 0.010, 6: 0.027, 9: 0.044, 12: 0.061, 18: 0.088, 24: 0.115},
        24: {3: 0.006, 6: 0.016, 9: 0.026, 12: 0.036, 18: 0.053, 24: 0.069},
    },
    "TRULY_INSOLVENT": {
        # Cannot pay regardless of offer — minimal uplift
        0:  {3: 0.012, 6: 0.028, 9: 0.044, 12: 0.060, 18: 0.086, 24: 0.110},
        6:  {3: 0.009, 6: 0.021, 9: 0.033, 12: 0.045, 18: 0.065, 24: 0.083},
        12: {3: 0.006, 6: 0.014, 9: 0.022, 12: 0.031, 18: 0.044, 24: 0.057},
        24: {3: 0.004, 6: 0.008, 9: 0.013, 12: 0.018, 18: 0.026, 24: 0.034},
    },
    "UNKNOWN": {
        # Conservative default
        0:  {3: 0.038, 6: 0.100, 9: 0.162, 12: 0.222, 18: 0.308, 24: 0.385},
        6:  {3: 0.030, 6: 0.080, 9: 0.132, 12: 0.182, 18: 0.252, 24: 0.315},
        12: {3: 0.022, 6: 0.058, 9: 0.096, 12: 0.133, 18: 0.186, 24: 0.235},
        24: {3: 0.013, 6: 0.035, 9: 0.057, 12: 0.080, 18: 0.113, 24: 0.143},
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CurvePVResult:
    """PV result for a single curve evaluation."""
    persona: str
    months_at_180plus: int
    staleness_bucket: int
    outstanding: float
    horizon_months: int
    pv_amount: float               # PV of all cash flows from this curve
    cumulative_recovery_pct: float # total % recovered by horizon (undiscounted)
    curve_type: str                # "A" or "B"
    is_default: bool               # True if using industry defaults, not fitted


@dataclass
class PaydownNPVResult:
    """Full NPV result comparing Curve B offer against Curve A baseline."""
    persona: str
    months_at_180plus: int
    outstanding: float
    pv_baseline: float             # PV(Curve_A) — natural recovery
    pv_with_offer: float           # PV(Curve_B) — with structured plan
    concession_cost: float         # principal + interest + charges waived + processing
    npv_offer: float               # PV(Curve_B) − concession_cost
    npv_baseline: float            # PV(Curve_A) — same as pv_baseline
    incremental_npv: float         # npv_offer − npv_baseline (gain over doing nothing)
    horizon_months: int
    is_default: bool


# ─────────────────────────────────────────────────────────────────────────────
# PAYDOWN CURVE ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class PaydownCurveEngine:
    """
    Computes PV of recovery cash flows using empirical paydown curves.

    Two operating modes:
      1. Default mode: uses industry-calibrated curves (MVP)
      2. Fitted mode: curves loaded from historical CardX cohort data via
         fit_from_history(cohort_df)

    Usage:
        engine = PaydownCurveEngine()
        result = engine.compute_npv(
            persona="SELECTIVE_DEFAULTER",
            months_at_180plus=3,
            outstanding=100_000,
            concession_cost=15_000,
            monthly_discount_rate=0.01,
            horizon_months=24,
        )
        # result.incremental_npv = gain from making the offer vs. doing nothing
    """

    def __init__(
        self,
        curve_a: Optional[Dict] = None,
        curve_b: Optional[Dict] = None,
    ):
        self._curve_a = curve_a or _DEFAULT_CURVE_A
        self._curve_b = curve_b or _DEFAULT_CURVE_B
        self._fitted   = (curve_a is not None)

    # ── PUBLIC ────────────────────────────────────────────────────────────────

    def compute_npv(
        self,
        persona: str,
        months_at_180plus: int,
        outstanding: float,
        concession_cost: float,
        monthly_discount_rate: float,
        horizon_months: int = 24,
    ) -> PaydownNPVResult:
        """
        Compute the NPV of a structured offer vs. natural recovery baseline.

        Args:
            persona: one of PERSONAS
            months_at_180plus: how long account has been at 180+ DPD
            outstanding: current outstanding balance (THB)
            concession_cost: total cost of concessions to CardX (THB)
                             = principal_waived + interest_waived
                               + charges_waived + processing_cost
            monthly_discount_rate: monthly discount rate (e.g. 0.01 for 12% pa)
            horizon_months: evaluation horizon (default 24m)

        Returns:
            PaydownNPVResult with pv_baseline, pv_with_offer, npv_offer,
            npv_baseline, incremental_npv
        """
        bucket = self._staleness_bucket(months_at_180plus)

        pv_a = self._pv_of_curve(
            self._curve_a, persona, bucket,
            outstanding, monthly_discount_rate, horizon_months
        )
        pv_b = self._pv_of_curve(
            self._curve_b, persona, bucket,
            outstanding, monthly_discount_rate, horizon_months
        )

        npv_offer    = pv_b - concession_cost
        npv_baseline = pv_a
        incremental  = npv_offer - npv_baseline

        return PaydownNPVResult(
            persona=persona,
            months_at_180plus=months_at_180plus,
            outstanding=outstanding,
            pv_baseline=round(pv_a, 2),
            pv_with_offer=round(pv_b, 2),
            concession_cost=round(concession_cost, 2),
            npv_offer=round(npv_offer, 2),
            npv_baseline=round(npv_baseline, 2),
            incremental_npv=round(incremental, 2),
            horizon_months=horizon_months,
            is_default=not self._fitted,
        )

    def get_curve(
        self,
        curve_type: str,
        persona: str,
        months_at_180plus: int,
        outstanding: float,
        monthly_discount_rate: float,
        horizon_months: int = 24,
    ) -> CurvePVResult:
        """
        Get PV result for a single curve (A or B).

        Args:
            curve_type: "A" or "B"
        """
        store   = self._curve_a if curve_type == "A" else self._curve_b
        bucket  = self._staleness_bucket(months_at_180plus)
        pv      = self._pv_of_curve(
            store, persona, bucket,
            outstanding, monthly_discount_rate, horizon_months
        )
        # undiscounted total recovery at horizon
        checkpoints = self._get_checkpoints(store, persona, bucket)
        max_checkpoint = max(c for c in _CHECKPOINTS if c <= horizon_months)
        cumulative_pct = checkpoints.get(max_checkpoint, 0.0)

        return CurvePVResult(
            persona=persona,
            months_at_180plus=months_at_180plus,
            staleness_bucket=bucket,
            outstanding=outstanding,
            horizon_months=horizon_months,
            pv_amount=round(pv, 2),
            cumulative_recovery_pct=round(cumulative_pct, 4),
            curve_type=curve_type,
            is_default=not self._fitted,
        )

    def incremental_recovery_pct(
        self,
        persona: str,
        months_at_180plus: int,
        horizon_months: int = 24,
    ) -> float:
        """
        Return incremental cumulative recovery pct: Curve_B − Curve_A at horizon.
        Useful for presentation / sanity check.
        """
        bucket = self._staleness_bucket(months_at_180plus)
        horizon_cp = max(c for c in _CHECKPOINTS if c <= horizon_months)

        a_pts = self._get_checkpoints(self._curve_a, persona, bucket)
        b_pts = self._get_checkpoints(self._curve_b, persona, bucket)

        return round(
            b_pts.get(horizon_cp, 0.0) - a_pts.get(horizon_cp, 0.0), 4
        )

    @classmethod
    def fit_from_history(cls, cohort_df) -> "PaydownCurveEngine":
        """
        Fit paydown curves from historical CardX collection cohort data.

        Expected cohort_df columns:
          - account_id
          - first_180dpd_date
          - persona                  (from RecoveryScorecard)
          - efs_flag                 ("Y" = plan agreed, "N" = natural)
          - month_offset             (months since first_180dpd_date)
          - cumulative_recovered_pct (% of original outstanding recovered by month)

        Returns a fitted PaydownCurveEngine instance.
        """
        import pandas as pd

        curve_a: Dict = {p: {b: {} for b in _STALENESS_BUCKETS} for p in PERSONAS}
        curve_b: Dict = {p: {b: {} for b in _STALENESS_BUCKETS} for p in PERSONAS}

        for persona in PERSONAS:
            for bucket in _STALENESS_BUCKETS:
                next_bucket = _STALENESS_BUCKETS[
                    _STALENESS_BUCKETS.index(bucket) + 1
                ] if bucket != _STALENESS_BUCKETS[-1] else 9999

                mask_persona = cohort_df["persona"] == persona
                mask_bucket  = (
                    (cohort_df["months_at_180plus"] >= bucket) &
                    (cohort_df["months_at_180plus"] < next_bucket)
                )
                cohort = cohort_df[mask_persona & mask_bucket]

                if len(cohort) < 50:
                    logger.warning(
                        "Insufficient data for persona=%s bucket=%d (n=%d), "
                        "using industry defaults",
                        persona, bucket, len(cohort)
                    )
                    curve_a[persona][bucket] = _DEFAULT_CURVE_A[persona][bucket]
                    curve_b[persona][bucket] = _DEFAULT_CURVE_B[persona][bucket]
                    continue

                for cp in _CHECKPOINTS:
                    cp_data = cohort[cohort["month_offset"] == cp]
                    if len(cp_data) > 0:
                        curve_a[persona][bucket][cp] = (
                            cp_data[cp_data["efs_flag"] == "N"]
                            ["cumulative_recovered_pct"].mean()
                        )
                        curve_b[persona][bucket][cp] = (
                            cp_data[cp_data["efs_flag"] == "Y"]
                            ["cumulative_recovered_pct"].mean()
                        )

        engine = cls(curve_a=curve_a, curve_b=curve_b)
        engine._fitted = True
        logger.info("PaydownCurveEngine fitted from historical data")
        return engine

    # ── PRIVATE ───────────────────────────────────────────────────────────────

    def _staleness_bucket(self, months_at_180plus: int) -> int:
        """Return lower bound of staleness bracket."""
        bucket = 0
        for b in _STALENESS_BUCKETS:
            if months_at_180plus >= b:
                bucket = b
        return bucket

    def _get_checkpoints(
        self,
        store: Dict,
        persona: str,
        bucket: int,
    ) -> Dict[int, float]:
        """Safely retrieve curve checkpoints with persona fallback."""
        p = persona if persona in store else "UNKNOWN"
        b = bucket if bucket in store.get(p, {}) else min(store.get(p, {0: {}}).keys())
        return store[p][b]

    def _pv_of_curve(
        self,
        store: Dict,
        persona: str,
        bucket: int,
        outstanding: float,
        monthly_rate: float,
        horizon_months: int,
    ) -> float:
        """
        Compute PV of all monthly cash flows implied by a paydown curve.

        The curve gives CUMULATIVE % recovered at checkpoint months.
        We interpolate to get incremental recovery each month, then discount.
        """
        checkpoints = self._get_checkpoints(store, persona, bucket)

        # Build monthly incremental cashflows from cumulative curve
        pv = 0.0
        prev_cumulative = 0.0

        for t in range(1, min(horizon_months, max(_CHECKPOINTS)) + 1):
            cumulative_t = self._interpolate(checkpoints, t)
            incremental  = max(cumulative_t - prev_cumulative, 0.0)
            cashflow     = outstanding * incremental
            discount     = (1 + monthly_rate) ** (-t)
            pv          += cashflow * discount
            prev_cumulative = cumulative_t

        return pv

    def _interpolate(self, checkpoints: Dict[int, float], t: int) -> float:
        """Linearly interpolate cumulative recovery % at month t."""
        months_sorted = sorted(checkpoints.keys())

        if t <= months_sorted[0]:
            # Linearly scale from 0
            return checkpoints[months_sorted[0]] * t / months_sorted[0]

        if t >= months_sorted[-1]:
            return checkpoints[months_sorted[-1]]

        for i in range(len(months_sorted) - 1):
            t0, t1 = months_sorted[i], months_sorted[i + 1]
            if t0 <= t <= t1:
                frac = (t - t0) / (t1 - t0)
                return checkpoints[t0] + frac * (checkpoints[t1] - checkpoints[t0])

        return checkpoints[months_sorted[-1]]


# ─────────────────────────────────────────────────────────────────────────────
# BALANCE DECOMPOSITION — concession cost helper
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BalanceDecomposition:
    """
    Splits outstanding into principal / interest / charges.
    Waivers are applied cheapest-first:
      1. penalty_charges   (lowest economic cost to waive — often already impaired)
      2. accrued_interest  (foregone interest income)
      3. principal         (hardest — actual capital loss)
    """
    principal: float
    accrued_interest: float
    penalty_charges: float
    total: float

    # Computed waiver components (filled by compute_waivers)
    charges_waived: float  = 0.0
    interest_waived: float = 0.0
    principal_waived: float = 0.0
    total_waived: float    = 0.0
    settlement_amount: float = 0.0


def compute_waivers(
    decomp: BalanceDecomposition,
    target_waiver_amount: float,
    processing_cost: float = 500.0,
) -> Tuple[BalanceDecomposition, float]:
    """
    Apply target_waiver_amount cheapest-first (charges → interest → principal).

    Returns:
        decomp: updated BalanceDecomposition with waiver fields set
        concession_cost: total cost to CardX (waivers + processing)
    """
    remaining = target_waiver_amount
    decomp = BalanceDecomposition(
        principal=decomp.principal,
        accrued_interest=decomp.accrued_interest,
        penalty_charges=decomp.penalty_charges,
        total=decomp.total,
    )

    # Step 1: waive charges first (cheapest)
    charges_waived = min(remaining, decomp.penalty_charges)
    remaining      = remaining - charges_waived

    # Step 2: waive interest (medium cost)
    interest_waived = min(remaining, decomp.accrued_interest)
    remaining       = remaining - interest_waived

    # Step 3: waive principal (most expensive)
    principal_waived = min(remaining, decomp.principal)

    total_waived = charges_waived + interest_waived + principal_waived

    decomp.charges_waived   = round(charges_waived, 2)
    decomp.interest_waived  = round(interest_waived, 2)
    decomp.principal_waived = round(principal_waived, 2)
    decomp.total_waived     = round(total_waived, 2)
    decomp.settlement_amount = round(decomp.total - total_waived, 2)

    # Concession cost to CardX:
    # - principal waived = full face value loss
    # - interest waived = foregone income (weight 0.6 — partially already at risk)
    # - charges waived = minimal (often already written down, weight 0.3)
    concession_cost = (
        principal_waived * 1.0
        + interest_waived * 0.6
        + charges_waived  * 0.3
        + processing_cost
    )

    return decomp, round(concession_cost, 2)


# ─────────────────────────────────────────────────────────────────────────────
# SMOKE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    engine = PaydownCurveEngine()

    print("\n=== Paydown Curve NPV: SELECTIVE_DEFAULTER, 3m at 180+ ===")
    result = engine.compute_npv(
        persona="SELECTIVE_DEFAULTER",
        months_at_180plus=3,
        outstanding=100_000,
        concession_cost=15_000,
        monthly_discount_rate=0.01,  # 12% pa
        horizon_months=24,
    )
    print(f"  PV baseline (no offer):  {result.pv_baseline:>10,.0f} THB")
    print(f"  PV with offer:           {result.pv_with_offer:>10,.0f} THB")
    print(f"  Concession cost:         {result.concession_cost:>10,.0f} THB")
    print(f"  NPV offer:               {result.npv_offer:>10,.0f} THB")
    print(f"  Incremental NPV:         {result.incremental_npv:>10,.0f} THB")
    print(f"  Incremental recovery %:  {engine.incremental_recovery_pct('SELECTIVE_DEFAULTER', 3):.1%}")

    print("\n=== Paydown Curve NPV: TRULY_INSOLVENT, 18m at 180+ ===")
    result2 = engine.compute_npv(
        persona="TRULY_INSOLVENT",
        months_at_180plus=18,
        outstanding=80_000,
        concession_cost=20_000,
        monthly_discount_rate=0.01,
        horizon_months=24,
    )
    print(f"  PV baseline (no offer):  {result2.pv_baseline:>10,.0f} THB")
    print(f"  PV with offer:           {result2.pv_with_offer:>10,.0f} THB")
    print(f"  NPV offer:               {result2.npv_offer:>10,.0f} THB")
    print(f"  Incremental NPV:         {result2.incremental_npv:>10,.0f} THB")
    print(f"  → Debt sale preferred if sale_price × 80,000 > {result2.npv_offer:,.0f}")

    print("\n=== Balance Decomposition: cheapest-first waivers ===")
    decomp = BalanceDecomposition(
        principal=70_000, accrued_interest=20_000, penalty_charges=10_000,
        total=100_000
    )
    decomp_out, concession = compute_waivers(decomp, target_waiver_amount=25_000)
    print(f"  Charges waived:   {decomp_out.charges_waived:>8,.0f} (all charges)")
    print(f"  Interest waived:  {decomp_out.interest_waived:>8,.0f}")
    print(f"  Principal waived: {decomp_out.principal_waived:>8,.0f}")
    print(f"  Settlement amt:   {decomp_out.settlement_amount:>8,.0f}")
    print(f"  Concession cost:  {concession:>8,.0f} (CardX economic cost)")
