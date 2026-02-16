"""
NPV Engine
===========
FICO-inspired Loan NPV and Portfolio NPV computation for collections.

FICO terminology used throughout:
  - Loan NPV            : NPV of a single TDR offer for one account
  - Portfolio NPV       : Sum of loan NPVs across all accounts
  - Take-up Probability : P(customer accepts the offered treatment)
  - Re-default Probability : P(customer defaults again after accepting TDR)
  - Collection Treatment Optimization : selecting best path per account

Three paths valued on same NPV basis:
  PATH 1 — TDR (Term Debt Restructure)
            Loan NPV = PV of survival-weighted cash flows − concession cost
  PATH 2 — Legal
            Legal NPV = P(success) × asset_value / (1+r)^months − legal_cost
  PATH 3 — Debt Sale
            Sale NPV = sale_price_pct × outstanding  (immediate, certain)

Best path = argmax(tdr_npv, legal_npv, debt_sale_npv).
Best offer = argmax(loan_npv) across all TDR offer structures.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class NPVConfig:
    """
    NPV parameters. Tune once historical TDR data is available.
    All rates are monthly unless stated.
    """

    # ── Discount rate ─────────────────────────────────────────────────────────
    annual_discount_rate: float     = 0.12      # CardX cost of capital (12% pa)

    # ── Re-default survival curve (FICO: "re-default probability") ────────────
    # Industry defaults — REPLACE with trained model predictions (Q8-A data)
    # Format: month → cumulative re-default rate
    redefault_curve_by_stage: Dict[str, Dict[int, float]] = field(
        default_factory=lambda: {
            "SM":         {6: 0.15, 12: 0.25, 24: 0.35, 36: 0.42, 48: 0.48, 60: 0.52},
            "NPL":        {6: 0.25, 12: 0.40, 24: 0.55, 36: 0.63, 48: 0.70, 60: 0.75},
            "CHARGEOFF":  {6: 0.35, 12: 0.55, 24: 0.70, 36: 0.78, 48: 0.83, 60: 0.87},
        }
    )

    # ── Legal parameters ──────────────────────────────────────────────────────
    legal_cost_fixed: float         = 15000.0   # THB — fixed legal filing cost
    legal_cost_pct: float           = 0.05      # 5% of outstanding (variable)
    legal_months_to_resolution: int = 18        # avg months to asset realisation
    asset_recovery_rate: float      = 0.60      # % of asset value recovered
    p_legal_success_no_asset: float = 0.10      # P(win) if no secured asset in bureau

    # ── Debt sale parameters ──────────────────────────────────────────────────
    # Sale price in cents on the dollar by stage
    debt_sale_price_by_stage: Dict[str, float] = field(
        default_factory=lambda: {
            "SM":        0.45,   # 45 cents
            "NPL":       0.25,   # 25 cents
            "CHARGEOFF": 0.08,   # 8 cents
        }
    )

    # ── Concession cost to CardX ──────────────────────────────────────────────
    processing_cost_per_tdr: float  = 500.0     # THB — agent + system cost

    # ── Minimum NPV threshold ─────────────────────────────────────────────────
    min_tdr_npv_vs_sale: float      = 0.05      # TDR must beat debt sale by 5%


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LoanNPV:
    """FICO: Loan Net Present Value for one TDR offer."""
    offer_id: str
    offer_type: str
    outstanding: float
    take_up_probability: float          # FICO term: P(acceptance)
    redefault_probability_12m: float    # FICO term
    loan_npv: float                     # PV of cash flows − concession
    concession_cost: float              # interest + fee waivers cost to CardX
    expected_cashflow: float            # take_up_probability × loan_npv
    monthly_instalment: float
    tenor_months: int
    haircut_pct: float                  # % of outstanding waived
    is_affordable: bool                 # instalment ≤ payment_capacity


@dataclass
class PathValuation:
    """Full path comparison for one account. FICO: Collection Treatment Optimization."""
    account_id: str
    outstanding: float
    stage: str

    # TDR path
    best_tdr_offer_id: str
    best_tdr_loan_npv: float
    best_tdr_take_up_prob: float
    best_tdr_expected_cashflow: float

    # Legal path
    legal_npv: float
    p_legal_success: float
    legal_viable: bool

    # Debt sale path
    debt_sale_npv: float
    debt_sale_price_pct: float

    # Recommended path
    recommended_path: str           # "TDR" | "LEGAL" | "DEBT_SALE" | "HOLD"
    recommended_path_npv: float
    path_rationale: str


# ─────────────────────────────────────────────────────────────────────────────
# NPV ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class NPVEngine:
    """
    FICO-inspired NPV computation for TDR, Legal, and Debt Sale paths.

    Key FICO concepts implemented:
      - Loan NPV: PV of survival-weighted cash flows
      - Portfolio NPV: sum across accounts (for capacity optimisation)
      - Take-up probability: from OfferAcceptanceModel
      - Re-default probability: from survival curve (rule-based now, trained later)
      - Collection Treatment Optimization: path comparison
    """

    def __init__(self, config: Optional[NPVConfig] = None):
        self.config = config or NPVConfig()
        self._monthly_rate = (1 + self.config.annual_discount_rate) ** (1/12) - 1

    # ── PUBLIC ────────────────────────────────────────────────────────────────

    def compute_loan_npv(
        self,
        offer_params: Dict,
        take_up_probability: float,
        stage: str,
        payment_capacity: float,
    ) -> LoanNPV:
        """
        Compute Loan NPV for a single TDR offer.
        FICO: "Loan Net Present Value"

        Args:
            offer_params: dict with keys:
                offer_id, offer_type, outstanding, haircut_pct,
                interest_rate_pa, tenor_months, fee_waiver_amount
            take_up_probability: FICO "take-up probability" — P(customer accepts)
            stage: SM | NPL | CHARGEOFF
            payment_capacity: monthly payment capacity from AffordabilityEngine
        """
        cfg         = self.config
        outstanding = float(offer_params["outstanding"])
        haircut_pct = float(offer_params.get("haircut_pct", 0.0))
        tenor       = int(offer_params.get("tenor_months", 24))
        rate_pa     = float(offer_params.get("interest_rate_pa", 0.0))
        fee_waiver  = float(offer_params.get("fee_waiver_amount", 0.0))

        # Principal after haircut
        principal_after_haircut = outstanding * (1 - haircut_pct)

        # Monthly instalment (standard annuity formula)
        monthly_rate = rate_pa / 12.0
        if monthly_rate > 0:
            instalment = (
                principal_after_haircut * monthly_rate
                / (1 - (1 + monthly_rate) ** (-tenor))
            )
        else:
            instalment = principal_after_haircut / tenor

        is_affordable = instalment <= payment_capacity

        # Re-default survival curve
        redefault_curve = cfg.redefault_curve_by_stage.get(
            stage, cfg.redefault_curve_by_stage["NPL"]
        )

        # PV of cash flows weighted by P(customer still paying at month t)
        pv_cashflows = 0.0
        for t in range(1, tenor + 1):
            # P(survive to month t) = 1 − cumulative re-default by t
            # Interpolate from curve checkpoints
            redefault_t = self._interpolate_redefault(redefault_curve, t)
            p_survive_t = max(1.0 - redefault_t, 0.0)

            # Discount factor
            discount = (1 + self._monthly_rate) ** (-t)

            pv_cashflows += p_survive_t * instalment * discount

        # Concession cost to CardX
        concession_cost = (
            outstanding * haircut_pct        # waived principal
            + fee_waiver                      # waived fees
            + cfg.processing_cost_per_tdr     # agent + system
        )

        # Loan NPV
        loan_npv = pv_cashflows - concession_cost

        # Re-default at 12 months (FICO KPI)
        redefault_12m = self._interpolate_redefault(redefault_curve, 12)

        # Expected cashflow = take_up_prob × loan_npv
        expected_cashflow = take_up_probability * max(loan_npv, 0)

        logger.debug(
            "LoanNPV: offer=%s pv_flows=%.0f concession=%.0f npv=%.0f",
            offer_params.get("offer_id"), pv_cashflows, concession_cost, loan_npv
        )

        return LoanNPV(
            offer_id=str(offer_params.get("offer_id", "unknown")),
            offer_type=str(offer_params.get("offer_type", "unknown")),
            outstanding=outstanding,
            take_up_probability=round(take_up_probability, 4),
            redefault_probability_12m=round(redefault_12m, 4),
            loan_npv=round(loan_npv, 2),
            concession_cost=round(concession_cost, 2),
            expected_cashflow=round(expected_cashflow, 2),
            monthly_instalment=round(instalment, 2),
            tenor_months=tenor,
            haircut_pct=haircut_pct,
            is_affordable=is_affordable,
        )

    def compute_legal_npv(
        self,
        outstanding: float,
        stage: str,
        bureau_asset_value: float = 0.0,
        has_secured_asset: bool   = False,
    ) -> Tuple[float, float, bool]:
        """
        Compute Legal Path NPV.
        Returns (legal_npv, p_legal_success, legal_viable).
        """
        cfg = self.config

        # P(legal success)
        if has_secured_asset and bureau_asset_value > 0:
            p_success = min(0.65, bureau_asset_value / outstanding)
        else:
            p_success = cfg.p_legal_success_no_asset

        # Asset realisation value (discounted for time)
        asset_recovery = bureau_asset_value * cfg.asset_recovery_rate
        months         = cfg.legal_months_to_resolution
        discount       = (1 + self._monthly_rate) ** (-months)
        pv_asset       = asset_recovery * discount * p_success

        # Legal cost
        legal_cost = (
            cfg.legal_cost_fixed
            + outstanding * cfg.legal_cost_pct
        )

        legal_npv = pv_asset - legal_cost
        legal_viable = (
            has_secured_asset
            and legal_npv > 0
            and outstanding > (legal_cost * 2)   # balance justifies cost
        )

        logger.debug(
            "LegalNPV: outstanding=%.0f p_success=%.2f npv=%.0f viable=%s",
            outstanding, p_success, legal_npv, legal_viable
        )

        return round(legal_npv, 2), round(p_success, 4), legal_viable

    def compute_debt_sale_npv(
        self,
        outstanding: float,
        stage: str,
        override_price_pct: Optional[float] = None,
    ) -> Tuple[float, float]:
        """
        Compute Debt Sale NPV.
        Immediate and certain — no discounting needed.
        Returns (sale_npv, sale_price_pct).
        """
        price_pct = override_price_pct or self.config.debt_sale_price_by_stage.get(
            stage, 0.15
        )
        sale_npv  = outstanding * price_pct
        return round(sale_npv, 2), round(price_pct, 4)

    def compare_paths(
        self,
        account_id: str,
        outstanding: float,
        stage: str,
        tdr_offers: List[LoanNPV],
        bureau_asset_value: float = 0.0,
        has_secured_asset: bool   = False,
        override_sale_price: Optional[float] = None,
    ) -> PathValuation:
        """
        FICO: Collection Treatment Optimization.
        Compare all paths on same Loan NPV basis, recommend best.
        """
        cfg = self.config

        # ── TDR: pick best offer ──────────────────────────────────────────────
        affordable_offers = [o for o in tdr_offers if o.is_affordable]
        all_offers        = affordable_offers or tdr_offers  # fallback if none affordable

        if all_offers:
            best_offer = max(all_offers, key=lambda o: o.expected_cashflow)
            tdr_ecf    = best_offer.expected_cashflow
            tdr_npv    = best_offer.loan_npv
            tdr_tup    = best_offer.take_up_probability
        else:
            best_offer = None
            tdr_ecf    = 0.0
            tdr_npv    = 0.0
            tdr_tup    = 0.0

        # ── Legal ─────────────────────────────────────────────────────────────
        legal_npv, p_legal, legal_viable = self.compute_legal_npv(
            outstanding, stage, bureau_asset_value, has_secured_asset
        )

        # ── Debt Sale ─────────────────────────────────────────────────────────
        sale_npv, sale_price_pct = self.compute_debt_sale_npv(
            outstanding, stage, override_sale_price
        )

        # ── Path selection ────────────────────────────────────────────────────
        path_values = {
            "TDR":       tdr_ecf,
            "LEGAL":     legal_npv if legal_viable else -np.inf,
            "DEBT_SALE": sale_npv,
        }

        recommended_path = max(path_values, key=path_values.get)
        recommended_npv  = path_values[recommended_path]

        # Hold if all paths have negative NPV
        if recommended_npv <= 0:
            recommended_path = "HOLD"
            recommended_npv  = 0.0

        # Build rationale
        rationale = self._build_rationale(
            recommended_path, tdr_ecf, tdr_npv, tdr_tup,
            legal_npv, legal_viable, sale_npv, best_offer
        )

        return PathValuation(
            account_id=account_id,
            outstanding=outstanding,
            stage=stage,
            best_tdr_offer_id=best_offer.offer_id if best_offer else "none",
            best_tdr_loan_npv=tdr_npv,
            best_tdr_take_up_prob=tdr_tup,
            best_tdr_expected_cashflow=tdr_ecf,
            legal_npv=legal_npv,
            p_legal_success=p_legal,
            legal_viable=legal_viable,
            debt_sale_npv=sale_npv,
            debt_sale_price_pct=sale_price_pct,
            recommended_path=recommended_path,
            recommended_path_npv=round(recommended_npv, 2),
            path_rationale=rationale,
        )

    def compute_portfolio_npv(
        self, path_valuations: List[PathValuation]
    ) -> Dict[str, float]:
        """
        FICO: Portfolio NPV — aggregate across all accounts.
        Used for capacity-constrained optimisation.
        """
        total_tdr   = sum(p.best_tdr_expected_cashflow for p in path_valuations)
        total_legal = sum(p.legal_npv for p in path_valuations if p.legal_viable)
        total_sale  = sum(p.debt_sale_npv for p in path_valuations)
        recommended = sum(p.recommended_path_npv for p in path_valuations)

        by_path = {}
        for path in ["TDR", "LEGAL", "DEBT_SALE", "HOLD"]:
            by_path[path] = sum(
                p.recommended_path_npv
                for p in path_valuations
                if p.recommended_path == path
            )

        return {
            "portfolio_npv_recommended": round(recommended, 2),
            "portfolio_tdr_npv":         round(total_tdr, 2),
            "portfolio_legal_npv":       round(total_legal, 2),
            "portfolio_sale_npv":        round(total_sale, 2),
            "accounts_tdr":              sum(1 for p in path_valuations if p.recommended_path == "TDR"),
            "accounts_legal":            sum(1 for p in path_valuations if p.recommended_path == "LEGAL"),
            "accounts_sale":             sum(1 for p in path_valuations if p.recommended_path == "DEBT_SALE"),
            "accounts_hold":             sum(1 for p in path_valuations if p.recommended_path == "HOLD"),
            "by_path":                   {k: round(v, 2) for k, v in by_path.items()},
        }

    # ── PRIVATE ───────────────────────────────────────────────────────────────

    def _interpolate_redefault(
        self, curve: Dict[int, float], month: int
    ) -> float:
        """Linearly interpolate re-default probability at arbitrary month."""
        checkpoints = sorted(curve.keys())
        if month <= checkpoints[0]:
            return curve[checkpoints[0]] * month / checkpoints[0]
        if month >= checkpoints[-1]:
            return curve[checkpoints[-1]]
        for i in range(len(checkpoints) - 1):
            t0, t1 = checkpoints[i], checkpoints[i + 1]
            if t0 <= month <= t1:
                frac = (month - t0) / (t1 - t0)
                return curve[t0] + frac * (curve[t1] - curve[t0])
        return curve[checkpoints[-1]]

    def _build_rationale(
        self,
        path, tdr_ecf, tdr_npv, tdr_tup,
        legal_npv, legal_viable, sale_npv, best_offer
    ) -> str:
        if path == "TDR":
            offer_desc = (
                f"{best_offer.offer_type} haircut={best_offer.haircut_pct:.0%} "
                f"tenor={best_offer.tenor_months}m instalment={best_offer.monthly_instalment:.0f}"
                if best_offer else "no offer"
            )
            return (
                f"TDR recommended: expected_cashflow={tdr_ecf:.0f} "
                f"take_up_prob={tdr_tup:.0%} loan_npv={tdr_npv:.0f}. "
                f"Best offer: {offer_desc}."
            )
        elif path == "LEGAL":
            return (
                f"LEGAL recommended: legal_npv={legal_npv:.0f} "
                f"beats TDR ({tdr_ecf:.0f}) and sale ({sale_npv:.0f}). "
                f"Secured asset identified in bureau."
            )
        elif path == "DEBT_SALE":
            return (
                f"DEBT SALE recommended: sale_npv={sale_npv:.0f} "
                f"beats TDR ({tdr_ecf:.0f}). "
                f"{'Legal not viable (no asset). ' if not legal_viable else ''}"
                f"Immediate recovery preferred."
            )
        else:
            return (
                f"HOLD: all paths have negative NPV. "
                f"TDR={tdr_ecf:.0f}, Legal={legal_npv:.0f}, Sale={sale_npv:.0f}."
            )
