"""
NPV Engine
===========
Paydown-curve-driven Loan NPV and Portfolio NPV for collections.

NPV framework (replaces FICO take-up-weighted formula):
  - No take-up probability assumed — offer NPV is grounded in empirical
    paydown curves from historical cohort data.
  - Curve A: natural recovery without any structured offer (baseline)
  - Curve B: recovery with structured plan agreed (EFS = Y, TDR agreed)

Three paths valued on the same absolute NPV basis:
  PATH 1 — TDR (Term Debt Restructure / Settlement)
            NPV_offer    = PV(Curve_B cashflows) − concession_cost
            NPV_baseline = PV(Curve_A cashflows)
            Incremental  = NPV_offer − NPV_baseline

  PATH 2 — Legal
            Legal NPV = P(success) × asset_value / (1+r)^months − legal_cost

  PATH 3 — Debt Sale
            Sale NPV = sale_price_pct × outstanding  (immediate, certain)

  PATH 4 — HOLD (do nothing)
            Hold NPV = PV(Curve_A)  — natural recovery with no concession

Best path = argmax(NPV_offer, legal_npv, debt_sale_npv, hold_npv).
Best offer = argmax(incremental_npv) across all TDR offer structures
             subject to instalment ≤ payment_capacity.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .paydown_curves import PaydownCurveEngine, PaydownNPVResult

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

    # ── Paydown curve horizon ─────────────────────────────────────────────────
    paydown_horizon_months: int     = 24        # months of recovery to evaluate

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
    """Paydown-curve-driven Loan NPV for one TDR offer."""
    offer_id: str
    offer_type: str
    outstanding: float
    persona: str                        # recovery persona driving curve selection
    months_at_180plus: int              # account staleness (months at 180+ DPD)
    pv_baseline: float                  # PV(Curve_A) — natural recovery without offer
    pv_with_offer: float                # PV(Curve_B) — recovery with structured plan
    loan_npv: float                     # PV(Curve_B) − concession_cost (absolute NPV)
    incremental_npv: float              # loan_npv − pv_baseline (gain over doing nothing)
    concession_cost: float              # waivers + processing cost to CardX
    expected_cashflow: float            # same as loan_npv (no take_up_prob; kept for API compat)
    monthly_instalment: float
    tenor_months: int
    haircut_pct: float                  # % of outstanding waived
    is_affordable: bool                 # instalment ≤ payment_capacity
    # Balance decomposition
    principal_waived: float  = 0.0
    interest_waived: float   = 0.0
    charges_waived: float    = 0.0


@dataclass
class PathValuation:
    """Full path comparison for one account."""
    account_id: str
    outstanding: float
    stage: str
    persona: str

    # TDR path (paydown-curve NPV)
    best_tdr_offer_id: str
    best_tdr_loan_npv: float           # PV(Curve_B) − concession
    best_tdr_incremental_npv: float    # gain over baseline
    best_tdr_expected_cashflow: float  # same as best_tdr_loan_npv (API compat)

    # HOLD path (do nothing — natural recovery)
    hold_npv: float                    # PV(Curve_A) without concession

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
    Paydown-curve-driven NPV computation for TDR, Legal, and Debt Sale paths.

    Key design:
      - No take-up probability — NPV grounded in empirical recovery curves
      - Curve A (baseline) vs Curve B (with offer) determine absolute NPVs
      - Path selected by argmax over TDR, Legal, Debt Sale, Hold (Curve A)
      - Balance decomposition: waivers applied cheapest-first
    """

    def __init__(
        self,
        config: Optional[NPVConfig] = None,
        curve_engine: Optional[PaydownCurveEngine] = None,
    ):
        self.config       = config or NPVConfig()
        self.curve_engine = curve_engine or PaydownCurveEngine()
        self._monthly_rate = (1 + self.config.annual_discount_rate) ** (1/12) - 1

    # ── PUBLIC ────────────────────────────────────────────────────────────────

    def compute_loan_npv(
        self,
        offer_params: Dict,
        persona: str,
        months_at_180plus: int,
        payment_capacity: float,
        concession_cost: Optional[float] = None,
    ) -> LoanNPV:
        """
        Compute Loan NPV for a single TDR offer using paydown curves.

        Args:
            offer_params: dict with keys:
                offer_id, offer_type, outstanding, haircut_pct,
                interest_rate_pa, tenor_months,
                principal_waived, interest_waived, charges_waived
            persona: recovery persona (drives curve selection)
            months_at_180plus: account staleness at 180+ DPD
            payment_capacity: monthly ATP from AffordabilityEngine
            concession_cost: override total cost (if balance decomposition computed
                             externally by OfferGenerator); auto-computed if None.
        """
        cfg         = self.config
        outstanding = float(offer_params["outstanding"])
        haircut_pct = float(offer_params.get("haircut_pct", 0.0))
        tenor       = int(offer_params.get("tenor_months", 24))
        rate_pa     = float(offer_params.get("interest_rate_pa", 0.0))

        # Waiver components (from balance decomposition if available)
        principal_waived = float(offer_params.get("principal_waived", outstanding * haircut_pct))
        interest_waived  = float(offer_params.get("interest_waived", 0.0))
        charges_waived   = float(offer_params.get("charges_waived", 0.0))

        # Monthly instalment on the settlement / restructured amount
        principal_after_haircut = outstanding * (1 - haircut_pct)
        monthly_rate = rate_pa / 12.0
        if monthly_rate > 0:
            instalment = (
                principal_after_haircut * monthly_rate
                / (1 - (1 + monthly_rate) ** (-tenor))
            )
        else:
            instalment = principal_after_haircut / tenor if tenor > 0 else principal_after_haircut

        is_affordable = instalment <= payment_capacity

        # Concession cost to CardX (cheapest-first economic weighting)
        if concession_cost is None:
            concession_cost = (
                principal_waived * 1.0            # full face value loss
                + interest_waived * 0.6           # foregone income
                + charges_waived * 0.3            # often already impaired
                + cfg.processing_cost_per_tdr     # agent + system
            )

        # Paydown curve NPV
        paydown_result = self.curve_engine.compute_npv(
            persona=persona,
            months_at_180plus=months_at_180plus,
            outstanding=outstanding,
            concession_cost=concession_cost,
            monthly_discount_rate=self._monthly_rate,
            horizon_months=cfg.paydown_horizon_months,
        )

        loan_npv      = paydown_result.npv_offer
        incremental   = paydown_result.incremental_npv
        pv_baseline   = paydown_result.pv_baseline
        pv_with_offer = paydown_result.pv_with_offer

        logger.debug(
            "LoanNPV: offer=%s pv_baseline=%.0f pv_offer=%.0f concession=%.0f "
            "loan_npv=%.0f incremental=%.0f",
            offer_params.get("offer_id"), pv_baseline, pv_with_offer,
            concession_cost, loan_npv, incremental,
        )

        return LoanNPV(
            offer_id=str(offer_params.get("offer_id", "unknown")),
            offer_type=str(offer_params.get("offer_type", "unknown")),
            outstanding=outstanding,
            persona=persona,
            months_at_180plus=months_at_180plus,
            pv_baseline=round(pv_baseline, 2),
            pv_with_offer=round(pv_with_offer, 2),
            loan_npv=round(loan_npv, 2),
            incremental_npv=round(incremental, 2),
            concession_cost=round(concession_cost, 2),
            expected_cashflow=round(loan_npv, 2),   # kept for API compat
            monthly_instalment=round(instalment, 2),
            tenor_months=tenor,
            haircut_pct=haircut_pct,
            is_affordable=is_affordable,
            principal_waived=round(principal_waived, 2),
            interest_waived=round(interest_waived, 2),
            charges_waived=round(charges_waived, 2),
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
        persona: str = "UNKNOWN",
        months_at_180plus: int = 0,
        bureau_asset_value: float = 0.0,
        has_secured_asset: bool   = False,
        override_sale_price: Optional[float] = None,
    ) -> PathValuation:
        """
        Compare all paths on absolute NPV basis; recommend best.

        Paths:
          TDR      = PV(Curve_B) − concession (best offer from grid)
          LEGAL    = P(success) × asset_value discounted − legal_cost
          DEBT_SALE= sale_price_pct × outstanding  (immediate, certain)
          HOLD     = PV(Curve_A) — natural recovery, zero concession
        """
        cfg = self.config

        # ── HOLD: natural recovery baseline ───────────────────────────────────
        hold_result = self.curve_engine.get_curve(
            curve_type="A",
            persona=persona,
            months_at_180plus=months_at_180plus,
            outstanding=outstanding,
            monthly_discount_rate=self._monthly_rate,
            horizon_months=cfg.paydown_horizon_months,
        )
        hold_npv = hold_result.pv_amount

        # ── TDR: pick best offer ──────────────────────────────────────────────
        # Prefer affordable offers; rank by incremental_npv (gain over baseline)
        affordable_offers = [o for o in tdr_offers if o.is_affordable]
        ranked_offers     = affordable_offers or tdr_offers  # fallback if none affordable

        if ranked_offers:
            best_offer = max(ranked_offers, key=lambda o: o.incremental_npv)
            tdr_npv    = best_offer.loan_npv
            tdr_incr   = best_offer.incremental_npv
            tdr_ecf    = best_offer.expected_cashflow
        else:
            best_offer = None
            tdr_npv    = 0.0
            tdr_incr   = 0.0
            tdr_ecf    = 0.0

        # ── Legal ─────────────────────────────────────────────────────────────
        legal_npv, p_legal, legal_viable = self.compute_legal_npv(
            outstanding, stage, bureau_asset_value, has_secured_asset
        )

        # ── Debt Sale ─────────────────────────────────────────────────────────
        sale_npv, sale_price_pct = self.compute_debt_sale_npv(
            outstanding, stage, override_sale_price
        )

        # ── Path selection (absolute NPV basis) ───────────────────────────────
        path_values = {
            "TDR":       tdr_npv,
            "LEGAL":     legal_npv if legal_viable else -np.inf,
            "DEBT_SALE": sale_npv,
            "HOLD":      hold_npv,
        }

        recommended_path = max(path_values, key=path_values.get)
        recommended_npv  = path_values[recommended_path]

        # Build rationale
        rationale = self._build_rationale(
            recommended_path, tdr_ecf, tdr_npv, tdr_incr,
            hold_npv, legal_npv, legal_viable, sale_npv, best_offer
        )

        return PathValuation(
            account_id=account_id,
            outstanding=outstanding,
            stage=stage,
            persona=persona,
            best_tdr_offer_id=best_offer.offer_id if best_offer else "none",
            best_tdr_loan_npv=tdr_npv,
            best_tdr_incremental_npv=tdr_incr,
            best_tdr_expected_cashflow=tdr_ecf,
            hold_npv=round(hold_npv, 2),
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
        Portfolio NPV — aggregate across all accounts.
        Used for capacity-constrained optimisation.
        """
        recommended = sum(p.recommended_path_npv for p in path_valuations)
        total_legal = sum(p.legal_npv for p in path_valuations if p.legal_viable)
        total_sale  = sum(p.debt_sale_npv for p in path_valuations)
        total_hold  = sum(p.hold_npv for p in path_valuations)
        total_tdr   = sum(p.best_tdr_loan_npv for p in path_valuations)

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
            "portfolio_hold_npv":        round(total_hold, 2),
            "accounts_tdr":              sum(1 for p in path_valuations if p.recommended_path == "TDR"),
            "accounts_legal":            sum(1 for p in path_valuations if p.recommended_path == "LEGAL"),
            "accounts_sale":             sum(1 for p in path_valuations if p.recommended_path == "DEBT_SALE"),
            "accounts_hold":             sum(1 for p in path_valuations if p.recommended_path == "HOLD"),
            "by_path":                   {k: round(v, 2) for k, v in by_path.items()},
        }

    # ── PRIVATE ───────────────────────────────────────────────────────────────

    def _build_rationale(
        self,
        path, tdr_ecf, tdr_npv, tdr_incr,
        hold_npv, legal_npv, legal_viable, sale_npv, best_offer
    ) -> str:
        if path == "TDR":
            offer_desc = (
                f"{best_offer.offer_type} haircut={best_offer.haircut_pct:.0%} "
                f"tenor={best_offer.tenor_months}m instalment={best_offer.monthly_instalment:.0f}"
                if best_offer else "no offer"
            )
            return (
                f"TDR recommended: loan_npv={tdr_npv:.0f} "
                f"(incremental over hold={tdr_incr:.0f}). "
                f"Best offer: {offer_desc}."
            )
        elif path == "LEGAL":
            return (
                f"LEGAL recommended: legal_npv={legal_npv:.0f} "
                f"beats TDR ({tdr_npv:.0f}) and sale ({sale_npv:.0f}). "
                f"Secured asset identified in bureau."
            )
        elif path == "DEBT_SALE":
            return (
                f"DEBT SALE recommended: sale_npv={sale_npv:.0f} "
                f"beats TDR ({tdr_npv:.0f}) and hold ({hold_npv:.0f}). "
                f"{'Legal not viable (no asset). ' if not legal_viable else ''}"
                f"Immediate recovery preferred."
            )
        elif path == "HOLD":
            return (
                f"HOLD recommended: natural recovery PV={hold_npv:.0f} "
                f"beats TDR ({tdr_npv:.0f}), "
                f"sale ({sale_npv:.0f}), legal ({legal_npv:.0f}). "
                f"No structured offer justified."
            )
        else:
            return (
                f"Path={path}: TDR={tdr_npv:.0f}, Hold={hold_npv:.0f}, "
                f"Legal={legal_npv:.0f}, Sale={sale_npv:.0f}."
            )
