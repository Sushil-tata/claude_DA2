"""
Affordability Engine (v2)
==========================
Uses ONLY confirmed available fields (Q1-D answers).
Never breaks on missing data — degrades gracefully with confidence flag.

Income waterfall (uses highest available tier):
  Tier 1 — Income estimation model output (future — not yet connected)
  Tier 2 — Bureau: secured loan monthly payment / 0.35 DSR assumption
  Tier 3 — Bureau: total monthly instalment / 0.40 DSR back-calc
  Tier 4 — CardX payment history: last_payment_amount × 12 / 0.15 obligation ratio
  Tier 5 — Segment median by DPD bucket + balance tier
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class AffordabilityConfig:
    max_dsr: float = 0.40
    living_expense_pct: float = 0.35
    cardx_share_of_capacity: float = 0.60
    mortgage_dsr_assumption: float = 0.35
    total_obligation_dsr_assumption: float = 0.40
    cardx_payment_obligation_ratio: float = 0.15
    segment_median_income: Dict = field(default_factory=lambda: {
        (0,0):15000,(0,1):25000,(0,2):40000,(0,3):70000,
        (1,0):12000,(1,1):20000,(1,2):35000,(1,3):60000,
        (2,0):10000,(2,1):18000,(2,2):30000,(2,3):55000,
        (3,0): 8000,(3,1):15000,(3,2):25000,(3,3):45000,
        (4,0): 7000,(4,1):12000,(4,2):20000,(4,3):35000,
        (5,0): 6000,(5,1):10000,(5,2):18000,(5,3):30000,
    })


@dataclass
class AffordabilityProfile:
    account_id: str
    estimated_monthly_income: float
    income_tier_used: int
    income_confidence: str
    total_monthly_obligations: float
    estimated_living_expenses: float
    disposable_income: float
    payment_capacity: float
    max_instalment: Dict
    dsr_current: float
    is_over_indebted: bool
    notes: List[str]


class AffordabilityEngine:
    def __init__(self, config: Optional[AffordabilityConfig] = None):
        self.config = config or AffordabilityConfig()

    def assess(self, account: pd.Series) -> AffordabilityProfile:
        c   = self.config
        aid = str(account.get("account_id", "unknown"))

        income, tier, confidence, notes = self._estimate_income(account)
        obligations = self._total_obligations(account)
        living      = income * c.living_expense_pct
        disposable  = max(income - obligations - living, 0)
        capacity    = disposable * c.cardx_share_of_capacity
        dsr         = obligations / max(income, 1)
        over        = dsr > c.max_dsr

        if over:
            notes.append(f"Over-indebted: DSR={dsr:.1%} > {c.max_dsr:.0%}")

        return AffordabilityProfile(
            account_id=aid,
            estimated_monthly_income=round(income, 2),
            income_tier_used=tier,
            income_confidence=confidence,
            total_monthly_obligations=round(obligations, 2),
            estimated_living_expenses=round(living, 2),
            disposable_income=round(disposable, 2),
            payment_capacity=round(capacity, 2),
            max_instalment={t: round(capacity, 2) for t in [12,24,36,48,60]},
            dsr_current=round(dsr, 4),
            is_over_indebted=over,
            notes=notes,
        )

    def assess_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        profiles = [self.assess(row) for _, row in df.iterrows()]
        return pd.DataFrame([{
            "account_id":               p.account_id,
            "estimated_monthly_income": p.estimated_monthly_income,
            "income_tier_used":         p.income_tier_used,
            "income_confidence":        p.income_confidence,
            "total_monthly_obligations":p.total_monthly_obligations,
            "payment_capacity":         p.payment_capacity,
            "dsr_current":              p.dsr_current,
            "is_over_indebted":         p.is_over_indebted,
        } for p in profiles])

    def _estimate_income(self, account: pd.Series):
        c     = self.config
        notes = []

        # Tier 1: income model (future)
        v = account.get("predicted_income")
        if v and not pd.isna(v) and float(v) > 0:
            return float(v), 1, "high", notes

        # Tier 2: bureau secured loan monthly payment
        v = account.get("bureau_secured_monthly_payment")
        if v and not pd.isna(v) and float(v) > 0:
            income = float(v) / c.mortgage_dsr_assumption
            notes.append(f"Tier 2: income from secured loan payment {float(v):,.0f}/mo")
            return income, 2, "medium", notes

        # Tier 3: bureau total monthly instalment
        v = account.get("bureau_monthly_instalment")
        if v and not pd.isna(v) and float(v) > 0:
            income = float(v) / c.total_obligation_dsr_assumption
            notes.append(f"Tier 3: income back-calc from total obligations {float(v):,.0f}/mo")
            return income, 3, "medium", notes

        # Tier 4: CardX payment history proxy
        # If customer paid X on CardX last year, their income must support that
        v = account.get("last_payment_amount")
        if v and not pd.isna(v) and float(v) > 0:
            # CardX payment is roughly 15% of monthly income obligation
            income = float(v) / c.cardx_payment_obligation_ratio
            notes.append(f"Tier 4: income proxy from last payment {float(v):,.0f}")
            return income, 4, "low", notes

        # Tier 5: segment median
        income = self._segment_median(account)
        notes.append("Tier 5: segment median — very low confidence")
        return income, 5, "low", notes

    def _total_obligations(self, account: pd.Series) -> float:
        v = account.get("bureau_monthly_instalment")
        if v and not pd.isna(v):
            return float(v)
        # Fallback: estimate from total paid on CardX
        paid12 = account.get("total_paid_12m", 0) or 0
        return float(paid12) / 12.0

    def _segment_median(self, account: pd.Series) -> float:
        dpd = int(account.get("days_past_due", 0) or 0)
        bal = float(account.get("balance", 0) or 0)
        bucket = int(min(dpd // 30, 5))
        tier   = 0 if bal < 1000 else 1 if bal < 5000 else 2 if bal < 10000 else 3
        return float(self.config.segment_median_income.get((bucket, tier), 15000))
