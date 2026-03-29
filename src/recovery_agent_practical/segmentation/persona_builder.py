"""
Persona Builder - Rule-Based Customer Segmentation
===================================================

Assigns recovery personas using only historical behavioral signals.
No predictive models - pure rule-based logic.

5 Personas:
-----------
1. ActivePayer: Regular payment activity, responsive to contact
2. SelectiveDefaulter: Has capacity but chooses not to pay
3. LiquidityConstrained: Wants to pay but lacks funds
4. Strategic: Sophisticated avoidance behavior
5. Dormant: No engagement despite contact attempts

Methodology:
-----------
Calculate 4 behavioral axes from historical data:
  - Payment behavior score (0-100)
  - Engagement score (0-100)
  - Capacity score (0-100)
  - Avoidance score (0-100)

Apply decision tree rules to assign persona.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PersonaAssignment:
    """Persona assignment with supporting scores"""
    account_id: str
    persona: str  # ACTIVE_PAYER | SELECTIVE_DEFAULTER | LIQUIDITY_CONSTRAINED | STRATEGIC | DORMANT

    # Underlying axis scores (0-100)
    payment_behavior_score: float
    engagement_score: float
    capacity_score: float
    avoidance_score: float

    # Supporting metrics
    confidence_level: str  # HIGH | MEDIUM | LOW
    data_completeness_pct: float
    flags: list[str]  # Additional context flags


# ─────────────────────────────────────────────────────────────────────────────
# PERSONA BUILDER
# ─────────────────────────────────────────────────────────────────────────────

class PersonaBuilder:
    """
    Rule-based persona assignment using historical behavior only.
    Uses stage-wise windows: 0-30, 31-90, 91-180, 181-365 days.
    """

    def __init__(self, payment_threshold: float = 500.0):
        """
        Args:
            payment_threshold: Minimum THB to count as meaningful payment
        """
        self.payment_threshold = payment_threshold

    def assign_persona(self, account: pd.Series) -> PersonaAssignment:
        """
        Assign persona based on behavioral axes.

        Expected fields in account:
        - Payment fields: payment_count_30d, payment_count_90d, payment_count_180d,
                         payment_amt_30d, payment_amt_90d, payment_amt_180d,
                         days_since_last_payment, ptp_kept_rate
        - Engagement fields: call_response_rate, sms_response_rate,
                            contacts_made_30d, contacts_made_90d
        - Capacity fields: bureau_monthly_instalment, bureau_total_outstanding,
                          balance, principal_outstanding, last_payment_amount
        - Avoidance fields: wrong_number_flag, dispute_flag, lawyer_mentioned,
                           sms_opt_out, last_contact_outcome
        """
        account_id = str(account.get("account_id", "unknown"))

        # Calculate behavioral axes
        payment_score = self._calculate_payment_behavior(account)
        engagement_score = self._calculate_engagement(account)
        capacity_score = self._calculate_capacity(account)
        avoidance_score = self._calculate_avoidance(account)

        # Assign persona using decision tree
        persona = self._apply_decision_tree(
            payment_score, engagement_score, capacity_score, avoidance_score
        )

        # Calculate confidence based on data completeness
        completeness, confidence = self._assess_confidence(account)

        # Generate context flags
        flags = self._generate_flags(account, payment_score, engagement_score, capacity_score, avoidance_score)

        return PersonaAssignment(
            account_id=account_id,
            persona=persona,
            payment_behavior_score=round(payment_score, 2),
            engagement_score=round(engagement_score, 2),
            capacity_score=round(capacity_score, 2),
            avoidance_score=round(avoidance_score, 2),
            confidence_level=confidence,
            data_completeness_pct=round(completeness, 2),
            flags=flags
        )

    # ── BEHAVIORAL AXES ───────────────────────────────────────────────────────

    def _calculate_payment_behavior(self, account: pd.Series) -> float:
        """
        Payment behavior score (0-100).
        Higher = more payment activity.

        Components:
        - Payment frequency (40%): weighted by recency
        - Payment amount ratio (30%): payments vs. balance
        - PTP kept rate (20%): promise-to-pay reliability
        - Recency (10%): days since last payment
        """
        score = 0.0

        # Payment frequency (weighted by recency)
        p30 = float(account.get("payment_count_30d", 0) or 0)
        p90 = float(account.get("payment_count_90d", 0) or 0)
        p180 = float(account.get("payment_count_180d", 0) or 0)

        # Weighted frequency: recent payments matter more
        freq = (p30 * 0.5 + p90 * 0.3 + p180 * 0.2)
        freq_score = min(100, freq * 20)  # 5+ payments = 100
        score += freq_score * 0.4

        # Payment amount ratio
        amt_90d = float(account.get("payment_amt_90d", 0) or 0)
        balance = float(account.get("balance", 1) or 1)
        payment_ratio = amt_90d / balance if balance > 0 else 0
        amt_score = min(100, payment_ratio * 100)  # 100% coverage = 100
        score += amt_score * 0.3

        # PTP kept rate
        ptp_rate = float(account.get("ptp_kept_rate", 0) or 0)
        score += ptp_rate * 100 * 0.2

        # Recency
        days_since = float(account.get("days_since_last_payment", 999) or 999)
        recency_score = max(0, 100 - (days_since / 3.65))  # Linear decay over 365d
        score += recency_score * 0.1

        return min(100, max(0, score))

    def _calculate_engagement(self, account: pd.Series) -> float:
        """
        Engagement score (0-100).
        Higher = more responsive to contact.

        Components:
        - Call response rate (40%)
        - SMS response rate (30%)
        - Contact recency (20%)
        - Contact frequency acceptance (10%): answers vs. attempts
        """
        score = 0.0

        # Call response rate
        call_rate = float(account.get("call_response_rate", 0) or 0)
        score += call_rate * 100 * 0.4

        # SMS response rate
        sms_rate = float(account.get("sms_response_rate", 0) or 0)
        score += sms_rate * 100 * 0.3

        # Contact recency
        days_since_contact = float(account.get("days_since_last_contact", 999) or 999)
        recency_score = max(0, 100 - (days_since_contact / 0.9))  # Decay over 90d
        score += recency_score * 0.2

        # Contact frequency acceptance
        contacts_made = float(account.get("contacts_made_30d", 0) or 0)
        calls_connected = float(account.get("calls_connected", 0) or 0)
        if contacts_made > 0:
            acceptance_rate = calls_connected / contacts_made
            score += min(100, acceptance_rate * 100) * 0.1

        return min(100, max(0, score))

    def _calculate_capacity(self, account: pd.Series) -> float:
        """
        Capacity score (0-100).
        Higher = more financial capacity to pay.

        Components:
        - Bureau DSR (40%): debt service ratio
        - Balance burden (30%): CardX balance vs. total exposure
        - Last payment size (20%): shows payment capacity
        - Bureau velocity (10%): new loan activity
        """
        score = 0.0

        # Bureau DSR
        bureau_instalment = float(account.get("bureau_monthly_instalment", 0) or 0)
        # Assume 30% income proxy from bureau_total_outstanding / 36
        total_outstanding = float(account.get("bureau_total_outstanding", 0) or 0)
        if total_outstanding > 0:
            implied_income = total_outstanding / 3  # Conservative estimate
            dsr = bureau_instalment / implied_income if implied_income > 0 else 1.0
            dsr_score = max(0, 100 - (dsr * 100))  # Lower DSR = higher capacity
            score += dsr_score * 0.4
        else:
            # No bureau data = use CardX balance as proxy
            balance = float(account.get("balance", 0) or 0)
            if balance < 50000:
                score += 70 * 0.4  # Small balance = likely has capacity
            elif balance < 200000:
                score += 50 * 0.4
            else:
                score += 30 * 0.4

        # Balance burden
        cardx_balance = float(account.get("balance", 0) or 0)
        if total_outstanding > 0:
            burden_ratio = cardx_balance / total_outstanding
            burden_score = max(0, 100 - (burden_ratio * 100))
            score += burden_score * 0.3
        else:
            score += 50 * 0.3  # Neutral if no bureau data

        # Last payment size
        last_payment = float(account.get("last_payment_amount", 0) or 0)
        if last_payment >= self.payment_threshold:
            payment_score = min(100, (last_payment / cardx_balance) * 100) if cardx_balance > 0 else 50
            score += payment_score * 0.2

        # Bureau velocity
        new_loan_12m = int(account.get("bureau_new_loan_12m", 0) or 0)
        if new_loan_12m == 0:
            score += 30 * 0.1  # No new loans = may be stressed
        elif new_loan_12m == 1:
            score += 70 * 0.1  # One new loan = accessing credit (capacity)
        else:
            score += 50 * 0.1  # Multiple = credit seeking

        return min(100, max(0, score))

    def _calculate_avoidance(self, account: pd.Series) -> float:
        """
        Avoidance score (0-100).
        Higher = more signs of strategic avoidance.

        Components:
        - Wrong number flag (25%)
        - Dispute/complaint flag (25%)
        - Lawyer mentioned (20%)
        - SMS opt-out (15%)
        - Refused to engage outcome (15%)
        """
        score = 0.0

        # Wrong number flag
        if account.get("wrong_number_flag", False):
            score += 25

        # Dispute flag
        if account.get("dispute_flag", False) or account.get("complaint_flag", False):
            score += 25

        # Lawyer mentioned
        if account.get("lawyer_mentioned", False) or account.get("legal_representation_flag", False):
            score += 20

        # SMS opt-out
        if account.get("sms_opt_out", False):
            score += 15

        # Last contact outcome = refused
        last_outcome = str(account.get("last_contact_outcome", "")).lower()
        if "refuse" in last_outcome or "hostile" in last_outcome or "legal" in last_outcome:
            score += 15

        return min(100, max(0, score))

    # ── DECISION TREE ─────────────────────────────────────────────────────────

    def _apply_decision_tree(
        self,
        payment_score: float,
        engagement_score: float,
        capacity_score: float,
        avoidance_score: float
    ) -> str:
        """
        Decision tree to assign persona based on axis scores.

        Logic:
        1. High avoidance (>50) → STRATEGIC
        2. Low engagement (<30) → DORMANT
        3. High payment (>60) → ACTIVE_PAYER
        4. High capacity (>50) + low payment (<40) → SELECTIVE_DEFAULTER
        5. Low capacity (<40) + moderate engagement (>30) → LIQUIDITY_CONSTRAINED
        6. Default → SELECTIVE_DEFAULTER
        """
        # Strategic avoidance
        if avoidance_score > 50:
            return "STRATEGIC"

        # Dormant (no engagement despite contact)
        if engagement_score < 30:
            return "DORMANT"

        # Active payer
        if payment_score > 60:
            return "ACTIVE_PAYER"

        # Selective defaulter (has capacity but not paying)
        if capacity_score > 50 and payment_score < 40:
            return "SELECTIVE_DEFAULTER"

        # Liquidity constrained (wants to engage but can't pay)
        if capacity_score < 40 and engagement_score > 30:
            return "LIQUIDITY_CONSTRAINED"

        # Default to selective defaulter
        return "SELECTIVE_DEFAULTER"

    # ── CONFIDENCE & FLAGS ────────────────────────────────────────────────────

    def _assess_confidence(self, account: pd.Series) -> tuple[float, str]:
        """
        Assess confidence based on data completeness.
        Returns (completeness_pct, confidence_level).
        """
        required_fields = [
            "payment_count_30d", "payment_count_90d", "payment_count_180d",
            "call_response_rate", "sms_response_rate",
            "bureau_monthly_instalment", "balance",
            "days_since_last_payment", "days_since_last_contact"
        ]

        present_count = sum(
            1 for f in required_fields
            if f in account.index and pd.notna(account[f])
        )

        completeness = (present_count / len(required_fields)) * 100

        if completeness >= 80:
            confidence = "HIGH"
        elif completeness >= 60:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        return completeness, confidence

    def _generate_flags(
        self,
        account: pd.Series,
        payment_score: float,
        engagement_score: float,
        capacity_score: float,
        avoidance_score: float
    ) -> list[str]:
        """Generate context flags for transparency."""
        flags = []

        # Payment flags
        if payment_score > 70:
            flags.append("high_payment_activity")
        elif payment_score < 20:
            flags.append("no_payment_activity")

        # Engagement flags
        if engagement_score > 70:
            flags.append("highly_responsive")
        elif engagement_score < 20:
            flags.append("unresponsive")

        # Capacity flags
        if capacity_score > 70:
            flags.append("high_capacity")
        elif capacity_score < 30:
            flags.append("low_capacity")

        # Avoidance flags
        if avoidance_score > 50:
            flags.append("avoidance_behavior")

        # Bureau flags
        if pd.notna(account.get("bureau_delinquent_other")) and account.get("bureau_delinquent_other", 0) > 0:
            flags.append("delinquent_elsewhere")

        if pd.notna(account.get("bureau_secured_loan_flag")) and account.get("bureau_secured_loan_flag", False):
            flags.append("has_secured_assets")

        # Stage flags
        stage = str(account.get("stage", "")).upper()
        if stage == "CHARGEOFF":
            flags.append("chargeoff_stage")

        return flags

    def assign_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Assign personas for a batch of accounts.
        Returns DataFrame with persona assignments.
        """
        results = []
        for _, row in df.iterrows():
            assignment = self.assign_persona(row)
            results.append({
                "account_id": assignment.account_id,
                "persona": assignment.persona,
                "payment_behavior_score": assignment.payment_behavior_score,
                "engagement_score": assignment.engagement_score,
                "capacity_score": assignment.capacity_score,
                "avoidance_score": assignment.avoidance_score,
                "confidence_level": assignment.confidence_level,
                "data_completeness_pct": assignment.data_completeness_pct,
                "flags": "; ".join(assignment.flags),
            })

        return pd.DataFrame(results)
