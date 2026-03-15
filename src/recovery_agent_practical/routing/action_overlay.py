"""
Action Overlay Router
=====================

Rule-based action recommendation using:
  persona × stage × balance_band × months_since_chargeoff × score_band

NO NPV optimization - pure rule-based routing.

Action Families:
---------------
SETTLEMENT_LUMP: One-time lump sum settlement offer
SETTLEMENT_PLAN: Structured payment plan over 6-12 months
AGENCY: Refer to external collection agency
LEGAL_REVIEW: Escalate to legal team for review
HOLD: No action - monitor only

Routing Logic:
-------------
1. Stage filter: NPL vs. CHARGEOFF
2. Persona mapping: Willingness + capacity signals
3. Balance band: Small (<50K), Medium (50-200K), Large (>200K)
4. Staleness: months_since_chargeoff (fresh vs. stale)
5. Score band: HOT/WARM/COLD/FROZEN from recovery scorecard
"""

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ActionRecommendation:
    """Action recommendation for one account"""
    account_id: str
    recommended_action: str  # SETTLEMENT_LUMP | SETTLEMENT_PLAN | AGENCY | LEGAL_REVIEW | HOLD

    # Routing inputs
    persona: str
    stage: str
    balance_band: str  # SMALL | MEDIUM | LARGE
    months_since_chargeoff: int
    score_band: str  # HOT | WARM | COLD | FROZEN

    # Supporting context
    reasoning: str  # Explanation of routing decision
    priority_tier: str  # TIER_1 | TIER_2 | TIER_3 (for work allocation)
    contact_channel: str  # PHONE | SMS | EMAIL | LEGAL_NOTICE
    offer_type: Optional[str] = None  # HAIRCUT_30 | HAIRCUT_50 | PLAN_6M | PLAN_12M

    # Metadata
    routing_version: str = "v1.0"


# ─────────────────────────────────────────────────────────────────────────────
# BALANCE BANDS
# ─────────────────────────────────────────────────────────────────────────────

def _get_balance_band(balance: float) -> str:
    """Categorize balance into bands"""
    if balance < 50_000:
        return "SMALL"
    elif balance < 200_000:
        return "MEDIUM"
    else:
        return "LARGE"


# ─────────────────────────────────────────────────────────────────────────────
# ACTION OVERLAY ROUTER
# ─────────────────────────────────────────────────────────────────────────────

class ActionOverlayRouter:
    """
    Rule-based action routing using persona, stage, balance, staleness, score.

    Decision tree approach - no optimization, pure business rules.
    """

    def __init__(
        self,
        small_balance_threshold: float = 50_000,
        medium_balance_threshold: float = 200_000,
        stale_chargeoff_months: int = 18,  # Chargeoff > 18 months = stale
    ):
        """
        Args:
            small_balance_threshold: Threshold for SMALL balance band
            medium_balance_threshold: Threshold for MEDIUM vs. LARGE
            stale_chargeoff_months: Months after which chargeoff is stale
        """
        self.small_threshold = small_balance_threshold
        self.medium_threshold = medium_balance_threshold
        self.stale_months = stale_chargeoff_months

    def route_action(
        self,
        account_id: str,
        persona: str,
        stage: str,
        balance: float,
        months_since_chargeoff: int,
        score_band: str,
        has_secured_assets: bool = False,
        bureau_delinquent_other: bool = False,
    ) -> ActionRecommendation:
        """
        Route account to action based on business rules.

        Args:
            account_id: Account identifier
            persona: ACTIVE_PAYER | SELECTIVE_DEFAULTER | LIQUIDITY_CONSTRAINED | STRATEGIC | DORMANT
            stage: SM | NPL | CHARGEOFF
            balance: Outstanding balance THB
            months_since_chargeoff: Months since chargeoff (0 if not chargeoff)
            score_band: HOT | WARM | COLD | FROZEN
            has_secured_assets: Bureau secured loan flag
            bureau_delinquent_other: Delinquent at other lenders

        Returns:
            ActionRecommendation with routing decision
        """
        balance_band = _get_balance_band(balance)
        stage = stage.upper()

        # ─────────────────────────────────────────────────────────────────────
        # STAGE 1: NPL (NOT CHARGEOFF YET)
        # ─────────────────────────────────────────────────────────────────────
        if stage in ["SM", "NPL"]:
            return self._route_npl(
                account_id, persona, stage, balance_band, score_band,
                has_secured_assets, bureau_delinquent_other
            )

        # ─────────────────────────────────────────────────────────────────────
        # STAGE 2: CHARGEOFF
        # ─────────────────────────────────────────────────────────────────────
        elif stage == "CHARGEOFF":
            is_stale = months_since_chargeoff >= self.stale_months

            return self._route_chargeoff(
                account_id, persona, balance_band, months_since_chargeoff,
                score_band, is_stale, has_secured_assets, bureau_delinquent_other
            )

        else:
            # Unknown stage - default to HOLD
            return ActionRecommendation(
                account_id=account_id,
                recommended_action="HOLD",
                persona=persona,
                stage=stage,
                balance_band=balance_band,
                months_since_chargeoff=months_since_chargeoff,
                score_band=score_band,
                reasoning=f"Unknown stage: {stage}. Default to HOLD.",
                priority_tier="TIER_3",
                contact_channel="NONE",
            )

    # ── NPL ROUTING ───────────────────────────────────────────────────────────

    def _route_npl(
        self,
        account_id: str,
        persona: str,
        stage: str,
        balance_band: str,
        score_band: str,
        has_secured_assets: bool,
        bureau_delinquent_other: bool,
    ) -> ActionRecommendation:
        """Route NPL accounts (not yet chargeoff)"""

        # Rule 1: ACTIVE_PAYER → Structured plan (likely to comply)
        if persona == "ACTIVE_PAYER":
            return ActionRecommendation(
                account_id=account_id,
                recommended_action="SETTLEMENT_PLAN",
                persona=persona,
                stage=stage,
                balance_band=balance_band,
                months_since_chargeoff=0,
                score_band=score_band,
                reasoning="Active payer persona with payment history - offer structured plan",
                priority_tier="TIER_1",
                contact_channel="PHONE",
                offer_type="PLAN_12M",
            )

        # Rule 2: SELECTIVE_DEFAULTER + HOT/WARM → Settlement lump (has capacity)
        if persona == "SELECTIVE_DEFAULTER" and score_band in ["HOT", "WARM"]:
            haircut = "HAIRCUT_30" if balance_band == "LARGE" else "HAIRCUT_50"
            return ActionRecommendation(
                account_id=account_id,
                recommended_action="SETTLEMENT_LUMP",
                persona=persona,
                stage=stage,
                balance_band=balance_band,
                months_since_chargeoff=0,
                score_band=score_band,
                reasoning="Selective defaulter with capacity - push for lump sum settlement",
                priority_tier="TIER_1",
                contact_channel="PHONE",
                offer_type=haircut,
            )

        # Rule 3: LIQUIDITY_CONSTRAINED + HOT/WARM → Payment plan
        if persona == "LIQUIDITY_CONSTRAINED" and score_band in ["HOT", "WARM"]:
            return ActionRecommendation(
                account_id=account_id,
                recommended_action="SETTLEMENT_PLAN",
                persona=persona,
                stage=stage,
                balance_band=balance_band,
                months_since_chargeoff=0,
                score_band=score_band,
                reasoning="Liquidity constrained but willing - offer affordable plan",
                priority_tier="TIER_1",
                contact_channel="PHONE",
                offer_type="PLAN_6M",
            )

        # Rule 4: STRATEGIC + has_secured_assets → Legal review
        if persona == "STRATEGIC" and has_secured_assets:
            return ActionRecommendation(
                account_id=account_id,
                recommended_action="LEGAL_REVIEW",
                persona=persona,
                stage=stage,
                balance_band=balance_band,
                months_since_chargeoff=0,
                score_band=score_band,
                reasoning="Strategic avoidance with secured assets - escalate to legal",
                priority_tier="TIER_2",
                contact_channel="LEGAL_NOTICE",
            )

        # Rule 5: DORMANT + COLD/FROZEN → Agency referral
        if persona == "DORMANT" and score_band in ["COLD", "FROZEN"]:
            return ActionRecommendation(
                account_id=account_id,
                recommended_action="AGENCY",
                persona=persona,
                stage=stage,
                balance_band=balance_band,
                months_since_chargeoff=0,
                score_band=score_band,
                reasoning="Dormant account with low recovery score - refer to agency",
                priority_tier="TIER_3",
                contact_channel="PHONE",
            )

        # Default NPL: Settlement plan
        return ActionRecommendation(
            account_id=account_id,
            recommended_action="SETTLEMENT_PLAN",
            persona=persona,
            stage=stage,
            balance_band=balance_band,
            months_since_chargeoff=0,
            score_band=score_band,
            reasoning="NPL account - attempt structured settlement",
            priority_tier="TIER_2",
            contact_channel="PHONE",
            offer_type="PLAN_12M",
        )

    # ── CHARGEOFF ROUTING ─────────────────────────────────────────────────────

    def _route_chargeoff(
        self,
        account_id: str,
        persona: str,
        balance_band: str,
        months_since_chargeoff: int,
        score_band: str,
        is_stale: bool,
        has_secured_assets: bool,
        bureau_delinquent_other: bool,
    ) -> ActionRecommendation:
        """Route chargeoff accounts"""

        # Rule 1: Fresh chargeoff + HOT → Aggressive settlement
        if months_since_chargeoff <= 6 and score_band == "HOT":
            return ActionRecommendation(
                account_id=account_id,
                recommended_action="SETTLEMENT_LUMP",
                persona=persona,
                stage="CHARGEOFF",
                balance_band=balance_band,
                months_since_chargeoff=months_since_chargeoff,
                score_band=score_band,
                reasoning="Fresh chargeoff with high recovery score - push lump settlement",
                priority_tier="TIER_1",
                contact_channel="PHONE",
                offer_type="HAIRCUT_50",
            )

        # Rule 2: LARGE balance + has_secured_assets → Legal review
        if balance_band == "LARGE" and has_secured_assets:
            return ActionRecommendation(
                account_id=account_id,
                recommended_action="LEGAL_REVIEW",
                persona=persona,
                stage="CHARGEOFF",
                balance_band=balance_band,
                months_since_chargeoff=months_since_chargeoff,
                score_band=score_band,
                reasoning="Large balance with secured assets - legal collection viable",
                priority_tier="TIER_2",
                contact_channel="LEGAL_NOTICE",
            )

        # Rule 3: Stale chargeoff + SMALL balance + FROZEN → HOLD
        if is_stale and balance_band == "SMALL" and score_band == "FROZEN":
            return ActionRecommendation(
                account_id=account_id,
                recommended_action="HOLD",
                persona=persona,
                stage="CHARGEOFF",
                balance_band=balance_band,
                months_since_chargeoff=months_since_chargeoff,
                score_band=score_band,
                reasoning="Stale small balance with frozen score - not economic to pursue",
                priority_tier="TIER_3",
                contact_channel="NONE",
            )

        # Rule 4: ACTIVE_PAYER or LIQUIDITY_CONSTRAINED + WARM/HOT → Payment plan
        if persona in ["ACTIVE_PAYER", "LIQUIDITY_CONSTRAINED"] and score_band in ["HOT", "WARM"]:
            return ActionRecommendation(
                account_id=account_id,
                recommended_action="SETTLEMENT_PLAN",
                persona=persona,
                stage="CHARGEOFF",
                balance_band=balance_band,
                months_since_chargeoff=months_since_chargeoff,
                score_band=score_band,
                reasoning="Willing persona with recovery potential - offer plan",
                priority_tier="TIER_1",
                contact_channel="SMS",
                offer_type="PLAN_6M",
            )

        # Rule 5: SELECTIVE_DEFAULTER + MEDIUM/LARGE → Agency
        if persona == "SELECTIVE_DEFAULTER" and balance_band in ["MEDIUM", "LARGE"]:
            return ActionRecommendation(
                account_id=account_id,
                recommended_action="AGENCY",
                persona=persona,
                stage="CHARGEOFF",
                balance_band=balance_band,
                months_since_chargeoff=months_since_chargeoff,
                score_band=score_band,
                reasoning="Selective defaulter with material balance - agency pressure",
                priority_tier="TIER_2",
                contact_channel="PHONE",
            )

        # Rule 6: STRATEGIC + not stale → Legal review
        if persona == "STRATEGIC" and not is_stale:
            return ActionRecommendation(
                account_id=account_id,
                recommended_action="LEGAL_REVIEW",
                persona=persona,
                stage="CHARGEOFF",
                balance_band=balance_band,
                months_since_chargeoff=months_since_chargeoff,
                score_band=score_band,
                reasoning="Strategic avoidance - escalate to legal",
                priority_tier="TIER_2",
                contact_channel="LEGAL_NOTICE",
            )

        # Rule 7: Stale + COLD/FROZEN → Agency or HOLD
        if is_stale and score_band in ["COLD", "FROZEN"]:
            if balance_band in ["MEDIUM", "LARGE"]:
                return ActionRecommendation(
                    account_id=account_id,
                    recommended_action="AGENCY",
                    persona=persona,
                    stage="CHARGEOFF",
                    balance_band=balance_band,
                    months_since_chargeoff=months_since_chargeoff,
                    score_band=score_band,
                    reasoning="Stale account with material balance - last attempt via agency",
                    priority_tier="TIER_3",
                    contact_channel="PHONE",
                )
            else:
                return ActionRecommendation(
                    account_id=account_id,
                    recommended_action="HOLD",
                    persona=persona,
                    stage="CHARGEOFF",
                    balance_band=balance_band,
                    months_since_chargeoff=months_since_chargeoff,
                    score_band=score_band,
                    reasoning="Stale small balance - not economic",
                    priority_tier="TIER_3",
                    contact_channel="NONE",
                )

        # Default chargeoff: Agency
        return ActionRecommendation(
            account_id=account_id,
            recommended_action="AGENCY",
            persona=persona,
            stage="CHARGEOFF",
            balance_band=balance_band,
            months_since_chargeoff=months_since_chargeoff,
            score_band=score_band,
            reasoning="Chargeoff default - refer to agency",
            priority_tier="TIER_3",
            contact_channel="PHONE",
        )

    def route_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Route actions for a batch of accounts.

        Expected columns:
        - account_id
        - persona
        - stage
        - balance
        - months_since_chargeoff
        - score_band
        - has_secured_assets (optional)
        - bureau_delinquent_other (optional)

        Returns:
            DataFrame with ActionRecommendation columns
        """
        results = []

        for _, row in df.iterrows():
            rec = self.route_action(
                account_id=str(row["account_id"]),
                persona=row["persona"],
                stage=row["stage"],
                balance=float(row["balance"]),
                months_since_chargeoff=int(row.get("months_since_chargeoff", 0)),
                score_band=row["score_band"],
                has_secured_assets=bool(row.get("has_secured_assets", False)),
                bureau_delinquent_other=bool(row.get("bureau_delinquent_other", False)),
            )

            results.append({
                "account_id": rec.account_id,
                "recommended_action": rec.recommended_action,
                "persona": rec.persona,
                "stage": rec.stage,
                "balance_band": rec.balance_band,
                "months_since_chargeoff": rec.months_since_chargeoff,
                "score_band": rec.score_band,
                "reasoning": rec.reasoning,
                "priority_tier": rec.priority_tier,
                "contact_channel": rec.contact_channel,
                "offer_type": rec.offer_type,
                "routing_version": rec.routing_version,
            })

        return pd.DataFrame(results)
