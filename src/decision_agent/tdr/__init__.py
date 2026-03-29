"""TDR / Offer Engine — Paydown-curve NPV + Affordability + Collection Treatment Optimization."""
from .affordability_engine import AffordabilityEngine, AffordabilityConfig, AffordabilityProfile
from .npv_engine import NPVEngine, NPVConfig, LoanNPV, PathValuation
from .offer_generator import OfferGenerator, PolicyConfig, OfferRecommendation, AuditLogRow
from .paydown_curves import (
    PaydownCurveEngine, PaydownNPVResult, CurvePVResult,
    BalanceDecomposition, compute_waivers,
    PERSONAS, _DEFAULT_CURVE_A, _DEFAULT_CURVE_B,
)
from .legal_queue_manager import (
    LegalQueueManager, LegalConfig, LegalCase, QueueSummary,
    CaseStatus, LegalAction, EscalationReason,
)

__all__ = [
    "AffordabilityEngine", "AffordabilityConfig", "AffordabilityProfile",
    "NPVEngine", "NPVConfig", "LoanNPV", "PathValuation",
    "OfferGenerator", "PolicyConfig", "OfferRecommendation", "AuditLogRow",
    "PaydownCurveEngine", "PaydownNPVResult", "CurvePVResult",
    "BalanceDecomposition", "compute_waivers", "PERSONAS",
    "LegalQueueManager", "LegalConfig", "LegalCase", "QueueSummary",
    "CaseStatus", "LegalAction", "EscalationReason",
]
