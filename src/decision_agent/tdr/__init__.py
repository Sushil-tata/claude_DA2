"""TDR / Offer Engine — FICO-inspired NPV + Affordability + Collection Treatment Optimization."""
from .affordability_engine import AffordabilityEngine, AffordabilityConfig, AffordabilityProfile
from .npv_engine import NPVEngine, NPVConfig, LoanNPV, PathValuation
from .offer_generator import OfferGenerator, PolicyConfig, OfferRecommendation, AuditLogRow, TakeUpEstimator
from .legal_queue_manager import (
    LegalQueueManager, LegalConfig, LegalCase, QueueSummary,
    CaseStatus, LegalAction, EscalationReason,
)

__all__ = [
    "AffordabilityEngine", "AffordabilityConfig", "AffordabilityProfile",
    "NPVEngine", "NPVConfig", "LoanNPV", "PathValuation",
    "OfferGenerator", "PolicyConfig", "OfferRecommendation", "AuditLogRow", "TakeUpEstimator",
    "LegalQueueManager", "LegalConfig", "LegalCase", "QueueSummary",
    "CaseStatus", "LegalAction", "EscalationReason",
]
