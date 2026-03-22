"""
Debt Collection Decision Intelligence — Module Exports
======================================================
Exposes the full NBO/NBA stack for collection recovery:

  DebtorPersonaSegmentation  — Rule + K-Means hybrid segmentation
  PersonaStrategyRecommender — Strategy assignment per persona
  NextBestActionEngine       — Channel / timing recommendation
  NextBestOfferEngine        — Offer type recommendation

  RecoveryAmountModel        — Two-stage beta regression: E[Amount | Recovery, Segment]
  RecoveryAmountModelSpark   — PySpark / Databricks wrapper
  DiscountElasticityModel    — P(Recovery | Segment, d) logistic GBM
  DiscountElasticityModelSpark — PySpark / Databricks wrapper
  ERVEngine                  — ERV assembler + d* optimiser
  ERVEngineSpark             — PySpark / Databricks wrapper
"""

from .persona_segmentation import DebtorPersonaSegmentation, PersonaStrategyRecommender
from .next_best_action import NextBestActionEngine, NextBestOfferEngine
from .recovery_amount_model import RecoveryAmountModel, RecoveryAmountModelSpark
from .discount_elasticity import DiscountElasticityModel, DiscountElasticityModelSpark
from .erv_engine import ERVEngine, ERVEngineSpark

__all__ = [
    # Segmentation
    "DebtorPersonaSegmentation",
    "PersonaStrategyRecommender",
    # Action & Offer
    "NextBestActionEngine",
    "NextBestOfferEngine",
    # Recovery Amount (Beta Regression)
    "RecoveryAmountModel",
    "RecoveryAmountModelSpark",
    # Discount Elasticity
    "DiscountElasticityModel",
    "DiscountElasticityModelSpark",
    # ERV Engine
    "ERVEngine",
    "ERVEngineSpark",
]
