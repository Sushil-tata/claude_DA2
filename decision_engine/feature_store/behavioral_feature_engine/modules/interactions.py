"""
Feature Interactions Module - Non-Linear Behavioral Patterns

Captures interactions between feature domains that reveal important patterns
not visible when looking at features independently.

Version: BFE_v1.2
Author: Behavioral Feature Engineering Team
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from datetime import date
import logging

logger = logging.getLogger(__name__)


class InteractionFeatureEngine:
    """
    Generate interaction features across behavioral domains.
    
    Key interaction types:
    1. RFM × Delinquency: High-value customers behaving badly (anomalies)
    2. Payment × Delinquency: Payment effort vs delinquency outcomes
    3. Vintage × Performance: Early vs late delinquency patterns
    4. Cure × Delinquency: Recovery propensity patterns
    """
    
    VERSION = "BFE_v1.2"
    MODULE_NAME = "bfe.interactions"
    
    def __init__(self):
        """Initialize interaction feature engine."""
        self.logger = logging.getLogger(__name__)
    
    def compute_features(
        self,
        base_features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compute interaction features from base features.
        
        Args:
            base_features: Dictionary with all computed features from other modules
                          (delinquency, payment, vintage modules)
            
        Returns:
            Dictionary of interaction features
        """
        features = {}
        
        # RFM × Delinquency interactions
        features.update(self._compute_rfm_delinquency_interactions(base_features))
        
        # Payment × Delinquency interactions
        features.update(self._compute_payment_delinquency_interactions(base_features))
        
        # Vintage × Performance interactions
        features.update(self._compute_vintage_performance_interactions(base_features))
        
        # Cure × Delinquency interactions
        features.update(self._compute_cure_delinquency_interactions(base_features))
        
        return features
    
    def _compute_rfm_delinquency_interactions(
        self,
        base_features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compute RFM × Delinquency interaction features.
        
        Key insights:
        - High RFM + High DPD = Anomaly (urgent recovery opportunity)
        - Low RFM + Stable = Low value but reliable
        - CHAMPIONS now delinquent = Life event, high recovery probability
        """
        features = {}
        
        # Get base features
        rfm_score = base_features.get("payment.rfm_composite_score", np.nan)
        rfm_segment = base_features.get("payment.rfm_segment", "UNKNOWN")
        rfm_detailed = base_features.get("payment.rfm_detailed_segment", "UNKNOWN")
        
        dpd_current = base_features.get("delinquency.dpd_current", np.nan)
        delinq_regime = base_features.get("delinquency.delinquency_regime", "UNKNOWN")
        
        # Interaction 1: High-value customer delinquent flag
        features["interaction_rfm_high_value_delinquent"] = (
            rfm_segment == "HIGH_VALUE" and dpd_current >= 30
        ) if not np.isnan(dpd_current) else np.nan
        
        # Interaction 2: Champions at risk (most urgent)
        features["interaction_champions_at_risk"] = (
            rfm_detailed == "CHAMPIONS" and dpd_current >= 30
        ) if not np.isnan(dpd_current) else np.nan
        
        # Interaction 3: Can't lose them severely delinquent (was loyal, now NPL)
        features["interaction_cant_lose_severely_delinquent"] = (
            rfm_detailed == "CANT_LOSE_THEM" and dpd_current >= 90
        ) if not np.isnan(dpd_current) else np.nan
        
        # Interaction 4: Low value but stable (never delinquent despite low RFM)
        features["interaction_low_value_stable"] = (
            rfm_segment in ["LOW_VALUE", "VERY_LOW_VALUE"] and 
            delinq_regime == "STABLE"
        )
        
        # Interaction 5: RFM score × DPD ratio (value/risk ratio)
        if not np.isnan(rfm_score) and not np.isnan(dpd_current) and dpd_current > 0:
            features["interaction_rfm_dpd_ratio"] = rfm_score / (dpd_current / 30.0)
        else:
            features["interaction_rfm_dpd_ratio"] = np.nan
        
        # Interaction 6: High RFM deteriorating (warning sign)
        features["interaction_high_rfm_deteriorating"] = (
            rfm_score >= 12 and delinq_regime == "DETERIORATING"
        ) if not np.isnan(rfm_score) else np.nan
        
        return features
    
    def _compute_payment_delinquency_interactions(
        self,
        base_features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compute Payment × Delinquency interaction features.
        
        Key insights:
        - Payments increasing + DPD increasing = Futile effort (structural problem)
        - Full payer now delinquent = Payment shock
        - Payment ratio high + DPD high = Interest/penalty problem
        """
        features = {}
        
        # Get base features
        payment_ratio_mean = base_features.get("payment.payment_ratio_6M_mean", np.nan)
        payment_ratio_slope = base_features.get("payment.payment_ratio_6M_slope", np.nan)
        payment_regime = base_features.get("payment.payment_regime", "UNKNOWN")
        
        dpd_current = base_features.get("delinquency.dpd_current", np.nan)
        dpd_slope = base_features.get("delinquency.dpd_6M_slope", np.nan)
        delinq_regime = base_features.get("delinquency.delinquency_regime", "UNKNOWN")
        
        # Interaction 1: Payment increasing but DPD increasing (futile effort)
        features["interaction_futile_payment_effort"] = (
            payment_ratio_slope > 0 and dpd_slope > 5
        ) if (not np.isnan(payment_ratio_slope) and not np.isnan(dpd_slope)) else np.nan
        
        # Interaction 2: Full payer now delinquent (payment shock)
        features["interaction_full_payer_now_delinquent"] = (
            payment_regime == "CONSISTENT_FULL_PAYER" and dpd_current >= 30
        ) if not np.isnan(dpd_current) else np.nan
        
        # Interaction 3: Erratic payer but stable DPD (payments don't affect delinquency)
        features["interaction_erratic_payer_stable_dpd"] = (
            payment_regime == "ERRATIC_PAYER" and delinq_regime == "STABLE"
        )
        
        # Interaction 4: Payment effort vs delinquency divergence
        if (not np.isnan(payment_ratio_slope) and not np.isnan(dpd_slope)):
            # Positive = payments improving, negative = delinquency worsening
            # Large negative value = divergence (bad sign)
            features["interaction_payment_dpd_divergence"] = (
                payment_ratio_slope - (dpd_slope / 10.0)
            )
        else:
            features["interaction_payment_dpd_divergence"] = np.nan
        
        # Interaction 5: High payment ratio but high DPD (debt growing faster)
        features["interaction_high_payment_high_dpd"] = (
            payment_ratio_mean >= 0.8 and dpd_current >= 60
        ) if (not np.isnan(payment_ratio_mean) and not np.isnan(dpd_current)) else np.nan
        
        return features
    
    def _compute_vintage_performance_interactions(
        self,
        base_features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compute Vintage × Performance interaction features.
        
        Key insights:
        - Early delinquency (first 6M) = Underwriting failure (high risk)
        - Late first delinquency (24M+) = Life event (recoverable)
        - Aged account deteriorating = Warning sign
        """
        features = {}
        
        # Get base features
        months_on_book = base_features.get("vintage.vintage_months_on_book", np.nan)
        early_delinq_flag = base_features.get("vintage.vintage_early_delinquency_flag", np.nan)
        months_since_first_delinq = base_features.get(
            "vintage.vintage_months_since_first_delinquency", np.nan
        )
        lifecycle_stage = base_features.get("vintage.vintage_lifecycle_stage", "UNKNOWN")
        
        dpd_current = base_features.get("delinquency.dpd_current", np.nan)
        delinq_regime = base_features.get("delinquency.delinquency_regime", "UNKNOWN")
        
        # Interaction 1: Early delinquency flag (delinquent in first 6 months)
        features["interaction_early_delinquency"] = early_delinq_flag
        
        # Interaction 2: Seasoned first delinquency (first-time delinquent after 24M)
        features["interaction_seasoned_first_delinquency"] = (
            months_on_book >= 24 and 
            months_since_first_delinq <= 3
        ) if (not np.isnan(months_on_book) and not np.isnan(months_since_first_delinq)) else np.nan
        
        # Interaction 3: Mature account deteriorating (warning)
        features["interaction_mature_deteriorating"] = (
            lifecycle_stage in ["MATURE", "AGED"] and 
            delinq_regime == "DETERIORATING"
        )
        
        # Interaction 4: New account severely delinquent (very high risk)
        features["interaction_new_account_severe"] = (
            months_on_book <= 12 and dpd_current >= 90
        ) if (not np.isnan(months_on_book) and not np.isnan(dpd_current)) else np.nan
        
        return features
    
    def _compute_cure_delinquency_interactions(
        self,
        base_features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compute Cure × Delinquency interaction features.
        
        Key insights:
        - Multiple cures = Serial curer (recoverable but high maintenance)
        - Cure followed by immediate re-delinquency = Structural problem
        - Never cured + high DPD = Lost cause
        """
        features = {}
        
        # Get base features
        cure_flag_6M = base_features.get("delinquency.cure_flag_6M", np.nan)
        cure_flag_12M = base_features.get("delinquency.cure_flag_12M", np.nan)
        re_delinq_flag_6M = base_features.get("delinquency.re_delinquency_flag_6M", np.nan)
        
        dpd_current = base_features.get("delinquency.dpd_current", np.nan)
        payment_regime = base_features.get("payment.payment_regime", "UNKNOWN")
        
        # Interaction 1: Serial curer flag (cured multiple times)
        features["interaction_serial_curer"] = (
            cure_flag_12M == True and 
            payment_regime in ["ERRATIC_PAYER", "MINIMAL_PAYER"]
        ) if cure_flag_12M is not np.nan else np.nan
        
        # Interaction 2: Rapid re-delinquency (cured then immediately delinquent)
        features["interaction_rapid_re_delinquency"] = (
            cure_flag_6M == True and 
            re_delinq_flag_6M == True
        ) if (cure_flag_6M is not np.nan and re_delinq_flag_6M is not np.nan) else np.nan
        
        # Interaction 3: Never cured + severely delinquent (lost cause)
        features["interaction_never_cured_severe"] = (
            cure_flag_12M == False and 
            dpd_current >= 90
        ) if (cure_flag_12M is not np.nan and not np.isnan(dpd_current)) else np.nan
        
        return features
    
    @staticmethod
    def get_feature_metadata() -> Dict[str, Dict[str, Any]]:
        """
        Get metadata for all interaction features.
        
        Returns:
            Dictionary mapping feature names to metadata
        """
        return {
            "interaction_rfm_high_value_delinquent": {
                "tier": 1,
                "definition": "High RFM customer currently delinquent (anomaly)",
                "feature_type": "binary",
                "signal_direction": "True = high recovery opportunity",
                "business_meaning": "Good customer hit a bump - urgent outreach needed",
                "use_cases": ["Collections prioritization", "Recovery prediction"]
            },
            "interaction_champions_at_risk": {
                "tier": 1,
                "definition": "CHAMPIONS segment (RFM top tier) now 30+ DPD",
                "feature_type": "binary",
                "signal_direction": "True = highest priority for retention",
                "business_meaning": "Best customers at risk - white-glove service required",
                "use_cases": ["VIP customer retention", "Collections strategy"]
            },
            "interaction_futile_payment_effort": {
                "tier": 1,
                "definition": "Payments increasing but delinquency worsening",
                "feature_type": "binary",
                "signal_direction": "True = structural problem (needs restructuring)",
                "business_meaning": "Customer trying to pay but debt growing faster",
                "use_cases": ["Restructuring candidates", "Forbearance decisions"]
            },
            "interaction_early_delinquency": {
                "tier": 1,
                "definition": "Delinquent in first 6 months (underwriting failure)",
                "feature_type": "binary",
                "signal_direction": "True = very high risk",
                "business_meaning": "Bad from start - likely underwriting miss",
                "use_cases": ["Underwriting quality", "Early warning systems"]
            },
            "interaction_seasoned_first_delinquency": {
                "tier": 1,
                "definition": "First-time delinquent after 24+ months",
                "feature_type": "binary",
                "signal_direction": "True = likely recoverable (life event)",
                "business_meaning": "Good customer hit unexpected issue",
                "use_cases": ["Recovery prediction", "Collections approach"]
            },
            "interaction_serial_curer": {
                "tier": 2,
                "definition": "Multiple cure episodes + erratic payment pattern",
                "feature_type": "binary",
                "signal_direction": "True = recoverable but high maintenance",
                "business_meaning": "Responds to pressure, will cure but needs contact",
                "use_cases": ["Collections intensity", "Contact strategy"]
            },
            "interaction_rfm_dpd_ratio": {
                "tier": 2,
                "definition": "RFM score divided by DPD severity (value/risk)",
                "feature_type": "continuous",
                "signal_direction": "higher = better risk-adjusted value",
                "business_meaning": "Customer value relative to current risk",
                "use_cases": ["Portfolio optimization", "Resource allocation"]
            }
        }


# Example usage and testing
if __name__ == "__main__":
    print("\n" + "="*80)
    print("FEATURE INTERACTIONS MODULE - EXAMPLE")
    print("="*80)
    
    # Create sample base features (from delinquency, payment, vintage modules)
    sample_base_features = {
        # Delinquency features
        "delinquency.dpd_current": 60,
        "delinquency.dpd_6M_slope": 8.5,
        "delinquency.delinquency_regime": "DETERIORATING",
        "delinquency.cure_flag_6M": True,
        "delinquency.cure_flag_12M": True,
        "delinquency.re_delinquency_flag_6M": False,
        
        # Payment features
        "payment.rfm_composite_score": 13,
        "payment.rfm_segment": "HIGH_VALUE",
        "payment.rfm_detailed_segment": "CHAMPIONS",
        "payment.payment_ratio_6M_mean": 0.92,
        "payment.payment_ratio_6M_slope": 0.05,
        "payment.payment_regime": "CONSISTENT_FULL_PAYER",
        
        # Vintage features
        "vintage.vintage_months_on_book": 28,
        "vintage.vintage_lifecycle_stage": "SEASONED",
        "vintage.vintage_months_since_first_delinquency": 2,
        "vintage.vintage_early_delinquency_flag": False
    }
    
    # Initialize engine
    engine = InteractionFeatureEngine()
    
    # Compute interaction features
    interaction_features = engine.compute_features(sample_base_features)
    
    print("\nInteraction Features Computed:")
    print("-" * 80)
    for key, value in sorted(interaction_features.items()):
        print(f"  {key:50s} : {value}")
    
    print(f"\nTotal interaction features: {len(interaction_features)}")
    
    print("\n" + "="*80)
    print("INTERPRETATION:")
    print("="*80)
    print("\nThis customer profile shows:")
    print("  - CHAMPIONS segment (RFM=13/15) but now 60 DPD")
    print("  - First-time delinquent after 28 months (seasoned account)")
    print("  - Payments increasing (+0.05) but DPD worsening (+8.5)")
    print("\n✓ interaction_champions_at_risk = True (URGENT)")
    print("✓ interaction_seasoned_first_delinquency = True (RECOVERABLE)")
    print("✓ interaction_futile_payment_effort = True (NEEDS RESTRUCTURING)")
    print("\n➜ Recommendation: White-glove outreach + payment plan offer")
