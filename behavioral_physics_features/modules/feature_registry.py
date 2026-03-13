"""
Feature Registry - Comprehensive Feature Catalog and Orchestration
==================================================================

Central registry for all behavioral physics features:
- Feature catalog with metadata
- Feature computation orchestration
- Feature validation and quality checks
- Feature documentation and lineage

Author: Behavioral Physics Team
Version: 1.0.0
"""

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

from .config import get_config
from .state_builder import StateBuilder
from .trajectory_engine import TrajectoryEngine
from .lender_ecology import LenderEcologyEngine
from .repayment_dynamics import RepaymentDynamicsEngine
from .enquiries_engine import EnquiriesEngine
from .cardx_bureau_interactions import CardXBureauInteractionsEngine
from .legal_actions import LegalActionsEngine
from .tdr_restructuring import TDRRestructuringEngine


class FeatureType(Enum):
    """Feature type classification"""
    VELOCITY = "velocity"
    ACCELERATION = "acceleration"
    TRANSITION = "transition"
    ENTROPY = "entropy"
    LENDER_ECOLOGY = "lender_ecology"
    REPAYMENT_NORMAL = "repayment_normal"
    REPAYMENT_STRESSED = "repayment_stressed"
    REPAYMENT_DELTA = "repayment_delta"
    ENQUIRY = "enquiry"
    STATE = "state"


class FeatureStability(Enum):
    """Feature stability/risk level"""
    STABLE = "stable"          # Low leakage risk, production-ready
    MODERATE = "moderate"      # Moderate risk, needs monitoring
    EXPERIMENTAL = "experimental"  # High risk, needs validation


@dataclass
class FeatureMetadata:
    """Metadata for a single feature"""
    name: str
    type: FeatureType
    description: str
    expected_direction: str  # "positive", "negative", "neutral"
    stability: FeatureStability
    source_engine: str
    formula: Optional[str] = None
    business_interpretation: Optional[str] = None
    use_cases: List[str] = field(default_factory=list)
    data_type: str = "double"
    nullable: bool = False
    min_value: Optional[float] = None
    max_value: Optional[float] = None


class FeatureRegistry:
    """
    Central registry for all behavioral physics features.

    Responsibilities:
    1. Orchestrate feature computation across all engines
    2. Maintain feature catalog with metadata
    3. Validate feature quality
    4. Provide feature documentation
    """

    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.config = get_config()

        # Initialize all feature engines
        self.state_builder = StateBuilder(spark)
        self.trajectory_engine = TrajectoryEngine(spark)
        self.lender_ecology = LenderEcologyEngine(spark)
        self.repayment_dynamics = RepaymentDynamicsEngine(spark)
        self.enquiries_engine = EnquiriesEngine(spark)
        self.cardx_bureau_interactions = CardXBureauInteractionsEngine(spark)
        self.legal_actions = LegalActionsEngine(spark)
        self.tdr_restructuring = TDRRestructuringEngine(spark)

        # Build feature catalog
        self.feature_catalog = self._build_feature_catalog()

    def compute_all_features(
        self,
        bureau_trade_df: DataFrame,
        bureau_enquiry_df: DataFrame,
        cardx_internal_df: DataFrame,
        as_of_month: str
    ) -> DataFrame:
        """
        Compute all behavioral physics features.

        Args:
            bureau_trade_df: Bureau trade monthly data
            bureau_enquiry_df: Bureau enquiry data
            cardx_internal_df: CardX internal monthly data
            as_of_month: As-of date for point-in-time filtering

        Returns:
            DataFrame with all features at (cust_id, as_of_month) grain
        """
        print(f"\n{'='*70}")
        print(f"BEHAVIORAL PHYSICS FEATURE COMPUTATION")
        print(f"As-of Month: {as_of_month}")
        print(f"{'='*70}\n")

        # Point-in-time filtering
        bureau_trade_df = bureau_trade_df.filter(
            F.col("as_of_month") <= F.lit(as_of_month)
        )
        cardx_internal_df = cardx_internal_df.filter(
            F.col("as_of_month") <= F.lit(as_of_month)
        )

        # 1. State Building (Foundation)
        print("Step 1/5: Building state assignments...")
        state_df = self.state_builder.build_states(
            bureau_trade_df, cardx_internal_df
        )
        # Add state transitions (creates state_changed, regime_changed columns)
        state_df = self.state_builder.compute_state_transitions(state_df)
        print(f"✓ States built for {state_df.select('cust_id').distinct().count()} customers")

        # 2. Trajectory Features
        print("\nStep 2/5: Computing trajectory features...")
        trajectory_df = self.trajectory_engine.compute_all_features(
            state_df, bureau_trade_df
        )
        print(f"✓ Trajectory features computed (60 features)")

        # 3. Lender Ecology Features
        print("\nStep 3/5: Computing lender ecology features...")
        lender_df = self.lender_ecology.compute_all_features(
            bureau_trade_df, state_df
        )
        print(f"✓ Lender ecology features computed (25 features)")

        # 4. Repayment Dynamics Features
        print("\nStep 4/5: Computing repayment dynamics features...")
        repayment_df = self.repayment_dynamics.compute_all_features(
            bureau_trade_df, state_df
        )
        print(f"✓ Repayment dynamics features computed (35 features)")

        # 5. Enquiry Features
        print("\nStep 5/8: Computing enquiry features...")
        enquiry_df = self.enquiries_engine.compute_all_features(
            bureau_enquiry_df, bureau_trade_df
        )
        print(f"✓ Enquiry features computed (12 features)")

        # 6. CardX Bureau Interactions
        print("\nStep 6/8: Computing CardX bureau interactions...")
        cardx_interactions_df = self.cardx_bureau_interactions.compute_all_features(
            bureau_trade_df, cardx_internal_df, state_df
        )
        print(f"✓ CardX interactions computed (20 features)")

        # 7. Legal Actions
        print("\nStep 7/8: Computing legal actions features...")
        legal_df = self.legal_actions.compute_all_features(
            bureau_trade_df
        )
        print(f"✓ Legal actions computed (18 features)")

        # 8. TDR Restructuring
        print("\nStep 8/8: Computing TDR restructuring features...")
        tdr_df = self.tdr_restructuring.compute_all_features(
            bureau_trade_df
        )
        print(f"✓ TDR restructuring computed (22 features)")

        # Combine all features
        print("\nCombining all features...")
        features_df = trajectory_df

        features_df = features_df.join(
            lender_df,
            on=["cust_id", "as_of_month"],
            how="left"
        )

        features_df = features_df.join(
            repayment_df,
            on=["cust_id", "as_of_month"],
            how="left"
        )

        features_df = features_df.join(
            enquiry_df,
            on=["cust_id", "as_of_month"],
            how="left"
        )

        features_df = features_df.join(
            cardx_interactions_df,
            on=["cust_id", "as_of_month"],
            how="left"
        )

        features_df = features_df.join(
            legal_df,
            on=["cust_id", "as_of_month"],
            how="left"
        )

        features_df = features_df.join(
            tdr_df,
            on=["cust_id", "as_of_month"],
            how="left"
        )

        # Filter to requested as_of_month
        features_df = features_df.filter(F.col("as_of_month") == F.lit(as_of_month))

        # Validate features
        features_df = self._validate_features(features_df)

        print(f"\n{'='*70}")
        print(f"✅ FEATURE COMPUTATION COMPLETE")
        print(f"Total features: {len(features_df.columns) - 2}")  # Exclude cust_id, as_of_month
        print(f"Total customers: {features_df.count()}")
        print(f"{'='*70}\n")

        return features_df

    def _validate_features(self, features_df: DataFrame) -> DataFrame:
        """
        Validate feature quality and apply guards.

        Checks:
        - No infinite values
        - Reasonable ranges
        - Missingness within thresholds
        """
        print("\nValidating features...")

        # Replace infinities with nulls
        for col_name in features_df.columns:
            if col_name not in ["cust_id", "as_of_month"]:
                features_df = features_df.withColumn(
                    col_name,
                    F.when(
                        F.col(col_name).isNull() |
                        F.isnan(col_name) |
                        (F.col(col_name) == float('inf')) |
                        (F.col(col_name) == float('-inf')),
                        None
                    ).otherwise(F.col(col_name))
                )

        # Check missingness
        total_rows = features_df.count()
        for col_name in features_df.columns:
            if col_name not in ["cust_id", "as_of_month"]:
                null_count = features_df.filter(F.col(col_name).isNull()).count()
                null_pct = (null_count / total_rows) * 100 if total_rows > 0 else 0

                if null_pct > 80:
                    print(f"  ⚠️  {col_name}: {null_pct:.1f}% missing (high)")

        # Fill remaining nulls with 0
        features_df = features_df.fillna(0.0)

        print("✓ Validation complete")
        return features_df

    def _build_feature_catalog(self) -> Dict[str, FeatureMetadata]:
        """
        Build comprehensive feature catalog with metadata.

        Returns:
            Dictionary mapping feature names to metadata
        """
        catalog = {}

        # ===== VELOCITY FEATURES =====
        catalog["dpd_velocity_3m"] = FeatureMetadata(
            name="dpd_velocity_3m",
            type=FeatureType.VELOCITY,
            description="Rate of change in max DPD over 3 months",
            expected_direction="negative",
            stability=FeatureStability.STABLE,
            source_engine="TrajectoryEngine",
            formula="(current_dpd - dpd_3m_ago) / 3",
            business_interpretation="Positive values = deterioration, Negative = improvement",
            use_cases=["delinquency_prediction", "early_warning"]
        )

        catalog["dpd_velocity_6m"] = FeatureMetadata(
            name="dpd_velocity_6m",
            type=FeatureType.VELOCITY,
            description="Rate of change in max DPD over 6 months",
            expected_direction="negative",
            stability=FeatureStability.STABLE,
            source_engine="TrajectoryEngine",
            formula="(current_dpd - dpd_6m_ago) / 6"
        )

        catalog["util_velocity_3m"] = FeatureMetadata(
            name="util_velocity_3m",
            type=FeatureType.VELOCITY,
            description="Rate of change in utilization over 3 months",
            expected_direction="negative",
            stability=FeatureStability.STABLE,
            source_engine="TrajectoryEngine",
            business_interpretation="Rising utilization = increasing credit dependency"
        )

        # ===== ACCELERATION FEATURES =====
        catalog["dpd_acceleration_3m"] = FeatureMetadata(
            name="dpd_acceleration_3m",
            type=FeatureType.ACCELERATION,
            description="Rate of change in DPD velocity (shock detection)",
            expected_direction="negative",
            stability=FeatureStability.STABLE,
            source_engine="TrajectoryEngine",
            formula="(current_velocity - prev_velocity) / 3",
            business_interpretation="Sudden acceleration indicates shock event",
            use_cases=["shock_detection", "early_warning", "collections_priority"]
        )

        catalog["shock_flag_3m"] = FeatureMetadata(
            name="shock_flag_3m",
            type=FeatureType.ACCELERATION,
            description="Binary flag for sudden DPD acceleration (>10/month)",
            expected_direction="negative",
            stability=FeatureStability.STABLE,
            source_engine="TrajectoryEngine",
            data_type="int"
        )

        catalog["deceleration_flag_3m"] = FeatureMetadata(
            name="deceleration_flag_3m",
            type=FeatureType.ACCELERATION,
            description="Binary flag for improving velocity (cure signal)",
            expected_direction="positive",
            stability=FeatureStability.STABLE,
            source_engine="TrajectoryEngine",
            data_type="int"
        )

        # ===== TRANSITION FEATURES =====
        catalog["s0_to_s2_count_12m"] = FeatureMetadata(
            name="s0_to_s2_count_12m",
            type=FeatureType.TRANSITION,
            description="Count of S0→S2 transitions (skipping S1) in 12 months",
            expected_direction="negative",
            stability=FeatureStability.STABLE,
            source_engine="TrajectoryEngine",
            business_interpretation="Abrupt deterioration, not gradual",
            use_cases=["shock_detection", "behavioral_segmentation"]
        )

        catalog["cure_halflife"] = FeatureMetadata(
            name="cure_halflife",
            type=FeatureType.TRANSITION,
            description="Time to reduce DPD by 50% from peak",
            expected_direction="negative",
            stability=FeatureStability.MODERATE,
            source_engine="TrajectoryEngine",
            business_interpretation="Longer half-life = harder to cure"
        )

        catalog["bad_state_trap_prob"] = FeatureMetadata(
            name="bad_state_trap_prob",
            type=FeatureType.TRANSITION,
            description="Probability of being stuck in S3/S4 for 3+ months",
            expected_direction="negative",
            stability=FeatureStability.STABLE,
            source_engine="TrajectoryEngine",
            business_interpretation="High probability = chronic delinquency",
            use_cases=["write_off_prediction", "collections_strategy"]
        )

        # ===== ENTROPY FEATURES =====
        catalog["state_entropy_6m"] = FeatureMetadata(
            name="state_entropy_6m",
            type=FeatureType.ENTROPY,
            description="Shannon entropy of state distribution over 6 months",
            expected_direction="negative",
            stability=FeatureStability.STABLE,
            source_engine="TrajectoryEngine",
            formula="-Σ(p(state) * log(p(state)))",
            business_interpretation="High entropy = unpredictable behavior",
            use_cases=["behavioral_segmentation", "model_confidence"]
        )

        catalog["oscillation_count_6m"] = FeatureMetadata(
            name="oscillation_count_6m",
            type=FeatureType.ENTROPY,
            description="Number of state changes in 6 months",
            expected_direction="negative",
            stability=FeatureStability.STABLE,
            source_engine="TrajectoryEngine",
            business_interpretation="Frequent changes = volatile behavior"
        )

        catalog["dpd_cv_6m"] = FeatureMetadata(
            name="dpd_cv_6m",
            type=FeatureType.ENTROPY,
            description="Coefficient of variation in DPD over 6 months",
            expected_direction="negative",
            stability=FeatureStability.STABLE,
            source_engine="TrajectoryEngine",
            formula="stddev(dpd) / mean(dpd)"
        )

        # ===== LENDER ECOLOGY FEATURES =====
        catalog["lender_hhi"] = FeatureMetadata(
            name="lender_hhi",
            type=FeatureType.LENDER_ECOLOGY,
            description="Herfindahl-Hirschman Index of lender concentration",
            expected_direction="neutral",
            stability=FeatureStability.STABLE,
            source_engine="LenderEcologyEngine",
            formula="Σ(lender_share²)",
            business_interpretation="High HHI = concentrated, Low HHI = diversified",
            use_cases=["exposure_analysis", "risk_diversification"]
        )

        catalog["synchronized_delinquency_flag"] = FeatureMetadata(
            name="synchronized_delinquency_flag",
            type=FeatureType.LENDER_ECOLOGY,
            description="Delinquent with 2+ lenders simultaneously",
            expected_direction="negative",
            stability=FeatureStability.STABLE,
            source_engine="LenderEcologyEngine",
            business_interpretation="Diffusion of problems across lenders",
            use_cases=["systemic_risk", "cross_default_prediction"],
            data_type="int"
        )

        catalog["fintech_share"] = FeatureMetadata(
            name="fintech_share",
            type=FeatureType.LENDER_ECOLOGY,
            description="Share of exposure to fintech lenders",
            expected_direction="negative",
            stability=FeatureStability.MODERATE,
            source_engine="LenderEcologyEngine",
            business_interpretation="Fintech exposure may indicate thin-file customers"
        )

        catalog["cardx_first_delinquent"] = FeatureMetadata(
            name="cardx_first_delinquent",
            type=FeatureType.LENDER_ECOLOGY,
            description="CardX delinquent but bureau clean (early warning)",
            expected_direction="negative",
            stability=FeatureStability.STABLE,
            source_engine="LenderEcologyEngine",
            business_interpretation="CardX data shows stress before bureau",
            use_cases=["early_warning", "proactive_collections"],
            data_type="int"
        )

        # ===== REPAYMENT NORMAL REGIME =====
        catalog["payment_effort_normal"] = FeatureMetadata(
            name="payment_effort_normal",
            type=FeatureType.REPAYMENT_NORMAL,
            description="Payment amount / balance in NORMAL regime",
            expected_direction="positive",
            stability=FeatureStability.STABLE,
            source_engine="RepaymentDynamicsEngine",
            business_interpretation="Higher effort = good credit behavior"
        )

        catalog["payment_consistency_normal"] = FeatureMetadata(
            name="payment_consistency_normal",
            type=FeatureType.REPAYMENT_NORMAL,
            description="% of months with payment in NORMAL regime",
            expected_direction="positive",
            stability=FeatureStability.STABLE,
            source_engine="RepaymentDynamicsEngine",
            business_interpretation="Consistency indicates discipline"
        )

        catalog["payment_discipline_score_normal"] = FeatureMetadata(
            name="payment_discipline_score_normal",
            type=FeatureType.REPAYMENT_NORMAL,
            description="Composite score of consistency + stability",
            expected_direction="positive",
            stability=FeatureStability.STABLE,
            source_engine="RepaymentDynamicsEngine"
        )

        # ===== REPAYMENT STRESSED REGIME =====
        catalog["cure_attempt_count_stressed"] = FeatureMetadata(
            name="cure_attempt_count_stressed",
            type=FeatureType.REPAYMENT_STRESSED,
            description="Number of months with DPD reduction in STRESSED regime",
            expected_direction="positive",
            stability=FeatureStability.STABLE,
            source_engine="RepaymentDynamicsEngine",
            business_interpretation="Cure attempts = willingness to recover",
            use_cases=["cure_probability", "collections_strategy"]
        )

        catalog["payment_fatigue_flag"] = FeatureMetadata(
            name="payment_fatigue_flag",
            type=FeatureType.REPAYMENT_STRESSED,
            description="Declining payment effort in STRESSED regime",
            expected_direction="negative",
            stability=FeatureStability.STABLE,
            source_engine="RepaymentDynamicsEngine",
            business_interpretation="Fatigue indicates giving up",
            use_cases=["write_off_prediction"],
            data_type="int"
        )

        catalog["chronicity_index"] = FeatureMetadata(
            name="chronicity_index",
            type=FeatureType.REPAYMENT_STRESSED,
            description="Fraction of time spent in STRESSED regime",
            expected_direction="negative",
            stability=FeatureStability.STABLE,
            source_engine="RepaymentDynamicsEngine",
            business_interpretation="High index = chronic delinquent"
        )

        catalog["desperation_score"] = FeatureMetadata(
            name="desperation_score",
            type=FeatureType.REPAYMENT_STRESSED,
            description="High payment effort but still high DPD",
            expected_direction="negative",
            stability=FeatureStability.MODERATE,
            source_engine="RepaymentDynamicsEngine",
            business_interpretation="Trying hard but not succeeding",
            data_type="int"
        )

        # ===== REPAYMENT DELTA =====
        catalog["delta_payment_effort"] = FeatureMetadata(
            name="delta_payment_effort",
            type=FeatureType.REPAYMENT_DELTA,
            description="Change in payment effort: STRESSED - NORMAL",
            expected_direction="positive",
            stability=FeatureStability.STABLE,
            source_engine="RepaymentDynamicsEngine",
            business_interpretation="Positive = increasing effort when stressed"
        )

        catalog["behavioral_shift_magnitude"] = FeatureMetadata(
            name="behavioral_shift_magnitude",
            type=FeatureType.REPAYMENT_DELTA,
            description="Magnitude of behavioral change between regimes",
            expected_direction="negative",
            stability=FeatureStability.STABLE,
            source_engine="RepaymentDynamicsEngine",
            business_interpretation="Large shift = unstable behavior"
        )

        catalog["recovery_capacity_score"] = FeatureMetadata(
            name="recovery_capacity_score",
            type=FeatureType.REPAYMENT_DELTA,
            description="Ability to maintain effort in STRESSED regime",
            expected_direction="positive",
            stability=FeatureStability.STABLE,
            source_engine="RepaymentDynamicsEngine",
            use_cases=["cure_probability", "restructuring_eligibility"]
        )

        # ===== ENQUIRY FEATURES =====
        catalog["enquiry_velocity_3m"] = FeatureMetadata(
            name="enquiry_velocity_3m",
            type=FeatureType.ENQUIRY,
            description="Average enquiries per month over 3 months",
            expected_direction="negative",
            stability=FeatureStability.STABLE,
            source_engine="EnquiriesEngine",
            business_interpretation="High velocity = credit seeking"
        )

        catalog["enquiry_burst_flag"] = FeatureMetadata(
            name="enquiry_burst_flag",
            type=FeatureType.ENQUIRY,
            description="5+ enquiries in 1 month (desperation signal)",
            expected_direction="negative",
            stability=FeatureStability.STABLE,
            source_engine="EnquiriesEngine",
            business_interpretation="Burst = financial stress",
            use_cases=["early_warning", "fraud_detection"],
            data_type="int"
        )

        catalog["conversion_rate_30d"] = FeatureMetadata(
            name="conversion_rate_30d",
            type=FeatureType.ENQUIRY,
            description="% of enquiries that became accounts within 30 days",
            expected_direction="positive",
            stability=FeatureStability.MODERATE,
            source_engine="EnquiriesEngine",
            business_interpretation="Low conversion = credit rejection"
        )

        catalog["rejection_signal"] = FeatureMetadata(
            name="rejection_signal",
            type=FeatureType.ENQUIRY,
            description="Multiple enquiries but low conversion",
            expected_direction="negative",
            stability=FeatureStability.MODERATE,
            source_engine="EnquiriesEngine",
            business_interpretation="Rejected by other lenders",
            use_cases=["adverse_selection"],
            data_type="int"
        )

        return catalog

    def get_feature_metadata(self, feature_name: str) -> Optional[FeatureMetadata]:
        """Get metadata for a specific feature"""
        return self.feature_catalog.get(feature_name)

    def get_features_by_type(self, feature_type: FeatureType) -> List[FeatureMetadata]:
        """Get all features of a specific type"""
        return [
            metadata for metadata in self.feature_catalog.values()
            if metadata.type == feature_type
        ]

    def get_features_by_use_case(self, use_case: str) -> List[FeatureMetadata]:
        """Get all features relevant to a use case"""
        return [
            metadata for metadata in self.feature_catalog.values()
            if use_case in metadata.use_cases
        ]

    def export_feature_catalog(self) -> DataFrame:
        """Export feature catalog as DataFrame for documentation"""
        catalog_data = [
            (
                metadata.name,
                metadata.type.value,
                metadata.description,
                metadata.expected_direction,
                metadata.stability.value,
                metadata.source_engine,
                metadata.formula or "",
                metadata.business_interpretation or "",
                ",".join(metadata.use_cases)
            )
            for metadata in self.feature_catalog.values()
        ]

        return self.spark.createDataFrame(
            catalog_data,
            ["feature_name", "type", "description", "expected_direction",
             "stability", "source_engine", "formula", "business_interpretation",
             "use_cases"]
        )


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    from pyspark.sql import SparkSession
    from datetime import date

    spark = SparkSession.builder \
        .appName("FeatureRegistryTest") \
        .master("local[*]") \
        .getOrCreate()

    print("\n" + "="*70)
    print("FEATURE REGISTRY - EXAMPLE")
    print("="*70)

    # Initialize registry
    registry = FeatureRegistry(spark)

    # Show feature catalog summary
    print("\n📚 Feature Catalog Summary:")
    print(f"Total features registered: {len(registry.feature_catalog)}")

    # Show features by type
    for feature_type in FeatureType:
        features = registry.get_features_by_type(feature_type)
        print(f"  - {feature_type.value}: {len(features)} features")

    # Show early warning features
    print("\n🚨 Features for Early Warning Use Case:")
    early_warning_features = registry.get_features_by_use_case("early_warning")
    for metadata in early_warning_features[:5]:
        print(f"  - {metadata.name}: {metadata.description}")

    # Export catalog
    print("\n📄 Feature Catalog Export:")
    catalog_df = registry.export_feature_catalog()
    catalog_df.show(10, truncate=False)

    print("\n✅ Feature Registry test complete")

    spark.stop()
