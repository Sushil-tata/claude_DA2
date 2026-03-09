"""
State Builder - Monthly State Assignment (S0-S4)
=================================================

Assigns behavioral states to each customer-month based on DPD.
Foundation for trajectory and regime analysis.

Author: Behavioral Physics Team
Version: 1.0.0
"""

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from typing import Dict, List

from .config import get_config


class StateBuilder:
    """
    Assigns monthly states based on DPD levels (Thai simplified classification).

    State Framework (Thai regulatory categories):
    - CURRENT: 0-30 DPD (Current accounts)
    - SM: 31-90 DPD (Special Mention)
    - NPL: 91-180 DPD (Non-Performing Loan)
    - CHARGE_OFF: 181+ DPD (Charge-off / Loss)
    """

    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.config = get_config()

    def build_states(
        self,
        bureau_trade_monthly_df: DataFrame,
        cardx_internal_monthly_df: DataFrame = None
    ) -> DataFrame:
        """
        Build monthly state assignments for each customer.

        Args:
            bureau_trade_monthly_df: Bureau trade lines
                Required columns: cust_id, as_of_month, lender_name, dpd
            cardx_internal_monthly_df: CardX internal data (optional)
                Required columns: cust_id, as_of_month, cardx_dpd

        Returns:
            DataFrame with columns:
                - cust_id
                - as_of_month
                - bureau_max_dpd (max DPD across all bureau accounts)
                - bureau_state (S0-S4)
                - bureau_regime (NORMAL/STRESSED)
                - cardx_dpd (if available)
                - cardx_state (if available)
                - cardx_regime (if available)
                - consolidated_state (worst of bureau and CardX)
                - consolidated_regime
        """
        # 1. Bureau state assignment
        bureau_states = self._compute_bureau_states(bureau_trade_monthly_df)

        # 2. CardX state assignment (if available)
        if cardx_internal_monthly_df is not None:
            cardx_states = self._compute_cardx_states(cardx_internal_monthly_df)

            # 3. Consolidate bureau + CardX
            consolidated = self._consolidate_states(bureau_states, cardx_states)
        else:
            # No CardX data - use bureau only
            consolidated = bureau_states.withColumn(
                "consolidated_state", F.col("bureau_state")
            ).withColumn(
                "consolidated_regime", F.col("bureau_regime")
            )

        return consolidated

    def _compute_bureau_states(self, bureau_trade_df: DataFrame) -> DataFrame:
        """
        Compute bureau-level states (max DPD across all accounts).

        Logic:
        - For each (cust_id, as_of_month), find MAX(dpd) across all lenders
        - Map max DPD to state (S0-S4)
        - Derive regime (NORMAL/STRESSED)
        """
        # Aggregate to customer-month level (max DPD)
        bureau_agg = bureau_trade_df.groupBy("cust_id", "as_of_month").agg(
            F.max("dpd").alias("bureau_max_dpd"),
            F.count("lender_name").alias("bureau_account_count")
        )

        # Map DPD to state using UDF
        state_mapping_udf = F.udf(
            lambda dpd: self.config.states.get_state_from_dpd(dpd or 0)
        )

        bureau_agg = bureau_agg.withColumn(
            "bureau_state",
            state_mapping_udf(F.col("bureau_max_dpd"))
        )

        # Derive regime
        regime_udf = F.udf(
            lambda state: "NORMAL" if self.config.states.is_normal_regime(state) else "STRESSED"
        )

        bureau_agg = bureau_agg.withColumn(
            "bureau_regime",
            regime_udf(F.col("bureau_state"))
        )

        return bureau_agg

    def _compute_cardx_states(self, cardx_internal_df: DataFrame) -> DataFrame:
        """
        Compute CardX-level states.

        Input columns: cust_id, as_of_month, cardx_dpd
        """
        # Map DPD to state
        state_mapping_udf = F.udf(
            lambda dpd: self.config.states.get_state_from_dpd(dpd or 0)
        )

        cardx_states = cardx_internal_df.select(
            "cust_id",
            "as_of_month",
            "cardx_dpd",
            state_mapping_udf(F.col("cardx_dpd")).alias("cardx_state")
        )

        # Derive regime
        regime_udf = F.udf(
            lambda state: "NORMAL" if self.config.states.is_normal_regime(state) else "STRESSED"
        )

        cardx_states = cardx_states.withColumn(
            "cardx_regime",
            regime_udf(F.col("cardx_state"))
        )

        return cardx_states

    def _consolidate_states(
        self,
        bureau_states: DataFrame,
        cardx_states: DataFrame
    ) -> DataFrame:
        """
        Consolidate bureau and CardX states (worst state wins).

        Logic:
        - S4 > S3 > S2 > S1 > S0 (hierarchy)
        - consolidated_state = max(bureau_state, cardx_state)
        """
        # Join bureau and CardX states
        consolidated = bureau_states.join(
            cardx_states,
            on=["cust_id", "as_of_month"],
            how="left"
        )

        # State hierarchy (S4 worst, S0 best)
        state_rank_map = {
            "S4": 4, "S3": 3, "S2": 2, "S1": 1, "S0": 0, "UNKNOWN": -1
        }

        state_rank_udf = F.udf(lambda state: state_rank_map.get(state, -1))

        # Compute consolidated state (worst state)
        consolidated = consolidated.withColumn(
            "bureau_rank", state_rank_udf(F.col("bureau_state"))
        ).withColumn(
            "cardx_rank", state_rank_udf(F.coalesce(F.col("cardx_state"), F.lit("S0")))
        ).withColumn(
            "max_rank", F.greatest(F.col("bureau_rank"), F.col("cardx_rank"))
        )

        # Map rank back to state
        reverse_map = {v: k for k, v in state_rank_map.items()}
        reverse_udf = F.udf(lambda rank: reverse_map.get(rank, "UNKNOWN"))

        consolidated = consolidated.withColumn(
            "consolidated_state",
            reverse_udf(F.col("max_rank"))
        )

        # Derive consolidated regime
        regime_udf = F.udf(
            lambda state: "NORMAL" if self.config.states.is_normal_regime(state) else "STRESSED"
        )

        consolidated = consolidated.withColumn(
            "consolidated_regime",
            regime_udf(F.col("consolidated_state"))
        )

        # Drop temporary columns
        consolidated = consolidated.drop("bureau_rank", "cardx_rank", "max_rank")

        return consolidated

    def compute_state_transitions(self, state_df: DataFrame) -> DataFrame:
        """
        Compute state transitions (for trajectory analysis).

        Adds columns:
        - prev_month_state (lag 1)
        - prev_month_regime
        - state_changed (boolean)
        - regime_changed (boolean)
        - transition_type (e.g., "S0_to_S2", "NORMAL_to_STRESSED")
        """
        # Window partitioned by customer, ordered by month
        window_spec = Window.partitionBy("cust_id").orderBy("as_of_month")

        # Lag state and regime
        state_df = state_df.withColumn(
            "prev_month_state",
            F.lag("consolidated_state", 1).over(window_spec)
        ).withColumn(
            "prev_month_regime",
            F.lag("consolidated_regime", 1).over(window_spec)
        )

        # Detect changes
        state_df = state_df.withColumn(
            "state_changed",
            F.col("consolidated_state") != F.col("prev_month_state")
        ).withColumn(
            "regime_changed",
            F.col("consolidated_regime") != F.col("prev_month_regime")
        )

        # Transition type
        state_df = state_df.withColumn(
            "transition_type",
            F.when(
                F.col("state_changed"),
                F.concat(
                    F.col("prev_month_state"),
                    F.lit("_to_"),
                    F.col("consolidated_state")
                )
            ).otherwise(F.lit("NO_CHANGE"))
        )

        # Regime transition type
        state_df = state_df.withColumn(
            "regime_transition_type",
            F.when(
                F.col("regime_changed"),
                F.concat(
                    F.col("prev_month_regime"),
                    F.lit("_to_"),
                    F.col("consolidated_regime")
                )
            ).otherwise(F.lit("NO_CHANGE"))
        )

        return state_df

    def add_state_streak_features(self, state_df: DataFrame) -> DataFrame:
        """
        Add state streak features (consecutive months in same state).

        Features:
        - current_state_streak (consecutive months in current state)
        - max_s3_streak (longest S3 streak)
        - max_s4_streak (longest S4 streak)
        """
        # Window for streak calculation
        window_spec = Window.partitionBy("cust_id").orderBy("as_of_month")

        # Create state change indicator
        state_df = state_df.withColumn(
            "state_change_indicator",
            (F.col("consolidated_state") != F.lag("consolidated_state", 1).over(window_spec))
            .cast("int")
        )

        # Cumulative sum of state changes (creates streak groups)
        state_df = state_df.withColumn(
            "state_group",
            F.sum("state_change_indicator").over(
                window_spec.rowsBetween(Window.unboundedPreceding, Window.currentRow)
            )
        )

        # Count months per state group (current streak)
        window_streak = Window.partitionBy("cust_id", "state_group").orderBy("as_of_month")

        state_df = state_df.withColumn(
            "current_state_streak",
            F.row_number().over(window_streak)
        )

        return state_df


# ============================================================================
# Example Usage & Testing
# ============================================================================

if __name__ == "__main__":
    from pyspark.sql import SparkSession
    import pandas as pd
    from datetime import date

    spark = SparkSession.builder \
        .appName("StateBuilderTest") \
        .master("local[*]") \
        .getOrCreate()

    print("\n" + "="*70)
    print("STATE BUILDER - EXAMPLE")
    print("="*70)

    # Create sample bureau trade data
    bureau_data = [
        ("CUST001", date(2024, 1, 31), "LENDER_A", 0),
        ("CUST001", date(2024, 1, 31), "LENDER_B", 15),
        ("CUST001", date(2024, 2, 29), "LENDER_A", 30),
        ("CUST001", date(2024, 2, 29), "LENDER_B", 45),
        ("CUST001", date(2024, 3, 31), "LENDER_A", 60),
        ("CUST001", date(2024, 3, 31), "LENDER_B", 90),
        ("CUST002", date(2024, 1, 31), "LENDER_C", 0),
        ("CUST002", date(2024, 2, 29), "LENDER_C", 0),
    ]

    bureau_df = spark.createDataFrame(
        bureau_data,
        ["cust_id", "as_of_month", "lender_name", "dpd"]
    )

    # Create sample CardX data
    cardx_data = [
        ("CUST001", date(2024, 1, 31), 0),
        ("CUST001", date(2024, 2, 29), 30),
        ("CUST001", date(2024, 3, 31), 45),
        ("CUST002", date(2024, 1, 31), 15),
        ("CUST002", date(2024, 2, 29), 30),
    ]

    cardx_df = spark.createDataFrame(
        cardx_data,
        ["cust_id", "as_of_month", "cardx_dpd"]
    )

    # Build states
    builder = StateBuilder(spark)
    states_df = builder.build_states(bureau_df, cardx_df)

    # Add transitions
    states_df = builder.compute_state_transitions(states_df)

    # Show results
    print("\nState Assignments:")
    states_df.select(
        "cust_id", "as_of_month",
        "bureau_max_dpd", "bureau_state", "bureau_regime",
        "cardx_dpd", "cardx_state",
        "consolidated_state", "consolidated_regime"
    ).orderBy("cust_id", "as_of_month").show(truncate=False)

    print("\nState Transitions:")
    states_df.select(
        "cust_id", "as_of_month",
        "prev_month_state", "consolidated_state",
        "state_changed", "transition_type"
    ).orderBy("cust_id", "as_of_month").show(truncate=False)

    print("\n✅ State Builder test complete")

    spark.stop()
