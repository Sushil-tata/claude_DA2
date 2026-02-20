"""
Recovery Agent Practical - Main Pipeline
=========================================

End-to-end pipeline orchestrator:
1. Load features (from existing feature pipeline)
2. Assign personas (PersonaBuilder)
3. Score recovery potential (RecoveryScorecard6M)
4. Route actions (ActionOverlayRouter)
5. Write daily scoring table + audit log

NO NPV optimization - pure rule-based approach.
"""

import logging
import uuid
from datetime import datetime
from typing import Optional, Literal

import pandas as pd

from recovery_agent_practical.segmentation.persona_builder import PersonaBuilder
from recovery_agent_practical.scoring.recovery_scorecard_6m import RecoveryScorecard6M
from recovery_agent_practical.routing.action_overlay import ActionOverlayRouter
from recovery_agent_practical.outputs.schemas import create_daily_scoring_row, create_audit_event

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

class RecoveryAgentPipeline:
    """
    Main orchestrator for recovery agent practical system.

    Workflow:
    --------
    1. Input: Feature DataFrame (from collections_feature_pipeline)
    2. Persona assignment (rule-based)
    3. Recovery scoring (6M scorecard)
    4. Action routing (rule-based overlay)
    5. Output: Daily scoring table + audit log
    """

    def __init__(
        self,
        scorecard_model_type: Literal["TWO_PART", "TWEEDIE"] = "TWO_PART",
        payment_threshold: float = 500.0,
        small_balance_threshold: float = 50_000,
        medium_balance_threshold: float = 200_000,
    ):
        """
        Args:
            scorecard_model_type: TWO_PART (default) or TWEEDIE
            payment_threshold: Minimum THB to count as recovery
            small_balance_threshold: SMALL balance threshold
            medium_balance_threshold: MEDIUM/LARGE threshold
        """
        self.scorecard_model_type = scorecard_model_type
        self.payment_threshold = payment_threshold

        # Initialize components
        self.persona_builder = PersonaBuilder(payment_threshold=payment_threshold)

        self.scorecard = RecoveryScorecard6M(
            model_type=scorecard_model_type,
            payment_threshold=payment_threshold,
        )

        self.action_router = ActionOverlayRouter(
            small_balance_threshold=small_balance_threshold,
            medium_balance_threshold=medium_balance_threshold,
        )

        self.pipeline_run_id = str(uuid.uuid4())
        self.is_scorecard_trained = False

    # ── TRAINING ──────────────────────────────────────────────────────────────

    def train_scorecard(
        self,
        features_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        val_split: float = 0.2,
    ) -> dict:
        """
        Train recovery scorecard.

        Args:
            features_df: Feature matrix with account_id
            labels_df: Labels with columns [account_id, recovery_amount_180d]
            val_split: Validation split ratio

        Returns:
            Training metrics dict
        """
        logger.info(f"Training {self.scorecard_model_type} scorecard on {len(features_df)} accounts")

        # Merge features with labels
        train_data = features_df.merge(labels_df, on="account_id", how="inner")

        if len(train_data) == 0:
            raise ValueError("No matching accounts between features and labels")

        # Extract features and labels
        X = train_data.drop(columns=["account_id", "recovery_amount_180d"], errors="ignore")
        y = train_data["recovery_amount_180d"]

        # Train/val split
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_split, random_state=42
        )

        # Train
        metrics = self.scorecard.train(X_train, y_train, X_val, y_val)

        self.is_scorecard_trained = True
        logger.info(f"Scorecard training complete. Val AUC: {metrics.get('val', {}).auc_roc if 'val' in metrics else 'N/A'}")

        return metrics

    # ── SCORING (INFERENCE) ──────────────────────────────────────────────────

    def score_batch(
        self,
        features_df: pd.DataFrame,
        score_date: str,
        write_outputs: bool = True,
        output_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Score a batch of accounts and optionally write outputs.

        Args:
            features_df: Feature DataFrame with account_id + all required features
            score_date: Score date (YYYY-MM-DD)
            write_outputs: Whether to write daily scoring table
            output_path: Path to write outputs (CSV or Parquet)

        Returns:
            Daily scoring DataFrame
        """
        if not self.is_scorecard_trained:
            raise ValueError("Scorecard not trained. Call train_scorecard() first.")

        logger.info(f"Scoring {len(features_df)} accounts for {score_date}")

        # Generate new run ID for this scoring batch
        self.pipeline_run_id = str(uuid.uuid4())

        # ──────────────────────────────────────────────────────────────────────
        # STEP 1: Persona Assignment
        # ──────────────────────────────────────────────────────────────────────
        logger.info("Step 1/3: Assigning personas...")
        persona_df = self.persona_builder.assign_batch(features_df)

        # Merge personas back to features
        enriched_df = features_df.merge(persona_df, on="account_id", how="left")

        # ──────────────────────────────────────────────────────────────────────
        # STEP 2: Recovery Scoring
        # ──────────────────────────────────────────────────────────────────────
        logger.info("Step 2/3: Scoring recovery potential...")
        recovery_scores_df = self.scorecard.score_batch(enriched_df, score_date)

        # Merge recovery scores
        enriched_df = enriched_df.merge(recovery_scores_df, on="account_id", how="left")

        # ──────────────────────────────────────────────────────────────────────
        # STEP 3: Action Routing
        # ──────────────────────────────────────────────────────────────────────
        logger.info("Step 3/3: Routing actions...")

        # Prepare routing input
        routing_input = enriched_df[[
            "account_id", "persona", "stage", "balance",
            "months_since_chargeoff", "score_band",
        ]].copy()

        # Add optional fields if available
        if "has_secured_assets" in enriched_df.columns:
            routing_input["has_secured_assets"] = enriched_df["has_secured_assets"]
        if "bureau_delinquent_other" in enriched_df.columns:
            routing_input["bureau_delinquent_other"] = enriched_df["bureau_delinquent_other"]

        actions_df = self.action_router.route_batch(routing_input)

        # Merge actions
        enriched_df = enriched_df.merge(
            actions_df[["account_id", "recommended_action", "priority_tier",
                       "contact_channel", "offer_type", "routing_reasoning", "balance_band"]],
            on="account_id",
            how="left"
        )

        # ──────────────────────────────────────────────────────────────────────
        # STEP 4: Create Daily Scoring Output
        # ──────────────────────────────────────────────────────────────────────
        logger.info("Creating daily scoring output...")

        daily_scoring_rows = []
        audit_events = []

        for _, row in enriched_df.iterrows():
            # Daily scoring row
            scoring_row = {
                "account_id": row["account_id"],
                "score_date": score_date,

                # Persona
                "persona": row["persona"],
                "payment_behavior_score": row["payment_behavior_score"],
                "engagement_score": row["engagement_score"],
                "capacity_score": row["capacity_score"],
                "avoidance_score": row["avoidance_score"],
                "persona_confidence": row["confidence_level"],
                "persona_flags": row["flags"],

                # Recovery score
                "score_band": row["score_band"],
                "p_recovery_6m": row["p_recovery"],
                "expected_recovery_amount": row["expected_recovery"],
                "model_version": row["model_version"],
                "model_type": row["model_type"],

                # Action
                "recommended_action": row["recommended_action"],
                "priority_tier": row["priority_tier"],
                "contact_channel": row["contact_channel"],
                "offer_type": row["offer_type"],
                "routing_reasoning": row["routing_reasoning"],

                # Context
                "stage": row["stage"],
                "balance": row["balance"],
                "balance_band": row["balance_band"],
                "days_past_due": row.get("days_past_due", 0),
                "months_since_chargeoff": row.get("months_since_chargeoff", 0),
                "has_secured_assets": row.get("has_secured_assets", False),
                "bureau_delinquent_other": row.get("bureau_delinquent_other", False),

                # Metadata
                "pipeline_run_id": self.pipeline_run_id,
                "created_at": datetime.now(),
            }

            daily_scoring_rows.append(scoring_row)

            # Audit events
            audit_events.append(create_audit_event(
                account_id=row["account_id"],
                event_type="PERSONA_ASSIGNED",
                event_detail=f"Persona: {row['persona']}, Confidence: {row['confidence_level']}",
                pipeline_run_id=self.pipeline_run_id,
            ))

            audit_events.append(create_audit_event(
                account_id=row["account_id"],
                event_type="SCORED",
                event_detail=f"Score band: {row['score_band']}, P(recovery): {row['p_recovery']:.3f}",
                pipeline_run_id=self.pipeline_run_id,
                model_version=row["model_version"],
            ))

            audit_events.append(create_audit_event(
                account_id=row["account_id"],
                event_type="ACTION_ROUTED",
                event_detail=f"Action: {row['recommended_action']}, Tier: {row['priority_tier']}",
                pipeline_run_id=self.pipeline_run_id,
            ))

        daily_scoring_df = pd.DataFrame(daily_scoring_rows)
        audit_log_df = pd.DataFrame(audit_events)

        # ──────────────────────────────────────────────────────────────────────
        # STEP 5: Write Outputs
        # ──────────────────────────────────────────────────────────────────────
        if write_outputs and output_path:
            logger.info(f"Writing outputs to {output_path}")

            # Daily scoring table
            scoring_file = f"{output_path}/daily_scoring_{score_date}.parquet"
            daily_scoring_df.to_parquet(scoring_file, index=False)
            logger.info(f"  ✓ Daily scoring: {scoring_file}")

            # Audit log
            audit_file = f"{output_path}/audit_log_{score_date}.parquet"
            audit_log_df.to_parquet(audit_file, index=False)
            logger.info(f"  ✓ Audit log: {audit_file}")

            # Summary stats
            summary = self._generate_summary(daily_scoring_df)
            summary_file = f"{output_path}/summary_{score_date}.txt"
            with open(summary_file, "w") as f:
                f.write(summary)
            logger.info(f"  ✓ Summary: {summary_file}")

        logger.info(f"✓ Scoring complete. {len(daily_scoring_df)} accounts scored.")
        return daily_scoring_df

    # ── UTILITIES ─────────────────────────────────────────────────────────────

    def _generate_summary(self, daily_scoring_df: pd.DataFrame) -> str:
        """Generate summary statistics"""
        summary = []
        summary.append("=" * 60)
        summary.append("RECOVERY AGENT PRACTICAL - SCORING SUMMARY")
        summary.append("=" * 60)
        summary.append(f"Total accounts scored: {len(daily_scoring_df)}")
        summary.append(f"Pipeline run ID: {self.pipeline_run_id}")
        summary.append("")

        # Persona distribution
        summary.append("PERSONA DISTRIBUTION:")
        persona_counts = daily_scoring_df["persona"].value_counts()
        for persona, count in persona_counts.items():
            pct = count / len(daily_scoring_df) * 100
            summary.append(f"  {persona:25s}: {count:5d} ({pct:5.1f}%)")
        summary.append("")

        # Score band distribution
        summary.append("SCORE BAND DISTRIBUTION:")
        band_counts = daily_scoring_df["score_band"].value_counts()
        for band in ["HOT", "WARM", "COLD", "FROZEN"]:
            count = band_counts.get(band, 0)
            pct = count / len(daily_scoring_df) * 100
            summary.append(f"  {band:10s}: {count:5d} ({pct:5.1f}%)")
        summary.append("")

        # Action distribution
        summary.append("ACTION DISTRIBUTION:")
        action_counts = daily_scoring_df["recommended_action"].value_counts()
        for action, count in action_counts.items():
            pct = count / len(daily_scoring_df) * 100
            summary.append(f"  {action:20s}: {count:5d} ({pct:5.1f}%)")
        summary.append("")

        # Priority distribution
        summary.append("PRIORITY DISTRIBUTION:")
        priority_counts = daily_scoring_df["priority_tier"].value_counts()
        for tier in ["TIER_1", "TIER_2", "TIER_3"]:
            count = priority_counts.get(tier, 0)
            pct = count / len(daily_scoring_df) * 100
            summary.append(f"  {tier:10s}: {count:5d} ({pct:5.1f}%)")
        summary.append("")

        # Expected recovery
        total_expected = daily_scoring_df["expected_recovery_amount"].sum()
        mean_expected = daily_scoring_df["expected_recovery_amount"].mean()
        summary.append(f"Total expected recovery: {total_expected:,.0f} THB")
        summary.append(f"Mean expected recovery: {mean_expected:,.0f} THB")
        summary.append("")

        summary.append("=" * 60)

        return "\n".join(summary)
