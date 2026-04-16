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
import os
import uuid
from datetime import datetime
from typing import Optional, Literal

import numpy as np
import pandas as pd

from recovery_agent_practical.segmentation.persona_builder import (
    PersonaBuilder, SEGMENTATION_FEATURES, AXIS_SCORE_COLUMNS,
)
from recovery_agent_practical.scoring.recovery_scorecard_6m import RecoveryScorecard6M
from recovery_agent_practical.routing.action_overlay import ActionOverlayRouter
from recovery_agent_practical.outputs.schemas import create_daily_scoring_row, create_audit_event

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# TREATMENT LOGGER
# ─────────────────────────────────────────────────────────────────────────────

class TreatmentLogger:
    """
    Persists treatment decisions to a log for causal tracking and validation.

    Schema per row:
      account_id, score_date, persona, signal_segment,
      recommended_action, p_recovery_6m, score_band, pipeline_run_id, logged_at

    One Parquet file per score_date. Appends to existing file if present.
    """

    LOG_COLUMNS = [
        "account_id", "score_date", "persona", "signal_segment",
        "recommended_action", "p_recovery_6m", "score_band", "pipeline_run_id",
    ]

    def __init__(self, log_path: str):
        """
        Args:
            log_path: Directory where treatment log Parquet files are written.
        """
        self.log_path = log_path

    def log_batch(self, scoring_df: pd.DataFrame) -> str:
        """
        Write treatment decisions from a scored batch.

        Args:
            scoring_df: Daily scoring DataFrame from score_batch()

        Returns:
            Path to the written log file.
        """
        if scoring_df.empty:
            logger.warning("TreatmentLogger: empty scoring_df, nothing logged.")
            return ""

        available_cols = [c for c in self.LOG_COLUMNS if c in scoring_df.columns]
        log_df = scoring_df[available_cols].copy()
        log_df["logged_at"] = datetime.now().isoformat()

        score_date = str(scoring_df["score_date"].iloc[0]) if "score_date" in scoring_df.columns else "unknown"
        log_file = os.path.join(self.log_path, f"treatment_log_{score_date}.parquet")

        if os.path.exists(log_file):
            existing = pd.read_parquet(log_file)
            log_df = pd.concat([existing, log_df], ignore_index=True)

        log_df.to_parquet(log_file, index=False)
        logger.info(f"TreatmentLogger: wrote {len(log_df)} rows → {log_file}")
        return log_file


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
        treatment_log_path: Optional[str] = None,
    ):
        """
        Args:
            scorecard_model_type   : TWO_PART (default) or TWEEDIE
            payment_threshold      : Minimum THB to count as recovery
            small_balance_threshold: SMALL balance threshold
            medium_balance_threshold: MEDIUM/LARGE threshold
            treatment_log_path     : Directory for treatment log Parquet files.
                                     If None, treatment logging is skipped.
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

        self.treatment_logger = (
            TreatmentLogger(treatment_log_path) if treatment_log_path else None
        )

        self.pipeline_run_id = str(uuid.uuid4())
        self.is_scorecard_trained = False

    # ── TRAINING ──────────────────────────────────────────────────────────────

    def train_scorecard(
        self,
        features_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        observation_date: str,
        val_split: float = 0.2,
    ) -> dict:
        """
        Train recovery scorecard using propensity features only.

        Feature separation enforced here:
        - SEGMENTATION_FEATURES and AXIS_SCORE_COLUMNS are explicitly excluded
          from the training feature set to prevent circular signal leakage.
        - Persona (categorical) and signal_segment may be added as conditioning
          variables AFTER segmentation — they are computed fresh inside this method.

        Args:
            features_df      : Feature matrix with account_id. Must contain
                               both SEGMENTATION_FEATURES (for persona) and
                               PROPENSITY_FEATURES (for scorecard).
            labels_df        : Labels with columns [account_id, recovery_amount_180d]
            observation_date : Snapshot date (YYYY-MM-DD) — passed to persona assignment
            val_split        : Validation split ratio

        Returns:
            Training metrics dict
        """
        logger.info(f"Training {self.scorecard_model_type} scorecard on {len(features_df)} accounts")

        # Step 1: Assign personas on segmentation features — separate pass
        seg_cols = ["account_id"] + [c for c in SEGMENTATION_FEATURES if c in features_df.columns]
        persona_df = self.persona_builder.assign_batch(
            features_df[seg_cols],
            observation_date=observation_date,
        )
        # Merge ONLY persona + signal_segment as conditioning variables
        # Do NOT include axis scores (structural_payment_score etc.)
        persona_conditioning = persona_df[["account_id", "persona", "signal_segment"]].copy()

        # Step 2: Build scorecard training set — propensity features + persona conditioning
        # Exclude: SEGMENTATION_FEATURES, AXIS_SCORE_COLUMNS, non-numeric, label
        excluded = set(SEGMENTATION_FEATURES) | set(AXIS_SCORE_COLUMNS) | {"account_id", "recovery_amount_180d"}

        train_data = features_df.merge(labels_df, on="account_id", how="inner")
        train_data = train_data.merge(persona_conditioning, on="account_id", how="left")

        if len(train_data) == 0:
            raise ValueError("No matching accounts between features and labels")

        # One-hot encode persona and signal_segment (categorical conditioning)
        train_data = pd.get_dummies(train_data, columns=["persona", "signal_segment"], dummy_na=False)

        numeric_cols = train_data.select_dtypes(include=[np.number]).columns.tolist()
        scorecard_cols = [c for c in numeric_cols if c not in excluded]

        X = train_data[scorecard_cols]
        y = train_data["recovery_amount_180d"]

        logger.info(
            f"Scorecard training: {len(scorecard_cols)} features "
            f"({len(scorecard_cols)} propensity + persona dummies). "
            f"Segmentation features excluded: {len([c for c in SEGMENTATION_FEATURES if c in features_df.columns])}."
        )

        # Train/val split
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_split, random_state=42
        )

        # Train
        metrics = self.scorecard.train(X_train, y_train, X_val, y_val)

        self.is_scorecard_trained = True
        logger.info(f"Scorecard training complete.")

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

        Feature separation enforced:
        - PersonaBuilder receives segmentation features only
        - Scorecard receives propensity features + persona/signal_segment dummies
        - Axis scores (structural_payment_score etc.) are excluded from scorecard input
        - Axis scores appear in daily output for audit purposes only

        Args:
            features_df : Feature DataFrame with account_id + all features
            score_date  : Score date (YYYY-MM-DD) — also used as observation_date
            write_outputs: Whether to write daily scoring table
            output_path : Directory to write outputs (Parquet)

        Returns:
            Daily scoring DataFrame
        """
        if not self.is_scorecard_trained:
            raise ValueError("Scorecard not trained. Call train_scorecard() first.")

        logger.info(f"Scoring {len(features_df)} accounts for {score_date}")
        self.pipeline_run_id = str(uuid.uuid4())

        # ──────────────────────────────────────────────────────────────────────
        # STEP 1: Persona Assignment (segmentation features only)
        # ──────────────────────────────────────────────────────────────────────
        logger.info("Step 1/3: Assigning personas...")
        seg_cols = ["account_id"] + [c for c in SEGMENTATION_FEATURES if c in features_df.columns]
        persona_df = self.persona_builder.assign_batch(
            features_df[seg_cols],
            observation_date=score_date,
        )

        # Merge ONLY persona + signal_segment into enriched_df (not axis scores)
        # Axis scores are kept separately for output/audit
        _persona_for_scorecard = persona_df[["account_id", "persona", "signal_segment"]].copy()
        _persona_for_output    = persona_df.copy()  # full — includes axis scores for audit

        enriched_df = features_df.merge(_persona_for_scorecard, on="account_id", how="left")

        # Exclude segmentation features from scorecard input
        _exclude = set(SEGMENTATION_FEATURES) | set(AXIS_SCORE_COLUMNS) | {"account_id"}
        _scorecard_cols = [c for c in enriched_df.columns if c not in _exclude]

        # One-hot encode persona + signal_segment before passing to scorecard
        enriched_for_scoring = pd.get_dummies(
            enriched_df[_scorecard_cols + ["account_id"]],
            columns=["persona", "signal_segment"],
            dummy_na=False,
        )

        # ──────────────────────────────────────────────────────────────────────
        # STEP 2: Recovery Scoring (propensity features + persona dummies)
        # ──────────────────────────────────────────────────────────────────────
        logger.info("Step 2/3: Scoring recovery potential...")
        recovery_scores_df = self.scorecard.score_batch(enriched_for_scoring, score_date)

        # Merge axis scores back to enriched_df for output only (not used by model)
        enriched_df = enriched_df.merge(
            _persona_for_output[["account_id"] + AXIS_SCORE_COLUMNS +
                                ["confidence_level", "data_completeness_pct", "flags"]],
            on="account_id",
            how="left",
        )

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
                       "contact_channel", "offer_type", "reasoning", "balance_band"]],
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
            # Daily scoring row — axis scores appear here for audit, NOT model features
            scoring_row = {
                "account_id": row["account_id"],
                "score_date": score_date,

                # Persona (structural segmentation)
                "persona": row["persona"],
                "signal_segment": row.get("signal_segment", "STANDARD"),
                # Axis scores: audit/transparency only — never model input
                "structural_payment_score": row.get("structural_payment_score"),
                "trajectory_score": row.get("trajectory_score"),
                "capacity_score": row.get("capacity_score"),
                "avoidance_score": row.get("avoidance_score"),
                "persona_confidence": row.get("confidence_level"),
                "data_completeness_pct": row.get("data_completeness_pct"),
                "persona_flags": row.get("flags"),

                # Recovery score (propensity)
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
                "routing_reasoning": row["reasoning"],

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
                event_detail=(
                    f"Persona: {row['persona']}, "
                    f"Segment: {row.get('signal_segment','STANDARD')}, "
                    f"Confidence: {row.get('confidence_level','?')}"
                ),
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
        # STEP 5: Treatment Logging (persistent causal tracking)
        # ──────────────────────────────────────────────────────────────────────
        if self.treatment_logger is not None:
            self.treatment_logger.log_batch(daily_scoring_df)

        # ──────────────────────────────────────────────────────────────────────
        # STEP 6: Write Outputs
        # ──────────────────────────────────────────────────────────────────────
        if write_outputs and output_path:
            logger.info(f"Writing outputs to {output_path}")

            scoring_file = f"{output_path}/daily_scoring_{score_date}.parquet"
            daily_scoring_df.to_parquet(scoring_file, index=False)
            logger.info(f"  ✓ Daily scoring: {scoring_file}")

            audit_file = f"{output_path}/audit_log_{score_date}.parquet"
            audit_log_df.to_parquet(audit_file, index=False)
            logger.info(f"  ✓ Audit log: {audit_file}")

            summary = self._generate_summary(daily_scoring_df)
            summary_file = f"{output_path}/summary_{score_date}.txt"
            with open(summary_file, "w") as f:
                f.write(summary)
            logger.info(f"  ✓ Summary: {summary_file}")

        logger.info(f"Scoring complete. {len(daily_scoring_df)} accounts scored.")
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

        # Signal segment distribution
        if "signal_segment" in daily_scoring_df.columns:
            summary.append("SIGNAL SEGMENT DISTRIBUTION:")
            seg_counts = daily_scoring_df["signal_segment"].value_counts()
            for seg, count in seg_counts.items():
                pct = count / len(daily_scoring_df) * 100
                summary.append(f"  {seg:25s}: {count:5d} ({pct:5.1f}%)")
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
