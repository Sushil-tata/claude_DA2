"""
Gradual Rollout Manager

Manages safe, gradual rollout of Challenger model from 0% → 100%.

Rollout Strategy:
- Phase 1: Shadow mode (0% traffic, log predictions)
- Phase 2: Gradual ramp (10% → 25% → 50% → 75%)
- Phase 3: Full rollout (100%)
- Rollback: Revert to 0% if stability issues detected

Safety Checks:
- Correlation with Champion >= 0.85
- MAE increase < 10%
- No data quality issues
- Manual approval gates at 50% and 100%

State Management:
- Current rollout % stored in Delta table
- History tracked for auditing
- Rollback mechanism with notifications
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

from decision_agent.validation.stability_monitor import StabilityMonitor

logger = logging.getLogger(__name__)


class RolloutManager:
    """
    Manages gradual rollout of Challenger model to production.

    Responsibilities:
    - Track current rollout percentage
    - Validate stability before increasing rollout
    - Automatic or manual rollout progression
    - Rollback if issues detected
    """

    def __init__(
        self,
        spark: SparkSession,
        config: Dict[str, Any]
    ):
        """
        Initialize rollout manager.

        Args:
            spark: Spark session
            config: Rollout configuration:
                - state_table: Delta table storing rollout state
                - rollout_schedule: List of rollout percentages [10, 25, 50, 75, 100]
                - stability_threshold: Minimum correlation (default: 0.85)
                - mae_increase_threshold: Max MAE increase % (default: 0.10)
                - manual_approval_at: Rollout % requiring manual approval [50, 100]
                - rollback_conditions: Conditions triggering automatic rollback
        """
        self.spark = spark
        self.config = config
        self.state_table = config.get("state_table", "decision_agent.rollout_state")
        self.rollout_schedule = config.get("rollout_schedule", [10, 25, 50, 75, 100])
        self.stability_threshold = config.get("stability_threshold", 0.85)
        self.mae_increase_threshold = config.get("mae_increase_threshold", 0.10)
        self.manual_approval_at = config.get("manual_approval_at", [50, 100])

        # Initialize stability monitor
        self.stability_monitor = StabilityMonitor(config.get("stability_config", {}))

        # Ensure state table exists
        self._initialize_state_table()

    def get_current_rollout_percentage(self) -> int:
        """
        Get current rollout percentage.

        Returns:
            Current rollout percentage (0-100)
        """
        current_state = self.spark.table(self.state_table) \
            .orderBy(F.col("timestamp").desc()) \
            .limit(1) \
            .collect()

        if current_state:
            return current_state[0]["rollout_percentage"]
        else:
            # No state yet, default to 0% (Champion only)
            return 0

    def set_rollout_percentage(
        self,
        percentage: int,
        reason: str,
        approved_by: Optional[str] = None
    ) -> bool:
        """
        Set rollout percentage (manual override).

        Args:
            percentage: Target rollout percentage (0-100)
            reason: Reason for change
            approved_by: Who approved this change (for audit)

        Returns:
            True if successful
        """
        if not 0 <= percentage <= 100:
            raise ValueError(f"Rollout percentage must be 0-100, got {percentage}")

        logger.info(f"Setting rollout percentage to {percentage}%")
        logger.info(f"Reason: {reason}")
        if approved_by:
            logger.info(f"Approved by: {approved_by}")

        # Record state change
        new_state = self.spark.createDataFrame([{
            "rollout_percentage": percentage,
            "timestamp": datetime.now().isoformat(),
            "reason": reason,
            "approved_by": approved_by or "automatic",
            "status": "active"
        }])

        new_state.write \
            .format("delta") \
            .mode("append") \
            .saveAsTable(self.state_table)

        logger.info(f"✓ Rollout percentage set to {percentage}%")
        return True

    def progress_rollout(
        self,
        champion_predictions_df: DataFrame,
        challenger_predictions_df: DataFrame,
        force: bool = False
    ) -> Dict[str, Any]:
        """
        Progress rollout to next stage if stability checks pass.

        Args:
            champion_predictions_df: Recent Champion predictions
            challenger_predictions_df: Recent Challenger predictions
            force: Skip stability checks (USE WITH CAUTION)

        Returns:
            Results dict with new rollout %, stability metrics, decision
        """
        logger.info("=" * 80)
        logger.info("Rollout Progression Check")
        logger.info("=" * 80)

        current_pct = self.get_current_rollout_percentage()
        logger.info(f"Current rollout: {current_pct}%")

        # Determine next rollout percentage
        next_pct = self._get_next_rollout_percentage(current_pct)

        if next_pct is None:
            logger.info("Already at 100% rollout. No progression needed.")
            return {
                "current_rollout": current_pct,
                "next_rollout": current_pct,
                "action": "no_change",
                "reason": "already_at_100_percent"
            }

        logger.info(f"Next rollout stage: {next_pct}%")

        results = {
            "current_rollout": current_pct,
            "next_rollout": next_pct,
            "timestamp": datetime.now().isoformat()
        }

        # Check if manual approval required
        if next_pct in self.manual_approval_at and not force:
            logger.warning(f"Manual approval required for {next_pct}% rollout")
            results["action"] = "manual_approval_required"
            results["reason"] = f"rollout to {next_pct}% requires manual approval"
            return results

        # Stability checks (unless forced)
        if not force:
            logger.info("Running stability checks...")
            stability_passed, stability_results = self.stability_monitor.evaluate_stability(
                champion_predictions_df,
                model_a_col="champion_prediction",
                model_b_col="challenger_prediction"
            )

            results["stability_check"] = stability_results

            if not stability_passed:
                logger.error("Stability check FAILED. Not progressing rollout.")
                results["action"] = "blocked"
                results["reason"] = "stability_check_failed"
                return results

            logger.info("✓ Stability check PASSED")

        # Progress rollout
        logger.info(f"Progressing rollout: {current_pct}% → {next_pct}%")
        self.set_rollout_percentage(
            percentage=next_pct,
            reason=f"automatic_progression_from_{current_pct}_percent",
            approved_by="rollout_manager"
        )

        results["action"] = "progressed"
        results["reason"] = "stability_checks_passed"

        logger.info("=" * 80)
        logger.info(f"✓ Rollout progressed to {next_pct}%")
        logger.info("=" * 80)

        return results

    def rollback(
        self,
        reason: str,
        approved_by: Optional[str] = None
    ) -> bool:
        """
        Rollback to 0% (Champion only).

        Args:
            reason: Reason for rollback
            approved_by: Who approved rollback

        Returns:
            True if successful
        """
        logger.warning("=" * 80)
        logger.warning("ROLLBACK INITIATED")
        logger.warning(f"Reason: {reason}")
        logger.warning("=" * 80)

        current_pct = self.get_current_rollout_percentage()

        self.set_rollout_percentage(
            percentage=0,
            reason=f"rollback_from_{current_pct}_percent: {reason}",
            approved_by=approved_by or "automatic_rollback"
        )

        logger.warning(f"✓ Rolled back to 0% (Champion only)")

        # TODO: Send alert to Slack/PagerDuty
        self._send_rollback_alert(current_pct, reason)

        return True

    def check_rollback_conditions(
        self,
        champion_predictions_df: DataFrame,
        challenger_predictions_df: DataFrame
    ) -> tuple[bool, Optional[str]]:
        """
        Check if rollback conditions are met.

        Args:
            champion_predictions_df: Champion predictions
            challenger_predictions_df: Challenger predictions

        Returns:
            Tuple of (should_rollback, reason)
        """
        # Check stability
        stability_passed, stability_results = self.stability_monitor.evaluate_stability(
            champion_predictions_df,
            model_a_col="champion_prediction",
            model_b_col="challenger_prediction"
        )

        if not stability_passed:
            correlation = stability_results.get("correlation", 0)
            if correlation < self.stability_threshold:
                return True, f"correlation_too_low: {correlation:.3f} < {self.stability_threshold}"

        # Check MAE increase
        champion_mae = champion_predictions_df.agg(
            F.mean(F.abs(F.col("champion_prediction") - F.col("actual_value")))
        ).collect()[0][0]

        challenger_mae = challenger_predictions_df.agg(
            F.mean(F.abs(F.col("challenger_prediction") - F.col("actual_value")))
        ).collect()[0][0]

        mae_increase = (challenger_mae - champion_mae) / champion_mae if champion_mae > 0 else 0

        if mae_increase > self.mae_increase_threshold:
            return True, f"mae_increased_too_much: {mae_increase:.1%} > {self.mae_increase_threshold:.1%}"

        # No rollback conditions met
        return False, None

    def _get_next_rollout_percentage(self, current_pct: int) -> Optional[int]:
        """Get next rollout percentage from schedule."""
        for pct in self.rollout_schedule:
            if pct > current_pct:
                return pct
        return None  # Already at max

    def _initialize_state_table(self):
        """Initialize rollout state table if not exists."""
        try:
            self.spark.table(self.state_table)
            logger.info(f"Rollout state table exists: {self.state_table}")
        except Exception:
            logger.info(f"Creating rollout state table: {self.state_table}")

            # Create initial state (0% rollout)
            initial_state = self.spark.createDataFrame([{
                "rollout_percentage": 0,
                "timestamp": datetime.now().isoformat(),
                "reason": "initial_state",
                "approved_by": "system",
                "status": "active"
            }])

            initial_state.write \
                .format("delta") \
                .mode("overwrite") \
                .saveAsTable(self.state_table)

            logger.info(f"✓ Created rollout state table: {self.state_table}")

    def _send_rollback_alert(self, from_pct: int, reason: str):
        """Send rollback alert to monitoring channels."""
        # Placeholder for Slack/PagerDuty integration
        logger.warning(f"ALERT: Rollback from {from_pct}% to 0%")
        logger.warning(f"Reason: {reason}")
        # TODO: Implement actual alerting


def get_rollout_status(spark: SparkSession, state_table: str = "decision_agent.rollout_state") -> Dict[str, Any]:
    """
    Get current rollout status.

    Args:
        spark: Spark session
        state_table: Rollout state table

    Returns:
        Current rollout status
    """
    current_state = spark.table(state_table) \
        .orderBy(F.col("timestamp").desc()) \
        .limit(1) \
        .collect()

    if current_state:
        row = current_state[0]
        return {
            "rollout_percentage": row["rollout_percentage"],
            "timestamp": row["timestamp"],
            "reason": row["reason"],
            "approved_by": row["approved_by"],
            "status": row["status"]
        }
    else:
        return {
            "rollout_percentage": 0,
            "timestamp": None,
            "reason": "no_state",
            "approved_by": None,
            "status": "unknown"
        }
