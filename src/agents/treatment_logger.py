"""
TreatmentLogger
===============
Utility class (NOT a BaseAgent) that appends every treatment decision to
recovery.treatment_log. Called by ModelAgent after writing model_agent_output.

This is the data collection foundation for Phase 2 uplift model training.
Without this log, UpliftModelTrainer.fit() cannot start — there is no labelled
treatment/control data.

IMPORTANT NOTES
---------------

treatment_log IS APPEND-ONLY — NEVER OVERWRITE:
  Every daily run appends to this table. It is a running log of all treatment
  decisions across the full history of the pipeline. Overwriting would destroy
  the historical record needed for uplift model training.

HOW OUTCOMES ARE JOINED BACK:
  paid_180d, recovery_amount, and other outcome labels are NOT written here.
  They are joined back to this table by a separate outcome labelling job that
  runs 30, 90, and 180 days after the decision date. That job matches on
  (account_id, execution_date) to attach the eventual outcome to the treatment
  decision record.

  Example join key: treatment_log.account_id = outcomes.account_id
                    treatment_log.execution_date = outcomes.decision_date

SCHEMA STABILITY:
  This table's schema must be stable. Adding new columns requires a migration
  (ALTER TABLE or Delta Lake schema evolution — set mergeSchema=true in the
  write path). Never remove or rename existing columns — downstream training
  jobs depend on the column names in UPLIFT_FEATURES.

PIPELINE VERSION:
  The pipeline_version column ("phase2_uplift_v1") lets you filter training
  data to records produced by the current model version. This is important
  when the action set or feature engineering changes between versions.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Columns extracted from model_agent_output with their defaults if absent
_EXTRACT_COLUMNS = {
    "account_id":          None,
    "execution_date":      None,
    "signal_segment":      "UNKNOWN",
    "holdout_group":       "UNKNOWN",
    "holdout_flag":        False,
    "recommended_action":  "UNKNOWN",
    "allocated_action":    None,         # from capacity_agent — may be absent
    "d_optimal":           None,
    "erv_at_d_optimal":    None,
    "propensity_30d":      None,
    "propensity_180d":     None,
    "behavioural_persona": "UNKNOWN",
    "final_segment":       None,
}

PIPELINE_VERSION = "phase2_uplift_v1"


class TreatmentLogger:
    """
    Appends treatment decisions from the model agent to recovery.treatment_log.

    This is a utility class, not a BaseAgent. It is called directly from
    within ModelAgent.execute() after writing model_agent_output.

    Usage:
        logger = TreatmentLogger(memory=self.memory, execution_date=self.execution_date)
        rows_logged = logger.log_treatments(model_output_df)
    """

    def __init__(self, memory, execution_date: str):
        """
        Args:
            memory:         AgentMemory instance (shared with calling agent).
            execution_date: Pipeline execution date (YYYY-MM-DD).
        """
        self.memory         = memory
        self.execution_date = execution_date
        self._logger        = logging.getLogger(__name__)

    def log_treatments(self, df: pd.DataFrame) -> int:
        """
        Extract treatment decision columns from model_agent_output and append
        to recovery.treatment_log.

        Args:
            df: model_agent_output DataFrame (output of ModelAgent.execute).
                Expected to contain the columns listed in _EXTRACT_COLUMNS.
                Missing columns are filled with their default values.

        Returns:
            int: number of rows appended to treatment_log.
        """
        if df is None or len(df) == 0:
            self._logger.warning("TreatmentLogger.log_treatments: empty DataFrame — nothing logged.")
            return 0

        log_df = self._extract_columns(df)
        log_df = self._add_metadata(log_df)

        # Append-only write — NEVER overwrite treatment_log
        self.memory.write("treatment_log", log_df, overwrite=False)

        rows_logged = len(log_df)
        self._logger.info(
            f"TreatmentLogger: appended {rows_logged:,} rows to treatment_log "
            f"| execution_date={self.execution_date} "
            f"| pipeline_version={PIPELINE_VERSION}"
        )
        return rows_logged

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _extract_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build the treatment_log DataFrame from model_agent_output.

        For columns absent in df, applies default values from _EXTRACT_COLUMNS.
        """
        result = {}

        for col, default in _EXTRACT_COLUMNS.items():
            if col in df.columns:
                result[col] = df[col].values
            else:
                self._logger.debug(
                    f"TreatmentLogger: column '{col}' not in model_agent_output "
                    f"— using default={default!r}"
                )
                result[col] = default

        log_df = pd.DataFrame(result, index=df.index)

        # Ensure execution_date is set from self if not in df
        if "execution_date" not in df.columns:
            log_df["execution_date"] = self.execution_date

        return log_df

    def _add_metadata(self, log_df: pd.DataFrame) -> pd.DataFrame:
        """Attach pipeline metadata columns to treatment_log rows."""
        log_df = log_df.copy()
        log_df["logged_at"]        = datetime.now(timezone.utc).isoformat()
        log_df["pipeline_version"] = PIPELINE_VERSION
        return log_df
