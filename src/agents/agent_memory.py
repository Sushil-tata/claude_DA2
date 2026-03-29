"""
AgentMemory
===========
Delta Lake as the shared state bus between all agents.

Design principles:
  - Every agent reads and writes through this class — no direct Spark table refs
  - All writes are partitioned by execution_date for idempotency
  - Audit log is append-only — never overwritten
  - Works in LOCAL mode (pandas + parquet) when Spark is not available,
    so agents can be unit-tested without a Databricks cluster

Table registry:
  recovery.model_scores         — written by recovery-engine-v2
  recovery.feature_output       — written by FeatureAgent
  recovery.model_agent_output   — written by ModelAgent
  recovery.constraint_overrides — written by ConstraintAgent
  recovery.nba_decisions        — written by DecisionAgent
  recovery.validation_results   — written by ValidationAgent
  recovery.nba_explanations     — written by ExplainAgent
  recovery.data_quality_metrics — written by DataQualityAgent
  recovery.agent_audit_log      — append-only, written by all agents via signal()
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Table registry ────────────────────────────────────────────────────────────
TABLES = {
    "model_scores":          "recovery.model_scores",
    "feature_output":        "recovery.feature_output",
    "model_agent_output":    "recovery.model_agent_output",
    "constraint_overrides":  "recovery.constraint_overrides",
    "nba_decisions":         "recovery.nba_decisions",
    "validation_results":    "recovery.validation_results",
    "nba_explanations":      "recovery.nba_explanations",
    "data_quality_metrics":  "recovery.data_quality_metrics",
    "agent_audit_log":       "recovery.agent_audit_log",
}

# Local parquet paths for non-Databricks execution
LOCAL_BASE = Path(os.environ.get("AGENT_LOCAL_DATA_DIR", "/tmp/agent_data"))


class AgentMemory:
    """
    Unified read/write interface over Delta Lake (Databricks) or
    local Parquet files (CI/CD, unit tests).

    Instantiate with spark=None to use local parquet mode.
    """

    def __init__(
        self,
        execution_date: str,
        spark=None,
        local_mode: bool = False,
    ):
        self.execution_date = execution_date
        self.spark = spark
        self.local_mode = local_mode or (spark is None)

        if self.local_mode:
            LOCAL_BASE.mkdir(parents=True, exist_ok=True)
            logger.info(f"AgentMemory: LOCAL mode | base={LOCAL_BASE}")
        else:
            logger.info(f"AgentMemory: DELTA mode | execution_date={execution_date}")

    # ── Write ─────────────────────────────────────────────────────────────────

    def write(self, table_key: str, df, overwrite: bool = True) -> None:
        """
        Write a DataFrame to the named table for this execution_date.

        Args:
            table_key:  Key from TABLES dict (e.g. "feature_output")
            df:         pandas DataFrame or Spark DataFrame
            overwrite:  Replace today's partition (default True — idempotent)
        """
        table_name = self._resolve(table_key)

        if self.local_mode:
            self._write_local(table_key, df)
        else:
            self._write_delta(table_name, df, overwrite)

    def _write_delta(self, table_name: str, df, overwrite: bool) -> None:
        import pyspark.sql.functions as F
        if hasattr(df, "toPandas"):
            spark_df = df
        else:
            spark_df = self.spark.createDataFrame(df)

        spark_df = spark_df.withColumn("execution_date", F.lit(self.execution_date))

        mode = "overwrite" if overwrite else "append"
        (
            spark_df.write
            .format("delta")
            .mode(mode)
            .option("replaceWhere", f"execution_date = '{self.execution_date}'")
            .saveAsTable(table_name)
        )
        logger.info(f"Wrote {spark_df.count()} rows → {table_name} (date={self.execution_date})")

    def _write_local(self, table_key: str, df) -> None:
        import pandas as pd
        if not isinstance(df, pd.DataFrame):
            df = df.toPandas()
        df["execution_date"] = self.execution_date
        path = LOCAL_BASE / f"{table_key}_{self.execution_date}.parquet"
        df.to_parquet(path, index=False)
        logger.info(f"[LOCAL] Wrote {len(df)} rows → {path}")

    # ── Read ──────────────────────────────────────────────────────────────────

    def read(self, table_key: str, as_pandas: bool = False):
        """
        Read this execution_date's partition from the named table.

        Returns Spark DataFrame in Delta mode, pandas DataFrame in local mode.
        Pass as_pandas=True to always get pandas (useful in agents that use sklearn).
        """
        table_name = self._resolve(table_key)

        if self.local_mode:
            return self._read_local(table_key)

        import pyspark.sql.functions as F
        df = (
            self.spark.read.table(table_name)
            .filter(F.col("execution_date") == self.execution_date)
        )
        if as_pandas:
            return df.toPandas()
        return df

    def _read_local(self, table_key: str):
        import pandas as pd
        path = LOCAL_BASE / f"{table_key}_{self.execution_date}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"[LOCAL] No data found for table='{table_key}' "
                f"date='{self.execution_date}' at {path}"
            )
        df = pd.read_parquet(path)
        logger.info(f"[LOCAL] Read {len(df)} rows ← {path}")
        return df

    # ── Signal (audit log) ────────────────────────────────────────────────────

    def signal(self, agent_signal) -> None:
        """
        Append an AgentSignal to the audit log.
        This is the ONLY table written in append mode (never overwritten).
        """
        import pandas as pd
        row = pd.DataFrame([{
            "agent_name":     agent_signal.agent_name,
            "status":         agent_signal.status.value,
            "execution_date": agent_signal.execution_date,
            "message":        agent_signal.message,
            "metadata":       json.dumps(agent_signal.metadata),
            "timestamp":      agent_signal.timestamp,
        }])

        if self.local_mode:
            log_path = LOCAL_BASE / f"agent_audit_log_{self.execution_date}.parquet"
            if log_path.exists():
                existing = pd.read_parquet(log_path)
                row = pd.concat([existing, row], ignore_index=True)
            row.to_parquet(log_path, index=False)
        else:
            spark_row = self.spark.createDataFrame(row)
            (
                spark_row.write
                .format("delta")
                .mode("append")
                .saveAsTable(TABLES["agent_audit_log"])
            )

    # ── Status check (used by downstream agents to poll upstream) ─────────────

    def get_agent_status(self, agent_name: str) -> Optional[str]:
        """Return the latest status for a given agent on this execution_date."""
        import pandas as pd
        try:
            if self.local_mode:
                log_path = LOCAL_BASE / f"agent_audit_log_{self.execution_date}.parquet"
                if not log_path.exists():
                    return None
                df = pd.read_parquet(log_path)
            else:
                df = (
                    self.spark.read.table(TABLES["agent_audit_log"])
                    .filter(f"execution_date = '{self.execution_date}'")
                    .filter(f"agent_name = '{agent_name}'")
                    .toPandas()
                )
            if df.empty:
                return None
            return df.sort_values("timestamp").iloc[-1]["status"]
        except Exception as e:
            logger.warning(f"Could not read agent status: {e}")
            return None

    # ── Internal ──────────────────────────────────────────────────────────────

    def _resolve(self, table_key: str) -> str:
        if table_key not in TABLES:
            raise KeyError(
                f"Unknown table key: '{table_key}'. "
                f"Valid keys: {list(TABLES.keys())}"
            )
        return TABLES[table_key]
