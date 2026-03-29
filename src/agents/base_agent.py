"""
BaseAgent
=========
All agents in the agentic NBA pipeline inherit from this class.

Responsibilities:
  - Enforce a standard run() lifecycle: start → execute → signal → complete
  - Write status signals to Delta Lake (via AgentMemory) so downstream
    agents and the Databricks Workflow can react to BLOCKED / FAILED states
  - Provide structured logging with agent name + execution_date context
  - Surface a clean interface for the Databricks Workflow entry point

Usage (subclass pattern):
    class FeatureAgent(BaseAgent):
        def execute(self) -> dict:
            df = self.memory.read("model_scores")
            ...
            self.memory.write("feature_output", result_df)
            return {"rows_written": len(result_df)}
"""

import logging
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class AgentStatus(str, Enum):
    PENDING   = "PENDING"
    STARTED   = "STARTED"
    COMPLETE  = "COMPLETE"
    BLOCKED   = "BLOCKED"   # upstream data problem — workflow should retry/alert
    FAILED    = "FAILED"    # unrecoverable — workflow should stop pipeline


@dataclass
class AgentSignal:
    agent_name:     str
    status:         AgentStatus
    execution_date: str
    message:        str
    metadata:       dict = field(default_factory=dict)
    timestamp:      str  = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class AgentBlockedException(Exception):
    """Raised when an agent detects a blocking data quality issue."""
    def __init__(self, message: str, retry_agent: Optional[str] = None):
        super().__init__(message)
        self.retry_agent = retry_agent


class BaseAgent(ABC):
    """
    Abstract base class for all pipeline agents.

    Subclasses must implement:
      execute(self) -> dict   — the agent's core logic, returns a result summary dict

    Subclasses may override:
      on_blocked(self, exc)   — custom handling when AgentBlockedException is raised
      on_failed(self, exc)    — custom handling on unexpected failure
    """

    def __init__(
        self,
        agent_name: str,
        execution_date: str,
        memory,                    # AgentMemory instance — injected at runtime
        dry_run: bool = False,
    ):
        self.agent_name     = agent_name
        self.execution_date = execution_date
        self.memory         = memory
        self.dry_run        = dry_run
        self._logger        = logging.getLogger(f"agent.{agent_name}")

    # ── Public entry point (called by Databricks task) ────────────────────────

    def run(self) -> AgentSignal:
        """
        Standard lifecycle:
          1. Signal STARTED to Delta audit log
          2. Call self.execute()
          3. Signal COMPLETE (or BLOCKED / FAILED on exception)
          4. Return final AgentSignal

        The Databricks Workflow reads BLOCKED/FAILED signals to decide
        whether to retry, alert, or halt the pipeline.
        """
        self._signal(AgentStatus.STARTED, "Agent started")
        self._logger.info(f"[{self.agent_name}] STARTED | date={self.execution_date} | dry_run={self.dry_run}")

        try:
            result = self.execute()
            signal = self._signal(
                AgentStatus.COMPLETE,
                "Agent completed successfully",
                metadata=result or {},
            )
            self._logger.info(f"[{self.agent_name}] COMPLETE | result={result}")
            return signal

        except AgentBlockedException as exc:
            signal = self._signal(
                AgentStatus.BLOCKED,
                str(exc),
                metadata={"retry_agent": exc.retry_agent},
            )
            self._logger.warning(f"[{self.agent_name}] BLOCKED | reason={exc} | retry={exc.retry_agent}")
            self.on_blocked(exc)
            raise   # re-raise so Databricks Workflow sees a non-zero exit

        except Exception as exc:
            tb = traceback.format_exc()
            signal = self._signal(
                AgentStatus.FAILED,
                str(exc),
                metadata={"traceback": tb},
            )
            self._logger.error(f"[{self.agent_name}] FAILED | error={exc}\n{tb}")
            self.on_failed(exc)
            raise

    # ── Abstract interface ────────────────────────────────────────────────────

    @abstractmethod
    def execute(self) -> dict:
        """
        Core agent logic. Must:
          - Read from self.memory as needed
          - Write results back to self.memory
          - Return a dict summarising what was done (used in audit log)
          - Raise AgentBlockedException if a blocking condition is detected
        """

    # ── Optional hooks ────────────────────────────────────────────────────────

    def on_blocked(self, exc: AgentBlockedException) -> None:
        """Override for custom BLOCKED handling (e.g., send Slack alert)."""

    def on_failed(self, exc: Exception) -> None:
        """Override for custom FAILED handling (e.g., page on-call)."""

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _signal(
        self,
        status: AgentStatus,
        message: str,
        metadata: Optional[dict] = None,
    ) -> AgentSignal:
        signal = AgentSignal(
            agent_name=self.agent_name,
            status=status,
            execution_date=self.execution_date,
            message=message,
            metadata=metadata or {},
        )
        if not self.dry_run:
            try:
                self.memory.signal(signal)
            except Exception as e:
                # Never let audit logging kill the pipeline
                self._logger.warning(f"Failed to write signal to Delta: {e}")
        return signal

    def log(self, message: str, level: str = "info") -> None:
        getattr(self._logger, level)(f"[{self.agent_name}] {message}")
