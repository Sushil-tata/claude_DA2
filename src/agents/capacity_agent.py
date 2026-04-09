"""
CapacityAllocationAgent
=======================
Task 3c — runs after ModelAgent, before DecisionAgent.
Applies portfolio-level capacity constraints per SIGNAL_SEGMENT.

Design principle:
  "Segmentation should describe behaviour, not prescribe action.
   Models should decide action through ERV optimisation."

  Capacity allocation is an OPERATIONAL constraint, not a modelling constraint.
  It limits HOW MANY accounts receive each action based on available resources,
  but does NOT pre-assign actions to segments.

Reads:
  recovery.model_agent_output

Responsibilities:
  - Enforces agency/legal/call capacity limits per SIGNAL_SEGMENT
  - Within each segment × action bucket, prioritises by net ERV descending
  - Accounts that exceed capacity are downgraded to next-best available action
  - Writes allocation flags for DecisionAgent to respect

Writes:
  recovery.capacity_allocation

──────────────────────────────────────────────────────────────────────────────
CONFIGURATION INSTRUCTIONS
──────────────────────────────────────────────────────────────────────────────
AGENCY_CAPACITY_SPLIT (default: A/B=60%, C=30%, D=10%)
    Fraction of total daily agency slots allocated to each SIGNAL_SEGMENT.
    Reflects the quality-weighted approach: stronger signal accounts
    receive priority access to the higher-cost agency action.

    How to override:
        CapacityAllocationAgent(..., agency_capacity_split={"A": 0.5, "B": 0.2, ...})

    How to read: if agency_daily_capacity=1000 and split A=0.60,
    then max 600 Segment A accounts can receive AGENCY action today.
    Remaining Segment A accounts above 600 are downgraded to AGENT_CALL.

AGENT_CALL_DAILY_CAPACITY (default: None = unlimited)
    Maximum number of AGENT_CALL actions per day across all segments.
    Set this based on contact centre headcount.
    Accounts exceeding this are downgraded to DIGITAL_NUDGE.

LEGAL_DAILY_CAPACITY (default: None = unlimited)
    Maximum LEGAL actions per day. Set based on legal team bandwidth.

AGENCY_DAILY_CAPACITY (default: None = unlimited)
    Maximum AGENCY referrals per day. Set based on agency contract limits.

    In Databricks task parameters:
        --agency-daily-capacity 500
        --agent-call-daily-capacity 2000
        --legal-daily-capacity 50

DOWNGRADE_MAP
    When an account cannot receive its preferred action due to capacity limits,
    it is downgraded to the next-best action in this order:
        AGENCY     → AGENT_CALL → DIGITAL_NUDGE → HOLD
        AGENT_CALL → DIGITAL_NUDGE → HOLD
        LEGAL      → AGENCY → AGENT_CALL → HOLD
        DIGITAL_NUDGE → HOLD
──────────────────────────────────────────────────────────────────────────────
"""

import logging

import pandas as pd

from agents.base_agent import AgentBlockedException, BaseAgent

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

# Default agency slot split by SIGNAL_SEGMENT.
# Segments A+B receive 90% of agency capacity — higher signal quality.
# See CONFIGURATION INSTRUCTIONS above.
DEFAULT_AGENCY_CAPACITY_SPLIT = {
    "A": 0.40,
    "B": 0.20,
    "C": 0.30,
    "D": 0.10,
}

# Action downgrade chain: if capacity exhausted, move to next action.
DOWNGRADE_MAP = {
    "AGENCY":        ["AGENT_CALL", "DIGITAL_NUDGE", "HOLD"],
    "AGENT_CALL":    ["DIGITAL_NUDGE", "HOLD"],
    "LEGAL":         ["AGENCY", "AGENT_CALL", "HOLD"],
    "DIGITAL_NUDGE": ["HOLD"],
    "HOLD":          [],
}


class CapacityAllocationAgent(BaseAgent):

    def __init__(
        self,
        execution_date: str,
        memory,
        dry_run: bool = False,
        agency_daily_capacity: int = None,
        agent_call_daily_capacity: int = None,
        legal_daily_capacity: int = None,
        agency_capacity_split: dict = None,
    ):
        """
        Args:
            execution_date:            Scoring date (YYYY-MM-DD).
            memory:                    AgentMemory instance.
            dry_run:                   If True, skip Delta writes.
            agency_daily_capacity:     Max agency referrals today. None = unlimited.
            agent_call_daily_capacity: Max agent calls today. None = unlimited.
            legal_daily_capacity:      Max legal actions today. None = unlimited.
            agency_capacity_split:     Fraction of agency slots per SIGNAL_SEGMENT.
                                       Must sum to 1.0. See module docstring.
        """
        super().__init__(
            agent_name="capacity_agent",
            execution_date=execution_date,
            memory=memory,
            dry_run=dry_run,
        )
        self.agency_daily_capacity     = agency_daily_capacity
        self.agent_call_daily_capacity = agent_call_daily_capacity
        self.legal_daily_capacity      = legal_daily_capacity
        self.agency_capacity_split     = {
            **DEFAULT_AGENCY_CAPACITY_SPLIT,
            **(agency_capacity_split or {}),
        }

    def execute(self) -> dict:
        self.log("Reading model_agent_output for capacity allocation")
        df = self._read_model_output()

        df = df.copy()
        df["allocated_action"]   = df["recommended_action"]
        df["capacity_downgraded"]= False
        df["downgrade_reason"]   = ""

        # ── Apply capacity limits per action ──────────────────────────────────
        df = self._apply_agency_capacity(df)
        df = self._apply_action_capacity(
            df, action="AGENT_CALL",
            daily_cap=self.agent_call_daily_capacity,
        )
        df = self._apply_action_capacity(
            df, action="LEGAL",
            daily_cap=self.legal_daily_capacity,
        )

        downgrade_count = df["capacity_downgraded"].sum()
        self.log(
            f"Capacity allocation done | rows={len(df):,} | "
            f"downgrades={downgrade_count} | "
            f"agency_split={self.agency_capacity_split}"
        )

        if not self.dry_run:
            self.memory.write("capacity_allocation", df[[
                "account_id", "score_date",
                "signal_segment", "recommended_action",
                "allocated_action", "capacity_downgraded", "downgrade_reason",
                "erv_at_d_optimal", "d_optimal",
            ]])

        action_dist = df["allocated_action"].value_counts().to_dict()
        return {
            "rows_processed":    len(df),
            "downgrades":        int(downgrade_count),
            "action_distribution": action_dist,
        }

    # ── Capacity rules ────────────────────────────────────────────────────────

    def _apply_agency_capacity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Allocates agency slots per SIGNAL_SEGMENT based on capacity split.
        Within each segment, prioritises accounts by net ERV descending.
        Accounts exceeding their segment's slot count are downgraded.
        """
        if self.agency_daily_capacity is None:
            return df  # unlimited — no allocation needed

        agency_mask = df["recommended_action"] == "AGENCY"
        if not agency_mask.any():
            return df

        for seg, fraction in self.agency_capacity_split.items():
            seg_agency_mask = agency_mask & (df["signal_segment"] == seg)
            if not seg_agency_mask.any():
                continue

            seg_capacity = int(self.agency_daily_capacity * fraction)
            seg_df       = df[seg_agency_mask].sort_values(
                "erv_at_d_optimal", ascending=False
            )
            overflow_idx = seg_df.index[seg_capacity:]

            if len(overflow_idx) > 0:
                downgrade_to = self._next_action(df.loc[overflow_idx, "allocated_action"])
                df.loc[overflow_idx, "allocated_action"]    = downgrade_to
                df.loc[overflow_idx, "capacity_downgraded"] = True
                df.loc[overflow_idx, "downgrade_reason"]    = (
                    f"agency_capacity_exceeded_seg_{seg}_limit_{seg_capacity}"
                )

        return df

    def _apply_action_capacity(
        self, df: pd.DataFrame, action: str, daily_cap: int
    ) -> pd.DataFrame:
        """
        Applies a simple total daily cap for a given action.
        Prioritises by ERV descending across all segments.
        Overflow accounts are downgraded via DOWNGRADE_MAP.
        """
        if daily_cap is None:
            return df

        action_mask = df["allocated_action"] == action
        if not action_mask.any():
            return df

        action_df    = df[action_mask].sort_values("erv_at_d_optimal", ascending=False)
        overflow_idx = action_df.index[daily_cap:]

        if len(overflow_idx) > 0:
            downgrade_to = DOWNGRADE_MAP.get(action, ["HOLD"])[0] \
                if DOWNGRADE_MAP.get(action) else "HOLD"
            df.loc[overflow_idx, "allocated_action"]    = downgrade_to
            df.loc[overflow_idx, "capacity_downgraded"] = True
            df.loc[overflow_idx, "downgrade_reason"]    = (
                f"{action}_daily_cap_{daily_cap}_exceeded"
            )

        return df

    @staticmethod
    def _next_action(current_actions: pd.Series) -> str:
        """Returns the first downgrade action from DOWNGRADE_MAP."""
        action = current_actions.iloc[0] if len(current_actions) > 0 else "HOLD"
        chain  = DOWNGRADE_MAP.get(action, [])
        return chain[0] if chain else "HOLD"

    def _read_model_output(self) -> pd.DataFrame:
        df = self.memory.read("model_agent_output")
        if hasattr(df, "toPandas"):
            df = df.toPandas()
        if df.empty:
            raise AgentBlockedException(
                "model_agent_output is empty — ModelAgent may have failed.",
                retry_agent="model_agent",
            )
        return df
