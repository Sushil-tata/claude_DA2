"""
DecisionAgent
=============
Task 5 — runs after BOTH ModelAgent and ConstraintAgent complete.

Reads:
  recovery.model_agent_output   — channel + timing recommendations
  recovery.constraint_overrides — business rule overrides

Responsibilities:
  - Merges model selections with constraint overrides
  - Calls claude_reasoner for qualifying accounts:
      * erv_at_d_optimal > THB 30,000
      * low_confidence_flag=True AND erv > THB 20,000
      * propensity_180d < 0.15 AND segment_label is NOT null
  - Produces final_d_optimal, final_channel, final_action per account
  - Writes one decision row per account to recovery.nba_decisions

Writes:
  recovery.nba_decisions
"""

import logging

import pandas as pd

from agents.base_agent import AgentBlockedException, BaseAgent
from agents.claude_reasoner import ClaudeReasoner

logger = logging.getLogger(__name__)


class DecisionAgent(BaseAgent):

    def __init__(self, execution_date: str, memory, dry_run: bool = False):
        super().__init__(
            agent_name="decision_agent",
            execution_date=execution_date,
            memory=memory,
            dry_run=dry_run,
        )
        self.reasoner = ClaudeReasoner(dry_run=dry_run)

    def execute(self) -> dict:
        self.log("Reading model_agent_output and constraint_overrides")
        model_df      = self._read_model_output()
        constraint_df = self._read_constraints()

        # ── Merge on account_id ───────────────────────────────────────────────
        decisions = model_df.merge(
            constraint_df[[
                "account_id", "constraint_override_flag", "override_reason",
                "final_d_optimal", "final_action", "legal_review_flag",
            ]],
            on="account_id",
            how="left",
        )

        # Fill defaults for accounts without constraint overrides
        decisions["constraint_override_flag"] = (
            decisions["constraint_override_flag"].fillna(False)
        )
        decisions["final_d_optimal"] = (
            decisions["final_d_optimal"].fillna(decisions["d_optimal"])
        )
        decisions["final_action"] = decisions["final_action"].fillna("PROCEED")
        decisions["legal_review_flag"] = decisions["legal_review_flag"].fillna(False)

        # ── Apply Claude reasoning for qualifying accounts ────────────────────
        decisions["claude_called"]       = False
        decisions["claude_recommendation"] = "PROCEED"
        decisions["claude_confidence"]   = "HIGH"
        decisions["claude_rationale"]    = ""
        decisions["claude_suggested_d"]  = None

        qualifying_mask = self._qualifying_for_claude(decisions)
        qualifying_idx  = decisions[qualifying_mask].index

        claude_call_count = 0
        if len(qualifying_idx) > 0:
            self.log(f"Calling claude_reasoner for {len(qualifying_idx)} accounts")
            for idx in qualifying_idx:
                row = decisions.loc[idx]
                advice = self.reasoner.advise({
                    "account_id":              row["account_id"],
                    "signal_quadrant":         row["signal_quadrant"],
                    "segment_label":           row.get("segment_label"),
                    "propensity_30d":          row.get("propensity_30d"),
                    "propensity_90d":          row.get("propensity_90d"),
                    "propensity_180d":         row.get("propensity_180d"),
                    "low_confidence_flag":     row.get("low_confidence_flag"),
                    "d_optimal":               row["final_d_optimal"],
                    "erv_at_d_optimal":        row["erv_at_d_optimal"],
                    "constraint_override_flag":row["constraint_override_flag"],
                })
                decisions.loc[idx, "claude_called"]         = advice["called_claude"]
                decisions.loc[idx, "claude_recommendation"] = advice["recommendation"]
                decisions.loc[idx, "claude_confidence"]     = advice["confidence"]
                decisions.loc[idx, "claude_rationale"]      = advice["rationale"]
                if advice.get("suggested_d_optimal"):
                    decisions.loc[idx, "claude_suggested_d"] = advice["suggested_d_optimal"]
                if advice["called_claude"]:
                    claude_call_count += 1

        # ── Apply Claude overrides where recommendation != PROCEED ────────────
        override_mask = decisions["claude_recommendation"].isin(["OVERRIDE", "REFER_HUMAN"])
        decisions.loc[override_mask & (decisions["claude_suggested_d"].notna()),
                      "final_d_optimal"] = decisions.loc[
            override_mask & (decisions["claude_suggested_d"].notna()), "claude_suggested_d"
        ]
        decisions.loc[decisions["claude_recommendation"] == "REFER_HUMAN",
                      "final_action"] = "REFER_HUMAN_REVIEW"

        # ── Final decision timestamp ──────────────────────────────────────────
        from datetime import datetime, timezone
        decisions["decision_timestamp"] = datetime.now(timezone.utc).isoformat()
        decisions["execution_date"]     = self.execution_date

        if not self.dry_run:
            self.memory.write("nba_decisions", decisions)

        action_dist = decisions["final_action"].value_counts().to_dict()
        self.log(
            f"Decisions written | rows={len(decisions):,} | "
            f"claude_calls={claude_call_count} | actions={action_dist}"
        )

        return {
            "rows_written":       len(decisions),
            "claude_calls_made":  claude_call_count,
            "action_distribution":action_dist,
            "refer_human_count":  int((decisions["final_action"] == "REFER_HUMAN_REVIEW").sum()),
        }

    # ── Claude qualification logic ────────────────────────────────────────────

    @staticmethod
    def _qualifying_for_claude(df: pd.DataFrame) -> pd.Series:
        """Boolean mask — True for accounts that need Claude reasoning."""
        high_value = df["erv_at_d_optimal"] > 30_000

        d_quadrant_high = (
            df["low_confidence_flag"].fillna(False) &
            (df["erv_at_d_optimal"] > 20_000)
        )

        unexpected_low = (
            (df["propensity_180d"] < 0.15) &
            df["segment_label"].notna()
        )

        return high_value | d_quadrant_high | unexpected_low

    # ── Readers ───────────────────────────────────────────────────────────────

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

    def _read_constraints(self) -> pd.DataFrame:
        df = self.memory.read("constraint_overrides")
        if hasattr(df, "toPandas"):
            df = df.toPandas()
        # Constraints are optional (no block if empty — model output stands)
        if df.empty:
            self.log("constraint_overrides is empty — using model output as-is", level="warning")
        return df
