"""
HoldoutAgent
============
Assigns 5% of accounts per SIGNAL_SEGMENT to a holdout (control) group for
causal uplift measurement. Runs after FeatureAgent and before ModelAgent.

Accounts flagged as holdout receive HOLD action regardless of ERV.
They form the control group used to measure the causal uplift of treatments
(contact, discount, channel) on P(recovery).

CONFIGURATION INSTRUCTIONS
---------------------------

HOLDOUT_PCT (default 5)
  Percentage of accounts per SIGNAL_SEGMENT assigned to the control group.

  How to override:
      HoldoutAgent(..., holdout_pct=10)

  Why deterministic hash (not random):
      Uses hash(account_id) % 100 < holdout_pct. This ensures the same account
      always falls in the control group across daily runs. Random assignment
      would contaminate the treatment/control comparison if an account is treated
      on some days and held out on others.

  When to increase holdout_pct:
      When you need more statistical power for uplift estimation (e.g., small
      treatment effect sizes, high outcome variance). More control accounts
      reduces variance in the incremental rate estimate.

  When to decrease holdout_pct:
      When the portfolio is small and holding out too many accounts reduces
      revenue materially. At 5% the revenue cost is minimal while still giving
      enough control observations for monthly model retraining.

SEGMENT_HOLDOUT_OVERRIDE (optional dict)
  Set different holdout percentages per SIGNAL_SEGMENT. Useful for segments
  where you want higher exploration to build training data faster.

  Example — higher holdout for low-signal Segment D:
      HoldoutAgent(..., segment_holdout_override={"D": 20})

  How to set:
      Pass a dict mapping segment label (str) to holdout_pct (int/float).
      Any segment not in the override dict uses the global holdout_pct.

  Practical use:
      Segment D accounts have low recovery probability and are normally HOLD
      anyway. Increasing their holdout percentage costs little revenue but
      builds labelled control data faster, improving D-segment uplift estimates.
"""

import hashlib
import logging
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from agents.base_agent import AgentBlockedException, BaseAgent

logger = logging.getLogger(__name__)


class HoldoutAgent(BaseAgent):
    """
    Assigns accounts to CONTROL or TREATMENT group using a deterministic hash.

    Pipeline position: after FeatureAgent, before ModelAgent.
    Reads:  recovery.feature_output
    Writes: recovery.feature_output  (same table — adds holdout columns, overwrite)
            recovery.treatment_log   (append mode — one row per holdout account)
    """

    agent_name = "holdout_agent"

    def __init__(
        self,
        execution_date: str,
        memory,
        dry_run: bool = False,
        holdout_pct: float = 5.0,
        segment_holdout_override: Optional[dict] = None,
    ):
        """
        Args:
            execution_date:           Scoring date (YYYY-MM-DD).
            memory:                   AgentMemory instance.
            dry_run:                  If True, skip all Delta writes.
            holdout_pct:              Global % per SIGNAL_SEGMENT assigned to
                                      control group. Default 5.
            segment_holdout_override: Optional dict of {segment: pct} overrides.
                                      e.g. {"D": 20} sets Segment D to 20%
                                      holdout regardless of global holdout_pct.
        """
        super().__init__(
            agent_name=self.agent_name,
            execution_date=execution_date,
            memory=memory,
            dry_run=dry_run,
        )
        self.holdout_pct = holdout_pct
        self.segment_holdout_override = segment_holdout_override or {}

    # ── Core logic ────────────────────────────────────────────────────────────

    def execute(self) -> dict:
        """
        Reads feature_output, assigns holdout flags, writes back.

        Returns:
            dict with keys: total_accounts, holdout_count, holdout_rate,
                            holdout_by_segment
        """
        self.log("Reading feature_output")
        df = self._read_feature_output()

        # Assign holdout columns
        df = self._assign_holdout(df)

        # Build treatment_log rows for holdout accounts only
        holdout_df = df[df["holdout_flag"]].copy()

        if not self.dry_run:
            # Overwrite feature_output with new holdout columns
            self.memory.write("feature_output", df, overwrite=True)
            self.log(f"Wrote feature_output with holdout columns | rows={len(df):,}")

            # Append holdout accounts to treatment_log
            if len(holdout_df) > 0:
                log_df = self._build_treatment_log(holdout_df)
                self.memory.write("treatment_log", log_df, overwrite=False)
                self.log(f"Appended {len(log_df):,} rows to treatment_log")

        # Summary stats
        total       = len(df)
        holdout_ct  = int(df["holdout_flag"].sum())
        holdout_rate = holdout_ct / total if total > 0 else 0.0
        holdout_by_segment = (
            df[df["holdout_flag"]]
            .groupby("signal_segment")
            .size()
            .to_dict()
        )

        self.log(
            f"Holdout assignment complete | total={total:,} | "
            f"holdout={holdout_ct:,} ({holdout_rate:.2%}) | "
            f"by_segment={holdout_by_segment}"
        )

        return {
            "total_accounts":    total,
            "holdout_count":     holdout_ct,
            "holdout_rate":      round(holdout_rate, 4),
            "holdout_by_segment": holdout_by_segment,
        }

    # ── Assignment logic ──────────────────────────────────────────────────────

    def _assign_holdout(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Assigns holdout_flag, holdout_group, holdout_assigned_date columns.

        Uses deterministic hash: hash(account_id) % 100 < holdout_pct(segment).
        This guarantees the same account always lands in the same group across
        daily pipeline runs — critical for valid causal measurement.
        """
        df = df.copy()

        def _get_pct(segment: str) -> float:
            return float(self.segment_holdout_override.get(segment, self.holdout_pct))

        def _is_holdout(row) -> bool:
            account_id = str(row["account_id"])
            pct = _get_pct(str(row.get("signal_segment", "")))
            # Use MD5 for a stable, well-distributed hash
            hash_val = int(hashlib.md5(account_id.encode()).hexdigest(), 16) % 100
            return hash_val < pct

        self.log("Computing holdout flags (deterministic hash)")
        df["holdout_flag"] = df.apply(_is_holdout, axis=1)
        df["holdout_group"] = df["holdout_flag"].map(
            {True: "CONTROL", False: "TREATMENT"}
        )
        df["holdout_assigned_date"] = self.execution_date

        return df

    # ── I/O helpers ───────────────────────────────────────────────────────────

    def _read_feature_output(self) -> pd.DataFrame:
        """Read feature_output; raise AgentBlockedException if empty."""
        raw = self.memory.read("feature_output")

        # Convert Spark DF to pandas if needed
        if hasattr(raw, "toPandas"):
            df = raw.toPandas()
        else:
            df = raw

        if df is None or len(df) == 0:
            raise AgentBlockedException(
                "feature_output is empty — FeatureAgent may not have completed successfully.",
                retry_agent="feature_agent",
            )

        self.log(f"Read feature_output | rows={len(df):,}")
        return df

    def _build_treatment_log(self, holdout_df: pd.DataFrame) -> pd.DataFrame:
        """
        Build treatment_log rows from holdout accounts.

        Schema: account_id, execution_date, signal_segment, holdout_group,
                holdout_flag, score_date, propensity_30d, erv_at_d_optimal,
                assigned_at
        """
        assigned_at = datetime.now(timezone.utc).isoformat()

        def _safe_col(df: pd.DataFrame, col: str, default=None):
            return df[col] if col in df.columns else default

        log_df = pd.DataFrame({
            "account_id":        holdout_df["account_id"],
            "execution_date":    self.execution_date,
            "signal_segment":    _safe_col(holdout_df, "signal_segment", "UNKNOWN"),
            "holdout_group":     holdout_df["holdout_group"],
            "holdout_flag":      holdout_df["holdout_flag"],
            "score_date":        _safe_col(holdout_df, "score_date", self.execution_date),
            "propensity_30d":    _safe_col(holdout_df, "propensity_30d"),
            "erv_at_d_optimal":  _safe_col(holdout_df, "erv_at_d_optimal"),
            "assigned_at":       assigned_at,
        })

        return log_df
