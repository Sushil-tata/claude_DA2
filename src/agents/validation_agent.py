"""
ValidationAgent
===============
Task 6 — runs after DecisionAgent. Gate before any action is taken.

Reads:
  recovery.nba_decisions — from DecisionAgent

Responsibilities:
  - Validates all decisions are within allowed business parameters
  - Flags for human review: constraint_override=True AND erv > THB 50,000
  - Raises BLOCKED if > 5% of decisions fail validation rules
  - Computes decision quality metrics

Writes:
  recovery.validation_results — one row per account with pass/fail + flags
"""

import logging

import pandas as pd

from agents.base_agent import AgentBlockedException, BaseAgent

logger = logging.getLogger(__name__)

# ── Validation thresholds ─────────────────────────────────────────────────────
HUMAN_REVIEW_ERV_THRESHOLD     = 50_000.0   # THB
MAX_FAILURE_RATE               = 0.05       # 5% hard block
MAX_DISCOUNT_HARD_CAP          = 0.65       # absolute maximum
MIN_PROPENSITY_FOR_CONTACT     = 0.05       # below this, contacting is wasteful


class ValidationAgent(BaseAgent):

    def __init__(self, execution_date: str, memory, dry_run: bool = False):
        super().__init__(
            agent_name="validation_agent",
            execution_date=execution_date,
            memory=memory,
            dry_run=dry_run,
        )

    def execute(self) -> dict:
        self.log("Reading nba_decisions for validation")
        decisions = self._read_decisions()

        results = decisions[[
            "account_id", "score_date", "signal_quadrant",
            "erv_at_d_optimal", "final_d_optimal", "final_action",
            "constraint_override_flag", "legal_review_flag",
            "propensity_180d", "low_confidence_flag",
        ]].copy()

        results["validation_passed"]    = True
        results["failure_reason"]       = ""
        results["human_review_required"]= False
        results["human_review_reason"]  = ""

        # ── Rule 1: Discount cap ──────────────────────────────────────────────
        bad_discount = results["final_d_optimal"] > MAX_DISCOUNT_HARD_CAP
        results.loc[bad_discount, "validation_passed"] = False
        results.loc[bad_discount, "failure_reason"]    = (
            f"final_d_optimal exceeds hard cap {MAX_DISCOUNT_HARD_CAP}"
        )

        # ── Rule 2: Contact on D-quadrant low confidence ───────────────────────
        bad_contact = (
            results["low_confidence_flag"].fillna(False) &
            (results["final_action"] != "HOLD") &
            (results["signal_quadrant"] == "D")
        )
        results.loc[bad_contact, "validation_passed"] = False
        results.loc[bad_contact, "failure_reason"]    = (
            "D-quadrant low_confidence account must be HOLD — contact blocked"
        )

        # ── Rule 3: Wasteful contact (near-zero propensity) ───────────────────
        wasteful = (
            (results["propensity_180d"] < MIN_PROPENSITY_FOR_CONTACT) &
            (results["final_action"].isin(["DIGITAL_ONLY", "DIGITAL_PLUS_CALL", "AGENT_CALL"]))
        )
        results.loc[wasteful, "validation_passed"] = False
        results.loc[wasteful, "failure_reason"]    = (
            f"propensity_180d < {MIN_PROPENSITY_FOR_CONTACT} but action requires contact"
        )

        # ── Rule 4: Human review flag ─────────────────────────────────────────
        human_review = (
            results["constraint_override_flag"].fillna(False) &
            (results["erv_at_d_optimal"] > HUMAN_REVIEW_ERV_THRESHOLD)
        )
        results.loc[human_review, "human_review_required"] = True
        results.loc[human_review, "human_review_reason"]   = (
            f"constraint_override=True AND erv > {HUMAN_REVIEW_ERV_THRESHOLD:,.0f} THB"
        )

        # ── Rule 5: Legal flag propagation ────────────────────────────────────
        if "legal_review_flag" in results.columns:
            results.loc[results["legal_review_flag"].fillna(False),
                        "human_review_required"] = True
            results.loc[results["legal_review_flag"].fillna(False),
                        "human_review_reason"]   = "legal_review_flagged"

        # ── Check failure rate ────────────────────────────────────────────────
        failure_rate  = (~results["validation_passed"]).mean()
        failure_count = (~results["validation_passed"]).sum()
        human_count   = results["human_review_required"].sum()

        if failure_rate > MAX_FAILURE_RATE:
            raise AgentBlockedException(
                f"Validation failure rate {failure_rate:.1%} exceeds threshold "
                f"{MAX_FAILURE_RATE:.0%} ({failure_count} of {len(results)} accounts failed). "
                "Pipeline halted — review DecisionAgent and ConstraintAgent outputs.",
                retry_agent="decision_agent",
            )

        if not self.dry_run:
            self.memory.write("validation_results", results)

        self.log(
            f"Validation complete | total={len(results):,} | "
            f"passed={results['validation_passed'].sum():,} | "
            f"failed={failure_count} | human_review={human_count}"
        )

        return {
            "total_decisions":    len(results),
            "passed":             int(results["validation_passed"].sum()),
            "failed":             int(failure_count),
            "failure_rate":       round(float(failure_rate), 4),
            "human_review_count": int(human_count),
        }

    def _read_decisions(self) -> pd.DataFrame:
        df = self.memory.read("nba_decisions")
        if hasattr(df, "toPandas"):
            df = df.toPandas()
        if df.empty:
            raise AgentBlockedException(
                "nba_decisions is empty — DecisionAgent may have failed.",
                retry_agent="decision_agent",
            )
        return df
