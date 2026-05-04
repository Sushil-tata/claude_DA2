"""
ExplainAgent
============
Task 7 — final agent. Runs after ValidationAgent.

Reads:
  recovery.validation_results — validated decisions
  recovery.model_scores       — original model scores for context

Responsibilities:
  - Produces human-readable narratives for the collector dashboard
  - Format: "Account X in Segment A with 61% 180d recovery probability.
             Recommended 30% discount (ERV: THB 12,500). Digital + call channel.
             Confidence: High."
  - Flags human_review_required accounts with prominent dashboard tags
  - Writes one explanation row per account

Writes:
  recovery.nba_explanations
"""

import logging

import pandas as pd

from agents.base_agent import AgentBlockedException, BaseAgent

logger = logging.getLogger(__name__)

CONFIDENCE_MAP = {
    (True,  "A"): ("High",   "✅"),
    (True,  "B"): ("High",   "✅"),
    (True,  "C"): ("Medium", "⚠️"),
    (True,  "D"): ("Low",    "🔴"),
    (False, "A"): ("High",   "✅"),
    (False, "B"): ("High",   "✅"),
    (False, "C"): ("Medium", "⚠️"),
    (False, "D"): ("Low",    "🔴"),
}

ACTION_LABEL = {
    "PROCEED":             "Proceed with recommended action",
    "HOLD":                "Hold — no outbound contact",
    "REFER_HUMAN_REVIEW":  "⚠️ Refer to senior collector for manual review",
    "DIGITAL_ONLY":        "Digital self-serve nudge",
    "DIGITAL_PLUS_CALL":   "Digital nudge + agent call",
    "AGENT_CALL":          "Agent call required",
}


class ExplainAgent(BaseAgent):

    def __init__(self, execution_date: str, memory, dry_run: bool = False):
        super().__init__(
            agent_name="explain_agent",
            execution_date=execution_date,
            memory=memory,
            dry_run=dry_run,
        )

    def execute(self) -> dict:
        self.log("Reading validation_results and model_scores")
        validated   = self._read_validated()
        model_scores = self._read_model_scores()

        # Merge for additional context (propensity_180d, segment_label etc.)
        if not model_scores.empty:
            merge_cols = ["account_id"] + [
                c for c in ["propensity_180d", "segment_label", "model_version"]
                if c in model_scores.columns and c not in validated.columns
            ]
            validated = validated.merge(
                model_scores[merge_cols], on="account_id", how="left"
            )

        explanations = validated.copy()
        explanations["narrative"]           = explanations.apply(self._build_narrative, axis=1)
        explanations["dashboard_tag"]       = explanations.apply(self._dashboard_tag, axis=1)
        explanations["confidence_label"]    = explanations.apply(self._confidence_label, axis=1)
        explanations["explanation_version"] = "v1.0"
        explanations["generated_at"]        = pd.Timestamp.utcnow().isoformat()

        if not self.dry_run:
            self.memory.write("nba_explanations", explanations)

        human_review_count = int(explanations["human_review_required"].fillna(False).sum())
        self.log(
            f"Explanations written | rows={len(explanations):,} | "
            f"human_review_flagged={human_review_count}"
        )

        return {
            "rows_written":       len(explanations),
            "human_review_count": human_review_count,
            "sample_narrative":   (
                explanations["narrative"].iloc[0] if len(explanations) > 0 else ""
            ),
        }

    # ── Narrative builders ────────────────────────────────────────────────────

    def _build_narrative(self, row) -> str:
        account_id  = row.get("account_id", "Unknown")
        segment     = row.get("segment_label") or "Unknown Segment"
        quadrant    = row.get("signal_quadrant", "D")
        p180        = row.get("propensity_180d", 0.0) or 0.0
        d_opt       = row.get("final_d_optimal", 0.0) or 0.0
        erv         = row.get("erv_at_d_optimal", 0.0) or 0.0
        action      = row.get("final_action", "HOLD")
        low_conf    = row.get("low_confidence_flag", False)
        override    = row.get("constraint_override_flag", False)
        human_rev   = row.get("human_review_required", False)

        confidence_label, _ = CONFIDENCE_MAP.get(
            (not low_conf, quadrant), ("Medium", "⚠️")
        )
        action_label = ACTION_LABEL.get(action, action)

        parts = [
            f"Account {account_id} | Quadrant {quadrant} | Segment: {segment}.",
            f"180-day recovery probability: {p180*100:.0f}%.",
            f"Recommended discount: {d_opt*100:.0f}% | ERV: THB {erv:,.0f}.",
            f"Action: {action_label}.",
            f"Confidence: {confidence_label}.",
        ]

        if override:
            parts.append("⚠️ Note: Business rule override applied to model recommendation.")
        if human_rev:
            parts.append(
                "🔴 HUMAN REVIEW REQUIRED — escalate to senior collector before actioning."
            )

        return " ".join(parts)

    def _dashboard_tag(self, row) -> str:
        if row.get("human_review_required"):
            return "HUMAN_REVIEW"
        if row.get("constraint_override_flag"):
            return "OVERRIDE_APPLIED"
        q = row.get("signal_quadrant", "D")
        if q == "A":
            return "PRIORITY_ACTION"
        if q in ("B", "C"):
            return "STANDARD"
        return "MONITOR_ONLY"

    def _confidence_label(self, row) -> str:
        low_conf = row.get("low_confidence_flag", False)
        q        = row.get("signal_quadrant", "D")
        label, _ = CONFIDENCE_MAP.get((not low_conf, q), ("Medium", ""))
        return label

    # ── Readers ───────────────────────────────────────────────────────────────

    def _read_validated(self) -> pd.DataFrame:
        df = self.memory.read("validation_results")
        if hasattr(df, "toPandas"):
            df = df.toPandas()
        if df.empty:
            raise AgentBlockedException(
                "validation_results is empty — ValidationAgent may have failed.",
                retry_agent="validation_agent",
            )
        return df

    def _read_model_scores(self) -> pd.DataFrame:
        try:
            df = self.memory.read("model_scores")
            if hasattr(df, "toPandas"):
                df = df.toPandas()
            return df
        except Exception as e:
            self.log(f"Could not read model_scores for enrichment: {e}", level="warning")
            return pd.DataFrame()
