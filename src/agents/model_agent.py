"""
ModelAgent
==========
Task 3 (parallel with ConstraintAgent). Runs after FeatureAgent.

Reads:
  recovery.feature_output — from FeatureAgent

Responsibilities:
  - Computes ERV for EVERY action across a discount grid
  - Selects the action + discount that maximises net ERV (after action cost)
  - Adds recommended_channel based on best action + signal_segment
  - Does NOT pre-assign actions based on segment rules
  - Passes clean model selections to DecisionAgent

Writes:
  recovery.model_agent_output

──────────────────────────────────────────────────────────────────────────────
CONFIGURATION INSTRUCTIONS
──────────────────────────────────────────────────────────────────────────────
ACTION_COSTS (default values in THB)
    Cost of each action, deducted from gross ERV before comparison.
    Adjust to reflect your actual cost-per-action from the collections P&L.

    DIGITAL_NUDGE      : ~10 THB  (SMS/push, near-zero variable cost)
    AGENT_CALL         : ~150 THB (avg handle time × agent cost per minute)
    AGENCY             : ~500 THB (agency commission proxy, flat fee component)
    LEGAL              : ~2000 THB (legal letter + case setup cost)
    HOLD               : 0 THB   (no outbound — no cost)

    How to override at runtime (Databricks task parameter):
        --action-costs '{"AGENT_CALL": 200, "AGENCY": 600}'

    How to override in code:
        ModelAgent(..., action_costs={"AGENT_CALL": 200})

ACTION PRIORITY FOR CHARGE-OFF ACCOUNTS
    For accounts in charge-off (signal_segment D or ERV below CHARGE_OFF_ERV_FLOOR):
      1. AGENCY is always evaluated BEFORE LEGAL
      2. LEGAL is only selected if ERV(LEGAL) > ERV(AGENCY) by at least LEGAL_UPLIFT_THRESHOLD
    This reflects operational reality: agency is lower-cost and faster to deploy;
    legal is an escalation path, not the default.

    LEGAL_UPLIFT_THRESHOLD (default: 5000 THB)
        Minimum additional ERV that LEGAL must offer over AGENCY before legal
        is selected. Set higher to reduce legal queue volume; set lower to
        escalate more aggressively.

DISCOUNT_GRID
    Discount levels evaluated for each account (as fractions, e.g. 0.20 = 20%).
    Default: 0%, 10%, 20%, 30%, 40%, 50%, 60%.
    Upper bound is capped by ConstraintAgent at segment-specific max and BOT 60% cap.
    Add finer steps (e.g. 0.05 increments) for more precise ERV optimisation
    at the cost of compute time.

ELASTICITY_ALPHA (default: 2.5)
    Controls how strongly discount increases acceptance probability.
    Calibrate per SIGNAL_SEGMENT using historical settlement data:
        Segment A: ~2.0 (strong signal, moderate sensitivity)
        Segment B: ~2.5
        Segment C: ~3.0 (less predictable, more responsive to discount)
        Segment D: ~3.5 (high discount needed to generate any response)
    Override via action_alphas dict in ModelAgent constructor.
──────────────────────────────────────────────────────────────────────────────
"""

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from agents.base_agent import AgentBlockedException, BaseAgent

DEFAULT_ELASTICITY_MODEL_DIR = Path(
    os.environ.get("ELASTICITY_MODEL_DIR", "models/elasticity_model")
)

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

# Discount levels evaluated per account. See CONFIGURATION INSTRUCTIONS above.
DISCOUNT_GRID = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60]

# Default action costs in THB. Override via ModelAgent constructor.
# See CONFIGURATION INSTRUCTIONS above.
DEFAULT_ACTION_COSTS = {
    "DIGITAL_NUDGE": 10,
    "AGENT_CALL":    150,
    "AGENCY":        500,
    "LEGAL":         2_000,
    "HOLD":          0,
}

# Default elasticity alpha per signal_segment.
# Higher alpha = more sensitive to discount. See CONFIGURATION INSTRUCTIONS.
DEFAULT_ELASTICITY_ALPHA = {
    "A": 2.0,
    "B": 2.5,
    "C": 3.0,
    "D": 3.5,
}

# Minimum ERV uplift (THB) that LEGAL must offer over AGENCY before legal is chosen.
DEFAULT_LEGAL_UPLIFT_THRESHOLD = 5_000.0

# Actions eligible for discount optimisation (others use fixed d=0)
DISCOUNTABLE_ACTIONS = {"DIGITAL_NUDGE", "AGENT_CALL", "AGENCY"}

# Channel mapping from best action to contact channel label
ACTION_TO_CHANNEL = {
    "DIGITAL_NUDGE": "DIGITAL_ONLY",
    "AGENT_CALL":    "AGENT_CALL",
    "AGENCY":        "AGENCY_REFERRAL",
    "LEGAL":         "LEGAL_QUEUE",
    "HOLD":          "HOLD",
}

# ── SIGNAL_SEGMENT strategy differentiation ──────────────────────────────────
# Each segment has a distinct objective that modifies how ERV is optimised.
# This is the operational expression of the segment — not just a feature.
#
# A (CardX + Bureau): Maximise recovery AMOUNT — customer has high signal quality,
#    keep discount low, extract maximum value. Penalise high discounts.
#
# B (CardX only): Balanced — moderate discount acceptable if it raises P(pay).
#    Standard ERV optimisation with no penalty.
#
# C (Bureau only): Reactivation — customer is known to bureau but disengaged
#    from CardX. Accept higher discount to re-establish contact. Reward contact.
#
# D (No signal): Exploration — no reliable signal. Use low-cost actions only.
#    Cap at DIGITAL_NUDGE or HOLD. High-cost actions (AGENT_CALL, LEGAL) suppressed.
#
# discount_penalty: multiplied against gross ERV to penalise high discounts.
#   1.0 = no penalty, 0.7 = 30% ERV haircut per unit of discount offered.
# allowed_actions: restricts which actions are evaluated for this segment.

SEGMENT_STRATEGY = {
    "A": {
        "objective":        "maximise_amount",
        "discount_penalty": 0.6,   # penalise high discounts — preserve recovery amount
        "allowed_actions":  {"DIGITAL_NUDGE", "AGENT_CALL", "HOLD"},
        "description":      "Optimise recovery amount — minimal discount",
    },
    "B": {
        "objective":        "balanced",
        "discount_penalty": 1.0,   # no penalty — standard ERV
        "allowed_actions":  {"DIGITAL_NUDGE", "AGENT_CALL", "AGENCY", "HOLD"},
        "description":      "Balanced ERV optimisation",
    },
    "C": {
        "objective":        "reactivation",
        "discount_penalty": 1.3,   # reward higher discounts — reactivation is the goal
        "allowed_actions":  {"DIGITAL_NUDGE", "AGENT_CALL", "AGENCY", "HOLD"},
        "description":      "Reactivation focus — accept higher discount to re-engage",
    },
    "D": {
        "objective":        "exploration",
        "discount_penalty": 1.0,
        "allowed_actions":  {"DIGITAL_NUDGE", "HOLD"},  # low-cost only
        "description":      "Exploration — low-cost actions, no agency/legal",
    },
}


class ModelAgent(BaseAgent):

    def __init__(
        self,
        execution_date: str,
        memory,
        dry_run: bool = False,
        action_costs: dict = None,
        action_alphas: dict = None,
        legal_uplift_threshold: float = DEFAULT_LEGAL_UPLIFT_THRESHOLD,
    ):
        """
        Args:
            execution_date:         Scoring date (YYYY-MM-DD).
            memory:                 AgentMemory instance.
            dry_run:                If True, skip all Delta writes.
            action_costs:           Override default action costs (THB).
                                    See module docstring for defaults.
            action_alphas:          Override default elasticity alphas per
                                    signal_segment. See module docstring.
            legal_uplift_threshold: Min ERV(LEGAL) - ERV(AGENCY) before
                                    LEGAL is chosen over AGENCY.
                                    Default: 5000 THB.
        """
        super().__init__(
            agent_name="model_agent",
            execution_date=execution_date,
            memory=memory,
            dry_run=dry_run,
        )
        self.action_costs           = {**DEFAULT_ACTION_COSTS, **(action_costs or {})}
        self.action_alphas          = {**DEFAULT_ELASTICITY_ALPHA, **(action_alphas or {})}
        self.legal_uplift_threshold = legal_uplift_threshold

        # Load elasticity model if available — replaces hardcoded alpha curves
        self._elasticity_curves: dict = {}   # account_id → {d: p_accept}
        self._elasticity_model_dir = DEFAULT_ELASTICITY_MODEL_DIR
        self._try_load_elasticity_model()

    def _try_load_elasticity_model(self) -> None:
        """Load elasticity model if fitted. Falls back to alpha curves silently."""
        import importlib.util as _ilu
        import pathlib as _pl
        model_path = self._elasticity_model_dir / "elasticity_model.pkl"
        if model_path.exists():
            try:
                _spec = _ilu.spec_from_file_location(
                    "elasticity_model_trainer",
                    _pl.Path(__file__).parent.parent / "models" / "elasticity_model_trainer.py",
                )
                _mod = _ilu.module_from_spec(_spec)
                _spec.loader.exec_module(_mod)
                ElasticityModelTrainer = _mod.ElasticityModelTrainer
                self._elasticity_trainer = ElasticityModelTrainer(
                    model_dir=self._elasticity_model_dir
                )
                self.log("Elasticity model found — will use model-based P(accept|d)")
            except Exception as e:
                self._elasticity_trainer = None
                self.log(f"Could not load elasticity model: {e} — using alpha fallback", level="warning")
        else:
            self._elasticity_trainer = None
            self.log(
                f"No elasticity model at {model_path} — "
                "using hardcoded alpha curves. "
                "Run ElasticityModelTrainer.fit() to enable model-based elasticity.",
                level="warning",
            )

    def execute(self) -> dict:
        self.log("Reading feature_output")
        df = self._read_features()

        output = df.copy()

        # ── Pre-compute elasticity curves if model is available ───────────────
        # Uses propensity_30d (P_1M) as primary signal per design spec.
        if getattr(self, "_elasticity_trainer", None) is not None:
            try:
                curves_df = self._elasticity_trainer.predict_accept_curve(output)
                # Store as {account_id: {d: p_accept}} for fast row-level lookup
                d_cols = [c for c in curves_df.columns if c.startswith("p_accept_d")]
                for _, row in curves_df.iterrows():
                    aid = row.get("account_id")
                    if aid is not None:
                        self._elasticity_curves[aid] = {
                            float(col.replace("p_accept_d", "")) / 100: row[col]
                            for col in d_cols
                        }
                self.log(f"Elasticity curves computed for {len(self._elasticity_curves):,} accounts")
            except Exception as e:
                self.log(f"Elasticity model inference failed: {e} — using alpha fallback", level="warning")
                self._elasticity_curves = {}

        # ── Compute per-action ERV and select best action ─────────────────────
        erv_results = output.apply(self._select_best_action, axis=1)

        output["recommended_action"]  = erv_results.apply(lambda r: r["action"])
        output["recommended_channel"] = output["recommended_action"].map(ACTION_TO_CHANNEL)
        output["d_optimal"]           = erv_results.apply(lambda r: r["d_optimal"])
        output["erv_at_d_optimal"]    = erv_results.apply(lambda r: r["erv_net"])
        output["erv_gross"]           = erv_results.apply(lambda r: r["erv_gross"])
        output["action_cost"]         = erv_results.apply(lambda r: r["action_cost"])
        output["erv_by_action"]       = erv_results.apply(lambda r: str(r["erv_by_action"]))

        # ── Contact timing from recovery_tier ─────────────────────────────────
        output["contact_timing"] = output["recovery_tier"].map({
            "TIER_1_HIGH_RECOVERY":  "WITHIN_24H",
            "TIER_2_MEDIUM_RECOVERY":"WITHIN_72H",
            "TIER_3_LOW_RECOVERY":   "WITHIN_7D",
        }).fillna("WITHIN_7D")

        # ── Segment D always HOLD unless ERV is strong enough ─────────────────
        seg_d_mask = output["signal_segment"] == "D"
        output.loc[seg_d_mask, "contact_timing"] = "NO_CONTACT"

        # ── Flag accounts needing constraint check ─────────────────────────────
        output["needs_constraint_check"] = (
            (output["d_optimal"] > 0.45) |
            (output["erv_at_d_optimal"] > 50_000) |
            (output["signal_segment"] == "C")
        )

        if not self.dry_run:
            self.memory.write("model_agent_output", output)

        action_dist  = output["recommended_action"].value_counts().to_dict()
        channel_dist = output["recommended_channel"].value_counts().to_dict()
        self.log(
            f"Model selections ready | rows={len(output):,} | "
            f"actions={action_dist} | channels={channel_dist}"
        )
        return {
            "rows_written":           len(output),
            "action_distribution":    action_dist,
            "channel_distribution":   channel_dist,
            "needs_constraint_check": int(output["needs_constraint_check"].sum()),
        }

    # ── Per-action ERV computation ────────────────────────────────────────────

    def _select_best_action(self, row) -> dict:
        """
        Computes net ERV for every action across the discount grid.
        Returns the action + discount that maximises net ERV.

        Agency-first rule for charge-off accounts:
          LEGAL is only selected over AGENCY if
          ERV(LEGAL) > ERV(AGENCY) + legal_uplift_threshold.
        """
        # propensity_30d (P_1M) is the primary decisioning signal
        # propensity_180d (P_6M) used only for AGENCY/LEGAL (longer-horizon actions)
        p30          = row.get("propensity_30d",    0.0) or 0.0
        p180         = row.get("propensity_180d",   0.0) or 0.0
        balance      = row.get("erv_at_d_optimal",  0.0) or 0.0
        seg          = row.get("signal_segment",    "D")
        alpha        = self.action_alphas.get(seg, 2.5)
        aid          = row.get("account_id")
        strategy     = SEGMENT_STRATEGY.get(seg, SEGMENT_STRATEGY["B"])
        allowed      = strategy["allowed_actions"]
        disc_penalty = strategy["discount_penalty"]

        elast_curves = self._elasticity_curves.get(aid, {})
        erv_by_action = {}

        # ── HOLD — always evaluated as baseline ───────────────────────────────
        erv_by_action["HOLD"] = {
            "erv_net": 0.0, "erv_gross": 0.0,
            "d_optimal": 0.0, "action_cost": 0,
        }

        # ── DIGITAL_NUDGE — P_1M primary, low-cost ───────────────────────────
        if "DIGITAL_NUDGE" in allowed:
            erv_by_action["DIGITAL_NUDGE"] = self._best_discounted_erv(
                p_base=p30, balance=balance, alpha=alpha * 0.5,
                action="DIGITAL_NUDGE", elast_curves=elast_curves,
                discount_penalty=disc_penalty,
            )

        # ── AGENT_CALL — P_1M primary ─────────────────────────────────────────
        if "AGENT_CALL" in allowed:
            erv_by_action["AGENT_CALL"] = self._best_discounted_erv(
                p_base=p30, balance=balance, alpha=alpha,
                action="AGENT_CALL", elast_curves=elast_curves,
                discount_penalty=disc_penalty,
            )

        # ── AGENCY — P_6M, evaluated before LEGAL ────────────────────────────
        if "AGENCY" in allowed:
            erv_by_action["AGENCY"] = self._best_discounted_erv(
                p_base=p180 * 0.6, balance=balance, alpha=alpha * 1.2,
                action="AGENCY", elast_curves=elast_curves,
                discount_penalty=disc_penalty,
            )

        # ── LEGAL — P_6M, escalation only ────────────────────────────────────
        if "LEGAL" in allowed:
            erv_by_action["LEGAL"] = self._best_discounted_erv(
                p_base=p180 * 0.4, balance=balance, alpha=0.5,
                action="LEGAL", elast_curves=elast_curves,
                discount_penalty=disc_penalty,
            )

        # ── Agency-first: LEGAL must beat AGENCY by threshold ─────────────────
        if "AGENCY" in erv_by_action and "LEGAL" in erv_by_action:
            if erv_by_action["LEGAL"]["erv_net"] <= \
               erv_by_action["AGENCY"]["erv_net"] + self.legal_uplift_threshold:
                erv_by_action["LEGAL"]["erv_net"] = -1.0

        # ── Select action with highest net ERV ────────────────────────────────
        best_action = max(erv_by_action, key=lambda a: erv_by_action[a]["erv_net"])
        best        = erv_by_action[best_action]

        return {
            "action":       best_action,
            "d_optimal":    best["d_optimal"],
            "erv_net":      best["erv_net"],
            "erv_gross":    best["erv_gross"],
            "action_cost":  best["action_cost"],
            "erv_by_action": {
                a: round(erv_by_action[a]["erv_net"], 2)
                for a in erv_by_action
            },
        }

    def _best_discounted_erv(
        self,
        p_base: float,
        balance: float,
        alpha: float,
        action: str,
        elast_curves: dict = None,
        discount_penalty: float = 1.0,
    ) -> dict:
        """
        Finds d in DISCOUNT_GRID that maximises net ERV for a given action.

        If elasticity model curves are provided (from ElasticityModelTrainer),
        uses model P(accept|d) directly.
        Otherwise falls back to parametric alpha curve:
            P(pay | d) = 1 − (1 − p_base) × (1 − d)^alpha

        ERV(d) = P(accept|d) × p_base × balance × (1−d) − cost(action)
        """
        cost          = self.action_costs.get(action, 0)
        best_erv_net  = -cost
        best_d        = 0.0
        best_erv_gross = 0.0

        for d in DISCOUNT_GRID:
            if elast_curves and d in elast_curves:
                # Model-based P(accept | d) — preferred
                p_accept = float(elast_curves[d])
                p_adj    = p_accept * p_base
            else:
                # Parametric alpha fallback
                p_adj = 1.0 - (1.0 - p_base) * ((1.0 - d) ** alpha)

            # discount_penalty modifies how discount is valued per segment strategy
            # Segment A: penalty < 1 → penalise high discounts (preserve amount)
            # Segment C: penalty > 1 → reward higher discounts (reactivation)
            erv_gross = p_adj * balance * (1.0 - d * discount_penalty)
            erv_net   = erv_gross - cost
            if erv_net > best_erv_net:
                best_erv_net   = erv_net
                best_d         = d
                best_erv_gross = erv_gross

        return {
            "erv_net":    round(best_erv_net,   2),
            "erv_gross":  round(best_erv_gross, 2),
            "d_optimal":  best_d,
            "action_cost": cost,
        }

    # ── Reader ────────────────────────────────────────────────────────────────

    def _read_features(self) -> pd.DataFrame:
        df = self.memory.read("feature_output")
        if hasattr(df, "toPandas"):
            df = df.toPandas()
        if df.empty:
            raise AgentBlockedException(
                "feature_output is empty — FeatureAgent may have failed.",
                retry_agent="feature_agent",
            )
        return df
