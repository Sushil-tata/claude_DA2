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

AMOUNT MODEL (E(recovery_amount))
    Two fitted AmountModelTrainer instances are loaded at startup:
        amount_model_30d.pkl  — E(amount) for DIGITAL_NUDGE and AGENT_CALL
                                Aligned with P_1M (propensity_30d) horizon.
        amount_model_180d.pkl — E(amount) for AGENCY and LEGAL
                                Aligned with P_6M (propensity_180d) horizon.

    When models are not yet fitted, ERV falls back to:
        outstanding_balance → total_outstanding → 0

    To activate model-based E(amount):
        Run AmountModelTrainer(outcome_window_days=30).fit(recovered_df)
        Run AmountModelTrainer(outcome_window_days=180).fit(recovered_df)
        Set AMOUNT_MODEL_DIR env variable to the path where models are saved.

    In Databricks:
        AMOUNT_MODEL_DIR = /dbfs/FileStore/models/amount_model/
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
DEFAULT_AMOUNT_MODEL_DIR = Path(
    os.environ.get("AMOUNT_MODEL_DIR", "models/amount_model")
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

# ── CLUSTER STRATEGY (primary strategic axis) ─────────────────────────────────
# Keyed by behavioural_persona (cluster label from PersonaClusterTrainer).
# Cluster label reflects the structural behavioural identity derived from
# 12–24M pre-CO trajectory and bureau features — NOT short-term propensity.
#
# Three-layer resolution in _select_best_action:
#   Layer 1: behavioural_persona → CLUSTER_STRATEGY (ERV horizon, action set, discount cap)
#   Layer 2: signal_segment      → SEGMENT_CONFIDENCE_MODIFIER (discount penalty, cost tolerance)
#   Layer 3: Dormant+LIMITED     → immediate HOLD override (no ERV, no spend)
#
# erv_horizon_days: which propensity model to use.
#   90d  → propensity_90d  (Sudden-Shock: responds quickly to right intervention)
#   180d → propensity_180d (Chronic/Structural: longer time needed)
#   360d → propensity_180d as proxy (Dormant: 360d model not yet built)
#
# Uplift note (Amendment 1): before any action, check uplift_score > threshold.
#   High propensity + low uplift → SUPPRESS (account will self-cure).
#   Low propensity + high uplift → PRIORITISE despite low propensity score.
#   The CLUSTER_STRATEGY here governs action type; uplift governs whether to act at all.

CLUSTER_STRATEGY = {
    "Sudden-Shock Distressed": {
        "objective":        "maximise_recovery_speed",
        "erv_horizon_days": 90,
        "discount_penalty": 0.7,    # was paying — don't over-discount, they will settle
        "discount_max":     0.35,   # cap at 35% — preserve value
        "allowed_actions":  {"AGENT_CALL", "AGENCY", "DIGITAL_NUDGE", "HOLD"},
        "agency_eligible":  True,
        "legal_eligible":   False,
        "description":      "Sudden onset — firm settlement, agency month 1–3",
    },
    "High-Engagement Chronic": {
        "objective":        "patient_structured_settlement",
        "erv_horizon_days": 180,
        "discount_penalty": 1.0,    # standard — needs time, not just discount
        "discount_max":     0.45,
        "allowed_actions":  {"AGENT_CALL", "AGENCY", "DIGITAL_NUDGE", "HOLD"},
        "agency_eligible":  True,
        "legal_eligible":   False,
        "description":      "Chronic but engaged — instalment plan preferred, 180d horizon",
    },
    "Structural Defaulter": {
        "objective":        "legal_posturing_or_targeted_settlement",
        "erv_horizon_days": 180,
        "discount_penalty": 0.8,    # bureau capacity intact — can pay, needs pressure
        "discount_max":     0.40,
        "allowed_actions":  {"AGENT_CALL", "AGENCY", "DIGITAL_NUDGE", "HOLD", "LEGAL"},
        "agency_eligible":  True,
        "legal_eligible":   True,   # paying other lenders = legal posturing viable
        "description":      "Bureau capacity intact — legal threat credible, firm offer",
    },
    "Dormant": {
        "objective":        "minimal_spend_portfolio_resolution",
        "erv_horizon_days": 360,
        "discount_penalty": 1.3,    # deep discount needed to unlock any recovery
        "discount_max":     0.60,   # BOT cap
        "allowed_actions":  {"DIGITAL_NUDGE", "HOLD"},
        "agency_eligible":  False,  # agency fees uneconomic at this recovery probability
        "legal_eligible":   False,
        "description":      "Dormant — digital only, flag for write-off or portfolio sale",
    },
}

# Default fallback if persona is missing or unrecognised
_CLUSTER_STRATEGY_DEFAULT = CLUSTER_STRATEGY["High-Engagement Chronic"]

# ── SIGNAL CONFIDENCE MODIFIER (Phase 1: binary — Amendment 4) ───────────────
# Applied ON TOP of CLUSTER_STRATEGY as a discount penalty multiplier.
# Reflects model confidence, not strategic intent.
#
# FULL_SIGNAL:    CardX + Bureau available. Trust model output; preserve value.
# LIMITED_SIGNAL: At least one source missing. Accept higher discount uncertainty.
#                 Do NOT escalate to expensive actions (agency, legal) for Dormant.

SEGMENT_CONFIDENCE_MODIFIER = {
    "FULL_SIGNAL": {
        "discount_penalty_multiplier": 0.85,  # rich signal — model reliable, less discount
        "cost_tolerance":              "high",  # expensive actions justified
        "low_confidence_flag":         False,
        "description":                 "Full signal — preserve value, model reliable",
    },
    "LIMITED_SIGNAL": {
        "discount_penalty_multiplier": 1.20,  # uncertain — accept more discount
        "cost_tolerance":              "low",  # avoid expensive actions for borderline accounts
        "low_confidence_flag":         True,
        "description":                 "Limited signal — higher discount tolerance, low-cost bias",
    },
}

_SEGMENT_MODIFIER_DEFAULT = SEGMENT_CONFIDENCE_MODIFIER["LIMITED_SIGNAL"]

# ── ELASTICITY MODEL CONFOUNDING NOTE (Amendment 2) ──────────────────────────
# Historical discount data is NOT randomly assigned — agents give higher discounts
# to higher-risk accounts. This creates spurious negative correlation between
# discount level and recovery rate.
#
# Phase 1 mitigations:
#   - Include prior_discount_max, signal_segment, balance_bucket, behavioural_segment
#     as control variables in the elasticity model (partial de-confounding)
#   - Treat elasticity output as rank ordering of acceptance probability only
#   - DO NOT interpret discount coefficient as causal effect of discount on acceptance
#   - Document this limitation in the model card
#
# Phase 2: controlled discount randomisation or instrumental variable approach.


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
        self._elasticity_curves: dict = {}
        self._elasticity_model_dir = DEFAULT_ELASTICITY_MODEL_DIR
        self._try_load_elasticity_model()

        # Load uplift (T-learner) models if available — enables causal ERV
        # causal ERV = τ(x) × E(amount) × (1−d) − cost(action)
        # Falls back to predictive ERV when no uplift model exists
        self._uplift_scores: dict = {}   # account_id → τ(x)
        self._uplift_model_dir = Path(
            os.environ.get("UPLIFT_MODEL_DIR", "models/uplift_models")
        )
        self._try_load_uplift_model()

        # Load amount models — E(recovery_amount | recovery occurred)
        # Two windows: 30d aligned with P_1M (tactical), 180d aligned with P_6M (agency/legal)
        # Replaces using outstanding_balance directly as the ERV "balance" term.
        # Falls back to outstanding_balance → total_outstanding → 0 if not fitted.
        self._amount_model_dir     = DEFAULT_AMOUNT_MODEL_DIR
        self._amount_trainer_30d   = None   # for DIGITAL_NUDGE, AGENT_CALL
        self._amount_trainer_180d  = None   # for AGENCY, LEGAL
        self._try_load_amount_models()

        # Per-account predicted amounts — populated in execute() before ERV loop
        # {account_id: {"amount_30d": float, "amount_180d": float}}
        self._predicted_amounts: dict = {}

        # Treatment logger — appends every action decision to treatment_log
        # This is the data collection foundation for Phase 2 uplift training
        self._treatment_logger = None  # initialised in execute() after memory is ready

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

    def _try_load_uplift_model(self) -> None:
        """Load T-learner uplift models if fitted. Falls back to predictive ERV silently."""
        import importlib.util as _ilu
        import pathlib as _pl
        any_model = any(
            (self._uplift_model_dir / f"{seg}_treated.pkl").exists()
            for seg in ["A", "B", "C", "D"]
        )
        if any_model:
            try:
                _spec = _ilu.spec_from_file_location(
                    "uplift_model_trainer",
                    _pl.Path(__file__).parent.parent / "models" / "uplift_model_trainer.py",
                )
                _mod = _ilu.module_from_spec(_spec)
                _spec.loader.exec_module(_mod)
                self._uplift_trainer = _mod.UpliftModelTrainer(
                    model_dir=self._uplift_model_dir
                )
                self.log("Uplift (T-learner) models found — causal ERV active")
            except Exception as e:
                self._uplift_trainer = None
                self.log(f"Could not load uplift models: {e} — using predictive ERV", level="warning")
        else:
            self._uplift_trainer = None
            self.log(
                f"No uplift models at {self._uplift_model_dir} — "
                "using predictive ERV (Phase 1 mode). "
                "Causal ERV activates automatically once UpliftModelTrainer.fit() "
                "has been run on champion/challenger data.",
                level="warning",
            )

    def _try_load_amount_models(self) -> None:
        """
        Load AmountModelTrainer for 30d and 180d windows if fitted models exist.

        30d model: used as E(amount) for DIGITAL_NUDGE and AGENT_CALL (tactical).
        180d model: used as E(amount) for AGENCY and LEGAL (strategic).

        Falls back silently to outstanding_balance column if models are not yet fitted.
        Run AmountModelTrainer(outcome_window_days=30).fit() and
        AmountModelTrainer(outcome_window_days=180).fit() on recovered accounts to activate.
        """
        import importlib.util as _ilu
        import pathlib as _pl

        _spec = _ilu.spec_from_file_location(
            "amount_model_trainer",
            _pl.Path(__file__).parent.parent / "models" / "amount_model_trainer.py",
        )
        _mod = _ilu.module_from_spec(_spec)

        for window, attr in [(30, "_amount_trainer_30d"), (180, "_amount_trainer_180d")]:
            model_path = self._amount_model_dir / f"amount_model_{window}d.pkl"
            if model_path.exists():
                try:
                    _spec.loader.exec_module(_mod)
                    trainer = _mod.AmountModelTrainer(
                        outcome_window_days=window,
                        model_dir=self._amount_model_dir,
                    )
                    setattr(self, attr, trainer)
                    self.log(f"Amount model {window}d loaded — E(amount) will use model predictions")
                except Exception as e:
                    self.log(
                        f"Could not load amount model {window}d: {e} — "
                        "using outstanding_balance fallback",
                        level="warning",
                    )
            else:
                self.log(
                    f"No amount model at {model_path} — "
                    f"using outstanding_balance as E(amount) proxy for {window}d window. "
                    f"Run AmountModelTrainer(outcome_window_days={window}).fit() to activate.",
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

        # ── Pre-compute uplift scores τ(x) per account ───────────────────────
        # When T-learner models are available:
        #   τ(x) = P(pay|treated,x) − P(pay|control,x)
        #   causal ERV = τ(x) × E(amount) × (1−d) − cost(action)
        #   Accounts with τ(x) ≤ 0 receive HOLD — intervening would not help.
        # When models are absent: τ(x) defaults to 1.0 (predictive ERV mode).
        causal_mode = False
        if getattr(self, "_uplift_trainer", None) is not None:
            try:
                tau_series = self._uplift_trainer.predict_uplift(output)
                self._uplift_scores = dict(zip(output["account_id"], tau_series))
                pct_positive = (tau_series > 0).mean()
                self.log(
                    f"Causal ERV active | τ(x) computed for {len(tau_series):,} accounts | "
                    f"positive uplift={pct_positive:.1%} | "
                    f"negative uplift (HOLD forced)={(1-pct_positive):.1%}"
                )
                causal_mode = True
            except Exception as e:
                self.log(f"Uplift inference failed: {e} — falling back to predictive ERV", level="warning")
                self._uplift_scores = {}

        # ── Pre-compute E(recovery_amount) per account ────────────────────────
        # Two windows:
        #   amount_30d  → used for DIGITAL_NUDGE and AGENT_CALL (tactical, P_1M)
        #   amount_180d → used for AGENCY and LEGAL (strategic, P_6M)
        #
        # When fitted models exist: model predictions are used.
        # Fallback priority: outstanding_balance → total_outstanding → 0.
        #
        # Models replace the earlier approach of using erv_at_d_optimal from
        # feature_output as the balance term — that column is the agent's own
        # output and is not available or meaningful as an input.
        self._predicted_amounts = {}
        if "outstanding_balance" in output.columns:
            _balance_fallback = output["outstanding_balance"].fillna(0.0)
        elif "total_outstanding" in output.columns:
            _balance_fallback = output["total_outstanding"].fillna(0.0)
        else:
            _balance_fallback = pd.Series(0.0, index=output.index)

        _aid_col = output["account_id"] if "account_id" in output.columns else output.index
        fallback_amounts = dict(zip(_aid_col, _balance_fallback))

        for window, attr in [(30, "_amount_trainer_30d"), (180, "_amount_trainer_180d")]:
            trainer = getattr(self, attr, None)
            key = f"amount_{window}d"
            if trainer is not None:
                try:
                    preds = trainer.predict(output)
                    for aid, val in zip(_aid_col, preds):
                        self._predicted_amounts.setdefault(aid, {})[key] = float(val)
                    self.log(
                        f"Amount model {window}d: E(amount) computed for "
                        f"{len(preds):,} accounts | "
                        f"mean={preds.mean():,.0f} THB"
                    )
                except Exception as e:
                    self.log(
                        f"Amount model {window}d inference failed: {e} — "
                        "using outstanding_balance fallback",
                        level="warning",
                    )
                    for aid, val in fallback_amounts.items():
                        self._predicted_amounts.setdefault(aid, {})[key] = val
            else:
                for aid, val in fallback_amounts.items():
                    self._predicted_amounts.setdefault(aid, {})[key] = val

        # ── Compute per-action ERV and select best action ─────────────────────
        erv_results = output.apply(self._select_best_action, axis=1)

        output["recommended_action"]  = erv_results.apply(lambda r: r["action"])
        output["recommended_channel"] = output["recommended_action"].map(ACTION_TO_CHANNEL)
        output["d_optimal"]           = erv_results.apply(lambda r: r["d_optimal"])
        output["erv_at_d_optimal"]    = erv_results.apply(lambda r: r["erv_net"])
        output["erv_gross"]           = erv_results.apply(lambda r: r["erv_gross"])
        output["action_cost"]         = erv_results.apply(lambda r: r["action_cost"])
        output["erv_by_action"]       = erv_results.apply(lambda r: str(r["erv_by_action"]))
        output["tau_uplift"]          = erv_results.apply(lambda r: r.get("tau", 1.0))
        output["erv_mode"]            = "causal" if causal_mode else "predictive"

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

            # ── Log all treatment decisions to treatment_log ──────────────────
            # This is the data collection foundation for uplift model training.
            # treatment_log is append-only — every run adds rows.
            # Outcomes (paid_180d) are joined back 30/90/180 days later.
            try:
                import importlib.util as _ilu
                import pathlib as _pl
                _spec = _ilu.spec_from_file_location(
                    "treatment_logger",
                    _pl.Path(__file__).parent / "treatment_logger.py",
                )
                _mod = _ilu.module_from_spec(_spec)
                _spec.loader.exec_module(_mod)
                logger_obj = _mod.TreatmentLogger(
                    memory=self.memory,
                    execution_date=self.execution_date,
                )
                rows_logged = logger_obj.log_treatments(output)
                self.log(f"Treatment log: {rows_logged:,} rows appended to recovery.treatment_log")
            except Exception as e:
                self.log(f"Treatment logging failed (non-blocking): {e}", level="warning")

        action_dist  = output["recommended_action"].value_counts().to_dict()
        channel_dist = output["recommended_channel"].value_counts().to_dict()
        neg_uplift   = int((output["tau_uplift"] <= 0).sum()) if causal_mode else 0
        self.log(
            f"Model selections ready | rows={len(output):,} | "
            f"mode={output['erv_mode'].iloc[0]} | "
            f"actions={action_dist} | negative_uplift_held={neg_uplift}"
        )
        return {
            "rows_written":           len(output),
            "erv_mode":               output["erv_mode"].iloc[0],
            "action_distribution":    action_dist,
            "channel_distribution":   channel_dist,
            "needs_constraint_check": int(output["needs_constraint_check"].sum()),
            "negative_uplift_held":   neg_uplift,
        }

    # ── Per-action ERV computation ────────────────────────────────────────────

    def _select_best_action(self, row) -> dict:
        """
        Computes net ERV for every action across the discount grid.
        Returns the action + discount that maximises net ERV.

        E(amount) is sourced from AmountModelTrainer predictions (pre-computed
        in execute()) rather than outstanding balance directly:
          - amount_30d → DIGITAL_NUDGE, AGENT_CALL (aligned with P_1M / 30d horizon)
          - amount_180d → AGENCY, LEGAL            (aligned with P_6M / 180d horizon)

        Strategy = SEGMENT_STRATEGY[signal_segment] × PERSONA_STRATEGY_MODIFIER[persona]:
          - Segment defines objective (maximise_amount / balanced / reactivation / exploration)
          - Persona refines discount tolerance and action eligibility
          - Example: Segment A + Stressed → tighter discount than A alone (preserve value
            for willing customers) but AGENCY unlocked for restructuring
          - Example: Segment B + Disconnected → AGENT_CALL/AGENCY removed (low-cost only)

        Agency-first rule for charge-off accounts:
          LEGAL is only selected over AGENCY if
          ERV(LEGAL) > ERV(AGENCY) + legal_uplift_threshold.
        """
        # ── Three-layer strategy resolution ───────────────────────────────────
        # Layer 1 (primary):   behavioural_persona → CLUSTER_STRATEGY
        #                       ERV horizon, action set, discount cap
        # Layer 2 (modifier):  signal_segment → SEGMENT_CONFIDENCE_MODIFIER
        #                       discount penalty multiplier, cost tolerance
        # Layer 3 (override):  Dormant + LIMITED_SIGNAL → immediate HOLD
        #                       (no uplift expected, no spend justified)

        persona = row.get("behavioural_persona", "High-Engagement Chronic")
        seg     = row.get("signal_segment",      "LIMITED_SIGNAL")
        aid     = row.get("account_id")
        alpha   = self.action_alphas.get(seg, 2.5)

        # Layer 1: cluster determines strategic objective + action envelope
        cluster_strat = CLUSTER_STRATEGY.get(persona, _CLUSTER_STRATEGY_DEFAULT)
        erv_horizon   = cluster_strat["erv_horizon_days"]
        disc_max      = cluster_strat["discount_max"]

        # Layer 2: signal segment modifies discount penalty
        seg_mod      = SEGMENT_CONFIDENCE_MODIFIER.get(seg, _SEGMENT_MODIFIER_DEFAULT)
        disc_penalty = cluster_strat["discount_penalty"] * seg_mod["discount_penalty_multiplier"]

        # Layer 3: Dormant + LIMITED_SIGNAL = no ERV, immediate HOLD
        # (low-cost digital only for Dormant + FULL_SIGNAL is handled via allowed_actions)
        if persona == "Dormant" and seg == "LIMITED_SIGNAL":
            return {
                "action": "HOLD", "d_optimal": 0.0,
                "erv_net": 0.0, "erv_gross": 0.0,
                "action_cost": 0, "tau": 1.0,
                "erv_by_action": {"HOLD": 0.0},
                "erv_horizon_days": erv_horizon,
                "behavioural_persona": persona,
                "signal_segment": seg,
            }

        # Allowed actions: cluster is authoritative
        # LIMITED_SIGNAL further restricts expensive actions for low-confidence accounts
        allowed = set(cluster_strat["allowed_actions"])
        if seg_mod["cost_tolerance"] == "low":
            allowed -= {"AGENCY", "LEGAL"}   # avoid expensive actions without full signal

        # Select propensity aligned to ERV horizon
        if erv_horizon and erv_horizon <= 90:
            p_primary = row.get("propensity_90d", row.get("propensity_30d", 0.0)) or 0.0
        else:
            p_primary = row.get("propensity_180d", 0.0) or 0.0   # 180d for 180d/360d horizons
        p30  = row.get("propensity_30d",  0.0) or 0.0
        p180 = row.get("propensity_180d", 0.0) or 0.0

        # E(recovery_amount) per horizon — from AmountModelTrainer or fallback
        amounts      = self._predicted_amounts.get(aid, {})
        amount_30d   = amounts.get("amount_30d",  0.0) or 0.0
        amount_180d  = amounts.get("amount_180d", 0.0) or 0.0

        elast_curves = self._elasticity_curves.get(aid, {})

        # τ(x) = causal uplift score from T-learner
        # τ = 1.0 means predictive mode (no causal model yet)
        # τ ≤ 0 means intervening causes no incremental benefit — force HOLD
        tau = self._uplift_scores.get(aid, 1.0)
        if tau <= 0:
            return {
                "action": "HOLD", "d_optimal": 0.0,
                "erv_net": 0.0, "erv_gross": 0.0,
                "action_cost": 0, "tau": tau,
                "erv_by_action": {"HOLD": 0.0},
            }

        erv_by_action = {}

        # ── HOLD — always evaluated as baseline ───────────────────────────────
        erv_by_action["HOLD"] = {
            "erv_net": 0.0, "erv_gross": 0.0,
            "d_optimal": 0.0, "action_cost": 0,
        }

        # ── Action ERV computation aligned to recoverability horizon ─────────
        # p_primary is aligned to the ERV horizon for this tier:
        #   TIER_1 (90d)  → propensity_90d
        #   TIER_2 (180d) → propensity_180d
        #   TIER_3 (360d) → propensity_180d (proxy; 360d model not yet available)
        # Using the tier-aligned propensity prevents TIER_3 from appearing
        # profitable at 30d when 360d recovery probability is actually very low.

        # ── DIGITAL_NUDGE — tier-aligned propensity ───────────────────────────
        if "DIGITAL_NUDGE" in allowed:
            erv_by_action["DIGITAL_NUDGE"] = self._best_discounted_erv(
                p_base=p_primary * 0.8, balance=amount_30d, alpha=alpha * 0.5,
                action="DIGITAL_NUDGE", elast_curves=elast_curves,
                discount_penalty=disc_penalty, discount_cap=disc_max,
            )

        # ── AGENT_CALL — tier-aligned propensity ─────────────────────────────
        if "AGENT_CALL" in allowed:
            erv_by_action["AGENT_CALL"] = self._best_discounted_erv(
                p_base=p_primary, balance=amount_30d, alpha=alpha,
                action="AGENT_CALL", elast_curves=elast_curves,
                discount_penalty=disc_penalty, discount_cap=disc_max,
            )

        # ── AGENCY — P_6M + amount_180d ───────────────────────────────────────
        if "AGENCY" in allowed:
            erv_by_action["AGENCY"] = self._best_discounted_erv(
                p_base=p180 * 0.6, balance=amount_180d, alpha=alpha * 1.2,
                action="AGENCY", elast_curves=elast_curves,
                discount_penalty=disc_penalty, discount_cap=disc_max,
            )

        # ── LEGAL — P_6M + amount_180d, escalation only ──────────────────────
        # Legal eligible only when cluster_strat marks it and signal is sufficient
        if "LEGAL" in allowed and cluster_strat.get("legal_eligible", False):
            erv_by_action["LEGAL"] = self._best_discounted_erv(
                p_base=p180 * 0.4, balance=amount_180d, alpha=0.5,
                action="LEGAL", elast_curves=elast_curves,
                discount_penalty=disc_penalty, discount_cap=disc_max,
            )

        # ── Agency-first: LEGAL must beat AGENCY by threshold ─────────────────
        if "AGENCY" in erv_by_action and "LEGAL" in erv_by_action:
            if erv_by_action["LEGAL"]["erv_net"] <= \
               erv_by_action["AGENCY"]["erv_net"] + self.legal_uplift_threshold:
                erv_by_action["LEGAL"]["erv_net"] = -1.0

        # ── Apply causal scaling: ERV × τ(x) when T-learner is active ──────────
        # τ < 1.0 discounts ERV for accounts with low incremental response
        # τ = 1.0 in predictive mode (no scaling applied)
        if tau != 1.0:
            for a in erv_by_action:
                if erv_by_action[a]["erv_net"] > 0:
                    erv_by_action[a]["erv_net"] = round(
                        erv_by_action[a]["erv_net"] * tau, 2
                    )

        # ── Select action with highest net causal ERV ─────────────────────────
        best_action = max(erv_by_action, key=lambda a: erv_by_action[a]["erv_net"])
        best        = erv_by_action[best_action]

        return {
            "action":               best_action,
            "d_optimal":            best["d_optimal"],
            "erv_net":              best["erv_net"],
            "erv_gross":            best["erv_gross"],
            "action_cost":          best["action_cost"],
            "tau":                  tau,
            "erv_horizon_days":    erv_horizon,
            "behavioural_persona": persona,
            "signal_segment":      seg,
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
        discount_cap: float = 0.60,
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
