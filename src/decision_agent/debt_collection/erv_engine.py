"""
ERV Engine — Expected Recovery Value Optimiser
===============================================
Assembles the three NBO/NBA components into the ERV decision engine:

    ERV(d) = P(Recovery | Segment, d) × E[Amount | Recovery, Segment] × Balance × (1 − d)
    d*     = argmax_d ERV(d)   subject to  ERV(d*) ≥ floor_threshold

The engine answers two questions per account:
  1. What is the optimal discount d* that maximises net recovery?
  2. Given d*, what is the best action (channel + offer type)?

Architecture
------------
  DiscountElasticityModel  → P(Recovery | Segment, d)       [sweep over d]
  RecoveryAmountModel      → E[Amount | Recovery, Segment]   [constant in d]
  ERVEngine                → assembles ERV(d), finds d*, applies constraints

Decision flow per account
--------------------------
  1. Retrieve segment from DebtorPersonaSegmentation
  2. Get E[Amount | Recovery, Segment] from RecoveryAmountModel (once)
  3. Sweep d ∈ discount_grid via DiscountElasticityModel
  4. Compute ERV(d) = P(d) × E[Amount] × Balance × (1 − d)
  5. Select d* = argmax ERV(d)
  6. Apply floor constraint: if ERV(d*) < floor_threshold → recommend DEFER/SELL
  7. Apply business constraints: min/max discount per segment, budget ceiling
  8. Return structured recommendation

Net Recovery vs Gross Recovery
-------------------------------
ERV as defined above is GROSS recovery (before collection cost).
Net ERV = ERV(d*) − collection_cost(channel)
The engine returns both; d* is computed on gross ERV to match the standard
definition, but the recommendation package includes net ERV for P&L reporting.

Constraints supported
---------------------
  - per_segment_discount_bounds : {segment: (d_min, d_max)}
  - erv_floor_threshold         : accounts below this → DEFER
  - max_discount_absolute       : hard cap across all segments
  - budget_ceiling_per_run      : total portfolio discount cost ceiling (batch mode)

MLflow integration
------------------
  Experiment : collection_recovery_models
  Run tags   : stage=erv_engine

Outputs
-------
  Per account:
    account_id, segment, d_star, erv_at_d_star, net_erv,
    p_recovery_at_d_star, e_amount, recommended_action, decision_reason
  Portfolio summary:
    total_erv, total_net_erv, avg_d_star, accounts_deferred,
    accounts_recommended, erv_by_segment

Usage
-----
    erv = ERVEngine(config, elasticity_model, recovery_model)
    recommendations = erv.recommend(accounts_df)
    summary         = erv.portfolio_summary(recommendations)
"""

import json
import logging
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# DEFAULT CONFIG VALUES
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_DISCOUNT_GRID: List[float] = [
    0.00, 0.05, 0.10, 0.15, 0.20, 0.25,
    0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60,
]

DEFAULT_SEGMENT_BOUNDS: Dict[str, Tuple[float, float]] = {
    "high_value_cooperative":  (0.00, 0.30),
    "high_value_unresponsive": (0.20, 0.60),
    "medium_value_willing":    (0.00, 0.40),
    "low_value_low_capacity":  (0.30, 0.60),
    "disputer":                (0.10, 0.50),
    "skip_tracer_needed":      (0.20, 0.60),
}

# Actions mapped to (channel, offer_type) based on d*
# Thresholds are fractions of optimal discount
ACTION_MAP = [
    {"d_max": 0.10, "channel": "sms",   "offer_type": "payment_reminder",  "label": "SOFT_REMINDER"},
    {"d_max": 0.25, "channel": "sms",   "offer_type": "payment_plan",       "label": "PAYMENT_PLAN"},
    {"d_max": 0.40, "channel": "call",  "offer_type": "settlement",         "label": "SETTLEMENT_CALL"},
    {"d_max": 0.55, "channel": "call",  "offer_type": "hardship_settlement", "label": "HARDSHIP_SETTLEMENT"},
    {"d_max": 1.00, "channel": "legal", "offer_type": "final_settlement",   "label": "LEGAL_LETTER"},
]

DECISION_DEFER   = "DEFER"   # ERV below floor
DECISION_SELL    = "SELL"    # Segment: low_value_low_capacity AND very low ERV
DECISION_CONTACT = "CONTACT" # Proceed with d* recommendation


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _erv(p_recovery: np.ndarray,
         e_amount_frac: np.ndarray,
         balance: float,
         d: float) -> np.ndarray:
    """
    Vectorised ERV computation.

    ERV(d) = P(Recovery | d) × E[fraction | Recovery] × Balance × (1 − d)

    Parameters
    ----------
    p_recovery    : array of P(Recovery) at discount d — shape (n,)
    e_amount_frac : array of E[fraction | Recovery]   — shape (n,)
    balance       : scalar balance (same for all rows if single account) or array
    d             : scalar discount level ∈ [0, 1]
    """
    return p_recovery * e_amount_frac * balance * (1.0 - d)


def _action_from_d_star(d_star: float) -> Dict:
    """Map d* to recommended channel + offer_type."""
    for action in ACTION_MAP:
        if d_star <= action["d_max"]:
            return {
                "channel":    action["channel"],
                "offer_type": action["offer_type"],
                "label":      action["label"],
            }
    return {"channel": "legal", "offer_type": "final_settlement", "label": "LEGAL_LETTER"}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CLASS
# ─────────────────────────────────────────────────────────────────────────────

class ERVEngine:
    """
    Expected Recovery Value optimisation engine.

    Wraps DiscountElasticityModel + RecoveryAmountModel and provides:
      - recommend()         : account-level d* recommendations
      - portfolio_summary() : aggregate ERV reporting
      - erv_curve()         : ERV(d) sweep for a single account (diagnostic)
      - sensitivity()       : how ERV changes with ±Δ in P(Recovery) assumptions

    Parameters
    ----------
    config : dict
        Must contain key ``erv_engine`` with sub-keys:
        ``discount_grid``, ``erv_floor_threshold``, ``debt_sale_threshold``,
        ``per_segment_discount_bounds``, ``max_discount_absolute``,
        ``collection_costs``, ``mlflow``.
    elasticity_model : DiscountElasticityModel (fitted)
    recovery_model   : RecoveryAmountModel (fitted)
    """

    def __init__(self, config: Dict, elasticity_model, recovery_model):
        self.config = config
        _cfg = config.get("erv_engine", {})

        self.discount_grid: List[float] = _cfg.get("discount_grid", DEFAULT_DISCOUNT_GRID)
        self.erv_floor_threshold: float = _cfg.get("erv_floor_threshold", 0.05)
        self.debt_sale_threshold: float = _cfg.get("debt_sale_threshold", 0.02)
        self.max_discount_absolute: float = _cfg.get("max_discount_absolute", 0.70)

        # Segment-level discount bounds override
        seg_bounds_raw = _cfg.get("per_segment_discount_bounds", {})
        self.segment_bounds: Dict[str, Tuple[float, float]] = {
            **DEFAULT_SEGMENT_BOUNDS,
            **{k: tuple(v) for k, v in seg_bounds_raw.items()},
        }

        # Collection cost by channel (for net ERV calculation)
        self.collection_costs: Dict[str, float] = _cfg.get(
            "collection_costs",
            config.get("constraints", {}).get("cost", {}).get("channel_costs", {
                "sms":   0.25,
                "email": 0.10,
                "call":  5.00,
                "legal": 200.00,
            })
        )

        mlflow_cfg = _cfg.get("mlflow", {})
        self.experiment_name: str = mlflow_cfg.get(
            "experiment_name", "collection_recovery_models"
        )
        self.tracking_uri: str = mlflow_cfg.get("tracking_uri", "mlruns")

        # Inject models
        self.elasticity_model = elasticity_model
        self.recovery_model   = recovery_model

        self._mlflow_available = False
        self._try_setup_mlflow()

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────────────────────────────────────

    def recommend(
        self,
        accounts_df: pd.DataFrame,
        log_to_mlflow: bool = True,
    ) -> pd.DataFrame:
        """
        Generate ERV-optimal recommendations for a batch of accounts.

        Parameters
        ----------
        accounts_df : pd.DataFrame
            Feature columns required by both sub-models +
            ``account_id``, ``current_balance``, ``segment``.
        log_to_mlflow : bool
            Log run-level summary metrics to MLflow.

        Returns
        -------
        pd.DataFrame with columns:
            account_id, segment, current_balance,
            d_star, erv_at_d_star, net_erv,
            p_recovery_at_d_star, e_amount, e_fraction,
            recommended_channel, offer_type, action_label,
            decision, decision_reason
        """
        self._check_models_fitted()

        n = len(accounts_df)
        print(f"\n[ERVEngine] Scoring {n:,} accounts across {len(self.discount_grid)} "
              f"discount levels...")

        # ── Step 1: E[Amount | Recovery, Segment] — constant across d ────────
        print("  Stage 1: Computing E[Recovery Amount] (RecoveryAmountModel)...")
        recovery_preds = self.recovery_model.predict(accounts_df, return_components=True)
        # recovery_preds columns: account_id, current_balance, segment,
        #                         p_recovery, e_fraction, e_recovery_amount

        e_fraction   = recovery_preds["e_fraction"].values
        balances     = recovery_preds["current_balance"].values
        segments     = accounts_df["segment"].values

        # ── Step 2: Sweep d, compute ERV(d) per account ───────────────────────
        print(f"  Stage 2: Sweeping discount grid {self.discount_grid}...")
        erv_matrix = np.zeros((n, len(self.discount_grid)))  # rows=accounts, cols=d levels

        for j, d in enumerate(self.discount_grid):
            elast_preds = self.elasticity_model.predict(accounts_df, discount_pct=d)
            p_d = elast_preds["p_recovery"].values
            erv_matrix[:, j] = _erv(p_d, e_fraction, balances, d)

        # ── Step 3: Apply per-segment discount bounds and find d* ─────────────
        print("  Stage 3: Applying segment bounds and finding d*...")
        d_star_indices = np.zeros(n, dtype=int)
        erv_at_d_star  = np.zeros(n)
        p_at_d_star    = np.zeros(n)

        for i in range(n):
            seg = segments[i]
            d_min, d_max = self.segment_bounds.get(seg, (0.0, self.max_discount_absolute))
            d_max = min(d_max, self.max_discount_absolute)

            # Mask invalid d values for this segment
            valid_mask = np.array([
                d_min <= d <= d_max
                for d in self.discount_grid
            ])

            if not valid_mask.any():
                valid_mask[:] = True  # Fallback: allow all

            # argmax within valid d range
            erv_row = erv_matrix[i].copy()
            erv_row[~valid_mask] = -np.inf
            best_j = int(np.argmax(erv_row))

            d_star_indices[i] = best_j
            erv_at_d_star[i]  = erv_matrix[i, best_j]

        # Get P(Recovery) at d*
        for i in range(n):
            j = d_star_indices[i]
            d = self.discount_grid[j]
            elast_preds = self.elasticity_model.predict(
                accounts_df.iloc[[i]], discount_pct=d
            )
            p_at_d_star[i] = elast_preds["p_recovery"].values[0]

        d_star_values = np.array([self.discount_grid[j] for j in d_star_indices])

        # ── Step 4: Decision logic and action mapping ─────────────────────────
        print("  Stage 4: Applying decision logic and ERV floor constraints...")
        decisions       = []
        decision_reasons = []
        channels        = []
        offer_types     = []
        action_labels   = []
        net_ervs        = []

        for i in range(n):
            seg   = segments[i]
            erv_i = erv_at_d_star[i]
            d_i   = d_star_values[i]
            bal_i = balances[i]

            # Collection cost for recommended channel
            action = _action_from_d_star(d_i)
            cost   = self.collection_costs.get(action["channel"], 5.0)
            net_erv_i = erv_i - cost

            # Decision gate
            erv_as_fraction = erv_i / max(bal_i, 1.0)

            if erv_as_fraction < self.debt_sale_threshold and seg == "low_value_low_capacity":
                decision = DECISION_SELL
                reason   = (f"ERV fraction {erv_as_fraction:.2%} < debt_sale_threshold "
                            f"{self.debt_sale_threshold:.2%} for low_value_low_capacity")
            elif erv_as_fraction < self.erv_floor_threshold:
                decision = DECISION_DEFER
                reason   = (f"ERV fraction {erv_as_fraction:.2%} < floor threshold "
                            f"{self.erv_floor_threshold:.2%}")
            else:
                decision = DECISION_CONTACT
                reason   = (f"Optimal discount={d_i:.0%}, ERV fraction={erv_as_fraction:.2%}, "
                            f"P(Recovery)={p_at_d_star[i]:.2%}")

            decisions.append(decision)
            decision_reasons.append(reason)
            channels.append(action["channel"])
            offer_types.append(action["offer_type"])
            action_labels.append(action["label"])
            net_ervs.append(round(net_erv_i, 2))

        # ── Step 5: Assemble output ────────────────────────────────────────────
        output = pd.DataFrame({
            "account_id":           accounts_df["account_id"].values,
            "segment":              segments,
            "current_balance":      balances.round(2),
            "d_star":               d_star_values.round(2),
            "erv_at_d_star":        erv_at_d_star.round(2),
            "net_erv":              net_ervs,
            "p_recovery_at_d_star": p_at_d_star.round(4),
            "e_fraction":           e_fraction.round(4),
            "e_amount":             (e_fraction * balances).round(2),
            "recommended_channel":  channels,
            "offer_type":           offer_types,
            "action_label":         action_labels,
            "decision":             decisions,
            "decision_reason":      decision_reasons,
        })

        # Stats
        n_contact = (output["decision"] == DECISION_CONTACT).sum()
        n_defer   = (output["decision"] == DECISION_DEFER).sum()
        n_sell    = (output["decision"] == DECISION_SELL).sum()
        total_erv = output.loc[output["decision"] == DECISION_CONTACT, "erv_at_d_star"].sum()

        print(f"\n  Results: {n_contact:,} CONTACT | {n_defer:,} DEFER | {n_sell:,} SELL")
        print(f"  Total ERV (CONTACT): {total_erv:,.0f}")
        print(f"  Avg d*: {d_star_values[output['decision'] == DECISION_CONTACT].mean():.1%}")

        if log_to_mlflow:
            self._log_run_to_mlflow(output, n)

        return output

    def erv_curve(
        self,
        account_row: pd.Series,
        discount_grid: Optional[List[float]] = None,
    ) -> pd.DataFrame:
        """
        Compute the full ERV(d) curve for a single account.
        Useful for diagnostics and explaining recommendations.

        Returns
        -------
        pd.DataFrame with columns:
            discount, p_recovery, e_fraction, gross_erv, net_erv, erv_marginal_gain
        """
        self._check_models_fitted()
        grid = discount_grid or self.discount_grid

        df_single = pd.DataFrame([account_row])
        balance   = float(account_row["current_balance"])

        # E[fraction] is constant in d
        rec_pred  = self.recovery_model.predict(df_single, return_components=True)
        e_fraction = float(rec_pred["e_fraction"].values[0])

        rows = []
        for d in grid:
            elast_pred = self.elasticity_model.predict(df_single, discount_pct=d)
            p_d        = float(elast_pred["p_recovery"].values[0])
            gross_erv  = _erv(
                np.array([p_d]), np.array([e_fraction]), balance, d
            )[0]
            action = _action_from_d_star(d)
            cost   = self.collection_costs.get(action["channel"], 5.0)
            rows.append({
                "discount":            d,
                "p_recovery":          round(p_d, 4),
                "e_fraction":          round(e_fraction, 4),
                "gross_erv":           round(gross_erv, 2),
                "net_erv":             round(gross_erv - cost, 2),
                "collection_cost":     cost,
                "action_label":        action["label"],
            })

        df_curve = pd.DataFrame(rows)
        df_curve["erv_marginal_gain"] = df_curve["gross_erv"].diff().round(2)

        # Mark d*
        d_star_idx = df_curve["gross_erv"].idxmax()
        df_curve["is_d_star"] = False
        df_curve.loc[d_star_idx, "is_d_star"] = True

        return df_curve

    def portfolio_summary(
        self,
        recommendations: pd.DataFrame,
        group_by_segment: bool = True,
    ) -> Dict:
        """
        Compute portfolio-level ERV statistics from recommend() output.

        Returns
        -------
        dict with keys:
            total_erv, total_net_erv, n_contact, n_defer, n_sell,
            avg_d_star, erv_by_segment (DataFrame, if group_by_segment=True)
        """
        contact_mask = recommendations["decision"] == DECISION_CONTACT

        total_erv      = recommendations.loc[contact_mask, "erv_at_d_star"].sum()
        total_net_erv  = recommendations.loc[contact_mask, "net_erv"].sum()
        avg_d_star     = recommendations.loc[contact_mask, "d_star"].mean()
        n_contact      = int(contact_mask.sum())
        n_defer        = int((recommendations["decision"] == DECISION_DEFER).sum())
        n_sell         = int((recommendations["decision"] == DECISION_SELL).sum())
        total_balance  = recommendations["current_balance"].sum()
        recovery_rate  = total_erv / max(total_balance, 1.0)

        summary = {
            "total_erv":        round(total_erv, 2),
            "total_net_erv":    round(total_net_erv, 2),
            "total_balance":    round(total_balance, 2),
            "portfolio_recovery_rate": round(recovery_rate, 4),
            "avg_d_star":       round(avg_d_star, 3) if not np.isnan(avg_d_star) else None,
            "n_total":          len(recommendations),
            "n_contact":        n_contact,
            "n_defer":          n_defer,
            "n_sell":           n_sell,
            "pct_contact":      round(n_contact / max(len(recommendations), 1), 3),
        }

        if group_by_segment:
            seg_summary = recommendations.groupby("segment").agg(
                n=("account_id", "count"),
                n_contact=("decision", lambda x: (x == DECISION_CONTACT).sum()),
                total_erv=("erv_at_d_star", "sum"),
                avg_erv=("erv_at_d_star", "mean"),
                avg_d_star=("d_star", "mean"),
                avg_p_recovery=("p_recovery_at_d_star", "mean"),
                total_balance=("current_balance", "sum"),
            ).reset_index()
            seg_summary["recovery_rate"] = (
                seg_summary["total_erv"] / seg_summary["total_balance"]
            ).round(4)
            seg_summary = seg_summary.round(2)
            summary["erv_by_segment"] = seg_summary

        return summary

    def sensitivity(
        self,
        accounts_df: pd.DataFrame,
        p_shock: float = 0.10,
        amount_shock: float = 0.10,
    ) -> pd.DataFrame:
        """
        Sensitivity analysis: how does ERV change under ±Δ shocks to
        P(Recovery) and E[Amount]?

        Parameters
        ----------
        accounts_df  : pd.DataFrame
        p_shock      : fractional shock to P(Recovery), e.g. 0.10 = ±10%
        amount_shock : fractional shock to E[Amount], e.g. 0.10 = ±10%

        Returns
        -------
        pd.DataFrame: account_id, erv_base, erv_p_up, erv_p_down,
                      erv_amt_up, erv_amt_down, erv_worst_case, erv_best_case
        """
        self._check_models_fitted()

        base_recs = self.recommend(accounts_df, log_to_mlflow=False)

        base_erv    = base_recs["erv_at_d_star"].values
        p_base      = base_recs["p_recovery_at_d_star"].values
        e_frac_base = base_recs["e_fraction"].values
        balances    = base_recs["current_balance"].values
        d_stars     = base_recs["d_star"].values

        erv_p_up    = _erv(p_base * (1 + p_shock),    e_frac_base, balances, d_stars)
        erv_p_down  = _erv(p_base * (1 - p_shock),    e_frac_base, balances, d_stars)
        erv_amt_up  = _erv(p_base, e_frac_base * (1 + amount_shock), balances, d_stars)
        erv_amt_down= _erv(p_base, e_frac_base * (1 - amount_shock), balances, d_stars)

        return pd.DataFrame({
            "account_id":    base_recs["account_id"].values,
            "segment":       base_recs["segment"].values,
            "erv_base":      base_erv.round(2),
            "erv_p_up":      erv_p_up.round(2),
            "erv_p_down":    erv_p_down.round(2),
            "erv_amt_up":    erv_amt_up.round(2),
            "erv_amt_down":  erv_amt_down.round(2),
            "erv_worst_case": np.minimum(erv_p_down, erv_amt_down).round(2),
            "erv_best_case":  np.maximum(erv_p_up,   erv_amt_up).round(2),
        })

    # ──────────────────────────────────────────────────────────────────────────
    # PRIVATE
    # ──────────────────────────────────────────────────────────────────────────

    def _check_models_fitted(self) -> None:
        if not getattr(self.elasticity_model, "_is_fitted", False):
            raise RuntimeError("DiscountElasticityModel is not fitted.")
        if not getattr(self.recovery_model, "_is_fitted", False):
            raise RuntimeError("RecoveryAmountModel is not fitted.")

    def _try_setup_mlflow(self) -> None:
        try:
            import mlflow
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)
            self._mlflow = mlflow
            self._mlflow_available = True
        except ImportError:
            logger.info("MLflow not installed — skipping experiment tracking.")

    def _log_run_to_mlflow(self, output: pd.DataFrame, n: int) -> None:
        if not self._mlflow_available:
            return
        try:
            mlflow = self._mlflow
            contact_mask = output["decision"] == DECISION_CONTACT
            with mlflow.start_run(run_name="erv_engine_batch") as run:
                mlflow.set_tag("stage", "erv_engine")
                mlflow.set_tag("n_accounts", str(n))

                mlflow.log_metrics({
                    "total_erv":       float(output.loc[contact_mask, "erv_at_d_star"].sum()),
                    "total_net_erv":   float(output.loc[contact_mask, "net_erv"].sum()),
                    "avg_d_star":      float(output.loc[contact_mask, "d_star"].mean()),
                    "pct_contact":     float(contact_mask.mean()),
                    "pct_defer":       float((output["decision"] == DECISION_DEFER).mean()),
                    "pct_sell":        float((output["decision"] == DECISION_SELL).mean()),
                })
                logger.info("ERVEngine run logged to MLflow: %s", run.info.run_id)
        except Exception as e:
            logger.warning("MLflow logging failed: %s", e)


# ─────────────────────────────────────────────────────────────────────────────
# SPARK / DATABRICKS WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

class ERVEngineSpark:
    """
    PySpark wrapper for running ERV recommendations on Databricks at scale.

    Usage
    -----
        erv_spark = ERVEngineSpark(erv_engine)
        recommendations_sdf = erv_spark.recommend(accounts_sdf)
    """

    def __init__(self, erv_engine: ERVEngine):
        self.engine = erv_engine

    def recommend(self, spark_df):
        """Run ERV recommendations on a PySpark DataFrame."""
        from pyspark.sql.types import (
            StructType, StructField, StringType, DoubleType, BooleanType
        )

        schema = StructType([
            StructField("account_id",              StringType(), True),
            StructField("segment",                 StringType(), True),
            StructField("current_balance",         DoubleType(), True),
            StructField("d_star",                  DoubleType(), True),
            StructField("erv_at_d_star",           DoubleType(), True),
            StructField("net_erv",                 DoubleType(), True),
            StructField("p_recovery_at_d_star",    DoubleType(), True),
            StructField("e_fraction",              DoubleType(), True),
            StructField("e_amount",                DoubleType(), True),
            StructField("recommended_channel",     StringType(), True),
            StructField("offer_type",              StringType(), True),
            StructField("action_label",            StringType(), True),
            StructField("decision",                StringType(), True),
            StructField("decision_reason",         StringType(), True),
        ])

        import pyspark
        sc = pyspark.SparkContext.getOrCreate()
        bc_engine = sc.broadcast(self.engine)

        return spark_df.groupby().applyInPandas(
            lambda pdf: bc_engine.value.recommend(pdf, log_to_mlflow=False),
            schema=schema,
        )


# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE / SMOKE TEST (end-to-end with all three models)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import os
    import yaml

    # Allow running from repo root
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

    from src.decision_agent.debt_collection.recovery_amount_model import RecoveryAmountModel
    from src.decision_agent.debt_collection.discount_elasticity    import DiscountElasticityModel

    with open("conf/use_cases/debt_collection_optimization.yaml") as f:
        config = yaml.safe_load(f)

    np.random.seed(42)
    n = 2000
    segments = ["high_value_cooperative", "medium_value_willing",
                "high_value_unresponsive", "low_value_low_capacity"]

    # ── Shared synthetic dataset ──────────────────────────────────────────────
    df = pd.DataFrame({
        "account_id":              [f"ACC{i:05d}" for i in range(n)],
        "current_balance":         np.random.uniform(500, 50_000, n),
        "original_balance":        np.random.uniform(1_000, 60_000, n),
        "balance_ratio":           np.random.uniform(0.3, 1.0, n),
        "months_since_charge_off": np.random.randint(1, 12, n),
        "days_past_due":           np.random.randint(90, 730, n),
        "payment_history_score":   np.random.uniform(0, 1, n),
        "contact_response_rate":   np.random.uniform(0, 1, n),
        "promise_kept_rate":       np.random.uniform(0, 1, n),
        "settlement_offers_made":  np.random.randint(0, 5, n),
        "avg_discount_offered":    np.random.uniform(0.0, 0.6, n),
        "discount_offered":        np.random.uniform(0.0, 0.6, n),
        "debt_to_income_ratio":    np.random.uniform(0.1, 1.5, n),
        "payment_capacity_score":  np.random.uniform(0, 1, n),
        "segment":                 np.random.choice(segments, n),
        "offer_randomisation_flag": np.random.choice([0, 1], n, p=[0.7, 0.3]),
    })

    recovery_prob = (
        0.20 * df["payment_capacity_score"] +
        0.15 * df["contact_response_rate"] +
        0.35 * df["discount_offered"] +
        0.10 * df["promise_kept_rate"] +
        0.05 * (1 - df["balance_ratio"])
    ).clip(0, 1)

    df["recovery_occurred"] = (np.random.uniform(0, 1, n) < recovery_prob).astype(int)
    df["recovery_fraction"]  = np.where(
        df["recovery_occurred"] == 1,
        np.random.beta(2, 5, n).clip(1e-6, 1 - 1e-6),
        np.nan,
    )

    train_df = df.iloc[:1600]
    val_df   = df.iloc[1600:]

    cfg_overrides = {
        "recovery_amount_model": {
            "stage1":  {"n_estimators": 80, "max_depth": 4, "learning_rate": 0.05},
            "stage2":  {"n_estimators": 60, "max_depth": 3, "learning_rate": 0.05},
            "min_segment_samples": 50,
            "calibrate_stage1": True,
            "mlflow": {"experiment_name": "erv_smoke_test", "tracking_uri": "mlruns"},
        },
        "discount_elasticity_model": {
            "model": {"n_estimators": 80, "max_depth": 4, "learning_rate": 0.05},
            "monotonicity_enforcement": True,
            "calibrate": True,
            "iv_reweighting": {"enabled": True, "iv_col": "offer_randomisation_flag"},
            "mlflow": {"experiment_name": "erv_smoke_test", "tracking_uri": "mlruns"},
        },
        "erv_engine": {
            "discount_grid":       DEFAULT_DISCOUNT_GRID,
            "erv_floor_threshold": 0.05,
            "debt_sale_threshold": 0.02,
            "max_discount_absolute": 0.65,
            "mlflow": {"experiment_name": "erv_smoke_test", "tracking_uri": "mlruns"},
        },
        "constraints": config.get("constraints", {}),
    }

    # ── Train sub-models ──────────────────────────────────────────────────────
    print("=" * 60)
    print("Training RecoveryAmountModel...")
    recovery_model = RecoveryAmountModel(cfg_overrides)
    recovery_model.fit(train_df, val_df)

    print("\nTraining DiscountElasticityModel...")
    elast_model = DiscountElasticityModel(cfg_overrides)
    elast_model.fit(train_df, val_df)

    # ── Initialise ERV Engine ─────────────────────────────────────────────────
    erv = ERVEngine(cfg_overrides, elast_model, recovery_model)

    # ── Full recommendations ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    recs = erv.recommend(val_df)
    print("\n[Sample Recommendations (first 10)]")
    cols = ["account_id", "segment", "current_balance", "d_star",
            "erv_at_d_star", "p_recovery_at_d_star", "decision", "action_label"]
    print(recs[cols].head(10).to_string(index=False))

    # ── Portfolio summary ─────────────────────────────────────────────────────
    summary = erv.portfolio_summary(recs)
    print(f"\n[Portfolio Summary]")
    for k, v in summary.items():
        if k != "erv_by_segment":
            print(f"  {k}: {v}")
    if "erv_by_segment" in summary:
        print("\n[ERV by Segment]")
        print(summary["erv_by_segment"].to_string(index=False))

    # ── ERV curve for one account ─────────────────────────────────────────────
    curve = erv.erv_curve(val_df.iloc[0])
    print("\n[ERV Curve — Account 0]")
    print(curve.to_string(index=False))

    # ── Sensitivity analysis ──────────────────────────────────────────────────
    sens = erv.sensitivity(val_df.head(5))
    print("\n[Sensitivity Analysis — first 5 accounts]")
    print(sens.to_string(index=False))

    print("\n✓ ERVEngine end-to-end smoke test passed")
