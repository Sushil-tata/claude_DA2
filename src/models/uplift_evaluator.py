"""
UpliftEvaluator
===============
Evaluation metrics for the Phase 2 uplift model in the CardX/SCB Thailand
debt collections recovery system.

Provides Qini coefficient, uplift curves, incremental recovery rates,
per-segment ROI, and negative-uplift account summaries.

All methods expect a DataFrame with tau (uplift score), paid (0/1 outcome),
and treatment (0/1 indicator) at minimum. Additional columns (erv_at_d_optimal,
signal_segment, cost_per_account) are required for specific methods.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Required column sets per method ──────────────────────────────────────────
_QINI_REQUIRED       = {"tau", "paid", "treatment"}
_CURVE_REQUIRED      = {"tau", "paid", "treatment"}
_INCREMENTAL_REQUIRED = {"tau", "paid", "treatment", "erv_at_d_optimal"}
_SEGMENT_ROI_REQUIRED = {"signal_segment", "treatment", "paid", "erv_at_d_optimal",
                          "cost_per_account", "tau"}
_NEG_UPLIFT_REQUIRED  = {"tau", "erv_at_d_optimal"}


class UpliftEvaluator:
    """
    Evaluation toolkit for T-learner uplift models.

    All methods are stateless — pass the DataFrame each time.
    All methods log their results at INFO level.
    """

    # ── 1. Qini coefficient ───────────────────────────────────────────────────

    def qini_coefficient(self, df: pd.DataFrame) -> float:
        """
        Compute the Qini coefficient for an uplift model.

        The Qini coefficient measures the area between the uplift curve
        (model-ranked) and the random baseline. Higher is better.
        A coefficient of 0 means the model performs no better than random.
        Negative values indicate the model is worse than random.

        Args:
            df: DataFrame with columns: tau (uplift score), paid (0/1 outcome),
                treatment (0/1 flag where 1 = treated, 0 = control).

        Returns:
            float: Qini coefficient (area between uplift curve and random baseline).
        """
        self._require_columns(df, _QINI_REQUIRED, method="qini_coefficient")

        df = df.copy().sort_values("tau", ascending=False).reset_index(drop=True)
        n = len(df)

        n_treatment_total = df["treatment"].sum()
        n_control_total   = (1 - df["treatment"]).sum()

        if n_treatment_total == 0 or n_control_total == 0:
            raise ValueError(
                "qini_coefficient: both treatment and control groups must be non-empty."
            )

        # Cumulative counts
        cum_treated_paid   = (df["paid"] * df["treatment"]).cumsum()
        cum_control_paid   = (df["paid"] * (1 - df["treatment"])).cumsum()
        cum_n              = np.arange(1, n + 1)

        # Qini curve: incremental gains over random
        qini_curve = cum_treated_paid - cum_control_paid * (n_treatment_total / n_control_total)

        # Area under Qini curve (trapezoid integration over n/N)
        x = cum_n / n
        area_model = float(np.trapz(qini_curve, x))

        # Random baseline area = straight line from 0 to final Qini value
        final_qini = float(qini_curve.iloc[-1])
        area_random = final_qini / 2.0

        qini_coef = area_model - area_random

        logger.info(
            f"qini_coefficient | n={n:,} | n_treated={n_treatment_total:.0f} | "
            f"n_control={n_control_total:.0f} | qini={qini_coef:.4f}"
        )
        return float(qini_coef)

    # ── 2. Uplift curve ───────────────────────────────────────────────────────

    def uplift_curve(self, df: pd.DataFrame) -> dict:
        """
        Compute the uplift curve (model vs random baseline).

        Sorts accounts by tau descending. For each cumulative percentile of
        accounts contacted, computes the incremental recovery rate above
        the control baseline.

        Args:
            df: DataFrame with columns: tau, paid (0/1), treatment (0/1).

        Returns:
            dict with keys:
              "x"        — list of cumulative % contacted (0.0 to 1.0)
              "y_uplift" — list of incremental recovery rates at each x
              "y_random" — list of random baseline values (straight line)
        """
        self._require_columns(df, _CURVE_REQUIRED, method="uplift_curve")

        df = df.copy().sort_values("tau", ascending=False).reset_index(drop=True)
        n = len(df)

        n_treatment_total = df["treatment"].sum()
        n_control_total   = (1 - df["treatment"]).sum()

        if n_treatment_total == 0 or n_control_total == 0:
            raise ValueError(
                "uplift_curve: both treatment and control groups must be non-empty."
            )

        cum_treated_paid   = (df["paid"] * df["treatment"]).cumsum()
        cum_control_paid   = (df["paid"] * (1 - df["treatment"])).cumsum()
        cum_treated_n      = df["treatment"].cumsum()
        cum_control_n      = (1 - df["treatment"]).cumsum()

        # Avoid division by zero
        treat_rate = cum_treated_paid / cum_treated_n.replace(0, np.nan)
        ctrl_rate  = cum_control_paid / cum_control_n.replace(0, np.nan)
        y_uplift   = (treat_rate - ctrl_rate).fillna(0.0).tolist()

        # x-axis: cumulative proportion of portfolio contacted
        x = (np.arange(1, n + 1) / n).tolist()

        # Random baseline: straight line from 0 to final uplift
        final_uplift = y_uplift[-1] if y_uplift else 0.0
        y_random = [final_uplift * xi for xi in x]

        logger.info(
            f"uplift_curve | n={n:,} | final_uplift={final_uplift:.4f} | "
            f"n_x_points={len(x)}"
        )
        return {"x": x, "y_uplift": y_uplift, "y_random": y_random}

    # ── 3. Incremental recovery ───────────────────────────────────────────────

    def incremental_recovery(self, df: pd.DataFrame) -> dict:
        """
        Compute actual incremental recovery rate: TREATMENT vs CONTROL.

        Args:
            df: DataFrame with columns: paid (0/1), treatment (0/1),
                erv_at_d_optimal (THB value per account).

        Returns:
            dict with keys:
              treatment_rate              — P(paid | treated)
              control_rate                — P(paid | control)
              incremental_rate            — treatment_rate - control_rate
              incremental_thb_per_account — incremental_rate × mean(erv_at_d_optimal)
        """
        self._require_columns(df, _INCREMENTAL_REQUIRED, method="incremental_recovery")

        treated_df = df[df["treatment"] == 1]
        control_df = df[df["treatment"] == 0]

        if len(treated_df) == 0 or len(control_df) == 0:
            raise ValueError(
                "incremental_recovery: both treatment and control groups must be non-empty."
            )

        treatment_rate = float(treated_df["paid"].mean())
        control_rate   = float(control_df["paid"].mean())
        incremental    = treatment_rate - control_rate
        mean_erv       = float(df["erv_at_d_optimal"].mean())
        incremental_thb = incremental * mean_erv

        result = {
            "treatment_rate":              round(treatment_rate, 4),
            "control_rate":                round(control_rate, 4),
            "incremental_rate":            round(incremental, 4),
            "incremental_thb_per_account": round(incremental_thb, 2),
        }

        logger.info(
            f"incremental_recovery | n_treated={len(treated_df):,} | "
            f"n_control={len(control_df):,} | "
            f"treatment_rate={treatment_rate:.4f} | control_rate={control_rate:.4f} | "
            f"incremental={incremental:.4f} | "
            f"incremental_thb_per_account={incremental_thb:.2f}"
        )
        return result

    # ── 4. Segment ROI ────────────────────────────────────────────────────────

    def segment_roi(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute per-SIGNAL_SEGMENT ROI of treatment vs control.

        ROI = (uplift × mean_erv - cost_per_account) / cost_per_account

        Args:
            df: DataFrame with columns: signal_segment, treatment (0/1),
                paid (0/1), erv_at_d_optimal, cost_per_account, tau.

        Returns:
            pd.DataFrame with columns:
              signal_segment, treatment_rate, control_rate, uplift,
              cost_per_account, roi, n_treated, n_control
        """
        self._require_columns(df, _SEGMENT_ROI_REQUIRED, method="segment_roi")

        rows = []
        for segment, seg_df in df.groupby("signal_segment"):
            treated = seg_df[seg_df["treatment"] == 1]
            control = seg_df[seg_df["treatment"] == 0]

            treatment_rate = float(treated["paid"].mean()) if len(treated) > 0 else 0.0
            control_rate   = float(control["paid"].mean()) if len(control) > 0 else 0.0
            uplift         = treatment_rate - control_rate
            mean_erv       = float(seg_df["erv_at_d_optimal"].mean())
            cost           = float(seg_df["cost_per_account"].mean())

            roi = ((uplift * mean_erv - cost) / cost) if cost > 0 else np.nan

            rows.append({
                "signal_segment":  segment,
                "treatment_rate":  round(treatment_rate, 4),
                "control_rate":    round(control_rate, 4),
                "uplift":          round(uplift, 4),
                "cost_per_account": round(cost, 2),
                "roi":             round(roi, 4) if not np.isnan(roi) else np.nan,
                "n_treated":       len(treated),
                "n_control":       len(control),
            })

        result_df = pd.DataFrame(rows).sort_values("signal_segment").reset_index(drop=True)

        logger.info(
            f"segment_roi | segments={result_df['signal_segment'].tolist()} | "
            f"roi_by_segment={dict(zip(result_df['signal_segment'], result_df['roi']))}"
        )
        return result_df

    # ── 5. Negative uplift summary ────────────────────────────────────────────

    def negative_uplift_summary(self, df: pd.DataFrame) -> dict:
        """
        Summarise accounts where tau < 0 — contacting HURTS recovery.

        Negative uplift means the model predicts treated accounts are LESS
        likely to pay than comparable untreated accounts. This can happen
        with strategic defaulters who dispute debts when contacted.

        Args:
            df: DataFrame with columns: tau, erv_at_d_optimal.

        Returns:
            dict with keys:
              count                             — number of negative-uplift accounts
              pct_of_portfolio                  — % of all accounts with tau < 0
              mean_erv_of_negative_uplift_accounts — avg ERV for those accounts
        """
        self._require_columns(df, _NEG_UPLIFT_REQUIRED, method="negative_uplift_summary")

        neg_df = df[df["tau"] < 0]
        total  = len(df)
        count  = len(neg_df)
        pct    = (count / total * 100.0) if total > 0 else 0.0
        mean_erv = float(neg_df["erv_at_d_optimal"].mean()) if count > 0 else 0.0

        result = {
            "count":                              count,
            "pct_of_portfolio":                   round(pct, 2),
            "mean_erv_of_negative_uplift_accounts": round(mean_erv, 2),
        }

        logger.info(
            f"negative_uplift_summary | total={total:,} | "
            f"negative_uplift_count={count:,} ({pct:.2f}%) | "
            f"mean_erv_negative={mean_erv:.2f}"
        )
        return result

    # ── Shared validation helper ──────────────────────────────────────────────

    @staticmethod
    def _require_columns(df: pd.DataFrame, required: set, method: str) -> None:
        """Raise ValueError with clear message if any required column is missing."""
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"UpliftEvaluator.{method}: missing required columns: "
                f"{sorted(missing)}. "
                f"Available columns: {sorted(df.columns.tolist())}"
            )
