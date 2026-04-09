"""
RecoverabilitySegmenter
=======================
Derives RECOVERABILITY_TIER for post-charge-off accounts (180+ DPD).

PURPOSE
-------
This is the PRIMARY STRATEGIC segmentation for charge-off recovery portfolios.
It answers one question before any other:

    "Is recovery structurally possible for this account, and through
     what mechanism — self-cure, agency settlement, legal, or write-off?"

This is NOT behavioural segmentation. BEHAVIOURAL_PERSONA remains as a
secondary tactical layer (contact channel, sequencing, tone). The
RECOVERABILITY_TIER drives the strategic objective and ERV horizon.

WHY THIS IS DIFFERENT FROM BEHAVIOURAL PERSONA
------------------------------------------------
BEHAVIOURAL_PERSONA clusters on current state (last 3 months of behaviour).
At charge-off, current behaviour is near-zero for most accounts — the card
is closed, transactions are zero, contact rates are 10–20%. Clustering on
degraded post-CO signals produces geometrically separated but
business-meaningless groups.

RECOVERABILITY_TIER is derived from:
  (1) The TRAJECTORY that led to charge-off — sudden shock or chronic decline?
  (2) The STRUCTURAL financial position at charge-off — can they pay?
  (3) The BUREAU evidence of external capacity — are they paying other lenders?
  (4) Any POST-CO engagement signals — even one payment changes everything.

The 24-month pre-charge-off window is more predictive than the 3-month
post-charge-off window. This segmenter uses it.

──────────────────────────────────────────────────────────────────────────────
RECOVERABILITY TIERS
──────────────────────────────────────────────────────────────────────────────
TIER_1_HIGH
    Structurally recoverable. Evidence of both capacity and motivation.
    Recovery is possible within 3–6 months with the right intervention.

    Typical profiles:
      - Sudden shock (job loss, medical) — was paying, then stopped abruptly
      - Strategic defaulter — paying other lenders, ignoring this card
      - Early post-CO payment — already engaging, just needs the right offer

    Decision:
      Agency placement month 1–3. Firm settlement offer. ERV on 90d horizon.
      Do NOT offer maximum discount immediately — these accounts will settle.

TIER_2_MODERATE
    Possibly recoverable. Mixed or deteriorating signals.
    Recovery requires patient approach; may take 6–12 months.

    Typical profiles:
      - Gradual decline but still some bureau payments
      - Made partial payments until 6 months before CO
      - Contact responsive but no payment

    Decision:
      Agency + digital. Flexible settlement offer. ERV on 180d horizon.
      Structured instalment plan often works here.

TIER_3_LOW
    Low recovery probability. Structural impairment likely.
    Recovery possible only with deep discount or long-horizon settlement.

    Typical profiles:
      - Chronic 12+ month delinquency trajectory
      - All bureau accounts delinquent simultaneously
      - Silent 6+ months post-CO with no prior engagement signals

    Decision:
      Low-cost digital only. Legal review for high-balance accounts.
      ERV on 360d horizon. Deep discount (50–60%) required.
      Minimal agency spend — ROI is low.

TIER_4_DORMANT
    Non-recoverable under current economics.
    Cost of further action exceeds expected recovery value.

    Typical profiles:
      - 18+ months post-CO with zero payment and zero contact response
      - All bureau accounts delinquent for 12+ months
      - No bureau activity (withdrawn from formal credit entirely)

    Decision:
      HOLD — no outbound. Flag for portfolio sale, final write-off, or
      legal judgment (last resort, low expected collection).

──────────────────────────────────────────────────────────────────────────────
SCORING RUBRIC (transparent, auditable)
──────────────────────────────────────────────────────────────────────────────
Four dimensions, each scored 0–3. Maximum total = 12.

DIMENSION 1 — TRAJECTORY SCORE (DPD path to charge-off)
    +3  Sudden shock: account was <60 DPD six months before CO
        (rapid onset — consistent with temporary liquidity crisis)
    +2  Semi-rapid: <90 DPD six months before CO, or ever partially cured
    +1  Some engagement: last payment within 6 months of CO, or
        months_first_delinquent < 6
    +0  Chronic: 90+ DPD for 12+ months before CO (structural default)

DIMENSION 2 — PAYMENT HISTORY SCORE (own obligation engagement)
    +3  Recent payment: last payment within 3 months of CO AND
        payment_decay_ratio > 0.3
    +2  Moderate: last payment within 6 months of CO, or
        ever_partially_cured = True
    +1  Weak: some payment in last 12M before CO
    +0  No payment history before CO (card never meaningfully serviced)

DIMENSION 3 — BUREAU STRUCTURAL SCORE (external capacity evidence)
    +3  Strong: paying mortgage or secured loan AND other accounts current
        (most powerful signal — strategic defaulter or isolated distress)
    +2  Moderate: other accounts current but no secured loan
    +1  Weak: some accounts current, majority delinquent
    +0  Systemic: all bureau accounts delinquent (structural impairment)

DIMENSION 4 — POST-CHARGE-OFF ENGAGEMENT SCORE
    +3  Any payment received after charge-off date (decisive signal)
    +2  Contact response post-CO (spoke with collections team)
    +1  Agency engagement (responded to agency without payment)
    +0  Complete silence post-CO

TIER THRESHOLDS (default, configurable)
    TIER_1_HIGH:      total score >= 8
    TIER_2_MODERATE:  total score 5–7
    TIER_3_LOW:       total score 2–4
    TIER_4_DORMANT:   total score 0–1
                      OR: months_since_co >= 24 AND score < 5 AND
                          no payment post-CO

TIME-DECAY OVERRIDE
    Accounts > 24 months past charge-off with score < 5 are forced to
    TIER_4 regardless of score. Recovery probability decays significantly
    with time and agency fees make these accounts uneconomic.
    Override: RecoverabilitySegmenter(dormancy_months_override=36)

──────────────────────────────────────────────────────────────────────────────
RECOVERABILITY SUBTYPE
──────────────────────────────────────────────────────────────────────────────
Within each tier, RECOVERABILITY_SUBTYPE identifies the driver:

TIER_1 subtypes:
    SHOCK_RECOVERY      — sudden onset, clear temporary distress
    STRATEGIC_DEFAULTER — bureau strong, paying other lenders
    ACTIVE_ENGAGER      — payment or contact post-CO already occurring
    STRONG_HISTORY      — paid consistently until close to CO date

TIER_2 subtypes:
    GRADUAL_DECLINE     — chronic but slow deterioration
    BUREAU_MODERATE     — some external capacity remaining
    SILENT_MODERATE     — moderate score but no post-CO engagement yet

TIER_3 subtypes:
    CHRONIC_IMPAIRED    — long history of delinquency, systemic
    DEEP_DISTRESS       — bureau widespread, no payment signals

TIER_4 subtypes:
    AGED_DORMANT        — time override triggered (>24M no payment)
    CONFIRMED_IMPAIRED  — all four dimensions score 0

The subtype is interpretable to collectors: a STRATEGIC_DEFAULTER requires
a completely different playbook than a SHOCK_RECOVERY account.

──────────────────────────────────────────────────────────────────────────────
REQUIRED FEATURES
──────────────────────────────────────────────────────────────────────────────
See RECOVERABILITY_FEATURES dict below.
All features are optional — scoring degrades gracefully when absent.
Minimum viable features: months_since_co + outstanding_balance_at_co.

In Databricks, pre-CO trajectory features must be joined from the account
history table (not model_scores or feature_store). Configure:
    PRE_CO_HISTORY_TABLE env variable → account DPD history table
    POST_CO_PAYMENT_TABLE env variable → collections payment table
──────────────────────────────────────────────────────────────────────────────
"""

import logging
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Required feature definitions ─────────────────────────────────────────────
# All features are optional. Scoring degrades when absent (treated as 0 / unknown).
# Add source table annotation to guide data engineering.

RECOVERABILITY_FEATURES = {
    # ── DPD TRAJECTORY (pre-charge-off) ──────────────────────────────────────
    # Source: account DPD history table (monthly snapshots)
    # These are the MOST IMPORTANT features. Derive from history table if not
    # already in feature store.
    "dpd_at_minus_3m":           "DPD 3 months before charge-off date",
    "dpd_at_minus_6m":           "DPD 6 months before charge-off date",
    "dpd_at_minus_12m":          "DPD 12 months before charge-off date",
    "months_first_delinquent":   "Total months account has been delinquent (at CO date)",
    "ever_partially_cured":      "1 if DPD ever went back below 90 after first delinquency",
    "dpd_velocity_3m":           "(dpd_current - dpd_at_minus_3m) / 3  [rate of DPD change]",
    "dpd_velocity_6m":           "(dpd_current - dpd_at_minus_6m) / 6",

    # ── PAYMENT HISTORY (pre-charge-off) ─────────────────────────────────────
    # Source: transaction / payment table
    "last_payment_months_before_co": "Months before CO date that last payment was made",
    "payment_decay_ratio":           "avg_payment_last_3M_pre_CO / avg_payment_6to12M_pre_CO "
                                     "(0 = collapsed, 1 = stable, >1 = increasing)",
    "payment_count_last_12m_pre_co": "Count of payments in final 12 months before CO",
    "avg_payment_amount_last_6m":    "Average payment amount in final 6 months before CO (THB)",

    # ── BUREAU STRUCTURAL POSITION (at charge-off date) ──────────────────────
    # Source: NCB/TUEF bureau pull at or near CO date
    "ncb_other_accounts_current":    "Count of other bureau accounts still current at CO",
    "ncb_accounts_delinquent_count": "Count of other bureau accounts delinquent at CO",
    "ncb_total_delinquent_pct":      "Fraction of total credit exposure that is delinquent "
                                     "(0 = this card only, 1 = everything delinquent)",
    "ncb_mortgage_current":          "1 if mortgage is current at CO date — strongest "
                                     "strategic defaulter signal",
    "ncb_secured_loan_current":      "1 if any secured loan (car, personal) is current",
    "ncb_score_at_co":               "Bureau score at charge-off date",
    "ncb_score_at_minus_6m":         "Bureau score 6 months before charge-off",
    "ncb_score_trend_6m":            "ncb_score_at_co - ncb_score_at_minus_6m "
                                     "(positive = improving, negative = deteriorating)",
    "ncb_new_credit_post_co":        "1 if any new credit approved after CO date "
                                     "(bureau sees financial recovery)",

    # ── POST-CHARGE-OFF ENGAGEMENT ────────────────────────────────────────────
    # Source: collections payment + contact log tables
    "any_payment_post_co":           "1 if any payment received after charge-off date "
                                     "(strongest available signal)",
    "payment_amount_post_co_total":  "Total THB received after charge-off date",
    "contact_response_post_co":      "1 if customer responded to any post-CO contact attempt",
    "agency_response_flag":          "1 if responded to agency (if already placed)",
    "months_since_co":               "Months elapsed since charge-off date",

    # ── ACCOUNT ECONOMICS ─────────────────────────────────────────────────────
    "outstanding_balance_at_co":     "Outstanding balance at time of charge-off (THB)",
    "credit_limit_at_co":            "Credit limit at charge-off date (THB)",
    "utilisation_at_co":             "Balance / credit_limit at CO (proxy for max exposure)",
}

# ── Tier configuration ────────────────────────────────────────────────────────

TIER_LABELS = {
    1: "TIER_1_HIGH",
    2: "TIER_2_MODERATE",
    3: "TIER_3_LOW",
    4: "TIER_4_DORMANT",
}

DEFAULT_TIER_THRESHOLDS = {
    "TIER_1_HIGH":     8,   # score >= 8
    "TIER_2_MODERATE": 5,   # score 5–7
    "TIER_3_LOW":      2,   # score 2–4
    "TIER_4_DORMANT":  0,   # score 0–1
}

# Accounts beyond this age (months) with score < dormancy_score_cap
# are forced to TIER_4 regardless of score.
DEFAULT_DORMANCY_MONTHS   = 24
DEFAULT_DORMANCY_SCORE_CAP = 5


class RecoverabilitySegmenter:
    """
    Derives RECOVERABILITY_TIER and RECOVERABILITY_SUBTYPE for charge-off accounts.

    Scoring is transparent and rule-based by default (no training data needed).
    Supports a model-based scorer (LightGBM) as an optional upgrade once
    recovery_180d labels are available.

    Usage:
        # Rule-based (default — works immediately):
        segmenter = RecoverabilitySegmenter()
        df["recoverability_tier"]    = segmenter.score(df)["tier"]
        df["recoverability_score"]   = segmenter.score(df)["score"]
        df["recoverability_subtype"] = segmenter.score(df)["subtype"]

        # Model-based (once labels available):
        segmenter.fit(df_with_outcomes)   # df must contain recovery_180d
        segmenter.save("models/recoverability/")
        tier_df = segmenter.score(df)
    """

    def __init__(
        self,
        tier_thresholds: dict    = None,
        dormancy_months: int     = DEFAULT_DORMANCY_MONTHS,
        dormancy_score_cap: int  = DEFAULT_DORMANCY_SCORE_CAP,
        model_dir: Path          = None,
    ):
        """
        Args:
            tier_thresholds:    Override default score thresholds per tier.
                                Format: {"TIER_1_HIGH": 8, "TIER_2_MODERATE": 5, ...}
                                Recalibrate on your portfolio using:
                                    RecoverabilitySegmenter.calibrate_thresholds(df_with_outcomes)
            dormancy_months:    Accounts older than this (months since CO) with
                                score < dormancy_score_cap are forced to TIER_4.
                                Default: 24 months. Increase to 36 if portfolio has
                                active long-tail agency placement.
            dormancy_score_cap: Score threshold below which dormancy override applies.
                                Default: 5 (TIER_2 boundary).
            model_dir:          If set, loads LightGBM model from this directory.
                                Rule-based scoring is used when no model exists.
        """
        self.tier_thresholds    = tier_thresholds or DEFAULT_TIER_THRESHOLDS
        self.dormancy_months    = dormancy_months
        self.dormancy_score_cap = dormancy_score_cap
        self.model_dir          = Path(model_dir) if model_dir else None
        self._model             = None

        if self.model_dir:
            self._try_load_model()

    # ── Public interface ──────────────────────────────────────────────────────

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Scores all accounts and assigns RECOVERABILITY_TIER and RECOVERABILITY_SUBTYPE.

        Args:
            df: DataFrame with charge-off account features.
                See RECOVERABILITY_FEATURES for required columns.
                Missing columns degrade scoring but do not raise errors.

        Returns:
            DataFrame with columns:
                recoverability_tier    — TIER_1_HIGH / TIER_2_MODERATE /
                                         TIER_3_LOW / TIER_4_DORMANT
                recoverability_score      — int 0–12 (sum of four dimension scores)
                recov_dim1_trajectory     — Dimension 1 score: DPD path to CO (0–3)
                recov_dim2_payment        — Dimension 2 score: pre-CO payment history (0–3)
                recov_dim3_bureau         — Dimension 3 score: bureau structural position (0–3)
                recov_dim4_post_co        — Dimension 4 score: post-CO engagement (0–3)
                recoverability_subtype — interpretable profile name
                recoverability_drivers — pipe-separated top factors
                (same index as df)
        """
        if self._model is not None:
            return self._score_model(df)
        return self._score_rules(df)

    def fit(self, df: pd.DataFrame) -> dict:
        """
        Trains a LightGBM recoverability model using recovery_180d as the label.
        Replaces rule-based scoring once enough labelled data is available.
        Saves model to model_dir.

        Args:
            df: DataFrame with RECOVERABILITY_FEATURES + recovery_180d (label).
                recovery_180d: 1 if any payment received within 180d of scoring date.

        Returns:
            dict: {auc, n_train, n_val, feature_importance, tier_recovery_rates}

        Training notes:
            - Use accounts that charged off at least 180 days ago (labelled cohort)
            - Include accounts with recovery_180d = 0 (not just recovered accounts)
            - Minimum 200 accounts per tier recommended before switching from rules
        """
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError(
                "LightGBM required for model-based scoring. "
                "pip install lightgbm  OR  use rule-based scoring (default)."
            )

        if "recovery_180d" not in df.columns:
            raise ValueError(
                "recovery_180d column required for fit(). "
                "This is 1 if any payment was received within 180d of the charge-off date. "
                "Use score() with rule-based scoring until labels are available."
            )

        logger.info(
            f"Training recoverability model | n={len(df):,} | "
            f"recovery_rate_180d={df['recovery_180d'].mean():.1%}"
        )

        X, feature_cols = self._prepare_model_features(df)
        y = df["recovery_180d"].astype(int).values

        # Time-based train/val split
        if "charge_off_date" in df.columns:
            df_sorted  = df.sort_values("charge_off_date")
            split_idx  = int(len(df_sorted) * 0.8)
            train_mask = df.index.isin(df_sorted.index[:split_idx])
        else:
            from sklearn.model_selection import train_test_split
            train_mask = pd.Series(False, index=df.index)
            train_idx, _ = train_test_split(
                df.index, test_size=0.2, random_state=42, stratify=y
            )
            train_mask[train_idx] = True

        X_train, X_val = X[train_mask], X[~train_mask]
        y_train, y_val = y[train_mask], y[~train_mask]

        params = {
            "objective":             "binary",
            "metric":                "auc",
            "num_leaves":            31,
            "learning_rate":         0.05,
            "class_weight":          "balanced",
            "feature_fraction":      0.8,
            "bagging_fraction":      0.8,
            "bagging_freq":          5,
            "n_estimators":          300,
            "early_stopping_rounds": 30,
            "verbose":               -1,
        }
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(y_val, model.predict_proba(X_val)[:, 1]))

        self._model        = model
        self._feature_cols = feature_cols

        if self.model_dir:
            self.model_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(
                {"model": model, "feature_cols": feature_cols},
                self.model_dir / "recoverability_model.pkl"
            )
            logger.info(f"Recoverability model saved → {self.model_dir}")

        feat_imp = dict(sorted(
            zip(feature_cols, model.feature_importances_),
            key=lambda x: x[1], reverse=True
        ))

        # Recovery rate by tier using rule-based scoring for comparison
        tier_df       = self._score_rules(df)
        df_with_tier  = df.copy()
        df_with_tier["recoverability_tier"] = tier_df["recoverability_tier"]
        tier_recovery = (
            df_with_tier.groupby("recoverability_tier")["recovery_180d"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "recovery_rate", "count": "n"})
            .to_dict()
        )

        logger.info(
            f"Recoverability model | AUC={auc:.4f} | "
            f"n_train={len(X_train):,} | n_val={len(X_val):,}"
        )
        return {
            "auc":                   round(auc, 4),
            "n_train":               len(X_train),
            "n_val":                 len(X_val),
            "recovery_rate_overall": round(float(df["recovery_180d"].mean()), 4),
            "top_features":          list(feat_imp.items())[:10],
            "tier_recovery_rates":   tier_recovery,
        }

    def calibrate_thresholds(self, df: pd.DataFrame) -> dict:
        """
        Suggests tier thresholds calibrated to your portfolio's recovery rate distribution.

        Run this after scoring a labelled cohort to validate that tier boundaries
        produce meaningful differentiation on actual recovery outcomes.

        Returns:
            dict of {tier: {threshold, n, recovery_rate}} with business interpretation.
        """
        if "recovery_180d" not in df.columns:
            raise ValueError("recovery_180d required for threshold calibration.")

        score_df = self.score(df)
        df_cal   = df[["recovery_180d"]].copy()
        df_cal["raw_score"] = score_df["recoverability_score"]

        results = {}
        for threshold in range(1, 12):
            tier1 = df_cal[df_cal["raw_score"] >= threshold]["recovery_180d"].mean()
            tier4 = df_cal[df_cal["raw_score"] < threshold]["recovery_180d"].mean()
            results[threshold] = {
                "above_threshold_n":             int((df_cal["raw_score"] >= threshold).sum()),
                "above_threshold_recovery_rate": round(float(tier1), 4) if not pd.isna(tier1) else None,
                "below_threshold_n":             int((df_cal["raw_score"] < threshold).sum()),
                "below_threshold_recovery_rate": round(float(tier4), 4) if not pd.isna(tier4) else None,
                "recovery_rate_gap":             round(float(tier1 - tier4), 4)
                                                 if not (pd.isna(tier1) or pd.isna(tier4)) else None,
            }

        # Recommend threshold where recovery gap is largest
        best_t1 = max(
            (t for t in results if results[t]["recovery_rate_gap"] is not None),
            key=lambda t: results[t]["recovery_rate_gap"],
            default=8,
        )
        logger.info(
            f"Calibration suggests TIER_1 threshold at score >= {best_t1} | "
            f"recovery gap = {results[best_t1]['recovery_rate_gap']:.1%}"
        )
        return {
            "threshold_analysis": results,
            "recommended_tier1_threshold": best_t1,
        }

    # ── Rule-based scoring ────────────────────────────────────────────────────

    def _score_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the four-dimension scoring rubric to every account.
        Returns a DataFrame with all score components, tier, and subtype.
        """
        results = []
        for _, row in df.iterrows():
            d1, d1_drivers = self._trajectory_score(row)
            d2, d2_drivers = self._payment_history_score(row)
            d3, d3_drivers = self._bureau_score(row)
            d4, d4_drivers = self._post_co_score(row)

            total = d1 + d2 + d3 + d4

            # ── Dormancy override ─────────────────────────────────────────────
            months_since_co = float(row.get("months_since_co", 0) or 0)
            any_payment     = int(row.get("any_payment_post_co", 0) or 0)
            dormant_override = (
                months_since_co >= self.dormancy_months
                and total < self.dormancy_score_cap
                and not any_payment
            )

            tier    = self._assign_tier(total, dormant_override)
            subtype = self._assign_subtype(tier, d1, d2, d3, d4, row)
            drivers = " | ".join(filter(None, d1_drivers + d2_drivers + d3_drivers + d4_drivers))

            results.append({
                "recoverability_tier":    tier,
                "recoverability_score":    int(total),
                "recov_dim1_trajectory":   int(d1),
                "recov_dim2_payment":      int(d2),
                "recov_dim3_bureau":       int(d3),
                "recov_dim4_post_co":      int(d4),
                "recoverability_subtype": subtype,
                "recoverability_drivers": drivers,
            })

        return pd.DataFrame(results, index=df.index)

    def _trajectory_score(self, row) -> tuple:
        """
        Dimension 1: DPD trajectory before charge-off.
        Sudden shock = temporary distress = more recoverable than chronic.
        Returns (score: int 0-3, drivers: list[str]).
        """
        dpd_minus_3m  = float(row.get("dpd_at_minus_3m",  180) or 180)
        dpd_minus_6m  = float(row.get("dpd_at_minus_6m",  180) or 180)
        dpd_minus_12m = float(row.get("dpd_at_minus_12m", 180) or 180)
        months_dlq    = float(row.get("months_first_delinquent", 24) or 24)
        cured         = int(row.get("ever_partially_cured", 0) or 0)
        last_pay_mo   = float(row.get("last_payment_months_before_co", 99) or 99)

        score   = 0
        drivers = []

        # Sudden shock: was relatively current 6 months before CO
        if dpd_minus_6m < 60:
            score   += 3
            drivers += ["sudden_shock_onset"]
        elif dpd_minus_6m < 90 or (dpd_minus_12m < 60 and dpd_minus_6m < 120):
            score   += 2
            drivers += ["semi_rapid_onset"]
        elif cured or last_pay_mo <= 6:
            score   += 1
            drivers += ["some_payment_engagement"]
        # else: chronic (score += 0)

        # Cap at 3
        score = min(score, 3)

        if score == 0:
            drivers += ["chronic_delinquency"]

        return score, drivers

    def _payment_history_score(self, row) -> tuple:
        """
        Dimension 2: Payment history on THIS account pre-charge-off.
        Did they engage with their own obligation before CO?
        Returns (score: int 0-3, drivers: list[str]).
        """
        last_pay_mo    = float(row.get("last_payment_months_before_co", 99) or 99)
        decay_ratio    = float(row.get("payment_decay_ratio", 0.0) or 0.0)
        payment_count  = float(row.get("payment_count_last_12m_pre_co", 0) or 0)
        cured          = int(row.get("ever_partially_cured", 0) or 0)
        avg_payment    = float(row.get("avg_payment_amount_last_6m", 0) or 0)

        score   = 0
        drivers = []

        if last_pay_mo <= 3 and decay_ratio > 0.3:
            score   += 3
            drivers += ["recent_payment_active"]
        elif last_pay_mo <= 6 or cured:
            score   += 2
            if last_pay_mo <= 6:
                drivers += ["payment_within_6m_of_co"]
            if cured:
                drivers += ["partial_cure_history"]
        elif payment_count >= 3 or avg_payment > 0:
            score   += 1
            drivers += ["some_pre_co_payments"]

        score = min(score, 3)

        if score == 0:
            drivers += ["no_payment_history"]

        return score, drivers

    def _bureau_score(self, row) -> tuple:
        """
        Dimension 3: External bureau structural capacity.
        Are they paying other lenders? Most predictive for long-term recoverability.

        High score here = capacity exists despite this default.
        Could be strategic defaulter (paying mortgage but ignoring card) OR
        genuinely isolated distress (this card only).
        Both are MORE recoverable than systemic impairment.

        Returns (score: int 0-3, drivers: list[str]).
        """
        other_current  = float(row.get("ncb_other_accounts_current",    0) or 0)
        other_dlq      = float(row.get("ncb_accounts_delinquent_count",  0) or 0)
        total_dlq_pct  = float(row.get("ncb_total_delinquent_pct",      1.0) or 1.0)
        mortgage_curr  = int(row.get("ncb_mortgage_current",     0) or 0)
        secured_curr   = int(row.get("ncb_secured_loan_current", 0) or 0)
        score_trend    = float(row.get("ncb_score_trend_6m",     0) or 0)
        new_credit     = int(row.get("ncb_new_credit_post_co",   0) or 0)

        score   = 0
        drivers = []

        # Mortgage or secured loan current = very strong capacity signal
        if mortgage_curr or secured_curr:
            score   += 3
            drivers += ["secured_loan_current"]
            if mortgage_curr:
                drivers += ["mortgage_current_strategic_signal"]
        elif other_current > 0 and total_dlq_pct < 0.5:
            score   += 2
            drivers += ["bureau_accounts_current"]
        elif other_current > 0 or total_dlq_pct < 0.8:
            score   += 1
            drivers += ["partial_bureau_capacity"]

        # Additional signals (bonus, doesn't increase dim beyond cap)
        if score_trend > 20 and score < 3:
            score   = min(score + 1, 3)
            drivers += ["bureau_score_trend_positive"]
        if new_credit and score < 3:
            score   = min(score + 1, 3)
            drivers += ["new_credit_approved_post_co"]

        score = min(score, 3)

        if score == 0:
            drivers += ["systemic_bureau_delinquency"]

        return score, drivers

    def _post_co_score(self, row) -> tuple:
        """
        Dimension 4: Post-charge-off engagement.
        Even one small payment after CO changes the entire prognosis.
        Returns (score: int 0-3, drivers: list[str]).
        """
        any_payment      = int(row.get("any_payment_post_co",       0) or 0)
        contact_resp     = int(row.get("contact_response_post_co",  0) or 0)
        agency_resp      = int(row.get("agency_response_flag",      0) or 0)
        months_since_co  = float(row.get("months_since_co",         0) or 0)

        score   = 0
        drivers = []

        if any_payment:
            score   += 3
            drivers += ["payment_received_post_co"]
        elif contact_resp:
            score   += 2
            drivers += ["contact_responded_post_co"]
        elif agency_resp:
            score   += 1
            drivers += ["agency_engagement"]
        elif months_since_co < 3:
            # Too early to have post-CO signals — don't penalise
            score   += 1
            drivers += ["fresh_co_benefit_of_doubt"]

        score = min(score, 3)

        if score == 0 and months_since_co >= 3:
            drivers += ["complete_silence_post_co"]

        return score, drivers

    # ── Tier + subtype assignment ─────────────────────────────────────────────

    def _assign_tier(self, score: int, dormant_override: bool) -> str:
        """Assigns RECOVERABILITY_TIER from total score and override flag."""
        if dormant_override:
            return "TIER_4_DORMANT"

        t1 = self.tier_thresholds.get("TIER_1_HIGH",     8)
        t2 = self.tier_thresholds.get("TIER_2_MODERATE", 5)
        t3 = self.tier_thresholds.get("TIER_3_LOW",      2)

        if score >= t1:
            return "TIER_1_HIGH"
        if score >= t2:
            return "TIER_2_MODERATE"
        if score >= t3:
            return "TIER_3_LOW"
        return "TIER_4_DORMANT"

    def _assign_subtype(
        self, tier: str,
        d1: int, d2: int, d3: int, d4: int,
        row,
    ) -> str:
        """
        Identifies the primary recovery driver within a tier.
        This guides the TACTICAL approach (settlement structure, channel, tone).
        """
        if tier == "TIER_1_HIGH":
            if d4 >= 3:
                return "ACTIVE_ENGAGER"        # already responding post-CO
            if d3 >= 3:
                return "STRATEGIC_DEFAULTER"   # paying other lenders
            if d1 >= 3:
                return "SHOCK_RECOVERY"        # sudden onset
            if d2 >= 3:
                return "STRONG_HISTORY"        # paid consistently until close to CO
            return "COMPOSITE_HIGH"            # no single dominant driver

        if tier == "TIER_2_MODERATE":
            if d3 >= 2:
                return "BUREAU_MODERATE"       # some external capacity
            if d1 >= 2:
                return "GRADUAL_DECLINE"       # slow trajectory, some hope
            return "SILENT_MODERATE"           # moderate score, not engaging yet

        if tier == "TIER_3_LOW":
            if d3 == 0:
                return "DEEP_DISTRESS"         # systemic bureau delinquency
            return "CHRONIC_IMPAIRED"          # long delinquency history

        # TIER_4_DORMANT
        months_since_co = float(row.get("months_since_co", 0) or 0)
        if months_since_co >= self.dormancy_months:
            return "AGED_DORMANT"              # time override triggered
        return "CONFIRMED_IMPAIRED"            # all dimensions zero

    # ── Model-based scoring ───────────────────────────────────────────────────

    def _score_model(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Model-based scoring using fitted LightGBM.

        Tier and total score come from the model probability.
        Dimension scores and subtype come from rule-based scoring (interpretability layer).
        recoverability_score is the model probability re-scaled to 0–12 to keep
        the same range as the rule-based score — label it clearly in reporting.

        Falls back to rule-based if model fails.
        """
        try:
            X, _ = self._prepare_model_features(df, feature_cols=self._feature_cols)
            model_proba = self._model.predict_proba(X)[:, 1]  # P(recovery_180d=1)

            # Map model probability to tier using fixed portfolio thresholds
            # (NOT percentile-based — percentile gives fixed tier sizes regardless
            #  of actual recoverability, which defeats the purpose of the model).
            # Thresholds are in probability space; set via calibrate_thresholds().
            t1_prob = 0.65   # P >= 0.65 → TIER_1_HIGH
            t2_prob = 0.40   # P >= 0.40 → TIER_2_MODERATE
            t3_prob = 0.20   # P >= 0.20 → TIER_3_LOW
            # P <  0.20 → TIER_4_DORMANT

            def _prob_to_tier(p: float) -> str:
                if p >= t1_prob:
                    return "TIER_1_HIGH"
                if p >= t2_prob:
                    return "TIER_2_MODERATE"
                if p >= t3_prob:
                    return "TIER_3_LOW"
                return "TIER_4_DORMANT"

            # Rule-based dim scores for interpretability (subtype, drivers, dim1–4)
            rule_df = self._score_rules(df)

            # Override tier and total score with model output
            rule_df["recoverability_tier"]  = [_prob_to_tier(p) for p in model_proba]
            rule_df["recoverability_score"]  = (model_proba * 12).round().astype(int)

            return rule_df

        except Exception as e:
            logger.warning(
                f"Model-based scoring failed: {e} — falling back to rule-based scoring."
            )
            return self._score_rules(df)

    def _prepare_model_features(
        self, df: pd.DataFrame, feature_cols: list = None
    ):
        """Prepares feature matrix for model training/inference."""
        available = feature_cols or [
            c for c in RECOVERABILITY_FEATURES if c in df.columns
        ]
        X = df[available].copy() if available else pd.DataFrame(index=df.index)

        # Impute missing — domain-meaningful defaults
        fill_map = {
            "dpd_at_minus_3m":             180,
            "dpd_at_minus_6m":             180,
            "dpd_at_minus_12m":            180,
            "months_first_delinquent":      24,
            "last_payment_months_before_co": 99,
            "payment_decay_ratio":           0.0,
            "payment_count_last_12m_pre_co": 0,
            "ncb_total_delinquent_pct":      1.0,
            "months_since_co":               0,
        }
        for col in X.columns:
            X[col] = X[col].fillna(fill_map.get(col, 0))

        # Encode binary flags as float
        for col in ["ever_partially_cured", "ncb_mortgage_current",
                    "ncb_secured_loan_current", "any_payment_post_co",
                    "contact_response_post_co", "agency_response_flag",
                    "ncb_new_credit_post_co"]:
            if col in X.columns:
                X[col] = X[col].astype(float)

        return X, available

    def _try_load_model(self) -> None:
        """Loads fitted LightGBM model if present."""
        model_path = self.model_dir / "recoverability_model.pkl"
        if model_path.exists():
            try:
                obj              = joblib.load(model_path)
                self._model      = obj["model"]
                self._feature_cols = obj["feature_cols"]
                logger.info(f"Recoverability model loaded from {model_path}")
            except Exception as e:
                self._model = None
                logger.warning(
                    f"Could not load recoverability model: {e} — "
                    "using rule-based scoring."
                )
        else:
            logger.warning(
                f"No recoverability model at {model_path} — "
                "using rule-based scoring. "
                "Run RecoverabilitySegmenter.fit(df_with_outcomes) to train."
            )

    def save(self, model_dir: Path) -> None:
        """Saves fitted model to model_dir."""
        if self._model is None:
            raise RuntimeError("No model fitted. Call fit() first.")
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model": self._model, "feature_cols": self._feature_cols},
            model_dir / "recoverability_model.pkl"
        )
        logger.info(f"Recoverability model saved → {model_dir}")
