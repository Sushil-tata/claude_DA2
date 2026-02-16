"""
Recovery Scorecard
==================
Two-part model for debt recovery prediction.

  Part 1 — P(recovery):   P(customer makes any payment within outcome_window days)
  Part 2 — E(amount):     Expected recovery amount given payment (log-normal regression)

Score banding maps probability to strategy tier:
  HOT    (score 8-10): High recovery likelihood → priority outreach
  WARM   (score 5-7):  Medium likelihood → standard collections
  COLD   (score 2-4):  Low likelihood → low-cost channels only
  FROZEN (score 0-1):  Very low → hold / legal / write-off review

Schema-independent: column names passed via config — no hardcoding.
Wire actual column names when schema is available.

Usage:
    sc = RecoveryScorecard(config)
    sc.fit(train_df, val_df)
    scores = sc.score(accounts_df)        # DataFrame: account_id, score, band, p_recovery, expected_amount
    print(sc.scorecard_report())
"""

import logging
import os
import pickle
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    mean_absolute_error, mean_squared_error,
)
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ScorecardConfig:
    """
    All column mappings and hyperparameters.
    Fill with actual column names when schema is provided.
    """

    # ── Identity ─────────────────────────────────────────────────────────────
    account_id_col: str         = "account_id"
    product_col: str            = "product"         # CC | SPC

    # ── Delinquency ──────────────────────────────────────────────────────────
    dpd_col: str                = "days_past_due"
    balance_col: str            = "balance"
    bucket_col: str             = "bucket"          # 0-5 or B0-B5
    stage_col: str              = "stage"           # NPL | CHARGEOFF | SM etc.

    # ── Payment history ───────────────────────────────────────────────────────
    last_payment_amount_col: str    = "last_payment_amount"
    last_payment_date_col: str      = "last_payment_date"
    payment_count_12m_col: str      = "payment_count_12m"
    total_paid_12m_col: str         = "total_paid_12m"

    # ── Contact / response ────────────────────────────────────────────────────
    total_contacts_col: str         = "total_contacts_90d"
    response_rate_col: str          = "response_rate_90d"
    last_contact_date_col: str      = "last_contact_date"
    promise_kept_rate_col: str      = "promise_kept_rate_6m"

    # ── Settlement history ────────────────────────────────────────────────────
    settlement_offered_col: str     = "settlement_offered"
    settlement_accepted_col: str    = "settlement_accepted"
    active_settlement_col: str      = "active_settlement"

    # ── Legal / flags ─────────────────────────────────────────────────────────
    legal_flag_col: str             = "legal_flag"
    deceased_col: str               = "deceased_flag"
    bankruptcy_col: str             = "bankruptcy_flag"

    # ── Outcome (training only) ───────────────────────────────────────────────
    outcome_pay_any_col: str        = "outcome_pay_any"       # binary
    outcome_pay_amount_col: str     = "outcome_pay_amount"    # continuous
    outcome_window_days: int        = 90                      # prediction horizon

    # ── Score banding ─────────────────────────────────────────────────────────
    score_bands: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "HOT":    (0.70, 1.00),
        "WARM":   (0.40, 0.70),
        "COLD":   (0.15, 0.40),
        "FROZEN": (0.00, 0.15),
    })

    # ── Model hyperparameters ─────────────────────────────────────────────────
    n_estimators_clf: int  = 150
    max_depth_clf: int     = 4
    learning_rate: float   = 0.05
    n_estimators_reg: int  = 100
    max_depth_reg: int     = 4

    # ── MLflow ────────────────────────────────────────────────────────────────
    mlflow_tracking_uri: str   = "mlruns"
    experiment_name: str       = "recovery_scorecard"


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE GROUPS (schema-independent logic — column names from config)
# ─────────────────────────────────────────────────────────────────────────────

class RecoveryFeatureBuilder:
    """
    Builds feature matrix from raw account data.
    All column references come from ScorecardConfig — no hardcoding.
    """

    def __init__(self, config: ScorecardConfig):
        self.config = config
        self.feature_cols: List[str] = []

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out = self._delinquency_features(out)
        out = self._payment_features(out)
        out = self._contact_features(out)
        out = self._settlement_features(out)
        out = self._capacity_features(out)
        out = self._fill_defaults(out)
        self.feature_cols = self._get_feature_cols(out)
        return out

    def _delinquency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        c = self.config
        if c.dpd_col in df.columns:
            dpd = df[c.dpd_col].clip(lower=0)
            df["dpd_norm"]        = (dpd / 720.0).clip(upper=1.0)   # normalise to 2yr max
            df["log_dpd"]         = np.log1p(dpd)
            df["is_npl"]          = (dpd.between(91, 180)).astype(int)
            df["is_chargeoff"]    = (dpd > 180).astype(int)
            df["is_sm"]           = (dpd.between(31, 90)).astype(int)
            df["dpd_bucket"]      = pd.cut(dpd,
                                           bins=[-1, 0, 30, 60, 90, 180, 10000],
                                           labels=[0, 1, 2, 3, 4, 5]).astype(float)
        if c.balance_col in df.columns:
            bal = df[c.balance_col].clip(lower=0)
            df["log_balance"]     = np.log1p(bal)
            df["balance_tier"]    = pd.cut(bal,
                                           bins=[-1, 999, 4999, 9999, 49999, 1e9],
                                           labels=[0, 1, 2, 3, 4]).astype(float)
        return df

    def _payment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        c = self.config
        if c.last_payment_amount_col in df.columns:
            df["log_last_payment"] = np.log1p(df[c.last_payment_amount_col].clip(lower=0))

        if c.last_payment_amount_col in df.columns and c.balance_col in df.columns:
            bal = df[c.balance_col].replace(0, np.nan)
            df["last_pay_to_balance"] = (
                df[c.last_payment_amount_col] / bal
            ).clip(0, 2).fillna(0)

        if c.payment_count_12m_col in df.columns:
            df["has_paid_12m"]    = (df[c.payment_count_12m_col] > 0).astype(int)
            df["payment_freq_12m"] = df[c.payment_count_12m_col].clip(0, 24) / 24.0

        if c.total_paid_12m_col in df.columns and c.balance_col in df.columns:
            bal = df[c.balance_col].replace(0, np.nan)
            df["recovery_rate_12m"] = (
                df[c.total_paid_12m_col] / bal
            ).clip(0, 2).fillna(0)

        if c.last_payment_date_col in df.columns:
            try:
                last_pay = pd.to_datetime(df[c.last_payment_date_col], errors="coerce")
                today    = pd.Timestamp.today()
                df["days_since_last_payment"] = (today - last_pay).dt.days.clip(0, 730).fillna(730)
                df["days_since_pay_norm"]     = df["days_since_last_payment"] / 730.0
                df["paid_in_last_90d"]        = (df["days_since_last_payment"] <= 90).astype(int)
                df["paid_in_last_180d"]       = (df["days_since_last_payment"] <= 180).astype(int)
            except Exception:
                pass
        return df

    def _contact_features(self, df: pd.DataFrame) -> pd.DataFrame:
        c = self.config
        if c.total_contacts_col in df.columns:
            df["contacts_norm"]   = (df[c.total_contacts_col].clip(0, 30) / 30.0)
            df["high_contact"]    = (df[c.total_contacts_col] > 10).astype(int)

        if c.response_rate_col in df.columns:
            df["responsive"]      = (df[c.response_rate_col] > 0.2).astype(int)

        if c.promise_kept_rate_col in df.columns:
            df["reliable_payer"]  = (df[c.promise_kept_rate_col] > 0.6).astype(int)

        if c.last_contact_date_col in df.columns:
            try:
                last_contact = pd.to_datetime(df[c.last_contact_date_col], errors="coerce")
                today        = pd.Timestamp.today()
                df["days_since_contact"] = (today - last_contact).dt.days.clip(0, 365).fillna(365)
                df["fresh_contact"]      = (df["days_since_contact"] <= 7).astype(int)
            except Exception:
                pass
        return df

    def _settlement_features(self, df: pd.DataFrame) -> pd.DataFrame:
        c = self.config
        if c.settlement_offered_col in df.columns and c.settlement_accepted_col in df.columns:
            offered  = df[c.settlement_offered_col].replace(0, np.nan)
            df["settlement_accept_rate"] = (
                df[c.settlement_accepted_col] / offered
            ).clip(0, 1).fillna(0)
            df["high_settlement_accept"] = (df["settlement_accept_rate"] > 0.5).astype(int)

        if c.active_settlement_col in df.columns:
            df["active_settlement"] = df[c.active_settlement_col].fillna(0).astype(int)
        return df

    def _capacity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Exclude legal/deceased — they are hard constraints, not features
        # But include flags as binary features for the model
        c = self.config
        for flag_col in [c.legal_flag_col, c.deceased_col, c.bankruptcy_col]:
            if flag_col in df.columns:
                df[f"flag_{flag_col}"] = df[flag_col].fillna(0).astype(int)
        return df

    def _fill_defaults(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric = df.select_dtypes(include=[np.number]).columns
        df[numeric] = df[numeric].fillna(0)
        return df

    def _get_feature_cols(self, df: pd.DataFrame) -> List[str]:
        exclude = {
            self.config.account_id_col,
            self.config.product_col,
            self.config.outcome_pay_any_col,
            self.config.outcome_pay_amount_col,
            self.config.last_payment_date_col,
            self.config.last_contact_date_col,
            "business_date", "split",
        }
        return [
            c for c in df.columns
            if c not in exclude
            and df[c].dtype in [np.float64, np.float32, np.int64, np.int32, np.int8, bool]
        ]


# ─────────────────────────────────────────────────────────────────────────────
# SCORECARD
# ─────────────────────────────────────────────────────────────────────────────

class RecoveryScorecard:
    """
    Two-part recovery scorecard.

      Part 1: GradientBoostingClassifier → P(any payment in outcome_window days)
      Part 2: GradientBoostingRegressor  → E(amount | paid) via log(1+y)

    Final score = P(recovery) mapped to 0-10 band.
    """

    def __init__(self, config: Optional[ScorecardConfig] = None):
        self.config         = config or ScorecardConfig()
        self.feature_builder = RecoveryFeatureBuilder(self.config)
        self.recovery_model: Optional[GradientBoostingClassifier] = None
        self.amount_model:   Optional[GradientBoostingRegressor]  = None
        self.feature_cols:   List[str] = []
        self._metrics:       Dict[str, float] = {}
        self._try_setup_mlflow()

    # ── PUBLIC ────────────────────────────────────────────────────────────────

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df:   pd.DataFrame,
    ) -> "RecoveryScorecard":
        """
        Train both models. DataFrames must contain outcome columns.
        Column names read from ScorecardConfig.
        """
        c = self.config

        print("Building recovery features...")
        train_feat = self.feature_builder.build(train_df)
        val_feat   = self.feature_builder.build(val_df)
        self.feature_cols = self.feature_builder.feature_cols

        X_train = train_feat[self.feature_cols].values.astype(float)
        X_val   = val_feat[self.feature_cols].values.astype(float)
        y_pay_train = train_df[c.outcome_pay_any_col].values
        y_pay_val   = val_df[c.outcome_pay_any_col].values
        y_amt_train = train_df[c.outcome_pay_amount_col].values
        y_amt_val   = val_df[c.outcome_pay_amount_col].values

        # ── Part 1: Recovery probability ──────────────────────────────────
        print(f"  Training recovery model ({len(X_train):,} samples)...")
        pos_weight     = (y_pay_train == 0).sum() / max((y_pay_train == 1).sum(), 1)
        sample_weights = np.where(y_pay_train == 1, pos_weight, 1.0)

        self.recovery_model = GradientBoostingClassifier(
            n_estimators=c.n_estimators_clf,
            max_depth=c.max_depth_clf,
            learning_rate=c.learning_rate,
            subsample=0.8,
            random_state=42,
        )
        self.recovery_model.fit(X_train, y_pay_train, sample_weight=sample_weights)

        p_recovery_val = self.recovery_model.predict_proba(X_val)[:, 1]
        auc   = roc_auc_score(y_pay_val, p_recovery_val) if len(np.unique(y_pay_val)) > 1 else float("nan")
        pr_auc = average_precision_score(y_pay_val, p_recovery_val)
        print(f"  Recovery model: AUC={auc:.4f}, PR-AUC={pr_auc:.4f}")

        # ── Part 2: Expected amount ────────────────────────────────────────
        print("  Training amount model...")
        self.amount_model = GradientBoostingRegressor(
            n_estimators=c.n_estimators_reg,
            max_depth=c.max_depth_reg,
            learning_rate=c.learning_rate,
            loss="squared_error",
            subsample=0.8,
            random_state=42,
        )
        self.amount_model.fit(X_train, np.log1p(y_amt_train))

        amt_pred_val = np.expm1(self.amount_model.predict(X_val)).clip(0)
        mae  = mean_absolute_error(y_amt_val, amt_pred_val)
        rmse = mean_squared_error(y_amt_val, amt_pred_val) ** 0.5
        print(f"  Amount model: MAE=${mae:.2f}, RMSE=${rmse:.2f}")

        self._metrics = {
            "recovery_auc":    round(auc, 4),
            "recovery_pr_auc": round(pr_auc, 4),
            "amount_mae":      round(mae, 2),
            "amount_rmse":     round(rmse, 2),
        }

        # Feature importance
        feat_names = self.feature_cols
        self._recovery_importance = dict(zip(
            feat_names, self.recovery_model.feature_importances_
        ))
        top5 = sorted(self._recovery_importance.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"  Top recovery features: {[(f, round(v,3)) for f,v in top5]}")

        self._log_to_mlflow()
        return self

    def score(
        self,
        accounts_df: pd.DataFrame,
        return_features: bool = False,
    ) -> pd.DataFrame:
        """
        Score accounts. Returns DataFrame with:
          account_id, p_recovery, expected_amount, score_0_10, band, strategy_tier
        """
        if self.recovery_model is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        feat_df = self.feature_builder.build(accounts_df)
        available = [c for c in self.feature_cols if c in feat_df.columns]
        missing   = [c for c in self.feature_cols if c not in feat_df.columns]

        X = feat_df[available].values.astype(float)
        if missing:
            X = np.column_stack([X, np.zeros((len(X), len(missing)))])

        p_recovery      = self.recovery_model.predict_proba(X)[:, 1]
        expected_amount = np.expm1(self.amount_model.predict(X)).clip(0) * p_recovery

        score_0_10 = np.round(p_recovery * 10).astype(int).clip(0, 10)
        bands      = pd.Series(p_recovery).apply(self._assign_band).values

        result = pd.DataFrame({
            self.config.account_id_col: accounts_df[self.config.account_id_col].values,
            "p_recovery":               np.round(p_recovery, 4),
            "expected_amount":          np.round(expected_amount, 2),
            "score_0_10":               score_0_10,
            "band":                     bands,
            "strategy_tier":            pd.Series(bands).map(self._band_to_strategy()).values,
        })

        if return_features:
            result = pd.concat([result, feat_df[available].reset_index(drop=True)], axis=1)

        return result

    def score_by_segment(
        self, accounts_df: pd.DataFrame, segment_col: str
    ) -> pd.DataFrame:
        """Score and summarise by segment (e.g. bucket, product)."""
        scores = self.score(accounts_df)
        combined = pd.concat([
            accounts_df[[self.config.account_id_col, segment_col]].reset_index(drop=True),
            scores.drop(columns=[self.config.account_id_col]),
        ], axis=1)

        return (
            combined.groupby(segment_col)
            .agg(
                n=("p_recovery", "count"),
                avg_p_recovery=("p_recovery", "mean"),
                median_p_recovery=("p_recovery", "median"),
                avg_expected_amount=("expected_amount", "mean"),
                total_expected_amount=("expected_amount", "sum"),
                pct_hot=("band", lambda x: (x == "HOT").mean()),
                pct_frozen=("band", lambda x: (x == "FROZEN").mean()),
            )
            .round(4)
            .reset_index()
        )

    def scorecard_report(self) -> str:
        """Print-friendly scorecard summary."""
        lines = [
            "=" * 55,
            "  RECOVERY SCORECARD REPORT",
            "=" * 55,
            "",
            "MODEL PERFORMANCE",
            f"  Recovery AUC     : {self._metrics.get('recovery_auc', 'n/a')}",
            f"  Recovery PR-AUC  : {self._metrics.get('recovery_pr_auc', 'n/a')}",
            f"  Amount MAE       : ${self._metrics.get('amount_mae', 'n/a')}",
            f"  Amount RMSE      : ${self._metrics.get('amount_rmse', 'n/a')}",
            "",
            "SCORE BANDS",
        ]
        for band, (lo, hi) in self.config.score_bands.items():
            strategy = self._band_to_strategy().get(band, "")
            lines.append(f"  {band:<8} P(recovery) {lo:.0%}–{hi:.0%}  →  {strategy}")

        if hasattr(self, "_recovery_importance"):
            lines += ["", "TOP RECOVERY FEATURES"]
            top = sorted(self._recovery_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            for feat, imp in top:
                lines.append(f"  {feat:<35} {imp:.4f}")

        lines += ["", "=" * 55]
        return "\n".join(lines)

    def save(self, model_dir: str) -> None:
        """Save models and config to disk."""
        import json
        os.makedirs(model_dir, exist_ok=True)
        with open(os.path.join(model_dir, "recovery_model.pkl"), "wb") as f:
            pickle.dump(self.recovery_model, f)
        with open(os.path.join(model_dir, "amount_model.pkl"), "wb") as f:
            pickle.dump(self.amount_model, f)
        with open(os.path.join(model_dir, "meta.json"), "w") as f:
            json.dump({
                "feature_cols": self.feature_cols,
                "metrics":      self._metrics,
            }, f, indent=2)
        logger.info("RecoveryScorecard saved to %s", model_dir)

    def load(self, model_dir: str) -> "RecoveryScorecard":
        """Load models from disk."""
        import json
        with open(os.path.join(model_dir, "recovery_model.pkl"), "rb") as f:
            self.recovery_model = pickle.load(f)
        with open(os.path.join(model_dir, "amount_model.pkl"), "rb") as f:
            self.amount_model = pickle.load(f)
        with open(os.path.join(model_dir, "meta.json")) as f:
            meta = json.load(f)
        self.feature_cols = meta["feature_cols"]
        self._metrics     = meta["metrics"]
        return self

    # ── PRIVATE ───────────────────────────────────────────────────────────────

    def _assign_band(self, p: float) -> str:
        for band, (lo, hi) in self.config.score_bands.items():
            if lo <= p <= hi:
                return band
        return "FROZEN"

    @staticmethod
    def _band_to_strategy() -> Dict[str, str]:
        return {
            "HOT":    "Priority outreach — AGENT_CALL / SETTLEMENT_ONE_TIME",
            "WARM":   "Standard collections — SMS / LINE / VOICE_IVR",
            "COLD":   "Low-cost only — EMAIL / SMS (no calls)",
            "FROZEN": "Hold — review for legal / write-off",
        }

    def _try_setup_mlflow(self) -> None:
        try:
            import mlflow
            mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
            mlflow.set_experiment(self.config.experiment_name)
            self._mlflow = mlflow
            self._mlflow_available = True
        except ImportError:
            self._mlflow_available = False

    def _log_to_mlflow(self) -> None:
        if not self._mlflow_available:
            return
        try:
            with self._mlflow.start_run(run_name="recovery_scorecard"):
                self._mlflow.log_params({
                    "n_estimators_clf": self.config.n_estimators_clf,
                    "max_depth_clf":    self.config.max_depth_clf,
                    "n_estimators_reg": self.config.n_estimators_reg,
                    "max_depth_reg":    self.config.max_depth_reg,
                    "learning_rate":    self.config.learning_rate,
                    "outcome_window":   self.config.outcome_window_days,
                })
                self._mlflow.log_metrics(self._metrics)
                import mlflow.sklearn
                mlflow.sklearn.log_model(self.recovery_model, "recovery_model")
                mlflow.sklearn.log_model(self.amount_model,   "amount_model")
        except Exception as e:
            logger.warning("MLflow logging failed: %s", e)
