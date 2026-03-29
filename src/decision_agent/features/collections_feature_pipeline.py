"""
Collections Feature Pipeline
==============================
Single entry point that takes raw CDX DataFrames (T1-T5) and produces
a model-ready, point-in-time safe feature matrix.

Input:  5 raw tables (post schema_mapper rename, or raw CDX columns)
Output: One flat DataFrame, one row per account, all features filled

Stages:
  1. Schema mapping     — rename CDX columns → internal names
  2. Account base       — required fields (DPD, balance, stage, bill_day, etc.)
  3. Delinquency        — DLNQ_HIST + AMT_IN_ARRS_* features (25 features)
  4. Billing cycle      — CC/SPC due-date cycle features (19 features)
  5. Action aggregation — outbound_calls, PTP, response rates (35 features)
  6. Demographics       — age, occupation, channel availability (8 features)
  7. Feature matrix     — join all, fill nulls, validate, return

Compatible with:
  - ModelTrainer (NBA): pass feature_cols list directly
  - RecoveryScorecard:  pass output DataFrame directly (replaces RecoveryFeatureBuilder)
  - TDR OfferGenerator: pass single enriched row via .enrich_one()

Point-in-time safety:
  - snapshot_date is threaded through every stage
  - Collection action windows are computed relative to snapshot_date
  - Billing cycle due dates computed at snapshot_date
  - No future data bleeds in (leakage guard checks can be run externally)
"""

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..data.schema_mapper import SchemaMapper, STALE_FIELDS
from .delinquency_features import DelinquencyFeatureBuilder
from .billing_cycle_features import BillingCycleCalculator, BillingCycleConfig
from .collection_action_aggregator import CollectionActionAggregator, ActionAggregatorConfig
from .bureau_features import BureauFeatureBuilder, BureauConfig, BUREAU_FEATURE_COLS

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    """
    Controls which feature groups are enabled and their sub-configs.
    Disable groups that aren't needed for a particular use case.
    """
    enable_delinquency:   bool = True
    enable_billing_cycle: bool = True
    enable_actions:       bool = True
    enable_demographics:  bool = True
    enable_bureau:        bool = True

    billing_config:  BillingCycleConfig    = field(default_factory=BillingCycleConfig)
    action_config:   ActionAggregatorConfig = field(default_factory=ActionAggregatorConfig)
    bureau_config:   BureauConfig          = field(default_factory=BureauConfig)

    # Null-fill strategy per dtype
    fill_numeric_with: float = 0.0
    fill_flag_with:    float = 0.0

    # Whether input DataFrames have CDX source column names (True)
    # or already have internal names post schema_mapper (False)
    raw_cdx_input: bool = True


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    """Full output of one pipeline run."""
    features_df:      pd.DataFrame    # model-ready feature matrix
    feature_cols:     List[str]       # ordered feature column names (excl. account_id / metadata)
    snapshot_date:    date
    account_count:    int
    missing_summary:  pd.DataFrame    # per-column null counts before fill
    stage_counts:     Dict[str, int]  # SM/NPL/CHARGEOFF distribution
    warnings:         List[str]


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

class CollectionsFeaturePipeline:
    """
    Orchestrates all feature engineering stages for the collections platform.

    Usage:
        pipeline = CollectionsFeaturePipeline()

        result = pipeline.run(
            card_df       = spark_df.toPandas(),   # T2: crcard_card_dly
            customer_df   = cust_df.toPandas(),    # T1: cust_prfl_dly
            actions_df    = actions_df,            # T4: efs_cax_extract_actions
            settlements_df= settlements_df,        # T5: efs_cax_extract_settlements
            snapshot_date = date(2026, 2, 9),
        )

        # Feed into NBA model
        trainer.train(result.features_df, result.feature_cols)

        # Feed into Recovery Scorecard
        scorecard.score(result.features_df)

        # Single account (on-demand TDR scoring)
        enriched = pipeline.enrich_one(account_id, card_df, ...)
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.cfg     = config or PipelineConfig()
        self.mapper  = SchemaMapper()
        self.dlnq    = DelinquencyFeatureBuilder()
        self.billing = BillingCycleCalculator(self.cfg.billing_config)
        self.actions = CollectionActionAggregator(self.cfg.action_config)

    # ── PUBLIC ────────────────────────────────────────────────────────────────

    def run(
        self,
        card_df:        pd.DataFrame,              # T2 — required
        customer_df:    Optional[pd.DataFrame] = None,  # T1 — optional
        actions_df:     Optional[pd.DataFrame] = None,  # T4 — optional
        settlements_df: Optional[pd.DataFrame] = None,  # T5 — optional
        snapshot_date:  Optional[date]         = None,
        bureau_df:      Optional[pd.DataFrame] = None,  # T6 NCB — optional
    ) -> PipelineResult:
        """
        Run the full feature pipeline. Returns PipelineResult.
        Gracefully handles missing optional tables.
        """
        snap     = snapshot_date or date.today()
        warnings = []

        logger.info("Collections feature pipeline starting — snapshot=%s", snap)

        # ── Stage 1: Schema mapping ───────────────────────────────────────────
        card_df        = self._map(card_df,        "card")
        customer_df    = self._map(customer_df,    "customer")
        actions_df     = self._map(actions_df,     "actions")
        settlements_df = self._map(settlements_df, "settlements")

        if customer_df is None:
            warnings.append("T1 customer_df not provided — demographics features will be empty")
        if actions_df is None:
            warnings.append("T4 actions_df not provided — contact features will be zero-filled")
        if settlements_df is None:
            warnings.append("T5 settlements_df not provided — PTP features will be zero-filled")

        # ── Stage 2: Base account features ───────────────────────────────────
        base_df = self._build_base(card_df, customer_df, snap)
        logger.info("  Base: %d accounts", len(base_df))

        # ── Stage 3: Delinquency features ────────────────────────────────────
        if self.cfg.enable_delinquency:
            dlnq_df = self.dlnq.extract_batch(base_df)
            dlnq_df = dlnq_df.drop(columns=["account_id"])
        else:
            dlnq_df = pd.DataFrame(index=base_df.index)

        # ── Stage 4: Billing cycle features ──────────────────────────────────
        if self.cfg.enable_billing_cycle:
            bill_df = self.billing.extract_batch(base_df)
            # Drop ordinal date cols — not useful as raw model features
            bill_df = bill_df.drop(columns=[
                c for c in bill_df.columns
                if "ordinal" in c or c == "account_id"
            ])
        else:
            bill_df = pd.DataFrame(index=base_df.index)

        # ── Stage 5: Action aggregation ───────────────────────────────────────
        if self.cfg.enable_actions and (
            actions_df is not None or settlements_df is not None
        ):
            act_df = self.actions.aggregate(
                actions_df     = actions_df     if actions_df     is not None else pd.DataFrame(),
                settlements_df = settlements_df if settlements_df is not None else pd.DataFrame(),
                snapshot_date  = snap,
                account_ids    = base_df["account_id"].tolist(),
            )
            if not act_df.empty:
                act_df = act_df.set_index("account_id").reindex(base_df["account_id"]).reset_index()
                act_df = act_df.drop(columns=["account_id"])
            else:
                act_df = pd.DataFrame(
                    self.actions._zero_action_features() | self.actions._zero_ptp_features(),
                    index=base_df.index
                )
        else:
            act_df = pd.DataFrame(
                {**self.actions._zero_action_features(), **self.actions._zero_ptp_features()},
                index=base_df.index
            )

        # ── Stage 6: Demographics features ────────────────────────────────────
        if self.cfg.enable_demographics and customer_df is not None:
            demo_df = self._build_demographics(base_df, customer_df)
        else:
            demo_df = self._empty_demographics(base_df)

        # ── Stage 6b: Bureau features ─────────────────────────────────────────
        if self.cfg.enable_bureau and bureau_df is not None:
            bur_builder = BureauFeatureBuilder(self.cfg.bureau_config)
            customer_id_col = "m_token" if "m_token" in base_df.columns else "account_id"
            bur_feat = bur_builder.build(bureau_df, base_df, customer_id_col=customer_id_col)
            bur_feat = bur_feat.drop(
                columns=[self.cfg.bureau_config.customer_id_col], errors="ignore"
            ).reset_index(drop=True)
        else:
            bur_feat = self._empty_bureau(base_df)

        # ── Stage 7: Assemble feature matrix ──────────────────────────────────
        feat_df = pd.concat(
            [base_df.reset_index(drop=True),
             bur_feat,
             dlnq_df.reset_index(drop=True),
             bill_df.reset_index(drop=True),
             act_df.reset_index(drop=True),
             demo_df.reset_index(drop=True)],
            axis=1,
        )

        # ── Stage 8: Null summary + fill ──────────────────────────────────────
        meta_cols = ["account_id", "stage", "data_date", "days_past_due", "balance"]
        feature_cols = [c for c in feat_df.columns if c not in meta_cols]

        missing_summary = self._null_summary(feat_df[feature_cols])
        if missing_summary["null_pct"].max() > 50:
            high_null = missing_summary[missing_summary["null_pct"] > 50]["column"].tolist()
            warnings.append(f"High null rate (>50%) in: {high_null[:5]}")

        feat_df[feature_cols] = feat_df[feature_cols].fillna(self.cfg.fill_numeric_with)

        stage_counts = feat_df["stage"].value_counts().to_dict() if "stage" in feat_df.columns else {}

        logger.info(
            "  Pipeline complete: %d accounts, %d features, stages=%s",
            len(feat_df), len(feature_cols), stage_counts,
        )

        return PipelineResult(
            features_df   = feat_df,
            feature_cols  = feature_cols,
            snapshot_date = snap,
            account_count = len(feat_df),
            missing_summary = missing_summary,
            stage_counts  = stage_counts,
            warnings      = warnings,
        )

    def enrich_one(
        self,
        account_id:     str,
        card_df:        pd.DataFrame,
        customer_df:    Optional[pd.DataFrame] = None,
        actions_df:     Optional[pd.DataFrame] = None,
        settlements_df: Optional[pd.DataFrame] = None,
        snapshot_date:  Optional[date]         = None,
    ) -> pd.Series:
        """
        Run pipeline for a single account. Returns enriched Series.
        Used for on-demand TDR / offer engine scoring.
        """
        # Filter to just this account
        def _filter(df, col):
            if df is None:
                return None
            return df[df[col].astype(str) == str(account_id)] if col in df.columns else None

        card_filt  = _filter(card_df,        "CARD_NUM" if self.cfg.raw_cdx_input else "account_id")
        cust_filt  = _filter(customer_df,    "CUST_NUM" if self.cfg.raw_cdx_input else "customer_id")
        act_filt   = _filter(actions_df,     "ACCOUNT_NUMBER" if self.cfg.raw_cdx_input else "action_account_id")
        ptp_filt   = _filter(settlements_df, "ACCOUNT_NUMBER" if self.cfg.raw_cdx_input else "ptp_account_id")

        if card_filt is None or card_filt.empty:
            raise ValueError(f"account_id {account_id} not found in card_df")

        result = self.run(card_filt, cust_filt, act_filt, ptp_filt, snapshot_date)
        rows   = result.features_df[result.features_df["account_id"].astype(str) == str(account_id)]

        if rows.empty:
            raise ValueError(f"No features produced for account_id {account_id}")

        return rows.iloc[0]

    # ── PRIVATE — STAGE BUILDERS ──────────────────────────────────────────────

    def _map(self, df: Optional[pd.DataFrame], label: str) -> Optional[pd.DataFrame]:
        """Apply schema mapper if raw CDX input."""
        if df is None:
            return None
        if self.cfg.raw_cdx_input:
            return self.mapper.to_internal(df)
        return df

    def _build_base(
        self,
        card_df:     pd.DataFrame,
        customer_df: Optional[pd.DataFrame],
        snap:        date,
    ) -> pd.DataFrame:
        """
        Build the base account DataFrame with core fields from T2 (+T1 join keys).
        One row per account.
        """
        base_cols = [
            "account_id", "balance", "principal_balance", "days_past_due",
            "stage", "credit_limit", "available_credit", "over_limit",
            "bill_day", "interest_rate", "data_date",
            "delinquency_history", "delinquency_number",
            "tdr_flag", "tdr_date", "tdr_instalment_amount", "tdr_instalment_count",
            "tdr_interest_rate", "payment_default_dt_pre_tdr",
            "last_payment_date", "last_payment_amount",
            "total_paid_since_cutoff",
            "pdue_30", "pdue_60", "pdue_90", "pdue_120", "pdue_180",
            "arrs_period_1", "arrs_period_2", "arrs_period_3",
            "arrs_period_4", "arrs_period_5", "arrs_period_6",
            "arrs_period_7", "arrs_period_8", "arrs_period_9",
            "write_off_date", "charge_off_date",
            "stop_from_dt", "stop_to_dt", "stop_reason_code",
            "product_code",   # mapped from PROD if available
            "m_token",        # join key for demographics (T1)
        ]

        # Keep only columns that exist in this DataFrame
        keep = [c for c in base_cols if c in card_df.columns]
        base = card_df[keep].copy()

        # Derive snapshot date if not in data
        if "data_date" not in base.columns:
            base["data_date"] = str(snap)

        # Derive utilisation
        if "balance" in base.columns and "credit_limit" in base.columns:
            base["utilisation_rate"] = (
                base["balance"] / base["credit_limit"].replace(0, np.nan)
            ).clip(0, 2).fillna(0)

        # Derive days_since_last_payment from LAST_PYMT_DT
        if "last_payment_date" in base.columns:
            base["days_since_last_payment"] = (
                pd.to_datetime(snap) - pd.to_datetime(base["last_payment_date"], errors="coerce")
            ).dt.days.fillna(999)

        # Approximate tdr_new_loan_amount from instalment × count
        if ("tdr_instalment_amount" in base.columns
                and "tdr_instalment_count" in base.columns):
            base["tdr_new_loan_amount"] = (
                base["tdr_instalment_amount"].fillna(0)
                * base["tdr_instalment_count"].fillna(0)
            )

        # Join customer_id from T1 if available
        if customer_df is not None and "m_token" in customer_df.columns:
            if "m_token" in base.columns:
                cust_keys = customer_df[["m_token", "customer_id"]].drop_duplicates("m_token")
                base = base.merge(cust_keys, on="m_token", how="left")

        return base.reset_index(drop=True)

    def _build_demographics(
        self, base_df: pd.DataFrame, customer_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Build demographics features from T1.
        Joined on m_token or customer_id.
        Returns DataFrame aligned to base_df index.
        """
        demo_cols = [
            "m_token", "customer_id",
            "date_of_birth", "gender", "occupation_group",
            "province", "customer_segment",
            "mktg_suppress_phone", "mktg_suppress_sms",
            "mktg_suppress_email", "mktg_suppress_line",
            "cardx_staff_flag",
        ]
        keep  = [c for c in demo_cols if c in customer_df.columns]
        cust  = customer_df[keep].drop_duplicates(
            subset=["m_token"] if "m_token" in keep else ["customer_id"]
        ).copy()

        # Age feature
        if "date_of_birth" in cust.columns:
            cust["age_years"] = (
                (pd.Timestamp.today() - pd.to_datetime(cust["date_of_birth"], errors="coerce"))
                .dt.days / 365.25
            ).fillna(0).astype(int)

        # Gender encode
        if "gender" in cust.columns:
            cust["gender_male_flag"] = (cust["gender"].str.upper() == "M").astype(float)

        # Channel availability flags (1 = channel open, 0 = suppressed)
        for ch_flag, col in [
            ("phone_available", "mktg_suppress_phone"),
            ("sms_available",   "mktg_suppress_sms"),
            ("email_available", "mktg_suppress_email"),
            ("line_available",  "mktg_suppress_line"),
        ]:
            if col in cust.columns:
                cust[ch_flag] = (cust[col].fillna(0).astype(int) == 0).astype(float)
            else:
                cust[ch_flag] = 1.0    # assume available if flag missing

        demo_feat_cols = [
            "age_years", "gender_male_flag",
            "phone_available", "sms_available", "email_available", "line_available",
            "cardx_staff_flag",
        ]
        demo_feat_cols = [c for c in demo_feat_cols if c in cust.columns]

        # Join to base on m_token
        join_key = "m_token" if ("m_token" in base_df.columns and "m_token" in cust.columns) else None

        if join_key:
            merged = base_df[["account_id", join_key]].merge(
                cust[[join_key] + demo_feat_cols],
                on=join_key,
                how="left",
            )
        elif "customer_id" in base_df.columns and "customer_id" in cust.columns:
            merged = base_df[["account_id", "customer_id"]].merge(
                cust[["customer_id"] + demo_feat_cols],
                on="customer_id",
                how="left",
            )
        else:
            return self._empty_demographics(base_df)

        return merged[demo_feat_cols].reset_index(drop=True)

    def _empty_demographics(self, base_df: pd.DataFrame) -> pd.DataFrame:
        cols = [
            "age_years", "gender_male_flag",
            "phone_available", "sms_available", "email_available", "line_available",
            "cardx_staff_flag",
        ]
        return pd.DataFrame(
            {c: 0.0 for c in cols},
            index=range(len(base_df)),
        )

    def _empty_bureau(self, base_df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {c: 0.0 for c in BUREAU_FEATURE_COLS},
            index=range(len(base_df)),
        )

    @staticmethod
    def _null_summary(df: pd.DataFrame) -> pd.DataFrame:
        null_counts = df.isnull().sum()
        return pd.DataFrame({
            "column":   null_counts.index,
            "null_count": null_counts.values,
            "null_pct": (null_counts.values / max(len(df), 1) * 100).round(1),
        }).query("null_count > 0").sort_values("null_pct", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# SMOKE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from datetime import date

    SNAP = date(2026, 2, 9)

    # ── Synthetic T2 (crcard_card_dly) — using internal names directly ────────
    card_df = pd.DataFrame([
        {
            "account_id": "ACC001", "balance": 45000, "principal_balance": 42000,
            "days_past_due": 60, "stage": "NPL", "credit_limit": 80000,
            "available_credit": 0, "bill_day": 5, "interest_rate": 0.18,
            "data_date": str(SNAP), "product_code": "CC",
            "delinquency_history": "000000001123",
            "arrs_period_1": 15000, "arrs_period_2": 12000, "arrs_period_3": 9000,
            "arrs_period_4": 5000,  "arrs_period_5": 2000,  "arrs_period_6": 0,
            "arrs_period_7": 0,     "arrs_period_8": 0,     "arrs_period_9": 0,
            "last_payment_amount": 3000, "last_payment_date": "2025-12-01",
            "tdr_flag": 0, "tdr_instalment_amount": None, "tdr_instalment_count": None,
            "pdue_30": 0, "pdue_60": 15000, "pdue_90": 0,
            "m_token": "TOK001",
        },
        {
            "account_id": "ACC002", "balance": 120000, "principal_balance": 115000,
            "days_past_due": 180, "stage": "CHARGEOFF", "credit_limit": 150000,
            "available_credit": 0, "bill_day": 20, "interest_rate": 0.20,
            "data_date": str(SNAP), "product_code": "SPC",
            "delinquency_history": "000001234567X",
            "arrs_period_1": 120000, "arrs_period_2": 115000, "arrs_period_3": 108000,
            "arrs_period_4": 98000,  "arrs_period_5": 85000,  "arrs_period_6": 65000,
            "arrs_period_7": 40000,  "arrs_period_8": 15000,  "arrs_period_9": 0,
            "last_payment_amount": 2000, "last_payment_date": "2025-06-15",
            "tdr_flag": 0, "tdr_instalment_amount": None, "tdr_instalment_count": None,
            "pdue_30": 0, "pdue_60": 0, "pdue_90": 0, "pdue_120": 0, "pdue_180": 120000,
            "m_token": "TOK002",
        },
        {
            "account_id": "ACC003", "balance": 8000, "principal_balance": 7500,
            "days_past_due": 35, "stage": "SM", "credit_limit": 50000,
            "available_credit": 42000, "bill_day": 15, "interest_rate": 0.15,
            "data_date": str(SNAP), "product_code": "CC",
            "delinquency_history": "000000000001",
            "arrs_period_1": 8000, "arrs_period_2": 0, "arrs_period_3": 0,
            "arrs_period_4": 0,    "arrs_period_5": 0, "arrs_period_6": 0,
            "arrs_period_7": 0,    "arrs_period_8": 0, "arrs_period_9": 0,
            "last_payment_amount": 5000, "last_payment_date": "2026-01-20",
            "tdr_flag": 0, "tdr_instalment_amount": None, "tdr_instalment_count": None,
            "pdue_30": 8000, "pdue_60": 0, "pdue_90": 0,
            "m_token": "TOK003",
        },
    ])

    # ── Synthetic T1 (cust_prfl_dly) ─────────────────────────────────────────
    customer_df = pd.DataFrame([
        {"m_token": "TOK001", "customer_id": "C001", "date_of_birth": "1985-03-10",
         "gender": "M", "occupation_group": "EMPLOYED", "province": "BKK",
         "customer_segment": "MASS", "cardx_staff_flag": 0,
         "mktg_suppress_sms": 0, "mktg_suppress_phone": 0,
         "mktg_suppress_email": 0, "mktg_suppress_line": 0},
        {"m_token": "TOK002", "customer_id": "C002", "date_of_birth": "1978-07-22",
         "gender": "F", "occupation_group": "SELF_EMPLOYED", "province": "CNX",
         "customer_segment": "AFFLUENT", "cardx_staff_flag": 0,
         "mktg_suppress_sms": 1, "mktg_suppress_phone": 0,
         "mktg_suppress_email": 0, "mktg_suppress_line": 1},
        {"m_token": "TOK003", "customer_id": "C003", "date_of_birth": "1992-11-05",
         "gender": "M", "occupation_group": "EMPLOYED", "province": "BKK",
         "customer_segment": "MASS", "cardx_staff_flag": 0,
         "mktg_suppress_sms": 0, "mktg_suppress_phone": 0,
         "mktg_suppress_email": 0, "mktg_suppress_line": 0},
    ])

    # ── Synthetic T4 (actions) ────────────────────────────────────────────────
    actions_df = pd.DataFrame([
        {"action_account_id": "ACC001", "action_category": "CALL", "action_result": "CONNECTED",  "action_date": "2026-02-07", "action_actor": "AGT01"},
        {"action_account_id": "ACC001", "action_category": "CALL", "action_result": "NO_ANSWER",  "action_date": "2026-02-09", "action_actor": "AGT01"},
        {"action_account_id": "ACC001", "action_category": "SMS",  "action_result": "RESPONDED",  "action_date": "2026-02-03", "action_actor": "SYS"},
        {"action_account_id": "ACC002", "action_category": "CALL", "action_result": "NO_ANSWER",  "action_date": "2026-02-08", "action_actor": "AGT02"},
        {"action_account_id": "ACC002", "action_category": "CALL", "action_result": "REFUSED",    "action_date": "2026-01-20", "action_actor": "AGT02"},
        {"action_account_id": "ACC002", "action_category": "FIELD_VISIT", "action_result": "CONNECTED", "action_date": "2026-01-10", "action_actor": "FLD01"},
        {"action_account_id": "ACC003", "action_category": "SMS",  "action_result": "NO_ANSWER",  "action_date": "2026-02-05", "action_actor": "SYS"},
    ])

    # ── Synthetic T5 (settlements) ────────────────────────────────────────────
    settlements_df = pd.DataFrame([
        {"ptp_account_id": "ACC001", "promise_status": "KEPT",   "ptp_date": "2026-02-05", "ptp_amount": 3000},
        {"ptp_account_id": "ACC001", "promise_status": "BROKEN", "ptp_date": "2025-12-10", "ptp_amount": 5000},
        {"ptp_account_id": "ACC002", "promise_status": "BROKEN", "ptp_date": "2026-01-15", "ptp_amount": 10000},
    ])

    # ── Run pipeline (internal names — raw_cdx_input=False) ───────────────────
    pipeline = CollectionsFeaturePipeline(PipelineConfig(raw_cdx_input=False))
    result   = pipeline.run(card_df, customer_df, actions_df, settlements_df, SNAP)

    print(f"\nAccounts: {result.account_count}  |  Features: {len(result.feature_cols)}")
    print(f"Stage mix: {result.stage_counts}")
    if result.warnings:
        print(f"Warnings: {result.warnings}")

    # Key feature groups summary
    print("\n── DELINQUENCY FEATURES (sample) ──")
    dlnq_cols = ["account_id", "dlnq_max_bucket_12m", "dlnq_current_streak",
                 "dlnq_worsening_count_12m", "arrs_trend_3m", "arrs_acceleration"]
    print(result.features_df[dlnq_cols].to_string(index=False))

    print("\n── BILLING CYCLE FEATURES (sample) ──")
    bill_cols = ["account_id", "days_past_cc_due", "days_past_spc_due",
                 "cycle_phase_cc", "cycles_missed_estimate", "payment_due_soon_flag"]
    print(result.features_df[bill_cols].to_string(index=False))

    print("\n── CONTACT FEATURES (sample) ──")
    act_cols = ["account_id", "outbound_calls_made", "calls_connected",
                "call_response_rate", "ptp_kept_rate", "escalation_flag",
                "voice_calls_today", "voice_calls_week"]
    print(result.features_df[act_cols].to_string(index=False))

    print("\n── DEMOGRAPHICS FEATURES ──")
    demo_cols = ["account_id", "age_years", "gender_male_flag",
                 "sms_available", "line_available"]
    print(result.features_df[demo_cols].to_string(index=False))

    print("\n── NULL SUMMARY (before fill) ──")
    if result.missing_summary.empty:
        print("  No nulls — all features complete")
    else:
        print(result.missing_summary.to_string(index=False))

    print(f"\n── FEATURE COLS (first 20 of {len(result.feature_cols)}) ──")
    print(result.feature_cols[:20])
