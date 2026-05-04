"""
Pipeline Adapter
================
Bridges CollectionsFeaturePipeline output to the two model consumers:

  1. RecoveryScorecard  — recovery P(pay) + expected amount model
  2. ModelTrainer (NBA) — S-Learner / X-Learner uplift model

Both consumers internally build features from raw account rows.
This adapter lets them accept *pre-built* feature matrices from the
pipeline instead, eliminating duplicate computation.

Usage — RecoveryScorecard
--------------------------
    from decision_agent.features.collections_feature_pipeline import (
        CollectionsFeaturePipeline, PipelineConfig
    )
    from decision_agent.features.pipeline_adapter import ScorecardAdapter
    from decision_agent.recovery.recovery_scorecard import RecoveryScorecard, ScorecardConfig

    pipeline   = CollectionsFeaturePipeline(PipelineConfig())
    train_res  = pipeline.run(card_df_train, customer_df, actions_df, settlements_df, snap_train)
    val_res    = pipeline.run(card_df_val,   customer_df, actions_df, settlements_df, snap_val)

    adapter    = ScorecardAdapter(RecoveryScorecard())
    adapter.fit_from_pipeline(train_res, val_res, train_labels, val_labels)

    # Scoring on new snapshot
    snap_res   = pipeline.run(card_df_now, customer_df, actions_df, settlements_df, today)
    scores     = adapter.score_from_pipeline(snap_res)

Usage — ModelTrainer (NBA)
---------------------------
    from decision_agent.features.pipeline_adapter import NBAAdapter
    from decision_agent.nba_builder.model_trainer import ModelTrainer

    adapter = NBAAdapter(ModelTrainer())
    adapter.train_from_pipeline(train_res, val_res, train_actions_labels, val_actions_labels)
    scores  = adapter.score_from_pipeline(snap_res, actions=["SMS", "AGENT_CALL", "NO_ACTION"])

Label DataFrames
-----------------
Both adapters expect a ``labels_df`` with at minimum:
  - ``account_id``           (join key)
  - ``outcome_pay_any``      (binary 0/1)
  - ``outcome_pay_amount``   (float ≥ 0)

NBAAdapter additionally requires:
  - ``action``               (str: action taken at time of label)
"""

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _merge_labels(features_df: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join pipeline features with outcome labels on account_id.
    Rows without labels are dropped with a warning.
    """
    merged = features_df.merge(labels_df, on="account_id", how="inner")
    dropped = len(features_df) - len(merged)
    if dropped:
        logger.warning(
            "pipeline_adapter: %d accounts dropped — no matching label (inner join on account_id)",
            dropped,
        )
    return merged


def _drop_non_feature_cols(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Return df with only account_id + feature_cols present."""
    keep = ["account_id"] + [c for c in feature_cols if c in df.columns]
    return df[keep]


def _numeric_feature_cols(df: pd.DataFrame, feature_cols: List[str]) -> List[str]:
    """
    Filter feature_cols to only those that are numeric in df.
    String/object columns (e.g. delinquency_history raw string, stage)
    cannot be passed directly into sklearn models.
    """
    numeric_types = {"int8","int16","int32","int64","float32","float64",
                     "uint8","uint16","uint32","uint64"}
    return [
        c for c in feature_cols
        if c in df.columns and df[c].dtype.name in numeric_types
    ]


# ──────────────────────────────────────────────────────────────────────────────
# SCORECARD ADAPTER
# ──────────────────────────────────────────────────────────────────────────────

class ScorecardAdapter:
    """
    Wraps RecoveryScorecard so it can consume PipelineResult objects
    instead of raw account DataFrames.

    The adapter bypasses RecoveryScorecard's internal RecoveryFeatureBuilder
    and injects the pre-built pipeline features directly into the model fit /
    score path.
    """

    def __init__(self, scorecard):
        """
        Args:
            scorecard: RecoveryScorecard instance (already constructed with config)
        """
        self.sc = scorecard

    # ── Training ──────────────────────────────────────────────────────────────

    def fit_from_pipeline(
        self,
        train_result,           # PipelineResult
        val_result,             # PipelineResult
        train_labels: pd.DataFrame,
        val_labels:   pd.DataFrame,
    ):
        """
        Fit the RecoveryScorecard using pre-built pipeline features.

        Args:
            train_result:   PipelineResult from pipeline.run() on training snapshot
            val_result:     PipelineResult from pipeline.run() on validation snapshot
            train_labels:   DataFrame with [account_id, outcome_pay_any, outcome_pay_amount]
            val_labels:     Same schema as train_labels

        Returns:
            self  (for chaining)
        """
        train_df = _merge_labels(train_result.features_df, train_labels)
        val_df   = _merge_labels(val_result.features_df,   val_labels)

        if len(train_df) == 0:
            raise ValueError("No training rows after label join — check account_id alignment.")
        if len(val_df) == 0:
            raise ValueError("No validation rows after label join — check account_id alignment.")

        # Only numeric columns can be fed to sklearn models
        feature_cols = _numeric_feature_cols(train_df, train_result.feature_cols)

        logger.info(
            "ScorecardAdapter.fit_from_pipeline: train=%d, val=%d, numeric_features=%d",
            len(train_df), len(val_df), len(feature_cols),
        )

        # Monkey-patch feature_builder.build to be a no-op identity,
        # and supply the numeric feature_cols directly.
        self.sc.feature_builder.feature_cols = feature_cols
        self.sc.feature_builder.build = lambda df: df   # bypass internal feature engineering

        self.sc.fit(train_df, val_df)

        # Pin the feature_cols from pipeline (scorecard.fit may reset them)
        self.sc.feature_cols = feature_cols

        return self

    # ── Scoring ───────────────────────────────────────────────────────────────

    def score_from_pipeline(
        self,
        snap_result,                    # PipelineResult
        return_features: bool = False,
    ) -> pd.DataFrame:
        """
        Score accounts using a pipeline snapshot result.

        Args:
            snap_result:      PipelineResult from pipeline.run() on current snapshot
            return_features:  If True, append feature columns to output

        Returns:
            DataFrame: account_id, score (0-10), band, p_recovery, expected_amount
        """
        if self.sc.recovery_model is None:
            raise RuntimeError("Model not trained. Call fit_from_pipeline() first.")

        feat_df = snap_result.features_df.copy()

        # Bypass internal feature builder for scoring; keep only numeric cols
        numeric_cols = _numeric_feature_cols(feat_df, self.sc.feature_cols)
        self.sc.feature_builder.build = lambda df: df
        self.sc.feature_builder.feature_cols = numeric_cols

        scores = self.sc.score(feat_df, return_features=return_features)
        return scores

    # ── Delegate everything else ──────────────────────────────────────────────

    def __getattr__(self, name):
        """Delegate any other method/attribute to the wrapped scorecard."""
        return getattr(self.sc, name)


# ──────────────────────────────────────────────────────────────────────────────
# NBA ADAPTER
# ──────────────────────────────────────────────────────────────────────────────

class NBAAdapter:
    """
    Wraps ModelTrainer (NBA) so it accepts PipelineResult objects directly.

    The ModelTrainer uses an S-Learner: action is encoded as a feature column
    alongside all account features. Labels must include which action was taken
    historically (``action`` column).
    """

    def __init__(self, trainer):
        """
        Args:
            trainer: ModelTrainer instance
        """
        self.trainer = trainer

    # ── Training ──────────────────────────────────────────────────────────────

    def train_from_pipeline(
        self,
        train_result,               # PipelineResult
        val_result,                 # PipelineResult
        train_labels: pd.DataFrame,
        val_labels:   pd.DataFrame,
        config: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Train the NBA model using pre-built pipeline features.

        Args:
            train_result:   PipelineResult from training snapshot
            val_result:     PipelineResult from validation snapshot
            train_labels:   DataFrame with [account_id, action, outcome_pay_any, outcome_pay_amount]
            val_labels:     Same schema as train_labels
            config:         Optional hyperparameter overrides (passed to ModelTrainer.train)

        Returns:
            Training report dict from ModelTrainer.train()
        """
        feature_cols = train_result.feature_cols

        train_df = _merge_labels(train_result.features_df, train_labels)
        val_df   = _merge_labels(val_result.features_df,   val_labels)

        if len(train_df) == 0:
            raise ValueError("No training rows after label join.")
        if len(val_df) == 0:
            raise ValueError("No validation rows after label join.")

        # Validate required label columns
        for col in ("action", "outcome_pay_any", "outcome_pay_amount"):
            if col not in train_df.columns:
                raise ValueError(f"train_labels must contain '{col}' column.")

        available_features = _numeric_feature_cols(train_df, feature_cols)
        missing = set(feature_cols) - set(available_features)
        if missing:
            logger.warning(
                "NBAAdapter: %d feature(s) excluded (non-numeric or missing): %s",
                len(missing), sorted(missing),
            )

        logger.info(
            "NBAAdapter.train_from_pipeline: train=%d, val=%d, features=%d",
            len(train_df), len(val_df), len(available_features),
        )

        report = self.trainer.train(train_df, val_df, available_features, config)
        return report

    # ── Scoring ───────────────────────────────────────────────────────────────

    def score_from_pipeline(
        self,
        snap_result,
        actions: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Score all accounts for all actions using pipeline snapshot.

        Args:
            snap_result:  PipelineResult from current snapshot
            actions:      Action list to score (default: all trained actions)

        Returns:
            DataFrame with columns: account_id, action, pay_any_score, amount_score, uplift
        """
        if self.trainer.pay_any_model is None:
            raise RuntimeError("Model not trained. Call train_from_pipeline() first.")

        feat_df = snap_result.features_df.copy()
        feature_cols = [c for c in self.trainer.feature_cols if c in feat_df.columns]

        scores = self.trainer.score_all_actions(feat_df, feature_cols, actions=actions)
        return scores

    # ── Delegate ──────────────────────────────────────────────────────────────

    def __getattr__(self, name):
        return getattr(self.trainer, name)


# ──────────────────────────────────────────────────────────────────────────────
# LABEL BUILDER HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def build_labels_from_outcomes(
    accounts_df: pd.DataFrame,
    outcome_window_days: int = 90,
    snapshot_date: Optional[Any] = None,
    payment_col: str = "last_payment_amount",
    payment_date_col: str = "last_payment_date",
    account_id_col: str = "account_id",
) -> pd.DataFrame:
    """
    Derive outcome labels from payment history columns in the pipeline output.

    This is a *proxy labelling* approach for when you don't have a separate
    outcomes table. Real production training should use a properly labelled
    dataset with forward-looking outcomes.

    Args:
        accounts_df:        Pipeline feature DataFrame (features_df from PipelineResult)
        outcome_window_days: Look-ahead window used to define 'paid within window'
        snapshot_date:      Reference date (uses 'data_date' col if None)
        payment_col:        Column name for last payment amount
        payment_date_col:   Column name for last payment date
        account_id_col:     Account identifier column

    Returns:
        DataFrame: [account_id, outcome_pay_any, outcome_pay_amount]

    Note:
        outcome_pay_any  = 1 if last_payment within outcome_window_days of snapshot
        outcome_pay_amount = last_payment_amount if paid, else 0
        This is a training proxy — replace with proper forward-looking labels
        from your outcomes table in production.
    """
    import pandas as pd
    from datetime import date, timedelta

    df = accounts_df.copy()

    if snapshot_date is None and "data_date" in df.columns:
        snap = pd.to_datetime(df["data_date"]).dt.date.iloc[0]
    elif snapshot_date is not None:
        snap = pd.to_datetime(snapshot_date).date() if not isinstance(snapshot_date, date) else snapshot_date
    else:
        snap = date.today()

    cutoff = snap - timedelta(days=outcome_window_days)

    labels = pd.DataFrame({account_id_col: df[account_id_col]})

    if payment_date_col in df.columns and payment_col in df.columns:
        pay_date = pd.to_datetime(df[payment_date_col], errors="coerce").dt.date
        pay_amt  = pd.to_numeric(df[payment_col], errors="coerce").fillna(0)

        paid_in_window = (pay_date >= cutoff) & (pay_date <= snap)

        labels["outcome_pay_any"]    = paid_in_window.astype(int).values
        labels["outcome_pay_amount"] = (pay_amt * paid_in_window).values
    else:
        logger.warning(
            "build_labels_from_outcomes: payment columns not found — defaulting outcomes to 0. "
            "Provide a proper outcomes table for real training."
        )
        labels["outcome_pay_any"]    = 0
        labels["outcome_pay_amount"] = 0.0

    return labels.reset_index(drop=True)
