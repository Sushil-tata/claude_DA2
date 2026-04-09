"""
PropensityAgent
===============
Task 1 in the agentic NBA pipeline. Runs before DataQualityAgent.

Reads:
  Enterprise feature table (cdx_mdz_prd.feature_store) — 100+ features
  recovery.model_scores (OPTIONAL) — if recovery-engine-v2 has already scored

Responsibilities:
  - Scores all delinquent accounts for propensity at three horizons:
      propensity_30d  (P_1M) — tactical actions: DIGITAL_NUDGE, AGENT_CALL
      propensity_90d  (P_3M) — intermediate monitoring signal
      propensity_180d (P_6M) — strategic actions: AGENCY, LEGAL
  - Uses PropensityModelTrainer (LightGBM per horizon, optionally per segment)
  - If no fitted models exist: falls back to scores from recovery-engine-v2
    (reads existing recovery.model_scores and passes them through unchanged)
  - Sets low_confidence_flag where model is uncertain (score near 0.5)
  - Writes to recovery.model_scores (overwrites for execution_date partition)

Writes:
  recovery.model_scores

──────────────────────────────────────────────────────────────────────────────
CONFIGURATION INSTRUCTIONS
──────────────────────────────────────────────────────────────────────────────
PROPENSITY_MODEL_DIR
    Path to trained PropensityModelTrainer .pkl files.
    Default: models/propensity/
    Set PROPENSITY_MODEL_DIR env variable.

    In Databricks:
        PROPENSITY_MODEL_DIR = /dbfs/FileStore/models/propensity/

ENTERPRISE_FEATURE_TABLE
    Fully-qualified Delta table name for enterprise features.
    Must match what FeatureAgent uses.
    Set ENTERPRISE_FEATURE_TABLE env variable.
    e.g. cdx_mdz_prd.feature_store

ENTERPRISE_FEATURE_JOIN_KEYS
    Join keys. Default: account_id,score_date
    Set ENTERPRISE_FEATURE_JOIN_KEYS env variable.

FALLBACK BEHAVIOUR
    If no PropensityModelTrainer models are found:
        - PropensityAgent reads recovery.model_scores written by recovery-engine-v2
        - Validates that propensity_30d and propensity_180d are present
        - Adds propensity_90d = (propensity_30d + propensity_180d) / 2 if missing
        - Proceeds without scoring — pipeline continues with upstream scores

    This means the pipeline is not blocked on Day 1 (cold start) even if
    PropensityModelTrainer has not been trained yet.

PER-SEGMENT MODELS
    Set per_segment=True to use segment-specific propensity models.
    Requires signal_segment to be derivable from enterprise features
    (bureau_pull_date, ncb_tradeline_count, propensity_30d).
    Default: False (global model per horizon).

LOW_CONFIDENCE_FLAG
    Set to True for accounts where the propensity score is uncertain.
    Threshold: abs(propensity_30d - 0.5) < LOW_CONFIDENCE_MARGIN.
    Default margin: 0.10 (scores between 0.40 and 0.60 flagged).
    DecisionAgent uses this flag to route borderline accounts to Claude.
──────────────────────────────────────────────────────────────────────────────
"""

import importlib.util as _ilu
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from agents.base_agent import AgentBlockedException, BaseAgent

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

DEFAULT_PROPENSITY_MODEL_DIR = Path(
    os.environ.get("PROPENSITY_MODEL_DIR", "models/propensity")
)

DEFAULT_ENTERPRISE_FEATURE_TABLE = os.environ.get("ENTERPRISE_FEATURE_TABLE", None)

_join_keys_env = os.environ.get("ENTERPRISE_FEATURE_JOIN_KEYS", "account_id,score_date")
DEFAULT_ENTERPRISE_JOIN_KEYS = [k.strip() for k in _join_keys_env.split(",")]

# Accounts with propensity_30d between (0.5 - margin) and (0.5 + margin)
# are flagged as low-confidence and routed to Claude in DecisionAgent
DEFAULT_LOW_CONFIDENCE_MARGIN = 0.10


class PropensityAgent(BaseAgent):

    def __init__(
        self,
        execution_date: str,
        memory,
        dry_run: bool = False,
        propensity_model_dir: Path = DEFAULT_PROPENSITY_MODEL_DIR,
        enterprise_feature_table: str = DEFAULT_ENTERPRISE_FEATURE_TABLE,
        enterprise_join_keys: list = None,
        per_segment: bool = False,
        low_confidence_margin: float = DEFAULT_LOW_CONFIDENCE_MARGIN,
        horizons: list = None,
    ):
        """
        Args:
            execution_date:           Scoring date (YYYY-MM-DD).
            memory:                   AgentMemory instance.
            dry_run:                  If True, skip all Delta writes.
            propensity_model_dir:     Trained PropensityModelTrainer .pkl files.
                                      Falls back to recovery-engine-v2 scores if empty.
                                      Set PROPENSITY_MODEL_DIR env var.
            enterprise_feature_table: Delta table with 100+ features.
                                      Set ENTERPRISE_FEATURE_TABLE env var.
            enterprise_join_keys:     Join keys. Default: [account_id, score_date].
            per_segment:              Use per-SIGNAL_SEGMENT models if True.
                                      Requires signal_segment derivable from features.
            low_confidence_margin:    Accounts within this margin of 0.5 on
                                      propensity_30d are flagged low_confidence.
                                      Default: 0.10 (flags 0.40–0.60 range).
            horizons:                 Horizons to score. Default: [30, 90, 180].
        """
        super().__init__(
            agent_name="propensity_agent",
            execution_date=execution_date,
            memory=memory,
            dry_run=dry_run,
        )
        self.propensity_model_dir    = Path(propensity_model_dir)
        self.enterprise_feature_table = enterprise_feature_table
        self.enterprise_join_keys    = enterprise_join_keys or DEFAULT_ENTERPRISE_JOIN_KEYS
        self.per_segment             = per_segment
        self.low_confidence_margin   = low_confidence_margin
        self.horizons                = horizons or [30, 90, 180]

        self._trainer = self._load_trainer()

    def _load_trainer(self):
        """Load PropensityModelTrainer if any fitted model exists."""
        any_model = any(
            (self.propensity_model_dir / f"{h}d_propensity_model.pkl").exists()
            for h in self.horizons
        )
        # Also check per-segment models
        if not any_model and self.per_segment:
            any_model = any(
                (self.propensity_model_dir / f"{seg}_{h}d_propensity_model.pkl").exists()
                for h in self.horizons for seg in ["A", "B", "C", "D"]
            )

        if any_model:
            try:
                _spec = _ilu.spec_from_file_location(
                    "propensity_model_trainer",
                    Path(__file__).parent.parent / "models" / "propensity_model_trainer.py",
                )
                _mod = _ilu.module_from_spec(_spec)
                _spec.loader.exec_module(_mod)
                trainer = _mod.PropensityModelTrainer(
                    horizons=self.horizons,
                    model_dir=self.propensity_model_dir,
                    per_segment=self.per_segment,
                )
                self.log(
                    f"PropensityModelTrainer loaded | "
                    f"horizons={self.horizons} | "
                    f"per_segment={self.per_segment}"
                )
                return trainer
            except Exception as e:
                self.log(
                    f"Could not load PropensityModelTrainer: {e} — "
                    "will use recovery-engine-v2 scores as fallback",
                    level="warning",
                )
                return None
        else:
            self.log(
                f"No fitted propensity models at {self.propensity_model_dir} — "
                "using recovery-engine-v2 scores as fallback. "
                "Run PropensityModelTrainer.fit() to activate in-pipeline scoring.",
                level="warning",
            )
            return None

    def execute(self) -> dict:

        if self._trainer is not None:
            # ── Model-based scoring ───────────────────────────────────────────
            # Read enterprise features → score → write model_scores
            self.log(
                f"Model-based propensity scoring | "
                f"horizons={self.horizons} | "
                f"feature_table={self.enterprise_feature_table or 'NOT SET'}"
            )
            df = self._read_enterprise_features()
            if df is None or df.empty:
                raise AgentBlockedException(
                    f"Enterprise feature table returned 0 rows for {self.execution_date}. "
                    f"Set ENTERPRISE_FEATURE_TABLE and ensure it is populated for this date. "
                    f"Current value: {self.enterprise_feature_table}",
                    retry_agent="propensity_agent",
                )

            scores_df = self._trainer.predict(df)

            # Merge scores back onto the account roster
            id_cols = [c for c in self.enterprise_join_keys if c in df.columns]
            output  = df[id_cols].copy()
            for col in scores_df.columns:
                output[col] = scores_df[col].values

            n_scored   = len(output)
            score_mode = "model"

        else:
            # ── Fallback: read existing recovery-engine-v2 scores ─────────────
            # Validate they exist and have required columns.
            self.log("Fallback mode — reading existing model_scores from recovery-engine-v2")
            output = self._read_existing_scores()
            if output is None or output.empty:
                raise AgentBlockedException(
                    "No propensity models fitted AND no existing model_scores found. "
                    "Either: (a) run PropensityModelTrainer.fit() and set PROPENSITY_MODEL_DIR, "
                    "or (b) ensure recovery-engine-v2 has written recovery.model_scores "
                    f"for execution_date={self.execution_date}.",
                    retry_agent="propensity_agent",
                )

            # Add propensity_90d if missing (interpolate between 30d and 180d)
            if "propensity_90d" not in output.columns:
                if "propensity_30d" in output.columns and "propensity_180d" in output.columns:
                    output["propensity_90d"] = (
                        output["propensity_30d"] * 0.4 +
                        output["propensity_180d"] * 0.6
                    ).clip(0.0, 1.0)
                    self.log(
                        "propensity_90d interpolated from 30d + 180d "
                        "(train PropensityModelTrainer for a direct 90d model)"
                    )

            n_scored   = len(output)
            score_mode = "fallback_upstream"

        # ── Set score_date ────────────────────────────────────────────────────
        output["score_date"] = self.execution_date

        # ── Low-confidence flag ───────────────────────────────────────────────
        # Accounts near the decision boundary (propensity_30d ≈ 0.5) are
        # uncertain — DecisionAgent routes these to Claude for edge-case reasoning.
        if "propensity_30d" in output.columns:
            output["low_confidence_flag"] = (
                (output["propensity_30d"] - 0.5).abs() < self.low_confidence_margin
            ).astype(int)
            n_low_conf = int(output["low_confidence_flag"].sum())
        else:
            output["low_confidence_flag"] = 0
            n_low_conf = 0

        # ── Score distribution summary ────────────────────────────────────────
        summary = {}
        for h in self.horizons:
            col = f"propensity_{h}d"
            if col in output.columns:
                summary[col] = {
                    "mean":   round(float(output[col].mean()), 4),
                    "pct_gt50": round(float((output[col] > 0.5).mean()), 4),
                    "null_pct": round(float(output[col].isna().mean()), 4),
                }

        if not self.dry_run:
            self.memory.write("model_scores", output)

        self.log(
            f"Propensity scoring complete | mode={score_mode} | "
            f"n_accounts={n_scored:,} | low_confidence={n_low_conf:,} | "
            f"summary={summary}"
        )
        return {
            "score_mode":       score_mode,
            "n_accounts_scored": n_scored,
            "n_low_confidence":  n_low_conf,
            "score_summary":     summary,
        }

    # ── Readers ───────────────────────────────────────────────────────────────

    def _read_enterprise_features(self) -> pd.DataFrame:
        """
        Reads enterprise feature table for execution_date.
        Tries Spark (Databricks) first, then pandas fallbacks.
        """
        if not self.enterprise_feature_table:
            self.log(
                "ENTERPRISE_FEATURE_TABLE not set. "
                "Cannot run model-based propensity scoring without features. "
                "Set ENTERPRISE_FEATURE_TABLE env var to your feature store table.",
                level="warning",
            )
            return None

        table = self.enterprise_feature_table
        date  = self.execution_date

        try:
            from pyspark.sql import SparkSession
            spark = SparkSession.getActiveSession()
            if spark is not None:
                schema_fields = [f.name for f in spark.table(table).schema]
                date_col = next(
                    (c for c in ["score_date", "feature_date", "snapshot_date"]
                     if c in schema_fields),
                    None,
                )
                sdf = spark.table(table)
                if date_col:
                    sdf = sdf.filter(f"{date_col} = '{date}'")
                df = sdf.toPandas()
                self.log(
                    f"Enterprise features read via Spark | "
                    f"table={table} | rows={len(df):,} | cols={len(df.columns)}"
                )
                return df
        except Exception as e:
            self.log(f"Spark read failed: {e} — trying pandas fallback", level="warning")

        try:
            df = pd.read_parquet(table)
            if "score_date" in df.columns:
                df = df[df["score_date"] == date]
            self.log(f"Enterprise features read via parquet | rows={len(df):,}")
            return df
        except Exception as e:
            self.log(f"Parquet read failed: {e}", level="warning")
            return None

    def _read_existing_scores(self) -> pd.DataFrame:
        """Read existing model_scores written by recovery-engine-v2."""
        try:
            df = self.memory.read("model_scores")
            if hasattr(df, "toPandas"):
                df = df.toPandas()
            missing = [
                c for c in ["propensity_30d", "propensity_180d", "account_id"]
                if c not in df.columns
            ]
            if missing:
                raise AgentBlockedException(
                    f"recovery.model_scores is missing required columns: {missing}. "
                    f"Ensure recovery-engine-v2 writes propensity_30d and propensity_180d.",
                    retry_agent="propensity_agent",
                )
            self.log(f"Existing model_scores read | rows={len(df):,}")
            return df
        except AgentBlockedException:
            raise
        except Exception as e:
            self.log(f"Could not read existing model_scores: {e}", level="warning")
            return None
