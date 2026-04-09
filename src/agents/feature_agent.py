"""
FeatureAgent
============
Task 2 in the agentic NBA pipeline. Runs after DataQualityAgent.

Reads:
  recovery.model_scores — validated scores from recovery-engine-v2

Responsibilities:
  - Selects and validates the feature columns needed downstream
  - Computes derived features: recovery_tier, erv_band, contact_priority
  - Derives SIGNAL_SEGMENT (A/B/C/D) based on data availability
  - Derives BEHAVIOURAL_PERSONA (Cooperative/Stressed/Sporadic/Disconnected)
  - Keeps willingness_score + capacity_score as model FEATURES (not segments)
  - Ensures point-in-time safety (no future-dated features)
  - Writes a clean feature_output table for ModelAgent + ConstraintAgent

Writes:
  recovery.feature_output

──────────────────────────────────────────────────────────────────────────────
CONFIGURATION INSTRUCTIONS
──────────────────────────────────────────────────────────────────────────────
BUREAU_SIGNAL_LAG_DAYS (default: 60)
    Maximum number of days since the last NCB/TUEF bureau pull for the
    bureau signal to be considered "available" for a given account.

    Why this matters:
    - SIGNAL_SEGMENT is derived from CardX signal availability + Bureau
      signal availability. If the bureau pull is older than this threshold,
      the account is treated as having NO bureau signal (Segment C or D).
    - Default of 60 days (2 months) reflects the standard NCB refresh
      cycle at CardX. Adjust if your bureau refresh cadence changes.

    How to override at runtime (Databricks task parameter):
        --bureau-signal-lag-days 90

    How to override in code:
        FeatureAgent(..., bureau_signal_lag_days=90)

    Impact of increasing this value:
    - More accounts classified as Segment A or C (bureau signal available)
    - Lower proportion of Segment D (no signal) accounts
    - Use a higher value only if bureau data is reliably refreshed more often

    Impact of decreasing this value:
    - More accounts fall to Segment D (no signal) → high-discount treatment
    - Use a lower value if bureau data quality is degrading at month 2

SIGNAL_SEGMENT definitions:
    A = CardX internal signal available  AND  Bureau signal available
    B = CardX internal signal available  AND  Bureau signal NOT available
    C = CardX internal signal NOT available  AND  Bureau signal available
    D = No CardX signal  AND  No Bureau signal  (lowest information)

    CardX signal is considered available if:
    - propensity_30d is not null (model was able to score the account)
    - score_date is not null

    Bureau signal is considered available if:
    - bureau_pull_date is present AND within BUREAU_SIGNAL_LAG_DAYS of score_date
    - OR ncb_tradeline_count > 0 (at least one tradeline returned from NCB)

BEHAVIOURAL_PERSONA definitions (derived from willingness + capacity scores):
    Cooperative   = willingness >= 0.5 AND capacity >= 0.5
    Stressed      = willingness >= 0.5 AND capacity <  0.5
    Sporadic      = willingness <  0.5 AND capacity >= 0.5
    Disconnected  = willingness <  0.5 AND capacity <  0.5

    willingness_score and capacity_score are passed as FEATURES to downstream
    models (propensity, amount, elasticity). They are NOT used as segmentation
    rules to pre-assign actions.

ENTERPRISE FEATURE TABLE (optional join — recommended)
──────────────────────────────────────────────────────────────────────────────
    Your DS team's enterprise feature engineering notebook produces ~100+
    features and writes them to a Delta table (e.g. cdx_mdz_prd.feature_store).
    Those 100 features were used to TRAIN the propensity models.

    The FeatureAgent does NOT recreate those features. It reads pre-computed
    propensity scores from recovery.model_scores (output of recovery-engine-v2).

    To make all 100 features available to downstream models (PersonaCluster,
    WillingnessModel, CapacityModel, AmountModel), configure:

        enterprise_feature_table: fully-qualified Delta table name
        enterprise_feature_join_keys: columns to join on (default: account_id + score_date)

    How to configure via env variables (Databricks cluster config):
        ENTERPRISE_FEATURE_TABLE=cdx_mdz_prd.feature_store
        ENTERPRISE_FEATURE_JOIN_KEYS=account_id,score_date

    How to configure in code:
        FeatureAgent(
            ...,
            enterprise_feature_table="cdx_mdz_prd.feature_store",
            enterprise_feature_join_keys=["account_id", "score_date"],
        )

    If enterprise_feature_table is not set:
        FeatureAgent only passes through columns present in recovery.model_scores.
        Downstream models fall back to their default feature subsets.

FEATURE PASSTHROUGH — ALL COLUMNS FLOW DOWNSTREAM
    FeatureAgent no longer filters to a hard column allowlist.
    All columns from model_scores + enterprise feature join pass through to
    feature_output. Downstream agents select only what they need.

    Columns that FeatureAgent itself ADDS (always present in output):
        signal_segment, behavioural_persona, final_segment
        willingness_score, capacity_score
        erv_band, recovery_tier, contact_priority

    BLOCKED columns (never passed downstream — PII / future-dated):
        PII_BLOCK_COLS list below. Add any PII fields your data contains.
──────────────────────────────────────────────────────────────────────────────
"""

import importlib.util as _ilu
import logging
import os
from pathlib import Path

import pandas as pd

from agents.base_agent import AgentBlockedException, BaseAgent
import importlib.util as _ilu
import pathlib as _pl

_pct_spec = _ilu.spec_from_file_location(
    "persona_cluster_trainer",
    _pl.Path(__file__).parent.parent / "models" / "persona_cluster_trainer.py",
)
_pct_mod = _ilu.module_from_spec(_pct_spec)
_pct_spec.loader.exec_module(_pct_mod)
PersonaClusterTrainer = _pct_mod.PersonaClusterTrainer

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

# See CONFIGURATION INSTRUCTIONS above before changing this value.
DEFAULT_BUREAU_SIGNAL_LAG_DAYS = 60

DEFAULT_PERSONA_MODEL_DIR         = Path(os.environ.get("PERSONA_MODEL_DIR",         "models/persona_clusters"))
DEFAULT_WILLINGNESS_MODEL_DIR     = Path(os.environ.get("WILLINGNESS_MODEL_DIR",     "models/willingness_model"))
DEFAULT_CAPACITY_MODEL_DIR        = Path(os.environ.get("CAPACITY_MODEL_DIR",        "models/capacity_model"))
# Enterprise feature table — set via env var or constructor param.
# This is the Delta table your DS team writes 100+ features to.
# Leave None to skip the join (FeatureAgent will use only model_scores columns).
DEFAULT_ENTERPRISE_FEATURE_TABLE = os.environ.get("ENTERPRISE_FEATURE_TABLE", None)

# Join keys between model_scores and enterprise feature table.
# Default: account_id + score_date for point-in-time safe join.
# Override via env: ENTERPRISE_FEATURE_JOIN_KEYS=account_id,score_date
_join_keys_env = os.environ.get("ENTERPRISE_FEATURE_JOIN_KEYS", "account_id,score_date")
DEFAULT_ENTERPRISE_JOIN_KEYS = [k.strip() for k in _join_keys_env.split(",")]

# PII columns — NEVER passed downstream regardless of source.
# Add any PII fields present in your enterprise feature table.
PII_BLOCK_COLS = {
    "national_id", "citizen_id", "passport_no",
    "full_name", "first_name", "last_name",
    "phone_number", "mobile_number", "email",
    "address", "home_address",
    "date_of_birth", "dob",
}

# ERV bands for ConstraintAgent rules (THB)
ERV_BANDS = [
    (50_000, "PREMIUM"),
    (30_000, "HIGH"),
    (10_000, "MEDIUM"),
    (0,      "LOW"),
]

# Willingness / capacity thresholds for BEHAVIOURAL_PERSONA derivation
WILLINGNESS_THRESHOLD = 0.5
CAPACITY_THRESHOLD    = 0.5


class FeatureAgent(BaseAgent):

    def __init__(
        self,
        execution_date: str,
        memory,
        dry_run: bool = False,
        bureau_signal_lag_days: int       = DEFAULT_BUREAU_SIGNAL_LAG_DAYS,
        persona_model_dir: Path           = DEFAULT_PERSONA_MODEL_DIR,
        willingness_model_dir: Path       = DEFAULT_WILLINGNESS_MODEL_DIR,
        capacity_model_dir: Path          = DEFAULT_CAPACITY_MODEL_DIR,
        enterprise_feature_table: str     = DEFAULT_ENTERPRISE_FEATURE_TABLE,
        enterprise_join_keys: list        = None,
    ):
        """
        Args:
            execution_date:           Scoring date (YYYY-MM-DD).
            memory:                   AgentMemory instance.
            dry_run:                  If True, skip all Delta writes.
            bureau_signal_lag_days:   Max days since last bureau pull for
                                      bureau signal to count as available.
                                      Default: 60 (2 months).
            persona_model_dir:        Fitted PersonaClusterTrainer models.
                                      Falls back to rule-based if not found.
                                      Set PERSONA_MODEL_DIR env var to override.
            willingness_model_dir:    Fitted WillingnessModelTrainer.
                                      Falls back to raw willingness_score column
                                      if column present, else 0.5 default.
                                      Set WILLINGNESS_MODEL_DIR to override.
            capacity_model_dir:       Fitted CapacityModelTrainer.
                                      Falls back to raw capacity_score column
                                      if column present, else 0.5 default.
                                      Set CAPACITY_MODEL_DIR to override.
            enterprise_feature_table: Fully-qualified Delta table name for your
                                      enterprise feature store (e.g.
                                      'cdx_mdz_prd.feature_store').
                                      All columns from this table are joined to
                                      model_scores and passed downstream.
                                      Set ENTERPRISE_FEATURE_TABLE env var to
                                      configure without code changes.
                                      Leave None to skip the join.
            enterprise_join_keys:     Columns to join on between model_scores
                                      and enterprise_feature_table.
                                      Default: ['account_id', 'score_date'].
                                      Set ENTERPRISE_FEATURE_JOIN_KEYS env var.
        """
        super().__init__(
            agent_name="feature_agent",
            execution_date=execution_date,
            memory=memory,
            dry_run=dry_run,
        )
        self.bureau_signal_lag_days       = bureau_signal_lag_days
        self.persona_model_dir            = Path(persona_model_dir)
        self.willingness_model_dir        = Path(willingness_model_dir)
        self.capacity_model_dir           = Path(capacity_model_dir)
        self.enterprise_feature_table     = enterprise_feature_table
        self.enterprise_join_keys         = enterprise_join_keys or DEFAULT_ENTERPRISE_JOIN_KEYS
        self._persona_trainer             = PersonaClusterTrainer(model_dir=self.persona_model_dir)
        self._willingness_trainer         = None
        self._capacity_trainer            = None
        self._try_load_score_models()
        self.log(
            f"bureau_signal_lag_days={self.bureau_signal_lag_days} | "
            f"persona_model_dir={self.persona_model_dir} | "
            f"willingness_model={'loaded' if self._willingness_trainer else 'fallback'} | "
            f"capacity_model={'loaded' if self._capacity_trainer else 'fallback'} | "
            f"enterprise_features={'YES: ' + str(self.enterprise_feature_table) if self.enterprise_feature_table else 'NO (set ENTERPRISE_FEATURE_TABLE to enable)'}"
        )

    def _try_load_score_models(self) -> None:
        """Load willingness + capacity models if fitted. Silent fallback if not."""
        for attr, model_dir, trainer_name, file_name in [
            ("_willingness_trainer", self.willingness_model_dir,
             "willingness_model_trainer", "willingness_model.pkl"),
            ("_capacity_trainer",    self.capacity_model_dir,
             "capacity_model_trainer",    "capacity_model_30d.pkl"),
        ]:
            model_path = model_dir / file_name
            if model_path.exists():
                try:
                    _spec = _ilu.spec_from_file_location(
                        trainer_name,
                        Path(__file__).parent.parent / "models" / f"{trainer_name}.py",
                    )
                    _mod = _ilu.module_from_spec(_spec)
                    _spec.loader.exec_module(_mod)
                    cls_name  = "WillingnessModelTrainer" if "willingness" in trainer_name \
                                else "CapacityModelTrainer"
                    setattr(self, attr, getattr(_mod, cls_name)(model_dir=model_dir))
                    self.log(f"{trainer_name} loaded from {model_dir}")
                except Exception as e:
                    self.log(f"Could not load {trainer_name}: {e} — using fallback", level="warning")
            else:
                self.log(
                    f"No fitted {trainer_name} at {model_path} — "
                    f"using column value or 0.5 default. "
                    f"Run {cls_name if 'cls_name' in dir() else trainer_name}.fit() "
                    f"to replace rule-based score.",
                    level="warning",
                )

    def _compute_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes willingness_score and capacity_score.

        Priority:
          1. Trained model output (WillingnessModelTrainer / CapacityModelTrainer)
          2. Raw column value if present in input (upstream pre-computed)
          3. Default 0.5 (neutral — no information)

        Both scores are written as numeric features, NOT used for segmentation rules.
        """
        df = df.copy()

        # willingness_score
        if self._willingness_trainer is not None:
            try:
                df["willingness_score"] = self._willingness_trainer.predict(df).values
                self.log("willingness_score: model-based")
            except Exception as e:
                self.log(f"Willingness model predict failed: {e} — using column/default", level="warning")
                if "willingness_score" not in df.columns:
                    df["willingness_score"] = None   # null = unknown, not neutral
        elif "willingness_score" not in df.columns:
            df["willingness_score"] = None
            self.log(
                "willingness_score: NULL (no model, no column). "
                "Rule-based persona fallback will assign 'Unknown'. "
                "Train WillingnessModelTrainer to activate real scores."
            )

        # capacity_score
        if self._capacity_trainer is not None:
            try:
                df["capacity_score"] = self._capacity_trainer.predict(df).values
                self.log("capacity_score: model-based")
            except Exception as e:
                self.log(f"Capacity model predict failed: {e} — using column/default", level="warning")
                if "capacity_score" not in df.columns:
                    df["capacity_score"] = None   # null = unknown, not neutral
        elif "capacity_score" not in df.columns:
            df["capacity_score"] = None
            self.log(
                "capacity_score: NULL (no model, no column). "
                "Rule-based persona fallback will assign 'Unknown'. "
                "Train CapacityModelTrainer to activate real scores."
            )

        return df

    def _join_enterprise_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Joins your enterprise feature table to model_scores on account_id + score_date.

        This is the bridge between your existing DS feature pipeline and the
        agent pipeline. The enterprise table is read once per execution and
        joined in-memory.

        - All columns from the enterprise table are added to df.
        - Duplicate column names (already in model_scores) are skipped —
          model_scores values take precedence (propensity scores are authoritative).
        - If the join produces zero matches, logs a warning but does NOT block.
        - PII columns are stripped AFTER this join in execute().

        Args:
            df: model_scores DataFrame (account_id + score_date + score columns).

        Returns:
            df enriched with enterprise feature columns (or unchanged if no table configured).
        """
        if not self.enterprise_feature_table:
            self.log(
                "No enterprise_feature_table configured — "
                "downstream models will use only model_scores columns. "
                "Set ENTERPRISE_FEATURE_TABLE env var to connect your feature store.",
                level="warning",
            )
            return df

        self.log(f"Joining enterprise feature table: {self.enterprise_feature_table}")
        try:
            # Read from Spark if available (Databricks), else read as Delta via pandas
            feat_df = self._read_enterprise_table()
            if feat_df is None or feat_df.empty:
                self.log(
                    f"Enterprise feature table {self.enterprise_feature_table} returned 0 rows "
                    f"— skipping join. Check table name and execution_date.",
                    level="warning",
                )
                return df

            # Validate join keys exist in both tables
            missing_left  = [k for k in self.enterprise_join_keys if k not in df.columns]
            missing_right = [k for k in self.enterprise_join_keys if k not in feat_df.columns]
            if missing_left or missing_right:
                self.log(
                    f"Join key mismatch — skipping enterprise join. "
                    f"Missing in model_scores: {missing_left}. "
                    f"Missing in feature table: {missing_right}. "
                    f"enterprise_join_keys={self.enterprise_join_keys}",
                    level="warning",
                )
                return df

            # Drop duplicate columns from feature table — model_scores is authoritative
            existing_cols   = set(df.columns)
            join_key_set    = set(self.enterprise_join_keys)
            new_feat_cols   = [
                c for c in feat_df.columns
                if c not in existing_cols or c in join_key_set
            ]
            feat_df = feat_df[new_feat_cols]

            enriched = df.merge(feat_df, on=self.enterprise_join_keys, how="left")

            match_rate = enriched[self.enterprise_join_keys[0]].notna().mean()
            added_cols = len(enriched.columns) - len(df.columns)
            self.log(
                f"Enterprise join complete | "
                f"added_cols={added_cols} | "
                f"match_rate={match_rate:.1%} | "
                f"rows={len(enriched):,}"
            )
            return enriched

        except Exception as e:
            self.log(
                f"Enterprise feature join failed: {e} — "
                "continuing without enterprise features. "
                "Downstream models will use model_scores columns only.",
                level="warning",
            )
            return df

    def _read_enterprise_table(self) -> pd.DataFrame:
        """
        Reads the enterprise feature table for execution_date.

        Tries Spark first (Databricks production), falls back to pandas Delta reader.
        Filters to execution_date to ensure point-in-time safety.
        """
        table = self.enterprise_feature_table
        date  = self.execution_date

        # Try Spark (Databricks)
        try:
            from pyspark.sql import SparkSession
            spark = SparkSession.getActiveSession()
            if spark is not None:
                # Point-in-time filter — only features as of execution_date
                # Assumes the feature table has a score_date or feature_date column
                date_col = next(
                    (c for c in ["score_date", "feature_date", "snapshot_date"]
                     if c in [f.name for f in spark.table(table).schema]),
                    None,
                )
                sdf = spark.table(table)
                if date_col:
                    sdf = sdf.filter(f"{date_col} = '{date}'")
                return sdf.toPandas()
        except Exception:
            pass

        # Fallback: pandas + delta-rs (local / non-Spark environments)
        try:
            import delta
            return delta.DeltaTable(table).toDF().toPandas()
        except Exception:
            pass

        # Last resort: try reading as a parquet path
        try:
            return pd.read_parquet(table)
        except Exception as e:
            raise RuntimeError(
                f"Could not read enterprise feature table '{table}'. "
                f"Tried Spark, delta-rs, and parquet. Error: {e}"
            )

    def execute(self) -> dict:
        self.log("Reading validated model_scores")
        df = self._read_scores()

        # ── Join enterprise feature table (your DS team's 100+ features) ──────
        # This brings all features used to train propensity/willingness/capacity
        # models into the same DataFrame so downstream agents can use them.
        # No features are recreated here — only joined from the feature store.
        df = self._join_enterprise_features(df)

        # ── Drop PII columns — never passed downstream ─────────────────────────
        pii_present = [c for c in df.columns if c.lower() in PII_BLOCK_COLS]
        if pii_present:
            df = df.drop(columns=pii_present)
            self.log(f"Dropped PII columns: {pii_present}")

        self.log(
            f"Feature columns available for downstream: {len(df.columns)} | "
            f"sample: {list(df.columns[:10])}"
        )

        # ── Compute model-based willingness + capacity scores ─────────────────
        # These replace rule-based weighted sums.
        # Both are FEATURES passed downstream — not segmentation rules.
        # If enterprise features are present they will be used by the models.
        df = self._compute_scores(df)

        # ── Derive computed features ──────────────────────────────────────────
        df["erv_band"]         = df["erv_at_d_optimal"].apply(self._erv_band)
        df["recovery_tier"]    = df.apply(self._recovery_tier, axis=1)
        df["contact_priority"] = df.apply(self._contact_priority, axis=1)

        # ── Derive SIGNAL_SEGMENT ─────────────────────────────────────────────
        df["signal_segment"] = df.apply(
            lambda r: self._signal_segment(r, self.bureau_signal_lag_days), axis=1
        )

        # ── Derive BEHAVIOURAL_PERSONA (cluster model; Segment D = rule-based) ─
        # predict_with_confidence returns both persona label and confidence ratio
        persona_result = self._derive_persona(df)
        if isinstance(persona_result, pd.DataFrame):
            df["behavioural_persona"] = persona_result["behavioural_persona"].values
            df["persona_confidence"]  = persona_result["persona_confidence"].values
        else:
            df["behavioural_persona"] = persona_result
            df["persona_confidence"]  = 1.0   # rule-based: no confidence metric

        # ── Trajectory features — persona stability over time ─────────────────
        # persona_previous: persona label from previous execution
        # days_in_current_persona: consecutive days in the current persona
        # persona_assigned_date: when the current persona was first assigned
        # These require reading yesterday's feature_output — fail silently if absent.
        df = self._add_trajectory_features(df)

        # ── Vulnerable / Hardship flag (BOT compliance dimension) ─────────────
        # Orthogonal to SIGNAL_SEGMENT and BEHAVIOURAL_PERSONA.
        # Identifies accounts showing signs of severe financial hardship.
        # DecisionAgent and ConstraintAgent apply additional guardrails for these.
        df["vulnerable_flag"] = df.apply(self._vulnerable_flag, axis=1)

        # ── Composite segment key (SIGNAL_SEGMENT + BEHAVIOURAL_PERSONA) ──────
        df["final_segment"] = df["signal_segment"] + "_" + df["behavioural_persona"]

        # ── Point-in-time safety check ────────────────────────────────────────
        if "score_date" in df.columns:
            future_rows = df[df["score_date"] > self.execution_date]
            if len(future_rows) > 0:
                raise AgentBlockedException(
                    f"PIT violation: {len(future_rows)} rows have "
                    f"score_date > execution_date ({self.execution_date}). "
                    "Possible data leakage.",
                    retry_agent="data_quality_agent",
                )

        if not self.dry_run:
            self.memory.write("feature_output", df)

        signal_dist      = df["signal_segment"].value_counts().to_dict()
        persona_dist     = df["behavioural_persona"].value_counts().to_dict()
        n_vulnerable     = int(df["vulnerable_flag"].sum()) \
                           if "vulnerable_flag" in df.columns else 0
        mean_confidence  = round(float(df["persona_confidence"].mean()), 3) \
                           if "persona_confidence" in df.columns else None
        n_dormant        = int((df["behavioural_persona"] == "Dormant").sum())

        self.log(
            f"Feature output ready | rows={len(df):,} | "
            f"signal_segments={signal_dist} | personas={persona_dist} | "
            f"dormant={n_dormant:,} | "
            f"vulnerable={n_vulnerable:,} | persona_confidence_mean={mean_confidence}"
        )
        return {
            "rows_written":                len(df),
            "feature_columns":             len(df.columns),
            "erv_band_dist":               df["erv_band"].value_counts().to_dict(),
            "recovery_tier_dist":          df["recovery_tier"].value_counts().to_dict(),
            "signal_segment_dist":         signal_dist,
            "behavioural_persona_dist":    persona_dist,
            "dormant_accounts":            n_dormant,
            "vulnerable_accounts":         n_vulnerable,
            "persona_confidence_mean":     mean_confidence,
        }

    # ── SIGNAL_SEGMENT derivation ─────────────────────────────────────────────

    @staticmethod
    def _has_cardx_signal(row) -> bool:
        """
        CardX internal signal is available when the propensity model was
        able to score the account (propensity_30d is not null).
        """
        return pd.notna(row.get("propensity_30d")) and pd.notna(row.get("score_date"))

    @staticmethod
    def _has_bureau_signal(row, lag_days: int) -> bool:
        """
        Bureau (NCB/TUEF) signal is available when:
          - bureau_pull_date is present AND within lag_days of score_date, OR
          - ncb_tradeline_count > 0 (at least one tradeline returned)

        lag_days is configurable via bureau_signal_lag_days (default 60).
        See module docstring for full guidance.
        """
        tradeline_count = row.get("ncb_tradeline_count", 0) or 0
        if tradeline_count > 0:
            return True

        bureau_pull = row.get("bureau_pull_date")
        score_date  = row.get("score_date")
        if pd.isna(bureau_pull) or pd.isna(score_date):
            return False

        try:
            pull_dt  = pd.Timestamp(bureau_pull)
            score_dt = pd.Timestamp(score_date)
            return (score_dt - pull_dt).days <= lag_days
        except Exception:
            return False

    @staticmethod
    def _signal_segment(row, lag_days: int) -> str:
        """
        Phase 1 (Amendment 4): binary split only.
          FULL_SIGNAL    = CardX signal AND Bureau signal available.
          LIMITED_SIGNAL = any signal missing (B, C, D combined).

        Phase 2 target: expand to four-quadrant A/B/C/D when ≥1,000 labelled
        accounts per quadrant with recovery outcomes are confirmed.
        """
        has_cardx  = FeatureAgent._has_cardx_signal(row)
        has_bureau = FeatureAgent._has_bureau_signal(row, lag_days)

        if has_cardx and has_bureau:
            return "FULL_SIGNAL"
        return "LIMITED_SIGNAL"

    # ── BEHAVIOURAL_PERSONA derivation ────────────────────────────────────────

    def _derive_persona(self, df: pd.DataFrame):
        """
        Attempts to use fitted PersonaClusterTrainer models first.
        Falls back to rule-based thresholds if models are not found.

        Cluster-based approach is preferred — it derives behavioural identity
        from data, not from manually defined rules.
        Rule-based fallback exists only for cold-start (no trained model yet).

        Returns:
            DataFrame with columns [behavioural_persona, persona_confidence]
            when cluster models are found; pd.Series of labels otherwise.
        """
        any_model = any(
            (self.persona_model_dir / f"{seg}_persona_model.pkl").exists()
            for seg in ["FULL_SIGNAL", "LIMITED_SIGNAL"]
        )

        if any_model:
            self.log("Using fitted cluster models for BEHAVIOURAL_PERSONA (+ confidence)")
            return self._persona_trainer.predict_with_confidence(df)

        self.log(
            "No fitted persona cluster models found at "
            f"{self.persona_model_dir} — using rule-based fallback. "
            "WARNING: rule-based personas require willingness_score and "
            "capacity_score to be model-derived (not null). "
            "If models are not trained, all accounts will be labelled 'Unknown'. "
            "Run PersonaClusterTrainer.fit() on historical data with recovery_180d "
            "outcomes to enable data-driven behavioural segmentation.",
            level="warning",
        )
        personas = df.apply(self._behavioural_persona_rules, axis=1)
        n_unknown = int((personas == "Unknown").sum())
        if n_unknown > 0:
            self.log(
                f"{n_unknown:,} accounts ({n_unknown/len(df):.0%}) labelled 'Unknown' persona "
                "— willingness_score or capacity_score is null. "
                "Train WillingnessModelTrainer + CapacityModelTrainer first, "
                "then PersonaClusterTrainer for full segmentation.",
                level="warning",
            )
        return personas

    @staticmethod
    def _behavioural_persona_rules(row) -> str:
        """
        Rule-based fallback for BEHAVIOURAL_PERSONA.
        Used only when no fitted PersonaClusterTrainer model exists.

        IMPORTANT — limitations of this fallback:
          1. Requires willingness_score and capacity_score to be model-derived
             (from WillingnessModelTrainer / CapacityModelTrainer). If these
             are null (no model trained), all accounts are labelled "Unknown".
          2. The 0.5 threshold is arbitrary — a boundary in model-output space,
             not a data-driven cluster boundary.
          3. Replace with PersonaClusterTrainer.fit() as soon as 3+ months of
             labeled outcome data (recovery_180d) is available.

        Persona assignment (STRICT threshold — > not >=):
          Cooperative  = willingness > 0.5 AND capacity > 0.5
          Stressed     = willingness > 0.5 AND capacity <= 0.5
          Sporadic     = willingness <= 0.5 AND capacity > 0.5
          Disconnected = willingness <= 0.5 AND capacity <= 0.5
          Unknown      = either score is null (no model fitted)
        """
        w = row.get("willingness_score")
        c = row.get("capacity_score")

        # Null means no model fitted — do not assign a false persona
        if w is None or c is None or (isinstance(w, float) and pd.isna(w)) \
                or (isinstance(c, float) and pd.isna(c)):
            return "Unknown"

        w = float(w)
        c = float(c)

        if w > WILLINGNESS_THRESHOLD and c > CAPACITY_THRESHOLD:
            return "Cooperative"
        if w > WILLINGNESS_THRESHOLD and c <= CAPACITY_THRESHOLD:
            return "Stressed"
        if w <= WILLINGNESS_THRESHOLD and c > CAPACITY_THRESHOLD:
            return "Sporadic"
        return "Disconnected"

    # ── Trajectory features ───────────────────────────────────────────────────

    def _add_trajectory_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds persona stability features by comparing today's persona against
        the most recent historical feature_output stored in AgentMemory.

        Columns added:
            persona_previous:       persona label from the most recent prior run
                                    (null if no history available)
            persona_assigned_date:  date when current persona streak began
                                    (execution_date if persona changed or no history)
            days_in_current_persona: consecutive days in the current persona
                                    (1 if new or changed; grows each day it stays)

        Falls back gracefully if no prior feature_output is found — these
        columns will be null/1 on the first run.

        CONFIGURATION: AgentMemory.read("feature_output_prev") reads the most
        recent prior partition. If your Delta table uses date-partitioned writes,
        this returns yesterday's data automatically.
        """
        df = df.copy()

        # Defaults — applied when no history is available
        df["persona_previous"]       = None
        df["persona_assigned_date"]  = self.execution_date
        df["days_in_current_persona"] = 1

        if "account_id" not in df.columns:
            self.log(
                "No account_id column — skipping trajectory features.",
                level="warning",
            )
            return df

        try:
            prev = self.memory.read("feature_output_prev")
            if hasattr(prev, "toPandas"):
                prev = prev.toPandas()

            if prev is None or prev.empty:
                return df

            persona_cols = ["account_id", "behavioural_persona",
                            "persona_assigned_date", "days_in_current_persona"]
            prev_avail   = [c for c in persona_cols if c in prev.columns]
            if "account_id" not in prev_avail or "behavioural_persona" not in prev_avail:
                return df

            prev = prev[prev_avail].rename(
                columns={"behavioural_persona": "persona_previous"}
            )
            df = df.merge(prev, on="account_id", how="left", suffixes=("", "_prev"))

            # Carry forward persona_assigned_date and days_in_current_persona
            # when persona has NOT changed; reset to today when it changed.
            same_persona = df["behavioural_persona"] == df["persona_previous"]

            # days_in_current_persona: increment when persona unchanged
            if "days_in_current_persona_prev" in df.columns:
                df["days_in_current_persona"] = df["days_in_current_persona_prev"].where(
                    same_persona, other=1
                ).fillna(1).astype(int) + same_persona.astype(int)
                df = df.drop(columns=["days_in_current_persona_prev"])

            # persona_assigned_date: keep prior date when unchanged, set today when changed
            if "persona_assigned_date_prev" in df.columns:
                df["persona_assigned_date"] = df["persona_assigned_date_prev"].where(
                    same_persona, other=self.execution_date
                ).fillna(self.execution_date)
                df = df.drop(columns=["persona_assigned_date_prev"])

            n_changed = int((~same_persona & df["persona_previous"].notna()).sum())
            self.log(
                f"Trajectory features: {n_changed:,} accounts changed persona "
                f"since last run | "
                f"mean_days_in_persona={df['days_in_current_persona'].mean():.1f}"
            )
        except Exception as e:
            self.log(
                f"Trajectory feature computation failed (non-blocking): {e}. "
                "persona_previous and days_in_current_persona will be null/1.",
                level="warning",
            )

        return df

    # ── Vulnerable / hardship flag ────────────────────────────────────────────

    @staticmethod
    def _vulnerable_flag(row) -> int:
        """
        Binary flag for accounts showing signs of severe financial hardship.

        Orthogonal to SIGNAL_SEGMENT and BEHAVIOURAL_PERSONA — a customer can
        be Cooperative AND vulnerable (willing and able but under severe stress).

        Used by ConstraintAgent and DecisionAgent to apply BOT-compliant
        guardrails: no aggressive settlement offers, escalate to human review,
        restrict to low-cost contact channels only.

        Rules (any one triggers the flag):
            1. dpd_current >= 180         — severely delinquent
            2. months_delinquent >= 12    — chronically delinquent (1+ year)
            3. broken_promise_count_3m >= 3 — repeated broken PTPs (distress signal)
            4. ncb_total_revolving_util >= 0.95 — near-maxed credit (financial crisis)
            5. days_since_last_payment >= 365   — no payment in 12+ months

        Returns:
            1 if any hardship indicator is present, 0 otherwise.
        """
        dpd           = row.get("dpd_current",               0) or 0
        months_dlq    = row.get("months_delinquent",          0) or 0
        broken_ptps   = row.get("broken_promise_count_3m",    0) or 0
        revolving_util = row.get("ncb_total_revolving_util",  0.0) or 0.0
        days_no_pay   = row.get("days_since_last_payment",    0) or 0

        if (dpd           >= 180  or
                months_dlq    >= 12   or
                broken_ptps   >= 3    or
                revolving_util >= 0.95 or
                days_no_pay   >= 365):
            return 1
        return 0

    # ── Existing derived feature logic ────────────────────────────────────────

    @staticmethod
    def _erv_band(erv: float) -> str:
        for threshold, label in ERV_BANDS:
            if erv >= threshold:
                return label
        return "LOW"

    @staticmethod
    def _recovery_tier(row) -> str:
        """Combine signal_segment + propensity_30d into a 3-tier classification.
        Uses P_1M (propensity_30d) as primary signal per design spec."""
        seg  = row.get("signal_segment", "D")
        p30  = row.get("propensity_30d", 0.0) or 0.0
        if seg in ("A", "B") and p30 >= 0.40:
            return "TIER_1_HIGH_RECOVERY"
        if seg in ("A", "B", "C") and p30 >= 0.20:
            return "TIER_2_MEDIUM_RECOVERY"
        return "TIER_3_LOW_RECOVERY"

    @staticmethod
    def _contact_priority(row) -> int:
        """1 = highest priority for outbound contact scheduling."""
        seg  = row.get("signal_segment", "D")
        erv  = row.get("erv_at_d_optimal", 0.0) or 0.0
        p30  = row.get("propensity_30d", 0.0) or 0.0
        if seg == "A" and erv >= 30_000:
            return 1
        if seg in ("A", "C") and erv >= 10_000:
            return 2
        if seg == "B":
            return 3
        return 4  # Segment D or low ERV

    def _read_scores(self) -> pd.DataFrame:
        df = self.memory.read("model_scores")
        if hasattr(df, "toPandas"):
            df = df.toPandas()
        if df.empty:
            raise AgentBlockedException(
                "model_scores is empty — DataQualityAgent may have been skipped.",
                retry_agent="data_quality_agent",
            )
        return df
