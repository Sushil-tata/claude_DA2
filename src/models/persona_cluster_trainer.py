"""
PersonaClusterTrainer
=====================
Trains behavioural persona clusters WITHIN each SIGNAL_SEGMENT.

Design principle:
  "Segmentation should describe behaviour, not prescribe action.
   Models should decide action through ERV optimisation."

This means:
  - Clusters describe WHO the customer is (stable behavioural identity)
  - Clusters do NOT map to actions (no SETTLE/PLAN/LEGAL/AGENCY labels)
  - Cluster labels are FEATURES for downstream models, not routing rules

──────────────────────────────────────────────────────────────────────────────
CONFIGURATION INSTRUCTIONS
──────────────────────────────────────────────────────────────────────────────
N_CLUSTERS_PER_SEGMENT (default: 4)
    Number of clusters to fit within each SIGNAL_SEGMENT group.
    4 clusters maps to the 4 behavioural personas used in reporting
    (Cooperative / Stressed / Sporadic / Disconnected), but the cluster
    labels are data-driven — not rule-assigned.

    How to override:
        PersonaClusterTrainer(n_clusters=5)

    Rule of thumb:
        - Segment A (rich signal): 4–5 clusters
        - Segment D (no signal):   2–3 clusters (less signal = less resolution)

CLUSTER_FEATURES
    Features used for clustering. All must be numeric.
    Missing values are imputed with column median before fitting.

    Current features:
        propensity_30d      — short-term recovery signal (primary)
        propensity_180d     — long-term structural signal
        willingness_score   — engagement proxy (contact + payment behaviour)
        capacity_score      — ability-to-pay proxy (bureau + card features)
        erv_at_d_optimal    — economic value of the account
        low_confidence_flag — model uncertainty signal

    Do NOT include:
        - PII fields (name, phone, address, NationalID)
        - Future-dated fields (labels, outcomes)
        - SIGNAL_SEGMENT itself (clustering is done within segment)

MODEL_PATH
    Where fitted cluster models are saved (one file per SIGNAL_SEGMENT).
    Default: models/persona_clusters/{segment}_persona_model.pkl
    Set PERSONA_MODEL_DIR env variable to override the base directory.

    In Databricks:
        Set PERSONA_MODEL_DIR to a DBFS path:
        e.g. /dbfs/FileStore/models/persona_clusters/
        Models are loaded by FeatureAgent at runtime via this path.
──────────────────────────────────────────────────────────────────────────────
"""

import importlib.util as _ilu
import logging
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def _load_validator():
    """Lazy-load ClusterBusinessValidator to avoid circular imports."""
    _spec = _ilu.spec_from_file_location(
        "cluster_business_validator",
        Path(__file__).parent / "cluster_business_validator.py",
    )
    _mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    return _mod.ClusterBusinessValidator

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

DEFAULT_N_CLUSTERS = 4

# Features used for clustering. See CONFIGURATION INSTRUCTIONS above.
CLUSTER_FEATURES = [
    "propensity_30d",
    "propensity_180d",
    "willingness_score",
    "capacity_score",
    "erv_at_d_optimal",
    "low_confidence_flag",
]

# Persona label map: cluster index → human-readable name.
# Assigned POST-FIT by ranking clusters on mean propensity_30d × capacity_score.
# This is descriptive naming only — does NOT prescribe any action.
PERSONA_RANK_LABELS = ["Cooperative", "Stressed", "Sporadic", "Disconnected"]

SIGNAL_SEGMENTS = ["A", "B", "C", "D"]

DEFAULT_MODEL_DIR = Path(
    os.environ.get("PERSONA_MODEL_DIR", "models/persona_clusters")
)


class PersonaClusterTrainer:

    def __init__(
        self,
        n_clusters: int = DEFAULT_N_CLUSTERS,
        model_dir: Path = DEFAULT_MODEL_DIR,
        random_state: int = 42,
        min_pairwise_recovery_diff: float = 0.05,
        skip_business_validation: bool = False,
    ):
        """
        Args:
            n_clusters:                  Number of clusters per SIGNAL_SEGMENT.
            model_dir:                   Save path. Set PERSONA_MODEL_DIR to override.
            random_state:                For reproducibility.
            min_pairwise_recovery_diff:  Min abs diff in recovery_180d between
                                         best and worst cluster for PASS.
                                         Default: 0.05 (5pp). See
                                         ClusterBusinessValidator docstring.
            skip_business_validation:    Set True only if outcome labels are not
                                         yet available (e.g. cold start). Models
                                         will be saved but validation is skipped
                                         with a warning. Remove this flag once
                                         outcome data is available.
        """
        self.n_clusters                  = n_clusters
        self.model_dir                   = Path(model_dir)
        self.random_state                = random_state
        self.min_pairwise_recovery_diff  = min_pairwise_recovery_diff
        self.skip_business_validation    = skip_business_validation
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def fit(self, df: pd.DataFrame) -> dict:
        """
        Fits one KMeans model per SIGNAL_SEGMENT on CLUSTER_FEATURES.
        Saves fitted models + scalers to model_dir.

        Args:
            df: DataFrame with SIGNAL_SEGMENT + CLUSTER_FEATURES columns.
                Expected source: recovery.feature_output or model_scores.

        Returns:
            dict of {segment: {"silhouette": float, "cluster_sizes": dict,
                               "persona_map": dict, "business_validation": dict}}

        ⚠ IMPORTANT — Business Validation:
            After fitting, validate() is called on clusters with outcome labels
            (recovery_30d, recovery_90d, recovery_180d) if present in df.
            If clusters are NOT differentiated on recovery_180d by ≥5pp, the fit
            is rejected for that segment and rule-based fallback is recommended.

            Include outcome columns in df for full validation.
            If outcomes not yet available (cold start), set skip_business_validation=True
            and validate retroactively once outcomes are labelled.
        """
        results = {}
        for seg in SIGNAL_SEGMENTS:
            seg_df = df[df["signal_segment"] == seg].copy()
            if len(seg_df) < self.n_clusters * 10:
                logger.warning(
                    f"Segment {seg}: only {len(seg_df)} rows — "
                    f"skipping cluster fit (need ≥ {self.n_clusters * 10})"
                )
                results[seg] = {"skipped": True, "reason": "insufficient_data"}
                continue

            X, scaler = self._prepare_features(seg_df)
            km        = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                n_init=10,
            )
            labels = km.fit_predict(X)

            sil = silhouette_score(X, labels) if len(set(labels)) > 1 else 0.0
            persona_map = self._assign_persona_labels(seg_df, labels)

            # ── Business validation before saving ─────────────────────────────
            # Clusters must be differentiated on recovery outcomes, not just
            # geometrically separated. If validation fails, log warning and
            # record result — do NOT save model for this segment.
            seg_df_labeled = seg_df.copy()
            seg_df_labeled["behavioural_persona"] = [
                persona_map.get(lbl, "Disconnected") for lbl in labels
            ]

            biz_validation = {"passed": True, "recommendation": "SKIPPED — no outcome columns"}
            if not self.skip_business_validation:
                outcome_cols = [c for c in ["recovery_30d", "recovery_90d",
                                            "recovery_180d", "recovery_amount"]
                                if c in seg_df_labeled.columns]
                if outcome_cols:
                    try:
                        Validator = _load_validator()
                        biz_validation = Validator(
                            min_pairwise_recovery_diff=self.min_pairwise_recovery_diff
                        ).validate(
                            seg_df_labeled,
                            cluster_col="behavioural_persona",
                            segment=seg,
                        )
                    except Exception as e:
                        biz_validation = {"passed": False, "error": str(e)}
                        logger.warning(f"Business validation error for segment {seg}: {e}")
                else:
                    logger.warning(
                        f"Segment {seg}: no outcome columns (recovery_*) in df — "
                        f"skipping business validation. Include recovery_180d in df "
                        f"to validate cluster separation on recovery outcomes."
                    )

            if not biz_validation.get("passed") and not self.skip_business_validation:
                if "SKIPPED" not in str(biz_validation.get("recommendation", "")):
                    logger.warning(
                        f"Segment {seg}: business validation FAILED — "
                        f"recommendation: {biz_validation.get('recommendation')}. "
                        f"Model NOT saved for this segment. "
                        f"Rule-based persona fallback will be used."
                    )
                    results[seg] = {
                        "silhouette":          round(sil, 4),
                        "cluster_sizes":       pd.Series(labels).value_counts().to_dict(),
                        "persona_map":         persona_map,
                        "business_validation": biz_validation,
                        "saved":               False,
                    }
                    continue   # skip _save — do not persist invalid clusters

            self._save(seg, km, scaler, persona_map)

            results[seg] = {
                "silhouette":          round(sil, 4),
                "cluster_sizes":       pd.Series(labels).value_counts().to_dict(),
                "persona_map":         persona_map,
                "business_validation": biz_validation,
                "saved":               True,
            }
            logger.info(
                f"Segment {seg} | n={len(seg_df):,} | "
                f"silhouette={sil:.3f} | "
                f"biz_validation={biz_validation.get('recommendation')} | "
                f"personas={persona_map}"
            )

        return results

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Applies fitted cluster models to assign BEHAVIOURAL_PERSONA per row.
        Loads models from model_dir. Called by FeatureAgent at inference time.

        Args:
            df: DataFrame with signal_segment + CLUSTER_FEATURES columns.

        Returns:
            pd.Series of persona labels (same index as df).
        """
        personas = pd.Series("Disconnected", index=df.index)

        for seg in SIGNAL_SEGMENTS:
            seg_mask = df["signal_segment"] == seg
            if not seg_mask.any():
                continue

            model_path = self._model_path(seg)
            if not model_path.exists():
                logger.warning(
                    f"No fitted model for segment {seg} at {model_path}. "
                    f"Defaulting to 'Disconnected'. Run PersonaClusterTrainer.fit() first."
                )
                continue

            km, scaler, persona_map = self._load(seg)
            seg_df = df[seg_mask].copy()
            X, _   = self._prepare_features(seg_df, scaler=scaler)
            labels = km.predict(X)
            personas[seg_mask] = [
                persona_map.get(lbl, "Disconnected") for lbl in labels
            ]

        return personas

    # ── Internal ──────────────────────────────────────────────────────────────

    def _prepare_features(
        self, df: pd.DataFrame, scaler: StandardScaler = None
    ):
        available = [c for c in CLUSTER_FEATURES if c in df.columns]
        X = df[available].copy()

        # Impute missing with median
        for col in X.columns:
            X[col] = X[col].fillna(X[col].median())

        # Boolean to float
        if "low_confidence_flag" in X.columns:
            X["low_confidence_flag"] = X["low_confidence_flag"].astype(float)

        if scaler is None:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = scaler.transform(X)

        return X_scaled, scaler

    def _assign_persona_labels(
        self, df: pd.DataFrame, labels: np.ndarray
    ) -> dict:
        """
        Ranks clusters by mean(propensity_30d × capacity_score) descending.
        Maps rank 0 → Cooperative, rank 1 → Stressed, etc.
        This is descriptive naming only.
        """
        df = df.copy()
        df["_cluster"] = labels

        p30 = "propensity_30d" if "propensity_30d" in df.columns else None
        cap = "capacity_score" if "capacity_score" in df.columns else None

        if p30 and cap:
            df["_score"] = df[p30].fillna(0) * df[cap].fillna(0)
        elif p30:
            df["_score"] = df[p30].fillna(0)
        else:
            df["_score"] = 0.0

        cluster_means = df.groupby("_cluster")["_score"].mean().sort_values(ascending=False)
        n = min(len(cluster_means), len(PERSONA_RANK_LABELS))
        return {
            int(cluster): PERSONA_RANK_LABELS[rank]
            for rank, cluster in enumerate(cluster_means.index[:n])
        }

    def _model_path(self, segment: str) -> Path:
        return self.model_dir / f"{segment}_persona_model.pkl"

    def _save(
        self, segment: str, km: KMeans,
        scaler: StandardScaler, persona_map: dict
    ) -> None:
        path = self._model_path(segment)
        joblib.dump({"km": km, "scaler": scaler, "persona_map": persona_map}, path)
        logger.info(f"Saved persona model → {path}")

    def _load(self, segment: str):
        obj = joblib.load(self._model_path(segment))
        return obj["km"], obj["scaler"], obj["persona_map"]
