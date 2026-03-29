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

import logging
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

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
    ):
        """
        Args:
            n_clusters:    Number of clusters per SIGNAL_SEGMENT.
                           See CONFIGURATION INSTRUCTIONS above.
            model_dir:     Directory to save fitted models.
                           Set PERSONA_MODEL_DIR env var to override.
            random_state:  For reproducibility.
        """
        self.n_clusters   = n_clusters
        self.model_dir    = Path(model_dir)
        self.random_state = random_state
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
                               "persona_map": dict}}
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

            self._save(seg, km, scaler, persona_map)

            results[seg] = {
                "silhouette":   round(sil, 4),
                "cluster_sizes":pd.Series(labels).value_counts().to_dict(),
                "persona_map":  persona_map,
            }
            logger.info(
                f"Segment {seg} | n={len(seg_df):,} | "
                f"silhouette={sil:.3f} | personas={persona_map}"
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
