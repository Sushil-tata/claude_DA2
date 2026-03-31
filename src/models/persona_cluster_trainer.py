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
    Domain-meaningful imputation applied (see _prepare_features).

    ⚠ CRITICAL DESIGN RULE — NO OUTCOME PROXIES IN CLUSTERING:
        propensity_30d, propensity_180d, willingness_score, capacity_score
        are EXPLICITLY EXCLUDED from CLUSTER_FEATURES.

        These are forward-looking / semi-target signals. Including them makes
        personas behave as propensity buckets:
          persona ≈ "people likely to pay" rather than "how people behave"

        Consequences of mixing them in:
          - Segmentation becomes latent supervised classification
          - Persona ≈ propensity → redundancy in the feature pipeline
          - ERV optimisation (which consumes propensity) becomes circular
          - Cluster interpretability degrades for collectors + compliance

        Rule: cluster on OBSERVABLE BEHAVIOUR only (what they do, not what
        models predict they'll do).

    Current features — three behavioural groups:
        PAYMENT BEHAVIOUR (what money moves):
            dpd_current, months_delinquent
            partial_payment_count_3m, broken_promise_count_3m
            days_since_last_payment, card_payment_pct_minimum_3m

        CONTACT / ENGAGEMENT (how they respond):
            contact_success_rate_3m, days_since_last_response
            digital_open_rate_3m, avg_response_lag_days

        BUREAU STRESS (structural financial position):
            ncb_total_revolving_util, ncb_enquiry_count_3m
            ncb_other_accounts_current

        TREATMENT HISTORY (collections exposure — prevents fatigue/anchoring):
            contact_attempts_total, days_since_last_offer
            prior_discount_max, n_prior_settlements_declined

    Do NOT include:
        - propensity_30d / propensity_180d / willingness_score / capacity_score
          (outcome proxies — see above)
        - low_confidence_flag (model uncertainty, not customer behaviour)
        - PII fields (name, phone, address, NationalID)
        - Future-dated fields (labels, outcomes)
        - SIGNAL_SEGMENT itself (clustering is done within segment)

ALGORITHM
    Default: KMeans (n_init=10, random_state=42).
    GMM option: PersonaClusterTrainer(algorithm="gmm") uses GaussianMixture.

    When to prefer GMM:
        - Customer profiles overlap significantly (soft boundaries expected)
        - Silhouette score < 0.25 with KMeans on your data
        - Portfolio has mixed Segment A/B accounts with similar scores

    When to keep KMeans:
        - Explainability to business stakeholders is important
        - Segment D (few features, sparse data) — GMM can be unstable
        - Cold start — KMeans is more robust with small n

    KMeans assumption of spherical clusters IS a limitation. If your feature
    space shows elongated or overlapping clusters (check PCA scatter plots),
    switch to GMM. Hierarchical clustering offline can validate KMeans structure.

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
from sklearn.mixture import GaussianMixture
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
#
# Two tiers of features — all are used when available (via enterprise join).
# PersonaClusterTrainer uses available = [c for c in CLUSTER_FEATURES if c in df.columns]
# so adding features here is safe — missing columns are skipped automatically.
#
# TIER 1 — Derived scores (always present after FeatureAgent runs)
#   These summarise propensity and engagement but lose granularity.
#   Clustering on scores alone is equivalent to a 2×2 matrix.
#
# TIER 2 — Raw behavioural signals (present when enterprise features are joined)
#   These give the clustering genuine behavioural resolution.
#   Without these, personas are only as differentiated as the 6 derived scores.
CLUSTER_FEATURES = [
    # ── Payment behaviour ─────────────────────────────────────────────────────
    # WHAT money moves — delinquency severity and payment trajectory
    "dpd_current",                # current DPD bucket — severity of delinquency
    "months_delinquent",          # how long account has been delinquent
    "partial_payment_count_3m",   # count of partial payments last 3 months
    "broken_promise_count_3m",    # broken PTPs — distinguishes intent vs. ability
    "days_since_last_payment",    # recency of any payment
    "card_payment_pct_minimum_3m",# payment as % of minimum due

    # ── Contact / engagement behaviour ───────────────────────────────────────
    # HOW the customer responds to outreach
    "contact_success_rate_3m",    # % of contact attempts that reached customer
    "days_since_last_response",   # recency of last meaningful engagement
    "digital_open_rate_3m",       # digital nudge open rate — passive engagement
    "avg_response_lag_days",      # how long customer takes to respond when contacted

    # ── Bureau structural position ────────────────────────────────────────────
    # WHAT financial stress exists beyond this account
    "ncb_total_revolving_util",   # total credit utilisation — financial stress
    "ncb_enquiry_count_3m",       # recent credit enquiries — financial search behaviour
    "ncb_other_accounts_current", # accounts still in good standing — capacity proxy

    # ── Treatment history ─────────────────────────────────────────────────────
    # WHAT collections exposure has already occurred — prevents over-contact
    # and discount anchoring (customer learns to wait for higher offers)
    "contact_attempts_total",     # total outbound attempts — contact fatigue signal
    "days_since_last_offer",      # recency of last settlement offer
    "prior_discount_max",         # highest discount ever offered — anchor risk
    "n_prior_settlements_declined",# declined offers — strategic behaviour signal

    # ⚠ DELIBERATELY EXCLUDED — see CONFIGURATION INSTRUCTIONS:
    #   propensity_30d, propensity_180d  — outcome proxies (target leakage)
    #   willingness_score, capacity_score — model outputs (circular dependency)
    #   low_confidence_flag              — model uncertainty, not behaviour
]

# Persona label map: cluster index → human-readable name.
# ⚠ Labels are assigned by ranking clusters on a BEHAVIOURAL composite score
# (contact_success_rate_3m + payment activity − broken_promise penalty).
# NOT ranked by propensity × capacity — that would reintroduce outcome leakage.
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
        algorithm: str = "kmeans",
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
            algorithm:                   Clustering algorithm. Options:
                                         "kmeans" (default) — fast, explainable,
                                           assumes spherical clusters.
                                         "gmm" — Gaussian Mixture Model, allows
                                           soft/overlapping clusters. Prefer when
                                           silhouette < 0.25 or profiles overlap.
                                         See CONFIGURATION INSTRUCTIONS for guidance.
        """
        if algorithm not in ("kmeans", "gmm"):
            raise ValueError(f"algorithm must be 'kmeans' or 'gmm', got: {algorithm!r}")
        self.n_clusters                  = n_clusters
        self.model_dir                   = Path(model_dir)
        self.random_state                = random_state
        self.min_pairwise_recovery_diff  = min_pairwise_recovery_diff
        self.skip_business_validation    = skip_business_validation
        self.algorithm                   = algorithm
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
            model, labels = self._fit_algorithm(X)

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

            self._save(seg, model, scaler, persona_map)

            results[seg] = {
                "silhouette":          round(sil, 4),
                "algorithm":           self.algorithm,
                "cluster_sizes":       pd.Series(labels).value_counts().to_dict(),
                "persona_map":         persona_map,
                "business_validation": biz_validation,
                "saved":               True,
            }
            logger.info(
                f"Segment {seg} | n={len(seg_df):,} | "
                f"algorithm={self.algorithm} | "
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
        result = self.predict_with_confidence(df)
        return result["behavioural_persona"]

    def predict_with_confidence(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Assigns BEHAVIOURAL_PERSONA and persona_confidence per row.

        persona_confidence is the ratio of distance-to-nearest-centroid over
        distance-to-own-centroid. Values close to 1.0 mean the account sits
        near its cluster boundary (low confidence); values > 2.0 are clearly
        assigned (high confidence).

        Args:
            df: DataFrame with signal_segment + CLUSTER_FEATURES columns.

        Returns:
            DataFrame with columns: behavioural_persona, persona_confidence
            (same index as df).
        """
        personas   = pd.Series("Disconnected", index=df.index, name="behavioural_persona")
        confidence = pd.Series(1.0,            index=df.index, name="persona_confidence")

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

            model, scaler, persona_map = self._load(seg)
            seg_df = df[seg_mask].copy()
            X, _   = self._prepare_features(seg_df, scaler=scaler)
            labels = self._predict_algorithm(model, X)

            personas[seg_mask] = [
                persona_map.get(lbl, "Disconnected") for lbl in labels
            ]

            # ── Cluster membership confidence ─────────────────────────────────
            # KMeans: confidence = dist_to_nearest_other_centroid / dist_to_own_centroid
            #   > 2.0 → clearly assigned; ≈ 1.0 → on the boundary
            # GMM: confidence = max posterior probability across clusters
            #   (already a soft assignment — use max probability directly)
            try:
                if isinstance(model, GaussianMixture):
                    probs = model.predict_proba(X)
                    confidence[seg_mask] = probs.max(axis=1)
                else:
                    centroids  = model.cluster_centers_
                    dists      = np.linalg.norm(
                        X[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2
                    )
                    own_dist   = dists[np.arange(len(labels)), labels]
                    dists_copy = dists.copy()
                    dists_copy[np.arange(len(labels)), labels] = np.inf
                    nearest_other_dist = dists_copy.min(axis=1)
                    confidence[seg_mask] = np.where(
                        own_dist > 0, nearest_other_dist / own_dist, 2.0
                    )
            except Exception as e:
                logger.warning(f"Confidence score computation failed for segment {seg}: {e}")

        return pd.DataFrame({"behavioural_persona": personas, "persona_confidence": confidence})

    # ── Internal ──────────────────────────────────────────────────────────────

    def _fit_algorithm(self, X: np.ndarray):
        """Fits the configured clustering algorithm. Returns (model, labels)."""
        if self.algorithm == "gmm":
            model = GaussianMixture(
                n_components=self.n_clusters,
                random_state=self.random_state,
                n_init=5,
                covariance_type="full",
            )
            model.fit(X)
            labels = model.predict(X)
        else:
            model = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                n_init=10,
            )
            labels = model.fit_predict(X)
        return model, labels

    def _predict_algorithm(self, model, X: np.ndarray) -> np.ndarray:
        """Predicts cluster labels. Works for both KMeans and GMM."""
        return model.predict(X)

    def _prepare_features(
        self, df: pd.DataFrame, scaler: StandardScaler = None
    ):
        available = [c for c in CLUSTER_FEATURES if c in df.columns]
        X = df[available].copy()

        # ── Missing indicator flags ───────────────────────────────────────────
        # A null value in these columns is itself a behavioural signal.
        # "days_since_last_payment is null" = this customer has NEVER paid.
        # Adding a binary flag preserves this signal after imputation, so the
        # algorithm can learn that "never paid" is distinct from "paid long ago".
        MISSING_INDICATOR_COLS = [
            "days_since_last_payment",    # null = never paid
            "days_since_last_response",   # null = never responded
            "contact_success_rate_3m",    # null = no contact history
            "days_since_last_offer",      # null = never received a settlement offer
        ]
        for col in MISSING_INDICATOR_COLS:
            if col in X.columns:
                X[f"_missing_{col}"] = X[col].isna().astype(float)

        # ── Domain-meaningful imputation ──────────────────────────────────────
        # Null in these columns means "never happened", not "data missing".
        # Median imputation would place absent customers near the average,
        # masking the behavioural signal (e.g. a customer who has never paid
        # should be far from customers who pay regularly).
        DOMAIN_FILL = {
            "days_since_last_payment":    999,  # never paid → worst case
            "days_since_last_response":   999,  # never responded → worst case
            "contact_success_rate_3m":      0,  # never reached → 0%
            "broken_promise_count_3m":      0,  # no PTPs → 0 broken
            "partial_payment_count_3m":     0,  # no partials → 0
            "avg_response_lag_days":      999,  # never responded → worst case
            "digital_open_rate_3m":         0,  # never opened → 0%
            "months_delinquent":            0,  # unknown → 0 (conservative)
            "dpd_current":                  0,  # unknown → 0 (conservative)
            "contact_attempts_total":       0,  # no contact history
            "days_since_last_offer":      999,  # never received offer
            "prior_discount_max":           0,  # never offered discount
            "n_prior_settlements_declined": 0,  # never declined an offer
        }
        for col in X.columns:
            if col.startswith("_missing_"):
                continue   # already binary — no imputation needed
            if col in DOMAIN_FILL:
                X[col] = X[col].fillna(DOMAIN_FILL[col])
            else:
                # Remaining numeric columns: median acceptable (rates, utilisation)
                X[col] = X[col].fillna(X[col].median())

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
        Ranks clusters by a BEHAVIOURAL composite score and maps to persona names.

        ⚠ Deliberately does NOT use propensity_30d × capacity_score for ranking.
        That approach reintroduces outcome leakage into persona naming:
            rank-by-propensity → "Cooperative" ≈ "high propensity"
            → segmentation and propensity become redundant

        Behavioural composite (all raw, no model outputs):
            + contact_success_rate_3m       — contactability (engagement axis)
            + card_payment_pct_minimum_3m   — payment activity (payment axis)
            + partial_payment_count_3m      — partial payment effort
            − broken_promise_count_3m × 0.2 — PTP reliability penalty
            − days_since_last_payment / 999  — payment recency penalty (normalised)

        Rank 0 (highest composite) → Cooperative (engaged + paying)
        Rank 1 → Stressed  (paying partially, some engagement)
        Rank 2 → Sporadic  (inconsistent across axes)
        Rank 3 → Disconnected (lowest — no contact, no payment)

        This is descriptive naming only — does NOT prescribe any action.
        """
        df = df.copy()
        df["_cluster"] = labels

        contact     = df.get("contact_success_rate_3m",    pd.Series(0.0, index=df.index)).fillna(0)
        pct_min     = df.get("card_payment_pct_minimum_3m",pd.Series(0.0, index=df.index)).fillna(0)
        partial_cnt = df.get("partial_payment_count_3m",   pd.Series(0.0, index=df.index)).fillna(0)
        broken      = df.get("broken_promise_count_3m",    pd.Series(0.0, index=df.index)).fillna(0)
        days_pay    = df.get("days_since_last_payment",    pd.Series(999, index=df.index)).fillna(999)

        df["_score"] = (
            contact
            + pct_min
            + partial_cnt * 0.1          # normalise count to similar scale
            - broken      * 0.2          # PTP reliability penalty
            - (days_pay / 999.0)         # recency penalty (0–1 range)
        )

        cluster_means = df.groupby("_cluster")["_score"].mean().sort_values(ascending=False)
        n = min(len(cluster_means), len(PERSONA_RANK_LABELS))
        return {
            int(cluster): PERSONA_RANK_LABELS[rank]
            for rank, cluster in enumerate(cluster_means.index[:n])
        }

    def _model_path(self, segment: str) -> Path:
        return self.model_dir / f"{segment}_persona_model.pkl"

    def _save(
        self, segment: str, model,
        scaler: StandardScaler, persona_map: dict
    ) -> None:
        path = self._model_path(segment)
        joblib.dump({
            "model":       model,
            "algorithm":   self.algorithm,
            "scaler":      scaler,
            "persona_map": persona_map,
        }, path)
        logger.info(f"Saved persona model ({self.algorithm}) → {path}")

    def _load(self, segment: str):
        obj = joblib.load(self._model_path(segment))
        # Backward compat: older saves used "km" key
        model = obj.get("model") or obj.get("km")
        return model, obj["scaler"], obj["persona_map"]
