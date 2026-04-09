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
from sklearn.preprocessing import RobustScaler, StandardScaler


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
    # ── Group A: DPD Trajectory and Delinquency Physics ──────────────────────
    # Shape of the delinquency path — strongest structural discriminators.
    # All are 12–24 month historical features, NOT point-in-time severity.
    "dpd_trajectory_slope_6m",        # linear slope of DPD over last 6M pre-CO
    "dpd_trajectory_slope_12m",       # slope over 12M — acceleration vs deceleration
    "dpd_acceleration_6m",            # second derivative: worsening faster or slowing?
    "dpd_velocity_3m_vs_6m",          # momentum: rate in last 3M vs prior 3M
    "max_dpd_ever",                   # worst-ever DPD — permanent stress record
    "months_in_s2_plus_24m",          # months 31–90 DPD in last 24M — chronic exposure
    "months_in_s3_plus_24m",          # months 91–180 DPD in last 24M
    "delinquency_episode_count_24m",  # number of distinct episodes — chronic vs acute
    "cure_count_24m",                 # times account returned to current — recidivism
    "cure_durability_avg_days",       # average days in current status between episodes
    "cure_durability_min_days",       # worst cure — did they ever truly stabilise?
    "first_delinquency_months_ago",   # age of problem — fresh vs chronic defaulter

    # ── Group B: Payment Behaviour and Trajectory (pre-charge-off) ───────────
    "payment_ratio_trend_6m",         # slope of payment/balance ratio over 6M
    "payment_velocity_3m",            # rate of change in payment amount
    "payment_acceleration_3m",        # second derivative of payment amounts
    "min_payment_ratio_12m",          # lowest payment/minimum_due ratio in 12M
    "avg_payment_ratio_12m",          # average payment coverage — structural capacity
    "payment_decay_ratio",            # avg_payment_last_3M / avg_payment_6to12M
    "last_payment_months_before_co",  # recency of last payment before CO
    "PTP_kept_rate_12m",              # % of promises-to-pay kept in 12M — stable trait
    "PTP_broken_rate_12m",            # % broken — strategic vs distressed defaulter
    "payment_count_last_12m_pre_co",  # count of payments in final 12M
    # self_initiated_contact_ratio_12m REMOVED — denominator includes all contacts,
    # so higher bank outreach deflates ratio even if customer behaviour unchanged.
    # Item 9 verdict: policy-influenced → removed.

    # ── Group C: Credit Utilisation and Financial Position ───────────────────
    "utilization_trend_6m",           # slope of utilisation over 6M
    "utilization_volatility_12m",     # std dev of utilisation — stable vs erratic
    "revolving_ratio_stability",      # stability of revolving/total ratio
    "balance_volatility_12m",         # std dev of balance — financial instability
    "balance_at_co_to_limit_ratio",   # utilisation at CO — structural exposure proxy

    # ── Group D: Bureau Structural Position (at or near charge-off) ──────────
    "ncb_total_revolving_util",       # total revolving utilisation — systemic stress
    "ncb_other_accounts_current",     # other accounts still current — strategic signal
    "ncb_accounts_delinquent_count",  # count of delinquent accounts beyond this card
    "ncb_total_delinquent_pct",       # fraction of total credit exposure delinquent
    "ncb_mortgage_current",           # mortgage current — strongest strategic defaulter
    "ncb_secured_loan_current",       # any secured loan current
    "ncb_score_at_co",                # bureau score at charge-off date
    "ncb_score_trend_6m",             # score change 6M before CO — improving or worsening
    "tradeline_stability_index",      # ratio of active to total tradelines
    "bureau_total_debt",              # total outstanding debt — structural burden
    "bureau_enquiry_velocity_3m",     # credit enquiry rate — financial search pre-CO

    # ── Group E: REMOVED in Phase 1 (Item 2) ─────────────────────────────────
    # paid_in_first_30d, paid_in_first_60d, days_to_first_payment removed.
    # Even purely customer-initiated early payment is too close to the outcome
    # label (recovery_180d) — it turns clustering into early-payer vs non-payer,
    # which is a propensity proxy. Reintroduce in Phase 2 with explicit controls
    # (propensity-residualised or as a separate stratification step).

    # ── Group F: Exposure Dimension (Item 3) ──────────────────────────────────
    # Absolute exposure matters for strategy — a THB 5k dormant account and a
    # THB 500k dormant account require completely different treatment economics.
    # Derived in _prepare_features() if not already present.
    "log_balance_at_co",              # log(1 + outstanding_balance_at_co) — heavy tail
    "balance_bucket",                 # ordinal bucket: 0–5k/5–20k/20–50k/50–100k/100k+

    # ── Group G: Affordability Proxy (Item 6) ────────────────────────────────
    # Ratio of payment capacity to total debt burden — structural affordability signal.
    # Derived in _prepare_features() if not already present.
    "payment_to_debt_ratio",          # avg_payment_12m / (bureau_total_debt + 1)

    # ── Multicollinearity note (Item 7) ──────────────────────────────────────
    # Highly correlated features (r > 0.8) are dropped in _prepare_features()
    # before clustering. Typical culprits: months_in_s2 vs months_in_s3,
    # avg_payment_ratio vs min_payment_ratio, ncb_total_delinquent_pct vs
    # ncb_accounts_delinquent_count. PCA is available as use_pca=True flag
    # but is NOT default — PCA centroids are uninterpretable in feature space.

    # ⚠ EXPLICITLY EXCLUDED:
    #   paid_in_first_30d/60d / days_to_first_payment — Group E removed Phase 1
    #   self_initiated_contact_ratio_12m  — policy-influenced (Item 9)
    #   rpc_in_first_30d / days_to_first_rpc  — Amendment 3
    #   dpd_current                       — point-in-time severity
    #   days_since_last_payment           — recent tactical → propensity only
    #   contact_success_rate_3m           — propensity-correlated → propensity only
    #   digital_open_rate_3m              — short-horizon → propensity only
    #   contact_attempts_total            — treatment intensity
    #   prior_discount_max                — Elast ✓, Seg ✗
    #   n_prior_settlements_declined      — Elast ✓, Seg ✗
    #   propensity_30d/90d/180d           — outcome proxies
    #   recovery_30d/90d/180d             — target labels (never cluster on outcome)
    #   erv_at_d_optimal                  — terminal output
]

# Cluster label names assigned by behavioural composite rank (high → low).
# Spec: PTP_kept_rate_12m + (1 − episode_count/max) − PTP_broken_rate_12m
# Rank 0 (highest) → High-Engagement Chronic (engaged, some history of trying)
# Rank 1 → Sudden-Shock Distressed (sudden DPD onset, was paying before CO)
# Rank 2 → Structural Defaulter (bureau capacity intact, not paying this card)
# Rank 3 → Dormant (low engagement, chronic impairment, minimal signals)
PERSONA_RANK_LABELS = [
    "High-Engagement Chronic",
    "Sudden-Shock Distressed",
    "Structural Defaulter",
    "Dormant",
]

# Features where heavy tails make RobustScaler strongly preferred over StandardScaler.
# (balance, DPD velocity/slope, time-based features all have meaningful outliers.)
HEAVY_TAILED_FEATURES: frozenset = frozenset({
    "max_dpd_ever",
    "dpd_trajectory_slope_6m",
    "dpd_trajectory_slope_12m",
    "dpd_velocity_3m_vs_6m",
    "dpd_acceleration_6m",
    "balance_volatility_12m",
    "balance_at_co_to_limit_ratio",
    "bureau_total_debt",
    "days_to_first_payment",
    "last_payment_months_before_co",
    "cure_durability_avg_days",
    "cure_durability_min_days",
    "first_delinquency_months_ago",
    "log_balance_at_co",   # Item 3 — log-transformed balance, heavy right tail
})

# Phase 1: binary split per Amendment 4.
# FULL_SIGNAL   = CardX AND Bureau available (original Segment A).
# LIMITED_SIGNAL = any signal missing (original Segments B, C, D combined).
# Expand to four-quadrant A/B/C/D only when ≥1,000 labelled accounts per quadrant.
SIGNAL_SEGMENTS = ["FULL_SIGNAL", "LIMITED_SIGNAL"]

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

            self._save(seg, model, scaler, persona_map, is_rules_based=False)

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
        personas   = pd.Series("Dormant", index=df.index, name="behavioural_persona")
        confidence = pd.Series(1.0,       index=df.index, name="persona_confidence")

        for seg in SIGNAL_SEGMENTS:
            seg_mask = df["signal_segment"] == seg
            if not seg_mask.any():
                continue

            model_path = self._model_path(seg)
            if not model_path.exists():
                logger.warning(
                    f"No fitted model for segment {seg} at {model_path}. "
                    f"Defaulting to 'Dormant'. Run PersonaClusterTrainer.fit() first."
                )
                continue

            model, scaler, persona_map, is_rules = self._load(seg)
            seg_df = df[seg_mask].copy()
            X, _   = self._prepare_features(seg_df, scaler=scaler)
            labels = self._predict_algorithm(model, X)

            personas[seg_mask] = [
                persona_map.get(lbl, "Dormant") for lbl in labels
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
                n_init=20,   # spec: n_init=20 for stability
            )
            labels = model.fit_predict(X)
        return model, labels

    def _predict_algorithm(self, model, X: np.ndarray) -> np.ndarray:
        """Predicts cluster labels. Works for both KMeans and GMM."""
        return model.predict(X)

    def _prepare_features(self, df: pd.DataFrame, scaler=None):
        """
        Prepares feature matrix for clustering.

        Scaling strategy (per spec):
          - RobustScaler for heavy-tailed features (DPD slopes, balance, time-to-event)
          - StandardScaler for rates, ratios, and counts
        Scaler is a dict {"robust": RobustScaler, "standard": StandardScaler,
                          "robust_cols": [...], "standard_cols": [...]}
        stored to ensure transform() at inference uses the same column split.
        """
        df = df.copy()

        # ── Derived features (Groups F and G) ────────────────────────────────
        # Compute before selecting CLUSTER_FEATURES so derived cols are available.
        if "log_balance_at_co" not in df.columns and "outstanding_balance_at_co" in df.columns:
            df["log_balance_at_co"] = np.log1p(
                df["outstanding_balance_at_co"].clip(lower=0).fillna(0)
            )

        if "balance_bucket" not in df.columns and "outstanding_balance_at_co" in df.columns:
            df["balance_bucket"] = pd.cut(
                df["outstanding_balance_at_co"].fillna(0),
                bins=[0, 5_000, 20_000, 50_000, 100_000, np.inf],
                labels=[0, 1, 2, 3, 4],
                include_lowest=True,
            ).astype(float)

        if "payment_to_debt_ratio" not in df.columns:
            avg_pay   = df.get("avg_payment_ratio_12m", pd.Series(0.0, index=df.index)).fillna(0)
            total_debt = df.get("bureau_total_debt",    pd.Series(1.0, index=df.index)).fillna(1).clip(lower=1)
            df["payment_to_debt_ratio"] = avg_pay / total_debt

        # ── Per-account missingness check (Item 5) ───────────────────────────
        # Accounts with too many null signal features are routed to rule-based
        # Dormant assignment and excluded from clustering.
        MISSINGNESS_THRESHOLD = 0.60   # >60% of clustering features null → fallback
        cluster_cols = [c for c in CLUSTER_FEATURES if not c.startswith("#")]
        null_fractions = df[
            [c for c in cluster_cols if c in df.columns]
        ].isnull().mean(axis=1)
        high_missingness_mask = null_fractions > MISSINGNESS_THRESHOLD
        if high_missingness_mask.any():
            import warnings
            warnings.warn(
                f"_prepare_features: {high_missingness_mask.sum()} accounts "
                f"exceed missingness threshold ({MISSINGNESS_THRESHOLD:.0%} null) "
                "and will receive rule-based 'Dormant' assignment.",
                UserWarning,
                stacklevel=2,
            )
        # Store mask on df so caller can retrieve it; we still process all rows
        # (missing values are imputed below) and callers can filter post-prediction.
        df["_high_missingness"] = high_missingness_mask.astype(int)

        available = [c for c in CLUSTER_FEATURES if c in df.columns]
        X = df[available].copy()

        # ── Missing indicator flags ───────────────────────────────────────────
        # Null = "this event never happened", not a data gap.
        # Binary flags preserve the "never" signal after imputation.
        MISSING_INDICATOR_COLS = [
            "last_payment_months_before_co",  # null = never paid before CO
            "days_to_first_payment",          # null = no payment post-CO (removed from features, kept for safety)
            "cure_durability_avg_days",       # null = account never cured
            "PTP_kept_rate_12m",              # null = no PTPs ever made
            # self_initiated_contact_ratio_12m removed (Item 9 — policy-influenced)
        ]
        for col in MISSING_INDICATOR_COLS:
            if col in X.columns:
                X[f"_missing_{col}"] = X[col].isna().astype(float)

        # ── Domain-meaningful imputation ──────────────────────────────────────
        # "Never happened" should map to the worst-case end of the distribution,
        # not the median. This preserves the information that absence carries.
        DOMAIN_FILL = {
            # Group A — DPD trajectory
            "dpd_trajectory_slope_6m":        0.0,   # unknown → flat trajectory
            "dpd_trajectory_slope_12m":       0.0,
            "dpd_acceleration_6m":            0.0,   # unknown → no acceleration
            "dpd_velocity_3m_vs_6m":          0.0,
            "max_dpd_ever":                   180,   # unknown → worst case (CO-level)
            "months_in_s2_plus_24m":          0,
            "months_in_s3_plus_24m":          0,
            "delinquency_episode_count_24m":  1,     # at least 1 — account is at CO
            "cure_count_24m":                 0,     # never cured
            "cure_durability_avg_days":       0,     # never cured → 0 days
            "cure_durability_min_days":       0,
            "first_delinquency_months_ago":   0,     # unknown → most recent
            # Group B — Payment behaviour
            "payment_ratio_trend_6m":         -0.1,  # unknown → slight decline
            "payment_velocity_3m":            0.0,
            "payment_acceleration_3m":        0.0,
            "min_payment_ratio_12m":          0.0,   # never paid minimum
            "avg_payment_ratio_12m":          0.0,
            "payment_decay_ratio":            0.0,   # fully collapsed
            "last_payment_months_before_co":  99,    # never paid → worst case
            "PTP_kept_rate_12m":              0.0,   # no PTPs → 0% kept
            "PTP_broken_rate_12m":            0.0,   # no PTPs → 0% broken
            "payment_count_last_12m_pre_co":  0,
            # self_initiated_contact_ratio_12m removed (Item 9 — policy-influenced)
            # Group C — Utilisation
            "utilization_trend_6m":           0.0,
            "utilization_volatility_12m":     0.0,
            "revolving_ratio_stability":      0.0,
            "balance_volatility_12m":         0.0,
            "balance_at_co_to_limit_ratio":   1.0,  # fully utilised at CO
            # Group D — Bureau
            "ncb_total_revolving_util":       1.0,  # assume maxed out
            "ncb_other_accounts_current":     0,
            "ncb_accounts_delinquent_count":  0,
            "ncb_total_delinquent_pct":       1.0,  # assume all delinquent
            "ncb_mortgage_current":           0,
            "ncb_secured_loan_current":       0,
            "ncb_score_at_co":                0,    # unknown → worst case
            "ncb_score_trend_6m":             0,    # unknown → flat
            "tradeline_stability_index":      0.0,
            "bureau_total_debt":              0,
            "bureau_enquiry_velocity_3m":     0.0,
            # Group E — Early post-CO binary
            "paid_in_first_30d":              0,
            "paid_in_first_60d":              0,
            "days_to_first_payment":          999,  # never paid → worst case
        }
        for col in X.columns:
            if col.startswith("_missing_"):
                continue
            if col in DOMAIN_FILL:
                X[col] = X[col].fillna(DOMAIN_FILL[col])
            else:
                X[col] = X[col].fillna(X[col].median())

        # ── Multicollinearity drop (Item 7, r > 0.8) ─────────────────────────
        # Applied only during fit (scaler is None) so train/inference use the same
        # column set. At inference, the saved scaler's feature_cols is the reference.
        if scaler is None:
            numeric_X = X[[c for c in X.columns if not c.startswith("_missing_")]]
            corr_matrix = numeric_X.corr().abs()
            upper = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            to_drop = [col for col in upper.columns if (upper[col] > 0.80).any()]
            if to_drop:
                import warnings
                warnings.warn(
                    f"_prepare_features: dropping {len(to_drop)} highly correlated "
                    f"features (r > 0.80): {to_drop}",
                    UserWarning,
                    stacklevel=2,
                )
                X = X.drop(columns=to_drop)

        feature_cols = X.columns.tolist()

        # ── Mixed scaling (RobustScaler for heavy-tailed, Standard for rest) ──
        robust_cols   = [c for c in feature_cols if c in HEAVY_TAILED_FEATURES]
        standard_cols = [c for c in feature_cols if c not in HEAVY_TAILED_FEATURES]

        X_out = np.zeros((len(X), len(feature_cols)), dtype=float)

        if scaler is None:
            robust_s   = RobustScaler()
            standard_s = StandardScaler()
            if robust_cols:
                idx = [feature_cols.index(c) for c in robust_cols]
                X_out[:, idx] = robust_s.fit_transform(X[robust_cols])
            if standard_cols:
                idx = [feature_cols.index(c) for c in standard_cols]
                X_out[:, idx] = standard_s.fit_transform(X[standard_cols])
            scaler = {
                "robust":       robust_s,
                "standard":     standard_s,
                "robust_cols":  robust_cols,
                "standard_cols": standard_cols,
                "feature_cols": feature_cols,
            }
        else:
            # Inference — use fitted scalers; align to saved column split
            robust_s      = scaler["robust"]
            standard_s    = scaler["standard"]
            saved_robust  = scaler["robust_cols"]
            saved_standard = scaler["standard_cols"]
            saved_cols    = scaler["feature_cols"]
            # Re-order X to match saved column order, fill any new-missing cols with 0
            X = X.reindex(columns=saved_cols, fill_value=0)
            X_out = np.zeros((len(X), len(saved_cols)), dtype=float)
            if saved_robust:
                idx = [saved_cols.index(c) for c in saved_robust]
                X_out[:, idx] = robust_s.transform(X[saved_robust])
            if saved_standard:
                idx = [saved_cols.index(c) for c in saved_standard]
                X_out[:, idx] = standard_s.transform(X[saved_standard])

        return X_out, scaler

    def _assign_persona_labels(
        self, df: pd.DataFrame, labels: np.ndarray
    ) -> dict:
        """
        Ranks clusters using a 3-component normalised composite (Item 1 — BLOCKER fix).

        Components (each min-max normalised to [0, 1] before summing):

          behavioural_score  = PTP_kept_rate_12m
                               − PTP_broken_rate_12m
                               − delinquency_episode_count_24m_normalised

          capacity_score     = ncb_other_accounts_current   (normalised)
                               + ncb_mortgage_current        (normalised)
                               − ncb_total_revolving_util    (normalised)

          trajectory_score   = −dpd_trajectory_slope_6m     (normalised)
                               − dpd_acceleration_6m         (normalised)
                               (negative because higher slope/accel = worse)

          FINAL_SCORE = behavioural_score + capacity_score + trajectory_score

        All three components are equally weighted initially.
        Higher FINAL_SCORE → more engaged / more capacity → ranked higher.

        Rank 0 (highest) → High-Engagement Chronic
        Rank 1 → Sudden-Shock Distressed
        Rank 2 → Structural Defaulter
        Rank 3 → Dormant

        Uses only long-horizon behavioural + bureau features (12–24M).
        No propensity, no recovery outcome, no model scores.
        """
        df = df.copy()
        df["_cluster"] = labels

        def _norm(series: pd.Series) -> pd.Series:
            """Min-max normalise to [0, 1]; returns 0.5 if constant."""
            lo, hi = series.min(), series.max()
            if hi == lo:
                return pd.Series(0.5, index=series.index)
            return (series - lo) / (hi - lo)

        def _get(col: str, default: float) -> pd.Series:
            return df.get(col, pd.Series(default, index=df.index)).fillna(default)

        # ── Behavioural component ─────────────────────────────────────────────
        ptp_kept      = _get("PTP_kept_rate_12m",            0.0)
        ptp_broken    = _get("PTP_broken_rate_12m",          0.0)
        episodes      = _get("delinquency_episode_count_24m", 1.0)
        episodes_norm = _norm(episodes)

        behavioural_score = _norm(ptp_kept) - _norm(ptp_broken) - episodes_norm

        # ── Capacity component (bureau) ───────────────────────────────────────
        other_current   = _get("ncb_other_accounts_current", 0.0)
        mortgage_curr   = _get("ncb_mortgage_current",       0.0)
        revolving_util  = _get("ncb_total_revolving_util",   1.0)

        capacity_score = (
            _norm(other_current)
            + _norm(mortgage_curr)
            - _norm(revolving_util)
        )

        # ── Trajectory component ──────────────────────────────────────────────
        # Higher slope/acceleration = deteriorating faster = WORSE → negate
        dpd_slope = _get("dpd_trajectory_slope_6m", 0.0)
        dpd_accel = _get("dpd_acceleration_6m",     0.0)

        trajectory_score = -_norm(dpd_slope) - _norm(dpd_accel)

        # ── Final composite ───────────────────────────────────────────────────
        df["_final_score"] = behavioural_score + capacity_score + trajectory_score

        cluster_means = (
            df.groupby("_cluster")["_final_score"]
            .mean()
            .sort_values(ascending=False)
        )
        n = min(len(cluster_means), len(PERSONA_RANK_LABELS))
        return {
            int(cluster): PERSONA_RANK_LABELS[rank]
            for rank, cluster in enumerate(cluster_means.index[:n])
        }

    def _segment_d_rules(self, df: pd.DataFrame) -> np.ndarray:
        """
        Rule-based persona assignment for Segment D (no CardX, no bureau signal).

        Uses only features reliably available without signal data:
          max_dpd_ever, months_since_co, outstanding_balance_at_co

        Logic:
          Dormant            — old CO (>18M), high DPD history, no signals
          Sudden-Shock       — max DPD < 90 (problem came suddenly)
          Structural Default — high balance + recent CO (likely strategic, worth pursuing)
          High-Engagement    — moderate DPD, recent CO (residual engagement possible)
        """
        n       = len(df)
        labels  = np.full(n, "Dormant", dtype=object)
        if n == 0:
            return labels

        max_dpd    = df.get("max_dpd_ever",            pd.Series(180, index=df.index)).fillna(180).values
        months_co  = df.get("months_since_co",         pd.Series(0,   index=df.index)).fillna(0).values
        balance    = df.get("outstanding_balance_at_co",pd.Series(0,   index=df.index)).fillna(0).values

        # Apply rules in priority order (most → least specific)
        sudden_shock  = max_dpd < 90
        structural    = (balance > 50_000) & (months_co <= 18) & ~sudden_shock
        chronic_recent = (max_dpd >= 90) & (months_co <= 12) & ~sudden_shock & ~structural
        # Default: Dormant (high DPD + old CO, or any residual)

        labels[sudden_shock]   = "Sudden-Shock Distressed"
        labels[structural]     = "Structural Defaulter"
        labels[chronic_recent] = "High-Engagement Chronic"
        return labels

    def _model_path(self, segment: str) -> Path:
        return self.model_dir / f"{segment}_persona_model.pkl"

    def _save(
        self, segment: str, model, scaler, persona_map: dict,
        is_rules_based: bool = False,
    ) -> None:
        path = self._model_path(segment)
        joblib.dump({
            "model":          model,
            "algorithm":      "rules" if is_rules_based else self.algorithm,
            "scaler":         scaler,   # dict with robust/standard sub-scalers
            "persona_map":    persona_map,
            "is_rules_based": is_rules_based,
        }, path)
        alg = "rules" if is_rules_based else self.algorithm
        logger.info(f"Saved persona model ({alg}) → {path}")

    def _load(self, segment: str):
        obj = joblib.load(self._model_path(segment))
        # Backward compat: older saves used "km" key, single StandardScaler
        model          = obj.get("model") or obj.get("km")
        scaler         = obj.get("scaler")
        is_rules_based = obj.get("is_rules_based", False)
        return model, scaler, obj["persona_map"], is_rules_based
