"""
ClusterBusinessValidator
========================
Validates that persona clusters are BUSINESS-meaningful, not just statistically
separated in feature space.

Core problem this solves:
  A clustering algorithm can produce 4 geometrically distinct clusters with a
  high silhouette score, but if all 4 clusters have similar recovery rates
  (e.g. 28%, 30%, 29%, 31%), the segmentation adds no business value.
  A portfolio manager cannot act differently on segments that recover at the
  same rate.

Validation principle:
  "Clusters are valid only if they are ACTIONABLY different — meaning they
   differ on recovery outcomes in a way that justifies treating them
   differently in the collections strategy."

──────────────────────────────────────────────────────────────────────────────
CONFIGURATION INSTRUCTIONS
──────────────────────────────────────────────────────────────────────────────
BUSINESS_METRICS
    The outcome columns used to assess cluster differentiation.
    All must be present in the validation DataFrame.

    recovery_30d:    payment rate within 30 days (short-term tactical signal)
    recovery_90d:    payment rate within 90 days (mid-term)
    recovery_180d:   payment rate within 180 days (MOST IMPORTANT — long-term)
    recovery_amount: mean THB recovered (economic value)
    erv_at_d_optimal:expected recovery value (model output)

    ⚠ recovery_180d is the critical metric. Clusters that are NOT differentiated
      on 180d recovery rate are not useful for collections decisioning — actions
      are built around 180d horizon.

MIN_PAIRWISE_RECOVERY_DIFF (default: 0.05 = 5 percentage points)
    Minimum absolute difference in recovery_180d between ANY two clusters
    for the validation to pass. If the best and worst performing cluster
    differ by less than this threshold, clusters are not business-valid.

    Example: clusters with recovery rates [0.28, 0.30, 0.29, 0.31]
    → max pairwise diff = 0.03 < 0.05 → FAIL
    Business interpretation: cannot justify different treatment strategies.

    How to change:
        ClusterBusinessValidator(min_pairwise_recovery_diff=0.08)
    Increase if you want stricter business separation requirements.
    Decrease if your portfolio has naturally low recovery variance.

MIN_CLUSTER_SIZE_PCT (default: 0.05 = 5% of segment)
    Minimum fraction of accounts that must fall in each cluster.
    Clusters smaller than this are likely noise — merge with nearest cluster
    or reduce n_clusters.

STAT_SIGNIFICANCE_ALPHA (default: 0.05)
    p-value threshold for Kruskal-Wallis test across clusters.
    Note: statistical significance is NECESSARY but NOT SUFFICIENT.
    A cluster split can be statistically significant but economically trivial.
    Both stat test AND min_pairwise_recovery_diff must pass.

RECOMMENDATION LOGIC
    PASS:           All checks pass — clusters are business-valid
    REDUCE_CLUSTERS:max pairwise diff too small → try n_clusters - 1
    MERGE_SMALL:    One or more clusters below MIN_CLUSTER_SIZE_PCT
    FAIL:           No significant differentiation on any recovery metric
                    → do not use these clusters; fall back to rule-based personas
──────────────────────────────────────────────────────────────────────────────
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

BUSINESS_METRICS = [
    "recovery_30d",
    "recovery_90d",
    "recovery_180d",    # primary — most important for collections strategy
    "recovery_amount",
    "erv_at_d_optimal",
]

DEFAULT_MIN_PAIRWISE_RECOVERY_DIFF = 0.05   # 5pp on recovery_180d
DEFAULT_MIN_CLUSTER_SIZE_PCT       = 0.05   # 5% of segment
DEFAULT_STAT_ALPHA                 = 0.05


class ClusterBusinessValidator:

    def __init__(
        self,
        min_pairwise_recovery_diff: float = DEFAULT_MIN_PAIRWISE_RECOVERY_DIFF,
        min_cluster_size_pct: float       = DEFAULT_MIN_CLUSTER_SIZE_PCT,
        stat_alpha: float                 = DEFAULT_STAT_ALPHA,
        primary_metric: str               = "recovery_180d",
    ):
        """
        Args:
            min_pairwise_recovery_diff: Min abs diff between best and worst cluster
                                        on primary_metric for PASS.
                                        Default: 0.05 (5pp). See CONFIGURATION.
            min_cluster_size_pct:       Min fraction of segment per cluster.
                                        Default: 0.05 (5%). See CONFIGURATION.
            stat_alpha:                 p-value threshold for Kruskal-Wallis.
            primary_metric:             Recovery metric used as the primary
                                        differentiation check. Default: recovery_180d.
        """
        self.min_pairwise_recovery_diff = min_pairwise_recovery_diff
        self.min_cluster_size_pct       = min_cluster_size_pct
        self.stat_alpha                 = stat_alpha
        self.primary_metric             = primary_metric

    def validate(
        self,
        df: pd.DataFrame,
        cluster_col: str = "behavioural_persona",
        segment: Optional[str] = None,
    ) -> dict:
        """
        Runs full business validation on cluster assignments.

        Args:
            df:          DataFrame with cluster_col + BUSINESS_METRICS columns.
                         Must contain outcome columns (recovery_30d, recovery_180d etc.)
                         joined from historical data — cannot validate without outcomes.
            cluster_col: Column containing cluster labels.
            segment:     SIGNAL_SEGMENT being validated (for logging only).

        Returns:
            dict with keys:
                passed (bool)
                recommendation (str): PASS / REDUCE_CLUSTERS / MERGE_SMALL / FAIL
                primary_metric_by_cluster (dict)
                max_pairwise_diff (float)
                stat_test_results (dict per metric)
                cluster_sizes (dict)
                cluster_profiles (DataFrame as dict)
                failure_reasons (list of str)
        """
        seg_label     = segment or "ALL"
        failure_reasons = []
        available_metrics = [m for m in BUSINESS_METRICS if m in df.columns]

        if not available_metrics:
            raise ValueError(
                f"No business metrics found in DataFrame. "
                f"Required at least one of: {BUSINESS_METRICS}. "
                f"Found columns: {list(df.columns)}"
            )
        if self.primary_metric not in df.columns:
            raise ValueError(
                f"Primary metric '{self.primary_metric}' not in DataFrame. "
                f"Available metrics: {available_metrics}. "
                f"Cannot validate without {self.primary_metric} — "
                f"this is the most important recovery signal for collections."
            )
        if cluster_col not in df.columns:
            raise ValueError(f"Cluster column '{cluster_col}' not found in DataFrame.")

        clusters      = df[cluster_col].unique()
        n_clusters    = len(clusters)
        cluster_sizes = df[cluster_col].value_counts(normalize=True).to_dict()

        # ── Check 1: Cluster size ─────────────────────────────────────────────
        small_clusters = [
            c for c, pct in cluster_sizes.items()
            if pct < self.min_cluster_size_pct
        ]
        if small_clusters:
            failure_reasons.append(
                f"Clusters too small (<{self.min_cluster_size_pct:.0%}): {small_clusters}. "
                f"Reduce n_clusters or merge small clusters."
            )

        # ── Cluster profiles on business metrics ─────────────────────────────
        profiles = df.groupby(cluster_col)[available_metrics].agg(["mean", "median", "std"])
        profiles.columns = ["_".join(c) for c in profiles.columns]
        profiles["n"]    = df[cluster_col].value_counts()

        # ── Check 2: Primary metric pairwise separation ───────────────────────
        primary_by_cluster = df.groupby(cluster_col)[self.primary_metric].mean().to_dict()
        values             = list(primary_by_cluster.values())
        max_pairwise_diff  = max(values) - min(values) if len(values) > 1 else 0.0

        if max_pairwise_diff < self.min_pairwise_recovery_diff:
            failure_reasons.append(
                f"Clusters NOT differentiated on {self.primary_metric}: "
                f"max pairwise diff = {max_pairwise_diff:.3f} "
                f"< threshold {self.min_pairwise_recovery_diff:.3f}. "
                f"Cluster recovery rates: {primary_by_cluster}. "
                f"Recommendation: reduce n_clusters or collect more labelled data."
            )

        # ── Check 3: Statistical significance per metric ──────────────────────
        stat_results = {}
        for metric in available_metrics:
            groups = [
                df.loc[df[cluster_col] == c, metric].dropna().values
                for c in clusters
                if len(df.loc[df[cluster_col] == c, metric].dropna()) >= 5
            ]
            if len(groups) < 2:
                stat_results[metric] = {"stat": None, "p_value": None, "significant": False}
                continue

            stat, p = stats.kruskal(*groups)
            significant = p < self.stat_alpha
            stat_results[metric] = {
                "stat":        round(float(stat), 4),
                "p_value":     round(float(p), 6),
                "significant": significant,
            }

            if metric == self.primary_metric and not significant:
                failure_reasons.append(
                    f"Kruskal-Wallis test NOT significant for {metric}: "
                    f"p={p:.4f} > {self.stat_alpha}. "
                    f"Clusters do not differ on primary recovery metric."
                )

        # ── Check 4: Effect size — are differences economically meaningful? ───
        # Cohen's d between best and worst cluster on primary metric
        best_cluster  = max(primary_by_cluster, key=primary_by_cluster.get)
        worst_cluster = min(primary_by_cluster, key=primary_by_cluster.get)
        best_vals  = df.loc[df[cluster_col] == best_cluster,  self.primary_metric].dropna()
        worst_vals = df.loc[df[cluster_col] == worst_cluster, self.primary_metric].dropna()

        pooled_std = np.sqrt(
            (best_vals.std() ** 2 + worst_vals.std() ** 2) / 2
        ) if len(best_vals) > 1 and len(worst_vals) > 1 else 1.0

        cohens_d = float(
            (best_vals.mean() - worst_vals.mean()) / pooled_std
        ) if pooled_std > 0 else 0.0

        if abs(cohens_d) < 0.2:
            failure_reasons.append(
                f"Effect size too small: Cohen's d = {cohens_d:.3f} "
                f"between best ({best_cluster}) and worst ({worst_cluster}) cluster. "
                f"Difference is statistically significant but economically trivial."
            )

        # ── Determine recommendation ──────────────────────────────────────────
        passed = len(failure_reasons) == 0

        if passed:
            recommendation = "PASS"
        elif small_clusters and max_pairwise_diff >= self.min_pairwise_recovery_diff:
            recommendation = "MERGE_SMALL"
        elif max_pairwise_diff < self.min_pairwise_recovery_diff:
            recommendation = "REDUCE_CLUSTERS"
        else:
            recommendation = "FAIL"

        # ── Log result ────────────────────────────────────────────────────────
        log_level = "info" if passed else "warning"
        getattr(logger, log_level)(
            f"[Segment {seg_label}] Cluster business validation: {recommendation} | "
            f"n_clusters={n_clusters} | "
            f"max_{self.primary_metric}_diff={max_pairwise_diff:.3f} | "
            f"cohens_d={cohens_d:.3f} | "
            f"cluster_recovery={primary_by_cluster}"
        )
        if not passed:
            for reason in failure_reasons:
                logger.warning(f"[Segment {seg_label}] FAIL reason: {reason}")

        return {
            "passed":                      passed,
            "recommendation":              recommendation,
            "segment":                     seg_label,
            "n_clusters":                  n_clusters,
            "primary_metric":              self.primary_metric,
            "primary_metric_by_cluster":   primary_by_cluster,
            "max_pairwise_diff":           round(max_pairwise_diff, 4),
            "cohens_d":                    round(cohens_d, 4),
            "stat_test_results":           stat_results,
            "cluster_sizes":               cluster_sizes,
            "cluster_profiles":            profiles.to_dict(),
            "failure_reasons":             failure_reasons,
        }

    def validate_all_segments(
        self,
        df: pd.DataFrame,
        cluster_col: str = "behavioural_persona",
        segment_col: str = "signal_segment",
    ) -> dict:
        """
        Runs validate() per SIGNAL_SEGMENT. Returns results and overall summary.

        Returns:
            dict: {segment: validation_result, ..., "summary": {...}}
        """
        results  = {}
        segments = df[segment_col].unique() if segment_col in df.columns else ["ALL"]

        for seg in segments:
            seg_df = df[df[segment_col] == seg] if segment_col in df.columns else df
            try:
                results[seg] = self.validate(seg_df, cluster_col=cluster_col, segment=seg)
            except Exception as e:
                results[seg] = {"passed": False, "recommendation": "ERROR", "error": str(e)}

        passed_segments = [s for s, r in results.items() if r.get("passed")]
        failed_segments = [s for s, r in results.items() if not r.get("passed")]

        results["summary"] = {
            "total_segments":   len(segments),
            "passed":           len(passed_segments),
            "failed":           len(failed_segments),
            "passed_segments":  passed_segments,
            "failed_segments":  failed_segments,
            "overall_pass":     len(failed_segments) == 0,
            "action": (
                "All cluster splits are business-valid. Proceed with persona assignment."
                if len(failed_segments) == 0
                else f"Cluster splits failed for segments: {failed_segments}. "
                     f"Do not use cluster-based personas for these segments. "
                     f"Fall back to rule-based personas or reduce n_clusters and refit."
            ),
        }

        logger.info(
            f"Business validation summary: "
            f"{len(passed_segments)}/{len(segments)} segments passed | "
            f"failed={failed_segments}"
        )
        return results
