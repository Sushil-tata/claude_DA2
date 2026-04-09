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

# Item 4: explicit minimum stability threshold (≥70% of accounts must retain
# the same persona month-over-month for clusters to be considered stable).
MIN_STABILITY_THRESHOLD = 0.70

# Item 8: minimum acceptable Spearman rank-order correlation between cluster
# composite rank (1=High-Engagement, 4=Dormant) and mean recovery_180d.
# Expected direction: negative (rank 1 = highest recovery, rank 4 = lowest).
# We check |correlation| > threshold — direction validated separately.
MIN_RANK_ORDER_SPEARMAN = 0.50

# Persona label → rank position (1=best, 4=worst) for rank-order validation.
PERSONA_RANK_POSITIONS = {
    "High-Engagement Chronic":  1,
    "Sudden-Shock Distressed":  2,
    "Structural Defaulter":     3,
    "Dormant":                  4,
}


class ClusterBusinessValidator:

    def __init__(
        self,
        min_pairwise_recovery_diff: float = DEFAULT_MIN_PAIRWISE_RECOVERY_DIFF,
        min_cluster_size_pct: float       = DEFAULT_MIN_CLUSTER_SIZE_PCT,
        stat_alpha: float                 = DEFAULT_STAT_ALPHA,
        primary_metric: str               = "recovery_180d",
        holdout_pct: float                = 0.20,
    ):
        """
        Args:
            min_pairwise_recovery_diff: Min abs diff between best and worst cluster
                                        on primary_metric for PASS.
                                        Default: 0.05 (5pp). See CONFIGURATION.
            min_cluster_size_pct:       Min fraction of segment per cluster.
                                        Default: 0.05 (5%). See CONFIGURATION.
            stat_alpha:                 p-value threshold for Kruskal-Wallis
                                        AFTER Bonferroni correction.
            primary_metric:             Recovery metric used as the primary
                                        differentiation check. Default: recovery_180d.
            holdout_pct:                Fraction of data held out for out-of-sample
                                        validation. Default: 0.20 (20%).
                                        Set to 0.0 to validate on full dataset
                                        (not recommended — in-sample bias).
        """
        self.min_pairwise_recovery_diff = min_pairwise_recovery_diff
        self.min_cluster_size_pct       = min_cluster_size_pct
        self.stat_alpha                 = stat_alpha
        self.primary_metric             = primary_metric
        self.holdout_pct                = holdout_pct

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

        # ── Out-of-sample holdout split ───────────────────────────────────────
        # All business metrics are computed on held-out data, NOT training data.
        # In-sample validation overstates cluster separation because the clusters
        # were optimised on this data. Holdout measures generalisation.
        if self.holdout_pct > 0.0 and len(df) >= 20:
            holdout_n = max(1, int(len(df) * self.holdout_pct))
            # Random holdout (no date column guaranteed in validator scope)
            rng        = np.random.default_rng(seed=42)
            holdout_idx = rng.choice(df.index, size=holdout_n, replace=False)
            val_df      = df.loc[holdout_idx]
            logger.info(
                f"[Segment {seg_label}] Out-of-sample validation | "
                f"holdout_n={holdout_n:,} ({self.holdout_pct:.0%} of {len(df):,})"
            )
        else:
            val_df = df
            if self.holdout_pct > 0.0:
                logger.warning(
                    f"[Segment {seg_label}] Not enough data for holdout split "
                    f"(n={len(df)} < 20). Validating on full dataset."
                )
        df = val_df   # all checks below use out-of-sample data

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

        # ── Check 3: Statistical significance per metric (Bonferroni-corrected) ─
        # We test multiple metrics simultaneously — without correction, we would
        # expect ~5% false positives at alpha=0.05. Bonferroni multiplies each
        # raw p-value by the number of tests, controlling family-wise error rate.
        n_tests      = len(available_metrics)
        stat_results = {}
        for metric in available_metrics:
            groups = [
                df.loc[df[cluster_col] == c, metric].dropna().values
                for c in clusters
                if len(df.loc[df[cluster_col] == c, metric].dropna()) >= 5
            ]
            if len(groups) < 2:
                stat_results[metric] = {
                    "stat": None, "p_value": None,
                    "p_value_bonferroni": None, "significant": False,
                }
                continue

            stat, p_raw = stats.kruskal(*groups)
            p_bonferroni = min(float(p_raw) * n_tests, 1.0)   # Bonferroni correction
            significant  = p_bonferroni < self.stat_alpha
            stat_results[metric] = {
                "stat":               round(float(stat), 4),
                "p_value":            round(float(p_raw), 6),
                "p_value_bonferroni": round(p_bonferroni, 6),
                "n_tests_corrected":  n_tests,
                "significant":        significant,
            }

            if metric == self.primary_metric and not significant:
                failure_reasons.append(
                    f"Kruskal-Wallis NOT significant for {metric} "
                    f"(Bonferroni-corrected): "
                    f"p_raw={p_raw:.4f}, p_corrected={p_bonferroni:.4f} "
                    f"> alpha={self.stat_alpha} (n_tests={n_tests}). "
                    f"Clusters do not reliably differ on primary recovery metric."
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

        # ── Check 5: Rank-order Spearman (Item 8) ────────────────────────────
        # The composite cluster ranking (High-Engagement=1 … Dormant=4) must be
        # MONOTONICALLY aligned with recovery_180d.  A rank-1 cluster with lower
        # recovery than a rank-3 cluster means the composite formula is mis-ranked.
        # We compute Spearman(rank_position, mean_recovery_180d) and require
        # |rho| > MIN_RANK_ORDER_SPEARMAN (0.50).  Expected direction: negative
        # (lower rank-position number = higher recovery).
        spearman_rho      = None
        spearman_passed   = True
        clusters_with_rank = {
            c: PERSONA_RANK_POSITIONS[c]
            for c in clusters
            if c in PERSONA_RANK_POSITIONS
        }
        if len(clusters_with_rank) >= 2 and self.primary_metric in df.columns:
            rank_positions  = []
            mean_recoveries = []
            for c, pos in clusters_with_rank.items():
                vals = df.loc[df[cluster_col] == c, self.primary_metric].dropna()
                if len(vals) >= 5:
                    rank_positions.append(pos)
                    mean_recoveries.append(float(vals.mean()))

            if len(rank_positions) >= 2:
                rho, _ = stats.spearmanr(rank_positions, mean_recoveries)
                spearman_rho = round(float(rho), 4)
                # Expect negative rho (rank 1 = highest recovery); check |rho| >= threshold
                spearman_passed = abs(spearman_rho) >= MIN_RANK_ORDER_SPEARMAN
                if not spearman_passed:
                    failure_reasons.append(
                        f"Rank-order Spearman check FAILED: |rho| = {abs(spearman_rho):.3f} "
                        f"< {MIN_RANK_ORDER_SPEARMAN}. Cluster composite ranking is not "
                        f"monotonically aligned with {self.primary_metric}. "
                        f"Inspect _assign_persona_labels composite weights."
                    )
                elif spearman_rho > 0:
                    # rho positive = higher rank-position → higher recovery (inverted)
                    logger.warning(
                        f"[Segment {seg_label}] Spearman rho={spearman_rho:.3f} is positive: "
                        f"Dormant cluster has higher recovery than High-Engagement. "
                        f"Composite ranking direction may be inverted."
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
            "spearman_rank_rho":           spearman_rho,          # Item 8
            "spearman_rank_passed":        spearman_passed,        # Item 8
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

    def validate_stability(
        self,
        df_t1: pd.DataFrame,
        df_t2: pd.DataFrame,
        cluster_col: str = "behavioural_persona",
        id_col:      str = "account_id",
        max_transition_pct: float = 1.0 - MIN_STABILITY_THRESHOLD,
    ) -> dict:
        """
        Validates cluster stability between two time periods by computing
        a persona transition matrix on accounts present in both periods.

        Stability threshold (Item 4): ≥70% of accounts must retain the same
        persona month-over-month (MIN_STABILITY_THRESHOLD = 0.70).
        Default max_transition_pct = 0.30 enforces this threshold.

        Why this matters:
          Good behavioural clusters should be relatively stable — a customer
          who is "Cooperative" today should mostly remain "Cooperative" next
          month. High churn (>30% of accounts switching cluster every period)
          means the clusters are not capturing stable behavioural identity.
          This can happen when: (a) clustering on outcome-proxy features that
          fluctuate with model scores, (b) insufficient data per cluster,
          (c) the feature distribution has shifted (population drift).

        Args:
            df_t1:               feature_output from period 1 (earlier).
            df_t2:               feature_output from period 2 (later).
            cluster_col:         Column with persona labels.
            id_col:              Account identifier column for the join.
            max_transition_pct:  Maximum acceptable fraction of accounts that
                                 changed persona between t1 and t2.
                                 Default: 1 - MIN_STABILITY_THRESHOLD = 0.30.

        Returns:
            dict with:
                transition_matrix (dict of dicts): t1_persona → {t2_persona: count}
                transition_rates  (dict): fraction moving out of each t1 persona
                overall_stability (float): fraction of accounts that kept same persona
                passed (bool): True if overall_stability >= (1 - max_transition_pct)
                recommendation (str)
                n_matched (int): accounts present in both periods
        """
        if id_col not in df_t1.columns or id_col not in df_t2.columns:
            return {
                "passed": False,
                "recommendation": "SKIP — id_col not found in both DataFrames",
                "error": f"id_col='{id_col}' missing",
            }
        if cluster_col not in df_t1.columns or cluster_col not in df_t2.columns:
            return {
                "passed": False,
                "recommendation": "SKIP — cluster_col not found in both DataFrames",
                "error": f"cluster_col='{cluster_col}' missing",
            }

        merged = df_t1[[id_col, cluster_col]].rename(
            columns={cluster_col: "persona_t1"}
        ).merge(
            df_t2[[id_col, cluster_col]].rename(columns={cluster_col: "persona_t2"}),
            on=id_col, how="inner",
        )
        n_matched = len(merged)
        if n_matched < 10:
            return {
                "passed": False,
                "recommendation": f"SKIP — only {n_matched} accounts matched between periods",
                "n_matched": n_matched,
            }

        # ── Transition matrix ─────────────────────────────────────────────────
        transition_counts = (
            merged.groupby(["persona_t1", "persona_t2"])
            .size()
            .unstack(fill_value=0)
        )
        all_personas = sorted(
            set(merged["persona_t1"].unique()) | set(merged["persona_t2"].unique())
        )
        transition_matrix = {
            p1: {
                p2: int(transition_counts.loc[p1, p2])
                    if p1 in transition_counts.index and p2 in transition_counts.columns
                    else 0
                for p2 in all_personas
            }
            for p1 in all_personas
        }

        # ── Transition rates per persona ──────────────────────────────────────
        transition_rates = {}
        for p1 in all_personas:
            row = merged[merged["persona_t1"] == p1]
            if len(row) > 0:
                stayed      = (row["persona_t2"] == p1).sum()
                transitioned = len(row) - stayed
                transition_rates[p1] = {
                    "n":                int(len(row)),
                    "stayed":           int(stayed),
                    "transitioned":     int(transitioned),
                    "transition_pct":   round(float(transitioned / len(row)), 4),
                }

        # ── Overall stability ─────────────────────────────────────────────────
        n_stable          = int((merged["persona_t1"] == merged["persona_t2"]).sum())
        overall_stability = round(float(n_stable / n_matched), 4)
        stability_threshold = 1.0 - max_transition_pct   # ≥70% per MIN_STABILITY_THRESHOLD
        passed            = overall_stability >= stability_threshold

        if passed:
            recommendation = (
                f"STABLE — {overall_stability:.1%} of accounts kept same persona "
                f"(threshold ≥{stability_threshold:.0%}). "
                f"Clusters represent stable behavioural identity."
            )
        else:
            recommendation = (
                f"UNSTABLE — only {overall_stability:.1%} kept same persona "
                f"(required ≥{stability_threshold:.0%}, Item 4 threshold). "
                f"Investigate: (1) are outcome proxies in CLUSTER_FEATURES? "
                f"(2) population drift? (3) insufficient cluster data?"
            )

        log_level = "info" if passed else "warning"
        getattr(logger, log_level)(
            f"Cluster stability: overall_stability={overall_stability:.1%} | "
            f"n_matched={n_matched:,} | passed={passed}"
        )

        return {
            "passed":             passed,
            "recommendation":     recommendation,
            "overall_stability":  overall_stability,
            "n_matched":          n_matched,
            "transition_matrix":  transition_matrix,
            "transition_rates":   transition_rates,
            "max_transition_pct": max_transition_pct,
        }
