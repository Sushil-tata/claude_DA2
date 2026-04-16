"""
Persona Validator
=================
Evaluates segmentation quality against actual recovery outcomes.

Validation criteria:
  1. Separation: recovery rate difference between best and worst personas ≥ 5pp
  2. Significance: Mann-Whitney U test (p < 0.05) between best and worst persona
  3. Distribution: no single persona accounts for > 50% of the population
  4. Signal segment distribution: no single segment > 60% (softer check)

Gating:
  - PASS:    all mandatory checks pass → segmentation is valid for production scoring
  - WARNING: distribution is skewed OR significance weak → flag but allow with monitoring
  - FAIL:    separation < 5pp → segmentation rejected; re-calibrate before scoring

Usage:
    validator = PersonaValidator(payment_threshold=500.0)
    report = validator.validate(
        persona_df=persona_df,        # output of PersonaBuilder.assign_batch()
        outcomes_df=outcomes_df,      # {account_id, observation_date, recovery_6m, recovery_12m}
        observation_date="2024-01-31"
    )
    if not report.gate_passed:
        raise RuntimeError(f"Segmentation rejected: {report.gate_reason}")

outcomes_df contract:
  - account_id        : str — matches persona_df.account_id
  - observation_date  : str (YYYY-MM-DD) — must match PersonaBuilder observation_date
  - recovery_6m       : float — total THB recovered in 6 months AFTER observation_date
  - recovery_12m      : float — total THB recovered in 12 months AFTER observation_date

Point-in-time safety:
  - recovery_6m and recovery_12m are FORWARD-LOOKING windows from observation_date
  - They must be computed from data AFTER observation_date (labels, not features)
  - The validator checks that observation_date is consistent between persona_df
    and outcomes_df — no date mismatch allowed
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

VALID_PERSONAS = {
    "ACTIVE_PAYER", "SELECTIVE_DEFAULTER", "LIQUIDITY_CONSTRAINED",
    "STRATEGIC", "DORMANT",
}

# Mandatory gate thresholds
MIN_SEPARATION_PP        = 0.05   # ≥5pp recovery rate difference (best vs worst persona)
MAX_PERSONA_DOMINANCE    = 0.50   # no single persona > 50% of accounts
SIGNIFICANCE_ALPHA       = 0.05   # p-value threshold for Mann-Whitney U
MIN_ACCOUNTS_PER_PERSONA = 30     # minimum accounts to compute a reliable rate

# Warning-only thresholds
MIN_PERSONA_POPULATION   = 0.05   # < 5% population per persona → WARNING (too thin to rely on)
MAX_SEGMENT_DOMINANCE    = 0.60   # signal_segment distribution soft cap
OOT_INSTABILITY_MAX_DRIFT = 0.05  # separation drift > 5pp across OOT dates → instability warning


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT DATACLASS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PersonaSeparationResult:
    """Per-persona recovery statistics."""
    persona: str
    n_accounts: int
    recovery_rate_6m: float    # proportion with recovery_6m > payment_threshold
    recovery_rate_12m: float
    mean_recovery_6m: float    # mean THB recovered
    mean_recovery_12m: float
    sufficient_n: bool         # True if n_accounts >= MIN_ACCOUNTS_PER_PERSONA


@dataclass
class SignificanceResult:
    """Mann-Whitney U test result between best and worst personas."""
    best_persona: str
    worst_persona: str
    statistic: float
    p_value: float
    is_significant: bool      # p_value < SIGNIFICANCE_ALPHA
    n_best: int
    n_worst: int
    test_name: str = "Mann-Whitney U"


@dataclass
class PairwiseSeparation:
    """Separation result between two personas."""
    persona_a: str
    persona_b: str
    rate_a_6m: float
    rate_b_6m: float
    separation_6m_pp: float    # rate_a - rate_b (signed; positive means A > B)
    separation_12m_pp: float
    p_value: Optional[float]   # Mann-Whitney p-value (one-sided: A > B)
    is_significant: bool


@dataclass
class ValidationReport:
    """Full validation report from PersonaValidator.validate()."""
    observation_date: str
    n_accounts_matched: int
    n_accounts_unmatched: int  # persona records with no outcome label

    # Per-persona stats
    persona_stats: List[PersonaSeparationResult] = field(default_factory=list)

    # Best-vs-worst separation (mandatory gate)
    best_persona: str = ""
    worst_persona: str = ""
    separation_6m_pp: float = 0.0
    separation_12m_pp: float = 0.0
    separation_passes: bool = False

    # All pairwise separations (informational — shows full separation structure)
    pairwise_separations: List[PairwiseSeparation] = field(default_factory=list)

    # Significance (best vs worst pair)
    significance: Optional[SignificanceResult] = None

    # Distribution
    persona_distribution: Dict[str, float] = field(default_factory=dict)
    dominant_persona: Optional[str] = None
    dominance_fraction: float = 0.0
    distribution_passes: bool = False     # no persona > MAX_PERSONA_DOMINANCE

    # Low-population warning (< 5% per persona)
    low_population_personas: List[str] = field(default_factory=list)

    # Signal segment distribution
    segment_distribution: Dict[str, float] = field(default_factory=dict)
    segment_distribution_passes: bool = True

    # Gate
    gate_status: str = "FAIL"      # PASS | WARNING | FAIL
    gate_reason: str = ""
    gate_passed: bool = False

    def to_text(self) -> str:
        """Human-readable validation report."""
        lines = [
            "=" * 70,
            "PERSONA VALIDATION REPORT",
            "=" * 70,
            f"Observation date : {self.observation_date}",
            f"Matched accounts : {self.n_accounts_matched}",
            f"Unmatched        : {self.n_accounts_unmatched}",
            "",
            "── PERSONA RECOVERY RATES ──────────────────────────────────────────",
        ]
        for ps in sorted(self.persona_stats, key=lambda x: -x.recovery_rate_6m):
            n_flag = "" if ps.sufficient_n else " [LOW N]"
            pop_flag = (
                " [< 5% POP]"
                if ps.persona in self.low_population_personas else ""
            )
            lines.append(
                f"  {ps.persona:25s}  n={ps.n_accounts:5d}  "
                f"6m={ps.recovery_rate_6m:5.1%}  12m={ps.recovery_rate_12m:5.1%}"
                f"  mean_6m={ps.mean_recovery_6m:,.0f} THB"
                f"{n_flag}{pop_flag}"
            )

        lines += [
            "",
            "── PAIRWISE SEPARATION (6M recovery rate, sorted by gap) ───────────",
        ]
        for pw in sorted(self.pairwise_separations, key=lambda x: -abs(x.separation_6m_pp)):
            sig_flag = "*" if pw.is_significant else " "
            p_str = f"p={pw.p_value:.3f}" if pw.p_value is not None else "p=N/A"
            lines.append(
                f"  {sig_flag} {pw.persona_a:25s} vs {pw.persona_b:25s} "
                f"gap={pw.separation_6m_pp:+.1%}  {p_str}"
            )
        lines.append("    (* = Mann-Whitney p < 0.05 for that pair)")

        lines += [
            "",
            "── MAX SEPARATION (mandatory gate) ─────────────────────────────────",
            f"  Best persona   : {self.best_persona}  ({self._get_rate(self.best_persona):.1%})",
            f"  Worst persona  : {self.worst_persona}  ({self._get_rate(self.worst_persona):.1%})",
            f"  Gap (6m)       : {self.separation_6m_pp:.1%}  (threshold ≥ {MIN_SEPARATION_PP:.0%})",
            f"  Gap (12m)      : {self.separation_12m_pp:.1%}",
            f"  PASS           : {self.separation_passes}",
            "",
            "── SIGNIFICANCE (best vs worst) ────────────────────────────────────",
        ]
        if self.significance:
            s = self.significance
            lines += [
                f"  Test    : {s.test_name} (one-sided: best > worst)",
                f"  Pair    : {s.best_persona} vs {s.worst_persona}",
                f"  p-value : {s.p_value:.4f}  (threshold < {SIGNIFICANCE_ALPHA})",
                f"  PASS    : {s.is_significant}",
            ]
        else:
            lines.append("  Not computed (insufficient data per persona)")

        lines += [
            "",
            "── DISTRIBUTION ────────────────────────────────────────────────────",
        ]
        for persona, frac in sorted(self.persona_distribution.items(), key=lambda x: -x[1]):
            dom_flag = " ← DOMINANCE FAIL" if frac > MAX_PERSONA_DOMINANCE else ""
            thin_flag = " ← THIN (<5%)" if frac < MIN_PERSONA_POPULATION else ""
            lines.append(
                f"  {persona:25s}: {frac:5.1%}{dom_flag}{thin_flag}"
            )
        lines.append(f"  Distribution PASS : {self.distribution_passes}")
        if self.low_population_personas:
            lines.append(
                f"  Low-pop WARNING   : {self.low_population_personas} "
                f"(< {MIN_PERSONA_POPULATION:.0%} each)"
            )

        lines += [
            "",
            "── SIGNAL SEGMENT DISTRIBUTION ─────────────────────────────────────",
        ]
        for seg, frac in sorted(self.segment_distribution.items(), key=lambda x: -x[1]):
            skew_flag = " ← SKEWED" if frac > MAX_SEGMENT_DOMINANCE else ""
            lines.append(f"  {seg:25s}: {frac:5.1%}{skew_flag}")

        lines += [
            "",
            "── GATE ────────────────────────────────────────────────────────────",
            f"  Status : {self.gate_status}",
            f"  Reason : {self.gate_reason}",
            "=" * 70,
        ]
        return "\n".join(lines)

    def _get_rate(self, persona: str) -> float:
        for ps in self.persona_stats:
            if ps.persona == persona:
                return ps.recovery_rate_6m
        return 0.0

    def _get_worst_rate(self) -> float:
        return self._get_rate(self.worst_persona)


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATOR
# ─────────────────────────────────────────────────────────────────────────────

class PersonaValidator:
    """
    Validates persona segmentation quality using actual recovery outcomes.

    Mandatory checks (gate fails if any fail):
      - Separation ≥ 5pp between best and worst persona recovery rates
      - Statistical significance (Mann-Whitney p < 0.05)
      - No single persona > 50% of population

    Warning checks (gate = WARNING, not FAIL):
      - Signal segment dominance > 60%
      - Low account counts per persona (< 30)
    """

    def __init__(self, payment_threshold: float = 500.0):
        """
        Args:
            payment_threshold: Minimum THB to count as recovery event (binary label)
        """
        self.payment_threshold = payment_threshold

    def validate(
        self,
        persona_df: pd.DataFrame,
        outcomes_df: pd.DataFrame,
        observation_date: str,
    ) -> ValidationReport:
        """
        Validate segmentation quality against actual recovery outcomes.

        Args:
            persona_df       : Output of PersonaBuilder.assign_batch().
                               Must have columns: account_id, observation_date,
                               persona, signal_segment.
            outcomes_df      : Outcome labels (FORWARD-LOOKING from observation_date).
                               Must have columns: account_id, observation_date,
                               recovery_6m, recovery_12m.
            observation_date : Snapshot date (YYYY-MM-DD). Used to filter both
                               DataFrames — ensures PIT consistency.

        Returns:
            ValidationReport with gate status and all metrics.
        """
        # ── PIT check: filter to observation_date ─────────────────────────────
        if "observation_date" in persona_df.columns:
            persona_slice = persona_df[
                persona_df["observation_date"] == observation_date
            ].copy()
        else:
            persona_slice = persona_df.copy()
            logger.warning(
                "persona_df has no observation_date column — "
                "cannot enforce PIT consistency."
            )

        if "observation_date" in outcomes_df.columns:
            outcomes_slice = outcomes_df[
                outcomes_df["observation_date"] == observation_date
            ].copy()
        else:
            outcomes_slice = outcomes_df.copy()

        # ── Merge persona with outcomes ───────────────────────────────────────
        merged = persona_slice.merge(
            outcomes_slice[["account_id", "recovery_6m", "recovery_12m"]],
            on="account_id",
            how="left",
        )

        n_matched = merged["recovery_6m"].notna().sum()
        n_unmatched = merged["recovery_6m"].isna().sum()

        report = ValidationReport(
            observation_date=observation_date,
            n_accounts_matched=int(n_matched),
            n_accounts_unmatched=int(n_unmatched),
        )

        if n_matched < 50:
            report.gate_status = "FAIL"
            report.gate_reason = (
                f"Insufficient matched accounts ({n_matched}) for validation. "
                "Need ≥50 accounts with outcome labels."
            )
            logger.error(report.gate_reason)
            return report

        # Only compute on matched rows
        merged_matched = merged[merged["recovery_6m"].notna()].copy()
        merged_matched["paid_6m"]  = (merged_matched["recovery_6m"]  > self.payment_threshold).astype(float)
        merged_matched["paid_12m"] = (merged_matched["recovery_12m"] > self.payment_threshold).astype(float)

        # ── 1. Per-persona statistics ─────────────────────────────────────────
        persona_stats = []
        persona_amounts: Dict[str, np.ndarray] = {}  # for significance test

        for persona in VALID_PERSONAS:
            subset = merged_matched[merged_matched["persona"] == persona]
            n = len(subset)
            if n == 0:
                continue

            rate_6m  = float(subset["paid_6m"].mean())
            rate_12m = float(subset["paid_12m"].mean())
            mean_6m  = float(subset["recovery_6m"].mean())
            mean_12m = float(subset["recovery_12m"].mean())

            persona_stats.append(PersonaSeparationResult(
                persona=persona,
                n_accounts=n,
                recovery_rate_6m=rate_6m,
                recovery_rate_12m=rate_12m,
                mean_recovery_6m=mean_6m,
                mean_recovery_12m=mean_12m,
                sufficient_n=(n >= MIN_ACCOUNTS_PER_PERSONA),
            ))
            persona_amounts[persona] = subset["recovery_6m"].values

        report.persona_stats = persona_stats

        if len(persona_stats) < 2:
            report.gate_status = "FAIL"
            report.gate_reason = (
                f"Only {len(persona_stats)} personas present in matched data — "
                "cannot compute separation."
            )
            return report

        # ── 2. Best-vs-worst separation (mandatory gate metric) ───────────────
        rates_6m  = {ps.persona: ps.recovery_rate_6m  for ps in persona_stats}
        rates_12m = {ps.persona: ps.recovery_rate_12m for ps in persona_stats}

        best_persona  = max(rates_6m, key=rates_6m.get)
        worst_persona = min(rates_6m, key=rates_6m.get)

        sep_6m  = rates_6m[best_persona]  - rates_6m[worst_persona]
        sep_12m = rates_12m[best_persona] - rates_12m[worst_persona]

        report.best_persona      = best_persona
        report.worst_persona     = worst_persona
        report.separation_6m_pp  = sep_6m
        report.separation_12m_pp = sep_12m
        report.separation_passes = sep_6m >= MIN_SEPARATION_PP

        # ── 3. Pairwise separation (all persona combinations) ─────────────────
        personas_present = [ps.persona for ps in persona_stats]
        pairwise = []
        for i, pa in enumerate(personas_present):
            for pb in personas_present[i + 1:]:
                sep_ab_6m  = rates_6m[pa]  - rates_6m[pb]
                sep_ab_12m = rates_12m[pa] - rates_12m[pb]

                # Mann-Whitney (one-sided: pa > pb)
                arr_a = persona_amounts.get(pa, np.array([]))
                arr_b = persona_amounts.get(pb, np.array([]))
                pw_p_val: Optional[float] = None
                pw_sig = False
                if len(arr_a) >= 10 and len(arr_b) >= 10:
                    # Test whichever direction is larger
                    a_is_higher = sep_ab_6m >= 0
                    hi, lo = (arr_a, arr_b) if a_is_higher else (arr_b, arr_a)
                    _, pw_p_val_raw = scipy_stats.mannwhitneyu(
                        hi, lo, alternative="greater"
                    )
                    pw_p_val = float(pw_p_val_raw)
                    pw_sig   = pw_p_val < SIGNIFICANCE_ALPHA

                pairwise.append(PairwiseSeparation(
                    persona_a=pa,
                    persona_b=pb,
                    rate_a_6m=rates_6m[pa],
                    rate_b_6m=rates_6m[pb],
                    separation_6m_pp=sep_ab_6m,
                    separation_12m_pp=sep_ab_12m,
                    p_value=pw_p_val,
                    is_significant=pw_sig,
                ))
        report.pairwise_separations = pairwise

        # ── 4. Significance (best vs worst — primary gate test) ───────────────
        best_amounts  = persona_amounts.get(best_persona,  np.array([]))
        worst_amounts = persona_amounts.get(worst_persona, np.array([]))

        if len(best_amounts) >= 10 and len(worst_amounts) >= 10:
            stat, p_val = scipy_stats.mannwhitneyu(
                best_amounts, worst_amounts, alternative="greater"
            )
            report.significance = SignificanceResult(
                best_persona=best_persona,
                worst_persona=worst_persona,
                statistic=float(stat),
                p_value=float(p_val),
                is_significant=bool(p_val < SIGNIFICANCE_ALPHA),
                n_best=len(best_amounts),
                n_worst=len(worst_amounts),
            )
        else:
            logger.warning(
                f"Insufficient data for significance test: "
                f"n_best={len(best_amounts)}, n_worst={len(worst_amounts)}"
            )

        # ── 5. Distribution check ─────────────────────────────────────────────
        dist = persona_slice["persona"].value_counts(normalize=True).to_dict()
        report.persona_distribution = {k: round(v, 4) for k, v in dist.items()}

        dominant = max(dist, key=dist.get) if dist else None
        dominant_frac = dist.get(dominant, 0.0) if dominant else 0.0
        report.dominant_persona    = dominant
        report.dominance_fraction  = dominant_frac
        report.distribution_passes = dominant_frac <= MAX_PERSONA_DOMINANCE

        # Low-population personas (< 5%): warning, not mandatory failure
        report.low_population_personas = [
            p for p, frac in dist.items() if frac < MIN_PERSONA_POPULATION
        ]

        # ── 6. Segment distribution (warning only) ────────────────────────────
        if "signal_segment" in persona_slice.columns:
            seg_dist = persona_slice["signal_segment"].value_counts(normalize=True).to_dict()
            report.segment_distribution = {k: round(v, 4) for k, v in seg_dist.items()}
            max_seg_frac = max(seg_dist.values()) if seg_dist else 0.0
            report.segment_distribution_passes = max_seg_frac <= MAX_SEGMENT_DOMINANCE
        else:
            report.segment_distribution_passes = True

        # ── 7. Gate decision ──────────────────────────────────────────────────
        fail_reasons: List[str] = []

        if not report.separation_passes:
            fail_reasons.append(
                f"max separation {sep_6m:.1%} < {MIN_SEPARATION_PP:.0%} required"
            )

        sig_passes = (
            report.significance is not None and report.significance.is_significant
        )
        if report.significance is not None and not sig_passes:
            fail_reasons.append(
                f"Mann-Whitney p={report.significance.p_value:.4f} ≥ {SIGNIFICANCE_ALPHA}"
            )

        if not report.distribution_passes:
            fail_reasons.append(
                f"persona '{dominant}' is {dominant_frac:.1%} of accounts "
                f"(max {MAX_PERSONA_DOMINANCE:.0%})"
            )

        if fail_reasons:
            report.gate_status = "FAIL"
            report.gate_reason = "; ".join(fail_reasons)
            report.gate_passed = False
        else:
            warn_reasons: List[str] = []
            low_n_ps = [ps.persona for ps in persona_stats if not ps.sufficient_n]
            if low_n_ps:
                warn_reasons.append(f"low sample count (<{MIN_ACCOUNTS_PER_PERSONA}): {low_n_ps}")
            if report.low_population_personas:
                warn_reasons.append(
                    f"thin population (<{MIN_PERSONA_POPULATION:.0%}): "
                    f"{report.low_population_personas}"
                )
            if not report.segment_distribution_passes:
                warn_reasons.append("signal_segment distribution skewed")

            if warn_reasons:
                report.gate_status = "WARNING"
                report.gate_reason = "; ".join(warn_reasons)
                report.gate_passed = True
            else:
                report.gate_status = "PASS"
                report.gate_reason = "all checks passed"
                report.gate_passed = True

        logger.info(
            f"PersonaValidator [{observation_date}]: gate={report.gate_status} | "
            f"sep_6m={sep_6m:.1%} | "
            f"p={report.significance.p_value:.4f if report.significance else 'N/A'} | "
            f"dominant={dominant} ({dominant_frac:.1%}) | "
            f"low_pop={report.low_population_personas}"
        )
        return report

    def run_oot_validation(
        self,
        persona_df: pd.DataFrame,
        outcomes_df: pd.DataFrame,
        observation_dates: List[str],
    ) -> Dict[str, object]:
        """
        Run validation across multiple observation dates and detect instability.

        Instability indicators (WARNING):
          - Gate status is not consistent across all dates (e.g. PASS then FAIL)
          - Separation_6m_pp drifts > OOT_INSTABILITY_MAX_DRIFT between any two dates

        Args:
            persona_df        : Combined persona_df for all dates (with observation_date col)
            outcomes_df       : Combined outcomes_df for all dates
            observation_dates : List of YYYY-MM-DD strings to evaluate independently

        Returns:
            Dict with:
              "by_date"    : {observation_date: ValidationReport}
              "stability"  : stability assessment dict
        """
        results: Dict[str, ValidationReport] = {}
        for obs_date in observation_dates:
            try:
                report = self.validate(persona_df, outcomes_df, obs_date)
                results[obs_date] = report
            except Exception as e:
                logger.error(f"OOT validation failed for {obs_date}: {e}")
                results[obs_date] = ValidationReport(
                    observation_date=obs_date,
                    n_accounts_matched=0,
                    n_accounts_unmatched=0,
                    gate_status="FAIL",
                    gate_reason=str(e),
                    gate_passed=False,
                )

        # ── Stability analysis ────────────────────────────────────────────────
        gates        = {d: r.gate_status    for d, r in results.items()}
        separations  = {d: r.separation_6m_pp for d, r in results.items()
                        if r.n_accounts_matched > 0}

        gate_values    = list(gates.values())
        gate_consistent = len(set(gate_values)) == 1

        sep_values      = list(separations.values())
        sep_drift       = (max(sep_values) - min(sep_values)) if len(sep_values) >= 2 else 0.0
        sep_stable      = sep_drift <= OOT_INSTABILITY_MAX_DRIFT

        stability: Dict[str, object] = {
            "gates_by_date":   gates,
            "gate_consistent": gate_consistent,
            "separation_by_date": separations,
            "separation_drift_pp": round(sep_drift, 4),
            "separation_stable": sep_stable,
            "overall_stable":  gate_consistent and sep_stable,
            "instability_warnings": [],
        }

        if not gate_consistent:
            stability["instability_warnings"].append(
                f"Gate status changed across OOT dates: {gates}"
            )
        if not sep_stable:
            stability["instability_warnings"].append(
                f"Separation drifted {sep_drift:.1%} across dates "
                f"(threshold {OOT_INSTABILITY_MAX_DRIFT:.0%}): {separations}"
            )

        logger.info(
            f"OOT validation: {len(results)} dates | "
            f"gates={gates} | sep_drift={sep_drift:.1%} | stable={stability['overall_stable']}"
        )
        return {"by_date": results, "stability": stability}

    @staticmethod
    def oot_summary_table(oot_result: Dict[str, object]) -> str:
        """
        Print a concise OOT summary table from run_oot_validation() output.

        Args:
            oot_result: Return value of run_oot_validation()

        Returns:
            Formatted string table.
        """
        by_date: Dict[str, ValidationReport] = oot_result.get("by_date", {})
        stability = oot_result.get("stability", {})

        lines = [
            "OOT VALIDATION SUMMARY",
            f"{'Date':12s}  {'Gate':8s}  {'Sep_6m':8s}  {'Sep_12m':8s}  "
            f"{'N_matched':10s}  {'Dominant%':10s}  {'Low_pop':20s}",
            "-" * 90,
        ]
        for date in sorted(by_date):
            r = by_date[date]
            low_pop = ",".join(r.low_population_personas) if r.low_population_personas else "-"
            lines.append(
                f"{date:12s}  {r.gate_status:8s}  {r.separation_6m_pp:+6.1%}   "
                f"{r.separation_12m_pp:+6.1%}   {r.n_accounts_matched:10d}  "
                f"{r.dominance_fraction:9.1%}   {low_pop:20s}"
            )

        lines += [
            "-" * 90,
            f"Drift:  sep_6m={stability.get('separation_drift_pp', 0):.1%}  "
            f"stable={stability.get('overall_stable')}",
        ]
        warnings = stability.get("instability_warnings", [])
        for w in warnings:
            lines.append(f"  ⚠  {w}")
        return "\n".join(lines)
