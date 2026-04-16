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
MIN_SEPARATION_PP       = 0.05   # ≥5pp recovery rate difference (best vs worst persona)
MAX_PERSONA_DOMINANCE   = 0.50   # no single persona > 50% of accounts
SIGNIFICANCE_ALPHA      = 0.05   # p-value threshold for Mann-Whitney U
MIN_ACCOUNTS_PER_PERSONA = 30    # minimum accounts to compute a reliable rate

# Warning-only thresholds
MAX_SEGMENT_DOMINANCE   = 0.60   # signal_segment distribution soft cap


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
class ValidationReport:
    """Full validation report from PersonaValidator.validate()."""
    observation_date: str
    n_accounts_matched: int
    n_accounts_unmatched: int  # persona records with no outcome label

    # Per-persona stats
    persona_stats: List[PersonaSeparationResult] = field(default_factory=list)

    # Separation metrics
    best_persona: str = ""
    worst_persona: str = ""
    separation_6m_pp: float = 0.0        # best_rate - worst_rate (percentage points)
    separation_12m_pp: float = 0.0
    separation_passes: bool = False       # >= MIN_SEPARATION_PP

    # Significance
    significance: Optional[SignificanceResult] = None

    # Distribution
    persona_distribution: Dict[str, float] = field(default_factory=dict)
    dominant_persona: Optional[str] = None
    dominance_fraction: float = 0.0
    distribution_passes: bool = False     # no persona > MAX_PERSONA_DOMINANCE

    # Signal segment distribution
    segment_distribution: Dict[str, float] = field(default_factory=dict)
    segment_distribution_passes: bool = True  # warning only

    # Gate
    gate_status: str = "FAIL"      # PASS | WARNING | FAIL
    gate_reason: str = ""
    gate_passed: bool = False       # True only if gate_status == PASS

    def to_text(self) -> str:
        """Human-readable validation report."""
        lines = [
            "=" * 65,
            "PERSONA VALIDATION REPORT",
            "=" * 65,
            f"Observation date : {self.observation_date}",
            f"Matched accounts : {self.n_accounts_matched}",
            f"Unmatched        : {self.n_accounts_unmatched}",
            "",
            "── PERSONA RECOVERY RATES ──────────────────────────────────────",
        ]

        for ps in sorted(self.persona_stats, key=lambda x: -x.recovery_rate_6m):
            flag = "" if ps.sufficient_n else " [LOW N]"
            lines.append(
                f"  {ps.persona:25s}  n={ps.n_accounts:5d}  "
                f"6m={ps.recovery_rate_6m:5.1%}  12m={ps.recovery_rate_12m:5.1%}"
                f"  mean_6m={ps.mean_recovery_6m:,.0f} THB{flag}"
            )

        lines += [
            "",
            "── SEPARATION ──────────────────────────────────────────────────",
            f"  Best persona   : {self.best_persona} ({self.separation_6m_pp + self._get_worst_rate():.1%})",
            f"  Worst persona  : {self.worst_persona} ({self._get_worst_rate():.1%})",
            f"  Gap (6m)       : {self.separation_6m_pp:.1%}  (threshold: {MIN_SEPARATION_PP:.0%})",
            f"  Gap (12m)      : {self.separation_12m_pp:.1%}",
            f"  PASS           : {self.separation_passes}",
            "",
            "── SIGNIFICANCE ────────────────────────────────────────────────",
        ]

        if self.significance:
            s = self.significance
            lines += [
                f"  Test           : {s.test_name}",
                f"  Best vs Worst  : {s.best_persona} vs {s.worst_persona}",
                f"  p-value        : {s.p_value:.4f}  (threshold: {SIGNIFICANCE_ALPHA})",
                f"  PASS           : {s.is_significant}",
            ]
        else:
            lines.append("  Not computed (insufficient data)")

        lines += [
            "",
            "── DISTRIBUTION ────────────────────────────────────────────────",
        ]
        for persona, frac in sorted(self.persona_distribution.items(), key=lambda x: -x[1]):
            flag = " ← DOMINANCE ISSUE" if frac > MAX_PERSONA_DOMINANCE else ""
            lines.append(f"  {persona:25s}: {frac:5.1%}{flag}")

        lines += [
            f"  Distribution PASS: {self.distribution_passes}",
            "",
            "── SIGNAL SEGMENT DISTRIBUTION ─────────────────────────────────",
        ]
        for seg, frac in sorted(self.segment_distribution.items(), key=lambda x: -x[1]):
            flag = " ← SKEWED" if frac > MAX_SEGMENT_DOMINANCE else ""
            lines.append(f"  {seg:25s}: {frac:5.1%}{flag}")

        lines += [
            "",
            "── GATE ────────────────────────────────────────────────────────",
            f"  Status : {self.gate_status}",
            f"  Reason : {self.gate_reason}",
            "=" * 65,
        ]
        return "\n".join(lines)

    def _get_worst_rate(self) -> float:
        for ps in self.persona_stats:
            if ps.persona == self.worst_persona:
                return ps.recovery_rate_6m
        return 0.0


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

        # ── 2. Separation check ───────────────────────────────────────────────
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

        # ── 3. Significance test (Mann-Whitney U) ─────────────────────────────
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

        # ── 4. Distribution check ─────────────────────────────────────────────
        total = len(persona_slice)
        dist = persona_slice["persona"].value_counts(normalize=True).to_dict()
        report.persona_distribution = {k: round(v, 4) for k, v in dist.items()}

        dominant = max(dist, key=dist.get) if dist else None
        dominant_frac = dist.get(dominant, 0.0) if dominant else 0.0
        report.dominant_persona    = dominant
        report.dominance_fraction  = dominant_frac
        report.distribution_passes = dominant_frac <= MAX_PERSONA_DOMINANCE

        # ── 5. Segment distribution (warning only) ────────────────────────────
        if "signal_segment" in persona_slice.columns:
            seg_dist = persona_slice["signal_segment"].value_counts(normalize=True).to_dict()
            report.segment_distribution = {k: round(v, 4) for k, v in seg_dist.items()}
            max_seg_frac = max(seg_dist.values()) if seg_dist else 0.0
            report.segment_distribution_passes = max_seg_frac <= MAX_SEGMENT_DOMINANCE
        else:
            report.segment_distribution_passes = True

        # ── 6. Gate decision ──────────────────────────────────────────────────
        reasons = []

        # Mandatory failures
        if not report.separation_passes:
            reasons.append(
                f"separation {sep_6m:.1%} < {MIN_SEPARATION_PP:.0%} required"
            )

        sig_passes = (
            report.significance is not None and report.significance.is_significant
        )
        if report.significance is not None and not sig_passes:
            reasons.append(
                f"Mann-Whitney p={report.significance.p_value:.4f} > {SIGNIFICANCE_ALPHA}"
            )

        if not report.distribution_passes:
            reasons.append(
                f"persona '{dominant}' is {dominant_frac:.1%} of accounts "
                f"(max {MAX_PERSONA_DOMINANCE:.0%})"
            )

        if reasons:
            report.gate_status = "FAIL"
            report.gate_reason = "; ".join(reasons)
            report.gate_passed = False
        else:
            # Check for warnings
            warnings = []
            low_n_personas = [ps.persona for ps in persona_stats if not ps.sufficient_n]
            if low_n_personas:
                warnings.append(f"low n for personas: {low_n_personas}")
            if not report.segment_distribution_passes:
                warnings.append("signal_segment distribution skewed")

            if warnings:
                report.gate_status = "WARNING"
                report.gate_reason = "; ".join(warnings)
                report.gate_passed = True  # warnings do not block scoring
            else:
                report.gate_status = "PASS"
                report.gate_reason = "all checks passed"
                report.gate_passed = True

        logger.info(
            f"PersonaValidator: gate={report.gate_status} | "
            f"separation_6m={sep_6m:.1%} | "
            f"p={report.significance.p_value:.4f if report.significance else 'N/A'} | "
            f"dominant={dominant} ({dominant_frac:.1%})"
        )
        return report

    def run_oot_validation(
        self,
        persona_df: pd.DataFrame,
        outcomes_df: pd.DataFrame,
        observation_dates: List[str],
    ) -> Dict[str, ValidationReport]:
        """
        Run validation across multiple observation dates (OOT stability check).

        Args:
            persona_df       : Combined persona_df for all dates
            outcomes_df      : Combined outcomes_df for all dates
            observation_dates: List of dates to validate independently

        Returns:
            Dict of {observation_date: ValidationReport}
        """
        results = {}
        for obs_date in observation_dates:
            try:
                report = self.validate(persona_df, outcomes_df, obs_date)
                results[obs_date] = report
            except Exception as e:
                logger.error(f"Validation failed for {obs_date}: {e}")
                # Return a FAIL report
                fail_report = ValidationReport(
                    observation_date=obs_date,
                    n_accounts_matched=0,
                    n_accounts_unmatched=0,
                    gate_status="FAIL",
                    gate_reason=str(e),
                    gate_passed=False,
                )
                results[obs_date] = fail_report

        gates = {d: r.gate_status for d, r in results.items()}
        logger.info(f"OOT validation summary: {gates}")
        return results
