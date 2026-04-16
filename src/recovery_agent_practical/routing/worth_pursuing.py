"""
Worth Pursuing — ERV-Based Value Layer
=======================================
Assigns a value band (HIGH / MEDIUM / LOW) and a "worth_pursuing" flag to each
account based on Expected Recovery Value (ERV).

Architecture position:
  - Runs AFTER scorecard scoring (requires p_recovery, expected_recovery)
  - Runs AFTER persona assignment (requires persona, signal_segment)
  - Output consumed by action routing as a CONDITIONING layer
  - Does NOT replace segmentation — it conditions effort intensity per segment

ERV definition (unchanged from existing scorecard output):
  ERV = P(recovery_6m) × E(recovery_amount | pays)
  Source: `expected_recovery` column in daily_scoring_df (RecoveryScorecard6M output)

Value bands:
  HIGH   — ERV above 67th percentile of reference population, OR balance override
  MEDIUM — ERV between 33rd and 67th percentile
  LOW    — ERV below 33rd percentile

"Worth pursuing" flag (conservative — both criteria must hold to flag NOT worth it):
  NOT_WORTH_PURSUING only when ALL THREE:
    1. erv_band == LOW
    2. balance < LOW_BALANCE_CEILING (50,000 THB)
    3. months_since_chargeoff > LONG_VINTAGE_MONTHS (24 months)

  This is intentionally conservative: missing any one condition → WORTH_PURSUING.

Thresholds:
  calibrate(reference_df) → derives p33/p67 from reference ERV distribution.
  Without calibration, falls back to hard-coded defaults (5,000 / 15,000 THB).

Persona × ERV combined decision matrix (informational — not a hard rule):

  Persona                | LOW ERV           | MEDIUM ERV          | HIGH ERV
  ─────────────────────────────────────────────────────────────────────────────
  ACTIVE_PAYER           | MONITOR           | LIGHT_TOUCH         | PRIORITISE
  SELECTIVE_DEFAULTER    | HOLD              | SETTLEMENT_OFFER    | LEGAL_REVIEW
  LIQUIDITY_CONSTRAINED  | MINIMAL_CONTACT   | PAYMENT_PLAN        | PAYMENT_PLAN
  STRATEGIC              | HOLD / AGENCY     | LEGAL_CONSIDERATION | LEGAL_REVIEW
  DORMANT                | AGENCY / WRITE_OFF| AGENCY              | FIELD_VISIT

  Neither dimension alone is sufficient:
    - Segmentation without ERV: can't prioritise effort within a persona group
    - ERV without segmentation: can't choose the right contact strategy

NO cost modelling. NO NPV. NO uplift. This layer is purely value-based prioritisation.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# Default ERV thresholds (THB) — used when calibrate() has not been called
DEFAULT_ERV_HIGH_THRESHOLD   = 15_000.0   # THB — top third (approx)
DEFAULT_ERV_LOW_THRESHOLD    =  5_000.0   # THB — bottom third (approx)

# Balance override: accounts above this are always HIGH regardless of ERV
HIGH_BALANCE_OVERRIDE        = 200_000.0  # THB

# "Not worth pursuing" conditions (ALL must hold simultaneously)
LOW_BALANCE_CEILING          = 50_000.0   # THB — must be small balance
LONG_VINTAGE_MONTHS          = 24         # months since first chargeoff

# Value band labels
BAND_HIGH   = "HIGH"
BAND_MEDIUM = "MEDIUM"
BAND_LOW    = "LOW"

# Persona × ERV combined strategy matrix (informational)
PERSONA_VALUE_MATRIX: Dict[str, Dict[str, str]] = {
    "ACTIVE_PAYER": {
        BAND_HIGH:   "PRIORITISE",
        BAND_MEDIUM: "LIGHT_TOUCH",
        BAND_LOW:    "MONITOR",
    },
    "SELECTIVE_DEFAULTER": {
        BAND_HIGH:   "LEGAL_REVIEW",
        BAND_MEDIUM: "SETTLEMENT_OFFER",
        BAND_LOW:    "HOLD",
    },
    "LIQUIDITY_CONSTRAINED": {
        BAND_HIGH:   "PAYMENT_PLAN",
        BAND_MEDIUM: "PAYMENT_PLAN",
        BAND_LOW:    "MINIMAL_CONTACT",
    },
    "STRATEGIC": {
        BAND_HIGH:   "LEGAL_REVIEW",
        BAND_MEDIUM: "LEGAL_CONSIDERATION",
        BAND_LOW:    "HOLD_OR_AGENCY",
    },
    "DORMANT": {
        BAND_HIGH:   "FIELD_VISIT",
        BAND_MEDIUM: "AGENCY",
        BAND_LOW:    "AGENCY_OR_WRITE_OFF",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ValueBandResult:
    """Value layer output for a single account."""
    account_id: str
    erv:               float   # Expected Recovery Value (THB)
    erv_band:          str     # HIGH | MEDIUM | LOW
    worth_pursuing:    bool    # False only when low ERV + low balance + long vintage
    collection_intensity: str  # PRIORITISE | STANDARD | MINIMAL | HOLD
    combined_strategy: str     # from PERSONA_VALUE_MATRIX


@dataclass
class ValueDistributionReport:
    """Distribution of accounts across value bands."""
    n_total: int
    band_counts: Dict[str, int] = field(default_factory=dict)
    band_fractions: Dict[str, float] = field(default_factory=dict)
    not_worth_pursuing_count: int = 0
    not_worth_pursuing_pct: float = 0.0
    erv_p33: float = 0.0
    erv_p67: float = 0.0
    erv_mean: float = 0.0
    erv_median: float = 0.0

    def to_text(self) -> str:
        lines = [
            "VALUE BAND DISTRIBUTION",
            f"  Total accounts : {self.n_total:,}",
            f"  ERV thresholds : LOW < {self.erv_p33:,.0f} THB ≤ MEDIUM < {self.erv_p67:,.0f} THB ≤ HIGH",
            f"  ERV mean       : {self.erv_mean:,.0f} THB",
            f"  ERV median     : {self.erv_median:,.0f} THB",
            "",
            f"  {'Band':8s}  {'Count':8s}  {'Fraction':8s}",
            "  " + "-" * 30,
        ]
        for band in [BAND_HIGH, BAND_MEDIUM, BAND_LOW]:
            n = self.band_counts.get(band, 0)
            f = self.band_fractions.get(band, 0.0)
            lines.append(f"  {band:8s}  {n:8,d}  {f:7.1%}")
        lines += [
            "",
            f"  Not worth pursuing : {self.not_worth_pursuing_count:,} "
            f"({self.not_worth_pursuing_pct:.1%})",
        ]
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATOR
# ─────────────────────────────────────────────────────────────────────────────

class WorthPursuingEvaluator:
    """
    Assigns ERV-based value bands and worth_pursuing flags.

    Step 1: calibrate(reference_df) — derive data-driven ERV thresholds from
            a reference population (33rd and 67th percentiles of ERV).
    Step 2: evaluate(scoring_df) — apply value bands to a scored batch.

    Input DataFrame (scoring_df) must contain:
      - account_id
      - expected_recovery       : float (ERV = P × E from scorecard)
      - p_recovery_6m           : float (P(recovery) from scorecard)
      - balance                 : float (current outstanding balance, THB)
      - months_since_chargeoff  : float (0 if not in chargeoff)
      - persona                 : str (from PersonaBuilder)
    """

    def __init__(self):
        self._erv_low_threshold  = DEFAULT_ERV_LOW_THRESHOLD
        self._erv_high_threshold = DEFAULT_ERV_HIGH_THRESHOLD
        self._is_calibrated      = False

    # ── CALIBRATION ───────────────────────────────────────────────────────────

    def calibrate(self, reference_df: pd.DataFrame) -> Tuple[float, float]:
        """
        Derive ERV thresholds from a reference population.

        Thresholds are the 33rd and 67th percentile of `expected_recovery`
        in the reference data. This ensures the three bands are approximately
        equal-sized on the reference population.

        Args:
            reference_df: DataFrame with `expected_recovery` column.
                          Should be a representative sample (≥ 500 accounts).

        Returns:
            (low_threshold, high_threshold) in THB.
        """
        if "expected_recovery" not in reference_df.columns:
            raise ValueError("reference_df must contain 'expected_recovery' column")

        erv = reference_df["expected_recovery"].dropna()
        if len(erv) < 100:
            logger.warning(
                f"calibrate() called with {len(erv)} accounts — "
                "thresholds may be unstable. Recommend ≥ 500."
            )

        self._erv_low_threshold  = float(np.percentile(erv, 33))
        self._erv_high_threshold = float(np.percentile(erv, 67))
        self._is_calibrated      = True

        logger.info(
            f"WorthPursuingEvaluator calibrated on {len(erv)} accounts. "
            f"ERV p33={self._erv_low_threshold:,.0f} THB, "
            f"p67={self._erv_high_threshold:,.0f} THB."
        )
        return self._erv_low_threshold, self._erv_high_threshold

    # ── EVALUATION ────────────────────────────────────────────────────────────

    def evaluate(self, scoring_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply value bands to a scored batch.

        Args:
            scoring_df: Daily scoring DataFrame (output of pipeline.score_batch()
                        or equivalent). Must contain expected_recovery, balance,
                        months_since_chargeoff, persona.

        Returns:
            scoring_df with added columns:
              erv_band            : HIGH | MEDIUM | LOW
              worth_pursuing      : True | False
              collection_intensity: PRIORITISE | STANDARD | MINIMAL | HOLD
              combined_strategy   : from PERSONA_VALUE_MATRIX
        """
        if not self._is_calibrated:
            logger.warning(
                "WorthPursuingEvaluator not calibrated — using default thresholds "
                f"({DEFAULT_ERV_LOW_THRESHOLD:,.0f} / {DEFAULT_ERV_HIGH_THRESHOLD:,.0f} THB). "
                "Call calibrate() with a reference population for data-driven thresholds."
            )

        df = scoring_df.copy()

        erv = df["expected_recovery"].fillna(0.0)
        balance = df.get("balance", pd.Series(0.0, index=df.index)).fillna(0.0)
        vintage = df.get("months_since_chargeoff",
                         pd.Series(0.0, index=df.index)).fillna(0.0)
        persona_col = df.get("persona",
                             pd.Series("DORMANT", index=df.index)).fillna("DORMANT")

        # ── Band assignment ───────────────────────────────────────────────────
        # Balance override: large balance → always HIGH regardless of ERV
        balance_override = balance >= HIGH_BALANCE_OVERRIDE

        erv_band = np.where(
            balance_override | (erv >= self._erv_high_threshold),
            BAND_HIGH,
            np.where(erv >= self._erv_low_threshold, BAND_MEDIUM, BAND_LOW),
        )
        df["erv_band"] = erv_band

        # ── Worth pursuing flag ───────────────────────────────────────────────
        # NOT worth pursuing only when all three hold simultaneously
        not_worth = (
            (df["erv_band"] == BAND_LOW) &
            (balance < LOW_BALANCE_CEILING) &
            (vintage > LONG_VINTAGE_MONTHS)
        )
        df["worth_pursuing"] = ~not_worth

        # ── Collection intensity ──────────────────────────────────────────────
        df["collection_intensity"] = np.select(
            [
                df["erv_band"] == BAND_HIGH,
                df["erv_band"] == BAND_MEDIUM,
                df["worth_pursuing"] == False,  # noqa: E712 — explicit comparison
            ],
            ["PRIORITISE", "STANDARD", "HOLD"],
            default="MINIMAL",
        )

        # ── Combined strategy (persona × erv_band matrix lookup) ─────────────
        def _combined(row: pd.Series) -> str:
            persona = str(row.get("persona", "DORMANT"))
            band    = str(row.get("erv_band", BAND_LOW))
            return PERSONA_VALUE_MATRIX.get(persona, {}).get(band, "STANDARD")

        df["combined_strategy"] = df.apply(_combined, axis=1)

        logger.info(
            f"WorthPursuingEvaluator: {len(df)} accounts evaluated. "
            f"Bands: HIGH={( df['erv_band']==BAND_HIGH).sum()}, "
            f"MEDIUM={(df['erv_band']==BAND_MEDIUM).sum()}, "
            f"LOW={(df['erv_band']==BAND_LOW).sum()}. "
            f"Not worth pursuing: {(~df['worth_pursuing']).sum()}."
        )
        return df

    # ── DISTRIBUTION REPORT ──────────────────────────────────────────────────

    def distribution_report(self, evaluated_df: pd.DataFrame) -> ValueDistributionReport:
        """
        Compute value band distribution statistics from an evaluated DataFrame.

        Args:
            evaluated_df: Output of evaluate() — must have erv_band, worth_pursuing,
                          expected_recovery columns.

        Returns:
            ValueDistributionReport with counts, fractions, and ERV statistics.
        """
        n = len(evaluated_df)
        erv = evaluated_df["expected_recovery"].fillna(0.0)
        band_counts = evaluated_df["erv_band"].value_counts().to_dict()
        band_fractions = {k: v / n for k, v in band_counts.items()}

        not_wp = int((~evaluated_df["worth_pursuing"]).sum())

        return ValueDistributionReport(
            n_total=n,
            band_counts=band_counts,
            band_fractions={k: round(v, 4) for k, v in band_fractions.items()},
            not_worth_pursuing_count=not_wp,
            not_worth_pursuing_pct=round(not_wp / n, 4) if n > 0 else 0.0,
            erv_p33=round(float(self._erv_low_threshold), 2),
            erv_p67=round(float(self._erv_high_threshold), 2),
            erv_mean=round(float(erv.mean()), 2),
            erv_median=round(float(erv.median()), 2),
        )

    # ── PERSONA × VALUE CROSS-TAB ─────────────────────────────────────────────

    @staticmethod
    def persona_value_crosstab(evaluated_df: pd.DataFrame) -> pd.DataFrame:
        """
        Cross-tabulate personas against value bands.

        Shows count and percentage of each band within each persona.
        Demonstrates how the two layers complement each other.

        Args:
            evaluated_df: Output of evaluate() with persona and erv_band columns.

        Returns:
            DataFrame with personas as rows, bands as columns, values as pct.
        """
        if "persona" not in evaluated_df.columns or "erv_band" not in evaluated_df.columns:
            raise ValueError("evaluated_df must have 'persona' and 'erv_band' columns")

        ct = pd.crosstab(
            evaluated_df["persona"],
            evaluated_df["erv_band"],
            normalize="index",
        ).rename(columns={
            BAND_HIGH:   "HIGH_pct",
            BAND_MEDIUM: "MEDIUM_pct",
            BAND_LOW:    "LOW_pct",
        })

        # Add count column
        counts = evaluated_df.groupby("persona").size().rename("n_accounts")
        ct = ct.join(counts)

        return ct.round(3)
