"""
Persona Builder - Rule-Based Customer Segmentation
===================================================
v2.0 — Redesigned with strict feature separation and snapshot enforcement

Core principle: Segmentation uses LONG-TERM STRUCTURAL signals only.
Short-window (≤30/90-day) payment and engagement signals are reserved
exclusively for the propensity scorecard (see PROPENSITY_FEATURES).

Axes (redesigned from v1):
  - structural_payment_score : cure history, re-default count, effort in NPL
  - trajectory_score         : bureau stage dynamics (shape type, slope, stickiness)
  - capacity_score           : bureau debt burden and DSR
  - avoidance_score          : persistent avoidance flags (wrong number, lawyer, opt-out)

Axis scores are written to the audit output ONLY.
They must never enter the propensity scorecard feature set.

Feature separation:
  - SEGMENTATION_FEATURES : long-term structural signals — used here ONLY
  - PROPENSITY_FEATURES   : short-window signals — used in scorecard ONLY

5 Personas:
  ACTIVE_PAYER | SELECTIVE_DEFAULTER | LIQUIDITY_CONSTRAINED | STRATEGIC | DORMANT

SIGNAL_SEGMENT — structural context overlay (data availability + history flags):
  BUREAU_THIN_FILE | CHARGEOFF_HISTORY | MULTI_LENDER_DISTRESS | STANDARD
  Behavioural distinctions (selective cycling, chronic NPL) are in persona layer.

observation_date enforcement:
  - assign_batch() requires observation_date (YYYY-MM-DD)
  - All window-based features must be pre-computed upstream relative to that date
  - assign_batch() raises ValueError if observation_date is not supplied
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE SET DEFINITIONS  (strict zero-overlap separation)
# ─────────────────────────────────────────────────────────────────────────────

# Features used in segmentation ONLY — must never appear in scorecard training
SEGMENTATION_FEATURES: List[str] = [
    # Long-term payment structure
    "payment_effort_ratio_npl",       # avg payment / balance during NPL months
    "cure_count_24m",                 # times cured from NPL/SM → CURRENT/X in 24m
    "re_default_count_24m",           # times re-defaulted after cure in 24m
    "months_dormant",                 # months with zero payment activity
    "pct_months_npl_24m",             # % of last 24m in NPL or worse
    "worst_stage_ever",               # worst DPD stage in full history

    # Bureau structural signals
    "bureau_months_on_book",          # vintage — longer = more data confidence
    "bureau_total_outstanding",       # absolute debt burden (used for ratio computation)
    "bureau_monthly_instalment",      # monthly debt service (structural)
    "bureau_delinquent_other",        # delinquent on other lenders (0/1 or count)
    "bureau_new_loan_12m",            # credit-seeking in last 12m
    "bureau_secured_loan_flag",       # has secured collateral

    # Bureau stage dynamics (from bureau_stage_dynamics.py via BureauFeatureAdapter)
    "dpd_shape_type",                 # CLIFF/SLIDE/OSCILLATOR/RECOVERING/STABLE
    "dpd_slope_12m",                  # trajectory direction (negative = improving)
    "stage_stickiness_score",         # prob of staying in current stage
    "re_default_rate_rolling_24m",    # chronic re-defaulter rate
    "cure_rate_sm_to_lower",          # cure capability from SM stage

    # Persistent avoidance / stance signals
    "wrong_number_flag",
    "dispute_flag",
    "complaint_flag",
    "lawyer_mentioned",
    "legal_representation_flag",
    "sms_opt_out",
    # last_contact_outcome: persistent avoidance stance (hostile/refuse/legal keywords only)
    # classified as SEGMENTATION because it captures structural stance, not recency
    "last_contact_outcome",
]

# Features reserved for propensity scorecard ONLY — zero overlap with SEGMENTATION_FEATURES
# GAP 3: last_contact_outcome removed — it is a persistent avoidance signal (see above)
PROPENSITY_FEATURES: List[str] = [
    "payment_count_30d",
    "payment_count_90d",
    "payment_count_180d",
    "payment_amt_30d",
    "payment_amt_90d",
    "payment_amt_180d",
    "days_since_last_payment",
    "ptp_kept_rate",
    "last_payment_amount",
    "call_response_rate",
    "sms_response_rate",
    "days_since_last_contact",
    "contacts_made_30d",
    "contacts_made_90d",
    "calls_connected",
]


def validate_feature_separation() -> None:
    """
    Programmatic guard against feature leakage.
    Raises AssertionError if any feature appears in both SEGMENTATION_FEATURES
    and PROPENSITY_FEATURES. Call at module import time and in pipeline init.
    """
    overlap = set(SEGMENTATION_FEATURES) & set(PROPENSITY_FEATURES)
    if overlap:
        raise AssertionError(
            f"Feature leakage: {sorted(overlap)} appear in both "
            "SEGMENTATION_FEATURES and PROPENSITY_FEATURES. "
            "Remove from one list before proceeding."
        )


# Run at import time — fails loudly if the lists drift
validate_feature_separation()

# Axis score columns produced by assign_batch() — for audit/output ONLY
AXIS_SCORE_COLUMNS: List[str] = [
    "structural_payment_score",
    "trajectory_score",
    "capacity_score",
    "avoidance_score",
]


# ─────────────────────────────────────────────────────────────────────────────
# DEFAULT THRESHOLDS  (overridden by calibrate())
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_THRESHOLDS: Dict[str, float] = {
    "avoidance_strategic_min":         50.0,
    "trajectory_dormant_max":          30.0,
    "structural_payment_dormant_max":  20.0,
    "structural_payment_active_min":   60.0,
    "capacity_selective_min":          50.0,
    "structural_payment_selective_max":40.0,
    "capacity_constrained_max":        40.0,
    "trajectory_constrained_min":      30.0,
}


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT DATACLASS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PersonaAssignment:
    """Persona assignment with structural axis scores and strategic overlay."""
    account_id: str
    observation_date: str  # Snapshot anchor — all windows relative to this date
    persona: str           # ACTIVE_PAYER | SELECTIVE_DEFAULTER | LIQUIDITY_CONSTRAINED | STRATEGIC | DORMANT

    # Axis scores (0-100) — audit/transparency ONLY, never scorecard input
    structural_payment_score: float
    trajectory_score: float
    capacity_score: float
    avoidance_score: float

    # Structural context overlay (data availability + history flags only)
    signal_segment: str    # BUREAU_THIN_FILE | CHARGEOFF_HISTORY
                           # | MULTI_LENDER_DISTRESS | STANDARD

    # Metadata
    confidence_level: str           # HIGH | MEDIUM | LOW
    data_completeness_pct: float
    flags: List[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# PERSONA BUILDER
# ─────────────────────────────────────────────────────────────────────────────

class PersonaBuilder:
    """
    Rule-based persona assignment using long-term structural signals only.

    v2.0 changes vs v1:
    - observation_date required — feature windows must be upstream-anchored
    - Axes redesigned: structural_payment, trajectory, capacity, avoidance
    - Short-window payment/engagement signals removed (moved to PROPENSITY_FEATURES)
    - Bureau stage dynamics features (dpd_shape_type, dpd_slope_12m, etc.) added
    - Thresholds calibratable from training data via calibrate()
    - SIGNAL_SEGMENT computed as strategic overlay
    """

    def __init__(
        self,
        payment_threshold: float = 500.0,
        thresholds: Optional[Dict[str, float]] = None,
    ):
        """
        Args:
            payment_threshold : Minimum THB to count as meaningful payment
            thresholds        : Override decision thresholds. Use calibrate() for
                                data-driven splits derived from actual portfolio.
        """
        self.payment_threshold = payment_threshold
        self.thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
        self._is_calibrated = False

    # ── CALIBRATION ───────────────────────────────────────────────────────────

    def calibrate(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Compute data-driven decision thresholds from a training sample.

        GAP 4 FIX: Thresholds are computed in SCORE SPACE (0-100), not raw
        feature space. This makes them scale-invariant and directly comparable
        to what the decision tree evaluates. Prior version used raw
        bureau_total_outstanding percentiles (e.g. 480,000 THB) as thresholds
        against 0-100 axis scores — that comparison was dimensionally incorrect.

        Method: compute all four axis scores for every row in df, then derive
        percentile cuts from the score distributions.

        Args:
            df: Training DataFrame with SEGMENTATION_FEATURES columns. Minimum
                100 rows recommended for stable percentile estimates.

        Returns:
            Dict of calibrated thresholds in score space (also stored in self.thresholds)
        """
        if len(df) < 10:
            logger.warning(
                f"calibrate() called with only {len(df)} rows — "
                "thresholds may be unstable. Falling back to defaults."
            )
            return dict(DEFAULT_THRESHOLDS)

        # Compute axis scores for every row in the training sample
        sp_scores, tr_scores, cap_scores, av_scores = [], [], [], []
        for _, row in df.iterrows():
            sp_scores.append(self._calculate_structural_payment(row))
            tr_scores.append(self._calculate_trajectory(row))
            cap_scores.append(self._calculate_capacity(row))
            av_scores.append(self._calculate_avoidance(row))

        sp = np.array(sp_scores)
        tr = np.array(tr_scores)
        cap = np.array(cap_scores)
        av = np.array(av_scores)

        calibrated: Dict[str, float] = {
            # structural_payment thresholds in score space
            "structural_payment_active_min":    float(np.percentile(sp, 60)),
            "structural_payment_selective_max": float(np.percentile(sp, 40)),
            "structural_payment_dormant_max":   float(np.percentile(sp, 20)),

            # trajectory thresholds in score space
            "trajectory_dormant_max":           float(np.percentile(tr, 30)),
            "trajectory_constrained_min":       float(np.percentile(tr, 30)),

            # capacity thresholds in score space (not raw THB amounts)
            "capacity_selective_min":           float(np.percentile(cap, 50)),
            "capacity_constrained_max":         float(np.percentile(cap, 30)),

            # avoidance: binary flag sum — fixed at 50 regardless of distribution
            "avoidance_strategic_min":          50.0,
        }

        self.thresholds.update(calibrated)
        self._is_calibrated = True

        logger.info(
            f"PersonaBuilder calibrated on {len(df)} accounts (score-space percentiles). "
            f"Thresholds: {calibrated}"
        )
        return calibrated

    # ── SINGLE ACCOUNT ────────────────────────────────────────────────────────

    def assign_persona(
        self,
        account: pd.Series,
        observation_date: str,
    ) -> PersonaAssignment:
        """
        Assign persona based on long-term structural behavioral axes.

        Args:
            account          : Series with fields from SEGMENTATION_FEATURES
            observation_date : Snapshot date (YYYY-MM-DD). Stored for audit trail.
                               All window features must be pre-computed upstream
                               relative to this date.

        Expected fields (see SEGMENTATION_FEATURES for full list):
          payment_effort_ratio_npl, cure_count_24m, re_default_count_24m,
          months_dormant, pct_months_npl_24m,
          bureau_months_on_book, bureau_total_outstanding, bureau_monthly_instalment,
          dpd_shape_type, dpd_slope_12m, stage_stickiness_score,
          re_default_rate_rolling_24m, cure_rate_sm_to_lower,
          wrong_number_flag, dispute_flag, lawyer_mentioned, sms_opt_out
        """
        account_id = str(account.get("account_id", "unknown"))

        structural_payment = self._calculate_structural_payment(account)
        trajectory         = self._calculate_trajectory(account)
        capacity           = self._calculate_capacity(account)
        avoidance          = self._calculate_avoidance(account)

        persona = self._apply_decision_tree(structural_payment, trajectory, capacity, avoidance, account)

        signal_segment = self._compute_signal_segment(
            account, structural_payment, trajectory, capacity, avoidance
        )

        completeness, confidence = self._assess_confidence(account)
        flags = self._generate_flags(account, structural_payment, trajectory, capacity, avoidance)

        return PersonaAssignment(
            account_id=account_id,
            observation_date=observation_date,
            persona=persona,
            structural_payment_score=round(structural_payment, 2),
            trajectory_score=round(trajectory, 2),
            capacity_score=round(capacity, 2),
            avoidance_score=round(avoidance, 2),
            signal_segment=signal_segment,
            confidence_level=confidence,
            data_completeness_pct=round(completeness, 2),
            flags=flags,
        )

    # ── STRUCTURAL AXES ───────────────────────────────────────────────────────

    def _calculate_structural_payment(self, account: pd.Series) -> float:
        """
        Structural payment score (0-100) using long-term cure/effort signals.
        Higher = stronger long-term payment behaviour.

        Components:
        - Payment effort in NPL months (35%) : avg payment / balance during NPL
        - Cure history (30%)                 : times cured from NPL/SM in 24m
        - Re-default penalty (25%)           : chronic re-defaulting reduces score
        - Dormancy penalty (10%)             : months with zero payment activity
        """
        score = 0.0

        # Payment effort in NPL — structural willingness signal
        effort = float(account.get("payment_effort_ratio_npl", 0) or 0)
        score += min(100, effort * 100) * 0.35

        # Cure history: 3+ cures in 24m → 100 pts
        cures = float(account.get("cure_count_24m", 0) or 0)
        score += min(100, cures * 33.3) * 0.30

        # Re-default penalty: each re-default removes 25 pts from this component
        redefaults = float(account.get("re_default_count_24m", 0) or 0)
        score -= min(100, redefaults * 25) * 0.25

        # Dormancy penalty: 12m dormant = full penalty
        dormant = float(account.get("months_dormant", 0) or 0)
        score -= min(100, dormant * 8.33) * 0.10

        return min(100.0, max(0.0, score))

    def _calculate_trajectory(self, account: pd.Series) -> float:
        """
        Trajectory score (0-100) using bureau stage dynamics signals.
        Higher = improving or stable DPD trajectory.

        Components:
        - DPD shape type (35%)      : RECOVERING/STABLE positive; CLIFF/SLIDE negative
        - DPD slope 12m (25%)       : negative slope = improving
        - Stage stickiness (20%)    : low stickiness in bad stage = cure potential
        - Re-default rate (20%)     : chronic pattern lowers trajectory
        """
        score = 50.0  # Neutral baseline

        # DPD shape type
        shape = str(account.get("dpd_shape_type", "") or "").upper()
        shape_delta = {
            "RECOVERING": 40,
            "STABLE":     20,
            "OSCILLATOR":  0,
            "SLIDE":     -20,
            "CLIFF":     -40,
        }.get(shape, 0)
        score += shape_delta * 0.35

        # DPD slope: negative = improving (DPD falling)
        slope = float(account.get("dpd_slope_12m", 0) or 0)
        slope_contribution = max(-30.0, min(30.0, -slope * 2))
        score += slope_contribution * 0.25

        # Stage stickiness: in bad stage, low stickiness = cure potential
        stickiness = float(account.get("stage_stickiness_score", 0.5) or 0.5)
        pct_npl = float(account.get("pct_months_npl_24m", 0) or 0)
        if pct_npl > 0.5:
            # Currently in bad stage: low stickiness is GOOD
            stickiness_contribution = (1.0 - stickiness) * 30 - 15
        else:
            stickiness_contribution = 0.0
        score += stickiness_contribution * 0.20

        # Re-default rate: 100% rate → -40 pts
        redefault_rate = float(account.get("re_default_rate_rolling_24m", 0) or 0)
        score -= redefault_rate * 40 * 0.20

        return min(100.0, max(0.0, score))

    def _calculate_capacity(self, account: pd.Series) -> float:
        """
        Capacity score (0-100) using bureau structural debt signals.
        Higher = more financial capacity to pay.

        Components:
        - Bureau DSR proxy (40%) : total instalment / (outstanding / 3)
        - Balance burden (30%)   : CardX balance vs. total bureau outstanding
        - Secured asset (20%)    : has collateral = implicit capacity
        - Credit access (10%)    : new loans taken = able to access credit market
        """
        score = 0.0

        bureau_instalment  = float(account.get("bureau_monthly_instalment", 0) or 0)
        total_outstanding  = float(account.get("bureau_total_outstanding", 0) or 0)
        cardx_balance      = float(account.get("balance", 0) or 0)

        # DSR proxy
        if total_outstanding > 0:
            implied_income = total_outstanding / 3.0
            dsr = bureau_instalment / implied_income if implied_income > 0 else 1.0
            score += max(0, 100 - dsr * 100) * 0.4
        else:
            # No bureau data: use CardX balance as fallback
            if cardx_balance < 50_000:
                score += 70 * 0.4
            elif cardx_balance < 200_000:
                score += 50 * 0.4
            else:
                score += 30 * 0.4

        # Balance burden
        if total_outstanding > 0:
            burden_ratio = cardx_balance / total_outstanding
            score += max(0, 100 - burden_ratio * 100) * 0.3
        else:
            score += 50 * 0.3

        # Secured asset
        if account.get("bureau_secured_loan_flag", False):
            score += 70 * 0.2
        else:
            score += 30 * 0.2

        # Credit access
        new_loans = int(account.get("bureau_new_loan_12m", 0) or 0)
        score += min(100, new_loans * 30) * 0.1

        return min(100.0, max(0.0, score))

    def _calculate_avoidance(self, account: pd.Series) -> float:
        """
        Avoidance score (0-100) using persistent structural avoidance flags.
        Higher = stronger strategic avoidance behaviour.

        Components (all binary persistent flags — unchanged from v1):
        - Wrong number flag (25%)
        - Dispute / complaint flag (25%)
        - Lawyer mentioned (20%)
        - SMS opt-out (15%)
        - Refused engagement outcome (15%)
        """
        score = 0.0

        if account.get("wrong_number_flag", False):
            score += 25

        if account.get("dispute_flag", False) or account.get("complaint_flag", False):
            score += 25

        if account.get("lawyer_mentioned", False) or account.get("legal_representation_flag", False):
            score += 20

        if account.get("sms_opt_out", False):
            score += 15

        last_outcome = str(account.get("last_contact_outcome", "") or "").lower()
        if any(kw in last_outcome for kw in ("refuse", "hostile", "legal")):
            score += 15

        return min(100.0, max(0.0, score))

    # ── DECISION TREE ─────────────────────────────────────────────────────────

    def _apply_decision_tree(
        self,
        structural_payment: float,
        trajectory: float,
        capacity: float,
        avoidance: float,
        account: Optional[pd.Series] = None,
    ) -> str:
        """
        Decision tree using calibratable thresholds from self.thresholds.

        Priority order (first match wins):
        1. High avoidance                   → STRATEGIC
        2. Low trajectory + low payment     → DORMANT
        3. High structural payment          → ACTIVE_PAYER
        4. Chronic cycling (cure+redefault) → SELECTIVE_DEFAULTER
           (GAP 5: moved from SIGNAL_SEGMENT — this is behavioral, not structural context)
        5. High capacity + low payment      → SELECTIVE_DEFAULTER
        6. Low capacity + moderate traj     → LIQUIDITY_CONSTRAINED
        7. Default                          → SELECTIVE_DEFAULTER
        """
        t = self.thresholds

        if avoidance > t["avoidance_strategic_min"]:
            return "STRATEGIC"

        if trajectory < t["trajectory_dormant_max"] and structural_payment < t["structural_payment_dormant_max"]:
            return "DORMANT"

        if structural_payment > t["structural_payment_active_min"]:
            return "ACTIVE_PAYER"

        # Chronic cycling: has demonstrated capacity by curing, but keeps re-defaulting.
        # Absorbed from SELECTIVE_CHRONIC signal segment (GAP 5 fix).
        if account is not None:
            cures     = float(account.get("cure_count_24m", 0) or 0)
            redefaults = float(account.get("re_default_count_24m", 0) or 0)
            if cures >= 2 and redefaults >= 2 and capacity > t["capacity_constrained_max"]:
                return "SELECTIVE_DEFAULTER"

        if capacity > t["capacity_selective_min"] and structural_payment < t["structural_payment_selective_max"]:
            return "SELECTIVE_DEFAULTER"

        if capacity < t["capacity_constrained_max"] and trajectory > t["trajectory_constrained_min"]:
            return "LIQUIDITY_CONSTRAINED"

        return "SELECTIVE_DEFAULTER"

    # ── SIGNAL SEGMENT ────────────────────────────────────────────────────────

    def _compute_signal_segment(
        self,
        account: pd.Series,
        structural_payment: float,
        trajectory: float,
        capacity: float,
        avoidance: float,
    ) -> str:
        """
        Compute structural context segment — layered ON TOP of persona.

        GAP 5 FIX: SIGNAL_SEGMENT now represents structural / data-availability
        context ONLY. Behavioural logic removed:
          - SELECTIVE_CHRONIC removed → absorbed into _apply_decision_tree()
          - CHRONIC_NPL removed → captured by trajectory_score (dpd_shape_type)
          - HIGH_WORTH_RECOVERY removed → belongs in action routing, not segmentation

        Remaining segments represent structural context that conditions treatment
        strategy without duplicating persona behavioural differentiation:

        - BUREAU_THIN_FILE       : <6 months on book — structural signals unreliable
        - CHARGEOFF_HISTORY      : worst stage ever CO or CO_DEEP — structural write-off history
        - MULTI_LENDER_DISTRESS  : delinquent on ≥1 other lenders — systemic financial stress
        - STANDARD               : no special structural context flag

        Priority: BUREAU_THIN_FILE → CHARGEOFF_HISTORY → MULTI_LENDER_DISTRESS → STANDARD
        """
        bureau_mob      = float(account.get("bureau_months_on_book", 0) or 0)
        worst_stage     = str(account.get("worst_stage_ever", "") or "").upper()
        delinquent_other = float(account.get("bureau_delinquent_other", 0) or 0)

        if bureau_mob < 6:
            return "BUREAU_THIN_FILE"

        if worst_stage in ("CO", "CO_DEEP"):
            return "CHARGEOFF_HISTORY"

        if delinquent_other >= 1:
            return "MULTI_LENDER_DISTRESS"

        return "STANDARD"

    # ── CONFIDENCE & FLAGS ────────────────────────────────────────────────────

    def _assess_confidence(self, account: pd.Series) -> Tuple[float, str]:
        """
        Assess confidence based on structural feature completeness.
        Required fields align with SEGMENTATION_FEATURES (core subset).
        """
        required_fields = [
            "payment_effort_ratio_npl",
            "cure_count_24m",
            "re_default_count_24m",
            "months_dormant",
            "pct_months_npl_24m",
            "bureau_total_outstanding",
            "bureau_months_on_book",
            "dpd_shape_type",
            "dpd_slope_12m",
            "stage_stickiness_score",
        ]

        present = sum(
            1 for f in required_fields
            if f in account.index and pd.notna(account[f])
        )
        completeness = (present / len(required_fields)) * 100

        confidence = "HIGH" if completeness >= 80 else ("MEDIUM" if completeness >= 60 else "LOW")
        return completeness, confidence

    def _generate_flags(
        self,
        account: pd.Series,
        structural_payment: float,
        trajectory: float,
        capacity: float,
        avoidance: float,
    ) -> List[str]:
        """Generate context flags for audit and transparency."""
        flags: List[str] = []

        if structural_payment > 70:
            flags.append("strong_structural_payment_history")
        elif structural_payment < 20:
            flags.append("no_meaningful_payment_history")

        if trajectory > 70:
            flags.append("improving_dpd_trajectory")
        elif trajectory < 30:
            flags.append("deteriorating_or_stagnant_trajectory")

        if capacity > 70:
            flags.append("high_capacity")
        elif capacity < 30:
            flags.append("low_capacity")

        if avoidance > 50:
            flags.append("avoidance_behaviour")

        redefaults = float(account.get("re_default_count_24m", 0) or 0)
        if redefaults >= 2:
            flags.append("repeat_re_defaulter")

        shape = str(account.get("dpd_shape_type", "") or "").upper()
        if shape == "RECOVERING":
            flags.append("recovery_trajectory")
        elif shape in ("CLIFF", "SLIDE"):
            flags.append("deterioration_trajectory")

        if pd.notna(account.get("bureau_delinquent_other")) and float(account.get("bureau_delinquent_other", 0) or 0) > 0:
            flags.append("delinquent_elsewhere")

        stage = str(account.get("stage", "") or "").upper()
        if stage in ("CO", "CO_DEEP"):
            flags.append("chargeoff_stage")

        return flags

    # ── BATCH PROCESSING ─────────────────────────────────────────────────────

    def assign_batch(
        self,
        df: pd.DataFrame,
        observation_date: str,
    ) -> pd.DataFrame:
        """
        Assign personas for a batch of accounts.

        Args:
            df               : DataFrame with SEGMENTATION_FEATURES columns + account_id.
                               Caller should filter to only segmentation-relevant columns
                               before passing in — do NOT pass the full feature matrix.
            observation_date : Snapshot date (YYYY-MM-DD). All feature windows must
                               be pre-computed upstream relative to this date.
                               Raises ValueError if not supplied.

        Returns:
            DataFrame with columns:
              account_id, observation_date, persona, signal_segment,
              structural_payment_score, trajectory_score, capacity_score,
              avoidance_score, confidence_level, data_completeness_pct, flags

        IMPORTANT — caller must NOT merge axis score columns (structural_payment_score,
        trajectory_score, capacity_score, avoidance_score) into the propensity
        scorecard feature set. These are audit columns only.
        Use AXIS_SCORE_COLUMNS constant to identify and exclude them.
        """
        if not observation_date:
            raise ValueError(
                "observation_date is required in assign_batch() — "
                "it anchors all feature window computations."
            )

        results = []
        for _, row in df.iterrows():
            a = self.assign_persona(row, observation_date)
            results.append({
                "account_id":               a.account_id,
                "observation_date":         a.observation_date,
                "persona":                  a.persona,
                "signal_segment":           a.signal_segment,
                # Axis scores: audit/transparency ONLY — excluded from scorecard
                "structural_payment_score": a.structural_payment_score,
                "trajectory_score":         a.trajectory_score,
                "capacity_score":           a.capacity_score,
                "avoidance_score":          a.avoidance_score,
                "confidence_level":         a.confidence_level,
                "data_completeness_pct":    a.data_completeness_pct,
                "flags":                    "; ".join(a.flags),
            })

        return pd.DataFrame(results)
