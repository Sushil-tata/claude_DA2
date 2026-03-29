"""
Model Output Contract
=====================
Single source of truth for:
  - Required columns in recovery.model_scores
  - PII blacklist (enforced by DeltaLakeWriter and claude_reasoner)
  - Column type and range validation

Used by:
  recovery_engine/output/delta_writer.py  — validates before writing
  src/agents/claude_reasoner.py           — strips PII before API call
  tests/integration/test_delta_integration.py — asserts no PII in Delta
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple
import pandas as pd


# ── PII Blacklist ─────────────────────────────────────────────────────────────
# Any column whose name contains one of these strings (case-insensitive)
# must NEVER be written to recovery.model_scores or passed to claude_reasoner.

PII_BLACKLIST: set[str] = {
    "name", "phone", "phone_number", "mobile", "address",
    "national_id", "id_card", "passport", "email",
    "date_of_birth", "dob", "firstname", "lastname",
    "surname", "citizen_id",
}

# Columns claude_reasoner.py is allowed to receive — explicit allowlist
CLAUDE_ALLOWED_COLUMNS: set[str] = {
    "account_id",
    "signal_quadrant",
    "segment_label",
    "propensity_30d",
    "propensity_90d",
    "propensity_180d",
    "low_confidence_flag",
    "d_optimal",
    "erv_at_d_optimal",
    "model_version",
    "constraint_override_flag",   # added by Constraint Agent
}

SIGNAL_QUADRANTS: set[str] = {"A", "B", "C", "D"}
DISCOUNT_LEVELS: list[float] = [0.20, 0.30, 0.40, 0.50, 0.60]


# ── Required columns and their types ─────────────────────────────────────────
@dataclass
class ColumnSpec:
    dtype: str                          # pandas dtype string
    nullable: bool = False
    range: Optional[Tuple[float, float]] = None
    enum: Optional[set] = None


REQUIRED_COLUMNS: dict[str, ColumnSpec] = {
    "account_id":          ColumnSpec("object"),
    "score_date":          ColumnSpec("object"),   # date / string
    "signal_quadrant":     ColumnSpec("object",    enum=SIGNAL_QUADRANTS),
    "segment_label":       ColumnSpec("object",    nullable=True),
    "propensity_30d":      ColumnSpec("float64",   range=(0.0, 1.0)),
    "propensity_90d":      ColumnSpec("float64",   range=(0.0, 1.0)),
    "propensity_180d":     ColumnSpec("float64",   range=(0.0, 1.0)),
    "low_confidence_flag": ColumnSpec("bool"),
    "d_optimal":           ColumnSpec("float64",   range=(0.20, 0.65)),
    "erv_at_d_optimal":    ColumnSpec("float64"),
    "model_version":       ColumnSpec("object"),
    "experiment_id":       ColumnSpec("object"),
}

# Elasticity grid columns (nullable — not required for D-quadrant)
for _d in [20, 30, 40, 50, 60]:
    REQUIRED_COLUMNS[f"e_amount_d{_d}"] = ColumnSpec("float64", nullable=True, range=(0.0, 1.0))
    REQUIRED_COLUMNS[f"p_accept_d{_d}"] = ColumnSpec("float64", nullable=True, range=(0.0, 1.0))
    REQUIRED_COLUMNS[f"erv_d{_d}"]      = ColumnSpec("float64", nullable=True)


# ── Validation ────────────────────────────────────────────────────────────────
class SchemaValidationError(Exception):
    pass


class PIIViolationError(Exception):
    pass


def validate_pii(df: pd.DataFrame, context: str = "") -> None:
    """Raise PIIViolationError if any PII column is present."""
    lower_cols = {c.lower() for c in df.columns}
    violations = []
    for col in df.columns:
        col_lower = col.lower()
        for pii_term in PII_BLACKLIST:
            if pii_term in col_lower:
                violations.append(col)
                break
    if violations:
        raise PIIViolationError(
            f"[{context}] PII columns detected — must never be written to Delta "
            f"or sent to Claude API: {violations}"
        )


def validate_schema(df: pd.DataFrame) -> list[str]:
    """
    Validate a DataFrame against the model_scores contract.
    Returns list of warning strings (empty = fully valid).
    Raises SchemaValidationError on hard failures.
    """
    errors = []
    warnings = []

    # 1. PII check (hard fail)
    validate_pii(df, context="model_scores")

    # 2. Required columns present
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaValidationError(f"Missing required columns: {missing}")

    # 3. Nullability
    for col, spec in REQUIRED_COLUMNS.items():
        if not spec.nullable and df[col].isnull().any():
            null_count = df[col].isnull().sum()
            errors.append(f"Column '{col}' has {null_count} nulls but is NOT nullable")

    # 4. Enum values
    for col, spec in REQUIRED_COLUMNS.items():
        if spec.enum and col in df.columns:
            bad = df[col].dropna()[~df[col].dropna().isin(spec.enum)]
            if len(bad) > 0:
                errors.append(
                    f"Column '{col}' has invalid values: {bad.unique().tolist()} "
                    f"(allowed: {spec.enum})"
                )

    # 5. Range checks
    for col, spec in REQUIRED_COLUMNS.items():
        if spec.range and col in df.columns:
            lo, hi = spec.range
            out_of_range = df[col].dropna()
            out_of_range = out_of_range[(out_of_range < lo) | (out_of_range > hi)]
            if len(out_of_range) > 0:
                errors.append(
                    f"Column '{col}' has {len(out_of_range)} values outside "
                    f"range [{lo}, {hi}]"
                )

    if errors:
        raise SchemaValidationError("\n".join(errors))

    return warnings
