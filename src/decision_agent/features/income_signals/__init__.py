"""
Income-specific signal extraction modules.

These modules extract income-specific features from bank transaction data:
- deposit_periodicity: Detect salary payment cadence (bi-weekly, monthly)
- deposit_stability: Measure deposit amount stability (CV, MAD, trimmed mean)
- counterparty_analysis: Identify employers and recurring payments (future)
"""

from decision_agent.features.income_signals.deposit_periodicity import (
    detect_deposit_periodicity,
    DepositPeriodicityDetector
)
from decision_agent.features.income_signals.deposit_stability import (
    compute_deposit_stability_metrics,
    DepositStabilityCalculator
)

__all__ = [
    "detect_deposit_periodicity",
    "DepositPeriodicityDetector",
    "compute_deposit_stability_metrics",
    "DepositStabilityCalculator",
]
