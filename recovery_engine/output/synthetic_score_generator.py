"""
SyntheticScoreGenerator
=======================
Generates realistic synthetic model_scores rows for each signal quadrant.

Used by the agentic team to bootstrap agent development before
recovery-engine-v2 produces real scores.

Realistic ranges per quadrant (from portfolio calibration):
  A: High propensity + High ERV  — settlement offer candidates
  B: High propensity + Low ERV   — digital self-serve nudge
  C: Low propensity  + High ERV  — agent call + escalation
  D: Low propensity  + Low ERV   — hold/monitor, low_confidence_flag=True

Output: /data/synthetic/model_scores_synthetic_{date}.parquet
Share with agentic team on Day 2 of contract week.
"""

import logging
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from contracts.model_output_contract import DISCOUNT_LEVELS, validate_schema

logger = logging.getLogger(__name__)

# Portfolio base rate used for D-quadrant propensity calibration
PORTFOLIO_BASE_RATE_30D  = 0.16
PORTFOLIO_BASE_RATE_180D = 0.24

QUADRANT_WEIGHTS = {"A": 0.20, "B": 0.30, "C": 0.25, "D": 0.25}

SEGMENT_LABELS_BY_QUADRANT = {
    "A": ["SALARY_LIKE", "PAYROLL", "SME_STABLE"],
    "B": ["SALARY_LIKE", "GIG_FREELANCE"],
    "C": ["SME_VOLATILE", "GIG_FREELANCE", "PASSIVE_INVESTOR"],
    "D": None,  # No segment label for D-quadrant
}


class SyntheticScoreGenerator:
    """
    Generates N synthetic model score rows with realistic distributions.

    Args:
        n:          Total number of accounts to generate
        seed:       Random seed for reproducibility
        score_date: Scoring date (defaults to today)
    """

    def __init__(
        self,
        n: int = 10_000,
        seed: int = 42,
        score_date: Optional[date] = None,
    ):
        self.n          = n
        self.rng        = np.random.default_rng(seed)
        self.score_date = str(score_date or date.today())

    def generate(self) -> pd.DataFrame:
        """
        Generate synthetic scores DataFrame.

        Returns:
            pandas DataFrame matching recovery.model_scores schema exactly.
        """
        rows = []
        quadrant_counts = {
            q: max(1, round(self.n * w))
            for q, w in QUADRANT_WEIGHTS.items()
        }

        for quadrant, count in quadrant_counts.items():
            rows.append(self._generate_quadrant(quadrant, count))

        df = pd.concat(rows, ignore_index=True)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
        df["account_id"]    = [f"ACC{str(i).zfill(7)}" for i in range(len(df))]
        df["score_date"]    = self.score_date
        df["model_version"] = "recovery-engine-v2.0.0-synthetic"
        df["experiment_id"] = f"synthetic-{self.score_date}"

        # Validate before returning
        validate_schema(df)
        logger.info(
            f"[SyntheticScoreGenerator] Generated {len(df)} rows | "
            f"quadrants={df['signal_quadrant'].value_counts().to_dict()}"
        )
        return df

    def generate_and_save(
        self,
        output_dir: Path = Path("/data/synthetic"),
    ) -> Path:
        """Generate and write to parquet. Returns the output path."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        df = self.generate()
        path = output_dir / f"model_scores_synthetic_{self.score_date}.parquet"
        df.to_parquet(path, index=False)
        logger.info(f"[SyntheticScoreGenerator] Saved → {path}")
        return path

    # ── Per-quadrant generators ───────────────────────────────────────────────

    def _generate_quadrant(self, quadrant: str, n: int) -> pd.DataFrame:
        gen = getattr(self, f"_quadrant_{quadrant.lower()}")
        return gen(n)

    def _quadrant_a(self, n: int) -> pd.DataFrame:
        p30  = self.rng.uniform(0.35, 0.75, n)
        p90  = np.clip(p30 + self.rng.uniform(0.03, 0.08, n), 0, 1)
        p180 = np.clip(p30 + self.rng.uniform(0.05, 0.12, n), 0, 1)
        erv_opt = np.maximum(0, self.rng.normal(12_000, 3_000, n))
        d_opt = self.rng.uniform(0.25, 0.40, n)
        return self._build_rows("A", n, p30, p90, p180, d_opt, erv_opt, low_conf=False)

    def _quadrant_b(self, n: int) -> pd.DataFrame:
        p30  = self.rng.uniform(0.20, 0.55, n)
        p90  = np.clip(p30 + self.rng.uniform(0.02, 0.07, n), 0, 1)
        p180 = np.clip(p30 + self.rng.uniform(0.04, 0.10, n), 0, 1)
        erv_opt = np.maximum(0, self.rng.normal(8_000, 2_500, n))
        d_opt = self.rng.uniform(0.30, 0.50, n)
        return self._build_rows("B", n, p30, p90, p180, d_opt, erv_opt, low_conf=False)

    def _quadrant_c(self, n: int) -> pd.DataFrame:
        p30  = self.rng.uniform(0.15, 0.45, n)
        p90  = np.clip(p30 + self.rng.uniform(0.02, 0.06, n), 0, 1)
        p180 = np.clip(p30 + self.rng.uniform(0.03, 0.08, n), 0, 1)
        erv_opt = np.maximum(0, self.rng.normal(6_000, 2_000, n))
        d_opt = self.rng.uniform(0.25, 0.40, n)
        return self._build_rows("C", n, p30, p90, p180, d_opt, erv_opt, low_conf=False)

    def _quadrant_d(self, n: int) -> pd.DataFrame:
        p30  = np.full(n, PORTFOLIO_BASE_RATE_30D * 0.5)  + self.rng.uniform(-0.02, 0.02, n)
        p90  = np.full(n, PORTFOLIO_BASE_RATE_30D * 0.7)  + self.rng.uniform(-0.02, 0.02, n)
        p180 = np.full(n, PORTFOLIO_BASE_RATE_180D * 0.5) + self.rng.uniform(-0.02, 0.02, n)
        p30  = np.clip(p30,  0, 1)
        p90  = np.clip(p90,  0, 1)
        p180 = np.clip(p180, 0, 1)
        erv_opt = np.maximum(0, self.rng.normal(2_000, 1_500, n))
        d_opt = np.full(n, 0.50)   # conservative default for D-quadrant
        return self._build_rows("D", n, p30, p90, p180, d_opt, erv_opt, low_conf=True)

    # ── Row builder ───────────────────────────────────────────────────────────

    def _build_rows(
        self,
        quadrant: str,
        n: int,
        p30: np.ndarray,
        p90: np.ndarray,
        p180: np.ndarray,
        d_opt: np.ndarray,
        erv_opt: np.ndarray,
        low_conf: bool,
    ) -> pd.DataFrame:
        seg_options = SEGMENT_LABELS_BY_QUADRANT[quadrant]
        if seg_options:
            segments = self.rng.choice(seg_options, size=n)
        else:
            segments = np.full(n, None, dtype=object)

        rows = {
            "signal_quadrant":     np.full(n, quadrant),
            "segment_label":       segments,
            "propensity_30d":      p30.round(4),
            "propensity_90d":      p90.round(4),
            "propensity_180d":     p180.round(4),
            "low_confidence_flag": np.full(n, low_conf),
            "d_optimal":           np.clip(d_opt, 0.20, 0.65).round(2),
            "erv_at_d_optimal":    erv_opt.round(2),
        }

        # Elasticity grid — interpolated from d_optimal and ERV
        for d_int in [20, 30, 40, 50, 60]:
            d_frac    = d_int / 100.0
            p_accept  = np.clip(p180 * (d_frac / 0.30) * self.rng.uniform(0.8, 1.2, n), 0, 1)
            e_amount  = np.clip(p_accept * self.rng.uniform(0.6, 1.0, n), 0, 1)
            erv       = erv_opt * (d_frac / d_opt) * self.rng.uniform(0.85, 1.15, n)
            rows[f"p_accept_d{d_int}"] = p_accept.round(4)
            rows[f"e_amount_d{d_int}"] = e_amount.round(4)
            rows[f"erv_d{d_int}"]      = np.maximum(0, erv).round(2)

        return pd.DataFrame(rows)
