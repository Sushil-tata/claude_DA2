"""
Training Data Generator
========================
Generates labeled synthetic training data for NBA model training.

Key design:
  - Each row = (account, action_taken, outcome)  ← S-Learner format
  - Actions are assigned randomly (simulating historical A/B test data)
  - Outcomes are generated with action-dependent probabilities
  - Temporal train/val/test split (no leakage)

Output schema:
  account_id, business_date, action (treatment),
  <all features>,
  outcome_pay_any (binary label),
  outcome_pay_amount (continuous label),
  split (train/val/test)
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from datetime import datetime, timedelta


# ─────────────────────────────────────────────────────────────────────────────
# TRUE EFFECT SIZES (ground truth, used to generate outcomes)
# These simulate the true uplift each action has vs NO_ACTION
# ─────────────────────────────────────────────────────────────────────────────
ACTION_EFFECTS = {
    "NO_ACTION":        {"pay_any_lift": 0.00, "amount_multiplier": 1.0},
    "SMS":              {"pay_any_lift": 0.08, "amount_multiplier": 1.1},
    "EMAIL":            {"pay_any_lift": 0.05, "amount_multiplier": 1.05},
    "LINE":             {"pay_any_lift": 0.10, "amount_multiplier": 1.12},
    "VOICE_IVR":        {"pay_any_lift": 0.12, "amount_multiplier": 1.15},
    "AGENT_CALL":       {"pay_any_lift": 0.20, "amount_multiplier": 1.30},
    "SETTLEMENT_ONE_TIME":    {"pay_any_lift": 0.30, "amount_multiplier": 0.70},
    "SETTLEMENT_PAYMENT_PLAN":{"pay_any_lift": 0.25, "amount_multiplier": 0.85},
}

ACTION_WEIGHTS = {      # historical distribution of actions taken
    "NO_ACTION":          0.30,
    "SMS":                0.25,
    "EMAIL":              0.15,
    "LINE":               0.10,
    "VOICE_IVR":          0.10,
    "AGENT_CALL":         0.08,
    "SETTLEMENT_ONE_TIME":0.01,
    "SETTLEMENT_PAYMENT_PLAN": 0.01,
}


class TrainingDataGenerator:
    """
    Generates synthetic training data in S-Learner format.
    Each account × action combination is a training row.
    """

    def __init__(
        self,
        n_accounts: int = 5000,
        n_months: int = 6,
        actions: Optional[List[str]] = None,
        seed: int = 42,
    ):
        self.n_accounts = n_accounts
        self.n_months   = n_months
        self.actions    = actions or list(ACTION_EFFECTS.keys())
        self.rng        = np.random.default_rng(seed)

    def generate(self) -> pd.DataFrame:
        """
        Generate full labeled training dataset.

        Returns:
            DataFrame with features + action + outcomes + split column
        """
        accounts = self._generate_accounts()
        history  = self._simulate_history(accounts)
        labeled  = self._add_outcomes(history)
        split    = self._temporal_split(labeled)
        return split

    def generate_scoring_batch(self, n: int = 1000) -> pd.DataFrame:
        """
        Generate accounts to score (no outcomes, just features).
        Simulates today's accounts needing NBA decisions.
        """
        accounts = self._generate_accounts(n=n, seed=999)
        accounts["business_date"] = datetime.today().strftime("%Y-%m-%d")
        return accounts

    # ──────────────────────────────────────────────────────────────────────────
    # PRIVATE
    # ──────────────────────────────────────────────────────────────────────────

    def _generate_accounts(self, n: Optional[int] = None, seed: Optional[int] = None) -> pd.DataFrame:
        """Generate base account features."""
        if n is None:
            n = self.n_accounts
        rng = np.random.default_rng(seed) if seed else self.rng

        dpd      = rng.integers(0, 360, size=n)
        balance  = rng.uniform(200, 80000, size=n).round(2)
        bucket   = np.clip(dpd // 30, 0, 5).astype(int)

        return pd.DataFrame({
            "account_id":        [f"ACC{str(i).zfill(7)}" for i in range(n)],
            "days_past_due":     dpd,
            "balance":           balance,
            "bucket":            bucket,
            "dpd_norm":          np.clip(dpd / 360.0, 0, 1).round(4),
            "log_balance":       np.log1p(balance).round(4),
            "log_dpd":           np.log1p(dpd).round(4),
            "balance_tier":      pd.cut(balance, bins=[-1,999,4999,9999,49999,1e9],
                                        labels=[0,1,2,3,4]).astype(float),
            "mobile_present":    rng.choice([1, 0], size=n, p=[0.85, 0.15]),
            "email_present":     rng.choice([1, 0], size=n, p=[0.70, 0.30]),
            "sms_optin":         rng.choice([1, 0], size=n, p=[0.75, 0.25]),
            "line_optin":        rng.choice([1, 0], size=n, p=[0.40, 0.60]),
            "dnc":               rng.choice([0, 1], size=n, p=[0.95, 0.05]),
            "fatigue_score":     rng.uniform(0, 0.8, size=n).round(4),
            "total_contacts_7d": rng.integers(0, 5, size=n),
            "days_since_last_sms":  rng.integers(0, 30, size=n),
            "days_since_last_call": rng.integers(0, 30, size=n),
            "active_settlement": rng.choice([0, 1], size=n, p=[0.95, 0.05]),
            "active_ptp":        rng.choice([0, 1], size=n, p=[0.90, 0.10]),
            "payment_count_30d": rng.integers(0, 4, size=n),
            "has_payment_30d":   rng.choice([1, 0], size=n, p=[0.45, 0.55]),
            "is_early_bucket":   (dpd.clip(1,60) == dpd).astype(int),
            "is_late_bucket":    (dpd > 90).astype(int),
            "mad_coverage_30d":  rng.uniform(0, 1.5, size=n).round(4),
        })

    def _simulate_history(self, accounts: pd.DataFrame) -> pd.DataFrame:
        """
        For each account create monthly snapshots with a random action assigned.
        This simulates historical randomised treatment data.
        """
        rows = []
        action_names = [a for a in self.actions if a in ACTION_WEIGHTS]
        action_probs = np.array([ACTION_WEIGHTS.get(a, 0.01) for a in action_names])
        action_probs /= action_probs.sum()

        base_date = datetime.today() - timedelta(days=self.n_months * 30)

        for month in range(self.n_months):
            snap_date = base_date + timedelta(days=month * 30)
            snap      = accounts.copy()
            snap["business_date"] = snap_date.strftime("%Y-%m-%d")
            snap["month_idx"]     = month

            # Assign historical action (randomly — real data would have EFS_executed_action)
            snap["action"] = self.rng.choice(action_names, size=len(snap), p=action_probs)
            rows.append(snap)

        return pd.concat(rows, ignore_index=True)

    def _add_outcomes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate outcomes based on account features + action taken.
        Outcome = pay_any (binary) + pay_amount (continuous).
        """
        df = df.copy()
        n  = len(df)

        # Base payment probability from account features (no action)
        base_prob = (
            0.40
            - 0.25 * df["dpd_norm"]
            + 0.10 * df["has_payment_30d"]
            + 0.05 * df["mad_coverage_30d"].clip(0, 1)
            - 0.05 * df["fatigue_score"]
            + 0.03 * df["is_early_bucket"]
            - 0.10 * df["is_late_bucket"]
        ).clip(0.02, 0.95)

        # Apply action effect
        action_lift = df["action"].map(
            {a: v["pay_any_lift"] for a, v in ACTION_EFFECTS.items()}
        ).fillna(0.0)
        pay_prob = (base_prob + action_lift).clip(0.02, 0.95)

        # Add noise
        pay_prob += self.rng.normal(0, 0.03, size=n)
        pay_prob  = pay_prob.clip(0.02, 0.95)

        # Binary outcome
        df["outcome_pay_any"] = (
            self.rng.uniform(0, 1, size=n) < pay_prob
        ).astype(int)

        # Amount outcome (zero-inflated: Tweedie-like)
        amount_mult = df["action"].map(
            {a: v["amount_multiplier"] for a, v in ACTION_EFFECTS.items()}
        ).fillna(1.0)

        base_amount = df["balance"] * base_prob * 0.6 * amount_mult
        noise       = self.rng.lognormal(0, 0.3, size=n)
        df["outcome_pay_amount"] = (
            df["outcome_pay_any"] * base_amount * noise
        ).clip(0).round(2)

        return df

    def _temporal_split(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Split by time: train=first 60%, val=next 20%, test=last 20%.
        Ensures no future leakage.
        """
        df = df.sort_values("business_date").copy()
        n  = len(df)
        train_end = int(n * 0.60)
        val_end   = int(n * 0.80)

        df["split"] = "train"
        df.iloc[train_end:val_end, df.columns.get_loc("split")] = "val"
        df.iloc[val_end:,         df.columns.get_loc("split")] = "test"
        return df
