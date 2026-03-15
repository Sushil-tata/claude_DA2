"""
Roll-Rate Outcome Labeller
==========================
Produces forward-looking training labels for the collections decision models.

Core concept
------------
For each account at snapshot date T, we observe what actually happened in the
next N days (the *outcome window*).  This gives us point-in-time-safe labels
that avoid target leakage.

Two input modes
---------------
  Mode A — Two snapshots (preferred on Databricks)
    snapshot_t:       T2 card DataFrame at the training snapshot date
    snapshot_t_plus:  T2 card DataFrame at T + outcome_window_days
    txn_df:           T3 transaction DataFrame between T and T+N

  Mode B — Single snapshot + synthetic (for dev / unit tests)
    snapshot_t only; outcomes are simulated from DPD/stage heuristics

Label columns produced
----------------------
For each outcome window W in [30, 60, 90]:
  outcome_pay_any_{W}d        binary  — any payment made in (T, T+W]
  outcome_pay_amount_{W}d     float   — total payment amount in (T, T+W]

Always:
  roll_bucket_t               int     — DPD bucket at snapshot T
  roll_bucket_horizon         int     — DPD bucket at T + primary_window_days
  roll_direction              str     — CURE / IMPROVE / STABLE / WORSEN / CHARGEOFF
  roll_buckets                int     — signed bucket delta (negative = improved)
  outcome_cured               binary  — DPD returned to 0 within primary window
  outcome_chargeoff           binary  — account charged off within primary window

DPD bucket mapping
------------------
  Bucket 0: DPD   0-29   (current)
  Bucket 1: DPD  30-59
  Bucket 2: DPD  60-89
  Bucket 3: DPD  90-119
  Bucket 4: DPD 120-179
  Bucket 5: DPD 180+
  Bucket 6: CHARGEOFF / WRITEOFF

Usage — Databricks (Mode A)
---------------------------
    from decision_agent.labels.roll_rate_labeller import RollRateLabeller, RollRateConfig

    cfg = RollRateConfig(outcome_windows=[30, 60, 90], primary_window_days=90)
    labeller = RollRateLabeller(cfg)

    labels = labeller.compute_labels(
        snapshot_t=card_df_jan,           # T2 at 2026-01-31
        snapshot_t_plus=card_df_apr,      # T2 at 2026-04-30  (+90d)
        txn_df=txn_df_feb_to_apr,         # T3 transactions between the two dates
    )
    # labels: one row per account, all outcome columns

Usage — Development / synthetic (Mode B)
-----------------------------------------
    labels = labeller.synthesize_labels(features_df)

Usage — Build full training dataset
-------------------------------------
    # Pairs of (snapshot_date, horizon_date) drive historical training
    pairs = [
        ("2025-06-30", "2025-09-30"),
        ("2025-07-31", "2025-10-31"),
        ("2025-08-31", "2025-11-30"),
    ]
    train_df = labeller.build_training_dataset(snapshot_pairs=pairs, features_pipeline=pipeline)
"""

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# DPD BUCKET MAPPING
# ─────────────────────────────────────────────────────────────────────────────

DPD_BUCKETS = [
    (0,    29,  0),   # Current
    (30,   59,  1),   # 1 month past due
    (60,   89,  2),   # 2 months past due
    (90,  119,  3),   # 3 months past due
    (120, 179,  4),   # 4-5 months past due
    (180, 9999, 5),   # 6+ months past due
]
CHARGEOFF_BUCKET = 6

BUCKET_LABELS = {
    0: "CURRENT",
    1: "DPD_30",
    2: "DPD_60",
    3: "DPD_90",
    4: "DPD_120",
    5: "DPD_180",
    6: "CHARGEOFF",
}

CHARGEOFF_STAGES = {"CHARGEOFF", "WRITEOFF", "CO", "WO"}


def dpd_to_bucket(dpd: float, stage: str = "") -> int:
    """Map DPD integer + stage string to bucket 0-6."""
    if pd.isna(dpd):
        return 0
    if str(stage).upper() in CHARGEOFF_STAGES:
        return CHARGEOFF_BUCKET
    dpd = int(dpd)
    for lo, hi, bucket in DPD_BUCKETS:
        if lo <= dpd <= hi:
            return bucket
    return 5  # DPD 180+ without explicit chargeoff


def bucket_to_direction(delta: int) -> str:
    """
    Convert signed bucket delta to roll direction label.
    delta = bucket_horizon - bucket_t
    """
    if delta <= -2:
        return "CURE"           # improved by 2+ buckets
    elif delta == -1:
        return "IMPROVE"        # improved by 1 bucket
    elif delta == 0:
        return "STABLE"         # no change
    elif delta == 1:
        return "WORSEN"         # worsened by 1 bucket
    else:
        return "CHARGEOFF"      # worsened by 2+ buckets (likely hit chargeoff)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RollRateConfig:
    """
    Controls outcome window sizes and thresholds.
    All monetary thresholds in Thai Baht.
    """

    # Outcome windows to label (days after snapshot)
    outcome_windows: List[int] = field(default_factory=lambda: [30, 60, 90])

    # Primary window used for roll_direction / roll_buckets
    primary_window_days: int = 90

    # Minimum payment amount to count as "paid" (THB)
    min_payment_to_count: float = 1.0

    # DPD threshold for "cured" (must reach this DPD or below)
    cure_dpd_threshold: int = 0

    # Column names in T2 snapshot DataFrames (post schema_mapper rename)
    account_id_col: str    = "account_id"
    dpd_col: str           = "days_past_due"
    stage_col: str         = "stage"
    balance_col: str       = "balance"
    data_date_col: str     = "data_date"

    # Column names in T3 transaction DataFrames (post schema_mapper rename)
    txn_account_id_col: str   = "account_id"
    txn_date_col: str         = "transaction_date"
    txn_amount_col: str       = "transaction_amount"
    txn_type_col: str         = "transaction_type"

    # Transaction type values that count as a payment
    payment_types: List[str] = field(default_factory=lambda: [
        "PAYMENT", "PAY", "PYMT", "CREDIT", "REPAYMENT",
        "SETTLEMENT", "PTP_PAYMENT", "PARTIAL_PAYMENT",
    ])

    # Whether to include intermediate windows (30d, 60d) in output
    # Set False to only produce primary_window labels (lighter output)
    include_intermediate_windows: bool = True


# ─────────────────────────────────────────────────────────────────────────────
# LABELLER
# ─────────────────────────────────────────────────────────────────────────────

class RollRateLabeller:
    """
    Builds roll-rate outcome labels from CDX T2 (card) + T3 (transaction) data.

    Point-in-time guarantee: all label computation uses data AFTER the
    snapshot date only.  The snapshot_t DataFrame must never contain
    forward-looking columns.
    """

    def __init__(self, config: Optional[RollRateConfig] = None):
        self.cfg = config or RollRateConfig()

    # ── Mode A: Two snapshots (Databricks production path) ────────────────────

    def compute_labels(
        self,
        snapshot_t:      pd.DataFrame,
        snapshot_t_plus: pd.DataFrame,
        txn_df:          Optional[pd.DataFrame] = None,
        snapshot_date_t: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Compute outcome labels from two T2 snapshots + optional T3 transactions.

        Args:
            snapshot_t:       T2 card data at training snapshot date T
            snapshot_t_plus:  T2 card data at T + primary_window_days
            txn_df:           T3 transactions between T and T+primary_window_days.
                              If None, payment labels are derived from arrears reduction.
            snapshot_date_t:  Override snapshot date (reads data_date col if None)

        Returns:
            DataFrame: one row per account, all label columns.
        """
        c = self.cfg

        # ── 1. DPD buckets at T and T+N ──────────────────────────────────────
        labels = self._compute_roll_buckets(snapshot_t, snapshot_t_plus)

        # ── 2. Payment labels ─────────────────────────────────────────────────
        if txn_df is not None and len(txn_df) > 0:
            payment_labels = self._compute_payment_labels_from_txn(txn_df, snapshot_t)
        else:
            # Fallback: infer payments from arrears reduction between snapshots
            logger.warning(
                "RollRateLabeller: txn_df not provided — "
                "inferring payment labels from arrears reduction. "
                "Provide T3 transactions for accurate payment labels."
            )
            payment_labels = self._infer_payments_from_arrears(snapshot_t, snapshot_t_plus)

        labels = labels.merge(payment_labels, on=c.account_id_col, how="left")

        # ── 3. Summary flags ──────────────────────────────────────────────────
        labels["outcome_cured"] = (
            (labels["roll_bucket_horizon"] == 0) & (labels["roll_bucket_t"] > 0)
        ).astype(int)

        labels["outcome_chargeoff"] = (
            labels["roll_bucket_horizon"] == CHARGEOFF_BUCKET
        ).astype(int)

        logger.info(
            "RollRateLabeller.compute_labels: %d accounts labelled | "
            "cure_rate=%.1f%% | chargeoff_rate=%.1f%% | pay_any_rate=%.1f%%",
            len(labels),
            labels["outcome_cured"].mean() * 100,
            labels["outcome_chargeoff"].mean() * 100,
            labels[f"outcome_pay_any_{c.primary_window_days}d"].mean() * 100,
        )

        return labels

    # ── Mode B: Synthetic labels (dev / unit tests) ────────────────────────────

    def synthesize_labels(
        self,
        features_df: pd.DataFrame,
        random_seed: int = 42,
    ) -> pd.DataFrame:
        """
        Generate realistic synthetic outcome labels from pipeline features.

        Uses DPD, stage, arrears, and contact features as signals to simulate
        plausible outcomes.  NOT for production — for development and testing only.

        Args:
            features_df: Output DataFrame from CollectionsFeaturePipeline.run()
            random_seed: Reproducibility seed

        Returns:
            DataFrame: [account_id, all label columns]
        """
        rng = np.random.default_rng(random_seed)
        c   = self.cfg
        df  = features_df.copy()
        n   = len(df)

        # ── Base pay probability from DPD and stage ───────────────────────────
        dpd   = df.get(c.dpd_col,   pd.Series(np.zeros(n))).fillna(0).values
        stage = df.get(c.stage_col, pd.Series(["SM"] * n)).fillna("SM").values

        # Logistic model: higher DPD → lower pay probability
        base_pay_prob = 0.80 * np.exp(-dpd / 60.0)

        # Stage adjustment
        stage_adj = np.where(
            np.isin(stage, list(CHARGEOFF_STAGES)), -0.30,
            np.where(dpd > 90, -0.15, 0.0)
        )

        # Contact boost (higher response rate → higher pay probability)
        response_rate = df.get("response_rate_90d", pd.Series(np.full(n, 0.3))).fillna(0.3).values
        contact_boost = 0.15 * response_rate

        pay_prob = np.clip(base_pay_prob + stage_adj + contact_boost, 0.01, 0.99)

        # ── Roll bucket at T ──────────────────────────────────────────────────
        bucket_t = np.array([
            dpd_to_bucket(d, s) for d, s in zip(dpd, stage)
        ])

        # ── Simulate roll at horizon ──────────────────────────────────────────
        #    Paying accounts tend to improve; non-paying tend to worsen
        paid_any = (rng.uniform(size=n) < pay_prob).astype(int)

        # Roll delta: paying accounts -1 to 0 buckets; non-paying +1 to +2 buckets
        roll_delta = np.where(
            paid_any == 1,
            rng.integers(-2, 1, size=n),   # -2, -1, or 0
            rng.integers(0, 3, size=n),    # 0, 1, or 2
        )

        bucket_horizon = np.clip(bucket_t + roll_delta, 0, CHARGEOFF_BUCKET)

        # Chargeoff accounts stay at chargeoff
        bucket_horizon = np.where(bucket_t == CHARGEOFF_BUCKET, CHARGEOFF_BUCKET, bucket_horizon)

        # ── Build labels DataFrame ────────────────────────────────────────────
        labels = pd.DataFrame({c.account_id_col: df[c.account_id_col].values})

        labels["roll_bucket_t"]       = bucket_t
        labels["roll_bucket_horizon"] = bucket_horizon
        labels["roll_buckets"]        = bucket_horizon - bucket_t
        labels["roll_direction"]      = [bucket_to_direction(int(d)) for d in labels["roll_buckets"]]
        labels["outcome_cured"]       = ((bucket_horizon == 0) & (bucket_t > 0)).astype(int)
        labels["outcome_chargeoff"]   = (bucket_horizon == CHARGEOFF_BUCKET).astype(int)

        # Payment labels for each window
        for w in c.outcome_windows:
            # Shorter windows → lower pay probability
            w_pay_prob = pay_prob * (w / c.primary_window_days)
            paid_w = (rng.uniform(size=n) < w_pay_prob).astype(int)

            # Amount: fraction of balance if paid, else 0
            balance = df.get(c.balance_col, pd.Series(np.full(n, 10000.0))).fillna(10000).values
            pay_fraction = rng.uniform(0.05, 0.30, size=n)
            pay_amount   = np.round(balance * pay_fraction * paid_w, 2)

            labels[f"outcome_pay_any_{w}d"]    = paid_w
            labels[f"outcome_pay_amount_{w}d"] = pay_amount

        logger.info(
            "RollRateLabeller.synthesize_labels: %d accounts | "
            "primary(%dd) pay_rate=%.1f%% | cure_rate=%.1f%%",
            n,
            c.primary_window_days,
            labels[f"outcome_pay_any_{c.primary_window_days}d"].mean() * 100,
            labels["outcome_cured"].mean() * 100,
        )

        return labels

    # ── Training dataset builder ───────────────────────────────────────────────

    def build_training_dataset(
        self,
        snapshot_pairs:    List[Tuple[str, str]],
        features_pipeline,
        card_loader,
        txn_loader         = None,
        customer_df:       Optional[pd.DataFrame] = None,
        actions_loader     = None,
        settlements_loader = None,
    ) -> pd.DataFrame:
        """
        Build a full historical training dataset by iterating over snapshot pairs.

        For each (snapshot_date, horizon_date) pair:
          1. Load T2 card data at snapshot_date → run feature pipeline → get features
          2. Load T2 card data at horizon_date + T3 transactions → compute labels
          3. Join features + labels → append to training set

        Args:
            snapshot_pairs:    List of (snapshot_date_str, horizon_date_str) tuples.
                               e.g. [("2025-06-30", "2025-09-30"), ...]
            features_pipeline: CollectionsFeaturePipeline instance
            card_loader:       Callable(date_str) → T2 card DataFrame
            txn_loader:        Callable(start_date, end_date) → T3 txn DataFrame (optional)
            customer_df:       T1 demographics DataFrame (reused across snapshots)
            actions_loader:    Callable(start_date, end_date) → T4 actions DataFrame (optional)
            settlements_loader:Callable(start_date, end_date) → T5 settlements DataFrame (optional)

        Returns:
            DataFrame: features + labels + snapshot_date, ready for model.fit()
        """
        all_records = []

        for snap_date_str, horizon_date_str in snapshot_pairs:
            logger.info(
                "Building training data: snapshot=%s, horizon=%s",
                snap_date_str, horizon_date_str,
            )

            snap_date    = pd.to_datetime(snap_date_str).date()
            horizon_date = pd.to_datetime(horizon_date_str).date()

            # ── Load data ─────────────────────────────────────────────────────
            card_t      = card_loader(snap_date_str)
            card_t_plus = card_loader(horizon_date_str)

            txn_df          = txn_loader(snap_date_str, horizon_date_str) if txn_loader else None
            actions_df      = actions_loader(snap_date_str, snap_date_str) if actions_loader else None
            settlements_df  = settlements_loader(snap_date_str, snap_date_str) if settlements_loader else None

            # ── Features at snapshot_t ────────────────────────────────────────
            feat_result = features_pipeline.run(
                card_df=card_t,
                customer_df=customer_df,
                actions_df=actions_df,
                settlements_df=settlements_df,
                snapshot_date=snap_date,
            )

            # ── Labels between snapshot_t and horizon ─────────────────────────
            labels = self.compute_labels(
                snapshot_t=card_t,
                snapshot_t_plus=card_t_plus,
                txn_df=txn_df,
                snapshot_date_t=snap_date_str,
            )

            # ── Join ──────────────────────────────────────────────────────────
            merged = feat_result.features_df.merge(
                labels,
                on=self.cfg.account_id_col,
                how="inner",
            )
            merged["snapshot_date"] = snap_date_str

            dropped = len(feat_result.features_df) - len(merged)
            if dropped:
                logger.warning(
                    "Snapshot %s: %d accounts dropped on label join",
                    snap_date_str, dropped,
                )

            all_records.append(merged)

        if not all_records:
            raise ValueError("No snapshot pairs produced any labelled records.")

        training_df = pd.concat(all_records, ignore_index=True)
        logger.info(
            "build_training_dataset: %d total records from %d snapshot pairs",
            len(training_df), len(snapshot_pairs),
        )

        return training_df

    # ── PRIVATE HELPERS ───────────────────────────────────────────────────────

    def _compute_roll_buckets(
        self,
        snap_t:      pd.DataFrame,
        snap_t_plus: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute DPD buckets at T and T+N; derive roll_direction and roll_buckets."""
        c = self.cfg

        t = snap_t[[c.account_id_col, c.dpd_col, c.stage_col]].copy()
        t["roll_bucket_t"] = [
            dpd_to_bucket(d, s)
            for d, s in zip(t[c.dpd_col].fillna(0), t[c.stage_col].fillna(""))
        ]

        h = snap_t_plus[[c.account_id_col, c.dpd_col, c.stage_col]].copy()
        h.columns = [c.account_id_col, f"{c.dpd_col}_h", f"{c.stage_col}_h"]
        h["roll_bucket_horizon"] = [
            dpd_to_bucket(d, s)
            for d, s in zip(h[f"{c.dpd_col}_h"].fillna(0), h[f"{c.stage_col}_h"].fillna(""))
        ]

        merged = t[[c.account_id_col, "roll_bucket_t"]].merge(
            h[[c.account_id_col, "roll_bucket_horizon"]],
            on=c.account_id_col,
            how="left",
        )

        # Accounts that disappeared from T+N snapshot are assumed charged off
        merged["roll_bucket_horizon"] = merged["roll_bucket_horizon"].fillna(CHARGEOFF_BUCKET).astype(int)

        merged["roll_buckets"]   = merged["roll_bucket_horizon"] - merged["roll_bucket_t"]
        merged["roll_direction"] = merged["roll_buckets"].apply(lambda d: bucket_to_direction(int(d)))

        return merged

    def _compute_payment_labels_from_txn(
        self,
        txn_df:     pd.DataFrame,
        snapshot_t: pd.DataFrame,
    ) -> pd.DataFrame:
        """Aggregate T3 transactions to per-account payment labels for each window."""
        c = self.cfg

        txn = txn_df.copy()
        txn[c.txn_date_col]   = pd.to_datetime(txn[c.txn_date_col], errors="coerce")
        txn[c.txn_amount_col] = pd.to_numeric(txn[c.txn_amount_col], errors="coerce").fillna(0)

        # Determine snapshot date per account from T2
        if c.data_date_col in snapshot_t.columns:
            snap_dates = snapshot_t[[c.account_id_col, c.data_date_col]].copy()
            snap_dates[c.data_date_col] = pd.to_datetime(snap_dates[c.data_date_col])
            txn = txn.merge(snap_dates, on=c.account_id_col, how="left")
        else:
            txn[c.data_date_col] = pd.NaT

        # Filter to payment types only
        if c.txn_type_col in txn.columns:
            payment_mask = txn[c.txn_type_col].str.upper().isin(
                [p.upper() for p in c.payment_types]
            )
            txn = txn[payment_mask]
        else:
            # No type column — treat all positive amounts as payments
            txn = txn[txn[c.txn_amount_col] > 0]

        accounts = snapshot_t[[c.account_id_col]].copy()
        results  = accounts.copy()

        for w in c.outcome_windows:
            if not c.include_intermediate_windows and w != c.primary_window_days:
                continue

            # Filter to transactions within window
            if c.data_date_col in txn.columns and txn[c.data_date_col].notna().any():
                cutoff = txn[c.data_date_col] + pd.Timedelta(days=w)
                window_txn = txn[txn[c.txn_date_col] <= cutoff].copy()
            else:
                window_txn = txn.copy()  # no date filtering possible

            # Aggregate
            agg = (
                window_txn.groupby(c.account_id_col)[c.txn_amount_col]
                .agg(total_paid="sum", payment_count="count")
                .reset_index()
            )

            results = results.merge(agg, on=c.account_id_col, how="left")
            results["total_paid"]    = results["total_paid"].fillna(0)
            results["payment_count"] = results["payment_count"].fillna(0)

            results[f"outcome_pay_any_{w}d"]    = (
                results["total_paid"] >= c.min_payment_to_count
            ).astype(int)
            results[f"outcome_pay_amount_{w}d"] = results["total_paid"].round(2)

            results = results.drop(columns=["total_paid", "payment_count"])

        return results

    def _infer_payments_from_arrears(
        self,
        snap_t:      pd.DataFrame,
        snap_t_plus: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Fallback: infer payment from arrears reduction between snapshots.
        Arrears decrease → payment made.  Less accurate than T3 transactions.
        """
        c = self.cfg

        arrs_col = "arrs_period_1"  # most recent arrears period

        t = snap_t[[c.account_id_col, c.balance_col]].copy()
        t[arrs_col] = snap_t.get(arrs_col, pd.Series(0, index=snap_t.index)).fillna(0)

        h = snap_t_plus[[c.account_id_col, c.balance_col]].copy()
        h.columns = [c.account_id_col, f"{c.balance_col}_h"]
        h[f"{arrs_col}_h"] = snap_t_plus.get(
            arrs_col, pd.Series(0, index=snap_t_plus.index)
        ).fillna(0)

        merged = t.merge(h, on=c.account_id_col, how="left")
        merged = merged.fillna(0)

        # Estimate payment as arrears reduction (proxy only)
        arrears_reduction = (merged[arrs_col] - merged[f"{arrs_col}_h"]).clip(lower=0)
        balance_reduction = (merged[c.balance_col] - merged[f"{c.balance_col}_h"]).clip(lower=0)
        inferred_payment  = arrears_reduction.combine(balance_reduction, max)

        results = merged[[c.account_id_col]].copy()

        for w in c.outcome_windows:
            if not c.include_intermediate_windows and w != c.primary_window_days:
                continue
            # Scale by window fraction (rough approximation)
            w_payment = (inferred_payment * (w / self.cfg.primary_window_days)).round(2)
            results[f"outcome_pay_any_{w}d"]    = (w_payment >= c.min_payment_to_count).astype(int)
            results[f"outcome_pay_amount_{w}d"] = w_payment

        return results


# ─────────────────────────────────────────────────────────────────────────────
# ROLL-RATE MATRIX (analytics helper)
# ─────────────────────────────────────────────────────────────────────────────

def compute_roll_rate_matrix(labels_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the roll-rate transition matrix from labelled data.

    Returns a DataFrame where rows = bucket at T, columns = bucket at T+N,
    values = percentage of accounts transitioning (row-normalised).

    Useful for validating label quality and tuning collection strategies.
    """
    ct = pd.crosstab(
        labels_df["roll_bucket_t"].map(BUCKET_LABELS),
        labels_df["roll_bucket_horizon"].map(BUCKET_LABELS),
        margins=False,
    )
    return ct.div(ct.sum(axis=1), axis=0).round(4) * 100


def label_summary(labels_df: pd.DataFrame, config: Optional[RollRateConfig] = None) -> str:
    """
    Print-friendly summary of label distribution.
    """
    cfg = config or RollRateConfig()
    w   = cfg.primary_window_days

    lines = [
        "=" * 55,
        f"  Roll-Rate Label Summary  (primary window: {w}d)",
        "=" * 55,
        f"  Accounts labelled:    {len(labels_df):,}",
        "",
        "  Roll direction distribution:",
    ]

    if "roll_direction" in labels_df.columns:
        rd = labels_df["roll_direction"].value_counts()
        for direction, cnt in rd.items():
            pct = cnt / len(labels_df) * 100
            lines.append(f"    {direction:<12} {cnt:>6,}  ({pct:.1f}%)")

    lines.append("")
    lines.append("  Payment outcomes:")

    for col in [f"outcome_pay_any_{w}d" for w in cfg.outcome_windows]:
        if col in labels_df.columns:
            rate = labels_df[col].mean() * 100
            lines.append(f"    {col:<30} {rate:.1f}%")

    lines.append("")

    if "outcome_cured" in labels_df.columns:
        lines.append(f"  Cure rate:            {labels_df['outcome_cured'].mean()*100:.1f}%")
    if "outcome_chargeoff" in labels_df.columns:
        lines.append(f"  Chargeoff rate:       {labels_df['outcome_chargeoff'].mean()*100:.1f}%")

    lines.append("=" * 55)
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# SMOKE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from decision_agent.features.collections_feature_pipeline import (
        CollectionsFeaturePipeline, PipelineConfig,
    )
    import pandas as pd

    # ── Synthetic T2 at snapshot T ────────────────────────────────────────────
    card_t = pd.DataFrame([
        {"account_id": "ACC001", "days_past_due": 60,  "stage": "NPL",      "balance": 45000, "data_date": "2026-01-31", "arrs_period_1": 5000},
        {"account_id": "ACC002", "days_past_due": 180, "stage": "CHARGEOFF", "balance": 120000,"data_date": "2026-01-31", "arrs_period_1": 50000},
        {"account_id": "ACC003", "days_past_due": 35,  "stage": "SM",        "balance": 8000,  "data_date": "2026-01-31", "arrs_period_1": 8000},
        {"account_id": "ACC004", "days_past_due": 0,   "stage": "CURRENT",   "balance": 25000, "data_date": "2026-01-31", "arrs_period_1": 0},
    ])

    # ── Synthetic T2 at T + 90d ───────────────────────────────────────────────
    card_t_plus = pd.DataFrame([
        {"account_id": "ACC001", "days_past_due": 0,   "stage": "CURRENT",   "balance": 40000, "data_date": "2026-04-30", "arrs_period_1": 0},      # cured
        {"account_id": "ACC002", "days_past_due": 180, "stage": "CHARGEOFF", "balance": 122000,"data_date": "2026-04-30", "arrs_period_1": 52000},   # stable chargeoff
        {"account_id": "ACC003", "days_past_due": 90,  "stage": "NPL",       "balance": 9000,  "data_date": "2026-04-30", "arrs_period_1": 9000},    # worsened
        {"account_id": "ACC004", "days_past_due": 30,  "stage": "SM",        "balance": 25500, "data_date": "2026-04-30", "arrs_period_1": 2000},    # worsened slightly
    ])

    # ── Synthetic T3 transactions ─────────────────────────────────────────────
    txn_df = pd.DataFrame([
        {"account_id": "ACC001", "transaction_date": "2026-02-15", "transaction_amount": 5000, "transaction_type": "PAYMENT"},
        {"account_id": "ACC001", "transaction_date": "2026-03-10", "transaction_amount": 5000, "transaction_type": "PAYMENT"},
        {"account_id": "ACC001", "transaction_date": "2026-04-05", "transaction_amount": 35000,"transaction_type": "PAYMENT"},
        {"account_id": "ACC003", "transaction_date": "2026-02-20", "transaction_amount": 500,  "transaction_type": "PAYMENT"},
    ])

    print("\n" + "="*55)
    print("  Mode A: Two-snapshot labels")
    print("="*55)
    labeller = RollRateLabeller(RollRateConfig(outcome_windows=[30, 60, 90]))
    labels_a = labeller.compute_labels(card_t, card_t_plus, txn_df)
    print(labels_a.to_string())

    print("\n" + label_summary(labels_a))

    print("\n" + "="*55)
    print("  Roll-Rate Transition Matrix")
    print("="*55)
    matrix = compute_roll_rate_matrix(labels_a)
    print(matrix.to_string())

    print("\n" + "="*55)
    print("  Mode B: Synthetic labels from pipeline features")
    print("="*55)

    from decision_agent.features.collections_feature_pipeline import CollectionsFeaturePipeline, PipelineConfig
    import warnings; warnings.filterwarnings("ignore")

    card_full = pd.DataFrame([
        {"account_id":"ACC001","balance":45000,"principal_balance":42000,"days_past_due":60,"stage":"NPL","credit_limit":50000,"available_credit":0,"over_limit":0,"bill_day":15,"interest_rate":0.18,"data_date":"2026-01-31","delinquency_history":"0001123456","delinquency_number":4,"arrs_period_1":5000,"arrs_period_2":4500,"arrs_period_3":4000,"arrs_period_4":2000,"arrs_period_5":0,"arrs_period_6":0,"arrs_period_7":0,"arrs_period_8":0,"arrs_period_9":0,"last_payment_amount":3000,"last_payment_date":"2025-12-01","tdr_flag":0,"tdr_instalment_amount":None,"tdr_instalment_count":None,"pdue_30":0,"pdue_60":15000,"pdue_90":0,"m_token":"TOK001"},
        {"account_id":"ACC002","balance":120000,"principal_balance":115000,"days_past_due":180,"stage":"CHARGEOFF","credit_limit":150000,"available_credit":0,"over_limit":5000,"bill_day":28,"interest_rate":0.20,"data_date":"2026-01-31","delinquency_history":"01234567X8","delinquency_number":8,"arrs_period_1":50000,"arrs_period_2":48000,"arrs_period_3":44000,"arrs_period_4":40000,"arrs_period_5":35000,"arrs_period_6":30000,"arrs_period_7":40000,"arrs_period_8":15000,"arrs_period_9":0,"last_payment_amount":2000,"last_payment_date":"2025-06-15","tdr_flag":0,"tdr_instalment_amount":None,"tdr_instalment_count":None,"pdue_30":0,"pdue_60":0,"pdue_90":0,"pdue_120":0,"pdue_180":120000,"m_token":"TOK002"},
        {"account_id":"ACC003","balance":8000,"principal_balance":7500,"days_past_due":35,"stage":"SM","credit_limit":50000,"available_credit":30000,"over_limit":0,"bill_day":5,"interest_rate":0.15,"data_date":"2026-01-31","delinquency_history":"0000000001","delinquency_number":1,"arrs_period_1":8000,"arrs_period_2":0,"arrs_period_3":0,"arrs_period_4":0,"arrs_period_5":0,"arrs_period_6":0,"arrs_period_7":0,"arrs_period_8":0,"arrs_period_9":0,"last_payment_amount":5000,"last_payment_date":"2026-01-20","tdr_flag":0,"tdr_instalment_amount":None,"tdr_instalment_count":None,"pdue_30":8000,"pdue_60":0,"pdue_90":0,"m_token":"TOK003"},
    ])

    pipeline = CollectionsFeaturePipeline(PipelineConfig())
    result   = pipeline.run(card_full, None, None, None)
    labels_b = labeller.synthesize_labels(result.features_df)
    print(labels_b[["account_id","roll_bucket_t","roll_bucket_horizon","roll_direction",
                     "outcome_cured","outcome_pay_any_90d","outcome_pay_amount_90d"]].to_string())

    print(f"\n✓ smoke test passed")
