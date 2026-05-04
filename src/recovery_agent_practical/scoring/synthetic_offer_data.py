"""
Synthetic Offer Data Generator
================================
Generates synthetic CO-stage settlement/installment offer data matching
the statistical profile of the CardX negotiation dataset (24,319 accounts,
observation window 2023-04-01 to 2026-04-30).

Statistical targets (from data summary):
  effective_acceptance mean : 0.935
  final_discount_pct mean   : 39.07,  range 0-100
  final_balance mean        : 70,307, range 596-1,610,983
  months_in_collection mean : 16.1,   range 1-36.87
  total_offers mean         : 1.06,   range 1-5
  is_settlement             : 86.5%
  offer_velocity mean       : 0.143,  range 0.03-4

Elasticity relationships embedded:
  P(accept) increases with discount_pct (diminishing returns)
  P(accept) decreases with months_in_collection beyond 24m
  Settlement has higher base P(accept) than installment
  Large balances (>100K) have higher P(accept) at any discount level
  Prior accepts strongly predict current acceptance
"""

import numpy as np
import pandas as pd
from typing import Optional


def generate_synthetic_offers(
    n: int = 5000,
    seed: int = 42,
    settlement_share: float = 0.865,
) -> pd.DataFrame:
    """
    Generate synthetic CO-stage offer dataset.

    Args:
        n               : Number of rows to generate
        seed            : Random seed for reproducibility
        settlement_share: Fraction of offers that are Settlement (vs Installment)

    Returns:
        DataFrame matching CardX offer data statistical profile
    """
    rng = np.random.default_rng(seed)

    # ── 1. ACCOUNT-LEVEL FEATURES ─────────────────────────────────────────────

    # Balance: log-normal, mean=70K, wide tail to 1.6M
    log_balance = rng.normal(loc=np.log(35_000), scale=1.1, size=n)
    final_balance = np.clip(np.exp(log_balance), 596, 1_610_983).round(2)

    balance_bucket = pd.cut(
        final_balance,
        bins=[0, 10_000, 50_000, 100_000, 500_000, np.inf],
        labels=["small_0_10k", "medium_10k_50k", "large_50k_100k",
                "very_large_100k_500k", "massive_500k_plus"]
    ).astype(str)

    # Months in collection: uniform-ish, mean ~16, range 1-37
    months_in_collection = np.clip(
        rng.gamma(shape=2.5, scale=6.5, size=n), 1, 36.87
    ).round(8)

    # ── 2. OFFER-LEVEL FEATURES ───────────────────────────────────────────────

    # Campaign type
    is_settlement = (rng.random(n) < settlement_share).astype(int)
    is_installment = 1 - is_settlement
    final_campaign = np.where(is_settlement, "Settlement", "Installment")
    last_campaign = final_campaign.copy()

    # Discount pct: mean ~39%, full range 0-100, roughly normal with wide spread
    # Settlement offers cluster lower (mean ~35%), installment higher (mean ~50%)
    base_discount = np.where(
        is_settlement,
        rng.normal(loc=35, scale=18, size=n),
        rng.normal(loc=50, scale=20, size=n)
    )
    final_discount_pct = np.clip(base_discount, 0, 100).round(2)

    # Discount tier and band
    discount_tier = pd.cut(
        final_discount_pct,
        bins=[0, 20, 40, 60, 80, 100],
        labels=["low_0_20", "medium_20_40", "high_40_60",
                "very_high_60_80", "extreme_80_100"],
        include_lowest=True
    ).astype(str)

    discount_band_labels = ["0-10%", "10-20%", "20-30%", "30-40%", "40-50%",
                            "50-60%", "60-70%", "70-80%", "80-90%", "90-100%"]
    final_offer_discount_band = pd.cut(
        final_discount_pct,
        bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        labels=discount_band_labels,
        include_lowest=True
    ).astype(str)
    last_discount_band = final_offer_discount_band.copy()

    # Offer amount = balance × (1 - discount/100)
    final_offer_amount = np.clip(
        (final_balance * (1 - final_discount_pct / 100)).round(0), 1, 1_490_000
    )

    # ── 3. OFFER HISTORY ──────────────────────────────────────────────────────

    # Total offers: 1-5, heavily skewed to 1
    total_offers = rng.choice([1, 2, 3, 4, 5], size=n,
                               p=[0.92, 0.05, 0.02, 0.005, 0.005])

    settlement_offers = np.where(
        is_settlement,
        np.minimum(total_offers, rng.integers(1, 5, size=n)),
        rng.integers(0, 2, size=n)
    )
    installment_offers = total_offers - settlement_offers
    installment_offers = np.clip(installment_offers, 0, total_offers)

    # Offer velocity: offers per month
    offer_velocity = np.clip(
        total_offers / np.maximum(months_in_collection, 1),
        0.03, 4
    ).round(8)

    # Min/max/avg discount from offer history (slight variation around final)
    noise = rng.normal(0, 2, size=n)
    min_discount_pct = np.clip(final_discount_pct - abs(noise) - 1, 0, 100).round(2)
    max_discount_pct = np.clip(final_discount_pct + abs(noise) + 1, 0, 100).round(2)
    avg_discount_pct = ((min_discount_pct + max_discount_pct) / 2).round(6)

    # ── 4. ELASTICITY-DRIVEN ACCEPTANCE ───────────────────────────────────────
    #
    # P(accept) is a function of:
    #   + discount_pct          (primary lever — S-curve)
    #   - months_in_collection  (fatigue beyond 24m — mild effect at CO stage)
    #   + large balance         (higher stakes = higher motivation)
    #   - installment type      (ongoing commitment = lower base rate)
    #   + prior accepts         (strongest predictor)

    # Logistic elasticity curve
    # Base log-odds varies by campaign type
    base_logit = np.where(is_settlement, 1.5, 0.8)

    # Discount effect: S-curve, strongest between 20-60%
    discount_effect = 0.04 * (final_discount_pct - 30)

    # Vintage effect: mild penalty beyond 24 months at CO stage
    vintage_effect = -0.02 * np.maximum(months_in_collection - 24, 0)

    # Balance effect: large balances more motivated to settle
    balance_effect = 0.3 * (np.log(final_balance) - np.log(50_000)) / np.log(10)

    log_odds = base_logit + discount_effect + vintage_effect + balance_effect
    p_accept = 1 / (1 + np.exp(-log_odds))

    # Draw acceptance with some noise
    effective_acceptance = (rng.random(n) < p_accept).astype(int)

    # Clip to match observed 93.5% acceptance rate approximately
    # Force acceptance=1 for top 93.5% by p_accept (selection bias proxy)
    threshold = np.percentile(p_accept, 100 - 93.5)
    effective_acceptance = (p_accept >= threshold).astype(int)
    # Add small random noise to avoid perfect determinism
    flip_mask = (rng.random(n) < 0.02)
    effective_acceptance = np.where(flip_mask, 1 - effective_acceptance, effective_acceptance)

    # ── 5. PRIOR ACCEPT FEATURES ──────────────────────────────────────────────

    has_prior_accepts = effective_acceptance.copy()
    n_prior_accepts_real = np.where(
        effective_acceptance == 1,
        rng.choice([1, 2, 3, 4], size=n, p=[0.90, 0.07, 0.02, 0.01]),
        0
    )

    avg_accepted_settlement_pct = np.where(
        (effective_acceptance == 1) & (is_settlement == 1),
        final_discount_pct + rng.normal(0, 1, size=n),
        np.nan
    )
    avg_accepted_installment_pct = np.where(
        (effective_acceptance == 1) & (is_installment == 1),
        final_discount_pct + rng.normal(0, 1, size=n),
        np.nan
    )
    avg_accepted_settlement_pct = np.clip(avg_accepted_settlement_pct, 0, 100)
    avg_accepted_installment_pct = np.clip(avg_accepted_installment_pct, 0, 100)

    # ── 6. CAMPAIGN MIX ───────────────────────────────────────────────────────

    campaign_mix = np.where(
        (settlement_offers > 0) & (installment_offers > 0), "Both",
        np.where(settlement_offers > 0, "Settlement Only", "Installment Only")
    )

    # ── 7. IDENTIFIERS & DATES ────────────────────────────────────────────────

    cust_ids = [f"C{str(i).zfill(6)}" for i in range(n)]
    account_nos = [f"A{str(i).zfill(6)}" for i in range(n)]
    card_nums = [f"K{str(i).zfill(6)}" for i in range(n)]

    start_date = pd.Timestamp("2023-04-01")
    end_date = pd.Timestamp("2026-04-30")
    date_range_days = (end_date - start_date).days
    offer_dates = [
        (start_date + pd.Timedelta(days=int(d))).strftime("%d/%m/%Y")
        for d in rng.integers(0, date_range_days, size=n)
    ]

    # ── 8. ASSEMBLE ───────────────────────────────────────────────────────────

    df = pd.DataFrame({
        "cust_id":                      cust_ids,
        "ACCOUNT_NO":                   account_nos,
        "CARD_NUM":                     card_nums,
        "effective_acceptance":         effective_acceptance,
        "final_discount_pct":           final_discount_pct,
        "discount_tier":                discount_tier,
        "final_offer_discount_band":    final_offer_discount_band,
        "final_offer_amount":           final_offer_amount,
        "final_balance":                final_balance,
        "balance_bucket":               balance_bucket,
        "final_dpd":                    np.nan,
        "final_campaign":               final_campaign,
        "is_settlement":                is_settlement,
        "is_installment":               is_installment,
        "campaign_mix":                 campaign_mix,
        "last_campaign":                last_campaign,
        "last_discount_band":           last_discount_band,
        "total_offers":                 total_offers,
        "settlement_offers":            settlement_offers,
        "installment_offers":           installment_offers,
        "min_discount_pct":             min_discount_pct,
        "max_discount_pct":             max_discount_pct,
        "avg_discount_pct":             avg_discount_pct,
        "n_prior_accepts_real":         n_prior_accepts_real,
        "has_prior_accepts":            has_prior_accepts,
        "avg_accepted_settlement_pct":  avg_accepted_settlement_pct,
        "avg_accepted_installment_pct": avg_accepted_installment_pct,
        "offer_velocity":               offer_velocity,
        "months_in_collection":         months_in_collection,
        "final_offer_date":             offer_dates,
        # audit cols
        "_audit_method":       "payment_in_window (replaced FC=15)",
        "_audit_source":       "lis_negotiation + spc_txn_dly (payment-validated)",
        "_observation_window": "2023-04-01 to 2026-04-30",
    })

    return df


def validate_synthetic(df: pd.DataFrame) -> None:
    """Print key statistics vs targets from data summary."""
    print("=== SYNTHETIC DATA VALIDATION ===\n")
    checks = [
        ("effective_acceptance mean",  df["effective_acceptance"].mean(),   0.935,  0.02),
        ("final_discount_pct mean",    df["final_discount_pct"].mean(),     39.07,  2.0),
        ("final_balance mean",         df["final_balance"].mean(),          70_307, 10_000),
        ("months_in_collection mean",  df["months_in_collection"].mean(),   16.10,  1.5),
        ("total_offers mean",          df["total_offers"].mean(),           1.06,   0.15),
        ("is_settlement mean",         df["is_settlement"].mean(),          0.865,  0.02),
        ("offer_velocity mean",        df["offer_velocity"].mean(),         0.143,  0.05),
    ]
    for name, actual, target, tol in checks:
        status = "PASS" if abs(actual - target) <= tol else "WARN"
        print(f"  [{status}] {name:40s}  actual={actual:10.4f}  target={target:10.4f}")

    print(f"\n  Rows: {len(df):,}")
    print(f"\n  balance_bucket distribution:")
    print(df["balance_bucket"].value_counts().to_string())
    print(f"\n  discount_tier distribution:")
    print(df["discount_tier"].value_counts().to_string())
    print(f"\n  Acceptance by campaign type:")
    print(df.groupby("final_campaign")["effective_acceptance"].mean().round(3).to_string())
    print(f"\n  Acceptance by discount tier:")
    print(df.groupby("discount_tier")["effective_acceptance"].mean().round(3).sort_values().to_string())


if __name__ == "__main__":
    df = generate_synthetic_offers(n=5000, seed=42)
    validate_synthetic(df)
    out = "/Users/sushilkumar/Desktop/claude/src/recovery_agent_practical/scoring/synthetic_offers.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")
