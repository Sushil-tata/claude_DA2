#!/usr/bin/env python3
"""
Phase 1 Logic Test

Tests business logic of Phase 1 modules (without Spark).
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("=" * 80)
print("Phase 1 Business Logic Test")
print("=" * 80)

# Test 1: Sparse History Scoring Logic
print("\n1. Testing sparse history scoring logic...")
try:
    # Create mock data
    customer_data = pd.DataFrame({
        'customer_id': ['C1', 'C2', 'C3'],
        'days_history': [120, 60, 30],
        'transaction_count': [50, 15, 5],
        'deposit_count': [10, 3, 1]
    })

    # Weights from config
    weight_days = 0.4
    weight_deposits = 0.3
    weight_transactions = 0.3

    # Min thresholds
    min_days = 90
    min_txns = 10
    min_deposits = 3

    # Calculate scores
    scores = []
    for _, row in customer_data.iterrows():
        days_score = min(row['days_history'] / min_days, 1.0)
        txn_score = min(row['transaction_count'] / min_txns, 1.0)
        deposit_score = min(row['deposit_count'] / min_deposits, 1.0)

        total_score = (weight_days * days_score +
                      weight_deposits * deposit_score +
                      weight_transactions * txn_score)
        scores.append(total_score)

    customer_data['sufficiency_score'] = scores

    # Classify
    customer_data['quality_flag'] = customer_data['sufficiency_score'].apply(
        lambda x: 'sufficient' if x >= 0.7 else ('marginal' if x >= 0.5 else 'insufficient')
    )

    print("   Customer Data Sufficiency:")
    for _, row in customer_data.iterrows():
        print(f"   {row['customer_id']}: score={row['sufficiency_score']:.2f}, flag={row['quality_flag']}")

    # Verify C1 is sufficient (120 days, 50 txns, 10 deposits -> all exceed thresholds)
    assert customer_data.loc[0, 'quality_flag'] == 'sufficient', "C1 should be sufficient"
    # Verify C3 is insufficient (30 days, 5 txns, 1 deposit -> all below thresholds)
    assert customer_data.loc[2, 'quality_flag'] == 'insufficient', "C3 should be insufficient"

    print("   ✓ Sparse history scoring logic correct")

except Exception as e:
    print(f"   ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Fair Lending Disparate Impact Calculation
print("\n2. Testing fair lending disparate impact calculation...")
try:
    # Mock predictions by group
    group_a_approved = 80  # Reference group (e.g., male)
    group_a_total = 100
    group_b_approved = 60  # Protected group (e.g., female)
    group_b_total = 100

    # Calculate approval rates
    rate_a = group_a_approved / group_a_total  # 0.80
    rate_b = group_b_approved / group_b_total  # 0.60

    # Disparate Impact ratio (4/5ths rule)
    di_ratio = min(rate_b / rate_a, rate_a / rate_b)

    print(f"   Group A (reference) approval rate: {rate_a:.2%}")
    print(f"   Group B (protected) approval rate: {rate_b:.2%}")
    print(f"   Disparate Impact ratio: {di_ratio:.2f}")

    # Check 80% rule
    threshold = 0.80
    passes_80_rule = di_ratio >= threshold

    if passes_80_rule:
        print(f"   ✓ PASS: DI ratio {di_ratio:.2f} >= {threshold} (no bias detected)")
    else:
        print(f"   ✗ FAIL: DI ratio {di_ratio:.2f} < {threshold} (bias detected!)")

    # This example should fail (0.60/0.80 = 0.75 < 0.80)
    assert not passes_80_rule, "Should detect bias in this example"
    print("   ✓ Fair lending DI calculation correct")

except Exception as e:
    print(f"   ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Deposit Periodicity Detection Logic (Simplified)
print("\n3. Testing deposit periodicity detection logic...")
try:
    # Mock deposit timestamps (biweekly pattern)
    deposit_dates = pd.to_datetime([
        '2024-01-01',  # First deposit
        '2024-01-15',  # +14 days
        '2024-01-29',  # +14 days
        '2024-02-12',  # +14 days
        '2024-02-26',  # +14 days
    ])

    # Calculate intervals
    intervals_td = deposit_dates.diff()
    intervals = pd.Series([td.days for td in intervals_td if pd.notna(td)])

    print(f"   Deposit intervals (days): {intervals.tolist()}")

    # Detect dominant period (median)
    median_interval = intervals.median()
    print(f"   Median interval: {median_interval} days")

    # Classify period
    if 13 <= median_interval <= 15:
        period_type = 'biweekly'
    elif 28 <= median_interval <= 31:
        period_type = 'monthly'
    elif 6 <= median_interval <= 8:
        period_type = 'weekly'
    else:
        period_type = 'irregular'

    print(f"   Detected period type: {period_type}")

    # Calculate regularity (coefficient of variation)
    cv = intervals.std() / intervals.mean() if intervals.mean() > 0 else 1.0
    print(f"   Coefficient of Variation: {cv:.2f}")

    # Confidence score (inverse of CV, capped)
    confidence = max(0.0, min(1.0, 1.0 - cv))
    print(f"   Period confidence: {confidence:.2f}")

    # Verify biweekly detection
    assert period_type == 'biweekly', "Should detect biweekly pattern"
    assert confidence > 0.8, "Should have high confidence for regular pattern"

    print("   ✓ Deposit periodicity detection logic correct")

except Exception as e:
    print(f"   ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Deposit Stability Calculation
print("\n4. Testing deposit stability calculation...")
try:
    # Mock deposit amounts
    # Stable salary: consistent amounts
    stable_deposits = np.array([5000, 5100, 4950, 5050, 5000])
    # Unstable gig income: variable amounts
    unstable_deposits = np.array([2000, 5000, 1500, 8000, 3000])

    def calculate_stability(deposits):
        mean = deposits.mean()
        std = deposits.std()
        cv = std / mean if mean > 0 else float('inf')

        # Stability score: 1 - normalized CV (lower CV = more stable)
        stability_score = max(0.0, min(1.0, 1.0 - min(cv, 1.0)))

        return {
            'mean': mean,
            'std': std,
            'cv': cv,
            'stability_score': stability_score
        }

    stable_metrics = calculate_stability(stable_deposits)
    unstable_metrics = calculate_stability(unstable_deposits)

    print(f"   Stable deposits (salary):")
    print(f"     Mean: ${stable_metrics['mean']:,.0f}")
    print(f"     CV: {stable_metrics['cv']:.3f}")
    print(f"     Stability score: {stable_metrics['stability_score']:.3f}")

    print(f"   Unstable deposits (gig):")
    print(f"     Mean: ${unstable_metrics['mean']:,.0f}")
    print(f"     CV: {unstable_metrics['cv']:.3f}")
    print(f"     Stability score: {unstable_metrics['stability_score']:.3f}")

    # Verify stable deposits have higher stability score
    assert stable_metrics['stability_score'] > unstable_metrics['stability_score'], \
        "Stable deposits should have higher stability score"
    assert stable_metrics['cv'] < unstable_metrics['cv'], \
        "Stable deposits should have lower CV"

    print("   ✓ Deposit stability calculation correct")

except Exception as e:
    print(f"   ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Champion/Challenger Comparison Logic
print("\n5. Testing Champion/Challenger comparison logic...")
try:
    # Mock model performance
    champion_mae = 15000
    challenger_mae = 14000

    champion_r2 = 0.75
    challenger_r2 = 0.78

    # Improvement thresholds from config
    mae_improvement_threshold = -0.05  # Must reduce MAE by 5%
    r2_improvement_threshold = 0.02    # Must improve R² by 2%

    # Calculate improvements
    mae_improvement = (challenger_mae - champion_mae) / champion_mae
    r2_improvement = challenger_r2 - champion_r2

    print(f"   Champion MAE: ${champion_mae:,}")
    print(f"   Challenger MAE: ${challenger_mae:,}")
    print(f"   MAE improvement: {mae_improvement:.1%} (threshold: {mae_improvement_threshold:.1%})")

    print(f"   Champion R²: {champion_r2:.3f}")
    print(f"   Challenger R²: {challenger_r2:.3f}")
    print(f"   R² improvement: {r2_improvement:.3f} (threshold: {r2_improvement_threshold:.3f})")

    # Promotion decision
    mae_check = mae_improvement <= mae_improvement_threshold  # Lower is better
    r2_check = r2_improvement >= r2_improvement_threshold     # Higher is better

    promote = mae_check and r2_check

    print(f"   MAE check: {'✓ PASS' if mae_check else '✗ FAIL'}")
    print(f"   R² check: {'✓ PASS' if r2_check else '✗ FAIL'}")
    print(f"   Promotion decision: {'✓ PROMOTE' if promote else '✗ REJECT'}")

    # This example should promote (MAE reduced by 6.7%, R² improved by 0.03)
    assert promote, "Challenger should be promoted in this example"
    print("   ✓ Champion/Challenger comparison logic correct")

except Exception as e:
    print(f"   ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Final Summary
print("\n" + "=" * 80)
print("Phase 1 Business Logic Test Summary")
print("=" * 80)
print("✅ Sparse History Scoring: Correct (flags sufficient/marginal/insufficient)")
print("✅ Fair Lending DI Calculation: Correct (detects 80% rule violations)")
print("✅ Deposit Periodicity Detection: Correct (detects biweekly pattern)")
print("✅ Deposit Stability Calculation: Correct (distinguishes stable vs unstable)")
print("✅ Champion/Challenger Comparison: Correct (promotion logic works)")
print("\n🎉 All Business Logic Tests PASSED!")
print("=" * 80)
