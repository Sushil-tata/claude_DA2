"""
Validate Thai Lender Classification
====================================

Checks actual lender names from bureau data and validates classification.

Run this in Databricks to:
1. Get top lenders by customer count
2. Test classification for each lender
3. Identify unclassified lenders (OTHER category)
4. Show distribution by lender type
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from behavioral_physics_features.modules.config import LenderTypeConfig
from datetime import datetime

def validate_lender_classification():
    """
    Validate lender classification on actual bureau data.
    """
    print("\n" + "="*80)
    print("LENDER CLASSIFICATION VALIDATION")
    print(f"Started: {datetime.now()}")
    print("="*80 + "\n")

    # Initialize Spark
    spark = SparkSession.builder \
        .appName("Validate_Lender_Classification") \
        .getOrCreate()

    # Initialize config
    config = LenderTypeConfig()

    # ========================================================================
    # STEP 1: Get Top Lenders from Bureau Account Table
    # ========================================================================
    print("STEP 1: Getting top lenders from bureau account table...")
    print("-" * 80)

    try:
        catalog = "cdx_mdz_prd.cdx_persist_mnf_res_db"
        bureau_account = spark.table(f"{catalog}.mnf_cra_rvw_s_account")

        # Get top lenders by customer count
        top_lenders = bureau_account.filter(
            F.col("MEMBERSHORTNAME").isNotNull()
        ).groupBy(
            "MEMBERSHORTNAME", "MEMBERCODE"
        ).agg(
            F.countDistinct("REF_NO").alias("customer_count"),
            F.count("*").alias("account_count")
        ).orderBy(
            F.desc("customer_count")
        ).limit(100)

        total_lenders = top_lenders.count()
        print(f"✓ Found {total_lenders} unique lenders")

        # Collect for analysis
        top_lenders_list = top_lenders.collect()

    except Exception as e:
        print(f"❌ Error loading bureau data: {e}")
        print("\nFalling back to test data...")

        # Fallback test data
        test_lenders = [
            ("ธนาคารกรุงเทพ จำกัด (มหาชน)", "BBL", 50000, 75000),
            ("KASIKORNBANK PUBLIC COMPANY LIMITED", "KBANK", 45000, 68000),
            ("SIAM COMMERCIAL BANK PUBLIC COMPANY LIMITED", "SCB", 40000, 62000),
            ("GOVERNMENT SAVINGS BANK", "GSB", 35000, 55000),
            ("ธนาคารกรุงไทย จำกัด (มหาชน)", "KTB", 30000, 48000),
            ("BANK FOR AGRICULTURE AND AGRICULTURAL COOPERATIVES", "BAAC", 28000, 45000),
            ("KRUNGSRI CONSUMER PUBLIC COMPANY LIMITED", "BAY", 25000, 40000),
            ("MUANG THAI CAPITAL PUBLIC COMPANY LIMITED", "MT", 20000, 35000),
            ("EASY BUY PUBLIC COMPANY LIMITED", "EB", 18000, 32000),
            ("RABBIT FINANCE COMPANY LIMITED", "RABBIT", 15000, 28000),
            ("AEON THANA SINSAP (THAILAND) PUBLIC COMPANY LIMITED", "AEON", 12000, 25000),
            ("TOYOTA LEASING (THAILAND) COMPANY LIMITED", "TOYOTA", 10000, 22000),
        ]

        from pyspark.sql.types import StructType, StructField, StringType, LongType
        schema = StructType([
            StructField("MEMBERSHORTNAME", StringType(), True),
            StructField("MEMBERCODE", StringType(), True),
            StructField("customer_count", LongType(), True),
            StructField("account_count", LongType(), True)
        ])

        top_lenders = spark.createDataFrame(test_lenders, schema)
        top_lenders_list = top_lenders.collect()
        print(f"✓ Using {len(top_lenders_list)} test lenders")

    # ========================================================================
    # STEP 2: Test Classification for Each Lender
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 2: Testing classification for top lenders...")
    print("-" * 80)

    results = []

    for row in top_lenders_list:
        lender_name = row["MEMBERSHORTNAME"]
        lender_id = row["MEMBERCODE"]
        customer_count = row["customer_count"]
        account_count = row["account_count"]

        # Test classification (only pass lender_name, not lender_id)
        lender_type = config.map_lender_type(lender_name)

        results.append({
            "lender_name": lender_name,
            "lender_id": lender_id,
            "lender_type": lender_type,
            "customer_count": customer_count,
            "account_count": account_count
        })

    # ========================================================================
    # STEP 3: Show Results
    # ========================================================================
    print("\nTop 30 Lenders by Customer Count:")
    print("="*120)
    print(f"{'Lender Name':<50} {'ID':<10} {'Type':<20} {'Customers':>12} {'Accounts':>12}")
    print("-"*120)

    for i, result in enumerate(results[:30], 1):
        lender_name = result["lender_name"][:48] if result["lender_name"] else "UNKNOWN"
        lender_id = result["lender_id"][:8] if result["lender_id"] else "N/A"
        lender_type = result["lender_type"]
        customers = result["customer_count"]
        accounts = result["account_count"]

        # Color code by type
        type_marker = "✅" if lender_type != "OTHER" else "⚠️ "

        print(f"{type_marker} {lender_name:<48} {lender_id:<10} {lender_type:<20} {customers:>12,} {accounts:>12,}")

    # ========================================================================
    # STEP 4: Distribution by Lender Type
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 4: Distribution by lender type...")
    print("-" * 80)

    # Count by type
    type_counts = {}
    type_customers = {}
    type_accounts = {}

    for result in results:
        ltype = result["lender_type"]
        type_counts[ltype] = type_counts.get(ltype, 0) + 1
        type_customers[ltype] = type_customers.get(ltype, 0) + result["customer_count"]
        type_accounts[ltype] = type_accounts.get(ltype, 0) + result["account_count"]

    total_customers = sum(type_customers.values())
    total_accounts = sum(type_accounts.values())

    print(f"\n{'Lender Type':<25} {'Lenders':>10} {'Customers':>15} {'%':>8} {'Accounts':>15} {'%':>8}")
    print("-"*90)

    # Sort by customer count
    for ltype in sorted(type_customers.keys(), key=lambda x: type_customers[x], reverse=True):
        lender_count = type_counts[ltype]
        customers = type_customers[ltype]
        accounts = type_accounts[ltype]
        cust_pct = (customers / total_customers * 100) if total_customers > 0 else 0
        acct_pct = (accounts / total_accounts * 100) if total_accounts > 0 else 0

        marker = "⚠️ " if ltype == "OTHER" else "✅ "
        print(f"{marker}{ltype:<23} {lender_count:>10} {customers:>15,} {cust_pct:>7.1f}% {accounts:>15,} {acct_pct:>7.1f}%")

    print("-"*90)
    print(f"{'TOTAL':<25} {len(results):>10} {total_customers:>15,} {'100.0%':>8} {total_accounts:>15,} {'100.0%':>8}")

    # ========================================================================
    # STEP 5: Identify Unclassified Lenders (OTHER)
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 5: Unclassified lenders (OTHER category)...")
    print("-" * 80)

    other_lenders = [r for r in results if r["lender_type"] == "OTHER"]

    if other_lenders:
        print(f"\n⚠️  Found {len(other_lenders)} lenders classified as OTHER:")
        print(f"\n{'Lender Name':<60} {'ID':<10} {'Customers':>12}")
        print("-"*90)

        for result in sorted(other_lenders, key=lambda x: x["customer_count"], reverse=True):
            lender_name = result["lender_name"][:58] if result["lender_name"] else "UNKNOWN"
            lender_id = result["lender_id"][:8] if result["lender_id"] else "N/A"
            customers = result["customer_count"]
            print(f"{lender_name:<60} {lender_id:<10} {customers:>12,}")

        other_pct = (type_customers.get("OTHER", 0) / total_customers * 100) if total_customers > 0 else 0

        print(f"\n⚠️  OTHER category represents {other_pct:.1f}% of customers")

        if other_pct > 5:
            print(f"\n🚨 ACTION REQUIRED: OTHER category >5% - add these lenders to config.py")
            print(f"\nTo fix:")
            print(f"  1. Review lenders above")
            print(f"  2. Identify correct category (SFI, COMMERCIAL_BANK, etc.)")
            print(f"  3. Add to LENDER_TYPE_MAPPING in config.py")
            print(f"  4. Re-run this validation")
        else:
            print(f"\n✅ OTHER category <5% - acceptable")

    else:
        print(f"\n✅ No lenders classified as OTHER - all lenders mapped!")

    # ========================================================================
    # STEP 6: Validation Summary
    # ========================================================================
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)

    total_types = len(type_counts)
    classified_lenders = sum(1 for r in results if r["lender_type"] != "OTHER")
    classification_rate = (classified_lenders / len(results) * 100) if results else 0

    print(f"\n✓ Total lenders analyzed: {len(results)}")
    print(f"✓ Lender types found: {total_types}")
    print(f"✓ Classification rate: {classification_rate:.1f}% ({classified_lenders}/{len(results)})")
    print(f"✓ Total customers: {total_customers:,}")
    print(f"✓ Total accounts: {total_accounts:,}")

    # Category breakdown
    print(f"\nCategory Breakdown:")
    expected_categories = ["SFI", "COMMERCIAL_BANK", "PERSONAL_LOAN", "LEASING", "FINTECH", "CARDX"]
    for cat in expected_categories:
        count = type_counts.get(cat, 0)
        customers = type_customers.get(cat, 0)
        pct = (customers / total_customers * 100) if total_customers > 0 else 0
        status = "✅" if count > 0 else "⚠️ "
        print(f"  {status} {cat:<20}: {count:>3} lenders, {customers:>12,} customers ({pct:>5.1f}%)")

    # Quality checks
    print(f"\nQuality Checks:")
    checks = []

    # Check 1: OTHER should be <5%
    other_pct = (type_customers.get("OTHER", 0) / total_customers * 100) if total_customers > 0 else 0
    check1 = "✅ PASS" if other_pct < 5 else f"❌ FAIL ({other_pct:.1f}% > 5%)"
    checks.append(("OTHER category <5%", check1))

    # Check 2: At least 4 categories should have data
    active_categories = sum(1 for cat in expected_categories if type_counts.get(cat, 0) > 0)
    check2 = "✅ PASS" if active_categories >= 4 else f"❌ FAIL (only {active_categories} categories)"
    checks.append(("At least 4 active categories", check2))

    # Check 3: COMMERCIAL_BANK should be largest
    largest_type = max(type_customers.keys(), key=lambda x: type_customers[x])
    check3 = "✅ PASS" if largest_type == "COMMERCIAL_BANK" else f"⚠️  WARNING (largest: {largest_type})"
    checks.append(("COMMERCIAL_BANK is largest", check3))

    for check_name, status in checks:
        print(f"  {status:<15} {check_name}")

    print("\n" + "="*80)
    print(f"Validation Completed: {datetime.now()}")
    print("="*80 + "\n")

    # Return results for further analysis
    return results, type_counts, type_customers


# ============================================================================
# Run Validation
# ============================================================================

if __name__ == "__main__":
    try:
        results, type_counts, type_customers = validate_lender_classification()

        print("\n✅ Validation completed successfully!")
        print("\nResults available as:")
        print("  - results: List of lender classification results")
        print("  - type_counts: Count of lenders by type")
        print("  - type_customers: Customer count by type")

        # Show next steps
        print("\n" + "="*80)
        print("NEXT STEPS")
        print("="*80)
        print("\n1. Review lenders classified as 'OTHER' above")
        print("2. Add missing lenders to config.py (LenderTypeConfig)")
        print("3. Re-run this script to verify")
        print("4. Run full pipeline with updated classification")

        print("\nTo add lenders to config.py:")
        print("  behavioral_physics_features/modules/config.py")
        print("  → LenderTypeConfig class")
        print("  → LENDER_TYPE_MAPPING dictionary")

    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
