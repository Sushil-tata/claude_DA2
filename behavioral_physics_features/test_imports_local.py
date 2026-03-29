"""
Local Import and Syntax Validation
===================================

This script validates that all modules can be imported and have correct syntax.
This can run locally WITHOUT Spark/Databricks.

For full tests WITH actual data, run in Databricks:
- test_xxx_handling.py
- test_all_fixes.py
- validate_lender_classification.py
"""

import sys
import importlib
from pathlib import Path

def test_imports():
    """Test that all modules can be imported"""
    print("\n" + "="*80)
    print("LOCAL VALIDATION - Import & Syntax Check")
    print("="*80 + "\n")

    modules_to_test = [
        "behavioral_physics_features.modules.config",
        "behavioral_physics_features.modules.bureau_schema_adapter",
        "behavioral_physics_features.modules.lender_ecology",
    ]

    results = []

    for module_name in modules_to_test:
        try:
            module = importlib.import_module(module_name)
            print(f"✅ {module_name:<60} PASS")
            results.append((module_name, "PASS", None))
        except Exception as e:
            print(f"❌ {module_name:<60} FAIL: {str(e)[:50]}")
            results.append((module_name, "FAIL", str(e)))

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    passed = sum(1 for _, status, _ in results if status == "PASS")
    failed = sum(1 for _, status, _ in results if status == "FAIL")

    print(f"\n✓ Passed: {passed}/{len(results)}")
    print(f"✗ Failed: {failed}/{len(results)}")

    if failed > 0:
        print("\n❌ Import validation FAILED")
        print("\nFailed modules:")
        for name, status, error in results:
            if status == "FAIL":
                print(f"  - {name}")
                print(f"    Error: {error}")
        return False
    else:
        print("\n✅ All imports successful!")
        return True

def test_config():
    """Test config module specifically"""
    print("\n" + "="*80)
    print("CONFIG VALIDATION")
    print("="*80 + "\n")

    try:
        from behavioral_physics_features.modules.config import LenderTypeConfig

        config = LenderTypeConfig()

        # Test Thai lender classification
        test_cases = [
            ("BANGKOK BANK", "BBL", "COMMERCIAL_BANK"),
            ("ธนาคารกรุงเทพ", "BBL", "COMMERCIAL_BANK"),
            ("GOVERNMENT SAVINGS BANK", "GSB", "SFI"),
            ("KASIKORN", "KBANK", "COMMERCIAL_BANK"),
            ("MUANG THAI", "MT", "PERSONAL_LOAN"),
            ("RABBIT FINANCE", "RABBIT", "FINTECH"),
            ("TOYOTA LEASING", "TOYOTA", "LEASING"),
            ("UNKNOWN LENDER", "UNK", "OTHER"),
        ]

        print("Testing lender type mapping:")
        all_passed = True

        for lender_name, lender_id, expected_type in test_cases:
            actual_type = config.map_lender_type(lender_name, lender_id)
            status = "✅" if actual_type == expected_type else "❌"

            if actual_type != expected_type:
                all_passed = False

            print(f"{status} {lender_name:<40} → {actual_type:<20} (expected: {expected_type})")

        if all_passed:
            print("\n✅ Config validation PASSED")
            return True
        else:
            print("\n❌ Config validation FAILED")
            return False

    except Exception as e:
        print(f"❌ Config validation FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all local validations"""
    print("\n" + "="*80)
    print("BEHAVIORAL PHYSICS - LOCAL VALIDATION")
    print("="*80)
    print("\nNote: This validates syntax and imports only.")
    print("For full tests with Spark/data, run in Databricks:")
    print("  - test_xxx_handling.py")
    print("  - test_all_fixes.py")
    print("  - validate_lender_classification.py")
    print("="*80)

    # Run tests
    import_test = test_imports()
    config_test = test_config()

    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)

    all_tests = [
        ("Import validation", import_test),
        ("Config validation", config_test),
    ]

    for test_name, passed in all_tests:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status:<15} {test_name}")

    total_passed = sum(1 for _, passed in all_tests if passed)
    total_tests = len(all_tests)

    print(f"\nTotal: {total_passed}/{total_tests} tests passed")

    if total_passed == total_tests:
        print("\n✅ LOCAL VALIDATION PASSED")
        print("\n📋 Next step: Run full tests in Databricks")
        return True
    else:
        print("\n❌ LOCAL VALIDATION FAILED")
        print("\n🔧 Fix errors above before running Databricks tests")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
