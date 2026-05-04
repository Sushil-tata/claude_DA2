"""
Tests for the data layer modules.

Tests data loader, quality analyzer, schema validator, and EDA engine.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    try:
        from data import (
            DataLoader,
            DataLoaderConfig,
            DataQualityAnalyzer,
            DataQualityConfig,
            SchemaValidator,
            DataSchema,
            FieldSchema,
            FieldType,
            EDAEngine,
            EDAConfig,
        )
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_data_structures():
    """Test that data structures can be instantiated."""
    print("\nTesting data structures...")

    try:
        from data import (
            DataLoaderConfig,
            DataQualityConfig,
            EDAConfig,
            FieldSchema,
            FieldType,
            DataSchema,
        )

        # Test configs
        loader_config = DataLoaderConfig(chunk_size=5000)
        assert loader_config.chunk_size == 5000
        print("✓ DataLoaderConfig works")

        quality_config = DataQualityConfig()
        assert quality_config.missing_threshold_warning == 0.05
        print("✓ DataQualityConfig works")

        eda_config = EDAConfig()
        assert eda_config.high_correlation_threshold == 0.7
        print("✓ EDAConfig works")

        # Test schema
        field = FieldSchema(
            name="age",
            field_type=FieldType.INTEGER,
            required=True,
            min_value=0,
            max_value=120,
        )
        assert field.name == "age"
        print("✓ FieldSchema works")

        schema = DataSchema(fields={"age": field})
        assert "age" in schema.fields
        print("✓ DataSchema works")

        return True

    except Exception as e:
        print(f"✗ Data structure test failed: {e}")
        return False


def test_basic_functionality():
    """Test basic functionality without external dependencies."""
    print("\nTesting basic functionality...")

    try:
        from data import (
            create_numeric_field,
            create_string_field,
            create_categorical_field,
        )

        # Test helper functions
        num_field = create_numeric_field("price", required=True, min_value=0)
        assert num_field.name == "price"
        assert num_field.required is True
        print("✓ create_numeric_field works")

        str_field = create_string_field("name", max_length=100)
        assert str_field.name == "name"
        assert str_field.max_length == 100
        print("✓ create_string_field works")

        cat_field = create_categorical_field("category", {"A", "B", "C"})
        assert cat_field.name == "category"
        assert "A" in cat_field.allowed_values
        print("✓ create_categorical_field works")

        return True

    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_module_structure():
    """Test module structure and documentation."""
    print("\nTesting module structure...")

    try:
        from data import data_loader, data_quality, schema_validator, eda_engine

        # Check docstrings
        assert data_loader.__doc__ is not None
        print("✓ data_loader has documentation")

        assert data_quality.__doc__ is not None
        print("✓ data_quality has documentation")

        assert schema_validator.__doc__ is not None
        print("✓ schema_validator has documentation")

        assert eda_engine.__doc__ is not None
        print("✓ eda_engine has documentation")

        # Check key classes exist
        assert hasattr(data_loader, "DataLoader")
        print("✓ DataLoader class exists")

        assert hasattr(data_quality, "DataQualityAnalyzer")
        print("✓ DataQualityAnalyzer class exists")

        assert hasattr(schema_validator, "SchemaValidator")
        print("✓ SchemaValidator class exists")

        assert hasattr(eda_engine, "EDAEngine")
        print("✓ EDAEngine class exists")

        return True

    except Exception as e:
        print(f"✗ Module structure test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 80)
    print("DATA LAYER MODULE TESTS")
    print("=" * 80)

    results = []

    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Data Structures", test_data_structures()))
    results.append(("Basic Functionality", test_basic_functionality()))
    results.append(("Module Structure", test_module_structure()))

    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    for test_name, passed in results:
        status = "PASSED" if passed else "FAILED"
        symbol = "✓" if passed else "✗"
        print(f"{symbol} {test_name}: {status}")

    all_passed = all(result[1] for result in results)
    print("\n" + "=" * 80)

    if all_passed:
        print("ALL TESTS PASSED ✓")
        return 0
    else:
        print("SOME TESTS FAILED ✗")
        return 1


if __name__ == "__main__":
    sys.exit(main())
