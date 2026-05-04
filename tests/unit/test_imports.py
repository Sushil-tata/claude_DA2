"""
Test that all modules can be imported successfully.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def test_config_loader_import():
    """Test config loader can be imported."""
    from decision_agent.utils.config_loader import ConfigLoader
    assert ConfigLoader is not None


def test_router_import():
    """Test router can be imported."""
    from decision_agent.orchestrator.router import route_use_case, list_available_use_cases
    assert route_use_case is not None
    assert list_available_use_cases is not None


def test_synthetic_data_import():
    """Test synthetic data generator can be imported."""
    from decision_agent.data.synthetic_data import SyntheticDataGenerator
    assert SyntheticDataGenerator is not None


def test_splits_import():
    """Test temporal splitter can be imported."""
    from decision_agent.data.splits import TemporalSplitter
    assert TemporalSplitter is not None


def test_asof_join_import():
    """Test as-of joiner can be imported."""
    from decision_agent.data.asof_join import AsOfJoiner, LeakageDetector
    assert AsOfJoiner is not None
    assert LeakageDetector is not None


def test_all_imports_together():
    """Test all core imports together."""
    from decision_agent.utils.config_loader import ConfigLoader
    from decision_agent.orchestrator.router import route_use_case
    from decision_agent.data.synthetic_data import SyntheticDataGenerator
    from decision_agent.data.splits import TemporalSplitter
    from decision_agent.data.asof_join import AsOfJoiner
    
    # All imports successful
    assert True


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
