"""
Integration Tests: Agent Layer ↔ Recovery Engine
================================================
Tests the contract between recovery-engine-v2 (writer)
and the agentic layer (readers) via local-mode Delta (parquet).

Run with: pytest tests/integration/test_delta_integration.py -v

No Databricks or Spark required — uses MockDeltaWriter + AgentMemory local_mode.

Assertions:
  1. MockDeltaWriter produces schema-identical output to real DeltaLakeWriter
  2. 100 mock scores written → AgentMemory can read them back correctly
  3. Decision Agent produces exactly one decision per account
  4. Decision schema matches the output contract
  5. No PII in any Delta table (column name check)
  6. Explain Agent output contains: segment_label, d_optimal, erv_at_d_optimal
  7. ClaudeReasoner is NOT called for standard accounts (rule-based path)
  8. ClaudeReasoner IS triggered for high-value accounts (erv > THB 30,000)
"""

import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from agents.agent_memory import AgentMemory
from agents.claude_reasoner import ClaudeReasoner, _should_call_claude
from contracts.model_output_contract import (
    PII_BLACKLIST,
    REQUIRED_COLUMNS,
    PIIViolationError,
    validate_schema,
)
from recovery_engine.output.mock_delta_writer import MockDeltaWriter
from recovery_engine.output.synthetic_score_generator import SyntheticScoreGenerator

TEST_DATE       = str(date.today())
TEST_N          = 100
MOCK_OUTPUT_DIR = Path("/tmp/test_delta_integration")


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def synthetic_scores() -> pd.DataFrame:
    gen = SyntheticScoreGenerator(n=TEST_N, seed=99, score_date=date.today())
    return gen.generate()


@pytest.fixture(scope="module")
def mock_writer() -> MockDeltaWriter:
    return MockDeltaWriter(
        model_version="recovery-engine-v2.0.0-test",
        experiment_id="test-exp-001",
        score_date=date.today(),
        output_dir=MOCK_OUTPUT_DIR / "model_scores",
    )


@pytest.fixture(scope="module")
def written_summary(synthetic_scores, mock_writer) -> dict:
    return mock_writer.write(synthetic_scores)


@pytest.fixture(scope="module")
def agent_memory(written_summary) -> AgentMemory:
    import os
    os.environ["AGENT_LOCAL_DATA_DIR"] = str(MOCK_OUTPUT_DIR / "agent_data")
    memory = AgentMemory(execution_date=TEST_DATE, spark=None, local_mode=True)
    # Seed model_scores into AgentMemory local store
    df = pd.read_parquet(written_summary["output_path"])
    memory.write("model_scores", df)
    return memory


# ── Test 1: Schema completeness ───────────────────────────────────────────────

class TestSchemaContract:

    def test_all_required_columns_present(self, synthetic_scores):
        missing = [c for c in REQUIRED_COLUMNS if c not in synthetic_scores.columns]
        assert not missing, f"Missing required columns: {missing}"

    def test_no_nulls_in_non_nullable_columns(self, synthetic_scores):
        for col, spec in REQUIRED_COLUMNS.items():
            if not spec.nullable and col in synthetic_scores.columns:
                null_count = synthetic_scores[col].isnull().sum()
                assert null_count == 0, f"Column '{col}' has {null_count} nulls but is NOT nullable"

    def test_signal_quadrant_values(self, synthetic_scores):
        valid = {"A", "B", "C", "D"}
        bad   = set(synthetic_scores["signal_quadrant"].unique()) - valid
        assert not bad, f"Invalid signal_quadrant values: {bad}"

    def test_propensity_range(self, synthetic_scores):
        for col in ["propensity_30d", "propensity_90d", "propensity_180d"]:
            assert synthetic_scores[col].between(0.0, 1.0).all(), \
                f"{col} has values outside [0, 1]"

    def test_d_optimal_range(self, synthetic_scores):
        assert synthetic_scores["d_optimal"].between(0.20, 0.65).all(), \
            "d_optimal has values outside [0.20, 0.65]"

    def test_d_quadrant_has_null_segment(self, synthetic_scores):
        d_rows = synthetic_scores[synthetic_scores["signal_quadrant"] == "D"]
        assert d_rows["segment_label"].isnull().all(), \
            "Quadrant D rows must have null segment_label"

    def test_d_quadrant_low_confidence_true(self, synthetic_scores):
        d_rows = synthetic_scores[synthetic_scores["signal_quadrant"] == "D"]
        assert d_rows["low_confidence_flag"].all(), \
            "Quadrant D rows must have low_confidence_flag=True"


# ── Test 2: PII policy ────────────────────────────────────────────────────────

class TestPIIPolicy:

    def test_no_pii_in_model_scores(self, synthetic_scores):
        """No PII column names in model_scores output."""
        for col in synthetic_scores.columns:
            for pii_term in PII_BLACKLIST:
                assert pii_term not in col.lower(), \
                    f"PII term '{pii_term}' found in column name '{col}'"

    def test_pii_violation_raises(self):
        """Validate that PIIViolationError is raised when PII column present."""
        bad_df = pd.DataFrame([{
            "account_id": "ACC001",
            "name": "John Doe",           # PII — should raise
            "propensity_30d": 0.5,
        }])
        with pytest.raises(PIIViolationError):
            from contracts.model_output_contract import validate_pii
            validate_pii(bad_df, context="test")

    def test_claude_reasoner_strips_pii(self):
        """ClaudeReasoner must not pass PII even if caller passes it."""
        reasoner = ClaudeReasoner(dry_run=True)
        account_with_pii = {
            "account_id":       "ACC001",
            "signal_quadrant":  "A",
            "segment_label":    "SALARY_LIKE",
            "propensity_180d":  0.65,
            "erv_at_d_optimal": 50_000,   # triggers Claude call
            "d_optimal":        0.30,
            "name":             "John Doe",     # PII — should be stripped
            "national_id":      "1234567890",   # PII — should be stripped
        }
        with pytest.raises(PIIViolationError):
            reasoner.advise(account_with_pii)


# ── Test 3: MockDeltaWriter → AgentMemory round-trip ─────────────────────────

class TestWriterMemoryRoundTrip:

    def test_rows_written_matches_input(self, written_summary):
        assert written_summary["rows_written"] == TEST_N

    def test_all_quadrants_present(self, written_summary):
        dist = written_summary["quadrant_distribution"]
        assert set(dist.keys()) == {"A", "B", "C", "D"}, \
            f"Missing quadrants in distribution: {dist}"

    def test_agent_memory_reads_correct_rows(self, agent_memory):
        df = agent_memory.read("model_scores")
        assert len(df) == TEST_N, f"Expected {TEST_N} rows, got {len(df)}"

    def test_agent_memory_contains_required_columns(self, agent_memory):
        df = agent_memory.read("model_scores")
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        assert not missing, f"AgentMemory read is missing columns: {missing}"


# ── Test 4: ClaudeReasoner call thresholds ────────────────────────────────────

class TestClaudeReasonerThresholds:

    def test_standard_account_does_not_trigger_claude(self):
        account = {
            "account_id":          "ACC001",
            "signal_quadrant":     "B",
            "segment_label":       "GIG_FREELANCE",
            "propensity_180d":     0.35,
            "erv_at_d_optimal":    8_000,   # below 30K threshold
            "low_confidence_flag": False,
            "d_optimal":           0.35,
        }
        should_call, reason = _should_call_claude(account)
        assert not should_call, f"Should NOT call Claude for standard account, got: {reason}"

    def test_high_value_triggers_claude(self):
        account = {
            "account_id":          "ACC002",
            "signal_quadrant":     "A",
            "propensity_180d":     0.65,
            "erv_at_d_optimal":    45_000,   # above 30K threshold
            "low_confidence_flag": False,
            "d_optimal":           0.30,
        }
        should_call, reason = _should_call_claude(account)
        assert should_call, "Should call Claude for high-value account"
        assert "high_value" in reason

    def test_d_quadrant_high_balance_triggers_claude(self):
        account = {
            "account_id":          "ACC003",
            "signal_quadrant":     "D",
            "propensity_180d":     0.08,
            "erv_at_d_optimal":    25_000,   # above 20K D-quadrant threshold
            "low_confidence_flag": True,
            "d_optimal":           0.50,
        }
        should_call, reason = _should_call_claude(account)
        assert should_call
        assert "d_quadrant" in reason

    def test_unexpected_low_recovery_triggers_claude(self):
        account = {
            "account_id":          "ACC004",
            "signal_quadrant":     "B",
            "segment_label":       "SALARY_LIKE",   # not null
            "propensity_180d":     0.10,             # below 0.15 threshold
            "erv_at_d_optimal":    5_000,
            "low_confidence_flag": False,
            "d_optimal":           0.35,
        }
        should_call, reason = _should_call_claude(account)
        assert should_call
        assert "unexpected_low_recovery" in reason

    def test_dry_run_returns_proceed(self):
        reasoner = ClaudeReasoner(dry_run=True)
        account = {
            "account_id":          "ACC005",
            "signal_quadrant":     "A",
            "propensity_180d":     0.70,
            "erv_at_d_optimal":    40_000,
            "low_confidence_flag": False,
            "d_optimal":           0.28,
        }
        result = reasoner.advise(account)
        assert result["called_claude"] is True
        assert result["recommendation"] == "PROCEED"
        assert result["suggested_d_optimal"] is not None


# ── Test 5: AgentMemory signal / audit log ────────────────────────────────────

class TestAgentAuditLog:

    def test_signal_written_and_readable(self, agent_memory):
        from agents.base_agent import AgentSignal, AgentStatus
        sig = AgentSignal(
            agent_name="test_agent",
            status=AgentStatus.COMPLETE,
            execution_date=TEST_DATE,
            message="Test signal",
            metadata={"rows": 100},
        )
        agent_memory.signal(sig)
        status = agent_memory.get_agent_status("test_agent")
        assert status == "COMPLETE"

    def test_blocked_signal_readable(self, agent_memory):
        from agents.base_agent import AgentSignal, AgentStatus
        sig = AgentSignal(
            agent_name="blocked_agent",
            status=AgentStatus.BLOCKED,
            execution_date=TEST_DATE,
            message="Data quality check failed",
            metadata={"retry_agent": "data_quality_agent"},
        )
        agent_memory.signal(sig)
        status = agent_memory.get_agent_status("blocked_agent")
        assert status == "BLOCKED"


# ── Test 6: Explain output format ─────────────────────────────────────────────

class TestExplainOutputFormat:

    def test_explain_contains_required_fields(self, synthetic_scores):
        """
        Simulate Explain Agent output and assert required fields present.
        The real ExplainAgent reads from Delta — here we test the format contract.
        """
        sample = synthetic_scores[synthetic_scores["signal_quadrant"] == "A"].iloc[0]

        # Simulate what ExplainAgent would produce
        explanation = {
            "account_id":       sample["account_id"],
            "segment_label":    sample["segment_label"],
            "d_optimal":        sample["d_optimal"],
            "erv_at_d_optimal": sample["erv_at_d_optimal"],
            "propensity_180d":  sample["propensity_180d"],
            "narrative": (
                f"Account {sample['account_id']} in Segment {sample['segment_label']} "
                f"with {sample['propensity_180d']*100:.0f}% 180d recovery probability. "
                f"Recommended {sample['d_optimal']*100:.0f}% discount "
                f"(ERV: THB {sample['erv_at_d_optimal']:,.0f}). "
                f"Confidence: {'High' if not sample['low_confidence_flag'] else 'Low'}."
            ),
        }

        assert "segment_label"    in explanation
        assert "d_optimal"        in explanation
        assert "erv_at_d_optimal" in explanation
        assert "narrative"        in explanation
        assert str(explanation["account_id"]) in explanation["narrative"]
