"""
run_agent.py — Databricks Wheel Entry Point
============================================
Single entry point for all 7 pipeline agents.
Referenced in agentic_nba_workflow.yml as:
    entry_point: run_agent

Usage (Databricks task parameters):
    --agent data_quality   --execution-date 2026-03-28
    --agent feature        --execution-date 2026-03-28
    --agent model          --execution-date 2026-03-28
    --agent constraint     --execution-date 2026-03-28
    --agent decision       --execution-date 2026-03-28 [--dry-run]
    --agent validation     --execution-date 2026-03-28
    --agent explain        --execution-date 2026-03-28

Local usage (no Spark):
    python jobs/run_agent.py --agent data_quality --execution-date 2026-03-28 --local
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("run_agent")

# ── Agent registry ────────────────────────────────────────────────────────────
AGENT_REGISTRY = {
    "propensity":   "agents.propensity_agent.PropensityAgent",
    "data_quality": "agents.data_quality_agent.DataQualityAgent",
    "feature":      "agents.feature_agent.FeatureAgent",
    "holdout":      "agents.holdout_agent.HoldoutAgent",
    "model":        "agents.model_agent.ModelAgent",
    "capacity":     "agents.capacity_agent.CapacityAllocationAgent",
    "constraint":   "agents.constraint_agent.ConstraintAgent",
    "decision":     "agents.decision_agent.DecisionAgent",
    "validation":   "agents.validation_agent.ValidationAgent",
    "explain":      "agents.explain_agent.ExplainAgent",
}


def _import_agent(dotted_path: str):
    module_path, class_name = dotted_path.rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _get_spark(local_mode: bool):
    if local_mode:
        return None
    try:
        from pyspark.sql import SparkSession
        spark = (
            SparkSession.builder
            .appName("agentic_nba_pipeline")
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
            .config(
                "spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog",
            )
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("WARN")
        return spark
    except Exception as e:
        logger.warning(f"Could not create SparkSession: {e}. Falling back to local mode.")
        return None


def main():
    parser = argparse.ArgumentParser(description="Run a named pipeline agent")
    parser.add_argument(
        "--agent",
        required=True,
        choices=list(AGENT_REGISTRY.keys()),
        help="Agent to run",
    )
    parser.add_argument(
        "--execution-date",
        required=True,
        help="Scoring date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Dry run — no Delta writes, no Claude API calls",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        default=False,
        help="Local mode — use parquet instead of Delta Lake (no Spark needed)",
    )
    args = parser.parse_args()

    logger.info(
        f"Starting agent={args.agent} | date={args.execution_date} | "
        f"dry_run={args.dry_run} | local={args.local}"
    )

    # ── Build dependencies ────────────────────────────────────────────────────
    spark       = _get_spark(local_mode=args.local)
    local_mode  = args.local or (spark is None)

    from agents.agent_memory import AgentMemory
    memory = AgentMemory(
        execution_date=args.execution_date,
        spark=spark,
        local_mode=local_mode,
    )

    # ── Instantiate and run agent ─────────────────────────────────────────────
    AgentClass = _import_agent(AGENT_REGISTRY[args.agent])
    agent = AgentClass(
        execution_date=args.execution_date,
        memory=memory,
        dry_run=args.dry_run,
    )

    signal = agent.run()

    logger.info(
        f"Agent finished | status={signal.status} | "
        f"message={signal.message[:120]}"
    )

    # Non-zero exit for BLOCKED/FAILED so Databricks Workflow stops the pipeline
    if signal.status.value in ("BLOCKED", "FAILED"):
        sys.exit(1)


if __name__ == "__main__":
    main()
