"""
Main entry point for decision agent use cases.

Loads YAML config, validates against schema, routes to appropriate pipeline.

Usage:
    python jobs/run_usecase.py --config conf/use_cases/income_estimation.yaml
    python jobs/run_usecase.py --config conf/use_cases/income_estimation.yaml --dry-run
"""

import argparse
import sys
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from decision_agent.utils.config_loader import ConfigLoader
from decision_agent.orchestrator.router import route_use_case, list_available_use_cases

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Decision Agent Platform - Use Case Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python jobs/run_usecase.py --config conf/use_cases/income_estimation.yaml
  python jobs/run_usecase.py --config conf/use_cases/income_estimation.yaml --dry-run
  python jobs/run_usecase.py --list-use-cases
        """
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to use case YAML configuration"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config only, don't execute pipeline"
    )

    parser.add_argument(
        "--list-use-cases",
        action="store_true",
        help="List all available use cases and exit"
    )

    args = parser.parse_args()

    # List use cases if requested
    if args.list_use_cases:
        print("\n" + "="*80)
        print("AVAILABLE USE CASES")
        print("="*80)
        use_cases = list_available_use_cases()
        for use_case_id, description in use_cases.items():
            print(f"  {use_case_id:25s} - {description}")
        print()
        return 0

    # Config path is required for non-list operations
    if not args.config:
        parser.error("--config is required (or use --list-use-cases)")

    # Find schema path (relative to this file)
    project_root = Path(__file__).parent.parent
    schema_path = project_root / "schemas" / "config_schemas" / "use_case_config.schema.json"

    if not schema_path.exists():
        logger.error(f"Schema not found: {schema_path}")
        return 1

    # Load and validate configuration
    logger.info("="*80)
    logger.info("DECISION AGENT PLATFORM - USE CASE RUNNER")
    logger.info("="*80)

    try:
        config_loader = ConfigLoader(str(schema_path))
        config = config_loader.load_use_case(args.config)

        logger.info(f"\nUse Case: {config['use_case_id']}")
        logger.info(f"Version: {config['version']}")
        logger.info(f"Description: {config.get('description', 'N/A')}")

        if args.dry_run:
            logger.info("\n✓ DRY RUN: Configuration is valid")
            logger.info("Pipeline would execute with the following config:")
            logger.info(f"  - Features: {len(config['features']['feature_list'])} features")
            logger.info(f"  - Model: {config['model']['algorithm']}")
            logger.info(f"  - Output: {config['output']['table_name']}")
            return 0

        # Execute pipeline
        logger.info("\n" + "="*80)
        logger.info("EXECUTING PIPELINE")
        logger.info("="*80)

        results = route_use_case(config["use_case_id"], config)

        logger.info("\n" + "="*80)
        logger.info("PIPELINE EXECUTION COMPLETE")
        logger.info("="*80)
        logger.info(f"Status: {results.get('status', 'unknown')}")
        logger.info(f"Message: {results.get('message', 'N/A')}")

        return 0

    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {e}")
        return 1
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
