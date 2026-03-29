#!/usr/bin/env python3
"""
Main entry point for running decision agent use cases.

Usage:
    python jobs/run_usecase.py --config conf/use_cases/income_estimation.yaml
    python jobs/run_usecase.py --config conf/use_cases/income_estimation.yaml --dry-run
    python jobs/run_usecase.py --config conf/use_cases/income_estimation.yaml --execution-date 2024-12-01
"""
import argparse
import sys
import logging
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from decision_agent.config.config_loader import load_config
from decision_agent.orchestrator.router import route_use_case, list_available_use_cases

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Run decision agent use case pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run income estimation with synthetic data
  python jobs/run_usecase.py --config conf/use_cases/income_estimation.yaml

  # Dry run to validate config
  python jobs/run_usecase.py --config conf/use_cases/income_estimation.yaml --dry-run

  # Run with specific execution date
  python jobs/run_usecase.py --config conf/use_cases/income_estimation.yaml --execution-date 2024-12-01

  # List available use cases
  python jobs/run_usecase.py --list
        """
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to use case YAML configuration file"
    )

    parser.add_argument(
        "--execution-date",
        type=str,
        default=datetime.now().strftime("%Y-%m-%d"),
        help="Execution date for pipeline (default: today)"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without executing pipeline"
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List available use cases"
    )

    parser.add_argument(
        "--local",
        action="store_true",
        help="Run in local mode (no Databricks connection required)"
    )

    args = parser.parse_args()

    # List available use cases
    if args.list:
        use_cases = list_available_use_cases()
        print("\nAvailable use cases:")
        for uc in use_cases:
            print(f"  - {uc}")
        print()
        return 0

    # Validate arguments
    if not args.config:
        parser.error("--config is required (or use --list to see available use cases)")

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return 1

    try:
        # Load and validate configuration
        logger.info(f"Loading configuration from: {config_path}")
        config = load_config(str(config_path), execution_date=args.execution_date)

        logger.info(f"Configuration loaded successfully:")
        logger.info(f"  Use Case: {config['use_case_id']}")
        logger.info(f"  Version: {config['version']}")
        logger.info(f"  Description: {config.get('description', 'N/A')}")
        logger.info(f"  Execution Date: {args.execution_date}")

        # Dry run: validate and exit
        if args.dry_run:
            logger.info("✓ Configuration is valid (dry-run mode)")
            logger.info("Pipeline steps that would execute:")
            logger.info("  1. Load/generate data")
            logger.info("  2. Create temporal splits")
            logger.info("  3. Compute features")
            logger.info("  4. Train model")
            logger.info("  5. Validate model")
            logger.info("  6. Score test set")
            logger.info("  7. Write decisions to Delta Lake")
            return 0

        # Execute pipeline
        logger.info(f"Starting pipeline execution...")
        logger.info("=" * 80)

        # Route to appropriate use case pipeline
        spark = None
        if not args.local:
            try:
                from decision_agent.utils.spark_utils import get_spark_session
                spark = get_spark_session()
                logger.info("✓ Spark session initialized")
            except Exception as e:
                logger.warning(f"Could not initialize Spark: {e}")
                logger.info("Running in local mode (limited functionality)")

        results = route_use_case(
            use_case_id=config['use_case_id'],
            config=config,
            spark=spark
        )

        # Log results
        logger.info("=" * 80)
        logger.info("Pipeline execution completed successfully!")
        logger.info("\nResults Summary:")
        for key, value in results.items():
            logger.info(f"  {key}: {value}")

        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
