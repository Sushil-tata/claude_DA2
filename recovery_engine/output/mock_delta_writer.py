"""
MockDeltaWriter
===============
Drop-in replacement for DeltaLakeWriter that writes to local Parquet.

Used for:
  - Unit and integration testing without Databricks
  - Agentic team to stub model outputs before recovery-engine-v2 is ready
  - CI/CD pipeline runs

Schema-identical to DeltaLakeWriter — the agentic team's agents will
read the same columns in the same types from the parquet file as they
would from the real Delta table.

Usage:
    writer = MockDeltaWriter(
        output_dir="/tmp/mock_delta",
        model_version="recovery-engine-v2.0.0-mock",
        experiment_id="mock-exp-001",
    )
    summary = writer.write(scores_df)
    print(summary["output_path"])   # pass this to AgentMemory in local_mode
"""

import logging
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

from contracts.model_output_contract import validate_schema

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path("/tmp/mock_delta/recovery/model_scores")


class MockDeltaWriter:
    """
    Writes validated model scores to a local Parquet file.
    Schema-identical to the real DeltaLakeWriter.
    """

    def __init__(
        self,
        model_version: str  = "recovery-engine-v2.0.0-mock",
        experiment_id: str  = "mock-experiment-001",
        score_date: Optional[date] = None,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
    ):
        self.model_version = model_version
        self.experiment_id = experiment_id
        self.score_date    = str(score_date or date.today())
        self.output_dir    = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write(self, df: pd.DataFrame) -> dict:
        """
        Validate and write to local parquet.

        Returns:
            dict with write summary including output_path
        """
        df = df.copy()
        df["score_date"]    = self.score_date
        df["model_version"] = self.model_version
        df["experiment_id"] = self.experiment_id

        # Same validation as the real writer
        warnings = validate_schema(df)

        output_path = self.output_dir / f"model_scores_{self.score_date}.parquet"
        df.to_parquet(output_path, index=False)

        summary = {
            "rows_written":          len(df),
            "score_date":            self.score_date,
            "model_version":         self.model_version,
            "output_path":           str(output_path),
            "quadrant_distribution": df["signal_quadrant"].value_counts().to_dict(),
            "schema_warnings":       warnings,
        }
        logger.info(f"[MockDeltaWriter] {len(df)} rows → {output_path}")
        return summary

    def read_back(self) -> pd.DataFrame:
        """Read the last written parquet — convenience for tests."""
        path = self.output_dir / f"model_scores_{self.score_date}.parquet"
        return pd.read_parquet(path)
