"""
Data Loader Module

Comprehensive data loading functionality supporting multiple formats and sources.
Includes connection pooling, data validation, chunking for large files, and robust
error handling.

Author: Principal Data Science Decision Agent
"""

import io
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from loguru import logger
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.pool import NullPool, QueuePool


class DataLoaderConfig:
    """Configuration for data loader."""

    def __init__(
        self,
        chunk_size: int = 10000,
        max_retries: int = 3,
        pool_size: int = 5,
        pool_recycle: int = 3600,
        encoding: str = "utf-8",
        **kwargs,
    ):
        self.chunk_size = chunk_size
        self.max_retries = max_retries
        self.pool_size = pool_size
        self.pool_recycle = pool_recycle
        self.encoding = encoding
        self.extra_config = kwargs


class DataLoader:
    """
    Comprehensive data loader supporting multiple formats and sources.

    Supports:
    - CSV, JSON, Excel files
    - SQL databases (PostgreSQL, MySQL, SQLite, etc.)
    - Large file handling with chunking
    - Connection pooling for databases
    - Automatic data type inference
    - Robust error handling and logging

    Examples:
        >>> loader = DataLoader()
        >>> df = loader.load_csv("data.csv")
        >>> df = loader.load_sql("SELECT * FROM users", connection_string)
        >>> for chunk in loader.load_csv_chunked("large_file.csv"):
        ...     process(chunk)
    """

    def __init__(self, config: Optional[DataLoaderConfig] = None):
        """
        Initialize DataLoader.

        Args:
            config: Configuration object for the loader
        """
        self.config = config or DataLoaderConfig()
        self._engines: Dict[str, Any] = {}
        logger.info("DataLoader initialized with config: {}", self.config.__dict__)

    def load_csv(
        self,
        filepath: Union[str, Path],
        infer_types: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Load CSV file into DataFrame.

        Args:
            filepath: Path to CSV file
            infer_types: Whether to infer optimal data types
            **kwargs: Additional arguments passed to pd.read_csv

        Returns:
            Loaded DataFrame

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is invalid
        """
        filepath = Path(filepath)
        logger.info("Loading CSV file: {}", filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        try:
            # Load CSV with default encoding
            df = pd.read_csv(
                filepath, encoding=self.config.encoding, low_memory=False, **kwargs
            )

            logger.info(
                "Loaded CSV: {} rows, {} columns", len(df), len(df.columns)
            )

            if infer_types:
                df = self._infer_types(df)

            return df

        except UnicodeDecodeError:
            logger.warning("UTF-8 decoding failed, trying alternate encodings")
            for encoding in ["latin-1", "iso-8859-1", "cp1252"]:
                try:
                    df = pd.read_csv(filepath, encoding=encoding, low_memory=False, **kwargs)
                    logger.info("Successfully loaded with {} encoding", encoding)
                    if infer_types:
                        df = self._infer_types(df)
                    return df
                except Exception:
                    continue
            raise ValueError(f"Could not decode file: {filepath}")

        except Exception as e:
            logger.error("Error loading CSV {}: {}", filepath, str(e))
            raise

    def load_csv_chunked(
        self,
        filepath: Union[str, Path],
        chunk_size: Optional[int] = None,
        infer_types: bool = True,
        **kwargs,
    ) -> Iterator[pd.DataFrame]:
        """
        Load CSV file in chunks for large files.

        Args:
            filepath: Path to CSV file
            chunk_size: Number of rows per chunk
            infer_types: Whether to infer optimal data types
            **kwargs: Additional arguments passed to pd.read_csv

        Yields:
            DataFrame chunks

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        filepath = Path(filepath)
        chunk_size = chunk_size or self.config.chunk_size

        logger.info("Loading CSV file in chunks: {}, chunk_size={}", filepath, chunk_size)

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        try:
            chunks = pd.read_csv(
                filepath,
                encoding=self.config.encoding,
                chunksize=chunk_size,
                low_memory=False,
                **kwargs,
            )

            for i, chunk in enumerate(chunks):
                logger.debug("Processing chunk {}: {} rows", i, len(chunk))
                if infer_types:
                    chunk = self._infer_types(chunk)
                yield chunk

        except Exception as e:
            logger.error("Error loading chunked CSV {}: {}", filepath, str(e))
            raise

    def load_json(
        self,
        filepath: Union[str, Path],
        orient: str = "records",
        **kwargs,
    ) -> pd.DataFrame:
        """
        Load JSON file into DataFrame.

        Args:
            filepath: Path to JSON file
            orient: JSON orientation ('records', 'index', 'columns', etc.)
            **kwargs: Additional arguments passed to pd.read_json

        Returns:
            Loaded DataFrame

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If JSON is invalid
        """
        filepath = Path(filepath)
        logger.info("Loading JSON file: {}", filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        try:
            df = pd.read_json(filepath, orient=orient, encoding=self.config.encoding, **kwargs)
            logger.info("Loaded JSON: {} rows, {} columns", len(df), len(df.columns))
            return df

        except ValueError as e:
            logger.error("Invalid JSON format in {}: {}", filepath, str(e))
            raise
        except Exception as e:
            logger.error("Error loading JSON {}: {}", filepath, str(e))
            raise

    def load_excel(
        self,
        filepath: Union[str, Path],
        sheet_name: Union[str, int, List, None] = 0,
        infer_types: bool = True,
        **kwargs,
    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Load Excel file into DataFrame.

        Args:
            filepath: Path to Excel file
            sheet_name: Sheet name, index, list of sheets, or None for all
            infer_types: Whether to infer optimal data types
            **kwargs: Additional arguments passed to pd.read_excel

        Returns:
            DataFrame or dict of DataFrames (if multiple sheets)

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If Excel file is invalid
        """
        filepath = Path(filepath)
        logger.info("Loading Excel file: {}, sheet: {}", filepath, sheet_name)

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        try:
            df = pd.read_excel(filepath, sheet_name=sheet_name, **kwargs)

            if isinstance(df, dict):
                logger.info("Loaded {} sheets from Excel", len(df))
                if infer_types:
                    df = {name: self._infer_types(sheet) for name, sheet in df.items()}
            else:
                logger.info("Loaded Excel: {} rows, {} columns", len(df), len(df.columns))
                if infer_types:
                    df = self._infer_types(df)

            return df

        except Exception as e:
            logger.error("Error loading Excel {}: {}", filepath, str(e))
            raise

    def load_sql(
        self,
        query: str,
        connection_string: str,
        params: Optional[Dict] = None,
        infer_types: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Load data from SQL database.

        Args:
            query: SQL query string
            connection_string: Database connection string
            params: Query parameters for parameterized queries
            infer_types: Whether to infer optimal data types
            **kwargs: Additional arguments passed to pd.read_sql

        Returns:
            Query results as DataFrame

        Raises:
            ValueError: If query or connection is invalid
        """
        logger.info("Executing SQL query on database")

        try:
            engine = self._get_engine(connection_string)

            with engine.connect() as conn:
                df = pd.read_sql(text(query), conn, params=params, **kwargs)

            logger.info("Loaded SQL result: {} rows, {} columns", len(df), len(df.columns))

            if infer_types:
                df = self._infer_types(df)

            return df

        except Exception as e:
            logger.error("Error executing SQL query: {}", str(e))
            raise

    def load_table(
        self,
        table_name: str,
        connection_string: str,
        schema: Optional[str] = None,
        columns: Optional[List[str]] = None,
        infer_types: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Load entire table from database.

        Args:
            table_name: Name of the table
            connection_string: Database connection string
            schema: Database schema name
            columns: Specific columns to load (None for all)
            infer_types: Whether to infer optimal data types
            **kwargs: Additional arguments passed to pd.read_sql_table

        Returns:
            Table data as DataFrame

        Raises:
            ValueError: If table doesn't exist
        """
        logger.info("Loading table: {}.{}", schema or "public", table_name)

        try:
            engine = self._get_engine(connection_string)

            df = pd.read_sql_table(
                table_name, engine, schema=schema, columns=columns, **kwargs
            )

            logger.info("Loaded table: {} rows, {} columns", len(df), len(df.columns))

            if infer_types:
                df = self._infer_types(df)

            return df

        except Exception as e:
            logger.error("Error loading table {}: {}", table_name, str(e))
            raise

    def load_sql_chunked(
        self,
        query: str,
        connection_string: str,
        chunk_size: Optional[int] = None,
        params: Optional[Dict] = None,
        infer_types: bool = True,
    ) -> Iterator[pd.DataFrame]:
        """
        Load SQL query results in chunks.

        Args:
            query: SQL query string
            connection_string: Database connection string
            chunk_size: Number of rows per chunk
            params: Query parameters
            infer_types: Whether to infer optimal data types

        Yields:
            DataFrame chunks

        Raises:
            ValueError: If query or connection is invalid
        """
        chunk_size = chunk_size or self.config.chunk_size
        logger.info("Executing chunked SQL query, chunk_size={}", chunk_size)

        try:
            engine = self._get_engine(connection_string)

            with engine.connect() as conn:
                chunks = pd.read_sql(
                    text(query), conn, params=params, chunksize=chunk_size
                )

                for i, chunk in enumerate(chunks):
                    logger.debug("Processing chunk {}: {} rows", i, len(chunk))
                    if infer_types:
                        chunk = self._infer_types(chunk)
                    yield chunk

        except Exception as e:
            logger.error("Error executing chunked SQL query: {}", str(e))
            raise

    def get_table_info(
        self, connection_string: str, table_name: str, schema: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get metadata about a database table.

        Args:
            connection_string: Database connection string
            table_name: Name of the table
            schema: Database schema name

        Returns:
            Dictionary with table metadata

        Raises:
            ValueError: If table doesn't exist
        """
        logger.info("Getting table info for: {}.{}", schema or "public", table_name)

        try:
            engine = self._get_engine(connection_string)
            inspector = inspect(engine)

            # Check if table exists
            if schema:
                tables = inspector.get_table_names(schema=schema)
            else:
                tables = inspector.get_table_names()

            if table_name not in tables:
                raise ValueError(f"Table '{table_name}' not found in schema '{schema}'")

            # Get column info
            columns = inspector.get_columns(table_name, schema=schema)

            # Get primary keys
            pk_constraint = inspector.get_pk_constraint(table_name, schema=schema)

            # Get foreign keys
            foreign_keys = inspector.get_foreign_keys(table_name, schema=schema)

            # Get indexes
            indexes = inspector.get_indexes(table_name, schema=schema)

            info = {
                "table_name": table_name,
                "schema": schema,
                "columns": columns,
                "primary_key": pk_constraint,
                "foreign_keys": foreign_keys,
                "indexes": indexes,
            }

            logger.info("Retrieved metadata for table: {}", table_name)
            return info

        except Exception as e:
            logger.error("Error getting table info for {}: {}", table_name, str(e))
            raise

    def list_tables(
        self, connection_string: str, schema: Optional[str] = None
    ) -> List[str]:
        """
        List all tables in database.

        Args:
            connection_string: Database connection string
            schema: Database schema name

        Returns:
            List of table names

        Raises:
            ValueError: If connection fails
        """
        logger.info("Listing tables in schema: {}", schema or "default")

        try:
            engine = self._get_engine(connection_string)
            inspector = inspect(engine)

            if schema:
                tables = inspector.get_table_names(schema=schema)
            else:
                tables = inspector.get_table_names()

            logger.info("Found {} tables", len(tables))
            return tables

        except Exception as e:
            logger.error("Error listing tables: {}", str(e))
            raise

    def _get_engine(self, connection_string: str):
        """
        Get or create SQLAlchemy engine with connection pooling.

        Args:
            connection_string: Database connection string

        Returns:
            SQLAlchemy engine
        """
        # Use connection string as cache key
        if connection_string not in self._engines:
            logger.debug("Creating new database engine")

            # Parse connection string to determine database type
            parsed = urlparse(connection_string)
            db_type = parsed.scheme.split("+")[0]

            # Configure pooling based on database type
            if db_type == "sqlite":
                # SQLite doesn't support connection pooling
                engine = create_engine(connection_string, poolclass=NullPool)
            else:
                engine = create_engine(
                    connection_string,
                    poolclass=QueuePool,
                    pool_size=self.config.pool_size,
                    pool_recycle=self.config.pool_recycle,
                    pool_pre_ping=True,  # Verify connections before using
                )

            self._engines[connection_string] = engine
            logger.debug("Engine created and cached")

        return self._engines[connection_string]

    def _infer_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Infer and optimize data types for DataFrame.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with optimized types
        """
        logger.debug("Inferring optimal data types")

        df = df.copy()

        for col in df.columns:
            col_type = df[col].dtype

            # Skip if already optimal
            if col_type in [np.bool_, "category"]:
                continue

            # Handle object columns
            if col_type == "object":
                # Try to convert to numeric
                try:
                    df[col] = pd.to_numeric(df[col])
                    continue
                except (ValueError, TypeError):
                    pass

                # Try to convert to datetime
                try:
                    df[col] = pd.to_datetime(df[col])
                    continue
                except (ValueError, TypeError):
                    pass

                # Convert to category if low cardinality
                if df[col].nunique() / len(df) < 0.5:
                    df[col] = df[col].astype("category")
                    continue

            # Optimize numeric types
            elif np.issubdtype(col_type, np.integer):
                df[col] = pd.to_numeric(df[col], downcast="integer")
            elif np.issubdtype(col_type, np.floating):
                df[col] = pd.to_numeric(df[col], downcast="float")

        logger.debug("Type inference complete")
        return df

    @contextmanager
    def transaction(self, connection_string: str):
        """
        Context manager for database transactions.

        Args:
            connection_string: Database connection string

        Yields:
            Database connection

        Example:
            >>> loader = DataLoader()
            >>> with loader.transaction(conn_str) as conn:
            ...     conn.execute("INSERT INTO ...")
            ...     conn.execute("UPDATE ...")
        """
        engine = self._get_engine(connection_string)
        conn = engine.connect()
        trans = conn.begin()

        try:
            yield conn
            trans.commit()
            logger.info("Transaction committed successfully")
        except Exception as e:
            trans.rollback()
            logger.error("Transaction rolled back: {}", str(e))
            raise
        finally:
            conn.close()

    def close_all_connections(self):
        """Close all database connections and dispose engines."""
        logger.info("Closing all database connections")

        for conn_str, engine in self._engines.items():
            try:
                engine.dispose()
                logger.debug("Disposed engine for: {}", conn_str[:50])
            except Exception as e:
                logger.warning("Error disposing engine: {}", str(e))

        self._engines.clear()
        logger.info("All connections closed")

    def __del__(self):
        """Cleanup on deletion."""
        self.close_all_connections()


# Utility functions
def detect_delimiter(filepath: Union[str, Path], sample_lines: int = 5) -> str:
    """
    Detect the delimiter used in a CSV file.

    Args:
        filepath: Path to the file
        sample_lines: Number of lines to sample

    Returns:
        Detected delimiter character
    """
    filepath = Path(filepath)

    with open(filepath, "r", encoding="utf-8") as f:
        sample = [f.readline() for _ in range(sample_lines)]

    # Try common delimiters
    delimiters = [",", ";", "\t", "|"]
    delimiter_counts = {}

    for delimiter in delimiters:
        counts = [line.count(delimiter) for line in sample if line.strip()]
        if counts and len(set(counts)) == 1:  # Consistent count across lines
            delimiter_counts[delimiter] = counts[0]

    if delimiter_counts:
        # Return delimiter with highest consistent count
        return max(delimiter_counts, key=delimiter_counts.get)

    return ","  # Default to comma


def get_file_info(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Get information about a file.

    Args:
        filepath: Path to the file

    Returns:
        Dictionary with file information
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    info = {
        "path": str(filepath.absolute()),
        "name": filepath.name,
        "extension": filepath.suffix,
        "size_bytes": filepath.stat().st_size,
        "size_mb": filepath.stat().st_size / (1024 * 1024),
        "modified": pd.Timestamp.fromtimestamp(filepath.stat().st_mtime),
    }

    return info
