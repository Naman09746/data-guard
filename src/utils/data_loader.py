"""
Data loading utilities.

Provides flexible data loading from various sources with validation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import pandas as pd

from src.core.exceptions import ValidationError
from src.core.logging_config import get_logger

logger = get_logger(__name__)


class DataLoader:
    """
    Utility class for loading data from various sources.
    
    Supports CSV, JSON, Parquet, and Excel files with
    automatic format detection and validation.
    """

    SUPPORTED_FORMATS = ["csv", "json", "parquet", "xlsx", "xls"]

    @classmethod
    def load(
        cls,
        source: str | Path,
        format: Literal["csv", "json", "parquet", "xlsx", "auto"] = "auto",
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Load data from a file.
        
        Args:
            source: Path to the data file.
            format: File format (auto-detect if not specified).
            **kwargs: Additional arguments passed to pandas reader.
        
        Returns:
            Loaded DataFrame.
        
        Raises:
            ValidationError: If file cannot be loaded.
        """
        path = Path(source)

        if not path.exists():
            raise ValidationError(f"File not found: {source}")

        # Auto-detect format
        if format == "auto":
            format = cls._detect_format(path)

        logger.info("loading_data", path=str(path), format=format)

        try:
            if format == "csv":
                df = pd.read_csv(path, **kwargs)
            elif format == "json":
                df = pd.read_json(path, **kwargs)
            elif format == "parquet":
                df = pd.read_parquet(path, **kwargs)
            elif format in ("xlsx", "xls"):
                df = pd.read_excel(path, **kwargs)
            else:
                raise ValidationError(f"Unsupported format: {format}")

            logger.info(
                "data_loaded",
                rows=len(df),
                columns=len(df.columns),
            )

            return df

        except Exception as e:
            raise ValidationError(f"Failed to load data: {e}")

    @classmethod
    def _detect_format(cls, path: Path) -> str:
        """Detect file format from extension."""
        suffix = path.suffix.lower().lstrip(".")

        if suffix in cls.SUPPORTED_FORMATS:
            return suffix
        elif suffix == "":
            # Try to detect from content
            with open(path, "rb") as f:
                header = f.read(4)
                if header.startswith(b"PAR1"):
                    return "parquet"
                elif header.startswith(b"PK"):
                    return "xlsx"
            return "csv"  # Default
        else:
            raise ValidationError(f"Unknown file format: {suffix}")

    @classmethod
    def load_train_test(
        cls,
        train_path: str | Path,
        test_path: str | Path,
        **kwargs: Any,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load train and test datasets.
        
        Args:
            train_path: Path to training data.
            test_path: Path to test data.
            **kwargs: Additional arguments passed to pandas reader.
        
        Returns:
            Tuple of (train_df, test_df).
        """
        train_df = cls.load(train_path, **kwargs)
        test_df = cls.load(test_path, **kwargs)
        return train_df, test_df

    @classmethod
    def sample_data(
        cls,
        df: pd.DataFrame,
        n: int | None = None,
        frac: float | None = None,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """
        Sample data from DataFrame.
        
        Args:
            df: Source DataFrame.
            n: Number of rows to sample.
            frac: Fraction of rows to sample.
            random_state: Random seed.
        
        Returns:
            Sampled DataFrame.
        """
        if n is not None and n >= len(df):
            return df

        return df.sample(n=n, frac=frac, random_state=random_state)
