"""
Completeness checker for data quality validation.

Detects and reports missing values with configurable thresholds
and various null detection strategies.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import pandas as pd

from src.core.config import get_settings
from src.core.logging_config import get_logger
from src.data_quality.validators.base import (
    BaseValidator,
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
)

logger = get_logger(__name__)


@dataclass
class CompletenessConfig:
    """Configuration for completeness checking."""

    # Global threshold for missing values (0.0 to 1.0)
    threshold: float = 0.05

    # Per-column thresholds (column_name -> threshold)
    column_thresholds: dict[str, float] | None = None

    # Values to consider as missing (besides null/NaN)
    missing_values: list[Any] | None = None

    # Whether to treat empty strings as missing
    empty_string_as_missing: bool = True

    # Whether to treat whitespace-only strings as missing
    whitespace_as_missing: bool = True

    # Columns to exclude from checking
    exclude_columns: list[str] | None = None

    # Minimum valid rows threshold (fail if too few valid rows)
    min_valid_rows_ratio: float = 0.1


class CompletenessChecker(BaseValidator[pd.DataFrame]):
    """
    Validates data completeness by checking for missing values.
    
    Features:
    - Configurable missing value thresholds (global and per-column)
    - Detection of null, NaN, empty strings, and whitespace
    - Custom missing value patterns (e.g., "N/A", "-", "null")
    - Row-level and column-level completeness metrics
    - Pattern analysis for systematic missingness
    
    Edge cases handled:
    - Empty DataFrames
    - All-null columns
    - Mixed null types (None, np.nan, pd.NA)
    - Implicit nulls (empty strings, sentinel values)
    """

    # Common sentinel values that may represent missing data
    DEFAULT_MISSING_VALUES: list[Any] = [
        "",
        "null",
        "NULL",
        "None",
        "none",
        "N/A",
        "n/a",
        "NA",
        "na",
        "NaN",
        "nan",
        "-",
        "--",
        ".",
        "?",
        "#N/A",
        "#NA",
        "#VALUE!",
        "#REF!",
        "(blank)",
        "(empty)",
    ]

    def __init__(
        self,
        config: CompletenessConfig | dict[str, Any] | None = None,
        name: str | None = None,
    ) -> None:
        """
        Initialize the completeness checker.
        
        Args:
            config: Completeness configuration.
            name: Validator name.
        """
        super().__init__(name=name or "CompletenessChecker")

        if isinstance(config, dict):
            self.config = CompletenessConfig(**config)
        elif config is None:
            settings = get_settings()
            self.config = CompletenessConfig(
                threshold=settings.quality.missing_value_threshold
            )
        else:
            self.config = config

    def validate(
        self,
        data: pd.DataFrame,
        threshold: float | None = None,
        columns: list[str] | None = None,
        **kwargs: Any,
    ) -> ValidationResult:
        """
        Validate data completeness.
        
        Args:
            data: DataFrame to validate.
            threshold: Override default missing value threshold.
            columns: Specific columns to check (None = all columns).
            **kwargs: Additional options.
        
        Returns:
            ValidationResult with completeness validation results.
        """
        start_time = perf_counter()
        issues: list[ValidationIssue] = []
        metrics: dict[str, Any] = {}

        try:
            # Check for empty data
            empty_issue = self._check_empty_data(data)
            if empty_issue:
                if empty_issue.severity == ValidationSeverity.CRITICAL:
                    return self._create_result([empty_issue], duration=perf_counter() - start_time)
                issues.append(empty_issue)
                # Continue with empty data handling
                return self._create_result(
                    issues,
                    metrics={"row_count": 0, "column_count": len(data.columns) if data is not None else 0},
                    duration=perf_counter() - start_time,
                )

            # Determine columns to check
            check_columns = columns or list(data.columns)
            if self.config.exclude_columns:
                check_columns = [c for c in check_columns if c not in self.config.exclude_columns]

            # Use provided threshold or config threshold
            effective_threshold = threshold if threshold is not None else self.config.threshold

            # Calculate completeness for each column
            column_metrics = {}
            for col in check_columns:
                if col not in data.columns:
                    issues.append(ValidationIssue(
                        message=f"Column '{col}' not found in data",
                        severity=ValidationSeverity.WARNING,
                        column=col,
                    ))
                    continue

                col_issues, col_metrics = self._check_column_completeness(
                    data[col],
                    col,
                    self.config.column_thresholds.get(col, effective_threshold)
                    if self.config.column_thresholds
                    else effective_threshold,
                )
                issues.extend(col_issues)
                column_metrics[col] = col_metrics

            # Calculate overall metrics
            total_cells = len(data) * len(check_columns)
            total_missing = sum(m.get("missing_count", 0) for m in column_metrics.values())
            overall_completeness = 1 - (total_missing / total_cells) if total_cells > 0 else 1.0

            # Check row-level completeness
            row_issues, row_metrics = self._check_row_completeness(data, check_columns)
            issues.extend(row_issues)

            # Pattern analysis for systematic missingness
            pattern_issues = self._analyze_missing_patterns(data, check_columns)
            issues.extend(pattern_issues)

            metrics = {
                "total_rows": len(data),
                "total_columns": len(check_columns),
                "total_cells": total_cells,
                "total_missing": int(total_missing),
                "overall_completeness": round(overall_completeness, 4),
                "column_metrics": {
                    col: {
                        "missing_count": int(m["missing_count"]),
                        "missing_ratio": round(m["missing_ratio"], 4),
                        "completeness": round(1 - m["missing_ratio"], 4),
                    }
                    for col, m in column_metrics.items()
                },
                **row_metrics,
            }

            duration = perf_counter() - start_time
            self._logger.info(
                "completeness_check_complete",
                overall_completeness=round(overall_completeness, 4),
                issues_found=len(issues),
                duration=round(duration, 4),
            )

            return self._create_result(issues, metrics, duration)

        except Exception as e:
            return self._handle_exception(e, "completeness_check")

    def _check_column_completeness(
        self,
        column: pd.Series,
        column_name: str,
        threshold: float,
    ) -> tuple[list[ValidationIssue], dict[str, Any]]:
        """Check completeness of a single column."""
        issues = []

        # Count standard missing values
        standard_missing = column.isna()

        # Check for additional missing value indicators
        additional_missing = pd.Series(False, index=column.index)

        # Empty strings
        if self.config.empty_string_as_missing and column.dtype == object:
            additional_missing |= column.astype(str).str.strip() == ""

        # Whitespace only
        if self.config.whitespace_as_missing and column.dtype == object:
            additional_missing |= column.astype(str).str.strip().str.len() == 0

        # Custom missing values
        if self.config.missing_values:
            additional_missing |= column.isin(self.config.missing_values)

        # Combine missing indicators
        is_missing = standard_missing | additional_missing
        missing_count = is_missing.sum()
        missing_ratio = missing_count / len(column) if len(column) > 0 else 0.0

        # Check against threshold
        if missing_ratio > threshold:
            missing_indices = column[is_missing].index.tolist()
            issues.append(ValidationIssue(
                message=f"Column '{column_name}' has {missing_ratio:.2%} missing values, "
                f"exceeding threshold of {threshold:.2%}",
                severity=ValidationSeverity.ERROR if missing_ratio > 0.5 else ValidationSeverity.WARNING,
                column=column_name,
                row_indices=missing_indices,
                details={
                    "missing_count": int(missing_count),
                    "total_count": len(column),
                    "missing_ratio": round(missing_ratio, 4),
                    "threshold": threshold,
                },
            ))

        # Check for all-null column
        if missing_ratio == 1.0:
            issues.append(ValidationIssue(
                message=f"Column '{column_name}' is entirely null/missing",
                severity=ValidationSeverity.CRITICAL,
                column=column_name,
            ))

        metrics = {
            "missing_count": missing_count,
            "missing_ratio": missing_ratio,
            "standard_nulls": int(standard_missing.sum()),
            "additional_missing": int(additional_missing.sum()),
        }

        return issues, metrics

    def _check_row_completeness(
        self,
        data: pd.DataFrame,
        columns: list[str],
    ) -> tuple[list[ValidationIssue], dict[str, Any]]:
        """Check row-level completeness."""
        issues = []

        subset = data[columns]

        # Calculate missing values per row
        missing_per_row = subset.isna().sum(axis=1)

        # Completely missing rows
        complete_missing_rows = (missing_per_row == len(columns))
        complete_missing_count = complete_missing_rows.sum()

        if complete_missing_count > 0:
            issues.append(ValidationIssue(
                message=f"{complete_missing_count} rows have all values missing",
                severity=ValidationSeverity.ERROR,
                row_indices=data[complete_missing_rows].index.tolist(),
                details={"completely_missing_rows": int(complete_missing_count)},
            ))

        # Check minimum valid rows
        valid_row_count = len(data) - complete_missing_count
        valid_ratio = valid_row_count / len(data) if len(data) > 0 else 0

        if valid_ratio < self.config.min_valid_rows_ratio:
            issues.append(ValidationIssue(
                message=f"Only {valid_ratio:.2%} of rows have valid data, "
                f"below minimum of {self.config.min_valid_rows_ratio:.2%}",
                severity=ValidationSeverity.CRITICAL,
                details={
                    "valid_rows": int(valid_row_count),
                    "total_rows": len(data),
                    "valid_ratio": round(valid_ratio, 4),
                },
            ))

        metrics = {
            "rows_with_all_missing": int(complete_missing_count),
            "rows_with_any_missing": int((missing_per_row > 0).sum()),
            "rows_complete": int((missing_per_row == 0).sum()),
            "avg_missing_per_row": round(missing_per_row.mean(), 2) if len(data) > 0 else 0,
        }

        return issues, metrics

    def _analyze_missing_patterns(
        self,
        data: pd.DataFrame,
        columns: list[str],
    ) -> list[ValidationIssue]:
        """Analyze patterns in missing data."""
        issues = []

        if len(data) < 10:  # Not enough data for pattern analysis
            return issues

        subset = data[columns]
        missing_mask = subset.isna()

        # Check for correlated missingness (columns that are often missing together)
        correlation_threshold = 0.8

        if len(columns) >= 2:
            for i, col1 in enumerate(columns[:-1]):
                for col2 in columns[i + 1:]:
                    if missing_mask[col1].sum() > 0 and missing_mask[col2].sum() > 0:
                        # Calculate correlation of missingness
                        both_missing = (missing_mask[col1] & missing_mask[col2]).sum()
                        either_missing = (missing_mask[col1] | missing_mask[col2]).sum()

                        if either_missing > 0:
                            correlation = both_missing / either_missing
                            if correlation > correlation_threshold:
                                issues.append(ValidationIssue(
                                    message=f"Columns '{col1}' and '{col2}' have correlated "
                                    f"missingness ({correlation:.0%})",
                                    severity=ValidationSeverity.INFO,
                                    details={
                                        "column_pair": [col1, col2],
                                        "correlation": round(correlation, 4),
                                    },
                                ))

        return issues

    def get_missing_summary(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Get a summary of missing values for each column.
        
        Args:
            data: DataFrame to analyze.
        
        Returns:
            DataFrame with missing value statistics.
        """
        summary_data = []

        for col in data.columns:
            missing_count = data[col].isna().sum()
            total = len(data)
            missing_ratio = missing_count / total if total > 0 else 0

            summary_data.append({
                "column": col,
                "dtype": str(data[col].dtype),
                "missing_count": missing_count,
                "total_count": total,
                "missing_ratio": round(missing_ratio, 4),
                "completeness": round(1 - missing_ratio, 4),
            })

        return pd.DataFrame(summary_data).sort_values(
            "missing_ratio", ascending=False
        ).reset_index(drop=True)
