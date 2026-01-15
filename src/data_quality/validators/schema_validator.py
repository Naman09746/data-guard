"""
Schema validator for data quality checks.

Validates data structure, column names, data types, and constraints
against expected schema definitions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import pandas as pd

from src.core.logging_config import get_logger
from src.data_quality.validators.base import (
    BaseValidator,
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
    ValidationStatus,
)

logger = get_logger(__name__)


@dataclass
class ColumnSchema:
    """Schema definition for a single column."""

    name: str
    dtype: str | list[str]  # Expected pandas dtype(s)
    nullable: bool = True
    unique: bool = False
    min_value: float | int | None = None
    max_value: float | int | None = None
    allowed_values: list[Any] | None = None
    pattern: str | None = None  # Regex pattern for string columns
    description: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ColumnSchema:
        """Create ColumnSchema from dictionary."""
        return cls(
            name=data["name"],
            dtype=data.get("dtype", "object"),
            nullable=data.get("nullable", True),
            unique=data.get("unique", False),
            min_value=data.get("min_value"),
            max_value=data.get("max_value"),
            allowed_values=data.get("allowed_values"),
            pattern=data.get("pattern"),
            description=data.get("description"),
        )


@dataclass
class DataFrameSchema:
    """Schema definition for an entire DataFrame."""

    columns: list[ColumnSchema]
    allow_extra_columns: bool = False
    strict_column_order: bool = False
    min_rows: int = 0
    max_rows: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DataFrameSchema:
        """Create DataFrameSchema from dictionary."""
        columns = [
            ColumnSchema.from_dict(col) if isinstance(col, dict) else col
            for col in data.get("columns", [])
        ]
        return cls(
            columns=columns,
            allow_extra_columns=data.get("allow_extra_columns", False),
            strict_column_order=data.get("strict_column_order", False),
            min_rows=data.get("min_rows", 0),
            max_rows=data.get("max_rows"),
        )

    @classmethod
    def infer_from_dataframe(cls, df: pd.DataFrame) -> DataFrameSchema:
        """Infer schema from a DataFrame."""
        columns = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            nullable = df[col].isna().any()
            unique = df[col].nunique() == len(df[col].dropna())

            columns.append(ColumnSchema(
                name=str(col),
                dtype=dtype,
                nullable=nullable,
                unique=unique,
            ))

        return cls(columns=columns, allow_extra_columns=False)


class SchemaValidator(BaseValidator[pd.DataFrame]):
    """
    Validates DataFrame structure and data types against a schema.
    
    Features:
    - Column presence and ordering validation
    - Data type checking with coercion support
    - Nullable and uniqueness constraints
    - Value range validation
    - Pattern matching for string columns
    - Extra column detection
    
    Edge cases handled:
    - Empty DataFrames
    - Mixed types in columns
    - Unicode column names
    - Nested/complex types
    """

    # Mapping of common dtype aliases
    DTYPE_ALIASES: dict[str, list[str]] = {
        "int": ["int8", "int16", "int32", "int64", "Int8", "Int16", "Int32", "Int64"],
        "float": ["float16", "float32", "float64", "Float32", "Float64"],
        "string": ["object", "string", "str"],
        "bool": ["bool", "boolean"],
        "datetime": ["datetime64[ns]", "datetime64", "datetime64[ns, UTC]"],
        "category": ["category"],
        "numeric": ["int8", "int16", "int32", "int64", "float16", "float32", "float64",
                   "Int8", "Int16", "Int32", "Int64", "Float32", "Float64"],
    }

    def __init__(
        self,
        schema: DataFrameSchema | dict[str, Any] | None = None,
        coerce_types: bool = False,
        name: str | None = None,
    ) -> None:
        """
        Initialize the schema validator.
        
        Args:
            schema: Expected schema (DataFrameSchema, dict, or None for inference).
            coerce_types: Whether to attempt type coercion before validation.
            name: Validator name.
        """
        super().__init__(name=name or "SchemaValidator")

        if isinstance(schema, dict):
            self.schema = DataFrameSchema.from_dict(schema)
        else:
            self.schema = schema

        self.coerce_types = coerce_types

    def validate(
        self,
        data: pd.DataFrame,
        schema: DataFrameSchema | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ValidationResult:
        """
        Validate DataFrame against schema.
        
        Args:
            data: DataFrame to validate.
            schema: Override schema for this validation.
            **kwargs: Additional validation options.
        
        Returns:
            ValidationResult with schema validation results.
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

            # Determine schema to use
            effective_schema = schema or self.schema
            if isinstance(effective_schema, dict):
                effective_schema = DataFrameSchema.from_dict(effective_schema)

            if effective_schema is None:
                return ValidationResult(
                    validator_name=self.name,
                    status=ValidationStatus.SKIPPED,
                    issues=[ValidationIssue(
                        message="No schema provided for validation",
                        severity=ValidationSeverity.INFO,
                    )],
                    duration_seconds=perf_counter() - start_time,
                )

            # Perform validations
            issues.extend(self._validate_row_count(data, effective_schema))
            issues.extend(self._validate_columns(data, effective_schema))
            issues.extend(self._validate_dtypes(data, effective_schema))
            issues.extend(self._validate_constraints(data, effective_schema))

            # Collect metrics
            metrics = {
                "total_columns": len(data.columns),
                "expected_columns": len(effective_schema.columns),
                "row_count": len(data),
                "schema_columns_validated": len(effective_schema.columns),
            }

            duration = perf_counter() - start_time
            self._logger.info(
                "schema_validation_complete",
                issues_found=len(issues),
                duration=round(duration, 4),
            )

            return self._create_result(issues, metrics, duration)

        except Exception as e:
            return self._handle_exception(e, "schema_validation")

    def _validate_row_count(
        self,
        data: pd.DataFrame,
        schema: DataFrameSchema,
    ) -> list[ValidationIssue]:
        """Validate row count constraints."""
        issues = []
        row_count = len(data)

        if row_count < schema.min_rows:
            issues.append(ValidationIssue(
                message=f"DataFrame has {row_count} rows, minimum required is {schema.min_rows}",
                severity=ValidationSeverity.ERROR,
                details={"row_count": row_count, "min_rows": schema.min_rows},
            ))

        if schema.max_rows is not None and row_count > schema.max_rows:
            issues.append(ValidationIssue(
                message=f"DataFrame has {row_count} rows, maximum allowed is {schema.max_rows}",
                severity=ValidationSeverity.ERROR,
                details={"row_count": row_count, "max_rows": schema.max_rows},
            ))

        return issues

    def _validate_columns(
        self,
        data: pd.DataFrame,
        schema: DataFrameSchema,
    ) -> list[ValidationIssue]:
        """Validate column presence and ordering."""
        issues = []
        actual_columns = set(data.columns)
        expected_columns = {col.name for col in schema.columns}

        # Check for missing columns
        missing = expected_columns - actual_columns
        if missing:
            issues.append(ValidationIssue(
                message=f"Missing required columns: {sorted(missing)}",
                severity=ValidationSeverity.ERROR,
                details={"missing_columns": sorted(missing)},
            ))

        # Check for extra columns
        extra = actual_columns - expected_columns
        if extra and not schema.allow_extra_columns:
            issues.append(ValidationIssue(
                message=f"Unexpected columns found: {sorted(extra)}",
                severity=ValidationSeverity.WARNING,
                details={"extra_columns": sorted(extra)},
            ))

        # Check column order if strict
        if schema.strict_column_order:
            expected_order = [col.name for col in schema.columns]
            actual_order = [c for c in data.columns if c in expected_columns]
            if actual_order != expected_order[:len(actual_order)]:
                issues.append(ValidationIssue(
                    message="Column order does not match expected schema",
                    severity=ValidationSeverity.WARNING,
                    details={
                        "expected_order": expected_order,
                        "actual_order": list(data.columns),
                    },
                ))

        return issues

    def _validate_dtypes(
        self,
        data: pd.DataFrame,
        schema: DataFrameSchema,
    ) -> list[ValidationIssue]:
        """Validate column data types."""
        issues = []

        for col_schema in schema.columns:
            if col_schema.name not in data.columns:
                continue

            actual_dtype = str(data[col_schema.name].dtype)
            expected_dtypes = (
                col_schema.dtype
                if isinstance(col_schema.dtype, list)
                else [col_schema.dtype]
            )

            # Expand dtype aliases
            expanded_expected = set()
            for dtype in expected_dtypes:
                if dtype in self.DTYPE_ALIASES:
                    expanded_expected.update(self.DTYPE_ALIASES[dtype])
                else:
                    expanded_expected.add(dtype)

            if actual_dtype not in expanded_expected:
                issues.append(ValidationIssue(
                    message=f"Column '{col_schema.name}' has dtype '{actual_dtype}', "
                    f"expected one of {sorted(expanded_expected)}",
                    severity=ValidationSeverity.ERROR,
                    column=col_schema.name,
                    details={
                        "actual_dtype": actual_dtype,
                        "expected_dtypes": sorted(expanded_expected),
                    },
                ))

        return issues

    def _validate_constraints(
        self,
        data: pd.DataFrame,
        schema: DataFrameSchema,
    ) -> list[ValidationIssue]:
        """Validate column constraints (nullable, unique, ranges, patterns)."""
        issues = []

        for col_schema in schema.columns:
            if col_schema.name not in data.columns:
                continue

            column = data[col_schema.name]

            # Nullable constraint
            if not col_schema.nullable and column.isna().any():
                null_count = column.isna().sum()
                null_indices = column[column.isna()].index.tolist()
                issues.append(ValidationIssue(
                    message=f"Column '{col_schema.name}' contains {null_count} null values "
                    "but is marked as non-nullable",
                    severity=ValidationSeverity.ERROR,
                    column=col_schema.name,
                    row_indices=null_indices,
                    details={"null_count": int(null_count)},
                ))

            # Uniqueness constraint
            if col_schema.unique:
                duplicates = column[column.duplicated(keep=False)]
                if len(duplicates) > 0:
                    dup_indices = duplicates.index.tolist()
                    issues.append(ValidationIssue(
                        message=f"Column '{col_schema.name}' should be unique but has "
                        f"{len(duplicates)} duplicate values",
                        severity=ValidationSeverity.ERROR,
                        column=col_schema.name,
                        row_indices=dup_indices,
                        details={"duplicate_count": len(duplicates)},
                    ))

            # Value range validation (for numeric columns)
            if col_schema.min_value is not None or col_schema.max_value is not None:
                issues.extend(self._validate_range(column, col_schema))

            # Allowed values validation
            if col_schema.allowed_values is not None:
                issues.extend(self._validate_allowed_values(column, col_schema))

            # Pattern validation (for string columns)
            if col_schema.pattern is not None:
                issues.extend(self._validate_pattern(column, col_schema))

        return issues

    def _validate_range(
        self,
        column: pd.Series,
        col_schema: ColumnSchema,
    ) -> list[ValidationIssue]:
        """Validate numeric range constraints."""
        issues = []

        if not pd.api.types.is_numeric_dtype(column):
            return issues

        non_null = column.dropna()

        if col_schema.min_value is not None:
            below_min = non_null[non_null < col_schema.min_value]
            if len(below_min) > 0:
                issues.append(ValidationIssue(
                    message=f"Column '{col_schema.name}' has {len(below_min)} values "
                    f"below minimum {col_schema.min_value}",
                    severity=ValidationSeverity.ERROR,
                    column=col_schema.name,
                    row_indices=below_min.index.tolist(),
                    details={
                        "min_allowed": col_schema.min_value,
                        "actual_min": float(non_null.min()),
                        "violation_count": len(below_min),
                    },
                ))

        if col_schema.max_value is not None:
            above_max = non_null[non_null > col_schema.max_value]
            if len(above_max) > 0:
                issues.append(ValidationIssue(
                    message=f"Column '{col_schema.name}' has {len(above_max)} values "
                    f"above maximum {col_schema.max_value}",
                    severity=ValidationSeverity.ERROR,
                    column=col_schema.name,
                    row_indices=above_max.index.tolist(),
                    details={
                        "max_allowed": col_schema.max_value,
                        "actual_max": float(non_null.max()),
                        "violation_count": len(above_max),
                    },
                ))

        return issues

    def _validate_allowed_values(
        self,
        column: pd.Series,
        col_schema: ColumnSchema,
    ) -> list[ValidationIssue]:
        """Validate allowed values constraint."""
        issues = []
        allowed = set(col_schema.allowed_values or [])

        non_null = column.dropna()
        invalid = non_null[~non_null.isin(allowed)]

        if len(invalid) > 0:
            invalid_values = invalid.unique().tolist()[:10]  # Limit to 10
            issues.append(ValidationIssue(
                message=f"Column '{col_schema.name}' has {len(invalid)} values "
                f"not in allowed set",
                severity=ValidationSeverity.ERROR,
                column=col_schema.name,
                row_indices=invalid.index.tolist(),
                details={
                    "allowed_values": list(allowed)[:20],
                    "invalid_values_sample": invalid_values,
                    "violation_count": len(invalid),
                },
            ))

        return issues

    def _validate_pattern(
        self,
        column: pd.Series,
        col_schema: ColumnSchema,
    ) -> list[ValidationIssue]:
        """Validate regex pattern constraint."""
        issues = []

        if not pd.api.types.is_string_dtype(column) and column.dtype != object:
            return issues

        try:
            pattern = re.compile(col_schema.pattern or "")
            non_null = column.dropna().astype(str)
            invalid = non_null[~non_null.str.match(pattern, na=False)]

            if len(invalid) > 0:
                invalid_samples = invalid.head(5).tolist()
                issues.append(ValidationIssue(
                    message=f"Column '{col_schema.name}' has {len(invalid)} values "
                    f"not matching pattern '{col_schema.pattern}'",
                    severity=ValidationSeverity.ERROR,
                    column=col_schema.name,
                    row_indices=invalid.index.tolist(),
                    details={
                        "pattern": col_schema.pattern,
                        "invalid_samples": invalid_samples,
                        "violation_count": len(invalid),
                    },
                ))
        except re.error as e:
            issues.append(ValidationIssue(
                message=f"Invalid regex pattern for column '{col_schema.name}': {e}",
                severity=ValidationSeverity.ERROR,
                column=col_schema.name,
                details={"pattern": col_schema.pattern, "error": str(e)},
            ))

        return issues

    def infer_schema(self, data: pd.DataFrame) -> DataFrameSchema:
        """
        Infer schema from a DataFrame.
        
        Args:
            data: DataFrame to analyze.
        
        Returns:
            Inferred DataFrameSchema.
        """
        return DataFrameSchema.infer_from_dataframe(data)
