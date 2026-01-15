"""
Custom exception classes for the Data Quality & Leakage Detection System.

This module defines a hierarchy of exceptions for precise error handling
and informative error messages throughout the application.
"""

from __future__ import annotations

from typing import Any


class BaseApplicationError(Exception):
    """
    Base exception for all application errors.
    
    Provides consistent error structure with error codes, details,
    and suggestions for resolution.
    """

    def __init__(
        self,
        message: str,
        error_code: str | None = None,
        details: dict[str, Any] | None = None,
        suggestion: str | None = None,
    ) -> None:
        """
        Initialize the base exception.
        
        Args:
            message: Human-readable error message.
            error_code: Machine-readable error code (e.g., "DQ001").
            details: Additional context about the error.
            suggestion: Suggested action to resolve the error.
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.suggestion = suggestion

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        result: dict[str, Any] = {
            "error": self.__class__.__name__,
            "message": self.message,
        }
        if self.error_code:
            result["error_code"] = self.error_code
        if self.details:
            result["details"] = self.details
        if self.suggestion:
            result["suggestion"] = self.suggestion
        return result

    def __repr__(self) -> str:
        """Return detailed string representation."""
        parts = [f"{self.__class__.__name__}({self.message!r}"]
        if self.error_code:
            parts.append(f", error_code={self.error_code!r}")
        if self.details:
            parts.append(f", details={self.details!r}")
        parts.append(")")
        return "".join(parts)


# =============================================================================
# Configuration Errors
# =============================================================================


class ConfigurationError(BaseApplicationError):
    """Raised when there is a configuration issue."""

    def __init__(
        self,
        message: str,
        config_key: str | None = None,
        expected_type: str | None = None,
        actual_value: Any = None,
        **kwargs: Any,
    ) -> None:
        details = kwargs.pop("details", {})
        if config_key:
            details["config_key"] = config_key
        if expected_type:
            details["expected_type"] = expected_type
        if actual_value is not None:
            details["actual_value"] = str(actual_value)
        super().__init__(message, details=details, **kwargs)


class InvalidConfigFileError(ConfigurationError):
    """Raised when a configuration file is invalid or cannot be parsed."""

    def __init__(self, filepath: str, parse_error: str | None = None) -> None:
        details = {"filepath": filepath}
        if parse_error:
            details["parse_error"] = parse_error
        super().__init__(
            message=f"Invalid configuration file: {filepath}",
            error_code="CFG001",
            details=details,
            suggestion="Check the file syntax and ensure it's valid YAML/JSON.",
        )


class MissingConfigError(ConfigurationError):
    """Raised when a required configuration is missing."""

    def __init__(self, config_key: str) -> None:
        super().__init__(
            message=f"Missing required configuration: {config_key}",
            error_code="CFG002",
            config_key=config_key,
            suggestion=f"Set the '{config_key}' configuration or environment variable.",
        )


# =============================================================================
# Data Quality Errors
# =============================================================================


class DataQualityError(BaseApplicationError):
    """Base exception for data quality validation errors."""

    def __init__(
        self,
        message: str,
        column: str | None = None,
        row_indices: list[int] | None = None,
        **kwargs: Any,
    ) -> None:
        details = kwargs.pop("details", {})
        if column:
            details["column"] = column
        if row_indices:
            details["row_indices"] = row_indices[:10]  # Limit to first 10
            details["total_affected_rows"] = len(row_indices)
        super().__init__(message, details=details, **kwargs)


class SchemaValidationError(DataQualityError):
    """Raised when data fails schema validation."""

    def __init__(
        self,
        message: str,
        expected_schema: dict[str, Any] | None = None,
        actual_schema: dict[str, Any] | None = None,
        mismatches: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        details = kwargs.pop("details", {})
        if expected_schema:
            details["expected_schema"] = expected_schema
        if actual_schema:
            details["actual_schema"] = actual_schema
        if mismatches:
            details["mismatches"] = mismatches
        super().__init__(
            message,
            error_code="DQ001",
            details=details,
            suggestion="Ensure data matches the expected schema.",
            **kwargs,
        )


class CompletenessError(DataQualityError):
    """Raised when data fails completeness checks."""

    def __init__(
        self,
        column: str,
        missing_ratio: float,
        threshold: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            message=f"Column '{column}' has {missing_ratio:.2%} missing values, "
            f"exceeding threshold of {threshold:.2%}",
            error_code="DQ002",
            column=column,
            details={
                "missing_ratio": missing_ratio,
                "threshold": threshold,
            },
            suggestion="Fill missing values or adjust the completeness threshold.",
            **kwargs,
        )


class ConsistencyError(DataQualityError):
    """Raised when data fails consistency checks."""

    def __init__(
        self,
        message: str,
        rule_name: str | None = None,
        violations: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        details = kwargs.pop("details", {})
        if rule_name:
            details["rule_name"] = rule_name
        if violations:
            details["violations"] = violations[:10]  # Limit to first 10
            details["total_violations"] = len(violations)
        super().__init__(
            message,
            error_code="DQ003",
            details=details,
            suggestion="Review and correct the inconsistent data.",
            **kwargs,
        )


class AccuracyError(DataQualityError):
    """Raised when data fails accuracy validation."""

    def __init__(
        self,
        column: str,
        validation_type: str,
        invalid_count: int,
        total_count: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            message=f"Column '{column}' has {invalid_count}/{total_count} invalid values "
            f"({validation_type} validation)",
            error_code="DQ004",
            column=column,
            details={
                "validation_type": validation_type,
                "invalid_count": invalid_count,
                "total_count": total_count,
                "invalid_ratio": invalid_count / total_count if total_count > 0 else 0,
            },
            suggestion="Review and correct invalid values or adjust validation rules.",
            **kwargs,
        )


class TimelinessError(DataQualityError):
    """Raised when data fails timeliness checks."""

    def __init__(
        self,
        message: str,
        data_age_hours: float | None = None,
        max_age_hours: float | None = None,
        **kwargs: Any,
    ) -> None:
        details = kwargs.pop("details", {})
        if data_age_hours is not None:
            details["data_age_hours"] = data_age_hours
        if max_age_hours is not None:
            details["max_age_hours"] = max_age_hours
        super().__init__(
            message,
            error_code="DQ005",
            details=details,
            suggestion="Ensure data is refreshed within the acceptable time window.",
            **kwargs,
        )


# =============================================================================
# Leakage Detection Errors
# =============================================================================


class LeakageDetectionError(BaseApplicationError):
    """Base exception for leakage detection errors."""

    pass


class TrainTestContaminationError(LeakageDetectionError):
    """Raised when train-test contamination is detected."""

    def __init__(
        self,
        duplicate_count: int,
        total_test_count: int,
        duplicate_indices: list[int] | None = None,
        **kwargs: Any,
    ) -> None:
        contamination_rate = duplicate_count / total_test_count if total_test_count > 0 else 0
        details = {
            "duplicate_count": duplicate_count,
            "total_test_count": total_test_count,
            "contamination_rate": contamination_rate,
        }
        if duplicate_indices:
            details["duplicate_indices"] = duplicate_indices[:20]
        super().__init__(
            message=f"Train-test contamination detected: {duplicate_count} duplicates "
            f"({contamination_rate:.2%} of test set)",
            error_code="LD001",
            details=details,
            suggestion="Remove duplicate rows from train/test split or use group-aware splitting.",
            **kwargs,
        )


class TargetLeakageError(LeakageDetectionError):
    """Raised when potential target leakage is detected."""

    def __init__(
        self,
        feature: str,
        correlation: float,
        threshold: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            message=f"Potential target leakage detected: feature '{feature}' has "
            f"correlation of {correlation:.4f} with target (threshold: {threshold})",
            error_code="LD002",
            details={
                "feature": feature,
                "correlation": correlation,
                "threshold": threshold,
            },
            suggestion="Investigate if this feature contains future information or is derived from target.",
            **kwargs,
        )


class FeatureLeakageError(LeakageDetectionError):
    """Raised when suspicious feature leakage patterns are detected."""

    def __init__(
        self,
        features: list[str],
        pattern: str | None = None,
        reason: str | None = None,
        **kwargs: Any,
    ) -> None:
        details: dict[str, Any] = {"features": features}
        if pattern:
            details["matched_pattern"] = pattern
        if reason:
            details["reason"] = reason
        super().__init__(
            message=f"Suspicious feature leakage detected in {len(features)} feature(s)",
            error_code="LD003",
            details=details,
            suggestion="Review flagged features and ensure they don't contain information "
            "unavailable at prediction time.",
            **kwargs,
        )


class TemporalLeakageError(LeakageDetectionError):
    """Raised when temporal leakage is detected."""

    def __init__(
        self,
        message: str,
        leaked_rows: int | None = None,
        time_column: str | None = None,
        **kwargs: Any,
    ) -> None:
        details = kwargs.pop("details", {})
        if leaked_rows is not None:
            details["leaked_rows"] = leaked_rows
        if time_column:
            details["time_column"] = time_column
        super().__init__(
            message,
            error_code="LD004",
            details=details,
            suggestion="Ensure proper time-based train/test splits with no future data leakage.",
            **kwargs,
        )


# =============================================================================
# Validation Errors
# =============================================================================


class ValidationError(BaseApplicationError):
    """Raised when input validation fails."""

    def __init__(
        self,
        message: str,
        field: str | None = None,
        value: Any = None,
        **kwargs: Any,
    ) -> None:
        details = kwargs.pop("details", {})
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = str(value)[:100]  # Truncate long values
        super().__init__(
            message,
            error_code="VAL001",
            details=details,
            **kwargs,
        )


class EmptyDataError(ValidationError):
    """Raised when input data is empty."""

    def __init__(self, data_name: str = "data") -> None:
        super().__init__(
            message=f"The provided {data_name} is empty",
            error_code="VAL002",
            suggestion=f"Ensure {data_name} contains at least one row.",
        )


class InvalidDataTypeError(ValidationError):
    """Raised when data type is invalid."""

    def __init__(
        self,
        expected_type: str,
        actual_type: str,
        field: str | None = None,
    ) -> None:
        super().__init__(
            message=f"Expected {expected_type}, got {actual_type}",
            error_code="VAL003",
            field=field,
            details={
                "expected_type": expected_type,
                "actual_type": actual_type,
            },
        )


# =============================================================================
# Processing Errors
# =============================================================================


class ProcessingError(BaseApplicationError):
    """Raised when data processing fails."""

    pass


class MemoryLimitError(ProcessingError):
    """Raised when memory limit is exceeded."""

    def __init__(
        self,
        required_mb: float,
        available_mb: float,
        operation: str | None = None,
    ) -> None:
        super().__init__(
            message=f"Memory limit exceeded: {required_mb:.1f}MB required, "
            f"{available_mb:.1f}MB available",
            error_code="PROC001",
            details={
                "required_mb": required_mb,
                "available_mb": available_mb,
                "operation": operation,
            },
            suggestion="Reduce data size, increase memory limit, or use chunked processing.",
        )


class TimeoutError(ProcessingError):
    """Raised when an operation times out."""

    def __init__(
        self,
        operation: str,
        timeout_seconds: float,
    ) -> None:
        super().__init__(
            message=f"Operation '{operation}' timed out after {timeout_seconds}s",
            error_code="PROC002",
            details={
                "operation": operation,
                "timeout_seconds": timeout_seconds,
            },
            suggestion="Increase timeout or optimize the operation.",
        )
