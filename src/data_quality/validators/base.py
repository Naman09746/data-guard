"""
Base validator class for data quality checks.

Provides the abstract interface and common functionality for all validators.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Generic, TypeVar

import pandas as pd

from src.core.logging_config import get_logger

logger = get_logger(__name__)

DataFrameType = TypeVar("DataFrameType", pd.DataFrame, Any)


class ValidationSeverity(str, Enum):
    """Severity levels for validation issues."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationStatus(str, Enum):
    """Status of a validation check."""

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    WARNING = "warning"


@dataclass
class ValidationIssue:
    """Represents a single validation issue found during checks."""

    message: str
    severity: ValidationSeverity
    column: str | None = None
    row_indices: list[int] | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert issue to dictionary."""
        result = {
            "message": self.message,
            "severity": self.severity.value,
        }
        if self.column:
            result["column"] = self.column
        if self.row_indices:
            result["row_indices"] = self.row_indices[:10]
            result["total_affected"] = len(self.row_indices)
        if self.details:
            result["details"] = self.details
        return result


@dataclass
class ValidationResult:
    """Result of a single validation check."""

    validator_name: str
    status: ValidationStatus
    issues: list[ValidationIssue] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def passed(self) -> bool:
        """Check if validation passed."""
        return self.status == ValidationStatus.PASSED

    @property
    def has_errors(self) -> bool:
        """Check if there are any error-level issues."""
        return any(
            issue.severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL)
            for issue in self.issues
        )

    @property
    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return any(issue.severity == ValidationSeverity.WARNING for issue in self.issues)

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "validator_name": self.validator_name,
            "status": self.status.value,
            "passed": self.passed,
            "issues": [issue.to_dict() for issue in self.issues],
            "metrics": self.metrics,
            "duration_seconds": round(self.duration_seconds, 4),
            "timestamp": self.timestamp.isoformat(),
        }


class BaseValidator(ABC, Generic[DataFrameType]):
    """
    Abstract base class for all data quality validators.
    
    Provides common functionality including:
    - Configuration management
    - Logging integration
    - Result standardization
    - Edge case handling
    """

    def __init__(self, name: str | None = None, **config: Any) -> None:
        """
        Initialize the validator.
        
        Args:
            name: Validator name (defaults to class name).
            **config: Validator-specific configuration.
        """
        self.name = name or self.__class__.__name__
        self.config = config
        self._logger = get_logger(f"validator.{self.name}")

    @abstractmethod
    def validate(self, data: DataFrameType, **kwargs: Any) -> ValidationResult:
        """
        Perform validation on the input data.
        
        Args:
            data: Input data to validate.
            **kwargs: Additional validation parameters.
        
        Returns:
            ValidationResult containing status, issues, and metrics.
        """
        pass

    def _check_empty_data(self, data: pd.DataFrame) -> ValidationIssue | None:
        """
        Check if data is empty and return an issue if so.
        
        Args:
            data: Input DataFrame to check.
        
        Returns:
            ValidationIssue if data is empty, None otherwise.
        """
        if data is None:
            return ValidationIssue(
                message="Input data is None",
                severity=ValidationSeverity.CRITICAL,
            )
        if len(data) == 0:
            return ValidationIssue(
                message="Input data is empty (0 rows)",
                severity=ValidationSeverity.WARNING,
            )
        if len(data.columns) == 0:
            return ValidationIssue(
                message="Input data has no columns",
                severity=ValidationSeverity.ERROR,
            )
        return None

    def _check_single_row(self, data: pd.DataFrame) -> ValidationIssue | None:
        """
        Check if data has only one row (may affect statistical checks).
        
        Args:
            data: Input DataFrame to check.
        
        Returns:
            ValidationIssue if data has only one row, None otherwise.
        """
        if len(data) == 1:
            return ValidationIssue(
                message="Data has only 1 row; statistical validations may be unreliable",
                severity=ValidationSeverity.INFO,
            )
        return None

    def _handle_exception(
        self,
        e: Exception,
        operation: str = "validation",
    ) -> ValidationResult:
        """
        Handle exceptions during validation and return error result.
        
        Args:
            e: Exception that occurred.
            operation: Description of the operation that failed.
        
        Returns:
            ValidationResult with error status.
        """
        self._logger.error(
            "validation_error",
            validator=self.name,
            operation=operation,
            error=str(e),
            error_type=type(e).__name__,
        )
        return ValidationResult(
            validator_name=self.name,
            status=ValidationStatus.FAILED,
            issues=[
                ValidationIssue(
                    message=f"Validation failed: {e}",
                    severity=ValidationSeverity.CRITICAL,
                    details={"exception_type": type(e).__name__},
                )
            ],
        )

    def _create_result(
        self,
        issues: list[ValidationIssue],
        metrics: dict[str, Any] | None = None,
        duration: float = 0.0,
    ) -> ValidationResult:
        """
        Create a validation result based on issues found.
        
        Args:
            issues: List of validation issues.
            metrics: Validation metrics.
            duration: Validation duration in seconds.
        
        Returns:
            ValidationResult with appropriate status.
        """
        # Determine status based on issues
        if any(i.severity == ValidationSeverity.CRITICAL for i in issues) or any(i.severity == ValidationSeverity.ERROR for i in issues):
            status = ValidationStatus.FAILED
        elif any(i.severity == ValidationSeverity.WARNING for i in issues):
            status = ValidationStatus.WARNING
        else:
            status = ValidationStatus.PASSED

        return ValidationResult(
            validator_name=self.name,
            status=status,
            issues=issues,
            metrics=metrics or {},
            duration_seconds=duration,
        )
