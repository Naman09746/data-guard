"""
Base detector class for leakage detection.

Provides the abstract interface and common functionality for all detectors.
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


class LeakageSeverity(str, Enum):
    """Severity levels for leakage issues."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class DetectionStatus(str, Enum):
    """Status of a detection check."""

    CLEAN = "clean"
    DETECTED = "detected"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class LeakageIssue:
    """Represents a single leakage issue found during detection."""

    message: str
    severity: LeakageSeverity
    leakage_type: str
    affected_features: list[str] = field(default_factory=list)
    affected_rows: int = 0
    details: dict[str, Any] = field(default_factory=dict)
    recommendation: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert issue to dictionary."""
        result = {
            "message": self.message,
            "severity": self.severity.value,
            "leakage_type": self.leakage_type,
        }
        if self.affected_features:
            result["affected_features"] = self.affected_features
        if self.affected_rows > 0:
            result["affected_rows"] = self.affected_rows
        if self.details:
            result["details"] = self.details
        if self.recommendation:
            result["recommendation"] = self.recommendation
        return result


@dataclass
class DetectionResult:
    """Result of a single detection check."""

    detector_name: str
    status: DetectionStatus
    issues: list[LeakageIssue] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def has_leakage(self) -> bool:
        """Check if leakage was detected."""
        return self.status == DetectionStatus.DETECTED

    @property
    def is_clean(self) -> bool:
        """Check if no leakage was found."""
        return self.status == DetectionStatus.CLEAN

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "detector_name": self.detector_name,
            "status": self.status.value,
            "has_leakage": self.has_leakage,
            "issues": [issue.to_dict() for issue in self.issues],
            "metrics": self.metrics,
            "duration_seconds": round(self.duration_seconds, 4),
            "timestamp": self.timestamp.isoformat(),
        }


class BaseDetector(ABC, Generic[DataFrameType]):
    """
    Abstract base class for all leakage detectors.
    
    Provides common functionality including:
    - Configuration management
    - Logging integration
    - Result standardization
    - Error handling
    """

    def __init__(self, name: str | None = None, **config: Any) -> None:
        self.name = name or self.__class__.__name__
        self.config = config
        self._logger = get_logger(f"detector.{self.name}")

    @abstractmethod
    def detect(
        self,
        train_data: DataFrameType,
        test_data: DataFrameType | None = None,
        target_column: str | None = None,
        **kwargs: Any,
    ) -> DetectionResult:
        """
        Perform leakage detection.
        
        Args:
            train_data: Training data.
            test_data: Test data (if applicable).
            target_column: Target column name.
            **kwargs: Additional parameters.
        
        Returns:
            DetectionResult containing status and issues.
        """
        pass

    def _check_empty_data(self, data: pd.DataFrame, name: str) -> LeakageIssue | None:
        """Check if data is empty."""
        if data is None:
            return LeakageIssue(
                message=f"{name} is None",
                severity=LeakageSeverity.CRITICAL,
                leakage_type="data_error",
            )
        if len(data) == 0:
            return LeakageIssue(
                message=f"{name} is empty",
                severity=LeakageSeverity.WARNING,
                leakage_type="data_error",
            )
        return None

    def _handle_exception(self, e: Exception, operation: str) -> DetectionResult:
        """Handle exceptions during detection."""
        self._logger.error(
            "detection_error",
            detector=self.name,
            operation=operation,
            error=str(e),
            error_type=type(e).__name__,
        )
        return DetectionResult(
            detector_name=self.name,
            status=DetectionStatus.ERROR,
            issues=[
                LeakageIssue(
                    message=f"Detection failed: {e}",
                    severity=LeakageSeverity.CRITICAL,
                    leakage_type="error",
                )
            ],
        )

    def _create_result(
        self,
        issues: list[LeakageIssue],
        metrics: dict[str, Any] | None = None,
        duration: float = 0.0,
    ) -> DetectionResult:
        """Create detection result based on issues found."""
        if any(i.severity == LeakageSeverity.CRITICAL for i in issues) or any(i.severity == LeakageSeverity.WARNING for i in issues):
            status = DetectionStatus.DETECTED
        else:
            status = DetectionStatus.CLEAN

        return DetectionResult(
            detector_name=self.name,
            status=status,
            issues=issues,
            metrics=metrics or {},
            duration_seconds=duration,
        )
