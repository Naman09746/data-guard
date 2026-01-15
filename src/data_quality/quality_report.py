"""
Quality Report generation and formatting.

Provides comprehensive reporting for data quality validation results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from src.data_quality.validators.base import ValidationResult, ValidationStatus


class QualityStatus(str, Enum):
    """Overall quality status."""

    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class QualityReport:
    """Comprehensive data quality report."""

    status: QualityStatus
    validation_results: list[ValidationResult]
    total_issues: int
    duration_seconds: float
    timestamp: datetime
    data_shape: tuple[int, int]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """Check if overall validation passed."""
        return self.status == QualityStatus.PASSED

    @property
    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return any(r.has_warnings for r in self.validation_results)

    @property
    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return any(r.has_errors for r in self.validation_results)

    def get_issues_by_severity(self, severity: str) -> list[dict[str, Any]]:
        """Get all issues of a specific severity."""
        issues = []
        for result in self.validation_results:
            for issue in result.issues:
                if issue.severity.value == severity:
                    issues.append(issue.to_dict())
        return issues

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of the quality report."""
        validators_passed = sum(1 for r in self.validation_results if r.passed)
        validators_failed = sum(
            1 for r in self.validation_results
            if r.status == ValidationStatus.FAILED
        )
        validators_warned = sum(
            1 for r in self.validation_results
            if r.status == ValidationStatus.WARNING
        )

        return {
            "status": self.status.value,
            "passed": self.passed,
            "data_rows": self.data_shape[0],
            "data_columns": self.data_shape[1],
            "total_issues": self.total_issues,
            "validators_run": len(self.validation_results),
            "validators_passed": validators_passed,
            "validators_failed": validators_failed,
            "validators_warned": validators_warned,
            "duration_seconds": round(self.duration_seconds, 4),
            "timestamp": self.timestamp.isoformat(),
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "summary": self.get_summary(),
            "validation_results": [r.to_dict() for r in self.validation_results],
            "metadata": self.metadata,
        }

    def to_markdown(self) -> str:
        """Generate markdown report."""
        lines = [
            "# Data Quality Report",
            "",
            f"**Status**: {self.status.value.upper()}",
            f"**Timestamp**: {self.timestamp.isoformat()}",
            f"**Duration**: {self.duration_seconds:.2f}s",
            "",
            "## Data Summary",
            "",
            f"- Rows: {self.data_shape[0]:,}",
            f"- Columns: {self.data_shape[1]:,}",
            "",
            "## Validation Results",
            "",
        ]

        for result in self.validation_results:
            status_emoji = "✅" if result.passed else "❌" if result.has_errors else "⚠️"
            lines.append(f"### {status_emoji} {result.validator_name}")
            lines.append(f"- Status: {result.status.value}")
            lines.append(f"- Duration: {result.duration_seconds:.4f}s")

            if result.issues:
                lines.append(f"- Issues: {len(result.issues)}")
                for issue in result.issues[:5]:  # Limit to 5 issues
                    lines.append(f"  - [{issue.severity.value.upper()}] {issue.message}")
            lines.append("")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"QualityReport(status={self.status.value}, "
            f"issues={self.total_issues}, "
            f"duration={self.duration_seconds:.2f}s)"
        )
