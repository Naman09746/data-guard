"""
Leakage Report generation and formatting.

Provides comprehensive reporting for leakage detection results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from src.leakage_detection.detectors.base import DetectionResult


class LeakageStatus(str, Enum):
    """Overall leakage status."""

    CLEAN = "clean"
    LEAKAGE_DETECTED = "leakage_detected"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class LeakageReport:
    """Comprehensive leakage detection report."""

    status: LeakageStatus
    detection_results: list[DetectionResult]
    total_issues: int
    duration_seconds: float
    timestamp: datetime
    train_shape: tuple[int, int]
    test_shape: tuple[int, int] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_clean(self) -> bool:
        """Check if no leakage was detected."""
        return self.status == LeakageStatus.CLEAN

    @property
    def has_leakage(self) -> bool:
        """Check if any leakage was detected."""
        return self.status == LeakageStatus.LEAKAGE_DETECTED

    def get_issues_by_type(self, leakage_type: str) -> list[dict[str, Any]]:
        """Get all issues of a specific type."""
        issues = []
        for result in self.detection_results:
            for issue in result.issues:
                if issue.leakage_type == leakage_type:
                    issues.append(issue.to_dict())
        return issues

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of the leakage report."""
        detectors_clean = sum(1 for r in self.detection_results if r.is_clean)
        detectors_detected = sum(1 for r in self.detection_results if r.has_leakage)

        return {
            "status": self.status.value,
            "is_clean": self.is_clean,
            "has_leakage": self.has_leakage,
            "train_rows": self.train_shape[0],
            "train_columns": self.train_shape[1],
            "test_rows": self.test_shape[0] if self.test_shape else None,
            "test_columns": self.test_shape[1] if self.test_shape else None,
            "total_issues": self.total_issues,
            "detectors_run": len(self.detection_results),
            "detectors_clean": detectors_clean,
            "detectors_detected": detectors_detected,
            "duration_seconds": round(self.duration_seconds, 4),
            "timestamp": self.timestamp.isoformat(),
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "summary": self.get_summary(),
            "detection_results": [r.to_dict() for r in self.detection_results],
            "metadata": self.metadata,
        }

    def to_markdown(self) -> str:
        """Generate markdown report."""
        status_emoji = "✅" if self.is_clean else "❌" if self.has_leakage else "⚠️"

        lines = [
            "# Leakage Detection Report",
            "",
            f"**Status**: {status_emoji} {self.status.value.replace('_', ' ').title()}",
            f"**Timestamp**: {self.timestamp.isoformat()}",
            f"**Duration**: {self.duration_seconds:.2f}s",
            "",
            "## Data Summary",
            "",
            f"- Train: {self.train_shape[0]:,} rows × {self.train_shape[1]} columns",
        ]

        if self.test_shape:
            lines.append(f"- Test: {self.test_shape[0]:,} rows × {self.test_shape[1]} columns")

        lines.extend(["", "## Detection Results", ""])

        for result in self.detection_results:
            emoji = "✅" if result.is_clean else "❌" if result.has_leakage else "⏭️"
            lines.append(f"### {emoji} {result.detector_name}")
            lines.append(f"- Status: {result.status.value}")
            lines.append(f"- Duration: {result.duration_seconds:.4f}s")

            if result.issues:
                lines.append(f"- Issues: {len(result.issues)}")
                for issue in result.issues[:5]:
                    lines.append(f"  - [{issue.severity.value.upper()}] {issue.message}")
                    if issue.recommendation:
                        lines.append(f"    - 💡 {issue.recommendation}")
            lines.append("")

        if self.has_leakage:
            lines.extend([
                "## ⚠️ Recommendations",
                "",
                "The following actions are recommended to address detected leakage:",
                "",
            ])

            recommendations = set()
            for result in self.detection_results:
                for issue in result.issues:
                    if issue.recommendation:
                        recommendations.add(issue.recommendation)

            for i, rec in enumerate(recommendations, 1):
                lines.append(f"{i}. {rec}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"LeakageReport(status={self.status.value}, "
            f"issues={self.total_issues}, "
            f"duration={self.duration_seconds:.2f}s)"
        )
