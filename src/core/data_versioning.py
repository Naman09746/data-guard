"""
Data Versioning and Scan History Module.

Enables tracking dataset evolution and detecting quality regressions
through hash-based versioning and scan history storage.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.core.logging_config import get_logger

logger = get_logger(__name__)


class ScanType(str, Enum):
    """Type of scan performed."""
    QUALITY = "quality"
    LEAKAGE = "leakage"
    FULL = "full"


class RegressionSeverity(str, Enum):
    """Severity of quality regression."""
    NONE = "none"
    MINOR = "minor"
    MAJOR = "major"
    CRITICAL = "critical"


@dataclass
class DatasetVersion:
    """Represents a versioned dataset."""
    version_hash: str
    schema_hash: str
    row_count: int
    column_count: int
    column_names: list[str]
    column_dtypes: dict[str, str]
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version_hash": self.version_hash,
            "schema_hash": self.schema_hash,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "column_names": self.column_names,
            "column_dtypes": self.column_dtypes,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatasetVersion":
        """Create from dictionary."""
        data = data.copy()
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)


@dataclass
class ScanRecord:
    """A single scan record in history."""
    scan_id: str
    scan_type: ScanType
    dataset_version: DatasetVersion
    status: str
    total_issues: int
    quality_score: float | None = None
    risk_level: str | None = None
    issues_by_severity: dict[str, int] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    duration_seconds: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "scan_id": self.scan_id,
            "scan_type": self.scan_type.value,
            "dataset_version": self.dataset_version.to_dict(),
            "status": self.status,
            "total_issues": self.total_issues,
            "quality_score": self.quality_score,
            "risk_level": self.risk_level,
            "issues_by_severity": self.issues_by_severity,
            "timestamp": self.timestamp.isoformat(),
            "duration_seconds": round(self.duration_seconds, 4),
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScanRecord":
        """Create from dictionary."""
        data = data.copy()
        data["scan_type"] = ScanType(data["scan_type"])
        data["dataset_version"] = DatasetVersion.from_dict(data["dataset_version"])
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


@dataclass
class ScanDiff:
    """Difference between two scans."""
    scan_before: ScanRecord
    scan_after: ScanRecord
    
    # Changes
    schema_changed: bool = False
    row_count_change: int = 0
    column_count_change: int = 0
    columns_added: list[str] = field(default_factory=list)
    columns_removed: list[str] = field(default_factory=list)
    
    # Quality changes
    issues_change: int = 0
    quality_score_change: float = 0.0
    
    # Regression detection
    has_regression: bool = False
    regression_severity: RegressionSeverity = RegressionSeverity.NONE
    regression_details: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "scan_before_id": self.scan_before.scan_id,
            "scan_after_id": self.scan_after.scan_id,
            "schema_changed": self.schema_changed,
            "row_count_change": self.row_count_change,
            "column_count_change": self.column_count_change,
            "columns_added": self.columns_added,
            "columns_removed": self.columns_removed,
            "issues_change": self.issues_change,
            "quality_score_change": round(self.quality_score_change, 4) if self.quality_score_change else 0,
            "has_regression": self.has_regression,
            "regression_severity": self.regression_severity.value,
            "regression_details": self.regression_details,
        }


@dataclass
class RegressionAlert:
    """Alert for quality regression."""
    alert_id: str
    scan_diff: ScanDiff
    severity: RegressionSeverity
    message: str
    recommendations: list[str] = field(default_factory=list)
    acknowledged: bool = False
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "severity": self.severity.value,
            "message": self.message,
            "recommendations": self.recommendations,
            "acknowledged": self.acknowledged,
            "created_at": self.created_at.isoformat(),
            "diff": self.scan_diff.to_dict(),
        }


def compute_dataset_hash(data: pd.DataFrame, include_content: bool = True) -> str:
    """
    Compute a deterministic hash for a dataset.
    
    Args:
        data: DataFrame to hash
        include_content: Whether to include data content in hash
        
    Returns:
        SHA-256 hash string
    """
    hasher = hashlib.sha256()
    
    # Include schema
    schema_str = json.dumps({
        "columns": list(data.columns),
        "dtypes": {c: str(data[c].dtype) for c in data.columns},
        "shape": list(data.shape),
    }, sort_keys=True)
    hasher.update(schema_str.encode())
    
    # Include content summary
    if include_content:
        # Sample of data for efficiency
        sample_size = min(1000, len(data))
        sample = data.head(sample_size)
        
        # Compute column-wise statistics
        for col in sample.columns:
            try:
                if np.issubdtype(sample[col].dtype, np.number):
                    stats = f"{col}:{sample[col].mean():.6f}:{sample[col].std():.6f}"
                else:
                    stats = f"{col}:{sample[col].nunique()}:{sample[col].iloc[0] if len(sample) > 0 else ''}"
                hasher.update(stats.encode())
            except Exception:
                hasher.update(f"{col}:error".encode())
    
    return hasher.hexdigest()


def compute_schema_hash(data: pd.DataFrame) -> str:
    """Compute hash of schema only (columns and dtypes)."""
    schema_str = json.dumps({
        "columns": sorted(data.columns.tolist()),
        "dtypes": {c: str(data[c].dtype) for c in sorted(data.columns)},
    }, sort_keys=True)
    return hashlib.sha256(schema_str.encode()).hexdigest()[:16]


class DataVersioner:
    """Creates version objects for datasets."""
    
    def __init__(self) -> None:
        self._logger = get_logger("data_versioner")
    
    def create_version(
        self,
        data: pd.DataFrame,
        metadata: dict[str, Any] | None = None,
    ) -> DatasetVersion:
        """
        Create a version object for a dataset.
        
        Args:
            data: DataFrame to version
            metadata: Optional additional metadata
            
        Returns:
            DatasetVersion object
        """
        version_hash = compute_dataset_hash(data)
        schema_hash = compute_schema_hash(data)
        
        return DatasetVersion(
            version_hash=version_hash[:16],
            schema_hash=schema_hash,
            row_count=len(data),
            column_count=len(data.columns),
            column_names=data.columns.tolist(),
            column_dtypes={c: str(data[c].dtype) for c in data.columns},
            metadata=metadata or {},
        )


class ScanHistoryStore:
    """
    JSON file-based storage for scan history.
    
    Stores scan records and enables history queries and diff computation.
    """
    
    def __init__(self, storage_path: str | Path | None = None) -> None:
        if storage_path:
            self.storage_path = Path(storage_path)
        else:
            self.storage_path = Path.home() / ".dq_scan_history"
        
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.history_file = self.storage_path / "scan_history.json"
        self._logger = get_logger("scan_history_store")
        
        self._ensure_history_file()
    
    def _ensure_history_file(self) -> None:
        """Ensure history file exists."""
        if not self.history_file.exists():
            self._save_history({"scans": [], "alerts": []})
    
    def _load_history(self) -> dict[str, Any]:
        """Load history from file."""
        try:
            with open(self.history_file) as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {"scans": [], "alerts": []}
    
    def _save_history(self, history: dict[str, Any]) -> None:
        """Save history to file."""
        with open(self.history_file, "w") as f:
            json.dump(history, f, indent=2, default=str)
    
    def add_scan(self, scan: ScanRecord) -> None:
        """
        Add a scan record to history.
        
        Args:
            scan: Scan record to add
        """
        history = self._load_history()
        history["scans"].append(scan.to_dict())
        
        # Keep only last 1000 scans
        if len(history["scans"]) > 1000:
            history["scans"] = history["scans"][-1000:]
        
        self._save_history(history)
        
        self._logger.info(
            "scan_added",
            scan_id=scan.scan_id,
            scan_type=scan.scan_type.value,
            version=scan.dataset_version.version_hash,
        )
    
    def get_scans(
        self,
        limit: int = 50,
        scan_type: ScanType | None = None,
        version_hash: str | None = None,
    ) -> list[ScanRecord]:
        """
        Get scan records from history.
        
        Args:
            limit: Maximum number of records
            scan_type: Filter by scan type
            version_hash: Filter by dataset version
            
        Returns:
            List of scan records
        """
        history = self._load_history()
        scans = [ScanRecord.from_dict(s) for s in history["scans"]]
        
        if scan_type:
            scans = [s for s in scans if s.scan_type == scan_type]
        
        if version_hash:
            scans = [s for s in scans if s.dataset_version.version_hash == version_hash]
        
        # Sort by timestamp descending
        scans.sort(key=lambda x: x.timestamp, reverse=True)
        
        return scans[:limit]
    
    def get_latest_scan(
        self,
        scan_type: ScanType | None = None,
        version_hash: str | None = None,
    ) -> ScanRecord | None:
        """Get the most recent scan."""
        scans = self.get_scans(limit=1, scan_type=scan_type, version_hash=version_hash)
        return scans[0] if scans else None
    
    def compare_scans(
        self,
        scan_before: ScanRecord,
        scan_after: ScanRecord,
    ) -> ScanDiff:
        """
        Compare two scans and compute diff.
        
        Args:
            scan_before: Earlier scan
            scan_after: Later scan
            
        Returns:
            ScanDiff with comparison results
        """
        v1 = scan_before.dataset_version
        v2 = scan_after.dataset_version
        
        # Schema changes
        schema_changed = v1.schema_hash != v2.schema_hash
        cols_added = [c for c in v2.column_names if c not in v1.column_names]
        cols_removed = [c for c in v1.column_names if c not in v2.column_names]
        
        # Issue changes
        issues_change = scan_after.total_issues - scan_before.total_issues
        
        # Quality score change
        q1 = scan_before.quality_score or 0
        q2 = scan_after.quality_score or 0
        quality_change = q2 - q1
        
        # Detect regression
        has_regression = False
        severity = RegressionSeverity.NONE
        details: list[str] = []
        
        if issues_change > 0:
            has_regression = True
            if issues_change >= 10:
                severity = RegressionSeverity.CRITICAL
                details.append(f"Critical: {issues_change} new issues detected")
            elif issues_change >= 5:
                severity = RegressionSeverity.MAJOR
                details.append(f"Major: {issues_change} new issues detected")
            else:
                severity = RegressionSeverity.MINOR
                details.append(f"Minor: {issues_change} new issues detected")
        
        if quality_change < -0.1:
            has_regression = True
            if quality_change < -0.3:
                severity = max(severity, RegressionSeverity.CRITICAL)
                details.append(f"Critical: Quality score dropped by {abs(quality_change):.1%}")
            elif quality_change < -0.15:
                severity = max(severity, RegressionSeverity.MAJOR)
                details.append(f"Major: Quality score dropped by {abs(quality_change):.1%}")
            else:
                severity = max(severity, RegressionSeverity.MINOR)
                details.append(f"Minor: Quality score dropped by {abs(quality_change):.1%}")
        
        if schema_changed:
            details.append(f"Schema changed: +{len(cols_added)} columns, -{len(cols_removed)} columns")
        
        return ScanDiff(
            scan_before=scan_before,
            scan_after=scan_after,
            schema_changed=schema_changed,
            row_count_change=v2.row_count - v1.row_count,
            column_count_change=v2.column_count - v1.column_count,
            columns_added=cols_added,
            columns_removed=cols_removed,
            issues_change=issues_change,
            quality_score_change=quality_change,
            has_regression=has_regression,
            regression_severity=severity,
            regression_details=details,
        )
    
    def add_alert(self, alert: RegressionAlert) -> None:
        """Add a regression alert."""
        history = self._load_history()
        history["alerts"].append(alert.to_dict())
        
        # Keep only last 100 alerts
        if len(history["alerts"]) > 100:
            history["alerts"] = history["alerts"][-100:]
        
        self._save_history(history)
    
    def get_alerts(
        self,
        limit: int = 20,
        unacknowledged_only: bool = False,
    ) -> list[dict[str, Any]]:
        """Get regression alerts."""
        history = self._load_history()
        alerts = history.get("alerts", [])
        
        if unacknowledged_only:
            alerts = [a for a in alerts if not a.get("acknowledged", False)]
        
        # Sort by creation time descending
        alerts.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        
        return alerts[:limit]
    
    def check_for_regression(
        self,
        current_scan: ScanRecord,
    ) -> RegressionAlert | None:
        """
        Check if current scan shows regression from previous.
        
        Args:
            current_scan: Current scan record
            
        Returns:
            RegressionAlert if regression detected, else None
        """
        previous = self.get_latest_scan(scan_type=current_scan.scan_type)
        
        if not previous:
            return None
        
        diff = self.compare_scans(previous, current_scan)
        
        if not diff.has_regression:
            return None
        
        # Generate alert
        alert_id = hashlib.md5(
            f"{current_scan.scan_id}:{previous.scan_id}".encode()
        ).hexdigest()[:12]
        
        recommendations = []
        if diff.issues_change > 0:
            recommendations.append("Review new data sources for quality issues")
            recommendations.append("Check recent changes to data pipeline")
        if diff.quality_score_change < -0.1:
            recommendations.append("Investigate root cause of quality degradation")
            recommendations.append("Consider reverting to previous data version")
        if diff.schema_changed:
            recommendations.append("Validate schema changes were intentional")
        
        alert = RegressionAlert(
            alert_id=alert_id,
            scan_diff=diff,
            severity=diff.regression_severity,
            message="; ".join(diff.regression_details),
            recommendations=recommendations,
        )
        
        self.add_alert(alert)
        
        self._logger.warning(
            "regression_detected",
            alert_id=alert_id,
            severity=diff.regression_severity.value,
            details=diff.regression_details,
        )
        
        return alert


# Convenience functions
def version_dataset(data: pd.DataFrame) -> DatasetVersion:
    """Quick function to version a dataset."""
    versioner = DataVersioner()
    return versioner.create_version(data)


def get_scan_history(limit: int = 50, storage_path: str | None = None) -> list[ScanRecord]:
    """Quick function to get scan history."""
    store = ScanHistoryStore(storage_path)
    return store.get_scans(limit=limit)
