"""
Alert System Module.

Provides a unified alert system for the data quality platform with
severity scoring, recommendations, and action tracking.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from src.core.logging_config import get_logger

logger = get_logger(__name__)


class AlertSeverity(str, Enum):
    """Severity levels for alerts."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    
    @classmethod
    def from_score(cls, score: float) -> "AlertSeverity":
        """Convert a score (0-1) to severity."""
        if score >= 0.8:
            return cls.CRITICAL
        elif score >= 0.6:
            return cls.ERROR
        elif score >= 0.4:
            return cls.WARNING
        return cls.INFO


class AlertType(str, Enum):
    """Types of alerts."""
    DATA_QUALITY = "data_quality"
    LEAKAGE = "leakage"
    DRIFT = "drift"
    REGRESSION = "regression"
    SYSTEM = "system"


class AlertStatus(str, Enum):
    """Status of an alert."""
    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    IGNORED = "ignored"


class RecommendedAction(str, Enum):
    """Recommended actions for alerts."""
    RETRAIN_MODEL = "retrain_model"
    REVALIDATE_DATA = "revalidate_data"
    EXCLUDE_FEATURE = "exclude_feature"
    INVESTIGATE = "investigate"
    ROLLBACK = "rollback"
    NOTIFY_TEAM = "notify_team"
    AUTO_REMEDIATE = "auto_remediate"


@dataclass
class ActionRecommendation:
    """A specific action recommendation."""
    action: RecommendedAction
    priority: int  # 1-10, higher = more urgent
    description: str
    estimated_impact: str
    automated: bool = False
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "action": self.action.value,
            "priority": self.priority,
            "description": self.description,
            "estimated_impact": self.estimated_impact,
            "automated": self.automated,
        }


@dataclass
class AffectedModel:
    """Information about a model affected by an alert."""
    model_id: str
    model_name: str
    impact_score: float  # 0-1
    last_trained: datetime | None = None
    features_affected: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "impact_score": round(self.impact_score, 4),
            "last_trained": self.last_trained.isoformat() if self.last_trained else None,
            "features_affected": self.features_affected,
        }


@dataclass
class Alert:
    """A unified alert object."""
    alert_id: str
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    status: AlertStatus = AlertStatus.OPEN
    
    # Context
    source: str = ""
    affected_features: list[str] = field(default_factory=list)
    affected_models: list[AffectedModel] = field(default_factory=list)
    
    # Actions
    recommendations: list[ActionRecommendation] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    acknowledged_at: datetime | None = None
    resolved_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "alert_type": self.alert_type.value,
            "severity": self.severity.value,
            "title": self.title,
            "message": self.message,
            "status": self.status.value,
            "source": self.source,
            "affected_features": self.affected_features,
            "affected_models": [m.to_dict() for m in self.affected_models],
            "recommendations": [r.to_dict() for r in self.recommendations],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Alert":
        """Create from dictionary."""
        data = data.copy()
        data["alert_type"] = AlertType(data["alert_type"])
        data["severity"] = AlertSeverity(data["severity"])
        data["status"] = AlertStatus(data["status"])
        data["affected_models"] = []  # Simplified for now
        data["recommendations"] = []  # Simplified for now
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        if data.get("acknowledged_at"):
            data["acknowledged_at"] = datetime.fromisoformat(data["acknowledged_at"])
        if data.get("resolved_at"):
            data["resolved_at"] = datetime.fromisoformat(data["resolved_at"])
        return cls(**data)


class AlertManager:
    """
    Manages alerts for the data quality platform.
    
    Features:
    - Alert creation and storage
    - Status management
    - Recommendation generation
    - Integration with drift detection
    """
    
    def __init__(self, storage_path: str | Path | None = None) -> None:
        if storage_path:
            self.storage_path = Path(storage_path)
        else:
            self.storage_path = Path.home() / ".dq_alerts"
        
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.alerts_file = self.storage_path / "alerts.json"
        self._logger = get_logger("alert_manager")
        
        self._ensure_alerts_file()
    
    def _ensure_alerts_file(self) -> None:
        """Ensure alerts file exists."""
        if not self.alerts_file.exists():
            self._save_alerts([])
    
    def _load_alerts(self) -> list[dict[str, Any]]:
        """Load alerts from file."""
        try:
            with open(self.alerts_file) as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []
    
    def _save_alerts(self, alerts: list[dict[str, Any]]) -> None:
        """Save alerts to file."""
        with open(self.alerts_file, "w") as f:
            json.dump(alerts, f, indent=2, default=str)
    
    def _generate_alert_id(self, prefix: str = "ALT") -> str:
        """Generate a unique alert ID."""
        timestamp = datetime.now(UTC).isoformat()
        return f"{prefix}_{hashlib.md5(timestamp.encode()).hexdigest()[:10]}"
    
    def create_alert(
        self,
        alert_type: AlertType,
        severity: AlertSeverity,
        title: str,
        message: str,
        source: str = "",
        affected_features: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Alert:
        """
        Create a new alert.
        
        Args:
            alert_type: Type of alert
            severity: Severity level
            title: Short title
            message: Detailed message
            source: Source of the alert
            affected_features: Features involved
            metadata: Additional data
            
        Returns:
            Created Alert object
        """
        alert = Alert(
            alert_id=self._generate_alert_id(),
            alert_type=alert_type,
            severity=severity,
            title=title,
            message=message,
            source=source,
            affected_features=affected_features or [],
            metadata=metadata or {},
        )
        
        # Generate recommendations based on alert type
        alert.recommendations = self._generate_recommendations(alert)
        
        # Save
        alerts = self._load_alerts()
        alerts.append(alert.to_dict())
        
        # Keep only last 500 alerts
        if len(alerts) > 500:
            alerts = alerts[-500:]
        
        self._save_alerts(alerts)
        
        self._logger.info(
            "alert_created",
            alert_id=alert.alert_id,
            type=alert_type.value,
            severity=severity.value,
        )
        
        return alert
    
    def _generate_recommendations(self, alert: Alert) -> list[ActionRecommendation]:
        """Generate recommendations based on alert type and severity."""
        recommendations: list[ActionRecommendation] = []
        
        if alert.alert_type == AlertType.DRIFT:
            if alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.ERROR]:
                recommendations.append(ActionRecommendation(
                    action=RecommendedAction.RETRAIN_MODEL,
                    priority=9,
                    description="Significant drift detected. Model retraining recommended to maintain accuracy.",
                    estimated_impact="High - may prevent prediction degradation",
                ))
            recommendations.append(ActionRecommendation(
                action=RecommendedAction.INVESTIGATE,
                priority=7,
                description="Investigate root cause of drift in affected features.",
                estimated_impact="Medium - understanding drift source helps prevention",
            ))
            
        elif alert.alert_type == AlertType.LEAKAGE:
            recommendations.append(ActionRecommendation(
                action=RecommendedAction.EXCLUDE_FEATURE,
                priority=10,
                description="Remove leaky features from model training to prevent overfitting.",
                estimated_impact="Critical - leaky features cause unrealistic accuracy",
            ))
            recommendations.append(ActionRecommendation(
                action=RecommendedAction.RETRAIN_MODEL,
                priority=8,
                description="Retrain model after removing leaky features.",
                estimated_impact="High - ensures model generalizes properly",
            ))
            
        elif alert.alert_type == AlertType.DATA_QUALITY:
            if alert.severity == AlertSeverity.CRITICAL:
                recommendations.append(ActionRecommendation(
                    action=RecommendedAction.ROLLBACK,
                    priority=9,
                    description="Critical quality issues detected. Consider rolling back to previous data version.",
                    estimated_impact="High - prevents bad data from affecting models",
                ))
            recommendations.append(ActionRecommendation(
                action=RecommendedAction.REVALIDATE_DATA,
                priority=7,
                description="Run full data validation after addressing issues.",
                estimated_impact="Medium - ensures quality is restored",
            ))
            
        elif alert.alert_type == AlertType.REGRESSION:
            recommendations.append(ActionRecommendation(
                action=RecommendedAction.INVESTIGATE,
                priority=8,
                description="Quality regression detected. Investigate recent changes to data pipeline.",
                estimated_impact="High - prevents ongoing quality degradation",
            ))
            if alert.severity == AlertSeverity.CRITICAL:
                recommendations.append(ActionRecommendation(
                    action=RecommendedAction.NOTIFY_TEAM,
                    priority=9,
                    description="Alert data engineering team about critical regression.",
                    estimated_impact="Medium - enables faster response",
                ))
        
        return recommendations
    
    def get_alerts(
        self,
        status: AlertStatus | None = None,
        alert_type: AlertType | None = None,
        severity: AlertSeverity | None = None,
        limit: int = 50,
    ) -> list[Alert]:
        """
        Get alerts with optional filtering.
        
        Args:
            status: Filter by status
            alert_type: Filter by type
            severity: Filter by severity
            limit: Maximum number to return
            
        Returns:
            List of Alert objects
        """
        alerts_data = self._load_alerts()
        alerts = []
        
        for data in alerts_data:
            try:
                alert = Alert.from_dict(data)
                
                if status and alert.status != status:
                    continue
                if alert_type and alert.alert_type != alert_type:
                    continue
                if severity and alert.severity != severity:
                    continue
                
                alerts.append(alert)
            except Exception:
                continue
        
        # Sort by created_at descending
        alerts.sort(key=lambda x: x.created_at, reverse=True)
        
        return alerts[:limit]
    
    def get_open_alerts(self) -> list[Alert]:
        """Get all open alerts."""
        return self.get_alerts(status=AlertStatus.OPEN)
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """
        Acknowledge an alert.
        
        Args:
            alert_id: ID of alert to acknowledge
            
        Returns:
            True if successful
        """
        alerts_data = self._load_alerts()
        
        for data in alerts_data:
            if data.get("alert_id") == alert_id:
                data["status"] = AlertStatus.ACKNOWLEDGED.value
                data["acknowledged_at"] = datetime.now(UTC).isoformat()
                data["updated_at"] = datetime.now(UTC).isoformat()
                self._save_alerts(alerts_data)
                self._logger.info("alert_acknowledged", alert_id=alert_id)
                return True
        
        return False
    
    def resolve_alert(self, alert_id: str, resolution_note: str = "") -> bool:
        """
        Resolve an alert.
        
        Args:
            alert_id: ID of alert to resolve
            resolution_note: Optional note about resolution
            
        Returns:
            True if successful
        """
        alerts_data = self._load_alerts()
        
        for data in alerts_data:
            if data.get("alert_id") == alert_id:
                data["status"] = AlertStatus.RESOLVED.value
                data["resolved_at"] = datetime.now(UTC).isoformat()
                data["updated_at"] = datetime.now(UTC).isoformat()
                if resolution_note:
                    data.setdefault("metadata", {})["resolution_note"] = resolution_note
                self._save_alerts(alerts_data)
                self._logger.info("alert_resolved", alert_id=alert_id)
                return True
        
        return False
    
    def create_drift_alert(
        self,
        feature_name: str,
        drift_score: float,
        drift_type: str,
        affected_models: list[str] | None = None,
    ) -> Alert:
        """
        Create a drift-specific alert.
        
        Args:
            feature_name: Name of drifted feature
            drift_score: Drift severity score (0-1)
            drift_type: Type of drift detected
            affected_models: Model IDs affected
            
        Returns:
            Created Alert
        """
        severity = AlertSeverity.from_score(drift_score)
        
        title = f"Drift detected in feature: {feature_name}"
        message = (
            f"Data drift of type '{drift_type}' detected in feature '{feature_name}' "
            f"with severity score {drift_score:.2f}. "
            f"This may affect model predictions and require retraining."
        )
        
        metadata = {
            "drift_score": drift_score,
            "drift_type": drift_type,
            "feature_name": feature_name,
        }
        
        if affected_models:
            metadata["affected_model_ids"] = affected_models
        
        return self.create_alert(
            alert_type=AlertType.DRIFT,
            severity=severity,
            title=title,
            message=message,
            source="DriftDetector",
            affected_features=[feature_name],
            metadata=metadata,
        )
    
    def create_leakage_alert(
        self,
        feature_name: str,
        risk_score: float,
        leakage_type: str,
    ) -> Alert:
        """
        Create a leakage-specific alert.
        
        Args:
            feature_name: Name of leaky feature
            risk_score: Risk score (0-1)
            leakage_type: Type of leakage
            
        Returns:
            Created Alert
        """
        severity = AlertSeverity.from_score(risk_score)
        
        title = f"Leakage risk in feature: {feature_name}"
        message = (
            f"Potential data leakage detected in feature '{feature_name}' "
            f"(type: {leakage_type}) with risk score {risk_score:.0%}. "
            f"This feature may be causing unrealistic model performance."
        )
        
        return self.create_alert(
            alert_type=AlertType.LEAKAGE,
            severity=severity,
            title=title,
            message=message,
            source="LeakageDetector",
            affected_features=[feature_name],
            metadata={
                "risk_score": risk_score,
                "leakage_type": leakage_type,
            },
        )
    
    def get_alert_summary(self) -> dict[str, Any]:
        """Get a summary of current alerts."""
        alerts = self.get_alerts(limit=500)
        
        by_status = {}
        by_severity = {}
        by_type = {}
        
        for alert in alerts:
            by_status[alert.status.value] = by_status.get(alert.status.value, 0) + 1
            by_severity[alert.severity.value] = by_severity.get(alert.severity.value, 0) + 1
            by_type[alert.alert_type.value] = by_type.get(alert.alert_type.value, 0) + 1
        
        open_alerts = [a for a in alerts if a.status == AlertStatus.OPEN]
        critical_open = [a for a in open_alerts if a.severity == AlertSeverity.CRITICAL]
        
        return {
            "total_alerts": len(alerts),
            "open_alerts": len(open_alerts),
            "critical_open": len(critical_open),
            "by_status": by_status,
            "by_severity": by_severity,
            "by_type": by_type,
        }


# Convenience functions
def create_alert(
    alert_type: AlertType,
    severity: AlertSeverity,
    title: str,
    message: str,
    **kwargs: Any,
) -> Alert:
    """Quick function to create an alert."""
    manager = AlertManager()
    return manager.create_alert(alert_type, severity, title, message, **kwargs)


def get_open_alerts() -> list[Alert]:
    """Quick function to get open alerts."""
    manager = AlertManager()
    return manager.get_open_alerts()
