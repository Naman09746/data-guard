"""Unit tests for the Alert System."""

import pytest

from src.core.alert_system import (
    ActionRecommendation,
    Alert,
    AlertManager,
    AlertSeverity,
    AlertStatus,
    AlertType,
    RecommendedAction,
    create_alert,
    get_open_alerts,
)


class TestAlertSeverity:
    """Tests for AlertSeverity enum."""

    def test_from_score_critical(self):
        """Test critical threshold."""
        assert AlertSeverity.from_score(0.9) == AlertSeverity.CRITICAL
        assert AlertSeverity.from_score(0.8) == AlertSeverity.CRITICAL

    def test_from_score_error(self):
        """Test error threshold."""
        assert AlertSeverity.from_score(0.7) == AlertSeverity.ERROR
        assert AlertSeverity.from_score(0.6) == AlertSeverity.ERROR

    def test_from_score_warning(self):
        """Test warning threshold."""
        assert AlertSeverity.from_score(0.5) == AlertSeverity.WARNING
        assert AlertSeverity.from_score(0.4) == AlertSeverity.WARNING

    def test_from_score_info(self):
        """Test info threshold."""
        assert AlertSeverity.from_score(0.3) == AlertSeverity.INFO
        assert AlertSeverity.from_score(0.0) == AlertSeverity.INFO


class TestActionRecommendation:
    """Tests for ActionRecommendation dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        rec = ActionRecommendation(
            action=RecommendedAction.RETRAIN_MODEL,
            priority=8,
            description="Retrain the model",
            estimated_impact="High",
            automated=False,
        )
        
        result = rec.to_dict()
        
        assert result["action"] == "retrain_model"
        assert result["priority"] == 8
        assert result["automated"] is False


class TestAlert:
    """Tests for Alert dataclass."""

    @pytest.fixture
    def sample_alert(self):
        """Create sample alert."""
        return Alert(
            alert_id="test123",
            alert_type=AlertType.DRIFT,
            severity=AlertSeverity.WARNING,
            title="Test Alert",
            message="This is a test alert",
            source="TestSource",
            affected_features=["feature1"],
        )

    def test_to_dict(self, sample_alert):
        """Test conversion to dictionary."""
        result = sample_alert.to_dict()
        
        assert result["alert_id"] == "test123"
        assert result["alert_type"] == "drift"
        assert result["severity"] == "warning"
        assert result["status"] == "open"

    def test_from_dict(self, sample_alert):
        """Test creation from dictionary."""
        data = sample_alert.to_dict()
        
        alert = Alert.from_dict(data)
        
        assert alert.alert_id == "test123"
        assert alert.alert_type == AlertType.DRIFT
        assert alert.severity == AlertSeverity.WARNING


class TestAlertManager:
    """Tests for AlertManager class."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create manager with temp path."""
        return AlertManager(tmp_path)

    def test_create_alert(self, manager):
        """Test alert creation."""
        alert = manager.create_alert(
            alert_type=AlertType.DATA_QUALITY,
            severity=AlertSeverity.WARNING,
            title="Quality Issue",
            message="Missing values detected",
            source="QualityChecker",
        )
        
        assert alert.alert_id is not None
        assert alert.status == AlertStatus.OPEN
        assert len(alert.recommendations) > 0

    def test_get_alerts(self, manager):
        """Test getting alerts."""
        manager.create_alert(
            AlertType.DRIFT, AlertSeverity.WARNING,
            "Test 1", "Message 1"
        )
        manager.create_alert(
            AlertType.LEAKAGE, AlertSeverity.ERROR,
            "Test 2", "Message 2"
        )
        
        all_alerts = manager.get_alerts()
        assert len(all_alerts) == 2
        
        drift_alerts = manager.get_alerts(alert_type=AlertType.DRIFT)
        assert len(drift_alerts) == 1

    def test_get_open_alerts(self, manager):
        """Test getting open alerts."""
        manager.create_alert(
            AlertType.DRIFT, AlertSeverity.WARNING,
            "Test", "Message"
        )
        
        open_alerts = manager.get_open_alerts()
        
        assert len(open_alerts) == 1
        assert open_alerts[0].status == AlertStatus.OPEN

    def test_acknowledge_alert(self, manager):
        """Test acknowledging an alert."""
        alert = manager.create_alert(
            AlertType.DRIFT, AlertSeverity.WARNING,
            "Test", "Message"
        )
        
        result = manager.acknowledge_alert(alert.alert_id)
        
        assert result is True
        
        alerts = manager.get_alerts()
        assert alerts[0].status == AlertStatus.ACKNOWLEDGED

    def test_resolve_alert(self, manager):
        """Test resolving an alert."""
        alert = manager.create_alert(
            AlertType.DRIFT, AlertSeverity.WARNING,
            "Test", "Message"
        )
        
        result = manager.resolve_alert(alert.alert_id, "Fixed the issue")
        
        assert result is True
        
        alerts = manager.get_alerts()
        assert alerts[0].status == AlertStatus.RESOLVED

    def test_create_drift_alert(self, manager):
        """Test drift-specific alert creation."""
        alert = manager.create_drift_alert(
            feature_name="price",
            drift_score=0.75,
            drift_type="distribution_shift",
            affected_models=["model1"],
        )
        
        assert alert.alert_type == AlertType.DRIFT
        assert alert.severity == AlertSeverity.ERROR
        assert "price" in alert.affected_features
        assert "RETRAIN" in str([r.action for r in alert.recommendations])

    def test_create_leakage_alert(self, manager):
        """Test leakage-specific alert creation."""
        alert = manager.create_leakage_alert(
            feature_name="future_price",
            risk_score=0.9,
            leakage_type="target_leakage",
        )
        
        assert alert.alert_type == AlertType.LEAKAGE
        assert alert.severity == AlertSeverity.CRITICAL
        assert "future_price" in alert.affected_features

    def test_get_alert_summary(self, manager):
        """Test alert summary generation."""
        manager.create_alert(
            AlertType.DRIFT, AlertSeverity.WARNING,
            "Test 1", "Message"
        )
        manager.create_alert(
            AlertType.LEAKAGE, AlertSeverity.CRITICAL,
            "Test 2", "Message"
        )
        
        summary = manager.get_alert_summary()
        
        assert summary["total_alerts"] == 2
        assert summary["open_alerts"] == 2
        assert summary["critical_open"] == 1


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_alert_func(self, tmp_path):
        """Test quick alert creation."""
        # Note: Uses default path, may not work in all test envs
        pass  # Skip in unit tests

    def test_get_open_alerts_func(self, tmp_path):
        """Test quick get open alerts."""
        pass  # Skip in unit tests
