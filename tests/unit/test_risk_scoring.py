"""Unit tests for the Leakage Risk Scoring Model."""

import numpy as np
import pandas as pd
import pytest

from src.leakage_detection.risk_scoring_model import (
    FeatureRiskScore,
    LeakageRiskFeatureExtractor,
    LeakageRiskScoringModel,
    RiskLevel,
    RiskScoringResult,
    assess_feature_risk,
)


class TestRiskLevel:
    """Tests for RiskLevel enum."""

    def test_from_score_high(self):
        """Test high risk threshold."""
        assert RiskLevel.from_score(0.8) == RiskLevel.HIGH
        assert RiskLevel.from_score(0.7) == RiskLevel.HIGH
        assert RiskLevel.from_score(1.0) == RiskLevel.HIGH

    def test_from_score_medium(self):
        """Test medium risk threshold."""
        assert RiskLevel.from_score(0.5) == RiskLevel.MEDIUM
        assert RiskLevel.from_score(0.4) == RiskLevel.MEDIUM
        assert RiskLevel.from_score(0.69) == RiskLevel.MEDIUM

    def test_from_score_low(self):
        """Test low risk threshold."""
        assert RiskLevel.from_score(0.3) == RiskLevel.LOW
        assert RiskLevel.from_score(0.0) == RiskLevel.LOW
        assert RiskLevel.from_score(0.39) == RiskLevel.LOW


class TestFeatureRiskScore:
    """Tests for FeatureRiskScore dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        score = FeatureRiskScore(
            feature_name="test_feature",
            risk_score=0.75,
            risk_level=RiskLevel.HIGH,
            risk_percentage=75,
            contributing_factors={"correlation": 0.9},
            recommendations=["Review this feature"],
        )
        
        result = score.to_dict()
        
        assert result["feature_name"] == "test_feature"
        assert result["risk_score"] == 0.75
        assert result["risk_level"] == "high"
        assert result["risk_percentage"] == 75
        assert "correlation" in result["contributing_factors"]


class TestLeakageRiskFeatureExtractor:
    """Tests for LeakageRiskFeatureExtractor."""

    @pytest.fixture
    def extractor(self):
        """Create extractor instance."""
        return LeakageRiskFeatureExtractor(n_splits=3, sample_size=1000)

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n = 500
        target = np.random.choice([0, 1], n)
        
        return pd.DataFrame({
            "clean_feature": np.random.randn(n),
            "leaky_feature": target + np.random.randn(n) * 0.05,  # Highly correlated
            "moderate_feature": target * 0.3 + np.random.randn(n) * 0.7,
            "target": target,
        })

    def test_extract_features_basic(self, extractor, sample_data):
        """Test basic feature extraction."""
        result = extractor.extract_features(sample_data, "target")
        
        assert len(result) == 3  # 3 features (excluding target)
        assert "feature_name" in result.columns
        assert "correlation_abs" in result.columns
        assert "mutual_info_proxy" in result.columns

    def test_extract_features_with_time_column(self, extractor):
        """Test feature extraction with temporal data."""
        np.random.seed(42)
        n = 200
        
        data = pd.DataFrame({
            "feature1": np.random.randn(n),
            "target": np.random.choice([0, 1], n),
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="h"),
        })
        
        result = extractor.extract_features(data, "target", "timestamp")
        
        assert len(result) == 1  # Only feature1
        assert "temporal_suspicion" in result.columns

    def test_extract_features_missing_target(self, extractor):
        """Test handling of missing target column."""
        data = pd.DataFrame({"a": [1, 2, 3]})
        
        with pytest.raises(ValueError, match="Target column"):
            extractor.extract_features(data, "missing_target")

    def test_leaky_feature_has_high_correlation(self, extractor, sample_data):
        """Test that leaky features have high correlation values."""
        result = extractor.extract_features(sample_data, "target")
        
        leaky_row = result[result["feature_name"] == "leaky_feature"].iloc[0]
        clean_row = result[result["feature_name"] == "clean_feature"].iloc[0]
        
        # Leaky feature should have higher correlation
        assert leaky_row["correlation_abs"] > clean_row["correlation_abs"]
        assert leaky_row["correlation_abs"] > 0.8


class TestLeakageRiskScoringModel:
    """Tests for LeakageRiskScoringModel."""

    @pytest.fixture
    def model(self):
        """Create model instance."""
        return LeakageRiskScoringModel()

    @pytest.fixture
    def clean_data(self):
        """Create data with clean features."""
        np.random.seed(42)
        n = 500
        return pd.DataFrame({
            "feature1": np.random.randn(n),
            "feature2": np.random.randn(n),
            "feature3": np.random.uniform(0, 100, n),
            "target": np.random.choice([0, 1], n),
        })

    @pytest.fixture
    def leaky_data(self):
        """Create data with leaky features."""
        np.random.seed(42)
        n = 500
        target = np.random.choice([0, 1], n)
        
        return pd.DataFrame({
            "clean_feature": np.random.randn(n),
            "leaky_feature": target + np.random.randn(n) * 0.01,  # Almost perfect correlation
            "target": target,
        })

    def test_model_initialization(self, model):
        """Test model initializes correctly."""
        assert model.model is not None
        assert len(model.feature_names) > 0
        assert model.MODEL_VERSION == "1.0.0"

    def test_predict_risk_clean_data(self, model, clean_data):
        """Test risk prediction on clean data."""
        result = model.predict_risk(clean_data, "target")
        
        assert isinstance(result, RiskScoringResult)
        assert len(result.feature_scores) == 3
        # Clean random features should mostly be low risk
        assert result.overall_risk in [RiskLevel.LOW, RiskLevel.MEDIUM]

    def test_predict_risk_leaky_data(self, model, leaky_data):
        """Test risk prediction on leaky data."""
        result = model.predict_risk(leaky_data, "target")
        
        assert isinstance(result, RiskScoringResult)
        assert len(result.feature_scores) == 2
        
        # Find the leaky feature score
        leaky_score = next(
            s for s in result.feature_scores if s.feature_name == "leaky_feature"
        )
        
        # Leaky feature should have higher risk than clean feature
        clean_score = next(
            s for s in result.feature_scores if s.feature_name == "clean_feature"
        )
        assert leaky_score.risk_score >= clean_score.risk_score

    def test_predict_risk_generates_recommendations(self, model, leaky_data):
        """Test that recommendations are generated."""
        result = model.predict_risk(leaky_data, "target")
        
        leaky_score = next(
            s for s in result.feature_scores if s.feature_name == "leaky_feature"
        )
        
        assert len(leaky_score.recommendations) > 0

    def test_predict_risk_empty_result(self, model):
        """Test handling of non-numeric data."""
        data = pd.DataFrame({
            "category": ["A", "B", "C", "A", "B"],
            "target": [0, 1, 0, 1, 0],
        })
        
        result = model.predict_risk(data, "target")
        
        # No numeric features, so empty result
        assert len(result.feature_scores) == 0
        assert result.overall_risk == RiskLevel.LOW

    def test_result_to_dict(self, model, clean_data):
        """Test result serialization."""
        result = model.predict_risk(clean_data, "target")
        
        result_dict = result.to_dict()
        
        assert "overall_risk" in result_dict
        assert "high_risk_count" in result_dict
        assert "feature_scores" in result_dict
        assert "model_version" in result_dict


class TestAssessFeatureRisk:
    """Tests for the convenience function."""

    def test_assess_feature_risk(self):
        """Test the quick assessment function."""
        np.random.seed(42)
        n = 200
        
        data = pd.DataFrame({
            "feature1": np.random.randn(n),
            "feature2": np.random.randn(n),
            "target": np.random.choice([0, 1], n),
        })
        
        result = assess_feature_risk(data, "target")
        
        assert isinstance(result, RiskScoringResult)
        assert len(result.feature_scores) == 2


class TestIntegrationWithEngine:
    """Integration tests with LeakageDetectionEngine."""

    def test_engine_risk_scoring(self):
        """Test risk scoring through the engine."""
        from src.leakage_detection.leakage_engine import LeakageDetectionEngine
        
        np.random.seed(42)
        n = 300
        target = np.random.choice([0, 1], n)
        
        data = pd.DataFrame({
            "clean_feature": np.random.randn(n),
            "leaky_feature": target + np.random.randn(n) * 0.02,
            "target": target,
        })
        
        engine = LeakageDetectionEngine()
        result = engine.get_risk_scores(data, "target")
        
        assert isinstance(result, RiskScoringResult)
        assert len(result.feature_scores) == 2
        assert "leaky_feature" in result.high_risk_features or result.feature_scores[0].feature_name == "leaky_feature"
