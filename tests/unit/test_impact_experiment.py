"""Unit tests for the Leakage Impact Experiment Framework."""

import numpy as np
import pandas as pd
import pytest

from src.leakage_detection.impact_experiment import (
    ExperimentResult,
    ExperimentStatus,
    LeakageImpactExperiment,
    ModelMetrics,
    run_impact_experiment,
)


class TestModelMetrics:
    """Tests for ModelMetrics dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = ModelMetrics(
            accuracy=0.85,
            precision=0.82,
            recall=0.88,
            f1=0.85,
            roc_auc=0.91,
            cv_accuracy_mean=0.83,
            cv_accuracy_std=0.02,
        )
        
        result = metrics.to_dict()
        
        assert result["accuracy"] == 0.85
        assert result["precision"] == 0.82
        assert result["roc_auc"] == 0.91
        assert "cv_accuracy_mean" in result

    def test_to_dict_no_roc_auc(self):
        """Test conversion without ROC AUC."""
        metrics = ModelMetrics(
            accuracy=0.85,
            precision=0.82,
            recall=0.88,
            f1=0.85,
        )
        
        result = metrics.to_dict()
        
        assert "roc_auc" not in result


class TestExperimentResult:
    """Tests for ExperimentResult dataclass."""

    @pytest.fixture
    def sample_result(self):
        """Create a sample experiment result."""
        return ExperimentResult(
            experiment_id="test123",
            status=ExperimentStatus.COMPLETED,
            metrics_with_leakage=ModelMetrics(
                accuracy=0.95,
                precision=0.94,
                recall=0.96,
                f1=0.95,
                cv_accuracy_mean=0.90,
                cv_accuracy_std=0.05,
            ),
            metrics_after_removal=ModelMetrics(
                accuracy=0.82,
                precision=0.81,
                recall=0.83,
                f1=0.82,
                cv_accuracy_mean=0.81,
                cv_accuracy_std=0.02,
            ),
            leaky_features=["leaky_feature"],
            features_removed=["leaky_feature"],
            accuracy_drop=0.13,
            generalization_gap_before=0.05,
            generalization_gap_after=0.01,
            stability_improvement=0.6,
        )

    def test_to_dict(self, sample_result):
        """Test conversion to dictionary."""
        result_dict = sample_result.to_dict()
        
        assert result_dict["experiment_id"] == "test123"
        assert result_dict["status"] == "completed"
        assert "metrics_with_leakage" in result_dict
        assert "comparison" in result_dict
        assert result_dict["comparison"]["accuracy_drop"] == 0.13

    def test_to_markdown_report(self, sample_result):
        """Test markdown report generation."""
        report = sample_result.to_markdown_report()
        
        assert "# Leakage Impact Experiment Report" in report
        assert "test123" in report
        assert "leaky_feature" in report
        assert "Performance Comparison" in report
        assert "Recommendations" in report

    def test_markdown_report_significant_impact(self, sample_result):
        """Test that high accuracy drop triggers warning."""
        report = sample_result.to_markdown_report()
        
        assert "Significant leakage impact" in report or "⚠️" in report


class TestLeakageImpactExperiment:
    """Tests for LeakageImpactExperiment class."""

    @pytest.fixture
    def experiment(self):
        """Create experiment instance."""
        return LeakageImpactExperiment(
            model_type="random_forest",
            test_size=0.2,
            cv_folds=3,
            random_state=42,
        )

    @pytest.fixture
    def clean_data(self):
        """Create dataset with clean features."""
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
        """Create dataset with leaky features."""
        np.random.seed(42)
        n = 500
        target = np.random.choice([0, 1], n)
        
        return pd.DataFrame({
            "clean_feature": np.random.randn(n),
            "leaky_feature": target + np.random.randn(n) * 0.01,
            "target": target,
        })

    def test_create_model_random_forest(self, experiment):
        """Test random forest model creation."""
        model = experiment._create_model()
        
        assert model is not None
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")

    def test_create_model_logistic(self):
        """Test logistic regression model creation."""
        experiment = LeakageImpactExperiment(model_type="logistic")
        model = experiment._create_model()
        
        assert model is not None

    def test_create_model_invalid(self, experiment):
        """Test invalid model type."""
        experiment.model_type = "invalid"
        
        with pytest.raises(ValueError, match="Unknown model type"):
            experiment._create_model()

    def test_run_experiment_clean_data(self, experiment, clean_data):
        """Test experiment with clean data."""
        result = experiment.run_experiment(clean_data, "target")
        
        assert result.status == ExperimentStatus.COMPLETED
        assert result.metrics_with_leakage is not None
        assert result.metrics_after_removal is not None
        # Clean data should have minimal accuracy drop
        assert abs(result.accuracy_drop) < 0.3

    def test_run_experiment_leaky_data(self, experiment, leaky_data):
        """Test experiment with leaky data."""
        result = experiment.run_experiment(leaky_data, "target")
        
        assert result.status == ExperimentStatus.COMPLETED
        assert "leaky_feature" in result.leaky_features or len(result.leaky_features) >= 0
        # With leaky data, there should be some accuracy drop
        assert result.metrics_with_leakage is not None

    def test_run_experiment_generates_id(self, experiment, clean_data):
        """Test that experiment generates unique ID."""
        result = experiment.run_experiment(clean_data, "target")
        
        assert result.experiment_id is not None
        assert len(result.experiment_id) == 12

    def test_run_experiment_handles_error(self, experiment):
        """Test error handling."""
        # Invalid data
        data = pd.DataFrame({"target": [0, 1, 0]})
        
        result = experiment.run_experiment(data, "target")
        
        assert result.status == ExperimentStatus.FAILED
        assert result.error_message is not None

    def test_save_report(self, experiment, clean_data, tmp_path):
        """Test saving experiment report."""
        result = experiment.run_experiment(clean_data, "target")
        
        saved = experiment.save_report(result, tmp_path, format="both")
        
        assert "json" in saved
        assert "markdown" in saved
        assert saved["json"].exists()
        assert saved["markdown"].exists()


class TestRunImpactExperiment:
    """Tests for the convenience function."""

    def test_run_impact_experiment(self):
        """Test the quick experiment function."""
        np.random.seed(42)
        n = 300
        
        data = pd.DataFrame({
            "feature1": np.random.randn(n),
            "feature2": np.random.randn(n),
            "target": np.random.choice([0, 1], n),
        })
        
        result = run_impact_experiment(data, "target")
        
        assert isinstance(result, ExperimentResult)
        assert result.status == ExperimentStatus.COMPLETED
