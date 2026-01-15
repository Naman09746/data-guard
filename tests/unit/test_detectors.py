"""Unit tests for leakage detectors."""

import numpy as np
import pandas as pd
import pytest

from src.leakage_detection.detectors.train_test_detector import TrainTestDetector
from src.leakage_detection.detectors.target_leakage_detector import TargetLeakageDetector
from src.leakage_detection.detectors.feature_leakage_detector import FeatureLeakageDetector
from src.leakage_detection.detectors.temporal_leakage_detector import TemporalLeakageDetector
from src.leakage_detection.detectors.base import DetectionStatus


class TestTrainTestDetector:
    """Tests for TrainTestDetector."""

    def test_clean_split(self, train_df, test_df):
        """Test detection with clean train/test split."""
        # Disable near-duplicate check as cosine similarity on random data can be high
        detector = TrainTestDetector(config={
            "check_duplicates": True,
            "check_near_duplicates": False,
        })
        
        result = detector.detect(train_df, test_df)
        
        # Should find no exact duplicates
        assert result.metrics.get("exact_duplicates", 0) == 0

    def test_contaminated_split(self, contaminated_train_test):
        """Test detection of train-test contamination."""
        train, test = contaminated_train_test
        detector = TrainTestDetector()
        
        result = detector.detect(train, test)
        
        assert result.has_leakage
        assert any("duplicate" in issue.message.lower() for issue in result.issues)

    def test_no_test_data(self, train_df):
        """Test handling when no test data provided."""
        detector = TrainTestDetector()
        
        result = detector.detect(train_df, None)
        
        assert result.status == DetectionStatus.SKIPPED


class TestTargetLeakageDetector:
    """Tests for TargetLeakageDetector."""

    def test_clean_features(self, train_df):
        """Test detection with clean features."""
        # Disable mutual info check as it can have false positives with random data
        detector = TargetLeakageDetector(config={
            "check_correlation": True,
            "check_mutual_info": False,
        })
        
        result = detector.detect(train_df, target_column="target")
        
        # Random features should not have high correlation with target
        correlation_issues = [i for i in result.issues if "correlation" in i.leakage_type]
        assert len(correlation_issues) == 0

    def test_leaky_features(self, leaky_features_df):
        """Test detection of target leakage."""
        detector = TargetLeakageDetector(config={
            "correlation_threshold": 0.9,
            "check_mutual_info": False,
        })
        
        result = detector.detect(leaky_features_df, target_column="target")
        
        assert result.has_leakage
        assert any("leaky_feature" in issue.affected_features for issue in result.issues)

    def test_no_target(self, train_df):
        """Test handling when no target provided."""
        detector = TargetLeakageDetector()
        
        result = detector.detect(train_df, target_column=None)
        
        assert result.status == DetectionStatus.SKIPPED


class TestFeatureLeakageDetector:
    """Tests for FeatureLeakageDetector."""

    def test_suspicious_names(self):
        """Test detection of suspicious feature names."""
        df = pd.DataFrame({
            "feature1": [1, 2, 3],
            "sales_future": [4, 5, 6],  # Suspicious
            "outcome_label": [7, 8, 9],  # Suspicious
        })
        
        detector = FeatureLeakageDetector()
        
        result = detector.detect(df)
        
        assert len(result.issues) > 0
        suspicious_features = []
        for issue in result.issues:
            suspicious_features.extend(issue.affected_features)
        
        assert "sales_future" in suspicious_features or "outcome_label" in suspicious_features

    def test_clean_names(self, train_df):
        """Test with clean feature names."""
        detector = FeatureLeakageDetector()
        
        result = detector.detect(train_df, target_column="target")
        
        # Should not flag normal feature names
        name_issues = [i for i in result.issues if "name" in i.leakage_type]
        assert len(name_issues) == 0


class TestTemporalLeakageDetector:
    """Tests for TemporalLeakageDetector."""

    def test_clean_temporal_split(self, train_df, test_df):
        """Test with proper temporal split."""
        detector = TemporalLeakageDetector()
        
        result = detector.detect(
            train_df, test_df,
            time_column="timestamp"
        )
        
        assert result.is_clean

    def test_temporal_overlap(self):
        """Test detection of temporal overlap."""
        train = pd.DataFrame({
            "feature": range(100),
            "timestamp": pd.date_range("2024-01-01", periods=100, freq="h"),
        })
        
        # Test overlaps with train
        test = pd.DataFrame({
            "feature": range(50),
            "timestamp": pd.date_range("2024-01-03", periods=50, freq="h"),  # Overlaps
        })
        
        detector = TemporalLeakageDetector()
        
        result = detector.detect(train, test, time_column="timestamp")
        
        assert result.has_leakage

    def test_no_time_column(self):
        """Test handling when no time column is found."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        detector = TemporalLeakageDetector()
        
        result = detector.detect(df)
        
        assert result.status == DetectionStatus.SKIPPED
