"""Integration tests for engines."""

import numpy as np
import pandas as pd
import pytest

from src.data_quality.quality_engine import DataQualityEngine
from src.data_quality.quality_report import QualityStatus
from src.leakage_detection.leakage_engine import LeakageDetectionEngine
from src.leakage_detection.leakage_report import LeakageStatus


class TestDataQualityEngine:
    """Integration tests for DataQualityEngine."""

    def test_full_validation(self, sample_df):
        """Test full quality validation pipeline."""
        engine = DataQualityEngine()
        
        report = engine.validate(sample_df)
        
        assert report is not None
        assert report.status in (QualityStatus.PASSED, QualityStatus.WARNING)
        assert len(report.validation_results) > 0
        assert report.duration_seconds > 0

    def test_selective_validation(self, sample_df):
        """Test selective validator execution."""
        engine = DataQualityEngine(config={
            "run_schema_validation": True,
            "run_completeness_check": True,
            "run_consistency_check": False,
            "run_accuracy_check": False,
            "run_timeliness_check": False,
        })
        
        report = engine.validate(sample_df)
        
        assert len(report.validation_results) == 2

    def test_quality_score(self, sample_df):
        """Test quality score calculation."""
        engine = DataQualityEngine()
        
        score = engine.get_quality_score(sample_df)
        
        assert 0.0 <= score <= 1.0

    def test_report_formats(self, sample_df):
        """Test report output formats."""
        engine = DataQualityEngine()
        report = engine.validate(sample_df)
        
        # Test dict format
        report_dict = report.to_dict()
        assert "summary" in report_dict
        assert "validation_results" in report_dict
        
        # Test markdown format
        report_md = report.to_markdown()
        assert "# Data Quality Report" in report_md


class TestLeakageDetectionEngine:
    """Integration tests for LeakageDetectionEngine."""

    def test_full_detection(self, train_df, test_df):
        """Test full leakage detection pipeline."""
        engine = LeakageDetectionEngine()
        
        report = engine.detect(
            train_df, test_df,
            target_column="target",
            time_column="timestamp"
        )
        
        assert report is not None
        assert report.status in (LeakageStatus.CLEAN, LeakageStatus.LEAKAGE_DETECTED)
        assert len(report.detection_results) > 0
        assert report.duration_seconds > 0

    def test_selective_detection(self, train_df, test_df):
        """Test selective detector execution."""
        engine = LeakageDetectionEngine(config={
            "run_train_test_detection": True,
            "run_target_leakage_detection": False,
            "run_feature_leakage_detection": False,
            "run_temporal_leakage_detection": False,
        })
        
        report = engine.detect(train_df, test_df)
        
        assert len(report.detection_results) == 1

    def test_quick_check(self, train_df, test_df):
        """Test quick check method."""
        engine = LeakageDetectionEngine()
        
        has_leakage = engine.quick_check(train_df, test_df, "target")
        
        assert isinstance(has_leakage, bool)

    def test_contaminated_detection(self, contaminated_train_test):
        """Test detection of contaminated split."""
        train, test = contaminated_train_test
        engine = LeakageDetectionEngine()
        
        report = engine.detect(train, test)
        
        assert report.has_leakage
        assert report.status == LeakageStatus.LEAKAGE_DETECTED

    def test_report_formats(self, train_df, test_df):
        """Test report output formats."""
        engine = LeakageDetectionEngine()
        report = engine.detect(train_df, test_df, target_column="target")
        
        # Test dict format
        report_dict = report.to_dict()
        assert "summary" in report_dict
        assert "detection_results" in report_dict
        
        # Test markdown format
        report_md = report.to_markdown()
        assert "# Leakage Detection Report" in report_md
