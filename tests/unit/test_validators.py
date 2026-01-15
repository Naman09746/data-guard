"""Unit tests for data quality validators."""

import numpy as np
import pandas as pd
import pytest

from src.data_quality.validators.schema_validator import SchemaValidator, DataFrameSchema
from src.data_quality.validators.completeness_checker import CompletenessChecker
from src.data_quality.validators.consistency_analyzer import ConsistencyAnalyzer, ConsistencyRule
from src.data_quality.validators.accuracy_validator import AccuracyValidator, AccuracyRule
from src.data_quality.validators.base import ValidationStatus, ValidationSeverity


class TestSchemaValidator:
    """Tests for SchemaValidator."""

    def test_valid_schema(self, sample_df):
        """Test validation with valid data."""
        schema = DataFrameSchema.infer_from_dataframe(sample_df)
        validator = SchemaValidator(schema=schema)
        
        result = validator.validate(sample_df)
        
        assert result.status in (ValidationStatus.PASSED, ValidationStatus.WARNING)

    def test_missing_columns(self, sample_df):
        """Test detection of missing columns."""
        schema = {
            "columns": [
                {"name": "id", "dtype": "int"},
                {"name": "missing_column", "dtype": "string"},
            ]
        }
        validator = SchemaValidator(schema=schema)
        
        result = validator.validate(sample_df)
        
        assert len(result.issues) > 0
        assert any("missing" in issue.message.lower() for issue in result.issues)

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        validator = SchemaValidator()
        empty_df = pd.DataFrame()
        
        result = validator.validate(empty_df)
        
        assert len(result.issues) > 0

    def test_schema_inference(self, sample_df):
        """Test schema inference from DataFrame."""
        validator = SchemaValidator()
        schema = validator.infer_schema(sample_df)
        
        assert len(schema.columns) == len(sample_df.columns)


class TestCompletenessChecker:
    """Tests for CompletenessChecker."""

    def test_complete_data(self, sample_df):
        """Test validation of complete data."""
        checker = CompletenessChecker()
        
        result = checker.validate(sample_df)
        
        assert result.passed or result.status == ValidationStatus.WARNING

    def test_detect_nulls(self, sample_df_with_nulls):
        """Test detection of null values."""
        checker = CompletenessChecker(config={"threshold": 0.01})
        
        result = checker.validate(sample_df_with_nulls)
        
        assert len(result.issues) > 0
        assert "missing" in result.issues[0].message.lower()

    def test_threshold_configuration(self):
        """Test threshold configuration."""
        df = pd.DataFrame({
            "a": [1, 2, None, 4, 5],  # 20% missing
            "b": [1, 2, 3, 4, 5],     # 0% missing
        })
        
        # Low threshold - should fail
        checker_low = CompletenessChecker(config={"threshold": 0.1})
        result_low = checker_low.validate(df)
        
        # High threshold - should pass
        checker_high = CompletenessChecker(config={"threshold": 0.5})
        result_high = checker_high.validate(df)
        
        assert not result_low.passed
        assert result_high.passed


class TestConsistencyAnalyzer:
    """Tests for ConsistencyAnalyzer."""

    def test_comparison_rule(self):
        """Test comparison rule."""
        df = pd.DataFrame({
            "start_date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
            "end_date": pd.to_datetime(["2024-01-15", "2024-01-15", "2024-03-15"]),  # Second is invalid
        })
        
        rule = ConsistencyRule(
            name="date_order",
            description="Start must be before end",
            rule_type="comparison",
            columns=["start_date", "end_date"],
            parameters={"operator": "<="},
        )
        
        analyzer = ConsistencyAnalyzer(rules=[rule], auto_detect=False)
        result = analyzer.validate(df)
        
        assert len(result.issues) > 0

    def test_auto_detection(self):
        """Test auto-detection of consistency rules."""
        df = pd.DataFrame({
            "start_date": pd.to_datetime(["2024-01-01", "2024-02-01"]),
            "end_date": pd.to_datetime(["2024-01-15", "2024-02-15"]),
        })
        
        analyzer = ConsistencyAnalyzer(auto_detect=True)
        result = analyzer.validate(df)
        
        # Should have detected and applied rules
        assert result.metrics.get("total_rules", 0) >= 1


class TestAccuracyValidator:
    """Tests for AccuracyValidator."""

    def test_range_validation(self):
        """Test range validation."""
        df = pd.DataFrame({
            "age": [25, 30, 150, 35, -5],  # 150 and -5 are invalid
        })
        
        rule = AccuracyRule(
            column="age",
            rule_type="range",
            parameters={"min": 0, "max": 120},
        )
        
        validator = AccuracyValidator(rules=[rule])
        result = validator.validate(df, check_outliers=False)
        
        assert len(result.issues) > 0

    def test_outlier_detection(self):
        """Test outlier detection."""
        np.random.seed(42)
        values = np.random.randn(100)
        values[0] = 100  # Obvious outlier
        
        df = pd.DataFrame({"value": values})
        
        validator = AccuracyValidator()
        result = validator.validate(df, check_outliers=True)
        
        assert any("outlier" in issue.message.lower() for issue in result.issues)

    def test_pattern_validation(self):
        """Test pattern validation."""
        df = pd.DataFrame({
            "email": ["test@example.com", "invalid-email", "user@domain.org"],
        })
        
        rule = AccuracyRule(
            column="email",
            rule_type="pattern",
            parameters={"pattern": r"^[\w\.-]+@[\w\.-]+\.\w+$"},
        )
        
        validator = AccuracyValidator(rules=[rule])
        result = validator.validate(df, check_outliers=False)
        
        assert len(result.issues) > 0
