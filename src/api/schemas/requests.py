"""
API Request schemas.

Pydantic models for API request validation.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class QualityValidationRequest(BaseModel):
    """Request schema for quality validation."""

    data: list[dict[str, Any]] = Field(
        ...,
        description="Data to validate as list of records",
        min_length=1,
    )
    run_schema_validation: bool = Field(
        default=True,
        description="Whether to run schema validation",
    )
    run_completeness_check: bool = Field(
        default=True,
        description="Whether to run completeness check",
    )
    run_consistency_check: bool = Field(
        default=True,
        description="Whether to run consistency check",
    )
    run_accuracy_check: bool = Field(
        default=True,
        description="Whether to run accuracy check",
    )
    run_timeliness_check: bool = Field(
        default=True,
        description="Whether to run timeliness check",
    )
    schema_definition: dict[str, Any] | None = Field(
        default=None,
        description="Optional schema definition for validation",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "data": [
                    {"id": 1, "name": "Alice", "age": 30},
                    {"id": 2, "name": "Bob", "age": 25},
                ],
                "run_schema_validation": True,
                "run_completeness_check": True,
            }
        }
    }


class LeakageDetectionRequest(BaseModel):
    """Request schema for leakage detection."""

    train_data: list[dict[str, Any]] = Field(
        ...,
        description="Training data as list of records",
        min_length=1,
    )
    test_data: list[dict[str, Any]] | None = Field(
        default=None,
        description="Test data as list of records",
    )
    target_column: str | None = Field(
        default=None,
        description="Name of the target column",
    )
    time_column: str | None = Field(
        default=None,
        description="Name of the time column for temporal checks",
    )
    run_train_test_detection: bool = Field(
        default=True,
        description="Whether to run train-test contamination detection",
    )
    run_target_leakage_detection: bool = Field(
        default=True,
        description="Whether to run target leakage detection",
    )
    run_feature_leakage_detection: bool = Field(
        default=True,
        description="Whether to run feature leakage detection",
    )
    run_temporal_leakage_detection: bool = Field(
        default=True,
        description="Whether to run temporal leakage detection",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "train_data": [
                    {"feature1": 1.0, "feature2": 2.0, "target": 0},
                    {"feature1": 2.0, "feature2": 3.0, "target": 1},
                ],
                "test_data": [
                    {"feature1": 1.5, "feature2": 2.5, "target": 0},
                ],
                "target_column": "target",
            }
        }
    }
