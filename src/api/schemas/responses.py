"""
API Response schemas.

Pydantic models for API response serialization.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class QualityValidationResponse(BaseModel):
    """Response schema for quality validation."""

    status: str = Field(description="Overall validation status")
    passed: bool = Field(description="Whether validation passed")
    total_issues: int = Field(description="Total number of issues found")
    duration_seconds: float = Field(description="Validation duration in seconds")
    summary: dict[str, Any] = Field(description="Validation summary")
    results: list[dict[str, Any]] = Field(description="Detailed validation results")

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "passed",
                "passed": True,
                "total_issues": 0,
                "duration_seconds": 0.123,
                "summary": {
                    "data_rows": 100,
                    "data_columns": 10,
                    "validators_run": 5,
                    "validators_passed": 5,
                },
                "results": [],
            }
        }
    }


class LeakageDetectionResponse(BaseModel):
    """Response schema for leakage detection."""

    status: str = Field(description="Overall detection status")
    is_clean: bool = Field(description="Whether no leakage was detected")
    has_leakage: bool = Field(description="Whether leakage was detected")
    total_issues: int = Field(description="Total number of issues found")
    duration_seconds: float = Field(description="Detection duration in seconds")
    summary: dict[str, Any] = Field(description="Detection summary")
    results: list[dict[str, Any]] = Field(description="Detailed detection results")

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "clean",
                "is_clean": True,
                "has_leakage": False,
                "total_issues": 0,
                "duration_seconds": 0.456,
                "summary": {
                    "train_rows": 1000,
                    "test_rows": 200,
                    "detectors_run": 4,
                    "detectors_clean": 4,
                },
                "results": [],
            }
        }
    }


class ErrorResponse(BaseModel):
    """Standard error response schema."""

    error: str = Field(description="Error type")
    message: str = Field(description="Error message")
    detail: str | None = Field(default=None, description="Additional error details")
    error_code: str | None = Field(default=None, description="Error code")
