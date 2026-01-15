"""
Data Quality API routes.

Provides endpoints for data quality validation.
"""

from __future__ import annotations

import io

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile

from src.api.schemas.requests import QualityValidationRequest
from src.api.schemas.responses import QualityValidationResponse
from src.core.logging_config import get_logger
from src.data_quality.quality_engine import DataQualityEngine

logger = get_logger(__name__)
router = APIRouter()


@router.post("/validate", response_model=QualityValidationResponse)
async def validate_data_quality(
    file: UploadFile = File(...),
    run_schema: bool = True,
    run_completeness: bool = True,
    run_consistency: bool = True,
    run_accuracy: bool = True,
    run_timeliness: bool = True,
) -> QualityValidationResponse:
    """
    Validate data quality for an uploaded CSV file.
    
    Args:
        file: CSV file to validate.
        run_schema: Whether to run schema validation.
        run_completeness: Whether to run completeness check.
        run_consistency: Whether to run consistency check.
        run_accuracy: Whether to run accuracy check.
        run_timeliness: Whether to run timeliness check.
    
    Returns:
        Quality validation results.
    """
    try:
        # Read uploaded file
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))

        logger.info(
            "quality_validation_request",
            filename=file.filename,
            rows=len(df),
            columns=len(df.columns),
        )

        # Configure engine
        engine = DataQualityEngine(config={
            "run_schema_validation": run_schema,
            "run_completeness_check": run_completeness,
            "run_consistency_check": run_consistency,
            "run_accuracy_check": run_accuracy,
            "run_timeliness_check": run_timeliness,
        })

        # Run validation
        report = engine.validate(df)

        return QualityValidationResponse(
            status=report.status.value,
            passed=report.passed,
            total_issues=report.total_issues,
            duration_seconds=round(report.duration_seconds, 4),
            summary=report.get_summary(),
            results=[r.to_dict() for r in report.validation_results],
        )

    except pd.errors.ParserError as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV format: {e}")
    except Exception as e:
        logger.error("quality_validation_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/validate/json", response_model=QualityValidationResponse)
async def validate_data_quality_json(
    request: QualityValidationRequest,
) -> QualityValidationResponse:
    """
    Validate data quality from JSON data.
    
    Args:
        request: Validation request with data and options.
    
    Returns:
        Quality validation results.
    """
    try:
        # Convert to DataFrame
        df = pd.DataFrame(request.data)

        logger.info(
            "quality_validation_json_request",
            rows=len(df),
            columns=len(df.columns),
        )

        # Configure engine
        engine = DataQualityEngine(config={
            "run_schema_validation": request.run_schema_validation,
            "run_completeness_check": request.run_completeness_check,
            "run_consistency_check": request.run_consistency_check,
            "run_accuracy_check": request.run_accuracy_check,
            "run_timeliness_check": request.run_timeliness_check,
        })

        # Run validation
        report = engine.validate(df)

        return QualityValidationResponse(
            status=report.status.value,
            passed=report.passed,
            total_issues=report.total_issues,
            duration_seconds=round(report.duration_seconds, 4),
            summary=report.get_summary(),
            results=[r.to_dict() for r in report.validation_results],
        )

    except Exception as e:
        logger.error("quality_validation_json_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def quality_health() -> dict[str, str]:
    """Quality service health check."""
    return {"status": "healthy", "service": "quality"}
