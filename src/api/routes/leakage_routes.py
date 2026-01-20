"""
Leakage Detection API routes.

Provides endpoints for data leakage detection, risk scoring, and experiments.
"""

from __future__ import annotations

import io
from typing import Any

import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from src.api.schemas.requests import LeakageDetectionRequest
from src.api.schemas.responses import LeakageDetectionResponse
from src.core.logging_config import get_logger
from src.leakage_detection.leakage_engine import LeakageDetectionEngine
from src.leakage_detection.risk_scoring_model import LeakageRiskScoringModel
from src.leakage_detection.impact_experiment import LeakageImpactExperiment
from src.core.data_versioning import ScanHistoryStore, ScanType
from src.core.alert_system import AlertManager, AlertType, AlertStatus

logger = get_logger(__name__)
router = APIRouter()


@router.post("/detect", response_model=LeakageDetectionResponse)
async def detect_leakage(
    train_file: UploadFile = File(...),
    test_file: UploadFile | None = File(None),
    target_column: str | None = Form(None),
    time_column: str | None = Form(None),
) -> LeakageDetectionResponse:
    """
    Detect data leakage in uploaded CSV files.
    
    Args:
        train_file: Training data CSV file.
        test_file: Test data CSV file (optional).
        target_column: Name of target column.
        time_column: Name of time column for temporal checks.
    
    Returns:
        Leakage detection results.
    """
    try:
        # Read train file
        train_content = await train_file.read()
        train_df = pd.read_csv(io.BytesIO(train_content))

        # Read test file if provided
        test_df = None
        if test_file:
            test_content = await test_file.read()
            test_df = pd.read_csv(io.BytesIO(test_content))

        logger.info(
            "leakage_detection_request",
            train_rows=len(train_df),
            test_rows=len(test_df) if test_df is not None else 0,
            target=target_column,
        )

        # Run detection
        engine = LeakageDetectionEngine()
        report = engine.detect(
            train_df,
            test_df,
            target_column=target_column,
            time_column=time_column,
        )

        return LeakageDetectionResponse(
            status=report.status.value,
            is_clean=report.is_clean,
            has_leakage=report.has_leakage,
            total_issues=report.total_issues,
            duration_seconds=round(report.duration_seconds, 4),
            summary=report.get_summary(),
            results=[r.to_dict() for r in report.detection_results],
        )

    except pd.errors.ParserError as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV format: {e}")
    except Exception as e:
        logger.error("leakage_detection_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/detect/json", response_model=LeakageDetectionResponse)
async def detect_leakage_json(
    request: LeakageDetectionRequest,
) -> LeakageDetectionResponse:
    """
    Detect data leakage from JSON data.
    
    Args:
        request: Detection request with data and options.
    
    Returns:
        Leakage detection results.
    """
    try:
        # Convert to DataFrames
        train_df = pd.DataFrame(request.train_data)
        test_df = pd.DataFrame(request.test_data) if request.test_data else None

        logger.info(
            "leakage_detection_json_request",
            train_rows=len(train_df),
            test_rows=len(test_df) if test_df is not None else 0,
        )

        # Configure engine
        engine = LeakageDetectionEngine(config={
            "run_train_test_detection": request.run_train_test_detection,
            "run_target_leakage_detection": request.run_target_leakage_detection,
            "run_feature_leakage_detection": request.run_feature_leakage_detection,
            "run_temporal_leakage_detection": request.run_temporal_leakage_detection,
        })

        # Run detection
        report = engine.detect(
            train_df,
            test_df,
            target_column=request.target_column,
            time_column=request.time_column,
        )

        return LeakageDetectionResponse(
            status=report.status.value,
            is_clean=report.is_clean,
            has_leakage=report.has_leakage,
            total_issues=report.total_issues,
            duration_seconds=round(report.duration_seconds, 4),
            summary=report.get_summary(),
            results=[r.to_dict() for r in report.detection_results],