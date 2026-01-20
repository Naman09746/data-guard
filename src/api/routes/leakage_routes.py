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
        )

    except Exception as e:
        logger.error("leakage_detection_json_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def leakage_health() -> dict[str, str]:
    """Leakage detection service health check."""
    return {"status": "healthy", "service": "leakage"}


# ============== NEW ENDPOINTS ==============


@router.post("/risk-scores")
async def get_risk_scores(
    data_file: UploadFile = File(...),
    target_column: str = Form(...),
    time_column: str | None = Form(None),
) -> dict[str, Any]:
    """
    Get ML-based leakage risk scores for all features.
    
    Returns probability-based risk assessments (0-100%) for each feature.
    """
    try:
        content = await data_file.read()
        df = pd.read_csv(io.BytesIO(content))
        
        logger.info(
            "risk_scoring_request",
            rows=len(df),
            target=target_column,
        )
        
        model = LeakageRiskScoringModel()
        result = model.predict_risk(df, target_column, time_column)
        
        return result.to_dict()
        
    except Exception as e:
        logger.error("risk_scoring_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/experiments/impact")
async def run_impact_experiment(
    data_file: UploadFile = File(...),
    target_column: str = Form(...),
    time_column: str | None = Form(None),
    model_type: str = Form("random_forest"),
) -> dict[str, Any]:
    """
    Run a before/after leakage impact experiment.
    
    Trains models with and without leaky features to demonstrate impact.
    """
    try:
        content = await data_file.read()
        df = pd.read_csv(io.BytesIO(content))
        
        logger.info(
            "impact_experiment_request",
            rows=len(df),
            target=target_column,
            model_type=model_type,
        )
        
        experiment = LeakageImpactExperiment(model_type=model_type)
        result = experiment.run_experiment(df, target_column, time_column)
        
        return result.to_dict()
        
    except Exception as e:
        logger.error("impact_experiment_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history")
async def get_scan_history(
    limit: int = 50,
    scan_type: str | None = None,
) -> dict[str, Any]:
    """
    Get scan history.
    """
    try:
        store = ScanHistoryStore()
        
        type_filter = None
        if scan_type:
            type_filter = ScanType(scan_type)
        
        scans = store.get_scans(limit=limit, scan_type=type_filter)
        
        return {
            "total": len(scans),
            "scans": [s.to_dict() for s in scans],
        }
        
    except Exception as e:
        logger.error("scan_history_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts")
async def get_alerts(
    status: str | None = None,
    alert_type: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    """
    Get alerts with optional filtering.
    """
    try:
        manager = AlertManager()
        
        status_filter = AlertStatus(status) if status else None
        type_filter = AlertType(alert_type) if alert_type else None
        
        alerts = manager.get_alerts(
            status=status_filter,
            alert_type=type_filter,
            limit=limit,
        )
        
        summary = manager.get_alert_summary()
        
        return {
            "summary": summary,
            "alerts": [a.to_dict() for a in alerts],
        }
        
    except Exception as e:
        logger.error("alerts_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str) -> dict[str, Any]:
    """Acknowledge an alert."""
    try:
        manager = AlertManager()
        success = manager.acknowledge_alert(alert_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        return {"status": "acknowledged", "alert_id": alert_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("acknowledge_alert_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alerts/{alert_id}/resolve")
async def resolve_alert(
    alert_id: str,
    resolution_note: str = Form(""),
) -> dict[str, Any]:
    """Resolve an alert."""
    try:
        manager = AlertManager()
        success = manager.resolve_alert(alert_id, resolution_note)
        
        if not success:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        return {"status": "resolved", "alert_id": alert_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("resolve_alert_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

