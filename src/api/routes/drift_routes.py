from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
import pandas as pd
import io
from typing import List, Dict, Any, Optional
from src.drift.drift_detector import DriftDetector, FeatureDrift
from src.core.database import get_db
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.models import Scan

router = APIRouter(tags=["ML Observability"])
detector = DriftDetector()

@router.post("/analyze")
async def analyze_drift(
    reference_file: UploadFile = File(...),
    current_file: UploadFile = File(...)
):
    """
    Compare two datasets (Reference vs Production) to detect distribution drift.
    """
    if not (reference_file.filename.endswith('.csv') and current_file.filename.endswith('.csv')):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
        
    try:
        # Read datasets
        ref_df = pd.read_csv(io.BytesIO(await reference_file.read()))
        curr_df = pd.read_csv(io.BytesIO(await current_file.read()))
        
        # Run drift analysis
        drift_results = detector.analyze_drift(ref_df, curr_df)
        summary = detector.get_drift_summary(drift_results)
        
        return {
            "summary": summary,
            "feature_drifts": [
                {
                    "name": f.feature_name,
                    "score": f.drift_score,
                    "is_drifted": f.is_drifted,
                    "p_value": f.p_value,
                    "dist_ref": f.distribution_reference,
                    "dist_curr": f.distribution_current
                } for f in drift_results
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Drift analysis failed: {str(e)}")
