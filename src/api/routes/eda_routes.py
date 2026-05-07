from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import pandas as pd
import io
from typing import Optional

from src.eda.profiler import EDAProfiler
from src.eda.schemas import EDAReport
from src.core.database import get_db
from src.core.models import Scan, EDAReport as EDAReportModel, Insight
from src.ai_insights.insight_engine import InsightEngine
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import Depends
from src.tasks.eda_tasks import run_eda_profile_task
import os
import uuid
from src.api.schemas.tasks import TaskResponse

router = APIRouter(tags=["EDA"])
profiler = EDAProfiler()
insight_engine = InsightEngine()

@router.post("/profile", response_model=TaskResponse)
async def profile_dataset(
    file: UploadFile = File(...),
    target_column: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db)
):
    """
    Upload a CSV file and get a full EDA profile.
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported for now.")
        
    try:
        # Read CSV
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # 1. Save file locally for background processing
        os.makedirs("storage/uploads", exist_ok=True)
        file_path = f"storage/uploads/{uuid.uuid4()}_{file.filename}"
        with open(file_path, "wb") as f:
            f.write(contents)

        # 2. Initialize Scan in PostgreSQL (Status: PENDING)
        scan = Scan(
            dataset_name=file.filename,
            dataset_hash=str(hash(contents)),
            scan_type="eda",
            status="processing"
        )
        db.add(scan)
        await db.commit()
        await db.refresh(scan)

        # 3. Trigger Background Task
        task = run_eda_profile_task.delay(
            file_path=file_path,
            dataset_name=file.filename,
            scan_id=scan.id,
            target_column=target_column
        )

        return {
            "task_id": task.id,
            "scan_id": scan.id,
            "status": "processing",
            "message": "Dataset analysis started in background."
        }
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Error initiating analysis: {str(e)}")

@router.get("/insights/{scan_id}")
async def get_scan_insights(
    scan_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """
    Fetch AI insights for a specific scan.
    """
    from sqlalchemy import select
    from src.core.models import Insight as InsightModel
    result = await db.execute(select(InsightModel).where(InsightModel.scan_id == scan_id))
    insight = result.scalar_one_or_none()
    
    if not insight:
        raise HTTPException(status_code=404, detail="Insights not found for this scan.")
        
    return insight

@router.get("/report/{scan_id}")
async def get_full_report(
    scan_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """
    Fetch the full EDA report (statistics) for a specific scan.
    """
    from sqlalchemy import select
    result = await db.execute(select(EDAReportModel).where(EDAReportModel.scan_id == scan_id))
    report = result.scalar_one_or_none()
    
    if not report:
        raise HTTPException(status_code=404, detail="Report not found for this scan.")
        
    return report.report_data
