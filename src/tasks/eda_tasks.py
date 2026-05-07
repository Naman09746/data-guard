import os
import pandas as pd
from src.core.celery_app import celery_app
from src.eda.profiler import EDAProfiler
from src.ai_insights.insight_engine import InsightEngine
from src.core.database import SessionLocal
from src.core.models import Scan, EDAReport as EDAReportModel, Insight
import asyncio

# Create profiler and engine
profiler = EDAProfiler()
insight_engine = InsightEngine()

@celery_app.task(bind=True, name="tasks.run_eda_profile")
def run_eda_profile_task(self, file_path: str, dataset_name: str, scan_id: str, target_column: str = None):
    """
    Background task to run full EDA profiling and AI insights.
    """
    self.update_state(state="PROGRESS", meta={"progress": 10, "status": "Reading data..."})
    
    if not os.path.exists(file_path):
        return {"error": f"File {file_path} not found"}

    try:
        # 1. Read Data
        df = pd.read_csv(file_path)
        self.update_state(state="PROGRESS", meta={"progress": 30, "status": "Calculating statistics..."})

        # 2. Run Profiling
        report = profiler.run_full_profile(df, dataset_name, target_column)
        self.update_state(state="PROGRESS", meta={"progress": 60, "status": "Generating AI insights..."})

        # 3. Generate AI Insights (Using event loop for async)
        
        insights = asyncio.run(insight_engine.generate_insights(report))
        self.update_state(state="PROGRESS", meta={"progress": 90, "status": "Saving results..."})

        # 4. Save to Database
        with SessionLocal() as db:
            # Update Scan
            scan = db.query(Scan).filter(Scan.id == scan_id).first()
            if scan:
                scan.status = "completed"
                scan.overall_health_score = report.overall_health_score
                scan.summary = {"shape": report.shape, "memory_mb": report.memory_mb}

            # Save EDA Report
            eda_db = EDAReportModel(
                scan_id=scan_id,
                report_data=report.model_dump(mode="json")
            )
            db.add(eda_db)

            # Save Insight
            insight_db = Insight(
                scan_id=scan_id,
                narrative=insights.get("narrative"),
                model_name=insights.get("model_name", "lily-1.5b")
            )
            db.add(insight_db)
            
            db.commit()

        return {"status": "SUCCESS", "scan_id": scan_id}

    except Exception as e:
        with SessionLocal() as db:
            scan = db.query(Scan).filter(Scan.id == scan_id).first()
            if scan:
                scan.status = "failed"
                db.commit()
        return {"status": "FAILED", "error": str(e)}
