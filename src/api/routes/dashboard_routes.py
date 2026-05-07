from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc
from src.core.database import get_db
from src.core.models import Scan, Alert
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List

router = APIRouter(tags=["Intelligence"])

@router.get("/stats")
async def get_global_stats(db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """
    Get aggregate metrics for the executive dashboard.
    """
    # 1. Basic Counts
    total_scans = await db.scalar(select(func.count(Scan.id)))
    active_alerts = await db.scalar(select(func.count(Alert.id)).where(Alert.status == "open"))
    
    # 2. Global Health Score (Weighted average of the last 50 scans)
    avg_health = await db.scalar(
        select(func.avg(Scan.overall_health_score))
        .where(Scan.overall_health_score.isnot(None))
    ) or 0.0

    # 3. Risk Distribution
    risk_dist_result = await db.execute(
        select(Scan.risk_level, func.count(Scan.id))
        .group_by(Scan.risk_level)
    )
    risk_distribution = {row[0] or "unknown": row[1] for row in risk_dist_result.all()}

    # 4. Scan Timeline (Last 30 days)
    thirty_days_ago = datetime.now(timezone.utc) - timedelta(days=30)
    timeline_result = await db.execute(
        select(
            func.date(Scan.created_at).label("date"),
            func.avg(Scan.overall_health_score).label("score"),
            func.count(Scan.id).label("count")
        )
        .where(Scan.created_at >= thirty_days_ago)
        .group_by(func.date(Scan.created_at))
        .order_by("date")
    )
    
    timeline = [
        {"date": str(row.date), "score": float(row.score or 0), "count": int(row.count)}
        for row in timeline_result.all()
    ]

    # 5. Recent Activity
    recent_scans = await db.execute(
        select(Scan).order_by(desc(Scan.created_at)).limit(5)
    )
    
    return {
        "summary": {
            "total_scans": total_scans,
            "active_alerts": active_alerts,
            "global_health_score": round(float(avg_health), 1),
            "total_issues_blocked": await db.scalar(select(func.sum(Scan.total_issues))) or 0
        },
        "risk_distribution": risk_distribution,
        "timeline": timeline,
        "recent_scans": [
            {
                "id": s.id,
                "name": s.dataset_name,
                "type": s.scan_type,
                "score": s.overall_health_score,
                "date": s.created_at.isoformat()
            } for s in recent_scans.scalars().all()
        ]
    }
