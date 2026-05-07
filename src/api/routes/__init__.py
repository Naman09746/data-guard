"""API routes package."""

from src.api.routes.leakage_routes import router as leakage_router
from src.api.routes.quality_routes import router as quality_router
from src.api.routes.eda_routes import router as eda_router
from src.api.routes.task_routes import router as task_router
from src.api.routes.drift_routes import router as drift_router
from src.api.routes.dashboard_routes import router as dashboard_router

__all__ = [
    "leakage_router", 
    "quality_router", 
    "eda_router", 
    "task_router", 
    "drift_router", 
    "dashboard_router"
]
