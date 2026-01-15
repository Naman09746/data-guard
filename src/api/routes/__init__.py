"""API routes package."""

from src.api.routes.leakage_routes import router as leakage_router
from src.api.routes.quality_routes import router as quality_router

__all__ = ["leakage_router", "quality_router"]
