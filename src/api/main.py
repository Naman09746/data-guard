"""
FastAPI application for Data Quality & Leakage Detection System.

Provides REST API endpoints for quality validation and leakage detection.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routes.leakage_routes import router as leakage_router
from src.api.routes.quality_routes import router as quality_router
from src.core.config import get_settings
from src.core.logging_config import get_logger, set_correlation_id

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    logger.info("application_starting")
    yield
    logger.info("application_shutting_down")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="Data Quality & Leakage Detection API",
        description="Production-grade API for validating data quality and detecting data leakage in ML pipelines",
        version="1.0.0",
        docs_url=settings.api.docs_url,
        redoc_url=settings.api.redoc_url,
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request logging middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next: Any) -> Any:
        import uuid
        correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4())[:8])
        set_correlation_id(correlation_id)

        logger.info(
            "request_started",
            method=request.method,
            path=request.url.path,
        )

        response = await call_next(request)

        logger.info(
            "request_completed",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
        )

        response.headers["X-Correlation-ID"] = correlation_id
        return response

    # Exception handlers
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error(
            "unhandled_exception",
            path=request.url.path,
            error=str(exc),
            error_type=type(exc).__name__,
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "InternalServerError",
                "message": "An unexpected error occurred",
                "detail": str(exc) if settings.debug else None,
            },
        )

    # Health check endpoints
    @app.get("/health", tags=["Health"])
    async def health_check() -> dict[str, str]:
        """Health check endpoint."""
        return {"status": "healthy"}

    @app.get("/ready", tags=["Health"])
    async def readiness_check() -> dict[str, str]:
        """Readiness check endpoint."""
        return {"status": "ready"}

    # Register routers
    app.include_router(
        quality_router,
        prefix=f"{settings.api.api_prefix}/quality",
        tags=["Data Quality"],
    )

    app.include_router(
        leakage_router,
        prefix=f"{settings.api.api_prefix}/leakage",
        tags=["Leakage Detection"],
    )

    return app


# Create default app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "src.api.main:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.api.reload,
    )
