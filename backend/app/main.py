"""
Main FastAPI application for Axiom Robotic AI Platform.
"""
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app

from app.core.config import settings
from app.core.logging import setup_logging, get_logger
from app.api.routes import auth, health, cameras, perception, robot, models

# Setup logging
setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Application lifespan events."""
    # Startup
    logger.info(
        "starting_axiom_platform",
        environment=settings.environment,
        debug=settings.debug,
    )

    # Initialize services here
    # - Database connection
    # - Redis connection
    # - Model loading
    # - Camera initialization

    logger.info("axiom_platform_started")

    yield

    # Shutdown
    logger.info("shutting_down_axiom_platform")

    # Cleanup services here
    # - Close database connections
    # - Close Redis connections
    # - Release GPU memory
    # - Stop camera streams

    logger.info("axiom_platform_shutdown_complete")


# Create FastAPI application
app = FastAPI(
    title="Axiom Robotic AI Platform",
    description="Real-time multimodal reasoning and perception for robotic operations",
    version="1.0.0",
    lifespan=lifespan,
    debug=settings.debug,
)


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(
        "unhandled_exception",
        path=request.url.path,
        method=request.method,
        error=str(exc),
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if settings.debug else "An error occurred",
        },
    )


@app.exception_handler(PermissionError)
async def permission_error_handler(request: Request, exc: PermissionError):
    """Permission error handler."""
    logger.warning(
        "permission_denied",
        path=request.url.path,
        method=request.method,
        error=str(exc),
    )
    return JSONResponse(
        status_code=403,
        content={"error": "Permission denied", "detail": str(exc)},
    )


# Include routers
app.include_router(auth.router, prefix="/api/v1/auth", tags=["Authentication"])
app.include_router(health.router, prefix="/api/v1", tags=["Health"])
app.include_router(cameras.router, prefix="/api/v1/cameras", tags=["Cameras"])
app.include_router(perception.router, prefix="/api/v1/perception", tags=["Perception"])
app.include_router(robot.router, prefix="/api/v1/robot", tags=["Robot Control"])
app.include_router(models.router, prefix="/api/v1/models", tags=["Models"])


# Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Axiom Robotic AI Platform",
        "version": "1.0.0",
        "status": "operational",
        "environment": settings.environment,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_config=None,  # Use our custom logging
    )
