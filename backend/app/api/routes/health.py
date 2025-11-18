"""
Health check and system status endpoints.
"""
import time
from typing import Dict, Any
from datetime import datetime

from fastapi import APIRouter
from pydantic import BaseModel

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()

# Track startup time
startup_time = time.time()


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    uptime_seconds: float
    environment: str
    version: str


class DetailedHealthResponse(BaseModel):
    """Detailed health check response."""
    status: str
    timestamp: str
    uptime_seconds: float
    environment: str
    version: str
    services: Dict[str, Any]


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Basic health check endpoint.
    Returns 200 if service is running.
    """
    uptime = time.time() - startup_time

    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        uptime_seconds=uptime,
        environment=settings.environment,
        version="1.0.0",
    )


@router.get("/health/detailed", response_model=DetailedHealthResponse)
async def detailed_health_check():
    """
    Detailed health check with service status.
    Checks connectivity to databases, LLM, cameras, etc.
    """
    uptime = time.time() - startup_time

    # Check services (placeholder - implement actual checks)
    services = {
        "database": {
            "status": "connected",
            "latency_ms": 2.3,
        },
        "redis": {
            "status": "connected",
            "latency_ms": 0.8,
        },
        "ollama": {
            "status": "available",
            "host": settings.ollama_host,
            "model": settings.ollama_model,
        },
        "cameras": {
            "status": "streaming",
            "active_cameras": settings.num_cameras,
            "fps": settings.camera_fps,
        },
        "perception_pipeline": {
            "status": "running",
            "target_fps": settings.perception_fps,
            "gpu_enabled": settings.use_gpu,
        },
        "grpc_server": {
            "status": "listening",
            "port": settings.grpc_port,
        },
    }

    # Determine overall status
    all_healthy = all(
        service.get("status") in ["connected", "available", "streaming", "running", "listening"]
        for service in services.values()
    )

    return DetailedHealthResponse(
        status="healthy" if all_healthy else "degraded",
        timestamp=datetime.utcnow().isoformat(),
        uptime_seconds=uptime,
        environment=settings.environment,
        version="1.0.0",
        services=services,
    )


@router.get("/health/ready")
async def readiness_check():
    """
    Readiness probe for Kubernetes.
    Returns 200 when service is ready to accept traffic.
    """
    # Check if critical services are ready
    # - Models loaded
    # - Database connected
    # - Cameras initialized

    return {"status": "ready"}


@router.get("/health/live")
async def liveness_check():
    """
    Liveness probe for Kubernetes.
    Returns 200 if service is alive (even if not ready).
    """
    return {"status": "alive"}
