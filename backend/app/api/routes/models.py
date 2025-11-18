"""
Model management and performance monitoring endpoints.
"""
from typing import Annotated, Dict, List
from pydantic import BaseModel

from fastapi import APIRouter, Depends

from app.core.config import settings
from app.core.security import User, Permission, check_permission
from app.core.logging import get_logger
from app.api.routes.auth import get_current_user

logger = get_logger(__name__)

router = APIRouter()


class ModelInfo(BaseModel):
    """Model information."""
    name: str
    version: str
    framework: str
    status: str
    loaded: bool
    device: str


class ModelPerformance(BaseModel):
    """Model performance metrics."""
    model_name: str
    accuracy: float
    latency_ms: float
    throughput_fps: float
    gpu_memory_mb: float
    accuracy_drift: float


class LayerMetrics(BaseModel):
    """Per-layer performance metrics."""
    layer_name: str
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    accuracy: float


@router.get("/", response_model=List[ModelInfo])
async def list_models(
    current_user: Annotated[User, Depends(get_current_user)]
):
    """
    Get list of all loaded models.
    """
    check_permission(current_user, Permission.VIEW_DIAGNOSTICS)

    # Mock model data
    models = [
        ModelInfo(
            name="yolov8n",
            version="8.0",
            framework="pytorch",
            status="active",
            loaded=True,
            device="cuda:0" if settings.use_gpu else "cpu",
        ),
        ModelInfo(
            name="segment_anything",
            version="vit_h",
            framework="pytorch",
            status="active",
            loaded=True,
            device="cuda:0" if settings.use_gpu else "cpu",
        ),
        ModelInfo(
            name="midas_depth",
            version="v2.1",
            framework="pytorch",
            status="active",
            loaded=True,
            device="cuda:0" if settings.use_gpu else "cpu",
        ),
        ModelInfo(
            name="ollama_llama",
            version="3.2",
            framework="ollama",
            status="active",
            loaded=True,
            device="cuda:0" if settings.use_gpu else "cpu",
        ),
    ]

    return models


@router.get("/performance", response_model=List[ModelPerformance])
async def get_model_performance(
    current_user: Annotated[User, Depends(get_current_user)]
):
    """
    Get performance metrics for all models.
    """
    check_permission(current_user, Permission.VIEW_DIAGNOSTICS)

    # Mock performance data
    performance = [
        ModelPerformance(
            model_name="yolov8n",
            accuracy=0.89,
            latency_ms=15.3,
            throughput_fps=65.4,
            gpu_memory_mb=450.2,
            accuracy_drift=0.02,
        ),
        ModelPerformance(
            model_name="segment_anything",
            accuracy=0.92,
            latency_ms=12.5,
            throughput_fps=80.0,
            gpu_memory_mb=1024.5,
            accuracy_drift=0.01,
        ),
        ModelPerformance(
            model_name="midas_depth",
            accuracy=0.87,
            latency_ms=18.7,
            throughput_fps=53.5,
            gpu_memory_mb=512.3,
            accuracy_drift=0.03,
        ),
    ]

    return performance


@router.get("/layers", response_model=List[LayerMetrics])
async def get_layer_metrics(
    current_user: Annotated[User, Depends(get_current_user)]
):
    """
    Get per-layer performance metrics for the 6-layer pipeline.
    """
    check_permission(current_user, Permission.VIEW_DIAGNOSTICS)

    # Mock layer metrics
    layers = [
        LayerMetrics(
            layer_name="normalization_segmentation",
            latency_p50_ms=8.2,
            latency_p95_ms=12.5,
            latency_p99_ms=15.3,
            accuracy=0.91,
        ),
        LayerMetrics(
            layer_name="object_detection_tracking",
            latency_p50_ms=15.3,
            latency_p95_ms=22.1,
            latency_p99_ms=28.4,
            accuracy=0.89,
        ),
        LayerMetrics(
            layer_name="depth_estimation",
            latency_p50_ms=18.7,
            latency_p95_ms=25.3,
            latency_p99_ms=31.2,
            accuracy=0.87,
        ),
        LayerMetrics(
            layer_name="liquid_boundary_detection",
            latency_p50_ms=14.1,
            latency_p95_ms=19.8,
            latency_p99_ms=24.5,
            accuracy=0.93,
        ),
        LayerMetrics(
            layer_name="temporal_modeling",
            latency_p50_ms=13.8,
            latency_p95_ms=18.2,
            latency_p99_ms=22.7,
            accuracy=0.88,
        ),
        LayerMetrics(
            layer_name="trajectory_prediction",
            latency_p50_ms=12.7,
            latency_p95_ms=16.9,
            latency_p99_ms=21.3,
            accuracy=0.86,
        ),
    ]

    return layers


@router.get("/{model_name}/status", response_model=ModelInfo)
async def get_model_status(
    model_name: str,
    current_user: Annotated[User, Depends(get_current_user)]
):
    """
    Get status of a specific model.
    """
    check_permission(current_user, Permission.VIEW_DIAGNOSTICS)

    return ModelInfo(
        name=model_name,
        version="1.0",
        framework="pytorch",
        status="active",
        loaded=True,
        device="cuda:0" if settings.use_gpu else "cpu",
    )


@router.post("/{model_name}/reload")
async def reload_model(
    model_name: str,
    current_user: Annotated[User, Depends(get_current_user)]
):
    """
    Reload a model (for updates or fallback).
    """
    check_permission(current_user, Permission.CONFIGURE_SYSTEM)

    logger.info("model_reload_requested", model=model_name, user=current_user.username)

    return {
        "message": f"Model {model_name} reload initiated",
        "status": "reloading",
    }


@router.post("/rollback")
async def rollback_models(
    current_user: Annotated[User, Depends(get_current_user)]
):
    """
    Rollback all models to previous version.
    Emergency fallback mechanism.
    """
    check_permission(current_user, Permission.CONFIGURE_SYSTEM)

    logger.warning("model_rollback_initiated", user=current_user.username)

    return {
        "message": "Model rollback initiated",
        "status": "rolling_back",
        "previous_version": "0.9.5",
    }
