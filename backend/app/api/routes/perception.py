"""
Perception pipeline configuration and status endpoints.
"""
from typing import Annotated, Dict, Any
from pydantic import BaseModel

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from app.core.config import settings
from app.core.security import User, Permission, check_permission
from app.core.logging import get_logger
from app.api.routes.auth import get_current_user

logger = get_logger(__name__)

router = APIRouter()


class PerceptionConfig(BaseModel):
    """Perception pipeline configuration."""
    fps: int
    use_gpu: bool
    batch_size: int
    enable_segmentation: bool = True
    enable_detection: bool = True
    enable_depth: bool = True
    enable_tracking: bool = True
    enable_liquid_detection: bool = True
    enable_prediction: bool = True


class PerceptionStatus(BaseModel):
    """Perception pipeline status."""
    running: bool
    fps: float
    latency_ms: float
    gpu_utilization: float
    layer_latencies: Dict[str, float]


class PerceptionOutput(BaseModel):
    """Perception pipeline output."""
    timestamp: str
    frame_id: int
    detections: list
    segmentation: dict
    depth_map: dict
    liquid_level: dict
    trajectory_prediction: dict


@router.get("/status", response_model=PerceptionStatus)
async def get_perception_status(
    current_user: Annotated[User, Depends(get_current_user)]
):
    """
    Get current perception pipeline status and performance metrics.
    """
    check_permission(current_user, Permission.VIEW_DIAGNOSTICS)

    # Mock status - replace with actual metrics
    return PerceptionStatus(
        running=True,
        fps=settings.perception_fps,
        latency_ms=75.3,
        gpu_utilization=0.82,
        layer_latencies={
            "normalization": 8.2,
            "segmentation": 12.5,
            "detection": 15.3,
            "depth": 18.7,
            "liquid_boundary": 14.1,
            "temporal_modeling": 13.8,
            "trajectory_prediction": 12.7,
        },
    )


@router.get("/config", response_model=PerceptionConfig)
async def get_perception_config(
    current_user: Annotated[User, Depends(get_current_user)]
):
    """
    Get current perception pipeline configuration.
    """
    check_permission(current_user, Permission.VIEW_DIAGNOSTICS)

    return PerceptionConfig(
        fps=settings.perception_fps,
        use_gpu=settings.use_gpu,
        batch_size=settings.batch_size,
    )


@router.post("/config")
async def update_perception_config(
    config: PerceptionConfig,
    current_user: Annotated[User, Depends(get_current_user)]
):
    """
    Update perception pipeline configuration.
    """
    check_permission(current_user, Permission.CONFIGURE_SYSTEM)

    logger.info(
        "perception_config_updated",
        config=config.model_dump(),
        user=current_user.username,
    )

    return {
        "message": "Perception configuration updated",
        "config": config,
    }


@router.post("/start")
async def start_perception(
    current_user: Annotated[User, Depends(get_current_user)]
):
    """
    Start the perception pipeline.
    """
    check_permission(current_user, Permission.CONFIGURE_SYSTEM)

    logger.info("perception_pipeline_started", user=current_user.username)

    return {"message": "Perception pipeline started", "status": "running"}


@router.post("/stop")
async def stop_perception(
    current_user: Annotated[User, Depends(get_current_user)]
):
    """
    Stop the perception pipeline.
    """
    check_permission(current_user, Permission.CONFIGURE_SYSTEM)

    logger.info("perception_pipeline_stopped", user=current_user.username)

    return {"message": "Perception pipeline stopped", "status": "stopped"}


@router.websocket("/ws/perception")
async def perception_stream(websocket: WebSocket):
    """
    WebSocket endpoint for real-time perception output.
    Streams detection results, segmentation masks, depth maps, etc.
    """
    await websocket.accept()

    logger.info("perception_stream_connected")

    try:
        # Stream perception outputs
        frame_id = 0
        while True:
            # Get perception output from pipeline
            # output = await perception_service.get_output()

            # Mock output
            output = {
                "timestamp": "2024-01-01T00:00:00Z",
                "frame_id": frame_id,
                "detections": [
                    {
                        "class": "bottle",
                        "confidence": 0.95,
                        "bbox": [100, 100, 200, 300],
                    }
                ],
                "segmentation": {
                    "available": True,
                    "classes": ["background", "bottle", "liquid"],
                },
                "depth_map": {"available": True, "min_depth": 0.5, "max_depth": 3.0},
                "liquid_level": {
                    "detected": True,
                    "level_mm": 150.5,
                    "confidence": 0.97,
                },
                "trajectory_prediction": {
                    "available": True,
                    "horizon_sec": 3.0,
                    "confidence": 0.89,
                },
            }

            await websocket.send_json(output)

            frame_id += 1

            # Wait for next frame
            import asyncio
            await asyncio.sleep(1.0 / settings.perception_fps)

    except WebSocketDisconnect:
        logger.info("perception_stream_disconnected")
    except Exception as e:
        logger.error("perception_stream_error", error=str(e))
        await websocket.close()
