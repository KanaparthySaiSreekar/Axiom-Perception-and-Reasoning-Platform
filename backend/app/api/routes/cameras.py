"""
Camera configuration and management endpoints.
"""
from typing import List, Annotated
from pydantic import BaseModel

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from app.core.config import settings
from app.core.security import User, Permission, check_permission
from app.core.logging import get_logger
from app.api.routes.auth import get_current_user

logger = get_logger(__name__)

router = APIRouter()


class CameraInfo(BaseModel):
    """Camera information."""
    id: int
    name: str
    resolution: tuple[int, int]
    fps: int
    status: str
    calibrated: bool
    pose: dict


class CameraConfig(BaseModel):
    """Camera configuration."""
    fps: int
    width: int
    height: int
    exposure: float = 0.0
    gain: float = 1.0


@router.get("/", response_model=List[CameraInfo])
async def list_cameras(current_user: Annotated[User, Depends(get_current_user)]):
    """
    Get list of all cameras with their status.
    """
    check_permission(current_user, Permission.VIEW_CAMERAS)

    # Mock camera data - replace with actual camera service
    cameras = [
        CameraInfo(
            id=i,
            name=f"Camera_{i}",
            resolution=(settings.camera_width, settings.camera_height),
            fps=settings.camera_fps,
            status="streaming",
            calibrated=True,
            pose={
                "position": [0.0, 0.0, 1.0],
                "rotation": [0.0, 0.0, 0.0, 1.0],
            },
        )
        for i in range(settings.num_cameras)
    ]

    return cameras


@router.get("/{camera_id}", response_model=CameraInfo)
async def get_camera(
    camera_id: int,
    current_user: Annotated[User, Depends(get_current_user)]
):
    """
    Get information about a specific camera.
    """
    check_permission(current_user, Permission.VIEW_CAMERAS)

    # Mock camera data
    return CameraInfo(
        id=camera_id,
        name=f"Camera_{camera_id}",
        resolution=(settings.camera_width, settings.camera_height),
        fps=settings.camera_fps,
        status="streaming",
        calibrated=True,
        pose={
            "position": [0.0, 0.0, 1.0],
            "rotation": [0.0, 0.0, 0.0, 1.0],
        },
    )


@router.post("/{camera_id}/config")
async def update_camera_config(
    camera_id: int,
    config: CameraConfig,
    current_user: Annotated[User, Depends(get_current_user)]
):
    """
    Update camera configuration.
    """
    check_permission(current_user, Permission.CONFIGURE_SYSTEM)

    logger.info(
        "camera_config_updated",
        camera_id=camera_id,
        config=config.model_dump(),
        user=current_user.username,
    )

    return {"message": f"Camera {camera_id} configuration updated", "config": config}


@router.post("/{camera_id}/calibrate")
async def calibrate_camera(
    camera_id: int,
    current_user: Annotated[User, Depends(get_current_user)]
):
    """
    Trigger camera calibration process.
    """
    check_permission(current_user, Permission.CONFIGURE_SYSTEM)

    logger.info("camera_calibration_started", camera_id=camera_id)

    return {
        "message": f"Camera {camera_id} calibration started",
        "status": "calibrating",
    }


@router.websocket("/ws/video/{camera_id}")
async def camera_video_stream(websocket: WebSocket, camera_id: int):
    """
    WebSocket endpoint for real-time camera video stream.
    Sends JPEG frames at configured FPS.
    """
    await websocket.accept()

    logger.info("camera_stream_connected", camera_id=camera_id)

    try:
        # Stream camera frames
        # This is a placeholder - implement actual camera streaming
        while True:
            # Get frame from camera service
            # frame = await camera_service.get_frame(camera_id)
            # await websocket.send_bytes(frame)

            # For now, send a status message
            await websocket.send_json({
                "camera_id": camera_id,
                "timestamp": "2024-01-01T00:00:00Z",
                "status": "streaming",
            })

            # Wait for next frame
            import asyncio
            await asyncio.sleep(1.0 / settings.camera_fps)

    except WebSocketDisconnect:
        logger.info("camera_stream_disconnected", camera_id=camera_id)
    except Exception as e:
        logger.error("camera_stream_error", camera_id=camera_id, error=str(e))
        await websocket.close()
