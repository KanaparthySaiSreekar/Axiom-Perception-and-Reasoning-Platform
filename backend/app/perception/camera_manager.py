"""
Multi-camera ingestion and synchronization system.
Handles N-camera synchronized capture with calibration and pose alignment.
"""
import asyncio
import time
from typing import List, Optional, Dict
from dataclasses import dataclass
from datetime import datetime

import cv2
import numpy as np

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CameraFrame:
    """Single camera frame with metadata."""
    camera_id: int
    frame: np.ndarray
    timestamp: float
    frame_number: int
    exposure: float
    gain: float


@dataclass
class SynchronizedFrameSet:
    """Synchronized frames from multiple cameras."""
    frames: List[CameraFrame]
    timestamp: float
    sync_quality: float  # 0-1, based on timestamp alignment


class CameraCalibration:
    """Camera calibration data."""

    def __init__(
        self,
        camera_id: int,
        intrinsic_matrix: np.ndarray,
        distortion_coeffs: np.ndarray,
        extrinsic_matrix: Optional[np.ndarray] = None,
    ):
        self.camera_id = camera_id
        self.intrinsic_matrix = intrinsic_matrix
        self.distortion_coeffs = distortion_coeffs
        self.extrinsic_matrix = extrinsic_matrix or np.eye(4)

    def undistort(self, frame: np.ndarray) -> np.ndarray:
        """Apply undistortion to frame."""
        return cv2.undistort(
            frame,
            self.intrinsic_matrix,
            self.distortion_coeffs,
        )


class Camera:
    """Single camera interface."""

    def __init__(
        self,
        camera_id: int,
        source: int = 0,
        width: int = 1920,
        height: int = 1080,
        fps: int = 30,
    ):
        self.camera_id = camera_id
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.capture: Optional[cv2.VideoCapture] = None
        self.calibration: Optional[CameraCalibration] = None
        self.frame_count = 0
        self.is_running = False
        self._last_frame: Optional[CameraFrame] = None

    async def initialize(self) -> bool:
        """Initialize camera connection."""
        try:
            logger.info(f"initializing_camera_{self.camera_id}", source=self.source)

            # For development, create synthetic frames
            # In production, use: self.capture = cv2.VideoCapture(self.source)
            self.is_running = True

            logger.info(f"camera_{self.camera_id}_initialized")
            return True

        except Exception as e:
            logger.error(f"camera_{self.camera_id}_init_failed", error=str(e))
            return False

    async def get_frame(self) -> Optional[CameraFrame]:
        """Capture a single frame."""
        if not self.is_running:
            return None

        try:
            # For development, create synthetic frame
            # In production, use: ret, frame = self.capture.read()
            frame = self._create_synthetic_frame()

            camera_frame = CameraFrame(
                camera_id=self.camera_id,
                frame=frame,
                timestamp=time.time(),
                frame_number=self.frame_count,
                exposure=0.0,
                gain=1.0,
            )

            self.frame_count += 1
            self._last_frame = camera_frame

            return camera_frame

        except Exception as e:
            logger.error(f"camera_{self.camera_id}_frame_capture_failed", error=str(e))
            return None

    def _create_synthetic_frame(self) -> np.ndarray:
        """Create synthetic frame for development."""
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Add camera ID text
        cv2.putText(
            frame,
            f"Camera {self.camera_id} - Frame {self.frame_count}",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        # Add timestamp
        cv2.putText(
            frame,
            datetime.now().isoformat(),
            (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )

        return frame

    async def stop(self):
        """Stop camera capture."""
        self.is_running = False
        if self.capture:
            self.capture.release()
        logger.info(f"camera_{self.camera_id}_stopped")


class MultiCameraManager:
    """
    Manages multiple cameras with synchronization.
    Ensures frames are captured within sync threshold.
    """

    def __init__(
        self,
        num_cameras: int = 4,
        target_fps: int = 30,
        sync_threshold_ms: float = 5.0,
    ):
        self.num_cameras = num_cameras
        self.target_fps = target_fps
        self.sync_threshold_ms = sync_threshold_ms
        self.cameras: List[Camera] = []
        self.calibrations: Dict[int, CameraCalibration] = {}
        self.is_running = False

    async def initialize(self) -> bool:
        """Initialize all cameras."""
        logger.info("initializing_multi_camera_system", num_cameras=self.num_cameras)

        # Create cameras
        for i in range(self.num_cameras):
            camera = Camera(
                camera_id=i,
                source=i,
                width=settings.camera_width,
                height=settings.camera_height,
                fps=settings.camera_fps,
            )

            if await camera.initialize():
                self.cameras.append(camera)

                # Load calibration (mock for now)
                self.calibrations[i] = self._create_mock_calibration(i)
            else:
                logger.error(f"failed_to_initialize_camera_{i}")

        self.is_running = len(self.cameras) > 0

        logger.info(
            "multi_camera_system_initialized",
            active_cameras=len(self.cameras),
        )

        return self.is_running

    def _create_mock_calibration(self, camera_id: int) -> CameraCalibration:
        """Create mock calibration data."""
        # Mock intrinsic matrix (typical for 1920x1080)
        fx = fy = 1000.0
        cx = settings.camera_width / 2
        cy = settings.camera_height / 2

        intrinsic = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1],
        ], dtype=np.float32)

        # Mock distortion coefficients
        distortion = np.zeros(5, dtype=np.float32)

        return CameraCalibration(
            camera_id=camera_id,
            intrinsic_matrix=intrinsic,
            distortion_coeffs=distortion,
        )

    async def capture_synchronized_frames(self) -> Optional[SynchronizedFrameSet]:
        """
        Capture synchronized frames from all cameras.
        Ensures all frames are within sync threshold.
        """
        if not self.is_running or not self.cameras:
            return None

        # Capture frames concurrently
        frame_tasks = [camera.get_frame() for camera in self.cameras]
        camera_frames = await asyncio.gather(*frame_tasks)

        # Filter out None frames
        valid_frames = [f for f in camera_frames if f is not None]

        if not valid_frames:
            return None

        # Check synchronization quality
        timestamps = [f.timestamp for f in valid_frames]
        max_diff = (max(timestamps) - min(timestamps)) * 1000  # Convert to ms

        sync_quality = 1.0 - min(max_diff / self.sync_threshold_ms, 1.0)

        if max_diff > self.sync_threshold_ms:
            logger.warning(
                "camera_sync_drift_detected",
                max_diff_ms=max_diff,
                threshold_ms=self.sync_threshold_ms,
            )

        return SynchronizedFrameSet(
            frames=valid_frames,
            timestamp=np.mean(timestamps),
            sync_quality=sync_quality,
        )

    async def stream_frames(self):
        """
        Continuously stream synchronized frames.
        Yields frames at target FPS.
        """
        frame_interval = 1.0 / self.target_fps

        while self.is_running:
            start_time = time.time()

            # Capture synchronized frames
            frame_set = await self.capture_synchronized_frames()

            if frame_set:
                yield frame_set

            # Maintain target FPS
            elapsed = time.time() - start_time
            sleep_time = max(0, frame_interval - elapsed)
            await asyncio.sleep(sleep_time)

    async def stop(self):
        """Stop all cameras."""
        logger.info("stopping_multi_camera_system")

        self.is_running = False

        # Stop all cameras concurrently
        stop_tasks = [camera.stop() for camera in self.cameras]
        await asyncio.gather(*stop_tasks)

        logger.info("multi_camera_system_stopped")


# Global camera manager instance
_camera_manager: Optional[MultiCameraManager] = None


async def get_camera_manager() -> MultiCameraManager:
    """Get or create global camera manager."""
    global _camera_manager

    if _camera_manager is None:
        _camera_manager = MultiCameraManager(
            num_cameras=settings.num_cameras,
            target_fps=settings.camera_fps,
            sync_threshold_ms=settings.camera_sync_threshold_ms,
        )
        await _camera_manager.initialize()

    return _camera_manager
