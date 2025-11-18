"""
Trajectory prediction module.
Predicts motion for 1-3 second horizon with uncertainty estimation.
"""
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from collections import deque

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TrajectoryPoint:
    """Single point in predicted trajectory."""
    position: np.ndarray  # [x, y, z]
    velocity: np.ndarray  # [vx, vy, vz]
    timestamp: float
    uncertainty: float


@dataclass
class TrajectoryPrediction:
    """Complete trajectory prediction."""
    object_id: int
    object_class: str
    current_position: np.ndarray
    current_velocity: np.ndarray
    predicted_points: List[TrajectoryPoint]
    horizon_sec: float
    confidence: float
    prediction_method: str


class PhysicsBasedPredictor:
    """
    Physics-informed trajectory prediction.
    Uses kinematic models with gravity and friction.
    """

    def __init__(self):
        self.gravity = np.array([0.0, 0.0, -9.81])  # m/s^2
        self.air_friction_coeff = 0.05
        logger.info("physics_based_predictor_initialized")

    def predict(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        dt: float = 0.033,  # 30 FPS
        num_steps: int = 90,  # 3 seconds at 30 FPS
    ) -> List[TrajectoryPoint]:
        """
        Predict trajectory using physics simulation.

        Args:
            position: Current position [x, y, z]
            velocity: Current velocity [vx, vy, vz]
            dt: Time step in seconds
            num_steps: Number of prediction steps

        Returns:
            List of predicted trajectory points
        """

        trajectory = []

        pos = position.copy()
        vel = velocity.copy()
        t = 0.0

        for step in range(num_steps):
            # Physics update
            # F = ma -> a = F/m
            # F = gravity + friction

            friction = -self.air_friction_coeff * vel

            acceleration = self.gravity + friction

            # Update velocity and position (Euler integration)
            vel = vel + acceleration * dt
            pos = pos + vel * dt

            t += dt

            # Uncertainty increases with prediction horizon
            uncertainty = 0.01 * (step + 1)  # Linear growth

            trajectory.append(TrajectoryPoint(
                position=pos.copy(),
                velocity=vel.copy(),
                timestamp=t,
                uncertainty=uncertainty,
            ))

        return trajectory


class LearningBasedPredictor:
    """
    Learning-based trajectory prediction.
    Uses historical data to predict motion patterns.
    """

    def __init__(self, history_length: int = 30):
        self.history_length = history_length
        self.position_history: deque = deque(maxlen=history_length)
        self.velocity_history: deque = deque(maxlen=history_length)
        logger.info("learning_based_predictor_initialized")

    def update_history(self, position: np.ndarray, velocity: np.ndarray):
        """Update tracking history."""
        self.position_history.append(position.copy())
        self.velocity_history.append(velocity.copy())

    def predict(
        self,
        dt: float = 0.033,
        num_steps: int = 90,
    ) -> Optional[List[TrajectoryPoint]]:
        """
        Predict trajectory using learned patterns.

        Args:
            dt: Time step in seconds
            num_steps: Number of prediction steps

        Returns:
            List of predicted trajectory points or None if insufficient data
        """

        if len(self.position_history) < 5:
            return None

        trajectory = []

        # Estimate current position and velocity from history
        positions = np.array(list(self.position_history))
        velocities = np.array(list(self.velocity_history))

        current_pos = positions[-1]
        current_vel = velocities[-1]

        # Estimate acceleration from velocity history
        if len(velocities) >= 2:
            accel = (velocities[-1] - velocities[-2]) / dt
        else:
            accel = np.zeros(3)

        # Predict with constant acceleration model
        pos = current_pos.copy()
        vel = current_vel.copy()
        t = 0.0

        for step in range(num_steps):
            # Update with learned acceleration
            vel = vel + accel * dt
            pos = pos + vel * dt
            t += dt

            # Uncertainty based on history variance
            pos_variance = np.var(positions, axis=0).mean()
            uncertainty = np.sqrt(pos_variance) * (step + 1) * 0.1

            trajectory.append(TrajectoryPoint(
                position=pos.copy(),
                velocity=vel.copy(),
                timestamp=t,
                uncertainty=uncertainty,
            ))

        return trajectory


class HybridTrajectoryPredictor:
    """
    Hybrid predictor combining physics-based and learning-based approaches.
    Provides 1-3 second prediction horizon with uncertainty quantification.
    """

    def __init__(self, horizon_sec: float = 3.0):
        self.horizon_sec = horizon_sec
        self.physics_predictor = PhysicsBasedPredictor()
        self.learning_predictors: dict = {}  # One per tracked object

        logger.info(
            "hybrid_trajectory_predictor_initialized",
            horizon_sec=horizon_sec
        )

    def predict_trajectory(
        self,
        object_id: int,
        object_class: str,
        position: np.ndarray,
        velocity: np.ndarray,
        fps: float = 30.0,
    ) -> TrajectoryPrediction:
        """
        Predict object trajectory.

        Args:
            object_id: Unique object identifier
            object_class: Object class (affects prediction method)
            position: Current position [x, y, z]
            velocity: Current velocity [vx, vy, vz]
            fps: Frame rate for prediction steps

        Returns:
            TrajectoryPrediction with confidence and uncertainty
        """

        dt = 1.0 / fps
        num_steps = int(self.horizon_sec * fps)

        # Get or create learning predictor for this object
        if object_id not in self.learning_predictors:
            self.learning_predictors[object_id] = LearningBasedPredictor()

        learning_predictor = self.learning_predictors[object_id]
        learning_predictor.update_history(position, velocity)

        # Get predictions from both methods
        physics_trajectory = self.physics_predictor.predict(
            position, velocity, dt, num_steps
        )

        learning_trajectory = learning_predictor.predict(dt, num_steps)

        # Fuse predictions or use best available
        if learning_trajectory and len(learning_predictor.position_history) >= 10:
            # Weighted fusion based on confidence in learning
            confidence = min(len(learning_predictor.position_history) / 30.0, 0.9)

            fused_trajectory = self._fuse_trajectories(
                physics_trajectory,
                learning_trajectory,
                physics_weight=1.0 - confidence,
                learning_weight=confidence,
            )

            method = "hybrid_fusion"
            final_confidence = 0.8 + confidence * 0.15

        else:
            # Use physics-based only
            fused_trajectory = physics_trajectory
            method = "physics_based"
            final_confidence = 0.75

        # Apply object-specific adjustments
        if object_class == "liquid":
            # Liquids have higher uncertainty
            for point in fused_trajectory:
                point.uncertainty *= 1.5
            final_confidence *= 0.9

        prediction = TrajectoryPrediction(
            object_id=object_id,
            object_class=object_class,
            current_position=position,
            current_velocity=velocity,
            predicted_points=fused_trajectory,
            horizon_sec=self.horizon_sec,
            confidence=final_confidence,
            prediction_method=method,
        )

        logger.debug(
            "trajectory_predicted",
            object_id=object_id,
            object_class=object_class,
            horizon_sec=self.horizon_sec,
            confidence=final_confidence,
            method=method,
        )

        return prediction

    def _fuse_trajectories(
        self,
        traj1: List[TrajectoryPoint],
        traj2: List[TrajectoryPoint],
        physics_weight: float,
        learning_weight: float,
    ) -> List[TrajectoryPoint]:
        """Fuse two trajectory predictions with weighted average."""

        fused = []

        for p1, p2 in zip(traj1, traj2):
            fused_pos = (
                physics_weight * p1.position +
                learning_weight * p2.position
            )

            fused_vel = (
                physics_weight * p1.velocity +
                learning_weight * p2.velocity
            )

            fused_uncertainty = np.sqrt(
                physics_weight * p1.uncertainty**2 +
                learning_weight * p2.uncertainty**2
            )

            fused.append(TrajectoryPoint(
                position=fused_pos,
                velocity=fused_vel,
                timestamp=p1.timestamp,
                uncertainty=fused_uncertainty,
            ))

        return fused

    def get_prediction_at_time(
        self,
        prediction: TrajectoryPrediction,
        time_sec: float,
    ) -> Optional[TrajectoryPoint]:
        """Get predicted state at specific time in future."""

        if time_sec > prediction.horizon_sec:
            return None

        # Find closest prediction point
        for point in prediction.predicted_points:
            if abs(point.timestamp - time_sec) < 0.05:  # 50ms tolerance
                return point

        return None

    def visualize_trajectory(
        self,
        prediction: TrajectoryPrediction
    ) -> dict:
        """
        Generate visualization data for trajectory.

        Returns:
            Dictionary with points and uncertainty bounds for rendering
        """

        positions = [p.position for p in prediction.predicted_points]
        uncertainties = [p.uncertainty for p in prediction.predicted_points]

        # Generate uncertainty bounds (tube around trajectory)
        upper_bound = [pos + unc for pos, unc in zip(positions, uncertainties)]
        lower_bound = [pos - unc for pos, unc in zip(positions, uncertainties)]

        return {
            "trajectory_line": positions,
            "uncertainty_upper": upper_bound,
            "uncertainty_lower": lower_bound,
            "confidence": prediction.confidence,
            "horizon_sec": prediction.horizon_sec,
        }


# Global predictor instance
_predictor: Optional[HybridTrajectoryPredictor] = None


def get_trajectory_predictor() -> HybridTrajectoryPredictor:
    """Get or create global trajectory predictor."""
    global _predictor

    if _predictor is None:
        _predictor = HybridTrajectoryPredictor(
            horizon_sec=settings.trajectory_horizon_sec
        )

    return _predictor
