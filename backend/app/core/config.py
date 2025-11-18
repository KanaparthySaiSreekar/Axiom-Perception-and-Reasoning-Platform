"""
Core configuration management for Axiom platform.
"""
from typing import List, Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # Server Configuration
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    workers: int = Field(default=4, description="Number of worker processes")
    debug: bool = Field(default=False, description="Debug mode")
    environment: str = Field(default="production", description="Environment")

    # Security
    secret_key: str = Field(..., description="Secret key for JWT")
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    access_token_expire_minutes: int = Field(default=30, description="Token expiry")

    # Database
    database_url: str = Field(
        default="postgresql+asyncpg://user:password@localhost:5432/axiom",
        description="Database connection URL"
    )
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL"
    )

    # LLM Configuration
    ollama_host: str = Field(
        default="http://localhost:11434",
        description="Ollama server URL"
    )
    ollama_model: str = Field(
        default="llama3.2:latest",
        description="Ollama model name"
    )
    llm_max_tokens: int = Field(default=2048, description="Max LLM tokens")
    llm_temperature: float = Field(default=0.7, description="LLM temperature")
    llm_timeout: int = Field(default=300, description="LLM timeout in seconds")

    # Hugging Face
    hf_token: Optional[str] = Field(default=None, description="Hugging Face API token")
    hf_model: str = Field(
        default="meta-llama/Llama-2-7b-chat-hf",
        description="Hugging Face model"
    )

    # Camera Configuration
    num_cameras: int = Field(default=4, description="Number of cameras")
    camera_fps: int = Field(default=30, description="Camera FPS")
    camera_width: int = Field(default=1920, description="Camera width")
    camera_height: int = Field(default=1080, description="Camera height")
    camera_sync_threshold_ms: float = Field(
        default=5.0,
        description="Camera sync threshold in ms"
    )

    # Perception Pipeline
    perception_fps: int = Field(default=30, description="Target perception FPS")
    perception_latency_budget_ms: int = Field(
        default=90,
        description="Perception latency budget in ms"
    )
    use_gpu: bool = Field(default=True, description="Use GPU acceleration")
    gpu_device: int = Field(default=0, description="GPU device ID")
    batch_size: int = Field(default=1, description="Batch size for inference")

    # Model Paths
    model_dir: str = Field(default="../models/weights", description="Model directory")
    yolo_model: str = Field(default="yolov8n.pt", description="YOLO model file")
    sam_model: str = Field(default="sam_vit_h.pth", description="SAM model file")
    depth_model: str = Field(default="midas_v21.pt", description="Depth model file")

    # Prediction
    liquid_level_accuracy_target: float = Field(
        default=0.95,
        description="Target accuracy for liquid level"
    )
    trajectory_horizon_sec: float = Field(
        default=3.0,
        description="Trajectory prediction horizon in seconds"
    )

    # WebSocket
    ws_max_connections: int = Field(
        default=100,
        description="Max WebSocket connections"
    )
    ws_heartbeat_interval: int = Field(
        default=30,
        description="WebSocket heartbeat interval in seconds"
    )

    # gRPC
    grpc_port: int = Field(default=50051, description="gRPC server port")
    grpc_max_workers: int = Field(default=10, description="Max gRPC workers")

    # Monitoring
    prometheus_port: int = Field(default=9090, description="Prometheus port")
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(default="json", description="Log format")

    # Robot Control
    robot_workspace_bounds: List[float] = Field(
        default=[-1.0, 1.0, -1.0, 1.0, 0.0, 2.0],
        description="Robot workspace boundaries [x_min, x_max, y_min, y_max, z_min, z_max]"
    )
    emergency_stop_enabled: bool = Field(
        default=True,
        description="Emergency stop enabled"
    )
    action_rate_limit: int = Field(
        default=10,
        description="Max actions per second"
    )

    @field_validator('robot_workspace_bounds')
    @classmethod
    def validate_workspace_bounds(cls, v):
        """Validate workspace bounds."""
        if len(v) != 6:
            raise ValueError("Workspace bounds must have 6 values")
        if v[0] >= v[1] or v[2] >= v[3] or v[4] >= v[5]:
            raise ValueError("Invalid workspace bounds")
        return v

    @property
    def yolo_model_path(self) -> str:
        """Get full path to YOLO model."""
        return f"{self.model_dir}/{self.yolo_model}"

    @property
    def sam_model_path(self) -> str:
        """Get full path to SAM model."""
        return f"{self.model_dir}/{self.sam_model}"

    @property
    def depth_model_path(self) -> str:
        """Get full path to depth model."""
        return f"{self.model_dir}/{self.depth_model}"


# Global settings instance
settings = Settings()
