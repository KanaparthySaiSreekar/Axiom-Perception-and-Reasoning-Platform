/**
 * Type definitions for Axiom Platform
 */

export enum UserRole {
  ADMIN = 'admin',
  OPERATOR = 'operator',
  OBSERVER = 'observer',
}

export interface User {
  username: string;
  email: string;
  full_name: string;
  role: UserRole;
}

export interface AuthToken {
  access_token: string;
  token_type: string;
  role: UserRole;
  expires_in: number;
}

export interface CameraInfo {
  id: number;
  name: string;
  resolution: [number, number];
  fps: number;
  status: string;
  calibrated: boolean;
  pose: {
    position: number[];
    rotation: number[];
  };
}

export interface Detection {
  id: number;
  class: string;
  confidence: number;
  bbox: number[]; // [x, y, w, h]
  center: number[];
  track_id?: number;
}

export interface PerceptionOutput {
  timestamp: string;
  frame_id: number;
  detections: Detection[];
  segmentation: {
    available: boolean;
    classes: string[];
  };
  depth_map: {
    available: boolean;
    min_depth: number;
    max_depth: number;
  };
  liquid_level: {
    detected: boolean;
    level_mm: number;
    confidence: number;
  };
  trajectory_prediction: {
    available: boolean;
    horizon_sec: number;
    confidence: number;
  };
}

export interface RobotTelemetry {
  timestamp: string;
  joint_positions: number[];
  joint_velocities: number[];
  joint_torques: number[];
  end_effector_pose: {
    position: number[];
    orientation: number[];
  };
  gripper_state: string;
  status: string;
}

export interface RobotAction {
  action_type: string;
  parameters: Record<string, any>;
  safety_override?: boolean;
}

export interface StructuredCommand {
  action: RobotAction;
  reasoning: string;
  safety_validated: boolean;
  estimated_duration_sec: number;
}

export interface ModelInfo {
  name: string;
  version: string;
  framework: string;
  status: string;
  loaded: boolean;
  device: string;
}

export interface ModelPerformance {
  model_name: string;
  accuracy: number;
  latency_ms: number;
  throughput_fps: number;
  gpu_memory_mb: number;
  accuracy_drift: number;
}

export interface LayerMetrics {
  layer_name: string;
  latency_p50_ms: number;
  latency_p95_ms: number;
  latency_p99_ms: number;
  accuracy: number;
}

export interface HealthStatus {
  status: string;
  timestamp: string;
  uptime_seconds: number;
  environment: string;
  version: string;
  services?: Record<string, any>;
}
