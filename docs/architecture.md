# Axiom Robotic AI Platform - System Architecture

## Overview
Full-stack robotic AI platform enabling real-time multimodal reasoning, multi-camera perception, prediction modules, and operator-facing controls.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FRONTEND (React)                          │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────┐   │
│  │  Multi-Cam  │  │  NL Console  │  │  Control Panel      │   │
│  │  Dashboard  │  │  & LLM UI    │  │  & Diagnostics      │   │
│  └─────────────┘  └──────────────┘  └─────────────────────┘   │
└────────────┬──────────────┬─────────────────┬──────────────────┘
             │              │                 │
        WebSocket         REST             WebSocket
             │              │                 │
┌────────────┴──────────────┴─────────────────┴──────────────────┐
│                    BACKEND (FastAPI + gRPC)                      │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              API Gateway & Load Balancer                  │  │
│  └────┬─────────────────┬──────────────────┬────────────────┘  │
│       │                 │                  │                    │
│  ┌────▼─────┐     ┌────▼─────┐      ┌────▼──────┐            │
│  │  gRPC    │     │   REST   │      │ WebSocket │            │
│  │  Service │     │   API    │      │  Streams  │            │
│  └────┬─────┘     └────┬─────┘      └────┬──────┘            │
│       │                │                   │                    │
│  ┌────┴────────────────┴───────────────────┴──────────────┐   │
│  │                   Core Engine                            │   │
│  │  ┌──────────────┐  ┌────────────┐  ┌────────────────┐ │   │
│  │  │  Perception  │  │    LLM     │  │   Prediction   │ │   │
│  │  │   Pipeline   │  │  Reasoning │  │    Modules     │ │   │
│  │  └──────────────┘  └────────────┘  └────────────────┘ │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │            6-Layer Deep Learning Pipeline                 │  │
│  │  Layer 1: Normalization & Segmentation                    │  │
│  │  Layer 2: Object Detection & Tracking                     │  │
│  │  Layer 3: Depth Estimation & Reconstruction              │  │
│  │  Layer 4: Liquid Boundary & Contour Modeling             │  │
│  │  Layer 5: Temporal Modeling for Liquid Dynamics          │  │
│  │  Layer 6: Short-Horizon Trajectory Prediction            │  │
│  └──────────────────────────────────────────────────────────┘  │
└───────────────────────────┬───────────────────────────────────┘
                            │
                   ┌────────┴────────┐
                   │  Multi-Camera   │
                   │  Input System   │
                   └─────────────────┘
```

## Technology Stack

### Backend
- **Framework**: FastAPI (async, high-performance)
- **gRPC**: High-throughput robotic control
- **WebSocket**: Real-time telemetry streaming
- **Deep Learning**: PyTorch, OpenCV, ONNX Runtime
- **LLM**: Ollama (local), Hugging Face Transformers
- **Vision Models**: YOLO, SAM, MiDaS, GroundingDINO

### Frontend
- **Framework**: Next.js 14 (React 18)
- **UI Library**: shadcn/ui + Tailwind CSS
- **Real-time**: WebSocket + React Query
- **Visualization**: Three.js, D3.js, Plotly
- **State Management**: Zustand

### Infrastructure
- **Containerization**: Docker + Docker Compose
- **Orchestration**: Kubernetes (optional)
- **Reverse Proxy**: Nginx
- **Monitoring**: Prometheus + Grafana
- **CI/CD**: GitHub Actions

## Data Flow

### Perception Pipeline
1. **Camera Ingestion** (30+ FPS)
   - Multi-camera synchronized capture
   - Calibration & pose alignment
   - Time synchronization (<5ms drift)

2. **6-Layer Processing** (<90ms total)
   - Frame normalization & segmentation (10ms)
   - Object detection & tracking (15ms)
   - Depth estimation (20ms)
   - Liquid boundary detection (15ms)
   - Temporal modeling (15ms)
   - Trajectory prediction (15ms)

3. **Scene Fusion**
   - Unified 3D scene generation
   - Multi-view geometry
   - Uncertainty quantification

### LLM Reasoning Flow
1. Natural language input from operator
2. Context augmentation with perception data
3. LLM inference (<300ms)
4. Structured command generation
5. Safety validation
6. Pre-execution preview
7. Robot controller dispatch

### Prediction Modules
1. **Liquid-Level Measurement**
   - Contour analysis + depth mapping
   - ≥95% accuracy requirement
   - Real-time confidence intervals

2. **Trajectory Prediction**
   - 1-3 second horizon
   - Physics-informed neural networks
   - Uncertainty propagation

## API Specifications

### gRPC Services
```protobuf
service RobotControl {
  rpc ExecuteAction(ActionRequest) returns (ActionResponse);
  rpc StreamTelemetry(Empty) returns (stream TelemetryData);
  rpc EmergencyStop(Empty) returns (StopResponse);
}
```

### REST Endpoints
- `POST /api/v1/auth/login` - Authentication
- `GET /api/v1/cameras` - Camera configuration
- `POST /api/v1/perception/config` - Update perception settings
- `GET /api/v1/health` - System health check
- `GET /api/v1/models/status` - Model performance metrics

### WebSocket Channels
- `/ws/video/{camera_id}` - Video stream
- `/ws/perception` - Perception overlay data
- `/ws/telemetry` - Robot telemetry
- `/ws/predictions` - Prediction outputs
- `/ws/llm` - LLM reasoning trace

## Performance Requirements

| Component | Requirement | Budget |
|-----------|-------------|--------|
| Camera FPS | ≥30 FPS | - |
| Perception latency | <90ms | Total pipeline |
| LLM latency | <300ms | End-to-end |
| Frontend update | ≥20 Hz | UI refresh |
| End-to-end latency | <150ms | Perception→UI |
| API response | <50ms | p95 |

## Security Architecture

### Authentication & Authorization
- JWT-based authentication
- Role-based access control (Admin, Operator, Observer)
- Action-level permissions
- API key management for external services

### Network Security
- TLS/SSL encryption for all connections
- WebSocket secure (WSS)
- gRPC with TLS
- Rate limiting on all endpoints

### Safety Constraints
- Command validation before execution
- Emergency stop override
- Action logging and audit trail
- Geofencing for robot workspace

## Scalability

### Horizontal Scaling
- Stateless API servers
- Load balancing across instances
- Distributed camera processing
- Model serving with multiple replicas

### Vertical Scaling
- GPU acceleration for deep learning
- Multi-threaded camera ingestion
- Async I/O for network operations

## Monitoring & Observability

### Metrics
- Per-layer latency (p50, p95, p99)
- Model accuracy drift detection
- Camera sync drift monitoring
- API request rates and errors
- Frontend render times

### Logging
- Structured JSON logging
- Distributed tracing (OpenTelemetry)
- Error aggregation and alerting
- Audit trail for robot actions

### Dashboards
- System health overview
- Performance metrics
- Model quality trends
- Resource utilization
- Incident timeline

## Deployment Architecture

### Development
```
docker-compose up
- Backend API: http://localhost:8000
- Frontend: http://localhost:3000
- Ollama: http://localhost:11434
```

### Production
```
Kubernetes cluster
- Multiple backend replicas
- Frontend CDN deployment
- Load balancer (Nginx Ingress)
- Persistent storage for models
- Redis for caching
```

## Open Source Components

### Deep Learning Models
- **YOLO v8/v9**: Object detection
- **Segment Anything (SAM)**: Segmentation
- **MiDaS**: Depth estimation
- **GroundingDINO**: Open-vocabulary detection
- **RAFT**: Optical flow

### LLM Options
- **Ollama**: Local deployment (Llama 3.2, Mistral, Phi-3)
- **Hugging Face**: Transformers library (free tier)
- **vLLM**: High-throughput inference server

### Supporting Libraries
- **OpenCV**: Computer vision
- **PyTorch**: Deep learning framework
- **ONNX Runtime**: Optimized inference
- **FastAPI**: Web framework
- **gRPC**: RPC framework

## Future Enhancements
1. Multi-robot coordination
2. Reinforcement learning integration
3. Digital twin simulation
4. Edge deployment optimization
5. Federated learning for privacy
