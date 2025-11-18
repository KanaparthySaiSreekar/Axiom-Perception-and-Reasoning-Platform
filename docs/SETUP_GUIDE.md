# Axiom Platform - Detailed Setup & Configuration Guide

## 📦 Installation Methods

### Method 1: Docker Compose (Production-Ready)

This is the recommended method for most users. It includes all services pre-configured and ready to run.

####Step-by-Step Installation

**1. Install Prerequisites**

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y docker.io docker-compose git curl

# For GPU support
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt update && sudo apt install -y nvidia-docker2
sudo systemctl restart docker

# Verify GPU access
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

**2. Clone Repository**

```bash
git clone https://github.com/yourusername/Axiom-Perception-and-Reasoning-Platform.git
cd Axiom-Perception-and-Reasoning-Platform
```

**3. Configure Environment**

```bash
# Backend configuration
cp backend/.env.example backend/.env

# Edit with your settings
nano backend/.env
```

**Important Environment Variables:**

```bash
# MUST CHANGE THESE for production
SECRET_KEY=<generate-with: openssl rand -hex 32>
DATABASE_URL=postgresql+asyncpg://axiom:<strong-password>@postgres:5432/axiom

# Optional: GPU configuration
USE_GPU=true
GPU_DEVICE=0  # Use first GPU

# Optional: Camera configuration
NUM_CAMERAS=4  # Adjust based on your setup
CAMERA_FPS=30

# Optional: LLM model selection
OLLAMA_MODEL=llama3.2:latest  # or mistral, phi-3, etc.
```

**4. Pull and Setup Ollama LLM**

```bash
# Start Ollama service only
docker-compose up -d ollama

# Wait for it to initialize
sleep 10

# Pull the LLM model (this may take 5-10 minutes)
docker exec -it axiom-ollama ollama pull llama3.2:latest

# Verify model is available
docker exec -it axiom-ollama ollama list

# Expected output:
# NAME                    ID              SIZE
# llama3.2:latest        abc123...       4.7 GB
```

**Alternative Models:**

```bash
# Smaller, faster model (recommended for limited hardware)
docker exec -it axiom-ollama ollama pull phi-3

# Larger, more capable model
docker exec -it axiom-ollama ollama pull llama3:70b

# Code-specialized model
docker exec -it axiom-ollama ollama pull codellama
```

**5. Start All Services**

```bash
# Start all services in background
docker-compose up -d

# View logs (optional)
docker-compose logs -f

# Check service status
docker-compose ps
```

**Expected Output:**

```
NAME                    IMAGE                  STATUS
axiom-backend           axiom-backend:latest   Up (healthy)
axiom-frontend          axiom-frontend:latest  Up
axiom-postgres          postgres:16-alpine     Up (healthy)
axiom-redis             redis:7-alpine         Up (healthy)
axiom-ollama            ollama/ollama:latest   Up (healthy)
axiom-nginx             nginx:alpine           Up
axiom-prometheus        prom/prometheus        Up
axiom-grafana           grafana/grafana        Up
```

**6. Verify Installation**

```bash
# Check backend health
curl http://localhost:8000/api/v1/health

# Expected response:
# {"status":"healthy","timestamp":"2024-01-01T00:00:00Z",...}

# Check frontend
curl -I http://localhost:3000

# Expected: HTTP/1.1 200 OK

# Check Ollama
curl http://localhost:11434/api/tags

# Check Grafana
curl -I http://localhost:3001

# Check all at once
./scripts/verify-installation.sh  # Provided in repo
```

**7. Access the Platform**

Open your browser and navigate to:

- **Main Dashboard**: http://localhost:3000
- **API Documentation**: http://localhost:8000/docs
- **Grafana Dashboards**: http://localhost:3001
- **Prometheus Metrics**: http://localhost:9091

**8. Login**

Use one of the default accounts:

| Username | Password | Role | Capabilities |
|----------|----------|------|-------------|
| `admin` | `admin123` | Admin | Full system access, user management |
| `operator` | `operator123` | Operator | Robot control, view diagnostics |
| `observer` | `observer123` | Observer | Read-only access |

⚠️ **IMPORTANT**: Change these passwords immediately in production!

---

### Method 2: Manual Installation (Development)

For developers who want more control or are modifying the codebase.

#### Backend Setup

**1. Install Python Dependencies**

```bash
cd backend

# Create virtual environment
python3.11 -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

**2. Setup PostgreSQL**

```bash
# Install PostgreSQL
sudo apt install postgresql-16

# Create database and user
sudo -u postgres psql << EOF
CREATE DATABASE axiom;
CREATE USER axiom WITH PASSWORD 'your_password_here';
GRANT ALL PRIVILEGES ON DATABASE axiom TO axiom;
\q
EOF
```

**3. Setup Redis**

```bash
# Install Redis
sudo apt install redis-server

# Start Redis
sudo systemctl start redis
sudo systemctl enable redis

# Verify
redis-cli ping
# Expected: PONG
```

**4. Setup Ollama**

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull model
ollama pull llama3.2:latest

# Start Ollama server (runs on port 11434 by default)
ollama serve &

# Verify
curl http://localhost:11434/api/tags
```

**5. Configure Backend**

```bash
# Copy environment template
cp .env.example .env

# Edit configuration
nano .env
```

Update these values:

```bash
DATABASE_URL=postgresql+asyncpg://axiom:your_password_here@localhost:5432/axiom
REDIS_URL=redis://localhost:6379/0
OLLAMA_HOST=http://localhost:11434
USE_GPU=true  # false if no GPU
```

**6. Run Database Migrations** (if applicable)

```bash
# Apply migrations
alembic upgrade head
```

**7. Start Backend Server**

```bash
# Development mode (with auto-reload)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Production mode
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

**8. Verify Backend**

```bash
# Health check
curl http://localhost:8000/api/v1/health

# API docs
open http://localhost:8000/docs
```

#### Frontend Setup

**1. Install Node.js**

```bash
# Using nvm (recommended)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
source ~/.bashrc
nvm install 18
nvm use 18

# Verify
node --version  # Should show v18.x.x
npm --version   # Should show v9.x.x or higher
```

**2. Install Dependencies**

```bash
cd frontend

# Install packages
npm install

# or with pnpm (faster)
npm install -g pnpm
pnpm install
```

**3. Configure Environment**

```bash
# Create environment file
cat > .env.local << EOF
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000
EOF
```

**4. Start Development Server**

```bash
# Development mode (with hot reload)
npm run dev

# Production build and start
npm run build
npm start
```

**5. Verify Frontend**

```bash
# Check frontend
curl http://localhost:3000

# Open in browser
open http://localhost:3000
```

---

## 🔧 Configuration Guide

### Backend Configuration Reference

Complete `.env` file with all available options:

```bash
# ============================================
# SERVER CONFIGURATION
# ============================================
HOST=0.0.0.0
PORT=8000
WORKERS=4
DEBUG=false
ENVIRONMENT=production

# ============================================
# SECURITY
# ============================================
# Generate with: openssl rand -hex 32
SECRET_KEY=your-super-secret-key-min-32-characters
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# ============================================
# DATABASE
# ============================================
DATABASE_URL=postgresql+asyncpg://axiom:password@localhost:5432/axiom
# Connection pool settings
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=10

# ============================================
# CACHE (Redis)
# ============================================
REDIS_URL=redis://localhost:6379/0
REDIS_MAX_CONNECTIONS=50

# ============================================
# LLM CONFIGURATION
# ============================================
# Ollama (local deployment)
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.2:latest
LLM_MAX_TOKENS=2048
LLM_TEMPERATURE=0.7
LLM_TIMEOUT=300

# Hugging Face (fallback)
HF_TOKEN=your_huggingface_token_here
HF_MODEL=meta-llama/Llama-2-7b-chat-hf

# ============================================
# CAMERA SYSTEM
# ============================================
NUM_CAMERAS=4
CAMERA_FPS=30
CAMERA_WIDTH=1920
CAMERA_HEIGHT=1080
CAMERA_SYNC_THRESHOLD_MS=5.0

# Camera sources (0=webcam, rtsp://... for IP cameras)
CAMERA_0_SOURCE=0
CAMERA_1_SOURCE=1
CAMERA_2_SOURCE=2
CAMERA_3_SOURCE=3

# ============================================
# PERCEPTION PIPELINE
# ============================================
PERCEPTION_FPS=30
PERCEPTION_LATENCY_BUDGET_MS=90
USE_GPU=true
GPU_DEVICE=0
BATCH_SIZE=1

# Model precision (fp16 for faster inference, fp32 for accuracy)
MODEL_PRECISION=fp16

# ============================================
# MODEL PATHS
# ============================================
MODEL_DIR=../models/weights
YOLO_MODEL=yolov8n.pt
SAM_MODEL=sam_vit_h.pth
DEPTH_MODEL=midas_v21.pt

# ============================================
# PREDICTION MODULES
# ============================================
LIQUID_LEVEL_ACCURACY_TARGET=0.95
TRAJECTORY_HORIZON_SEC=3.0

# ============================================
# WEBSOCKET
# ============================================
WS_MAX_CONNECTIONS=100
WS_HEARTBEAT_INTERVAL=30

# ============================================
# gRPC
# ============================================
GRPC_PORT=50051
GRPC_MAX_WORKERS=10

# ============================================
# MONITORING
# ============================================
PROMETHEUS_PORT=9090
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT=json  # json or console

# ============================================
# ROBOT CONTROL
# ============================================
# Workspace bounds: [x_min, x_max, y_min, y_max, z_min, z_max]
ROBOT_WORKSPACE_BOUNDS=[-1.0,1.0,-1.0,1.0,0.0,2.0]
EMERGENCY_STOP_ENABLED=true
ACTION_RATE_LIMIT=10
```

### Frontend Configuration Reference

`.env.local` file options:

```bash
# API Endpoints
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000

# Feature Flags
NEXT_PUBLIC_ENABLE_3D_VIEWER=true
NEXT_PUBLIC_ENABLE_VOICE_INPUT=false

# UI Configuration
NEXT_PUBLIC_DEFAULT_THEME=dark
NEXT_PUBLIC_CAMERA_GRID_ROWS=2
NEXT_PUBLIC_CAMERA_GRID_COLS=2

# Performance
NEXT_PUBLIC_WS_RECONNECT_DELAY=5000
NEXT_PUBLIC_MAX_MESSAGE_HISTORY=100
```

---

## 📚 Usage Examples

### Example 1: Basic Natural Language Command

```bash
# Via API
curl -X POST http://localhost:8000/api/v1/robot/llm/command \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"command": "Pick up the red bottle", "reasoning": ""}'

# Response:
{
  "action": {
    "action_type": "pick",
    "parameters": {
      "object": "bottle",
      "color": "red",
      "position": [0.5, 0.2, 0.8],
      "approach_vector": [0, 0, -1]
    }
  },
  "reasoning": "Detected red bottle at position (0.5, 0.2, 0.8). Planning top-down grasp.",
  "safety_validated": true,
  "estimated_duration_sec": 3.5
}
```

### Example 2: Monitor Perception Output

```javascript
// Frontend WebSocket client
const ws = new WebSocket('ws://localhost:8000/ws/perception');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Frame:', data.frame_id);
  console.log('Detections:', data.detections);
  console.log('Liquid Level:', data.liquid_level);
};
```

### Example 3: Get System Performance

```bash
# Get detailed health status
curl http://localhost:8000/api/v1/health/detailed \
  -H "Authorization: Bearer YOUR_TOKEN"

# Get model performance metrics
curl http://localhost:8000/api/v1/models/performance \
  -H "Authorization: Bearer YOUR_TOKEN"

# Get per-layer metrics
curl http://localhost:8000/api/v1/models/layers \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### Example 4: Camera Calibration

```python
# Python calibration example
import requests

API_URL = "http://localhost:8000"
TOKEN = "your_jwt_token"

headers = {"Authorization": f"Bearer {TOKEN}"}

# Get camera info
response = requests.get(f"{API_URL}/api/v1/cameras/0", headers=headers)
camera = response.json()

# Trigger calibration
response = requests.post(f"{API_URL}/api/v1/cameras/0/calibrate", headers=headers)
print(response.json())
```

---

## 🧪 Testing the System

### Automated Test Suite

```bash
# Backend tests
cd backend
pytest tests/ -v --cov=app

# Frontend tests
cd frontend
npm test

# Integration tests
./scripts/run-integration-tests.sh
```

### Manual Testing Checklist

- [ ] All services start without errors
- [ ] Can login with default credentials
- [ ] Camera feeds display in dashboard
- [ ] Natural language commands process correctly
- [ ] Robot telemetry updates in real-time
- [ ] Emergency stop functions
- [ ] Grafana dashboards load
- [ ] Perception pipeline shows detections
- [ ] Liquid level measurements appear
- [ ] No error messages in logs

---

## 📊 Performance Tuning

### GPU Optimization

```bash
# Set persistent mode
sudo nvidia-smi -pm 1

# Set power limit (adjust for your GPU)
sudo nvidia-smi -pl 300

# Monitor GPU usage
watch -n 1 nvidia-smi
```

### Database Optimization

```sql
-- PostgreSQL performance tuning
ALTER SYSTEM SET shared_buffers = '4GB';
ALTER SYSTEM SET effective_cache_size = '12GB';
ALTER SYSTEM SET maintenance_work_mem = '1GB';
ALTER SYSTEM SET work_mem = '104857kB';
SELECT pg_reload_conf();
```

### Backend Scaling

```yaml
# docker-compose.override.yml
services:
  backend:
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '4'
          memory: 8G
```

---

## 🚨 Troubleshooting Guide

### Issue: Ollama Not Responding

**Symptoms**: Backend logs show "LLM service unavailable"

**Solutions**:

```bash
# Check Ollama status
docker-compose logs ollama

# Restart Ollama
docker-compose restart ollama

# Re-pull model
docker exec -it axiom-ollama ollama pull llama3.2:latest

# Check model list
docker exec -it axiom-ollama ollama list

# Test Ollama directly
curl http://localhost:11434/api/tags
```

### Issue: High Perception Latency

**Symptoms**: Latency > 100ms in diagnostics

**Solutions**:

1. **Enable GPU acceleration**:
```bash
# In backend/.env
USE_GPU=true
GPU_DEVICE=0

# Verify GPU access
nvidia-smi
```

2. **Reduce model size**:
```bash
# Use nano variant of YOLO
YOLO_MODEL=yolov8n.pt  # Instead of yolov8m.pt
```

3. **Adjust batch size**:
```bash
BATCH_SIZE=1  # Reduce if latency is high
```

4. **Lower camera resolution**:
```bash
CAMERA_WIDTH=1280
CAMERA_HEIGHT=720
```

### Issue: Frontend Can't Connect

**Symptoms**: "Connection Error" on dashboard

**Solutions**:

```bash
# Check backend is running
curl http://localhost:8000/api/v1/health

# Check backend logs
docker-compose logs backend

# Verify CORS settings
# In backend/app/main.py, ensure allow_origins includes frontend URL

# Check network connectivity
docker-compose ps
docker network inspect axiom-network
```

### Issue: Database Connection Errors

**Symptoms**: "Could not connect to database"

**Solutions**:

```bash
# Check PostgreSQL status
docker-compose logs postgres

# Verify database exists
docker exec -it axiom-postgres psql -U axiom -l

# Reset database (WARNING: destroys data)
docker-compose down -v
docker-compose up -d postgres
```

### Issue: Out of GPU Memory

**Symptoms**: CUDA out of memory errors

**Solutions**:

```bash
# Reduce batch size
BATCH_SIZE=1

# Use smaller models
YOLO_MODEL=yolov8n.pt  # Smallest variant

# Enable model quantization
MODEL_PRECISION=fp16

# Reduce camera count
NUM_CAMERAS=2
```

---

## 📖 Additional Resources

- **Architecture Documentation**: `docs/architecture.md`
- **Deployment Guide**: `docs/DEPLOYMENT.md`
- **API Reference**: http://localhost:8000/docs
- **Model Documentation**: `docs/models/`
- **Contributing Guidelines**: `CONTRIBUTING.md`

---

## 🎓 Training & Tutorials

### Tutorial 1: Your First Robot Command

```bash
1. Start the platform: `docker-compose up -d`
2. Open dashboard: http://localhost:3000
3. Login as operator: operator / operator123
4. Open Natural Language Console
5. Type: "Go to home position"
6. Click "Execute Command"
7. Watch robot telemetry update
```

### Tutorial 2: Adding a Custom Camera

```bash
# Edit docker-compose.yml
services:
  backend:
    environment:
      NUM_CAMERAS: 5
      CAMERA_4_SOURCE: "rtsp://192.168.1.100:554/stream"

# Restart
docker-compose restart backend
```

### Tutorial 3: Creating Custom Dashboards

```bash
1. Access Grafana: http://localhost:3001
2. Login: admin / admin
3. Click "+" → "Import"
4. Upload dashboard JSON from `infrastructure/grafana/dashboards/`
5. Select Prometheus datasource
6. Click "Import"
```

---

**Last Updated**: 2024-01-01
**Version**: 1.0.0
**Support**: GitHub Issues
