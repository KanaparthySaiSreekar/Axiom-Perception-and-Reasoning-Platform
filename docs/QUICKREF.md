# Axiom Platform - Quick Reference Card

## 🚀 Common Commands

### Docker Compose Operations

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f [service-name]

# Stop all services
docker-compose down

# Rebuild and restart
docker-compose up -d --build

# Scale backend
docker-compose up -d --scale backend=3

# View service status
docker-compose ps

# Execute command in container
docker exec -it axiom-backend bash
```

### Service URLs

| Service | URL | Default Credentials |
|---------|-----|---------------------|
| Frontend Dashboard | http://localhost:3000 | operator/operator123 |
| Backend API Docs | http://localhost:8000/docs | (use JWT token) |
| Grafana | http://localhost:3001 | admin/admin |
| Prometheus | http://localhost:9091 | (no auth) |
| Backend Health | http://localhost:8000/api/v1/health | (public) |

### API Quick Examples

**Login:**
```bash
curl -X POST http://localhost:8000/api/v1/auth/login \
  -d "username=operator&password=operator123"

# Save token
export TOKEN="<your-jwt-token>"
```

**Get Cameras:**
```bash
curl http://localhost:8000/api/v1/cameras \
  -H "Authorization: Bearer $TOKEN"
```

**Execute Natural Language Command:**
```bash
curl -X POST http://localhost:8000/api/v1/robot/llm/command \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"command":"Pick up the bottle"}'
```

**Get System Metrics:**
```bash
curl http://localhost:8000/metrics
```

## 🔧 Troubleshooting Quick Fixes

| Problem | Quick Fix |
|---------|-----------|
| Service won't start | `docker-compose down && docker-compose up -d` |
| Out of memory | `docker system prune -a` |
| Database errors | `docker-compose restart postgres` |
| Ollama not responding | `docker exec -it axiom-ollama ollama pull llama3.2:latest` |
| Frontend 404 errors | Check `NEXT_PUBLIC_API_URL` in `.env.local` |
| High latency | Enable GPU: `USE_GPU=true` in backend/.env |
| Can't login | Reset: `docker-compose restart backend` |

## 📊 Key Performance Metrics

**Target Values:**
- Perception Pipeline: <90ms
- LLM Response: <300ms
- Camera FPS: ≥30
- End-to-End: <150ms

**Check Metrics:**
```bash
# System health
curl http://localhost:8000/api/v1/health/detailed

# Model performance
curl http://localhost:8000/api/v1/models/performance

# Layer latencies
curl http://localhost:8000/api/v1/models/layers
```

## 🔒 Default Users & Permissions

| User | Password | Permissions |
|------|----------|-------------|
| admin | admin123 | All permissions |
| operator | operator123 | Control robot, view diagnostics |
| observer | observer123 | View only |

**Change passwords in production!**

## 📝 File Locations

```
backend/
  app/main.py              - Main FastAPI app
  app/core/config.py       - Configuration
  app/models/pipeline.py   - 6-layer DL pipeline
  app/llm/service.py       - LLM integration
  .env                     - Environment config

frontend/
  src/app/page.tsx         - Main dashboard
  src/components/          - UI components
  .env.local               - Frontend config

infrastructure/
  nginx/nginx.conf         - Reverse proxy config
  prometheus/              - Monitoring config

docker-compose.yml         - Service orchestration
```

## 🎯 Quick Tasks

**Add a new camera:**
```bash
# Edit docker-compose.yml
NUM_CAMERAS=5
# Restart backend
docker-compose restart backend
```

**Change LLM model:**
```bash
docker exec -it axiom-ollama ollama pull mistral
# Update backend/.env: OLLAMA_MODEL=mistral
docker-compose restart backend
```

**Enable GPU:**
```bash
# In backend/.env
USE_GPU=true
GPU_DEVICE=0
# Restart
docker-compose restart backend
```

**View logs:**
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend

# Last 100 lines
docker-compose logs --tail=100 backend
```

## 📞 Getting Help

1. Check logs: `docker-compose logs -f`
2. Health endpoint: `curl http://localhost:8000/api/v1/health/detailed`
3. GitHub Issues: https://github.com/yourusername/axiom/issues
4. Documentation: `docs/` directory
5. Setup Guide: `docs/SETUP_GUIDE.md`

## 🚨 Emergency Procedures

**System Unresponsive:**
```bash
docker-compose restart
```

**Complete Reset (destroys data):**
```bash
docker-compose down -v
docker-compose up -d
```

**Backup Database:**
```bash
docker exec axiom-postgres pg_dump -U axiom axiom > backup.sql
```

**Restore Database:**
```bash
cat backup.sql | docker exec -i axiom-postgres psql -U axiom axiom
```

---

**Quick Start in 60 Seconds:**

```bash
git clone <repo-url>
cd Axiom-Perception-and-Reasoning-Platform
cp backend/.env.example backend/.env
docker-compose up -d
# Wait 30 seconds
open http://localhost:3000
# Login: operator/operator123
```

---

**Version**: 1.0.0
**Last Updated**: 2024-01-01
