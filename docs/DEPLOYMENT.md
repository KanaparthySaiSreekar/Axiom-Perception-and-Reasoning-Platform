# Deployment Guide

This guide covers deploying the Axiom Robotic AI Platform in production environments.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Docker Compose Deployment](#docker-compose-deployment)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Production Configuration](#production-configuration)
- [Security Hardening](#security-hardening)
- [Monitoring Setup](#monitoring-setup)
- [Backup & Recovery](#backup--recovery)

## Prerequisites

### Hardware Requirements

**Minimum (Development)**:
- CPU: 8 cores
- RAM: 16 GB
- Storage: 100 GB SSD
- GPU: Optional (CPU mode available)

**Recommended (Production)**:
- CPU: 16+ cores
- RAM: 64 GB
- Storage: 500 GB NVMe SSD
- GPU: NVIDIA GPU with 8+ GB VRAM (RTX 3080 or better)
- Network: 1 Gbps+

### Software Requirements

- Docker 24.0+
- Docker Compose 2.20+
- NVIDIA Docker Runtime (for GPU support)
- Linux kernel 5.4+ (Ubuntu 22.04 LTS recommended)

## Docker Compose Deployment

### 1. Clone Repository

```bash
git clone <repository-url>
cd Axiom-Perception-and-Reasoning-Platform
```

### 2. Configure Environment

```bash
# Copy and edit backend environment
cp backend/.env.example backend/.env
nano backend/.env

# Important: Change these values
SECRET_KEY=<generate-secure-random-key>
DATABASE_URL=postgresql+asyncpg://axiom:<secure-password>@postgres:5432/axiom
```

### 3. Pull Ollama Model

```bash
# Start only Ollama service
docker-compose up -d ollama

# Wait for Ollama to start
sleep 10

# Pull LLM model
docker exec -it axiom-ollama ollama pull llama3.2:latest

# Verify model is available
docker exec -it axiom-ollama ollama list
```

### 4. Start All Services

```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

### 5. Verify Deployment

```bash
# Check backend health
curl http://localhost:8000/api/v1/health

# Check frontend
curl http://localhost:3000

# Check Grafana
curl http://localhost:3001
```

## Production Configuration

### Environment Variables

**Backend (`backend/.env`)**:

```bash
# Security
SECRET_KEY=<strong-random-key-min-32-chars>
DEBUG=false
ENVIRONMENT=production

# Database
DATABASE_URL=postgresql+asyncpg://axiom:<secure-password>@postgres:5432/axiom

# LLM
OLLAMA_HOST=http://ollama:11434
OLLAMA_MODEL=llama3.2:latest

# Performance
WORKERS=4
USE_GPU=true
BATCH_SIZE=4

# Monitoring
LOG_LEVEL=INFO
LOG_FORMAT=json
```

### Docker Compose Production Overrides

Create `docker-compose.prod.yml`:

```yaml
version: '3.8'

services:
  backend:
    restart: always
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '4'
          memory: 8G
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  frontend:
    restart: always
    deploy:
      replicas: 2
      resources:
        limits:
          cpus: '2'
          memory: 2G

  postgres:
    restart: always
    volumes:
      - /data/axiom/postgres:/var/lib/postgresql/data

  redis:
    restart: always
    volumes:
      - /data/axiom/redis:/data

  nginx:
    restart: always
    ports:
      - "443:443"
    volumes:
      - ./infrastructure/nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - /etc/letsencrypt:/etc/letsencrypt:ro
```

Deploy with:

```bash
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

## Security Hardening

### 1. SSL/TLS Configuration

```bash
# Install Certbot
sudo apt install certbot

# Get SSL certificate
sudo certbot certonly --standalone -d your-domain.com

# Update nginx configuration
# Add SSL directives to infrastructure/nginx/nginx.conf
```

### 2. Firewall Rules

```bash
# Allow only necessary ports
sudo ufw allow 22/tcp   # SSH
sudo ufw allow 80/tcp   # HTTP
sudo ufw allow 443/tcp  # HTTPS
sudo ufw enable
```

### 3. Secrets Management

```bash
# Use Docker secrets (Swarm mode)
echo "<secret-key>" | docker secret create axiom_secret_key -

# Update docker-compose.yml to use secrets
```

### 4. Network Security

```yaml
# In docker-compose.yml
networks:
  axiom-network:
    driver: bridge
    internal: false  # Set to true for isolated network
    ipam:
      config:
        - subnet: 172.28.0.0/16
```

### 5. User Permissions

```bash
# Run containers as non-root
# Add to Dockerfile:
USER nobody:nogroup
```

## Monitoring Setup

### Prometheus Configuration

**`infrastructure/prometheus/prometheus.yml`**:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'axiom-backend'
    static_configs:
      - targets: ['backend:9090']

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']

  - job_name: 'postgres-exporter'
    static_configs:
      - targets: ['postgres-exporter:9187']
```

### Grafana Dashboards

1. Access Grafana: http://localhost:3001
2. Login: admin/admin
3. Import dashboards:
   - Axiom System Overview (provided in repo)
   - Node Exporter Full
   - PostgreSQL Database

### Alerting Rules

**`infrastructure/prometheus/alerts.yml`**:

```yaml
groups:
  - name: axiom_alerts
    interval: 30s
    rules:
      - alert: HighLatency
        expr: perception_latency_ms > 100
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Perception pipeline latency exceeded threshold"

      - alert: ServiceDown
        expr: up == 0
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Service {{ $labels.job }} is down"
```

## Backup & Recovery

### Database Backup

```bash
# Create backup script
cat > /usr/local/bin/axiom-backup.sh << 'EOF'
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR=/backups/axiom

# Database backup
docker exec axiom-postgres pg_dump -U axiom axiom | \
  gzip > $BACKUP_DIR/axiom_db_$DATE.sql.gz

# Model weights backup
tar -czf $BACKUP_DIR/models_$DATE.tar.gz models/weights/

# Keep last 7 days
find $BACKUP_DIR -mtime +7 -delete
EOF

chmod +x /usr/local/bin/axiom-backup.sh

# Add to crontab (daily at 2 AM)
echo "0 2 * * * /usr/local/bin/axiom-backup.sh" | crontab -
```

### Recovery

```bash
# Restore database
gunzip < /backups/axiom/axiom_db_20240101_020000.sql.gz | \
  docker exec -i axiom-postgres psql -U axiom axiom

# Restore model weights
tar -xzf /backups/axiom/models_20240101_020000.tar.gz
```

## Scaling

### Horizontal Scaling

```bash
# Scale backend instances
docker-compose up -d --scale backend=5

# Scale frontend instances
docker-compose up -d --scale frontend=3
```

### Load Balancing

Configure Nginx for load balancing:

```nginx
upstream backend_cluster {
    least_conn;
    server backend_1:8000;
    server backend_2:8000;
    server backend_3:8000;
}

server {
    location /api/ {
        proxy_pass http://backend_cluster;
    }
}
```

## Health Checks

### Automated Health Monitoring

```bash
# Create health check script
cat > /usr/local/bin/axiom-health.sh << 'EOF'
#!/bin/bash

# Check backend
if ! curl -f http://localhost:8000/api/v1/health; then
    echo "Backend health check failed"
    # Send alert
fi

# Check frontend
if ! curl -f http://localhost:3000; then
    echo "Frontend health check failed"
fi

# Check GPU
nvidia-smi || echo "GPU not available"
EOF

chmod +x /usr/local/bin/axiom-health.sh

# Run every 5 minutes
echo "*/5 * * * * /usr/local/bin/axiom-health.sh" | crontab -
```

## Troubleshooting Production Issues

### High Latency

```bash
# Check GPU utilization
nvidia-smi

# Check backend logs
docker-compose logs backend | grep latency

# Monitor metrics
curl http://localhost:8000/metrics
```

### Memory Issues

```bash
# Check memory usage
docker stats

# Adjust memory limits in docker-compose.yml
# Restart services
docker-compose restart backend
```

### Database Connection Issues

```bash
# Check postgres logs
docker-compose logs postgres

# Check connections
docker exec axiom-postgres psql -U axiom -c "SELECT count(*) FROM pg_stat_activity;"

# Increase connection pool size in backend/.env
```

## Disaster Recovery

### Complete System Restore

```bash
# 1. Restore from backups
./restore-backup.sh

# 2. Rebuild and start services
docker-compose build --no-cache
docker-compose up -d

# 3. Verify health
./axiom-health.sh

# 4. Check all services
docker-compose ps
```

## Maintenance

### Update Procedure

```bash
# 1. Backup current state
./axiom-backup.sh

# 2. Pull latest code
git pull origin main

# 3. Rebuild images
docker-compose build

# 4. Rolling update (zero downtime)
docker-compose up -d --no-deps --build backend
docker-compose up -d --no-deps --build frontend

# 5. Verify deployment
./axiom-health.sh
```

### Log Rotation

```bash
# Configure Docker logging
# Add to docker-compose.yml:
logging:
  driver: "json-file"
  options:
    max-size: "10m"
    max-file: "3"
```

## Performance Tuning

### GPU Optimization

```bash
# Set GPU persistence mode
nvidia-smi -pm 1

# Set power limit (adjust based on your GPU)
nvidia-smi -pl 300

# Monitor GPU metrics
nvidia-smi dmon -s pucvmet
```

### Database Optimization

```sql
-- PostgreSQL tuning
ALTER SYSTEM SET shared_buffers = '4GB';
ALTER SYSTEM SET effective_cache_size = '12GB';
ALTER SYSTEM SET maintenance_work_mem = '1GB';
ALTER SYSTEM SET checkpoint_completion_target = '0.9';
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = '100';
ALTER SYSTEM SET random_page_cost = '1.1';
ALTER SYSTEM SET effective_io_concurrency = '200';
ALTER SYSTEM SET work_mem = '104857kB';
ALTER SYSTEM SET min_wal_size = '1GB';
ALTER SYSTEM SET max_wal_size = '4GB';
SELECT pg_reload_conf();
```

## Support

For deployment issues:
- GitHub Issues: <repository-url>/issues
- Documentation: docs/
- Email: support@axiom-platform.ai (placeholder)

---

Last updated: 2024-01-01
