# Docker Deployment Guide

This guide covers how to deploy the AI Cybersecurity Tool using Docker containers.

## Quick Start

### Development Environment
```bash
# Build and start development environment
make dev

# Or manually:
docker-compose build
docker-compose up -d
```

### Production Environment
```bash
# Build and start production environment
make prod

# Or manually:
docker-compose -f docker-compose.prod.yml build
docker-compose -f docker-compose.prod.yml up -d
```

## Services

### API Service (Port 5001)
- **Container**: `ai-cybersecurity-api`
- **Health Check**: `http://localhost:5001/health`
- **Endpoints**:
  - `GET /health` - Health check
  - `POST /predict` - Threat prediction
  - `GET /stats` - System statistics
  - `GET /alerts` - Recent alerts
  - `GET /system/info` - System information

### Frontend Service (Port 3000)
- **Container**: `ai-cybersecurity-dashboard`
- **Health Check**: `http://localhost:3000`
- **Features**: Real-time dashboard with WebSocket integration

### Nginx Reverse Proxy (Port 80/443)
- **Container**: `ai-cybersecurity-nginx` (production only)
- **Features**: Load balancing, SSL termination, rate limiting
- **Health Check**: `http://localhost/health`

## Docker Commands

### Development
```bash
make build      # Build images
make up         # Start services
make down       # Stop services
make logs       # View logs
make test-dev   # Test environment
```

### Production
```bash
make build-prod # Build production images
make up-prod    # Start production services
make down-prod  # Stop production services
make test-prod  # Test production environment
```

### Maintenance
```bash
make clean      # Clean up Docker resources
make logs-api   # View API logs only
make logs-frontend # View frontend logs only
```

## Configuration

### Environment Variables
- `FLASK_ENV=production` - Flask environment
- `FLASK_APP=api/app.py` - Flask application
- `PYTHONPATH=/app` - Python path
- `REACT_APP_API_URL=http://localhost:5001` - API URL for frontend

### Volumes
- `./models:/app/models:ro` - Read-only model files
- `./data:/app/data:ro` - Read-only data files
- `./logs:/app/logs` - Log files
- `logs:/app/logs` - Persistent log volume (production)

### Networks
- `cybersecurity-network` - Bridge network for service communication

## Health Checks

All services include health checks:
- **API**: HTTP GET to `/health` endpoint
- **Frontend**: HTTP GET to root endpoint
- **Nginx**: HTTP GET to `/health` endpoint

## Resource Limits (Production)

### API Service
- **Memory**: 2GB limit, 512MB reservation
- **CPU**: 1.0 limit, 0.5 reservation

### Frontend Service
- **Memory**: 512MB limit, 256MB reservation
- **CPU**: 0.5 limit, 0.25 reservation

## Security Features

### Nginx Configuration
- Rate limiting (10 req/s for API, 30 req/s for frontend)
- Security headers (X-Frame-Options, X-XSS-Protection, etc.)
- WebSocket support for real-time communication

### Container Security
- Non-root user execution
- Read-only model and data volumes
- Health checks for service monitoring
- Resource limits to prevent resource exhaustion

## Troubleshooting

### Common Issues

1. **Port Conflicts**
   ```bash
   # Check what's using the ports
   lsof -i :5001
   lsof -i :3000
   ```

2. **Build Failures**
   ```bash
   # Clean and rebuild
   make clean
   make build
   ```

3. **Service Not Starting**
   ```bash
   # Check logs
   make logs
   # Check health
   make test-dev
   ```

4. **Model Files Missing**
   ```bash
   # Ensure models are in the correct location
   ls -la models/
   # Should contain: baseline_model.pkl, scaler.pkl, feature_names.pkl
   ```

### Logs
```bash
# View all logs
make logs

# View specific service logs
make logs-api
make logs-frontend

# Follow logs in real-time
docker-compose logs -f
```

## Performance Monitoring

### Resource Usage
```bash
# Check container resource usage
docker stats

# Check specific container
docker stats ai-cybersecurity-api
```

### Health Monitoring
```bash
# Test API health
curl http://localhost:5001/health

# Test frontend
curl http://localhost:3000

# Test nginx (production)
curl http://localhost/health
```

## Scaling

### Horizontal Scaling
```bash
# Scale API service
docker-compose -f docker-compose.prod.yml up -d --scale api=3

# Scale frontend service
docker-compose -f docker-compose.prod.yml up -d --scale frontend=2
```

### Load Balancing
The nginx configuration automatically load balances between multiple API and frontend instances.

## Backup and Recovery

### Data Backup
```bash
# Backup models and data
tar -czf backup-$(date +%Y%m%d).tar.gz models/ data/

# Backup logs
docker-compose exec api tar -czf /app/logs/backup-$(date +%Y%m%d).tar.gz /app/logs/
```

### Recovery
```bash
# Restore from backup
tar -xzf backup-YYYYMMDD.tar.gz

# Restart services
make up-prod
```
