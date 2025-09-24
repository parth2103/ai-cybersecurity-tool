# AI Cybersecurity Tool - Docker Operations

.PHONY: help build up down logs clean test-dev test-prod

# Default target
help:
	@echo "AI Cybersecurity Tool - Docker Commands"
	@echo "======================================"
	@echo ""
	@echo "Development:"
	@echo "  make build      - Build Docker images"
	@echo "  make up         - Start development environment"
	@echo "  make down       - Stop development environment"
	@echo "  make logs       - View logs"
	@echo "  make test-dev   - Test development environment"
	@echo ""
	@echo "Production:"
	@echo "  make build-prod - Build production images"
	@echo "  make up-prod    - Start production environment"
	@echo "  make down-prod  - Stop production environment"
	@echo "  make test-prod  - Test production environment"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean      - Clean up Docker resources"
	@echo "  make logs-api   - View API logs only"
	@echo "  make logs-frontend - View frontend logs only"

# Development environment
build:
	@echo "Building development Docker images..."
	docker-compose build

up:
	@echo "Starting development environment..."
	docker-compose up -d
	@echo "Services started:"
	@echo "  API: http://localhost:5001"
	@echo "  Dashboard: http://localhost:3000"

down:
	@echo "Stopping development environment..."
	docker-compose down

logs:
	docker-compose logs -f

logs-api:
	docker-compose logs -f api

logs-frontend:
	docker-compose logs -f frontend

# Production environment
build-prod:
	@echo "Building production Docker images..."
	docker-compose -f docker-compose.prod.yml build

up-prod:
	@echo "Starting production environment..."
	docker-compose -f docker-compose.prod.yml up -d
	@echo "Services started:"
	@echo "  API: http://localhost:5001"
	@echo "  Dashboard: http://localhost:3000"
	@echo "  Nginx: http://localhost:80"

down-prod:
	@echo "Stopping production environment..."
	docker-compose -f docker-compose.prod.yml down

# Testing
test-dev:
	@echo "Testing development environment..."
	@echo "Testing API health..."
	@curl -f http://localhost:5001/health || echo "API health check failed"
	@echo "Testing frontend..."
	@curl -f http://localhost:3000 || echo "Frontend health check failed"
	@echo "Testing threat detection..."
	@python3 test_comprehensive_threats.py

test-prod:
	@echo "Testing production environment..."
	@echo "Testing API health..."
	@curl -f http://localhost:5001/health || echo "API health check failed"
	@echo "Testing frontend..."
	@curl -f http://localhost:3000 || echo "Frontend health check failed"
	@echo "Testing nginx..."
	@curl -f http://localhost/health || echo "Nginx health check failed"

# Cleanup
clean:
	@echo "Cleaning up Docker resources..."
	docker-compose down -v
	docker-compose -f docker-compose.prod.yml down -v
	docker system prune -f
	docker volume prune -f

# Quick start for development
dev: build up
	@echo "Development environment ready!"
	@echo "API: http://localhost:5001"
	@echo "Dashboard: http://localhost:3000"

# Quick start for production
prod: build-prod up-prod
	@echo "Production environment ready!"
	@echo "API: http://localhost:5001"
	@echo "Dashboard: http://localhost:3000"
	@echo "Nginx: http://localhost:80"
