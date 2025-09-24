#!/bin/bash

# AI Cybersecurity Tool - Docker Setup Script
# This script installs Docker and Docker Compose on macOS

set -e

echo "🐳 AI Cybersecurity Tool - Docker Setup"
echo "========================================"

# Check if Docker is already installed
if command -v docker &> /dev/null; then
    echo "✅ Docker is already installed"
    docker --version
else
    echo "📦 Installing Docker Desktop for macOS..."
    
    # Check if Homebrew is installed
    if ! command -v brew &> /dev/null; then
        echo "❌ Homebrew is required but not installed"
        echo "Please install Homebrew first: https://brew.sh/"
        exit 1
    fi
    
    # Install Docker Desktop
    brew install --cask docker
    
    echo "✅ Docker Desktop installed"
    echo "⚠️  Please start Docker Desktop from Applications folder"
    echo "⚠️  Then run this script again to continue setup"
    exit 0
fi

# Check if Docker Compose is available
if docker compose version &> /dev/null; then
    echo "✅ Docker Compose is available"
    docker compose version
else
    echo "❌ Docker Compose not available"
    echo "Please ensure Docker Desktop is running"
    exit 1
fi

# Check if Docker daemon is running
if ! docker info &> /dev/null; then
    echo "❌ Docker daemon is not running"
    echo "Please start Docker Desktop and try again"
    exit 1
fi

echo "✅ Docker is ready!"

# Test the Docker configuration
echo ""
echo "🧪 Testing Docker Configuration..."
echo "================================="

# Build the API image
echo "Building API image..."
docker build -t ai-cybersecurity-api .

# Build the frontend image
echo "Building frontend image..."
docker build -t ai-cybersecurity-frontend ./frontend/cybersecurity-dashboard

echo ""
echo "✅ Docker setup complete!"
echo ""
echo "🚀 Quick Start Commands:"
echo "  make dev     - Start development environment"
echo "  make prod    - Start production environment"
echo "  make help    - Show all available commands"
echo ""
echo "📚 Documentation:"
echo "  DOCKER.md    - Complete Docker deployment guide"
echo ""
echo "🌐 Services will be available at:"
echo "  API: http://localhost:5001"
echo "  Dashboard: http://localhost:3000"
