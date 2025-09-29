#!/bin/bash

# AI Cybersecurity Tool Deployment Script
# This script handles deployment to different environments

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$PROJECT_ROOT")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Help function
show_help() {
    cat << EOF
AI Cybersecurity Tool Deployment Script

Usage: $0 [OPTIONS] ENVIRONMENT

ENVIRONMENTS:
    dev         Deploy to development environment
    staging     Deploy to staging environment
    prod        Deploy to production environment

OPTIONS:
    -h, --help              Show this help message
    -b, --build             Build Docker images before deployment
    -p, --push              Push Docker images to registry
    -t, --tag TAG           Use specific tag for Docker images (default: latest)
    -c, --config CONFIG     Use specific config file
    -d, --dry-run           Show what would be deployed without actually deploying
    -v, --verbose           Enable verbose output
    --skip-tests            Skip running tests before deployment
    --skip-migration        Skip database migrations
    --force                 Force deployment even if tests fail

EXAMPLES:
    $0 dev --build
    $0 prod --build --push --tag v1.0.0
    $0 staging --dry-run
    $0 prod --config custom.env

EOF
}

# Default values
ENVIRONMENT=""
BUILD_IMAGES=false
PUSH_IMAGES=false
DOCKER_TAG="latest"
CONFIG_FILE=""
DRY_RUN=false
VERBOSE=false
SKIP_TESTS=false
SKIP_MIGRATION=false
FORCE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -b|--build)
            BUILD_IMAGES=true
            shift
            ;;
        -p|--push)
            PUSH_IMAGES=true
            shift
            ;;
        -t|--tag)
            DOCKER_TAG="$2"
            shift 2
            ;;
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -d|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        --skip-migration)
            SKIP_MIGRATION=true
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        dev|staging|prod)
            ENVIRONMENT="$1"
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate environment
if [[ -z "$ENVIRONMENT" ]]; then
    log_error "Environment is required"
    show_help
    exit 1
fi

if [[ ! "$ENVIRONMENT" =~ ^(dev|staging|prod)$ ]]; then
    log_error "Invalid environment: $ENVIRONMENT. Must be dev, staging, or prod"
    exit 1
fi

# Set default config file if not provided
if [[ -z "$CONFIG_FILE" ]]; then
    CONFIG_FILE="$PROJECT_ROOT/deploy/${ENVIRONMENT}.env"
fi

# Check if config file exists
if [[ ! -f "$CONFIG_FILE" ]]; then
    log_error "Config file not found: $CONFIG_FILE"
    exit 1
fi

# Load environment variables
log_info "Loading configuration from: $CONFIG_FILE"
source "$CONFIG_FILE"

# Set Docker image name
DOCKER_IMAGE="ai-cybersecurity-tool:${DOCKER_TAG}"

# Functions
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    # Check if Docker is running
    if ! docker info &> /dev/null; then
        log_error "Docker is not running"
        exit 1
    fi
    
    # Check if kubectl is installed (for Kubernetes deployment)
    if [[ "$ENVIRONMENT" == "prod" ]] && ! command -v kubectl &> /dev/null; then
        log_warning "kubectl is not installed. Kubernetes deployment will be skipped."
    fi
    
    log_success "Prerequisites check passed"
}

run_tests() {
    if [[ "$SKIP_TESTS" == true ]]; then
        log_warning "Skipping tests"
        return 0
    fi
    
    log_info "Running tests..."
    
    cd "$PROJECT_ROOT"
    
    # Run unit tests
    if ! python -m pytest tests/unit_test.py -v; then
        if [[ "$FORCE" == false ]]; then
            log_error "Unit tests failed"
            exit 1
        else
            log_warning "Unit tests failed but continuing due to --force flag"
        fi
    fi
    
    # Run integration tests
    if ! python -m pytest tests/integration_test.py -v; then
        if [[ "$FORCE" == false ]]; then
            log_error "Integration tests failed"
            exit 1
        else
            log_warning "Integration tests failed but continuing due to --force flag"
        fi
    fi
    
    log_success "Tests completed"
}

build_images() {
    if [[ "$BUILD_IMAGES" == false ]]; then
        log_info "Skipping image build"
        return 0
    fi
    
    log_info "Building Docker images..."
    
    cd "$PROJECT_ROOT"
    
    # Build API image
    log_info "Building API image: $DOCKER_IMAGE"
    if [[ "$VERBOSE" == true ]]; then
        docker build -t "$DOCKER_IMAGE" .
    else
        docker build -t "$DOCKER_IMAGE" . > /dev/null
    fi
    
    # Build frontend image
    FRONTEND_IMAGE="ai-cybersecurity-frontend:${DOCKER_TAG}"
    log_info "Building frontend image: $FRONTEND_IMAGE"
    if [[ "$VERBOSE" == true ]]; then
        docker build -t "$FRONTEND_IMAGE" ./frontend/cybersecurity-dashboard
    else
        docker build -t "$FRONTEND_IMAGE" ./frontend/cybersecurity-dashboard > /dev/null
    fi
    
    log_success "Docker images built successfully"
}

push_images() {
    if [[ "$PUSH_IMAGES" == false ]]; then
        log_info "Skipping image push"
        return 0
    fi
    
    log_info "Pushing Docker images to registry..."
    
    # Check if registry is configured
    if [[ -z "$DOCKER_REGISTRY" ]]; then
        log_warning "DOCKER_REGISTRY not set. Skipping push."
        return 0
    fi
    
    # Push API image
    log_info "Pushing API image to registry"
    docker tag "$DOCKER_IMAGE" "${DOCKER_REGISTRY}/${DOCKER_IMAGE}"
    docker push "${DOCKER_REGISTRY}/${DOCKER_IMAGE}"
    
    # Push frontend image
    FRONTEND_IMAGE="ai-cybersecurity-frontend:${DOCKER_TAG}"
    docker tag "$FRONTEND_IMAGE" "${DOCKER_REGISTRY}/${FRONTEND_IMAGE}"
    docker push "${DOCKER_REGISTRY}/${FRONTEND_IMAGE}"
    
    log_success "Docker images pushed successfully"
}

deploy_docker_compose() {
    log_info "Deploying with Docker Compose..."
    
    cd "$PROJECT_ROOT"
    
    # Select compose file based on environment
    if [[ "$ENVIRONMENT" == "prod" ]]; then
        COMPOSE_FILE="docker-compose.prod.yml"
    else
        COMPOSE_FILE="docker-compose.yml"
    fi
    
    # Update environment variables in compose file
    envsubst < "$CONFIG_FILE" > .env
    
    if [[ "$DRY_RUN" == true ]]; then
        log_info "Dry run - would execute: docker-compose -f $COMPOSE_FILE up -d"
        return 0
    fi
    
    # Stop existing containers
    docker-compose -f "$COMPOSE_FILE" down
    
    # Start new containers
    docker-compose -f "$COMPOSE_FILE" up -d
    
    # Wait for services to be ready
    log_info "Waiting for services to be ready..."
    sleep 10
    
    # Check health
    if curl -f http://localhost:5001/health > /dev/null 2>&1; then
        log_success "API is healthy"
    else
        log_error "API health check failed"
        exit 1
    fi
    
    log_success "Docker Compose deployment completed"
}

deploy_kubernetes() {
    if [[ "$ENVIRONMENT" != "prod" ]]; then
        log_info "Kubernetes deployment only supported for production"
        return 0
    fi
    
    log_info "Deploying to Kubernetes..."
    
    cd "$PROJECT_ROOT"
    
    if [[ "$DRY_RUN" == true ]]; then
        log_info "Dry run - would execute: kubectl apply -f deploy/kubernetes/"
        return 0
    fi
    
    # Apply Kubernetes manifests
    kubectl apply -f deploy/kubernetes/deployment.yaml
    kubectl apply -f deploy/kubernetes/ingress.yaml
    
    # Wait for deployment to be ready
    log_info "Waiting for deployment to be ready..."
    kubectl rollout status deployment/ai-cybersecurity-tool --timeout=300s
    
    log_success "Kubernetes deployment completed"
}

run_migrations() {
    if [[ "$SKIP_MIGRATION" == true ]]; then
        log_warning "Skipping database migrations"
        return 0
    fi
    
    log_info "Running database migrations..."
    
    # Database migrations would go here
    # For now, just create the database directory
    mkdir -p "$PROJECT_ROOT/data"
    
    log_success "Database migrations completed"
}

cleanup() {
    log_info "Cleaning up temporary files..."
    
    # Remove temporary .env file
    if [[ -f "$PROJECT_ROOT/.env" ]]; then
        rm "$PROJECT_ROOT/.env"
    fi
    
    log_success "Cleanup completed"
}

# Main deployment function
main() {
    log_info "Starting deployment to $ENVIRONMENT environment"
    log_info "Docker image: $DOCKER_IMAGE"
    log_info "Config file: $CONFIG_FILE"
    
    if [[ "$DRY_RUN" == true ]]; then
        log_warning "DRY RUN MODE - No actual deployment will occur"
    fi
    
    # Run deployment steps
    check_prerequisites
    run_tests
    build_images
    push_images
    run_migrations
    
    # Deploy based on environment
    if [[ "$ENVIRONMENT" == "prod" ]]; then
        deploy_kubernetes
    else
        deploy_docker_compose
    fi
    
    cleanup
    
    log_success "Deployment to $ENVIRONMENT completed successfully!"
    
    # Show access information
    if [[ "$ENVIRONMENT" == "prod" ]]; then
        log_info "Production deployment accessible at: https://api.yourdomain.com"
    else
        log_info "Development deployment accessible at: http://localhost:5001"
        log_info "Frontend accessible at: http://localhost:3000"
    fi
}

# Trap to ensure cleanup on exit
trap cleanup EXIT

# Run main function
main
