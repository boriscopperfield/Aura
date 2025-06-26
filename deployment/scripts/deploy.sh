#!/bin/bash

# AURA Deployment Script
# This script deploys AURA to a Kubernetes cluster

set -e

# Configuration
NAMESPACE="aura"
IMAGE_TAG="${IMAGE_TAG:-latest}"
REGISTRY="${REGISTRY:-your-registry.com}"
IMAGE_NAME="${REGISTRY}/aura:${IMAGE_TAG}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if kubectl is installed
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed or not in PATH"
        exit 1
    fi
    
    # Check if docker is installed
    if ! command -v docker &> /dev/null; then
        log_error "docker is not installed or not in PATH"
        exit 1
    fi
    
    # Check if we can connect to the cluster
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    log_info "Prerequisites check passed"
}

# Build and push Docker image
build_and_push_image() {
    log_info "Building Docker image..."
    
    # Build the image
    docker build -t "${IMAGE_NAME}" .
    
    if [ $? -ne 0 ]; then
        log_error "Failed to build Docker image"
        exit 1
    fi
    
    log_info "Pushing Docker image to registry..."
    
    # Push the image
    docker push "${IMAGE_NAME}"
    
    if [ $? -ne 0 ]; then
        log_error "Failed to push Docker image"
        exit 1
    fi
    
    log_info "Docker image built and pushed successfully"
}

# Create namespace
create_namespace() {
    log_info "Creating namespace..."
    
    kubectl apply -f deployment/kubernetes/namespace.yaml
    
    # Wait for namespace to be ready
    kubectl wait --for=condition=Ready namespace/${NAMESPACE} --timeout=60s
    
    log_info "Namespace created successfully"
}

# Deploy secrets
deploy_secrets() {
    log_info "Deploying secrets..."
    
    # Check if secrets already exist
    if kubectl get secret aura-secrets -n ${NAMESPACE} &> /dev/null; then
        log_warn "Secrets already exist. Skipping secret creation."
        log_warn "Please update secrets manually if needed."
    else
        kubectl apply -f deployment/kubernetes/secrets.yaml
        log_info "Secrets deployed successfully"
    fi
}

# Deploy storage
deploy_storage() {
    log_info "Deploying storage..."
    
    kubectl apply -f deployment/kubernetes/storage.yaml
    
    # Wait for PVCs to be bound
    log_info "Waiting for PVCs to be bound..."
    kubectl wait --for=condition=Bound pvc/aura-workspace-pvc -n ${NAMESPACE} --timeout=300s
    kubectl wait --for=condition=Bound pvc/aura-logs-pvc -n ${NAMESPACE} --timeout=300s
    kubectl wait --for=condition=Bound pvc/redis-data-pvc -n ${NAMESPACE} --timeout=300s
    kubectl wait --for=condition=Bound pvc/postgres-data-pvc -n ${NAMESPACE} --timeout=300s
    
    log_info "Storage deployed successfully"
}

# Deploy configuration
deploy_config() {
    log_info "Deploying configuration..."
    
    kubectl apply -f deployment/kubernetes/configmap.yaml
    
    log_info "Configuration deployed successfully"
}

# Deploy databases
deploy_databases() {
    log_info "Deploying databases..."
    
    # Deploy PostgreSQL
    kubectl apply -f deployment/kubernetes/postgres.yaml
    
    # Deploy Redis
    kubectl apply -f deployment/kubernetes/redis.yaml
    
    # Wait for databases to be ready
    log_info "Waiting for databases to be ready..."
    kubectl wait --for=condition=Available deployment/aura-postgres -n ${NAMESPACE} --timeout=300s
    kubectl wait --for=condition=Available deployment/aura-redis -n ${NAMESPACE} --timeout=300s
    
    log_info "Databases deployed successfully"
}

# Deploy main application
deploy_application() {
    log_info "Deploying AURA application..."
    
    # Update image in deployment
    sed -i.bak "s|image: aura:latest|image: ${IMAGE_NAME}|g" deployment/kubernetes/deployment.yaml
    
    kubectl apply -f deployment/kubernetes/deployment.yaml
    
    # Restore original deployment file
    mv deployment/kubernetes/deployment.yaml.bak deployment/kubernetes/deployment.yaml
    
    # Wait for deployment to be ready
    log_info "Waiting for application to be ready..."
    kubectl wait --for=condition=Available deployment/aura-main -n ${NAMESPACE} --timeout=600s
    
    log_info "Application deployed successfully"
}

# Verify deployment
verify_deployment() {
    log_info "Verifying deployment..."
    
    # Check pod status
    kubectl get pods -n ${NAMESPACE}
    
    # Check service status
    kubectl get services -n ${NAMESPACE}
    
    # Check ingress status
    kubectl get ingress -n ${NAMESPACE}
    
    # Test health endpoint
    log_info "Testing health endpoint..."
    
    # Port forward to test locally
    kubectl port-forward service/aura-main-service 8080:80 -n ${NAMESPACE} &
    PORT_FORWARD_PID=$!
    
    sleep 5
    
    if curl -f http://localhost:8080/health &> /dev/null; then
        log_info "Health check passed"
    else
        log_warn "Health check failed - application may still be starting"
    fi
    
    # Clean up port forward
    kill $PORT_FORWARD_PID 2>/dev/null || true
    
    log_info "Deployment verification completed"
}

# Cleanup function
cleanup() {
    log_info "Cleaning up..."
    
    # Kill any background processes
    jobs -p | xargs -r kill 2>/dev/null || true
}

# Set up cleanup trap
trap cleanup EXIT

# Main deployment function
deploy() {
    log_info "Starting AURA deployment..."
    
    check_prerequisites
    build_and_push_image
    create_namespace
    deploy_secrets
    deploy_storage
    deploy_config
    deploy_databases
    deploy_application
    verify_deployment
    
    log_info "AURA deployment completed successfully!"
    log_info "Access your AURA instance at: https://aura.yourdomain.com"
    log_info "Monitor with: kubectl get pods -n ${NAMESPACE} -w"
}

# Rollback function
rollback() {
    log_info "Rolling back AURA deployment..."
    
    kubectl rollout undo deployment/aura-main -n ${NAMESPACE}
    kubectl wait --for=condition=Available deployment/aura-main -n ${NAMESPACE} --timeout=300s
    
    log_info "Rollback completed successfully"
}

# Uninstall function
uninstall() {
    log_warn "This will completely remove AURA from the cluster!"
    read -p "Are you sure? (y/N): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_info "Uninstalling AURA..."
        
        kubectl delete namespace ${NAMESPACE}
        
        log_info "AURA uninstalled successfully"
    else
        log_info "Uninstall cancelled"
    fi
}

# Parse command line arguments
case "${1:-deploy}" in
    deploy)
        deploy
        ;;
    rollback)
        rollback
        ;;
    uninstall)
        uninstall
        ;;
    *)
        echo "Usage: $0 {deploy|rollback|uninstall}"
        exit 1
        ;;
esac