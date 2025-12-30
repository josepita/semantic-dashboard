#!/bin/bash
# ============================================================================
# Manual Deployment Script
# ============================================================================
# This script deploys the application manually (without CI/CD)
#
# Usage:
#   chmod +x scripts/deploy.sh
#   ./scripts/deploy.sh
# ============================================================================

set -e

echo "============================================================================"
echo "Embedding Insights Suite - Manual Deployment"
echo "============================================================================"

# ============================================================================
# Configuration
# ============================================================================
APP_DIR="/opt/embedding-insights-suite"

# Change to app directory
if [ ! -d "$APP_DIR" ]; then
    echo "Error: Application directory not found: $APP_DIR"
    echo "Run setup-server.sh first"
    exit 1
fi

cd $APP_DIR

# ============================================================================
# 1. Pull latest changes
# ============================================================================
echo ""
echo "[1/6] Pulling latest changes from Git..."

# Stash any local changes
git stash

# Pull latest
git pull origin main || git pull origin master

# Apply stashed changes (if any)
git stash pop || true

# ============================================================================
# 2. Check environment variables
# ============================================================================
echo ""
echo "[2/6] Checking environment variables..."

if [ ! -f "$APP_DIR/.env" ]; then
    echo "Error: .env file not found!"
    echo "Copy .env.example to .env and configure it:"
    echo "  cp .env.example .env"
    echo "  nano .env"
    exit 1
fi

# Check for placeholder values
if grep -q "your-.*-key-here" .env; then
    echo "Warning: .env contains placeholder values"
    echo "Make sure to replace them with actual API keys"
    read -p "Continue anyway? (y/N): " confirm
    if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
        exit 1
    fi
fi

# ============================================================================
# 3. Backup current data
# ============================================================================
echo ""
echo "[3/6] Creating backup of current data..."

BACKUP_DIR="/opt/backups/embedding-insights"
mkdir -p $BACKUP_DIR

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/workspace_pre_deploy_$TIMESTAMP.tar.gz"

if [ -d "$APP_DIR/workspace" ]; then
    tar -czf $BACKUP_FILE workspace/
    echo "Backup created: $BACKUP_FILE"

    # Keep only last 10 backups
    ls -t $BACKUP_DIR/workspace_pre_deploy_*.tar.gz | tail -n +11 | xargs -r rm
else
    echo "No workspace directory found, skipping backup"
fi

# ============================================================================
# 4. Build Docker images
# ============================================================================
echo ""
echo "[4/6] Building Docker images..."

# Build with no cache if requested
read -p "Build with no cache? (y/N): " no_cache

if [ "$no_cache" = "y" ] || [ "$no_cache" = "Y" ]; then
    docker compose build --no-cache
else
    docker compose build
fi

# ============================================================================
# 5. Deploy with zero downtime
# ============================================================================
echo ""
echo "[5/6] Deploying application..."

# Stop old containers
echo "Stopping old containers..."
docker compose down

# Start new containers
echo "Starting new containers..."
docker compose up -d --remove-orphans

# Wait for services to start
echo "Waiting for services to start..."
sleep 15

# ============================================================================
# 6. Health checks
# ============================================================================
echo ""
echo "[6/6] Running health checks..."

# Check if containers are running
echo "Checking container status..."
docker compose ps

# Try to curl health endpoints
echo ""
echo "Checking application health..."

# Function to check health
check_health() {
    local service=$1
    local port=$2
    local url="http://localhost:$port/_stcore/health"

    if curl -sf $url > /dev/null; then
        echo "✓ $service is healthy"
        return 0
    else
        echo "✗ $service is NOT responding"
        return 1
    fi
}

# Check each service
HEALTH_OK=true

check_health "GSC Insights" 8501 || HEALTH_OK=false
check_health "Content Analyzer" 8502 || HEALTH_OK=false
check_health "Linking Optimizer" 8503 || HEALTH_OK=false
check_health "Nginx" 80 || HEALTH_OK=false

# ============================================================================
# Results
# ============================================================================
echo ""
echo "============================================================================"

if [ "$HEALTH_OK" = true ]; then
    echo "Deployment successful!"
    echo "============================================================================"
    echo ""
    echo "Applications are running at:"
    echo "  - GSC Insights: http://localhost:8501"
    echo "  - Content Analyzer: http://localhost:8502"
    echo "  - Linking Optimizer: http://localhost:8503"
    echo ""
    echo "Check logs with: docker compose logs -f"
else
    echo "Deployment completed with warnings"
    echo "============================================================================"
    echo ""
    echo "Some services are not responding properly."
    echo "Check logs with:"
    echo "  docker compose logs -f"
    echo ""
    echo "To rollback, restore from backup:"
    echo "  cd $APP_DIR"
    echo "  tar -xzf $BACKUP_FILE"
    echo "  docker compose up -d"
fi

echo ""
echo "============================================================================"
