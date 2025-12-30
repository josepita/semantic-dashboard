#!/bin/bash
# ============================================================================
# Backup Script for Embedding Insights Suite
# ============================================================================
# Creates a backup of all workspace data and configuration
#
# Usage:
#   chmod +x scripts/backup.sh
#   ./scripts/backup.sh [backup_name]
# ============================================================================

set -e

echo "============================================================================"
echo "Embedding Insights Suite - Backup"
echo "============================================================================"

# ============================================================================
# Configuration
# ============================================================================
APP_DIR="/opt/embedding-insights-suite"
BACKUP_DIR="/opt/backups/embedding-insights"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Custom backup name or timestamp
if [ ! -z "$1" ]; then
    BACKUP_NAME="$1_$TIMESTAMP"
else
    BACKUP_NAME="full_backup_$TIMESTAMP"
fi

BACKUP_FILE="$BACKUP_DIR/${BACKUP_NAME}.tar.gz"

# Create backup directory
mkdir -p $BACKUP_DIR

# Change to app directory
if [ ! -d "$APP_DIR" ]; then
    echo "Error: Application directory not found: $APP_DIR"
    exit 1
fi

cd $APP_DIR

# ============================================================================
# 1. Backup workspace data
# ============================================================================
echo ""
echo "[1/4] Backing up workspace data..."

if [ -d "workspace" ]; then
    WORKSPACE_SIZE=$(du -sh workspace | cut -f1)
    echo "Workspace size: $WORKSPACE_SIZE"

    # Create temporary directory for backup
    TEMP_BACKUP_DIR="/tmp/embedding-insights-backup-$TIMESTAMP"
    mkdir -p $TEMP_BACKUP_DIR

    # Copy workspace
    echo "Copying workspace..."
    cp -r workspace $TEMP_BACKUP_DIR/

    # Count projects
    PROJECT_COUNT=$(find workspace/projects -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
    echo "Projects: $PROJECT_COUNT"
else
    echo "No workspace directory found"
    exit 1
fi

# ============================================================================
# 2. Backup configuration files
# ============================================================================
echo ""
echo "[2/4] Backing up configuration..."

# Copy .env (without sensitive data in logs)
if [ -f ".env" ]; then
    cp .env $TEMP_BACKUP_DIR/.env.backup
    echo "Environment variables backed up"
fi

# Copy docker-compose.yml
if [ -f "docker-compose.yml" ]; then
    cp docker-compose.yml $TEMP_BACKUP_DIR/
    echo "Docker compose configuration backed up"
fi

# Copy nginx configs
if [ -d "nginx" ]; then
    cp -r nginx $TEMP_BACKUP_DIR/
    echo "Nginx configuration backed up"
fi

# Create backup metadata
cat > $TEMP_BACKUP_DIR/backup_metadata.txt << EOF
Backup Created: $(date)
Hostname: $(hostname)
Backup Name: $BACKUP_NAME
Projects Count: $PROJECT_COUNT
Workspace Size: $WORKSPACE_SIZE
Git Commit: $(git rev-parse HEAD 2>/dev/null || echo "N/A")
Git Branch: $(git branch --show-current 2>/dev/null || echo "N/A")
Docker Images:
$(docker images | grep embedding-insights || echo "N/A")
EOF

echo "Metadata created"

# ============================================================================
# 3. Create compressed archive
# ============================================================================
echo ""
echo "[3/4] Creating compressed archive..."

cd /tmp
tar -czf $BACKUP_FILE embedding-insights-backup-$TIMESTAMP/

# Get backup size
BACKUP_SIZE=$(du -sh $BACKUP_FILE | cut -f1)
echo "Backup size: $BACKUP_SIZE"

# Cleanup temp directory
rm -rf $TEMP_BACKUP_DIR

# ============================================================================
# 4. Cleanup old backups
# ============================================================================
echo ""
echo "[4/4] Cleaning up old backups..."

# Keep only last 30 backups
BACKUP_COUNT=$(ls -1 $BACKUP_DIR/*.tar.gz 2>/dev/null | wc -l)
echo "Total backups: $BACKUP_COUNT"

if [ $BACKUP_COUNT -gt 30 ]; then
    echo "Removing old backups (keeping last 30)..."
    ls -t $BACKUP_DIR/*.tar.gz | tail -n +31 | xargs -r rm
    echo "Cleanup complete"
fi

# ============================================================================
# Results
# ============================================================================
echo ""
echo "============================================================================"
echo "Backup completed successfully!"
echo "============================================================================"
echo ""
echo "Backup file: $BACKUP_FILE"
echo "Backup size: $BACKUP_SIZE"
echo "Projects backed up: $PROJECT_COUNT"
echo ""
echo "To restore this backup:"
echo "  cd $APP_DIR"
echo "  tar -xzf $BACKUP_FILE"
echo "  cp embedding-insights-backup-$TIMESTAMP/workspace . -r"
echo "  cp embedding-insights-backup-$TIMESTAMP/.env.backup .env"
echo "  docker compose up -d"
echo ""
echo "============================================================================"

# Optional: Upload to remote storage
read -p "Upload to remote storage? (y/N): " upload

if [ "$upload" = "y" ] || [ "$upload" = "Y" ]; then
    echo ""
    echo "Remote upload not configured yet."
    echo "Configure rsync, S3, or other backup solution in this script."
    # Example for rsync:
    # rsync -avz $BACKUP_FILE user@backup-server:/backups/
    # Example for S3:
    # aws s3 cp $BACKUP_FILE s3://your-bucket/backups/
fi

echo ""
echo "Backup complete: $BACKUP_FILE"
