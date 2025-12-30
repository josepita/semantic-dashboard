#!/bin/bash
# ============================================================================
# Embedding Insights Suite - Server Setup Script
# ============================================================================
# This script sets up a fresh Ubuntu/Debian server for deployment
# Tested on: Ubuntu 22.04 LTS, Debian 11+
#
# Usage:
#   chmod +x scripts/setup-server.sh
#   sudo ./scripts/setup-server.sh
# ============================================================================

set -e  # Exit on error

echo "============================================================================"
echo "Embedding Insights Suite - Server Setup"
echo "============================================================================"

# ============================================================================
# 1. Update system
# ============================================================================
echo ""
echo "[1/8] Updating system packages..."
apt-get update
apt-get upgrade -y

# ============================================================================
# 2. Install required packages
# ============================================================================
echo ""
echo "[2/8] Installing required packages..."
apt-get install -y \
    curl \
    git \
    vim \
    htop \
    ufw \
    fail2ban \
    certbot \
    python3-certbot-nginx \
    build-essential \
    apt-transport-https \
    ca-certificates \
    software-properties-common

# ============================================================================
# 3. Install Docker
# ============================================================================
echo ""
echo "[3/8] Installing Docker..."

# Add Docker's official GPG key
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# Set up Docker repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker Engine
apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Start Docker service
systemctl start docker
systemctl enable docker

# Add current user to docker group (if not root)
if [ "$USER" != "root" ]; then
    usermod -aG docker $USER
    echo "User $USER added to docker group. Please log out and back in for changes to take effect."
fi

# ============================================================================
# 4. Configure Firewall (UFW)
# ============================================================================
echo ""
echo "[4/8] Configuring firewall..."

# Allow SSH
ufw allow 22/tcp

# Allow HTTP and HTTPS
ufw allow 80/tcp
ufw allow 443/tcp

# Enable firewall
echo "y" | ufw enable

ufw status

# ============================================================================
# 5. Create application directory
# ============================================================================
echo ""
echo "[5/8] Creating application directory..."

APP_DIR="/opt/embedding-insights-suite"
mkdir -p $APP_DIR
mkdir -p /opt/backups/embedding-insights

# Set permissions
chmod -R 755 $APP_DIR
chmod -R 755 /opt/backups/embedding-insights

echo "Application directory: $APP_DIR"

# ============================================================================
# 6. Clone repository
# ============================================================================
echo ""
echo "[6/8] Cloning repository..."

read -p "Enter your GitHub repository URL (e.g., https://github.com/user/repo.git): " REPO_URL

if [ ! -z "$REPO_URL" ]; then
    cd /opt
    git clone $REPO_URL embedding-insights-suite
    cd embedding-insights-suite
else
    echo "No repository URL provided. You'll need to clone manually."
    echo "Run: cd /opt && git clone YOUR_REPO_URL embedding-insights-suite"
fi

# ============================================================================
# 7. Set up environment variables
# ============================================================================
echo ""
echo "[7/8] Setting up environment variables..."

if [ -f "$APP_DIR/.env.example" ]; then
    cp $APP_DIR/.env.example $APP_DIR/.env
    echo "Created .env file from .env.example"
    echo ""
    echo "IMPORTANT: Edit $APP_DIR/.env and add your API keys!"
    echo "Run: nano $APP_DIR/.env"
else
    echo "Warning: .env.example not found. You'll need to create .env manually."
fi

# ============================================================================
# 8. Configure swap (for servers with limited RAM)
# ============================================================================
echo ""
echo "[8/8] Configuring swap space..."

# Check if swap already exists
if [ $(swapon --show | wc -l) -eq 0 ]; then
    # Create 4GB swap file
    fallocate -l 4G /swapfile
    chmod 600 /swapfile
    mkswap /swapfile
    swapon /swapfile

    # Make swap permanent
    echo '/swapfile none swap sw 0 0' | tee -a /etc/fstab

    # Configure swappiness
    sysctl vm.swappiness=10
    echo 'vm.swappiness=10' | tee -a /etc/sysctl.conf

    echo "Swap configured: 4GB"
else
    echo "Swap already exists, skipping..."
fi

# ============================================================================
# Setup Complete
# ============================================================================
echo ""
echo "============================================================================"
echo "Server setup complete!"
echo "============================================================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Edit environment variables:"
echo "   nano $APP_DIR/.env"
echo ""
echo "2. Configure domain names in Nginx:"
echo "   nano $APP_DIR/nginx/conf.d/apps.conf"
echo "   - Replace 'tudominio.com' with your actual domains"
echo ""
echo "3. Set up SSL certificates (after DNS is configured):"
echo "   bash $APP_DIR/scripts/ssl-setup.sh"
echo ""
echo "4. Build and start the application:"
echo "   cd $APP_DIR"
echo "   docker compose up -d"
echo ""
echo "5. Check status:"
echo "   docker compose ps"
echo "   docker compose logs -f"
echo ""
echo "6. Set up GitHub Actions secrets for CI/CD:"
echo "   - SERVER_HOST: Your server IP"
echo "   - SERVER_USER: Your SSH user"
echo "   - SSH_PRIVATE_KEY: Your SSH private key"
echo "   - DOMAIN_*: Your domain names"
echo ""
echo "============================================================================"
