#!/bin/bash
# ============================================================================
# SSL Certificate Setup with Let's Encrypt (Certbot)
# ============================================================================
# This script configures SSL certificates for all three apps
#
# Prerequisites:
# - DNS records must point to your server IP
# - Nginx must be running
# - Ports 80 and 443 must be open
#
# Usage:
#   chmod +x scripts/ssl-setup.sh
#   sudo ./scripts/ssl-setup.sh
# ============================================================================

set -e

echo "============================================================================"
echo "SSL Certificate Setup"
echo "============================================================================"

# ============================================================================
# Configuration
# ============================================================================
echo ""
echo "Enter your domain names (press Enter after each, empty line to finish):"
echo ""

# Read domains
DOMAINS=()

read -p "GSC Insights domain (e.g., gsc.yourdomain.com): " DOMAIN_GSC
if [ ! -z "$DOMAIN_GSC" ]; then
    DOMAINS+=("$DOMAIN_GSC")
fi

read -p "Content Analyzer domain (e.g., content.yourdomain.com): " DOMAIN_CONTENT
if [ ! -z "$DOMAIN_CONTENT" ]; then
    DOMAINS+=("$DOMAIN_CONTENT")
fi

read -p "Linking Optimizer domain (e.g., linking.yourdomain.com): " DOMAIN_LINKING
if [ ! -z "$DOMAIN_LINKING" ]; then
    DOMAINS+=("$DOMAIN_LINKING")
fi

# Contact email for Let's Encrypt
read -p "Email for SSL certificate notifications: " CERT_EMAIL

if [ ${#DOMAINS[@]} -eq 0 ]; then
    echo "No domains provided. Exiting."
    exit 1
fi

if [ -z "$CERT_EMAIL" ]; then
    echo "No email provided. Exiting."
    exit 1
fi

# ============================================================================
# Stop containers temporarily
# ============================================================================
echo ""
echo "Stopping containers..."
cd /opt/embedding-insights-suite
docker compose down

# ============================================================================
# Obtain SSL certificates
# ============================================================================
echo ""
echo "Obtaining SSL certificates for:"
for domain in "${DOMAINS[@]}"; do
    echo "  - $domain"
done
echo ""

# Create SSL directory
mkdir -p /opt/embedding-insights-suite/nginx/ssl

# Build certbot command
CERTBOT_DOMAINS=""
for domain in "${DOMAINS[@]}"; do
    CERTBOT_DOMAINS="$CERTBOT_DOMAINS -d $domain"
done

# Run certbot
certbot certonly \
    --standalone \
    --non-interactive \
    --agree-tos \
    --email "$CERT_EMAIL" \
    $CERTBOT_DOMAINS \
    --preferred-challenges http

# ============================================================================
# Link certificates to Nginx directory
# ============================================================================
echo ""
echo "Linking certificates to Nginx directory..."

for domain in "${DOMAINS[@]}"; do
    # Create symbolic links
    ln -sf /etc/letsencrypt/live/$domain/fullchain.pem /opt/embedding-insights-suite/nginx/ssl/$domain.crt
    ln -sf /etc/letsencrypt/live/$domain/privkey.pem /opt/embedding-insights-suite/nginx/ssl/$domain.key
done

# ============================================================================
# Update Nginx configuration for HTTPS
# ============================================================================
echo ""
echo "Updating Nginx configuration for HTTPS..."

NGINX_CONF="/opt/embedding-insights-suite/nginx/conf.d/apps.conf"

if [ -f "$NGINX_CONF" ]; then
    # Backup original
    cp $NGINX_CONF ${NGINX_CONF}.backup

    # Create HTTPS configuration
    cat > $NGINX_CONF << 'EOF'
# Embedding Insights Suite - Nginx Configuration with SSL
# ========================================================

upstream gsc-insights {
    server gsc-insights:8501;
}

upstream content-analyzer {
    server content-analyzer:8502;
}

upstream linking-optimizer {
    server linking-optimizer:8503;
}

# ============================================================================
# GSC Insights - HTTP to HTTPS redirect
# ============================================================================
server {
    listen 80;
    server_name DOMAIN_GSC;
    return 301 https://$server_name$request_uri;
}

# GSC Insights - HTTPS
server {
    listen 443 ssl http2;
    server_name DOMAIN_GSC;

    ssl_certificate /etc/nginx/ssl/DOMAIN_GSC.crt;
    ssl_certificate_key /etc/nginx/ssl/DOMAIN_GSC.key;

    # SSL configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;

    # HSTS
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

    location / {
        proxy_pass http://gsc-insights;
        proxy_http_version 1.1;

        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        proxy_read_timeout 86400;
        proxy_connect_timeout 86400;
        proxy_send_timeout 86400;
        proxy_buffering off;
    }

    location /_stcore/stream {
        proxy_pass http://gsc-insights/_stcore/stream;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 86400;
    }
}

# ============================================================================
# Content Analyzer - HTTP to HTTPS redirect
# ============================================================================
server {
    listen 80;
    server_name DOMAIN_CONTENT;
    return 301 https://$server_name$request_uri;
}

# Content Analyzer - HTTPS
server {
    listen 443 ssl http2;
    server_name DOMAIN_CONTENT;

    ssl_certificate /etc/nginx/ssl/DOMAIN_CONTENT.crt;
    ssl_certificate_key /etc/nginx/ssl/DOMAIN_CONTENT.key;

    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

    location / {
        proxy_pass http://content-analyzer;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 86400;
        proxy_connect_timeout 86400;
        proxy_send_timeout 86400;
        proxy_buffering off;
    }

    location /_stcore/stream {
        proxy_pass http://content-analyzer/_stcore/stream;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 86400;
    }
}

# ============================================================================
# Linking Optimizer - HTTP to HTTPS redirect
# ============================================================================
server {
    listen 80;
    server_name DOMAIN_LINKING;
    return 301 https://$server_name$request_uri;
}

# Linking Optimizer - HTTPS
server {
    listen 443 ssl http2;
    server_name DOMAIN_LINKING;

    ssl_certificate /etc/nginx/ssl/DOMAIN_LINKING.crt;
    ssl_certificate_key /etc/nginx/ssl/DOMAIN_LINKING.key;

    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

    location / {
        proxy_pass http://linking-optimizer;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 86400;
        proxy_connect_timeout 86400;
        proxy_send_timeout 86400;
        proxy_buffering off;
    }

    location /_stcore/stream {
        proxy_pass http://linking-optimizer/_stcore/stream;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 86400;
    }
}

# ============================================================================
# Default Server (Redirect to GSC Insights)
# ============================================================================
server {
    listen 80 default_server;
    listen 443 ssl http2 default_server;
    server_name _;

    ssl_certificate /etc/nginx/ssl/DOMAIN_GSC.crt;
    ssl_certificate_key /etc/nginx/ssl/DOMAIN_GSC.key;

    return 301 https://DOMAIN_GSC;
}
EOF

    # Replace placeholders
    if [ ! -z "$DOMAIN_GSC" ]; then
        sed -i "s/DOMAIN_GSC/$DOMAIN_GSC/g" $NGINX_CONF
    fi

    if [ ! -z "$DOMAIN_CONTENT" ]; then
        sed -i "s/DOMAIN_CONTENT/$DOMAIN_CONTENT/g" $NGINX_CONF
    fi

    if [ ! -z "$DOMAIN_LINKING" ]; then
        sed -i "s/DOMAIN_LINKING/$DOMAIN_LINKING/g" $NGINX_CONF
    fi

    echo "Nginx configuration updated with HTTPS"
fi

# ============================================================================
# Set up automatic renewal
# ============================================================================
echo ""
echo "Setting up automatic certificate renewal..."

# Create renewal hook
cat > /etc/letsencrypt/renewal-hooks/deploy/restart-docker.sh << 'HOOK'
#!/bin/bash
# Restart Docker containers after certificate renewal
cd /opt/embedding-insights-suite
docker compose restart nginx
HOOK

chmod +x /etc/letsencrypt/renewal-hooks/deploy/restart-docker.sh

# Test renewal
certbot renew --dry-run

echo "Automatic renewal configured"

# ============================================================================
# Start containers
# ============================================================================
echo ""
echo "Starting containers with HTTPS..."
cd /opt/embedding-insights-suite
docker compose up -d

# Wait for services
sleep 10

# ============================================================================
# Verify SSL
# ============================================================================
echo ""
echo "Verifying SSL certificates..."

for domain in "${DOMAINS[@]}"; do
    echo "Checking $domain..."
    curl -sI https://$domain | head -n 1 || echo "  Warning: Could not reach $domain"
done

# ============================================================================
# Complete
# ============================================================================
echo ""
echo "============================================================================"
echo "SSL setup complete!"
echo "============================================================================"
echo ""
echo "Your applications are now available at:"
echo ""
for domain in "${DOMAINS[@]}"; do
    echo "  https://$domain"
done
echo ""
echo "Certificates will auto-renew every 60 days."
echo "Test renewal with: certbot renew --dry-run"
echo ""
echo "============================================================================"
