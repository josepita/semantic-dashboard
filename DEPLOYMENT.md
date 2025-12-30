# Deployment Guide - Embedding Insights Suite

Complete guide for deploying the Embedding Insights Suite to a production server.

## Table of Contents

- [Overview](#overview)
- [Server Requirements](#server-requirements)
- [Deployment Options](#deployment-options)
- [Quick Start](#quick-start)
- [Detailed Setup](#detailed-setup)
- [CI/CD Setup](#cicd-setup)
- [Monitoring & Maintenance](#monitoring--maintenance)
- [Troubleshooting](#troubleshooting)

---

## Overview

The Embedding Insights Suite consists of 3 Streamlit applications:

1. **GSC Insights** (Port 8501) - SEO position analysis and reporting
2. **Content Analyzer** (Port 8502) - Content optimization and semantic analysis
3. **Linking Optimizer** (Port 8503) - Internal linking recommendations

All apps share a common `/workspace` directory for persistent data storage.

### Architecture

```
Internet
    ↓
Nginx (Port 80/443) - Reverse Proxy + SSL
    ↓
┌─────────────┬──────────────────┬──────────────────┐
│             │                  │                  │
GSC Insights  Content Analyzer  Linking Optimizer
  (8501)          (8502)             (8503)
    │               │                  │
    └───────────────┴──────────────────┘
                    ↓
            Shared /workspace
            (DuckDB + Files)
```

---

## Server Requirements

### Minimum Specifications

- **OS**: Ubuntu 22.04 LTS or Debian 11+ (recommended)
- **RAM**: 4-8 GB (NLP models require significant memory)
- **CPU**: 2+ cores
- **Storage**: 20 GB SSD (10 GB for Docker, 10 GB for data)
- **Network**: Public IP address with open ports 80, 443, 22

### Software Requirements

- Docker 24.0+
- Docker Compose 2.0+
- Git
- Nginx (for reverse proxy)
- Certbot (for SSL certificates)

### Recommended Providers

1. **VPS (Self-managed)**:
   - Hetzner Cloud (€4.5-9/month) - Excellent performance/price
   - DigitalOcean Droplets ($6-12/month)
   - Linode ($5-12/month)
   - Vultr ($6-12/month)

2. **PaaS (Managed)**:
   - Railway ($5-20/month) - Easy Docker deployment
   - Render ($7-25/month) - Zero config deployment
   - Fly.io ($5-15/month) - Global edge deployment

3. **Streamlit Cloud** (Free tier available):
   - Pros: Zero config, free tier
   - Cons: Limited resources, no persistent storage, single app

**Recommendation**: Hetzner Cloud CX21 (€4.5/month) or DigitalOcean Basic Droplet ($6/month) for full control and persistent storage.

---

## Deployment Options

### Option A: VPS Deployment (Recommended)

Full control, persistent storage, custom domains, SSL.

**Steps**: See [Detailed Setup](#detailed-setup) below.

### Option B: Railway Deployment

Quick deployment with managed infrastructure.

**Steps**:
1. Fork this repository
2. Create account at [Railway.app](https://railway.app)
3. Click "New Project" → "Deploy from GitHub repo"
4. Select your fork
5. Add environment variables from `.env.example`
6. Deploy!

### Option C: Render Deployment

Similar to Railway, with free tier.

**Steps**:
1. Create account at [Render.com](https://render.com)
2. Click "New" → "Web Service"
3. Connect GitHub repository
4. Select "Docker" as environment
5. Add environment variables
6. Deploy

### Option D: Streamlit Cloud

**Limitations**:
- Only 1 app per account (free tier)
- No persistent storage (data lost on restart)
- Limited resources

**Not recommended** for production multi-app deployment.

---

## Quick Start

For experienced DevOps engineers who want to deploy quickly:

```bash
# 1. Clone repository
git clone https://github.com/your-username/EmbeddingDashboard.git /opt/embedding-insights-suite
cd /opt/embedding-insights-suite

# 2. Run setup script
sudo bash scripts/setup-server.sh

# 3. Configure environment
cp .env.example .env
nano .env  # Add your API keys

# 4. Configure domains
nano nginx/conf.d/apps.conf  # Replace tudominio.com

# 5. Start application
docker compose up -d

# 6. Setup SSL (after DNS configured)
sudo bash scripts/ssl-setup.sh

# Done! Check status
docker compose ps
docker compose logs -f
```

---

## Detailed Setup

### Step 1: Server Preparation

#### 1.1. Access Your Server

```bash
ssh root@your-server-ip
```

Or with key-based authentication:

```bash
ssh -i ~/.ssh/your-key.pem root@your-server-ip
```

#### 1.2. Update System

```bash
apt update && apt upgrade -y
```

#### 1.3. Run Automated Setup Script

The setup script installs Docker, configures firewall, and prepares the application directory.

```bash
# Download repository
cd /opt
git clone https://github.com/your-username/EmbeddingDashboard.git embedding-insights-suite
cd embedding-insights-suite

# Make script executable
chmod +x scripts/setup-server.sh

# Run setup
sudo bash scripts/setup-server.sh
```

The script will:
- ✅ Install Docker and Docker Compose
- ✅ Configure UFW firewall (ports 22, 80, 443)
- ✅ Create application directory (`/opt/embedding-insights-suite`)
- ✅ Set up 4GB swap space
- ✅ Clone repository
- ✅ Create `.env` from template

**Manual alternative** (if you prefer step-by-step):

<details>
<summary>Click to expand manual installation steps</summary>

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo apt install docker-compose-plugin

# Install other dependencies
sudo apt install -y git vim htop ufw certbot python3-certbot-nginx

# Configure firewall
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable

# Create app directory
sudo mkdir -p /opt/embedding-insights-suite
sudo chmod -R 755 /opt/embedding-insights-suite
```

</details>

---

### Step 2: Configuration

#### 2.1. Environment Variables

Edit the `.env` file with your API keys:

```bash
cd /opt/embedding-insights-suite
nano .env
```

**Required variables**:

```bash
# API Keys
OPENAI_API_KEY=sk-proj-your-key
GEMINI_API_KEY=AIzaSy-your-key
SERPROBOT_API_KEY=your-key

# Domains
DOMAIN_GSC_INSIGHTS=gsc.yourdomain.com
DOMAIN_CONTENT_ANALYZER=content.yourdomain.com
DOMAIN_LINKING_OPTIMIZER=linking.yourdomain.com
```

See `.env.example` for all available options.

#### 2.2. Domain Configuration

Update Nginx configuration with your actual domains:

```bash
nano nginx/conf.d/apps.conf
```

Replace all instances of `tudominio.com` with your actual domain names:

```nginx
server_name gsc.yourdomain.com;  # Change this
server_name content.yourdomain.com;  # Change this
server_name linking.yourdomain.com;  # Change this
```

#### 2.3. DNS Configuration

Before proceeding, configure DNS records to point to your server:

```
A Record: gsc.yourdomain.com → your-server-ip
A Record: content.yourdomain.com → your-server-ip
A Record: linking.yourdomain.com → your-server-ip
```

Verify DNS propagation:

```bash
dig gsc.yourdomain.com
dig content.yourdomain.com
dig linking.yourdomain.com
```

---

### Step 3: Deployment

#### 3.1. Build and Start Containers

```bash
cd /opt/embedding-insights-suite

# Build Docker images
docker compose build

# Start all services
docker compose up -d

# Check status
docker compose ps
```

Expected output:

```
NAME                      STATUS    PORTS
gsc-insights              Up        0.0.0.0:8501->8501/tcp
content-analyzer          Up        0.0.0.0:8502->8502/tcp
linking-optimizer         Up        0.0.0.0:8503->8503/tcp
nginx                     Up        0.0.0.0:80->80/tcp, 0.0.0.0:443->443/tcp
```

#### 3.2. Verify Applications

Test that apps are running:

```bash
# Test locally
curl http://localhost:8501/_stcore/health
curl http://localhost:8502/_stcore/health
curl http://localhost:8503/_stcore/health

# Test through Nginx
curl http://your-server-ip
```

#### 3.3. View Logs

```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f gsc-insights
docker compose logs -f nginx
```

---

### Step 4: SSL Setup

#### 4.1. Automated SSL Setup

Run the SSL setup script (after DNS is configured):

```bash
sudo bash scripts/ssl-setup.sh
```

The script will:
- ✅ Obtain Let's Encrypt certificates for all 3 domains
- ✅ Configure Nginx for HTTPS
- ✅ Set up automatic renewal
- ✅ Restart containers

#### 4.2. Manual SSL Setup

<details>
<summary>Click for manual SSL configuration</summary>

```bash
# Stop containers temporarily
docker compose down

# Obtain certificates
sudo certbot certonly --standalone \
  -d gsc.yourdomain.com \
  -d content.yourdomain.com \
  -d linking.yourdomain.com \
  --agree-tos \
  --email your-email@example.com

# Link certificates
sudo ln -s /etc/letsencrypt/live/gsc.yourdomain.com/fullchain.pem \
  /opt/embedding-insights-suite/nginx/ssl/gsc.yourdomain.com.crt

sudo ln -s /etc/letsencrypt/live/gsc.yourdomain.com/privkey.pem \
  /opt/embedding-insights-suite/nginx/ssl/gsc.yourdomain.com.key

# Repeat for other domains...

# Update Nginx config (see scripts/ssl-setup.sh for HTTPS config)

# Restart containers
docker compose up -d
```

</details>

#### 4.3. Verify HTTPS

```bash
# Test SSL
curl -I https://gsc.yourdomain.com
curl -I https://content.yourdomain.com
curl -I https://linking.yourdomain.com

# Check SSL grade (optional)
# Visit: https://www.ssllabs.com/ssltest/
```

---

## CI/CD Setup

### GitHub Actions Configuration

#### 1. Create GitHub Secrets

Go to your repository → Settings → Secrets and variables → Actions → New repository secret

Add the following secrets:

```
SERVER_HOST = your-server-ip
SERVER_USER = root (or your SSH user)
SSH_PRIVATE_KEY = (paste your SSH private key)
SERVER_PORT = 22
DOMAIN_GSC_INSIGHTS = gsc.yourdomain.com
DOMAIN_CONTENT_ANALYZER = content.yourdomain.com
DOMAIN_LINKING_OPTIMIZER = linking.yourdomain.com
```

#### 2. Generate SSH Key (if needed)

```bash
# On your local machine
ssh-keygen -t ed25519 -C "github-actions"

# Copy public key to server
ssh-copy-id -i ~/.ssh/id_ed25519.pub root@your-server-ip

# Copy private key to GitHub Secrets
cat ~/.ssh/id_ed25519
```

#### 3. Enable GitHub Actions

The workflow file is already created at `.github/workflows/deploy.yml`.

It will automatically:
- ✅ Run tests on every push
- ✅ Build Docker images
- ✅ Push images to GitHub Container Registry
- ✅ Deploy to server via SSH
- ✅ Run health checks

#### 4. Trigger Deployment

```bash
# Commit and push changes
git add .
git commit -m "Deploy to production"
git push origin main

# GitHub Actions will automatically deploy
```

Watch the deployment progress:
- Go to your repository → Actions tab
- Click on the latest workflow run

---

## Monitoring & Maintenance

### Health Checks

```bash
# Check container status
docker compose ps

# Check resource usage
docker stats

# Check logs
docker compose logs -f

# Check disk usage
df -h
du -sh /opt/embedding-insights-suite/workspace
```

### Backups

#### Automated Backups

Backups are configured in GitHub Actions to run weekly.

#### Manual Backups

```bash
# Run backup script
sudo bash scripts/backup.sh

# Or create custom backup
cd /opt/embedding-insights-suite
tar -czf backup_$(date +%Y%m%d).tar.gz workspace/

# Copy to remote storage
rsync -avz backup_*.tar.gz user@backup-server:/backups/
```

#### Restore from Backup

```bash
# Stop containers
docker compose down

# Extract backup
cd /opt/embedding-insights-suite
tar -xzf /opt/backups/embedding-insights/workspace_20240101_120000.tar.gz

# Restart containers
docker compose up -d
```

### Updates

#### Update Application Code

```bash
cd /opt/embedding-insights-suite
git pull origin main
docker compose build
docker compose up -d
```

Or use the automated deployment script:

```bash
sudo bash scripts/deploy.sh
```

#### Update Docker Images

```bash
docker compose pull
docker compose up -d
```

#### Update System Packages

```bash
sudo apt update && sudo apt upgrade -y
sudo reboot
```

### SSL Certificate Renewal

Certificates auto-renew via certbot cron job.

To manually renew:

```bash
sudo certbot renew
docker compose restart nginx
```

Test renewal:

```bash
sudo certbot renew --dry-run
```

---

## Troubleshooting

### Container Won't Start

```bash
# Check logs
docker compose logs -f service-name

# Common issues:
# - Port already in use
# - Missing environment variables
# - Insufficient memory

# Check ports
sudo netstat -tulpn | grep 8501
sudo netstat -tulpn | grep 80

# Free up memory
docker system prune -a
```

### Out of Memory

```bash
# Check memory usage
free -h
docker stats

# Add swap space
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Nginx 502 Bad Gateway

```bash
# Check if backend containers are running
docker compose ps

# Restart containers
docker compose restart

# Check Nginx config
docker compose exec nginx nginx -t

# Check Nginx logs
docker compose logs nginx
```

### SSL Certificate Issues

```bash
# Check certificate expiration
sudo certbot certificates

# Force renewal
sudo certbot renew --force-renewal

# Check Nginx SSL config
docker compose exec nginx nginx -t
```

### Application Not Accessible

```bash
# Check firewall
sudo ufw status

# Check if ports are open
curl http://localhost:8501/_stcore/health
curl http://localhost:80

# Check DNS
dig gsc.yourdomain.com

# Test from outside
curl -I http://your-server-ip
```

### Workspace Data Lost

```bash
# Check if volume is mounted
docker compose ps -a
docker inspect container-name | grep -A 10 Mounts

# Restore from backup
cd /opt/embedding-insights-suite
tar -xzf /opt/backups/latest-backup.tar.gz

# Restart containers
docker compose restart
```

### Performance Issues

```bash
# Check resource usage
htop
docker stats

# Increase container resources (edit docker-compose.yml)
# Add under each service:
deploy:
  resources:
    limits:
      memory: 2G
      cpus: '1.0'

# Optimize database
# Connect to container
docker compose exec gsc-insights bash
python -c "import duckdb; conn = duckdb.connect('/app/workspace/projects/*/data.duckdb'); conn.execute('VACUUM'); conn.execute('ANALYZE');"
```

---

## Advanced Configuration

### Custom Domains

To use different domain patterns (e.g., subpath instead of subdomain):

Edit `nginx/conf.d/apps.conf`:

```nginx
# Instead of subdomain
server_name yourdomain.com;

location /gsc {
    proxy_pass http://gsc-insights;
    # ... rest of config
}

location /content {
    proxy_pass http://content-analyzer;
    # ... rest of config
}

location /linking {
    proxy_pass http://linking-optimizer;
    # ... rest of config
}
```

### Resource Limits

Edit `docker-compose.yml`:

```yaml
services:
  gsc-insights:
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
        reservations:
          memory: 1G
          cpus: '0.5'
```

### Custom Nginx Configuration

Edit `nginx/nginx.conf` for global settings or `nginx/conf.d/apps.conf` for app-specific routing.

Common customizations:
- Rate limiting
- IP whitelisting
- Custom headers
- Caching

Example rate limiting:

```nginx
http {
    limit_req_zone $binary_remote_addr zone=one:10m rate=10r/s;

    server {
        location / {
            limit_req zone=one burst=20;
            # ... rest of config
        }
    }
}
```

---

## Production Checklist

Before going live, verify:

- [ ] Environment variables configured (`.env`)
- [ ] Domains configured in Nginx
- [ ] DNS records pointing to server
- [ ] SSL certificates obtained and configured
- [ ] Firewall rules configured (UFW)
- [ ] Backup system configured
- [ ] CI/CD pipeline tested
- [ ] Health checks passing
- [ ] Resource monitoring set up (optional: Prometheus, Grafana)
- [ ] Error tracking configured (optional: Sentry)
- [ ] Log aggregation configured (optional: ELK, Loki)
- [ ] Alerts configured (optional: Uptime monitoring)

---

## Support & Resources

- **Documentation**: See [README.md](README.md) for application features
- **ROADMAP**: See [ROADMAP.md](ROADMAP.md) for planned features
- **Usage Guide**: See [shared/USAGE.md](shared/USAGE.md) for API documentation
- **Issues**: Report bugs at your GitHub repository issues page

---

## License

See [LICENSE](LICENSE) file for details.

---

**Last Updated**: 2024
**Version**: 1.0.0
