#!/bin/bash
# Startup script for cloud deployment (Linux + NVIDIA GPU).
# Run once after the instance boots:  bash startup.sh

set -euo pipefail

# ── Config — edit before running ──────────────────────────────────────────────
REPO_URL="https://github.com/your-username/multi-agent-system.git"
APP_DIR="/workspace/multi-agent-system"
API_KEY=""          # optional: set a key to protect the /query endpoint

# ── Install system dependencies ───────────────────────────────────────────────
echo "==> Installing dependencies..."
apt-get update -q
apt-get install -y -q git curl

# Docker Compose v2 plugin
if ! docker compose version &>/dev/null 2>&1; then
    echo "==> Installing Docker Compose plugin..."
    mkdir -p /usr/local/lib/docker/cli-plugins
    curl -fsSL "https://github.com/docker/compose/releases/latest/download/docker-compose-linux-x86_64" \
        -o /usr/local/lib/docker/cli-plugins/docker-compose
    chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
fi

# ── Clone repo ────────────────────────────────────────────────────────────────
if [ ! -d "$APP_DIR" ]; then
    echo "==> Cloning repository..."
    git clone "$REPO_URL" "$APP_DIR"
else
    echo "==> Repository already exists, pulling latest..."
    git -C "$APP_DIR" pull
fi

cd "$APP_DIR"

# ── Environment ───────────────────────────────────────────────────────────────
if [ ! -f .env ]; then
    echo "==> Creating .env..."
    cat > .env <<EOF
API_KEY=${API_KEY}
EOF
fi

# ── Launch ────────────────────────────────────────────────────────────────────
echo "==> Starting services..."
docker compose -f docker-compose.cloud.yml up --build -d

echo ""
echo "==> Done. Services starting up — model pull may take a few minutes."
echo "    App will be available at http://$(hostname -I | awk '{print $1}'):8000"
echo "    Follow logs with: docker compose -f docker-compose.cloud.yml logs -f"
