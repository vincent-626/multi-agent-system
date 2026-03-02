#!/bin/bash
# Startup script for cloud deployment (Linux + NVIDIA GPU, no Docker).
# Runs Ollama, Qdrant, and the app directly as processes.
# Run once after the instance boots:  bash startup.sh

set -euo pipefail

# ── Config — edit before running ──────────────────────────────────────────────
REPO_URL="https://github.com/your-username/multi-agent-system.git"
APP_DIR="/workspace/multi-agent-system"
API_KEY=""          # optional: set a key to protect the /query endpoint

# ── System dependencies ───────────────────────────────────────────────────────
echo "==> Installing system dependencies..."
apt-get update -q
apt-get install -y -q git curl

# ── Ollama ────────────────────────────────────────────────────────────────────
if ! command -v ollama &>/dev/null; then
    echo "==> Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
fi

echo "==> Starting Ollama..."
nohup ollama serve > /var/log/ollama.log 2>&1 &
sleep 5

echo "==> Pulling models (this may take a while)..."
ollama pull qwen3
ollama pull qwen3:1.7b
ollama pull nomic-embed-text

# ── Qdrant ────────────────────────────────────────────────────────────────────
if [ ! -f /usr/local/bin/qdrant ]; then
    echo "==> Installing Qdrant..."
    curl -fsSL https://github.com/qdrant/qdrant/releases/latest/download/qdrant-x86_64-unknown-linux-musl.tar.gz \
        | tar -xz -C /usr/local/bin
    chmod +x /usr/local/bin/qdrant
fi

mkdir -p /var/lib/qdrant
echo "==> Starting Qdrant..."
nohup env QDRANT__SERVICE__HOST=127.0.0.1 QDRANT__STORAGE__STORAGE_PATH=/var/lib/qdrant qdrant > /var/log/qdrant.log 2>&1 &
sleep 2

# ── Clone repo ────────────────────────────────────────────────────────────────
if [ ! -d "$APP_DIR" ]; then
    echo "==> Cloning repository..."
    git clone "$REPO_URL" "$APP_DIR"
else
    echo "==> Repository already exists, pulling latest..."
    git -C "$APP_DIR" pull
fi

cd "$APP_DIR"

# ── uv ────────────────────────────────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
    echo "==> Installing uv..."
    curl -fsSL https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# ── Environment ───────────────────────────────────────────────────────────────
if [ ! -f .env ]; then
    echo "==> Creating .env..."
    cat > .env <<EOF
API_KEY=${API_KEY}
EOF
fi

# ── Install Python dependencies ───────────────────────────────────────────────
echo "==> Installing Python dependencies..."
uv sync --frozen

# ── Launch app ────────────────────────────────────────────────────────────────
echo "==> Starting app..."
nohup env OLLAMA_BASE_URL=http://localhost:11434 QDRANT_URL=http://localhost:6333 API_KEY="${API_KEY}" \
    uv run uvicorn src.server:app --host 0.0.0.0 --port 8000 \
    > /var/log/app.log 2>&1 &
sleep 3

# ── Cloudflare Tunnel ─────────────────────────────────────────────────────────
if [ ! -f /usr/local/bin/cloudflared ]; then
    echo "==> Installing cloudflared..."
    curl -fsSL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
        -o /usr/local/bin/cloudflared
    chmod +x /usr/local/bin/cloudflared
fi

echo "==> Starting Cloudflare Tunnel..."
nohup cloudflared tunnel --no-autoupdate --url http://localhost:8000 > /var/log/cloudflared.log 2>&1 &

echo "==> Waiting for tunnel URL..."
for i in $(seq 1 20); do
    URL=$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' /var/log/cloudflared.log 2>/dev/null | head -1)
    if [ -n "$URL" ]; then
        break
    fi
    sleep 2
done

echo ""
echo "==> Done."
if [ -n "${URL:-}" ]; then
    echo "    URL  : $URL"
else
    echo "    URL  : (not yet available — check: grep trycloudflare /var/log/cloudflared.log)"
fi
echo "    Logs:"
echo "      ollama      : tail -f /var/log/ollama.log"
echo "      qdrant      : tail -f /var/log/qdrant.log"
echo "      app         : tail -f /var/log/app.log"
echo "      cloudflared : tail -f /var/log/cloudflared.log"
