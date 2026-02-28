#!/bin/bash

# Render assigns a port via $PORT env var. Default to 8000 locally.
export PORT=${PORT:-8000}

# The bot talks to the backend locally inside the same container
export BACKEND_URL="http://localhost:${PORT}"

echo "🚀 Starting FastAPI Backend on port $PORT..."
python3 -m uvicorn app.main:app --host 0.0.0.0 --port "$PORT" &

# Poll the health endpoint until the backend is up and chunks are loaded
echo "⏳ Waiting for backend to finish loading PDFs..."
for i in $(seq 1 60); do
    sleep 3
    CHUNKS=$(curl -s "http://localhost:${PORT}/health" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('chunks_loaded', 0))" 2>/dev/null)
    if [ "$CHUNKS" -gt "0" ] 2>/dev/null; then
        echo "✅ Backend ready! $CHUNKS chunks loaded."
        break
    fi
    echo "... still loading ($i/60)..."
done

echo "🤖 Starting Telegram Bot..."
python3 -m app.telegram_bot
