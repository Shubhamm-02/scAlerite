#!/bin/bash

# For Render Background Worker: no port binding needed.
# We run the FastAPI backend internally and the Telegram bot polls Telegram.

echo "🚀 Starting FastAPI Backend..."
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 &

# Wait until the backend confirms PDFs are loaded
echo "⏳ Waiting for backend to load PDFs..."
for i in $(seq 1 60); do
    sleep 3
    CHUNKS=$(curl -s "http://localhost:8000/health" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('chunks_loaded', 0))" 2>/dev/null)
    if [ "$CHUNKS" -gt "0" ] 2>/dev/null; then
        echo "✅ Backend ready! $CHUNKS chunks loaded."
        break
    fi
    echo "... still loading ($i/60)..."
done

export BACKEND_URL="http://localhost:8000"

echo "🤖 Starting Telegram Bot..."
python3 -m app.telegram_bot
