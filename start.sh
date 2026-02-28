#!/bin/bash

# Render assigns a dynamic port (usually 10000), default to 8000 if running locally
export PORT=${PORT:-8000}

echo "🚀 Starting FastAPI Backend on port $PORT..."
python3 -m uvicorn app.main:app --host 0.0.0.0 --port $PORT &

# Wait a few seconds to let FastAPI fully load the FAISS vectors before the bot tries to connect
echo "⏳ Waiting for backend to load PDFs..."
sleep 5

# Tell the bot to connect to the local FastAPI port
export BACKEND_URL="http://localhost:$PORT"

echo "🤖 Starting Telegram Bot..."
python3 -m app.telegram_bot
