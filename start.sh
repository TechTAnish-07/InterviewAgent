#!/bin/bash
set -e

echo "Starting LiveKit Agent Worker in background..."
python agent.py start &

echo "Starting FastAPI Web Server..."
exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
