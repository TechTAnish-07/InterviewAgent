#!/bin/bash
set -e

echo "Starting LiveKit Agent Worker in background..."
python agent.py start &

echo "Starting FastAPI Web Server on port ${PORT:-10000}..."
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-10000}"
