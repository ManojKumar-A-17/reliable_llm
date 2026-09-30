#!/usr/bin/env bash
# Start only the backend (see ../start.sh to start everything).
cd "$(dirname "$0")"
source ../venv/bin/activate
exec uvicorn api.server:app --reload --reload-dir rllm --reload-dir api --port 8000
