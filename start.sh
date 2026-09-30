#!/usr/bin/env bash
# Start backend (port 8000) and frontend (port 3000) together. Ctrl+C stops both.
set -euo pipefail
cd "$(dirname "$0")"

[ -d venv ] || { echo "Run ./setup.sh first."; exit 1; }
[ -f backend/.env ] || { echo "backend/.env is missing — run ./setup.sh or copy backend/.env.example"; exit 1; }
# shellcheck disable=SC1091
source venv/bin/activate

cleanup() { trap - INT TERM EXIT; kill 0 2>/dev/null || true; }
trap cleanup INT TERM EXIT

(cd backend && uvicorn api.server:app --reload --reload-dir rllm --reload-dir api --port 8000) &
(cd frontend && npm run dev -- --port 3000) &

sleep 2
printf "\n  Reliable LLM is starting…\n  → App:      http://localhost:3000\n  → API docs: http://localhost:8000/docs\n\n  Press Ctrl+C to stop.\n\n"
wait
