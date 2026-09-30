#!/usr/bin/env bash
# One-time setup: Python venv + backend deps, frontend deps, and backend/.env.
set -euo pipefail
cd "$(dirname "$0")"

say() { printf "\n\033[1;34m▸ %s\033[0m\n" "$*"; }

command -v python3 >/dev/null || { echo "python3 is required (3.10+)"; exit 1; }
command -v npm >/dev/null || { echo "Node.js 18+ is required (https://nodejs.org)"; exit 1; }

if [ ! -d venv ]; then
  say "Creating Python virtual environment (venv/)"
  python3 -m venv venv
fi
# shellcheck disable=SC1091
source venv/bin/activate

say "Installing backend dependencies"
pip install --upgrade pip >/dev/null
pip install -r backend/requirements.txt

say "Installing frontend dependencies"
(cd frontend && npm install)

if [ ! -f backend/.env ]; then
  cp backend/.env.example backend/.env
  say "Created backend/.env — open it and add at least one API key (GEMINI_API_KEY or GROQ_API_KEY)"
else
  say "backend/.env already exists — left unchanged"
fi

say "Running backend tests (offline)"
(cd backend && python -m pytest -q tests) || echo "Tests failed — see output above"

say "Done. Start everything with:  ./start.sh"
