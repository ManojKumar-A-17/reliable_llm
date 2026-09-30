#!/usr/bin/env bash
# Optional: move v1 files that v2 no longer uses into _legacy/ (nothing is deleted).
# Review _legacy/ afterwards and delete it yourself when you're happy.
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p _legacy
move() { if [ -e "$1" ]; then mkdir -p "_legacy/$(dirname "$1")"; mv "$1" "_legacy/$1" && echo "moved $1"; fi; }

move backend/src
move backend/configs
move backend/experiments
move backend/main.py
move backend/scripts/build_index.py
move backend/data/index/knowledge.faiss
move backend/data/index/knowledge.pkl
move frontend/src/components
move frontend/src/index.css
move frontend/src/App.module.css
move frontend/rllm-frontend
move tests/test_agents.py
move data/knowledge_base.txt
echo "Done. v1 files are in _legacy/."
