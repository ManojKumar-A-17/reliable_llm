"""
backend/api/server.py
---------------------
ASGI entry point. From the backend/ directory:

    uvicorn api.server:app --reload --port 8000

All logic lives in the ``rllm`` package; this file only builds the app.
"""

import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from rllm.api import create_app  # noqa: E402

app = create_app()
