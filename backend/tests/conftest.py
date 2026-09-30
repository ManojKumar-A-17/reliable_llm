"""Shared fixtures: an offline fake LLM and a hashing embedder — no network, no ML downloads."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import List, Optional

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rllm.config import Settings  # noqa: E402
from rllm.knowledge import HashEmbedder  # noqa: E402
from rllm.llm.base import LLMError, LLMResponse  # noqa: E402
from rllm.llm.providers import Provider  # noqa: E402

DOC_MD = b"""# Solar System Facts

The Sun is a G-type main-sequence star. It contains about 99.86 percent of the mass of the Solar System.

## Mars

Mars is the fourth planet from the Sun. Mars has two small moons named Phobos and Deimos.
Olympus Mons on Mars is the tallest volcano known in the Solar System.

## Jupiter

Jupiter is the largest planet in the Solar System. Jupiter has a Great Red Spot, which is a giant storm.
"""


class FakeProvider(Provider):
    """Scripted provider. `fail` is a list of LLMErrors raised on successive calls first."""

    def __init__(self, name: str = "fake", fail: Optional[List[LLMError]] = None, rpm: int = 0,
                 answer: Optional[str] = None, verify_json: Optional[dict] = None):
        super().__init__({"strong": f"{name}-strong", "fast": f"{name}-fast"}, {"strong": rpm, "fast": rpm})
        self.name = name
        self.label = name.title()
        self.fail = list(fail or [])
        self.calls: List[str] = []
        self.answer = answer or "Mars has two moons, Phobos and Deimos [1]. Olympus Mons is the tallest volcano in the Solar System [1]."
        self.verify_json = verify_json

    async def complete(self, req, model, on_token=None):
        self.calls.append(req.purpose)
        if self.fail:
            raise self.fail.pop(0)
        if req.purpose in ("answer", "sample"):
            text = self.answer
            if on_token:
                for w in re.findall(r"\S+\s*", text):
                    on_token(w)
        elif req.purpose == "verify":
            text = json.dumps(self.verify_json or {"claims": [
                {"claim": "Mars has two moons, Phobos and Deimos.", "cited": [1], "verdict": "supported",
                 "source": 1, "quote": "Mars has two small moons named Phobos and Deimos", "note": "stated"},
                {"claim": "Olympus Mons is the tallest volcano in the Solar System.", "cited": [1],
                 "verdict": "supported", "source": 1,
                 "quote": "Olympus Mons on Mars is the tallest volcano known in the Solar System", "note": "stated"},
            ]})
        elif req.purpose == "reflect":
            text = '```json\n{"answers_question": true, "score": 0.9, "issues": [{"type": "omission", "severity": "low", "detail": "Could mention Mars is the fourth planet."}], "summary": "Accurate."}\n```'
        else:
            text = "ok"
        return LLMResponse(text=text, provider=self.name, model=model, latency_ms=1)


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(_env_file=None, DATA_DIR=str(tmp_path / "data"), RERANK_ENABLED=False,
                    GEMINI_API_KEY="", GROQ_API_KEY="", OPENROUTER_API_KEY="", OLLAMA_ENABLED=False,
                    NUM_SAMPLES=2, LLM_MAX_WAIT_SECS=2.0)


@pytest.fixture
def embedder():
    return HashEmbedder(256)
