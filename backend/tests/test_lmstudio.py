"""LM Studio provider: built from settings, no auth header, serialised calls (mocked HTTP)."""

import asyncio
import json
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rllm.config import Settings  # noqa: E402
from rllm.llm import LLMRequest  # noqa: E402
from rllm.llm.providers import build_providers  # noqa: E402


def _settings(**kw):
    return Settings(_env_file=None, LLM_PROVIDER_ORDER="lmstudio", LMSTUDIO_ENABLED=True,
                    LMSTUDIO_MODEL="big", LMSTUDIO_FAST_MODEL="", **kw)


def test_lmstudio_skipped_when_disabled():
    assert build_providers(Settings(_env_file=None, LLM_PROVIDER_ORDER="lmstudio")) == []


def test_lmstudio_built_from_settings():
    [p] = build_providers(_settings())
    assert p.name == "lmstudio" and p.max_concurrency == 1
    assert p.model_for("strong") == "big" and p.model_for("fast") == "big"
    assert p.rpm_for("strong") == 0


def test_lmstudio_request_has_no_auth_header():
    seen = []

    def handler(request: httpx.Request):
        seen.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": "Hi [1]."},
                                                      "finish_reason": "stop"}]})

    [p] = build_providers(_settings())
    p._transport = httpx.MockTransport(handler)
    resp = asyncio.run(p.complete(LLMRequest(prompt="q"), "big"))
    assert resp.text == "Hi [1]." and resp.provider == "lmstudio"
    assert str(seen[0].url) == "http://localhost:1234/v1/chat/completions"
    assert "authorization" not in seen[0].headers
    assert json.loads(seen[0].content)["model"] == "big"


def test_lmstudio_restarts_server_when_unreachable(monkeypatch):
    import subprocess
    from rllm.llm import providers as prov
    state = {"up": False, "started": 0}

    def handler(request: httpx.Request):
        if not state["up"]:
            raise httpx.ConnectError("refused", request=request)
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": []})
        return httpx.Response(200, json={"choices": [{"message": {"content": "Back [1]."}}]})

    def fake_run(cmd, **kw):
        assert cmd[1:] == ["server", "start"]
        state["started"] += 1
        state["up"] = True

    monkeypatch.setattr(prov, "_lms_executable", lambda: "lms")
    monkeypatch.setattr(subprocess, "run", fake_run)
    [p] = build_providers(_settings())
    p._transport = httpx.MockTransport(handler)
    resp = asyncio.run(p.complete(LLMRequest(prompt="q"), "big"))
    assert resp.text == "Back [1]." and state["started"] == 1
