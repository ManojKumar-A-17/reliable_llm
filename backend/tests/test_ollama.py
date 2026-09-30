"""Ollama native provider: thinking suppression, streaming filter, errors (mocked HTTP)."""

import asyncio
import json
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rllm.llm import LLMError, LLMRequest, OllamaProvider  # noqa: E402
from rllm.llm.providers import _ThinkFilter, strip_thinking  # noqa: E402


def run(c):
    return asyncio.run(c)


def test_strip_thinking():
    assert strip_thinking("<think>hmm\nplan</think>\n\nMars has two moons [1].") == "Mars has two moons [1]."
    assert strip_thinking("<think>never closed") == ""
    assert strip_thinking("plain") == "plain"
    # thinking-only models: template opens <think>, output only has the closing tag
    assert strip_thinking("Source 1 says... so\n</think>\n\nThe answer [1].") == "The answer [1]."


def test_think_filter_handles_split_tags():
    out = []
    f = _ThinkFilter(out.append)
    for piece in ["<thi", "nk>secret rea", "soning</th", "ink>Hello ", "world [1]", "."]:
        f.feed(piece)
    f.flush()
    assert "".join(out) == "Hello world [1]."


def test_ollama_non_stream_body_and_parse():
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        assert request.url.path == "/api/chat"
        return httpx.Response(200, json={"message": {"content": "<think>x</think>{\"a\": 1}"},
                                         "done": True, "prompt_eval_count": 10, "eval_count": 3})

    p = OllamaProvider("http://localhost:11434/v1", {"strong": "qwen3:4b"}, num_ctx=8192,
                       transport=httpx.MockTransport(handler))
    r = run(p.complete(LLMRequest("q", system="s", json_mode=True, max_tokens=100), "qwen3:4b"))
    assert r.text == '{"a": 1}' and r.input_tokens == 10
    b = seen[0]
    assert b["think"] is False and b["format"] == "json"
    assert b["options"]["num_ctx"] == 8192 and b["options"]["num_predict"] == 100


def test_ollama_stream_and_think_param_fallback():
    calls = []

    def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        if "think" in body:
            return httpx.Response(400, json={"error": "invalid option: think"})
        lines = [{"message": {"content": "Hi "}, "done": False},
                 {"message": {"content": "there"}, "done": False},
                 {"message": {"content": ""}, "done": True, "eval_count": 2}]
        return httpx.Response(200, text="\n".join(json.dumps(l) for l in lines))

    p = OllamaProvider("http://h:11434", {"strong": "m"}, transport=httpx.MockTransport(handler))
    toks = []
    r = run(p.complete(LLMRequest("q"), "m", on_token=toks.append))
    assert r.text == "Hi there" and "".join(toks) == "Hi there" and len(calls) == 2


def test_ollama_model_not_found_message():
    def handler(request):
        return httpx.Response(404, json={"error": "model 'llama3.1:8b' not found"})
    p = OllamaProvider("http://h:11434", {"strong": "llama3.1:8b"}, transport=httpx.MockTransport(handler))
    with pytest.raises(LLMError) as ei:
        run(p.complete(LLMRequest("q"), "llama3.1:8b"))
    assert ei.value.kind == "not_found" and "ollama list" in ei.value.message


def test_think_filter_orphan_close_resets():
    out, resets = [], []
    f = _ThinkFilter(out.append, lambda: (resets.append(1), out.clear()))
    for piece in ["Let me look at source 1", "... done.</thi", "nk>\n\nFinal answer [1]."]:
        f.feed(piece)
    f.flush()
    assert resets == [1] and "".join(out).strip() == "Final answer [1]."
