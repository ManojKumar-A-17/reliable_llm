"""
LLM provider adapters.

* :class:`GeminiProvider` talks to the native Gemini REST API (no SDK — the
  old ``google-generativeai`` package is deprecated). Native REST gives us
  control over thinking budget, safety settings and JSON output.
* :class:`OpenAICompatProvider` covers Groq, OpenRouter and Ollama, which all
  expose an OpenAI-style ``/chat/completions`` endpoint.

Both support streaming via an ``on_token`` callback and raise
:class:`~rllm.llm.base.LLMError` with a normalised ``kind`` so the router can
decide whether to wait, retry, or fail over.
"""

from __future__ import annotations

import json
import re
import time
from typing import Dict, Optional

import httpx

from .. import telemetry
from .base import (
    AUTH, BAD_REQUEST, DAILY_QUOTA, EMPTY, NETWORK, NOT_FOUND, RATE_LIMIT, SERVER,
    LLMError, LLMRequest, LLMResponse, TokenCallback, parse_duration, parse_retry_from_text,
)


class Provider:
    """Interface every provider implements."""

    name: str = "base"
    label: str = "Base"

    def __init__(self, models: Dict[str, str], rpm: Dict[str, int], timeout: float = 90.0):
        self.models = models          # {"strong": "...", "fast": "..."}
        self.rpm = rpm                # {"strong": 5, "fast": 10}; 0 = unlimited
        self.timeout = timeout

    def model_for(self, role: str) -> str:
        return self.models.get(role) or self.models["strong"]

    def rpm_for(self, role: str) -> int:
        return self.rpm.get(role, self.rpm.get("strong", 0))

    async def complete(self, req: LLMRequest, model: str,
                       on_token: Optional[TokenCallback] = None) -> LLMResponse:
        raise NotImplementedError

    def describe(self) -> dict:
        return {"name": self.name, "label": self.label, "models": dict(self.models)}


def _status_to_kind(status: int, body: str) -> str:
    low = body.lower()
    if status == 429:
        if "perday" in low.replace("_", "").replace(" ", "") or "per day" in low or "daily" in low:
            return DAILY_QUOTA
        return RATE_LIMIT
    if status in (401, 403):
        return AUTH
    if status == 404:
        return NOT_FOUND
    if status in (400, 413, 422):
        if "quota" in low and "exceed" in low:
            return RATE_LIMIT
        return BAD_REQUEST
    if status >= 500:
        return SERVER
    return BAD_REQUEST


def _error_message(body: str) -> str:
    try:
        data = json.loads(body)
        err = data.get("error", data)
        if isinstance(err, dict):
            return str(err.get("message") or err)
        return str(err)
    except Exception:
        return body[:400]


# ════════════════════════════════════════════════════════════════════════════
# Gemini (native REST)
# ════════════════════════════════════════════════════════════════════════════

_GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"
_SAFETY = [
    {"category": c, "threshold": "BLOCK_NONE"}
    for c in (
        "HARM_CATEGORY_HARASSMENT",
        "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "HARM_CATEGORY_DANGEROUS_CONTENT",
    )
]


class GeminiProvider(Provider):
    name = "gemini"
    label = "Google Gemini"

    def __init__(self, api_key: str, models: Dict[str, str], rpm: Dict[str, int],
                 thinking_level: str = "low", timeout: float = 90.0,
                 transport: Optional[httpx.AsyncBaseTransport] = None):
        super().__init__(models, rpm, timeout)
        self._key = api_key
        self._thinking_level = (thinking_level or "").strip().lower()
        self._no_thinking_cfg: set[str] = set()   # models that rejected thinkingConfig
        self._no_json_mode: set[str] = set()
        self._transport = transport

    def _body(self, req: LLMRequest, model: str) -> dict:
        gen: dict = {
            "temperature": req.temperature,
            "maxOutputTokens": req.max_tokens,
        }
        if req.json_mode and model not in self._no_json_mode:
            gen["responseMimeType"] = "application/json"
        if self._thinking_level and model not in self._no_thinking_cfg:
            gen["thinkingConfig"] = {"thinkingLevel": self._thinking_level}
        body = {
            "contents": [{"role": "user", "parts": [{"text": req.prompt}]}],
            "generationConfig": gen,
            "safetySettings": _SAFETY,
        }
        if req.system:
            body["systemInstruction"] = {"parts": [{"text": req.system}]}
        return body

    @staticmethod
    def _retry_after(body: str) -> Optional[float]:
        try:
            for d in json.loads(body).get("error", {}).get("details", []) or []:
                if "retryDelay" in d:
                    return parse_duration(d["retryDelay"])
        except Exception:
            pass
        return parse_retry_from_text(body)

    @staticmethod
    def _extract(payload: dict) -> tuple[str, str]:
        text, finish = "", ""
        for cand in payload.get("candidates", []) or []:
            finish = cand.get("finishReason", finish) or finish
            for part in (cand.get("content") or {}).get("parts", []) or []:
                if part.get("thought"):
                    continue
                text += part.get("text", "")
            break
        return text, finish

    def _raise_for(self, status: int, body: str, model: str) -> None:
        kind = _status_to_kind(status, body)
        raise LLMError(kind, f"Gemini {status}: {_error_message(body)}",
                       retry_after=self._retry_after(body) if kind == RATE_LIMIT else None,
                       provider=self.name, model=model)

    async def complete(self, req: LLMRequest, model: str,
                       on_token: Optional[TokenCallback] = None) -> LLMResponse:
        # One silent retry if the model rejects an optional config field.
        for _ in range(3):
            try:
                return await self._complete_once(req, model, on_token)
            except LLMError as e:
                low = e.message.lower()
                if e.kind == BAD_REQUEST and "thinking" in low and model not in self._no_thinking_cfg:
                    self._no_thinking_cfg.add(model)
                    continue
                if e.kind == BAD_REQUEST and ("mime" in low or "json" in low) and req.json_mode \
                        and model not in self._no_json_mode:
                    self._no_json_mode.add(model)
                    continue
                raise
        raise LLMError(BAD_REQUEST, "Gemini rejected the request configuration",
                       provider=self.name, model=model)

    async def _complete_once(self, req: LLMRequest, model: str,
                             on_token: Optional[TokenCallback]) -> LLMResponse:
        headers = {"x-goog-api-key": self._key, "Content-Type": "application/json"}
        body = self._body(req, model)
        t0 = time.perf_counter()
        text, finish, usage = "", "", {}
        try:
            async with httpx.AsyncClient(timeout=self.timeout, transport=self._transport) as client:
                if on_token is None:
                    r = await client.post(f"{_GEMINI_BASE}/models/{model}:generateContent",
                                          headers=headers, json=body)
                    if r.status_code != 200:
                        self._raise_for(r.status_code, r.text, model)
                    payload = r.json()
                    text, finish = self._extract(payload)
                    usage = payload.get("usageMetadata", {}) or {}
                else:
                    url = f"{_GEMINI_BASE}/models/{model}:streamGenerateContent?alt=sse"
                    async with client.stream("POST", url, headers=headers, json=body) as r:
                        if r.status_code != 200:
                            self._raise_for(r.status_code, (await r.aread()).decode(errors="replace"), model)
                        async for line in r.aiter_lines():
                            if not line.startswith("data:"):
                                continue
                            try:
                                payload = json.loads(line[5:].strip())
                            except json.JSONDecodeError:
                                continue
                            delta, fin = self._extract(payload)
                            finish = fin or finish
                            usage = payload.get("usageMetadata", usage) or usage
                            if delta:
                                text += delta
                                on_token(delta)
        except httpx.TimeoutException as e:
            raise LLMError(NETWORK, f"Gemini timeout: {e}", provider=self.name, model=model)
        except httpx.HTTPError as e:
            raise LLMError(NETWORK, f"Gemini network error: {e}", provider=self.name, model=model)

        if not text.strip():
            raise LLMError(EMPTY, f"Gemini returned no text (finishReason={finish or 'unknown'})",
                           provider=self.name, model=model)
        return LLMResponse(
            text=text.strip(), provider=self.name, model=model,
            latency_ms=int((time.perf_counter() - t0) * 1000),
            input_tokens=int(usage.get("promptTokenCount", 0) or 0),
            output_tokens=int(usage.get("candidatesTokenCount", 0) or 0),
            meta={"finish_reason": finish},
        )


# ════════════════════════════════════════════════════════════════════════════
# OpenAI-compatible (Groq, OpenRouter, Ollama)
# ════════════════════════════════════════════════════════════════════════════

class OpenAICompatProvider(Provider):

    def __init__(self, name: str, label: str, base_url: str, api_key: str,
                 models: Dict[str, str], rpm: Dict[str, int], timeout: float = 90.0,
                 extra_headers: Optional[Dict[str, str]] = None,
                 transport: Optional[httpx.AsyncBaseTransport] = None):
        super().__init__(models, rpm, timeout)
        self.name = name
        self.label = label
        self._base = base_url.rstrip("/")
        self._key = api_key
        self._headers = extra_headers or {}
        self._no_json_mode: set[str] = set()
        self._transport = transport

    def _retry_after(self, resp_headers: httpx.Headers, body: str) -> Optional[float]:
        for h in ("retry-after", "x-ratelimit-reset-requests"):
            if h in resp_headers:
                v = parse_duration(resp_headers[h])
                if v is not None:
                    return v
        return parse_retry_from_text(body)

    async def complete(self, req: LLMRequest, model: str,
                       on_token: Optional[TokenCallback] = None) -> LLMResponse:
        try:
            return await self._complete_once(req, model, on_token)
        except LLMError as e:
            if e.kind == BAD_REQUEST and req.json_mode and model not in self._no_json_mode:
                # Many free models don't support response_format; our prompts
                # still ask for JSON, so just retry without the flag.
                self._no_json_mode.add(model)
                return await self._complete_once(req, model, on_token)
            raise

    async def _complete_once(self, req: LLMRequest, model: str,
                             on_token: Optional[TokenCallback]) -> LLMResponse:
        messages = []
        if req.system:
            messages.append({"role": "system", "content": req.system})
        messages.append({"role": "user", "content": req.prompt})
        body: dict = {
            "model": model,
            "messages": messages,
            "temperature": req.temperature,
            "max_tokens": req.max_tokens,
            "stream": on_token is not None,
        }
        if req.json_mode and model not in self._no_json_mode:
            body["response_format"] = {"type": "json_object"}

        headers = {"Content-Type": "application/json", **self._headers}
        if self._key:
            headers["Authorization"] = f"Bearer {self._key}"

        url = f"{self._base}/chat/completions"
        t0 = time.perf_counter()
        text, usage, finish = "", {}, ""
        try:
            async with httpx.AsyncClient(timeout=self.timeout, transport=self._transport) as client:
                if on_token is None:
                    r = await client.post(url, headers=headers, json=body)
                    if r.status_code != 200:
                        self._raise_for(r.status_code, r.text, r.headers, model)
                    payload = r.json()
                    choice = (payload.get("choices") or [{}])[0]
                    text = (choice.get("message") or {}).get("content") or ""
                    finish = choice.get("finish_reason") or ""
                    usage = payload.get("usage") or {}
                else:
                    async with client.stream("POST", url, headers=headers, json=body) as r:
                        if r.status_code != 200:
                            raw = (await r.aread()).decode(errors="replace")
                            self._raise_for(r.status_code, raw, r.headers, model)
                        async for line in r.aiter_lines():
                            if not line.startswith("data:"):
                                continue
                            data = line[5:].strip()
                            if data == "[DONE]":
                                break
                            try:
                                payload = json.loads(data)
                            except json.JSONDecodeError:
                                continue
                            if payload.get("error"):
                                raise LLMError(SERVER, f"{self.label} stream error: {payload['error']}",
                                               provider=self.name, model=model)
                            choice = (payload.get("choices") or [{}])[0]
                            delta = (choice.get("delta") or {}).get("content") or ""
                            finish = choice.get("finish_reason") or finish
                            usage = payload.get("usage") or usage
                            if delta:
                                text += delta
                                on_token(delta)
        except httpx.TimeoutException as e:
            raise LLMError(NETWORK, f"{self.label} timeout: {e}", provider=self.name, model=model)
        except httpx.ConnectError as e:
            raise LLMError(NETWORK, f"{self.label} unreachable at {self._base}: {e}",
                           provider=self.name, model=model)
        except httpx.HTTPError as e:
            raise LLMError(NETWORK, f"{self.label} network error: {e}", provider=self.name, model=model)

        if not text.strip():
            raise LLMError(EMPTY, f"{self.label} returned no text (finish={finish or 'unknown'})",
                           provider=self.name, model=model)
        return LLMResponse(
            text=text.strip(), provider=self.name, model=model,
            latency_ms=int((time.perf_counter() - t0) * 1000),
            input_tokens=int(usage.get("prompt_tokens", 0) or 0),
            output_tokens=int(usage.get("completion_tokens", 0) or 0),
            meta={"finish_reason": finish},
        )

    def _raise_for(self, status: int, body: str, headers: httpx.Headers, model: str) -> None:
        kind = _status_to_kind(status, body)
        raise LLMError(kind, f"{self.label} {status}: {_error_message(body)}",
                       retry_after=self._retry_after(headers, body) if kind == RATE_LIMIT else None,
                       provider=self.name, model=model)


# ════════════════════════════════════════════════════════════════════════════
# Ollama (native /api/chat — needed for num_ctx and to turn off "thinking")
# ════════════════════════════════════════════════════════════════════════════

_THINK_BLOCK = re.compile(r"<think>.*?(</think>|$)", re.S | re.I)


def strip_thinking(text: str) -> str:
    """Remove <think>…</think> reasoning that some local models inline.

    Thinking-only models (e.g. qwen3 "2507") get the opening <think> from the
    chat template, so their output is "reasoning…</think>answer" — keep only
    what follows the last closing tag.
    """
    text = _THINK_BLOCK.sub("", text or "")
    if "</think>" in text.lower():
        text = text[text.lower().rfind("</think>") + 8:]
    return text.strip()


class _ThinkFilter:
    """Streaming filter: drops everything between <think> and </think>."""

    def __init__(self, on_token: TokenCallback, on_reset=None):
        self.on_token = on_token
        self.on_reset = on_reset
        self.buf = ""
        self.inside = False

    def feed(self, delta: str) -> None:
        self.buf += delta
        while True:
            orphan = -1 if self.inside else self.buf.lower().find("</think>")
            if orphan != -1:
                # Reasoning was streamed without an opening tag: discard it.
                if self.on_reset:
                    self.on_reset()
                self.buf = self.buf[orphan + 8:]
                continue
            if self.inside:
                end = self.buf.lower().find("</think>")
                if end == -1:
                    self.buf = self.buf[-8:]       # keep a tail in case the tag is split
                    return
                self.buf = self.buf[end + 8:]
                self.inside = False
            else:
                start = self.buf.lower().find("<think>")
                if start == -1:
                    safe = len(self.buf) - 8 if "<" in self.buf[-8:] else len(self.buf)
                    if safe > 0:
                        self.on_token(self.buf[:safe])
                        self.buf = self.buf[safe:]
                    return
                if start:
                    self.on_token(self.buf[:start])
                self.buf = self.buf[start + 7:]
                self.inside = True

    def flush(self) -> None:
        if self.buf and not self.inside:
            self.on_token(self.buf)
        self.buf = ""


class OllamaProvider(Provider):
    name = "ollama"
    label = "Ollama (local)"
    max_concurrency = 1          # one generation at a time (see LLMRouter)

    def __init__(self, base_url: str, models: Dict[str, str], num_ctx: int = 8192,
                 think: bool = False, timeout: float = 300.0, max_tokens: int = 700,
                 transport: Optional[httpx.AsyncBaseTransport] = None):
        super().__init__(models, {"strong": 0, "fast": 0}, timeout)
        self._base = base_url.rstrip("/").removesuffix("/v1")
        self._num_ctx = num_ctx
        self._think = think
        self._max_tokens = max_tokens
        self._no_think_param: set[str] = set()   # older Ollama / models that reject "think"
        self._transport = transport

    async def complete(self, req: LLMRequest, model: str,
                       on_token: Optional[TokenCallback] = None) -> LLMResponse:
        try:
            return await self._complete_once(req, model, on_token)
        except LLMError as e:
            if e.kind == BAD_REQUEST and "think" in e.message.lower() and model not in self._no_think_param:
                self._no_think_param.add(model)
                return await self._complete_once(req, model, on_token)
            raise

    async def _complete_once(self, req: LLMRequest, model: str,
                             on_token: Optional[TokenCallback]) -> LLMResponse:
        messages = []
        if req.system:
            messages.append({"role": "system", "content": req.system})
        messages.append({"role": "user", "content": req.prompt})
        # Reasoning models spend tokens thinking before they answer; give them room.
        budget = req.max_tokens * 4 if self._think else min(req.max_tokens, self._max_tokens)
        body: dict = {
            "model": model,
            "messages": messages,
            "stream": on_token is not None,
            "options": {"temperature": req.temperature, "num_predict": budget, "num_ctx": self._num_ctx},
            "keep_alive": "30m",     # don't unload/reload the model between pipeline steps
        }
        if model not in self._no_think_param:
            body["think"] = self._think
        if req.json_mode:
            body["format"] = "json"

        url = f"{self._base}/api/chat"
        t0 = time.perf_counter()
        text, usage = "", {}
        filt = _ThinkFilter(on_token, lambda: telemetry.emit("answer_reset", {"reason": "thinking"})) \
            if on_token else None
        try:
            async with httpx.AsyncClient(timeout=self.timeout, transport=self._transport) as client:
                if on_token is None:
                    r = await client.post(url, json=body)
                    if r.status_code != 200:
                        self._raise_for(r.status_code, r.text, model)
                    payload = r.json()
                    text = (payload.get("message") or {}).get("content") or ""
                    usage = payload
                else:
                    async with client.stream("POST", url, json=body) as r:
                        if r.status_code != 200:
                            self._raise_for(r.status_code, (await r.aread()).decode(errors="replace"), model)
                        async for line in r.aiter_lines():
                            if not line.strip():
                                continue
                            try:
                                payload = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            if payload.get("error"):
                                raise LLMError(SERVER, f"Ollama: {payload['error']}", provider=self.name, model=model)
                            delta = (payload.get("message") or {}).get("content") or ""
                            if delta:
                                text += delta
                                filt.feed(delta)
                            if payload.get("done"):
                                usage = payload
                        filt.flush()
        except httpx.TimeoutException as e:
            raise LLMError(NETWORK, f"Ollama timeout after {self.timeout:.0f}s: {e}", provider=self.name, model=model)
        except httpx.ConnectError:
            raise LLMError(NETWORK, f"Ollama is not running at {self._base} — start it with `ollama serve`",
                           provider=self.name, model=model)
        except httpx.HTTPError as e:
            raise LLMError(NETWORK, f"Ollama network error: {e}", provider=self.name, model=model)

        text = strip_thinking(text)
        if not text:
            raise LLMError(EMPTY, "Ollama returned no text (the model may have used its whole budget thinking)",
                           provider=self.name, model=model)
        return LLMResponse(
            text=text, provider=self.name, model=model,
            latency_ms=int((time.perf_counter() - t0) * 1000),
            input_tokens=int(usage.get("prompt_eval_count", 0) or 0),
            output_tokens=int(usage.get("eval_count", 0) or 0),
            meta={"finish_reason": usage.get("done_reason", "")},
        )

    def _raise_for(self, status: int, body: str, model: str) -> None:
        kind = _status_to_kind(status, body)
        msg = _error_message(body)
        if kind == NOT_FOUND:
            msg += f" — run `ollama pull {model}` or set OLLAMA_MODEL to a name from `ollama list`"
        raise LLMError(kind, f"Ollama {status}: {msg}", provider=self.name, model=model)


# ════════════════════════════════════════════════════════════════════════════
# Factory
# ════════════════════════════════════════════════════════════════════════════

def build_providers(settings) -> list[Provider]:
    """Instantiate every configured provider in the configured order."""
    out: list[Provider] = []
    for name in settings.provider_order:
        if name == "gemini" and settings.GEMINI_API_KEY:
            out.append(GeminiProvider(
                api_key=settings.GEMINI_API_KEY,
                models={"strong": settings.GEMINI_MODEL, "fast": settings.GEMINI_FAST_MODEL},
                rpm={"strong": settings.GEMINI_RPM, "fast": settings.GEMINI_FAST_RPM},
                thinking_level=settings.GEMINI_THINKING_LEVEL,
                timeout=settings.LLM_TIMEOUT_SECS,
            ))
        elif name == "groq" and settings.GROQ_API_KEY:
            out.append(OpenAICompatProvider(
                "groq", "Groq", "https://api.groq.com/openai/v1", settings.GROQ_API_KEY,
                models={"strong": settings.GROQ_MODEL, "fast": settings.GROQ_FAST_MODEL},
                rpm={"strong": settings.GROQ_RPM, "fast": settings.GROQ_FAST_RPM},
                timeout=settings.LLM_TIMEOUT_SECS,
            ))
        elif name == "openrouter" and settings.OPENROUTER_API_KEY:
            out.append(OpenAICompatProvider(
                "openrouter", "OpenRouter", "https://openrouter.ai/api/v1", settings.OPENROUTER_API_KEY,
                models={"strong": settings.OPENROUTER_MODEL, "fast": settings.OPENROUTER_FAST_MODEL},
                rpm={"strong": settings.OPENROUTER_RPM, "fast": settings.OPENROUTER_FAST_RPM},
                timeout=settings.LLM_TIMEOUT_SECS,
                extra_headers={"HTTP-Referer": "http://localhost:3000", "X-Title": "Reliable LLM"},
            ))
        elif name == "ollama" and settings.OLLAMA_ENABLED:
            out.append(OllamaProvider(
                base_url=settings.OLLAMA_BASE_URL,
                models={"strong": settings.OLLAMA_MODEL,
                        "fast": settings.OLLAMA_FAST_MODEL or settings.OLLAMA_MODEL},
                num_ctx=settings.OLLAMA_NUM_CTX, think=settings.OLLAMA_THINK,
                max_tokens=settings.OLLAMA_MAX_TOKENS,
                timeout=max(settings.LLM_TIMEOUT_SECS, 300),
            ))
    return out
