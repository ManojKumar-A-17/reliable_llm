"""
LLM router: proactive rate limiting, retries, and cross-provider failover.

Why this exists
---------------
Free tiers are tiny (Gemini flash: 5 requests/minute, ~20/day). Instead of
firing requests and eating 429s, each (provider, role) pair gets a sliding-
window limiter that paces calls *before* they are sent. When a provider does
push back, we honour its retry hint — or, if the wait is long and another
provider is configured, switch to it immediately.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

from .. import telemetry
from .base import (
    AUTH, BAD_REQUEST, DAILY_QUOTA, EXHAUSTED, NO_PROVIDER, NOT_FOUND, RATE_LIMIT,
    LLMError, LLMRequest, LLMResponse, TokenCallback,
)
from .providers import Provider

log = logging.getLogger("rllm.llm")


class RateLimiter:
    """Sliding 60-second window limiter with an explicit cooldown."""

    def __init__(self, rpm: int):
        self.rpm = max(0, int(rpm))
        self._times: deque[float] = deque()
        self._cooldown_until = 0.0
        self._lock: Optional[asyncio.Lock] = None

    def _prune(self, now: float) -> None:
        while self._times and now - self._times[0] >= 60.0:
            self._times.popleft()

    def predicted_wait(self) -> float:
        now = time.monotonic()
        wait = max(0.0, self._cooldown_until - now)
        if self.rpm:
            self._prune(now)
            if len(self._times) >= self.rpm:
                wait = max(wait, 60.0 - (now - self._times[0]) + 0.05)
        return wait

    def cooldown(self, seconds: float) -> None:
        self._cooldown_until = max(self._cooldown_until, time.monotonic() + max(0.0, seconds))

    def cooling_for(self) -> float:
        return max(0.0, self._cooldown_until - time.monotonic())

    async def acquire(self, on_wait=None) -> None:
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            while True:
                wait = self.predicted_wait()
                if wait <= 0:
                    if self.rpm:
                        self._times.append(time.monotonic())
                    return
                if on_wait:
                    on_wait(wait)
                await asyncio.sleep(wait)


class LLMRouter:

    def __init__(self, providers: List[Provider], max_retries: int = 3, max_wait: float = 25.0):
        self.providers = providers
        self.max_retries = max(1, max_retries)
        self.max_wait = max_wait
        self._limiters: Dict[Tuple[str, str], RateLimiter] = {}
        # Local servers (Ollama) run one generation at a time on CPU; sending
        # several at once only makes every request slower and time out.
        self._slots: Dict[str, asyncio.Semaphore] = {}
        self.stats = {"calls": 0, "failovers": 0, "rate_limited": 0}

    # ── Introspection ──────────────────────────────────────────────────
    @property
    def ready(self) -> bool:
        return bool(self.providers)

    def describe(self) -> list[dict]:
        out = []
        for p in self.providers:
            d = p.describe()
            d["cooling_secs"] = round(max(
                self._limiter(p, "strong").cooling_for(),
                0.0,
            ), 1)
            out.append(d)
        return out

    def _limiter(self, p: Provider, role: str) -> RateLimiter:
        key = (p.name, p.model_for(role))
        if key not in self._limiters:
            self._limiters[key] = RateLimiter(p.rpm_for(role))
        return self._limiters[key]

    # ── Core ───────────────────────────────────────────────────────────
    async def complete(self, req: LLMRequest, on_token: Optional[TokenCallback] = None) -> LLMResponse:
        if not self.providers:
            raise LLMError(NO_PROVIDER,
                           "No LLM provider is configured. Add GEMINI_API_KEY or GROQ_API_KEY "
                           "(or enable Ollama) in backend/.env and restart the server.")

        errors: List[str] = []
        # Prefer providers that are not cooling down, but keep order otherwise.
        ordered = sorted(self.providers, key=lambda p: self._limiter(p, req.role).cooling_for() > self.max_wait)

        for idx, provider in enumerate(ordered):
            has_next = idx < len(ordered) - 1
            model = provider.model_for(req.role)
            limiter = self._limiter(provider, req.role)
            streamed = {"any": False}

            def _tok(t: str, _s=streamed):
                _s["any"] = True
                if on_token:
                    on_token(t)

            attempt = 0
            while attempt < self.max_retries:
                attempt += 1
                wait = limiter.predicted_wait()
                if wait > self.max_wait and has_next:
                    errors.append(f"{provider.name}: busy for {wait:.0f}s")
                    break

                def _on_wait(w, _p=provider, _m=model):
                    if w >= 1.0:
                        telemetry.notice(
                            f"{_p.label} free-tier limit reached — waiting {w:.0f}s before next call",
                            level="wait", provider=_p.name, model=_m, wait_secs=round(w, 1),
                            purpose=req.purpose,
                        )
                await limiter.acquire(on_wait=_on_wait)

                try:
                    self.stats["calls"] += 1
                    slots = getattr(provider, "max_concurrency", 0)
                    if slots:
                        sem = self._slots.setdefault(provider.name, asyncio.Semaphore(slots))
                        if sem.locked():
                            telemetry.notice(f"{provider.label} is busy with another step — queued",
                                             level="wait", provider=provider.name, purpose=req.purpose)
                        async with sem:
                            resp = await provider.complete(req, model, on_token=_tok if on_token else None)
                    else:
                        resp = await provider.complete(req, model, on_token=_tok if on_token else None)
                    telemetry.emit("llm_call", {
                        "provider": provider.name, "model": model, "purpose": req.purpose,
                        "latency_ms": resp.latency_ms, "input_tokens": resp.input_tokens,
                        "output_tokens": resp.output_tokens, "attempt": attempt, "ok": True,
                    })
                    return resp
                except LLMError as e:
                    log.warning("%s/%s %s failed (%s): %s", provider.name, model, req.purpose, e.kind, e.short())
                    telemetry.emit("llm_call", {
                        "provider": provider.name, "model": model, "purpose": req.purpose,
                        "attempt": attempt, "ok": False, "error": e.kind, "message": e.short(),
                    })
                    errors.append(f"{provider.name}: {e.short()}")

                    if streamed["any"]:
                        # Partial text was already shown — tell the UI to reset it.
                        telemetry.emit("answer_reset", {"reason": e.kind})
                        streamed["any"] = False

                    if e.kind in (AUTH, NOT_FOUND, BAD_REQUEST):
                        if e.kind == AUTH:
                            telemetry.notice(f"{provider.label}: API key rejected — check backend/.env",
                                             level="error", provider=provider.name)
                        elif e.kind == NOT_FOUND:
                            telemetry.notice(f"{provider.label}: model '{model}' is not available on this key",
                                             level="error", provider=provider.name)
                        break
                    if e.kind == DAILY_QUOTA:
                        limiter.cooldown(3600)
                        telemetry.notice(f"{provider.label} daily free quota used up for {model}",
                                         level="error", provider=provider.name)
                        break
                    if e.kind == RATE_LIMIT:
                        self.stats["rate_limited"] += 1
                        delay = e.retry_after if e.retry_after is not None else 5.0 * attempt
                        limiter.cooldown(delay)
                        if (delay > self.max_wait and has_next) or delay > 120:
                            break
                        continue
                    if e.transient:
                        await asyncio.sleep(min(8.0, 1.5 * attempt))
                        continue
                    break

            if has_next:
                self.stats["failovers"] += 1
                nxt = ordered[idx + 1]
                telemetry.notice(f"Switching from {provider.label} to {nxt.label}",
                                 level="failover", provider=nxt.name)

        raise LLMError(EXHAUSTED, "All LLM providers failed: " + " | ".join(errors[-4:]))
