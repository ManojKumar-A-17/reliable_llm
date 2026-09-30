"""
rllm.telemetry
--------------
A tiny, context-local event bus.

The pipeline binds an emitter for the duration of a run; deep layers (the
LLM router, the rate limiter) call :func:`emit` to surface things like
"rate-limited, waiting 12s" to the UI without threading a callback through
every function signature. Outside a run, :func:`emit` is a no-op.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Callable, Iterator, Optional

Emitter = Callable[[str, dict], None]

_emitter: ContextVar[Optional[Emitter]] = ContextVar("rllm_emitter", default=None)


def emit(event: str, data: dict) -> None:
    fn = _emitter.get()
    if fn is not None:
        try:
            fn(event, data)
        except Exception:  # telemetry must never break the pipeline
            pass


def notice(message: str, level: str = "info", **extra) -> None:
    emit("notice", {"level": level, "message": message, **extra})


@contextmanager
def bind(fn: Emitter) -> Iterator[None]:
    token = _emitter.set(fn)
    try:
        yield
    finally:
        _emitter.reset(token)
