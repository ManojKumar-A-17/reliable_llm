"""Shared types for the LLM layer."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

TokenCallback = Callable[[str], None]

# Error kinds -----------------------------------------------------------------
RATE_LIMIT = "rate_limit"      # per-minute limit; wait and retry
DAILY_QUOTA = "daily_quota"    # per-day limit; fail over, cool down
AUTH = "auth"                  # bad / missing key
BAD_REQUEST = "bad_request"    # our request is wrong for this model
NOT_FOUND = "not_found"        # model name not available
SERVER = "server"              # 5xx — transient
NETWORK = "network"            # connection / timeout — transient
EMPTY = "empty"                # model returned no text
NO_PROVIDER = "no_provider"
EXHAUSTED = "exhausted"

_TRANSIENT = {RATE_LIMIT, SERVER, NETWORK, EMPTY}


class LLMError(Exception):
    def __init__(self, kind: str, message: str, retry_after: Optional[float] = None,
                 provider: str = "", model: str = ""):
        super().__init__(message)
        self.kind = kind
        self.message = message
        self.retry_after = retry_after
        self.provider = provider
        self.model = model

    @property
    def transient(self) -> bool:
        return self.kind in _TRANSIENT

    def short(self) -> str:
        msg = self.message.strip().splitlines()[0] if self.message else self.kind
        return msg[:220]


@dataclass
class LLMRequest:
    prompt: str
    system: Optional[str] = None
    temperature: float = 0.2
    max_tokens: int = 1024
    json_mode: bool = False
    role: str = "strong"          # "strong" (answer/judge) | "fast" (samples)
    purpose: str = ""             # label for telemetry, e.g. "answer", "verify"


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    latency_ms: int
    input_tokens: int = 0
    output_tokens: int = 0
    meta: Dict = field(default_factory=dict)


# Helpers ---------------------------------------------------------------------

_DURATION_RE = re.compile(r"(?:(\d+(?:\.\d+)?)h)?(?:(\d+(?:\.\d+)?)m(?!s))?(?:(\d+(?:\.\d+)?)s)?")


def parse_duration(value: str) -> Optional[float]:
    """Parse '37s', '1m2.5s', '2m59.56s', '12' → seconds."""
    if value is None:
        return None
    value = str(value).strip()
    try:
        return float(value)
    except ValueError:
        pass
    m = _DURATION_RE.fullmatch(value)
    if not m or not any(m.groups()):
        return None
    h, mi, s = (float(g) if g else 0.0 for g in m.groups())
    return h * 3600 + mi * 60 + s


_RETRY_TEXT_RE = re.compile(
    r"(?:retry|try again)\s*(?:in|after)?\s*(\d+(?:\.\d+)?)\s*(ms|s|sec|seconds)?", re.I
)


def parse_retry_from_text(text: str) -> Optional[float]:
    m = _RETRY_TEXT_RE.search(text or "")
    if not m:
        return None
    val = float(m.group(1))
    return val / 1000 if (m.group(2) or "").lower() == "ms" else val
