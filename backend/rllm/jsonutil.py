"""Tolerant JSON extraction from LLM output."""

from __future__ import annotations

import json
import re
from typing import Any, Optional

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S | re.I)


def _balanced(text: str, open_ch: str, close_ch: str) -> Optional[str]:
    start = text.find(open_ch)
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == open_ch:
                depth += 1
            elif c == close_ch:
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        start = text.find(open_ch, start + 1)
    return None


def _loads(s: str) -> Any:
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        # common LLM slip: trailing commas
        return json.loads(re.sub(r",\s*([}\]])", r"\1", s))


def extract_json(text: str) -> Optional[Any]:
    """Return the first JSON object/array found in *text*, or None."""
    if not text:
        return None
    candidates = [m.group(1) for m in _FENCE.finditer(text)] + [text]
    for cand in candidates:
        cand = cand.strip()
        try:
            return _loads(cand)
        except (json.JSONDecodeError, ValueError):
            pass
        for o, c in (("{", "}"), ("[", "]")):
            block = _balanced(cand, o, c)
            if block:
                try:
                    return _loads(block)
                except (json.JSONDecodeError, ValueError):
                    continue
    return None
