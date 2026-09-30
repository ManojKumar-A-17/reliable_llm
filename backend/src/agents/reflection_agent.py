"""
reflection_agent.py
--------------------
Self-reflection: reviews the generated answer for mistakes, unsupported
claims, ambiguities, and missing information.
Now uses Gemini instead of Groq — behaviour unchanged.
"""

import json
import re
import time
from dataclasses import dataclass

from src.models.gemini_client import GeminiClient
from configs.settings import settings


@dataclass
class ReflectionResult:
    question:            str
    original_answer:     str
    mistakes:            list
    unsupported_claims:  list
    ambiguities:         list
    contradictions:      list
    missing_information: list
    reasoning:           str
    improved_answer:     str
    reflection_score:    float
    reflection_time:     float


_SYSTEM = (
    "You are an expert AI reviewer. "
    "Critically review the given answer — do NOT re-answer the question. "
    "Return ONLY valid JSON."
)

_PROMPT_TEMPLATE = """Question:
{question}

Answer:
{answer}

Perform the following:
1. Identify factual mistakes.
2. Identify unsupported claims.
3. Identify ambiguities.
4. Identify missing information.
5. Identify contradictions.
6. Compute a reflection_score from 0.0 (many issues) to 1.0 (perfect).

Return ONLY valid JSON:
{{
    "mistakes": ["..."],
    "unsupported_claims": ["..."],
    "ambiguities": ["..."],
    "contradictions": ["..."],
    "missing_information": ["..."],
    "reasoning": "One sentence summary.",
    "improved_answer": "Corrected answer if needed, else repeat the original.",
    "reflection_score": 0.85
}}
"""


class ReflectionAgent:

    def __init__(self):
        self.llm = GeminiClient(
            model_name=settings.REFLECTION_MODEL,
            api_key=settings.require_gemini_key(),
        )

    def _extract_json(self, text: str):
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
        return None

    def _score_reflection(self, mistakes, unsupported, ambiguities, contradictions, missing):
        score = 1.0
        score -= len(mistakes)      * 0.20
        score -= len(unsupported)   * 0.15
        score -= len(contradictions)* 0.25
        score -= len(ambiguities)   * 0.10
        score -= len(missing)       * 0.08
        return max(0.0, round(score, 4))

    def run(self, question: str, answer: str) -> ReflectionResult:
        print("[Reflection] Processing…")
        t0 = time.time()

        prompt = _PROMPT_TEMPLATE.format(question=question, answer=answer)
        raw = self.llm.generate(prompt=prompt, system=_SYSTEM, temperature=0.2, max_tokens=600)
        elapsed = round(time.time() - t0, 2)

        data = self._extract_json(raw) or {}

        mistakes     = data.get("mistakes", []) or []
        unsupported  = data.get("unsupported_claims", []) or []
        ambiguities  = data.get("ambiguities", []) or []
        contradictions = data.get("contradictions", []) or []
        missing      = data.get("missing_information", []) or []

        score = data.get("reflection_score")
        if score is None:
            score = self._score_reflection(mistakes, unsupported, ambiguities, contradictions, missing)
        else:
            score = max(0.0, min(1.0, float(score)))

        print(f"[Reflection] Score: {score:.2f}  ({elapsed}s)")

        return ReflectionResult(
            question=question,
            original_answer=answer,
            mistakes=mistakes,
            unsupported_claims=unsupported,
            ambiguities=ambiguities,
            contradictions=contradictions,
            missing_information=missing,
            reasoning=data.get("reasoning", ""),
            improved_answer=data.get("improved_answer", answer),
            reflection_score=round(score, 4),
            reflection_time=elapsed,
        )
