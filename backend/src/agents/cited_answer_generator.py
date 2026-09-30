"""
cited_answer_generator.py
--------------------------
Generates an answer that cites retrieved evidence chunks inline.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import List

from src.models.gemini_client import GeminiClient
from src.knowledge.knowledge_base import SearchResult
from configs.settings import settings


_SYSTEM = (
    "You are a precise, evidence-based assistant. "
    "Answer using ONLY the provided source passages. "
    "Cite every factual claim with [Source N]. "
    "If sources are insufficient, state what is unknown. "
    "Give a direct, concise answer with no preamble."
)


@dataclass
class CitedAnswer:
    question:   str
    answer:     str
    sources:    List[SearchResult]
    cited_nums: List[int]
    gen_time:   float


def _build_prompt(question: str, sources: List[SearchResult]) -> str:
    context_blocks = "\n\n".join(
        sr.chunk.as_context_block(i + 1)
        for i, sr in enumerate(sources)
    )
    return (
        f"Sources:\n{context_blocks}\n\n"
        f"Question: {question}\n\n"
        "Answer (cite sources inline as [Source N]):"
    )


def _extract_cited_nums(text: str) -> List[int]:
    return sorted(set(int(m) for m in re.findall(r"\[Source\s+(\d+)\]", text, re.IGNORECASE)))


class CitedAnswerGenerator:

    def __init__(self):
        self.llm = GeminiClient(
            model_name=settings.ANSWER_MODEL,
            api_key=settings.require_gemini_key(),
        )

    def run(self, question: str, sources: List[SearchResult]) -> CitedAnswer:
        if not sources:
            return CitedAnswer(
                question=question,
                answer="No relevant sources were found in the knowledge base.",
                sources=[], cited_nums=[], gen_time=0.0,
            )

        t0 = time.time()
        answer = self.llm.generate(
            prompt=_build_prompt(question, sources),
            system=_SYSTEM,
            temperature=0.2,
            max_tokens=800,
        )
        gen_time = round(time.time() - t0, 2)

        cited_nums = _extract_cited_nums(answer)
        if not cited_nums and sources:
            cited_nums = [1]

        return CitedAnswer(
            question=question,
            answer=answer,
            sources=sources,
            cited_nums=cited_nums,
            gen_time=gen_time,
        )
