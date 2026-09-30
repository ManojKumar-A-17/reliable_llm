"""
sampler.py
----------
Generates N sample answers WITH retrieved context for consistency scoring.
The old version sampled without context, making consistency meaningless.
"""

from __future__ import annotations

from typing import List

from configs.settings import settings
from src.knowledge.knowledge_base import SearchResult


def _build_prompt(question: str, sources: List[SearchResult]) -> str:
    if not sources:
        return f"Question: {question}"
    blocks = "\n\n".join(
        sr.chunk.as_context_block(i + 1) for i, sr in enumerate(sources)
    )
    return f"Sources:\n{blocks}\n\nQuestion: {question}"


class ResponseSampler:
    """
    Draw N independent answers at higher temperature for consistency scoring.
    Each sample uses the same retrieved context as the main answer.
    """

    def __init__(self, generator):
        self.generator = generator

    def sample(
        self,
        question: str,
        sources: List[SearchResult] = None,
        num_samples: int = None,
    ) -> List[str]:
        n = num_samples or settings.NUM_SAMPLES
        prompt = _build_prompt(question, sources or [])
        responses: List[str] = []

        for i in range(n):
            answer = self.generator.generate(
                prompt=prompt,
                do_sample=True,
                temperature=settings.TEMPERATURE,
                top_p=settings.TOP_P,
                max_tokens=400,
            )
            if answer:
                responses.append(answer)

        return responses
