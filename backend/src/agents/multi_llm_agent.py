"""
multi_llm_agent.py
------------------
Queries multiple LLMs in parallel via the Groq API and returns
all responses for comparison.

Primary model (first in MULTI_LLM_MODELS) drives the confidence
pipeline; others are shown for comparison only.
"""

from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List

from groq import Groq

from configs.config import GROQ_API_KEY, MULTI_LLM_MODELS

# System prompt applied to every model call
_SYSTEM_PROMPT = (
    "You are a helpful, knowledgeable assistant. "
    "Answer the question accurately and concisely based on the provided context. "
    "If the context does not contain enough information, use your general knowledge. "
    "Give a direct answer without any preamble or meta-commentary."
)


def _strip_think(text: str) -> str:
    """Remove <think>…</think> reasoning blocks emitted by qwen3 models."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return cleaned if cleaned else text.strip()


@dataclass
class ModelResponse:
    model_name: str
    display_name: str
    answer: str
    generation_time: float
    error: str = ""

    def as_dict(self) -> dict:
        return {
            "model_name":      self.model_name,
            "display_name":    self.display_name,
            "answer":          self.answer,
            "generation_time": round(self.generation_time, 2),
            "error":           self.error,
        }


@dataclass
class MultiLLMResult:
    question: str
    responses: List[ModelResponse]
    total_time: float

    def best_answer(self) -> str:
        """Return the first successful answer (primary model)."""
        for resp in self.responses:
            if resp.answer and not resp.error:
                return resp.answer
        for resp in self.responses:
            if resp.answer:
                return resp.answer
        return "No answer available."

    def as_dict_list(self) -> List[dict]:
        return [r.as_dict() for r in self.responses]


class MultiLLMAgent:
    """
    Queries all configured LLMs in parallel and collects their answers.
    """

    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)
        self.models: dict = MULTI_LLM_MODELS

    def _query_model(
        self,
        model_name: str,
        display_name: str,
        question: str,
        context: str,
    ) -> ModelResponse:
        t_start = time.time()
        try:
            user_content = (
                f"Context:\n{context}\n\nQuestion: {question}"
                if context.strip()
                else f"Question: {question}"
            )
            completion = self.client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": user_content},
                ],
                temperature=0.3,
                max_tokens=800,
            )
            raw = completion.choices[0].message.content.strip()
            answer = _strip_think(raw)
            return ModelResponse(
                model_name=model_name,
                display_name=display_name,
                answer=answer,
                generation_time=time.time() - t_start,
            )
        except Exception as exc:
            return ModelResponse(
                model_name=model_name,
                display_name=display_name,
                answer="",
                generation_time=time.time() - t_start,
                error=str(exc),
            )

    def run(self, question: str, context: str = "") -> MultiLLMResult:
        t_start = time.time()
        model_items = list(self.models.items())

        futures = {}
        with ThreadPoolExecutor(max_workers=len(model_items)) as executor:
            for model_name, display_name in model_items:
                fut = executor.submit(
                    self._query_model,
                    model_name, display_name, question, context,
                )
                futures[fut] = model_name

            results_by_model: dict[str, ModelResponse] = {}
            for fut in as_completed(futures):
                resp = fut.result()
                results_by_model[resp.model_name] = resp

        ordered_responses = [
            results_by_model[model_name]
            for model_name, _ in model_items
            if model_name in results_by_model
        ]

        return MultiLLMResult(
            question=question,
            responses=ordered_responses,
            total_time=round(time.time() - t_start, 2),
        )
