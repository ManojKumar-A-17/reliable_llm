import json
import re
import time
from dataclasses import dataclass

from src.models.groq_generator import GroqGenerator
from configs.config import GROQ_API_KEY, REFLECTION_MODEL


@dataclass
class ReasonCheckResult:
    """
    Output of the Reason Checker Agent.
    """

    question: str
    answer: str
    reason_score: float
    logical_contradictions: list
    unsupported_conclusions: list
    circular_reasoning: list
    missing_justification: list
    summary: str
    reasoning_time: float


class ReasonCheckerAgent:

    def __init__(self):
        self.generator = GroqGenerator(
            model_name=REFLECTION_MODEL,
            api_key=GROQ_API_KEY
        )

    def _build_prompt(self, question, answer):
        return f"""
You are an expert reasoning auditor.

Your task is to analyze the reasoning quality of the provided answer.

Question:
{question}

Answer:
{answer}

Look for:
1. Logical contradictions.
2. Unsupported conclusions.
3. Circular reasoning.
4. Missing justification.

Return ONLY valid JSON.

{{
    "reason_score": 0.72,
    "logical_contradictions": ["contradiction 1"],
    "unsupported_conclusions": ["unsupported conclusion 1"],
    "circular_reasoning": ["circular argument 1"],
    "missing_justification": ["missing evidence 1"],
    "summary": "A concise reasoning analysis."
}}
"""

    def _extract_json(self, text):
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return match.group()
        return None

    def _heuristic_reason_score(self, answer, logical_contradictions, unsupported_conclusions,
                                circular_reasoning, missing_justification):
        score = 1.0
        score -= len(logical_contradictions) * 0.25
        score -= len(unsupported_conclusions) * 0.20
        score -= len(circular_reasoning) * 0.15
        score -= len(missing_justification) * 0.10
        return max(0.0, round(score, 4))

    def run(self, question, answer):
        print("[Reason Checker] Processing...")
        start = time.time()
        prompt = self._build_prompt(question, answer)
        response = self.generator.generate(prompt)
        elapsed = round(time.time() - start, 2)

        try:
            json_text = self._extract_json(response)
            data = json.loads(json_text)
        except Exception:
            data = {
                "reason_score": None,
                "logical_contradictions": [],
                "unsupported_conclusions": [],
                "circular_reasoning": [],
                "missing_justification": [],
                "summary": response
            }

        logical_contradictions = data.get("logical_contradictions", []) or []
        unsupported_conclusions = data.get("unsupported_conclusions", []) or []
        circular_reasoning = data.get("circular_reasoning", []) or []
        missing_justification = data.get("missing_justification", []) or []

        reason_score = data.get("reason_score")
        if reason_score is None:
            reason_score = self._heuristic_reason_score(
                answer,
                logical_contradictions,
                unsupported_conclusions,
                circular_reasoning,
                missing_justification
            )
        else:
            reason_score = max(0.0, min(1.0, float(reason_score)))

        print(f"[Reason Checker] Intermediate Score: {reason_score:.2f}")
        print(f"[Reason Checker] Finished in {elapsed:.2f}s")

        return ReasonCheckResult(
            question=question,
            answer=answer,
            reason_score=round(reason_score, 4),
            logical_contradictions=logical_contradictions,
            unsupported_conclusions=unsupported_conclusions,
            circular_reasoning=circular_reasoning,
            missing_justification=missing_justification,
            summary=data.get("summary", "Reasoning analysis completed."),
            reasoning_time=elapsed
        )
