import json
import re
import time
from dataclasses import dataclass

from src.models.groq_generator import GroqGenerator
from configs.config import GROQ_API_KEY, CONSISTENCY_MODEL


@dataclass
class ConsistencyCheckResult:
    """
    Output of the Consistency Checker Agent.
    """

    question: str
    responses: list
    agreement_score: float
    conflicting_entities: list
    conflicting_facts: list
    summary: str
    consistency_time: float


class ConsistencyCheckerAgent:

    def __init__(self):
        self.generator = GroqGenerator(
            model_name=CONSISTENCY_MODEL,
            api_key=GROQ_API_KEY
        )

    def _build_prompt(self, question, responses):
        responses_text = "\n".join([f"Response {i+1}: {resp}" for i, resp in enumerate(responses)])
        return f"""
You are an expert consistency analyzer.

Your task is to analyze multiple responses and determine their semantic consistency.

Question:
{question}

Responses:
{responses_text}

Perform the following:

1. Compare the responses at the semantic level.
2. Detect conflicting entities, dates, numbers, names, and locations.
3. Provide an agreement score from 0-1.
4. Return structured findings.

Return ONLY valid JSON.

{{
    "agreement_score": 0.2,
    "conflicting_entities": ["capital city"],
    "conflicting_facts": ["Paris vs Lyon"],
    "summary": "Responses disagree on the capital city."
}}
"""

    def _extract_json(self, text):
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return match.group()
        return None

    def run(self, question, responses):
        print("[Consistency Checker] Processing...")
        prompt = self._build_prompt(question, responses)
        start = time.time()
        response = self.generator.generate(prompt)
        elapsed = round(time.time() - start, 2)

        try:
            json_text = self._extract_json(response)
            data = json.loads(json_text)
        except Exception:
            data = {
                "agreement_score": 0.5,
                "conflicting_entities": [],
                "conflicting_facts": [],
                "summary": response
            }

        agreement_score = data.get("agreement_score")
        if agreement_score is None:
            agreement_score = 0.5
        else:
            agreement_score = max(0.0, min(1.0, float(agreement_score)))

        conflicting_entities = data.get("conflicting_entities", []) or []
        conflicting_facts = data.get("conflicting_facts", []) or []

        print(f"[Consistency Checker] Intermediate Score: {agreement_score:.2f}")
        print(f"[Consistency Checker] Finished in {elapsed:.2f}s")

        return ConsistencyCheckResult(
            question=question,
            responses=responses,
            agreement_score=round(agreement_score, 4),
            conflicting_entities=conflicting_entities,
            conflicting_facts=conflicting_facts,
            summary=data.get("summary", "Consistency analysis completed."),
            consistency_time=elapsed
        )
