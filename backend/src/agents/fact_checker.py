import json
import re
import time
from dataclasses import dataclass

from src.models.groq_generator import GroqGenerator
from configs.config import GROQ_API_KEY, FACT_CHECKER_MODEL


@dataclass
class FactCheckResult:
    """
    Output of the Fact Checker Agent.
    """

    question: str
    answer: str
    claims: list
    claim_results: list
    verified_claims: list
    unverified_claims: list
    fact_check_score: float
    fact_check_time: float


class FactCheckerAgent:

    def __init__(self):
        self.generator = GroqGenerator(
            model_name=FACT_CHECKER_MODEL,
            api_key=GROQ_API_KEY
        )

    def _build_prompt(self, question, answer):
        return f"""
You are an expert fact-checker and researcher.

Your task is to analyze the given answer and identify verifiable claims.

Question:
{question}

Answer:
{answer}

Perform the following:

1. Split the answer into individual factual claims.
2. For each claim, classify it as Supported, Partially Supported, Unsupported, or Cannot Verify.
3. Assign each claim a confidence score between 0 and 1.
4. Compute an overall_fact_score as the average of the claim confidence scores.

Return ONLY valid JSON.

{{
    "claims": [
        "claim 1",
        "claim 2"
    ],
    "claim_results": [
        {{"claim": "claim 1", "status": "Supported", "confidence": 0.92}},
        {{"claim": "claim 2", "status": "Partially Supported", "confidence": 0.68}}
    ],
    "verified_claims": ["claim 1"],
    "unverified_claims": ["claim 2"],
    "fact_check_score": 0.8
}}
"""

    def _extract_json(self, text):
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return match.group()
        return None

    def _split_claims(self, answer):
        sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
        claims = [s.strip() for s in sentences if s.strip()]
        if not claims:
            claims = [answer.strip()] if answer.strip() else []
        return claims

    def run(self, question, answer):
        print("[Fact Checker] Processing...")
        prompt = self._build_prompt(question, answer)
        start = time.time()
        response = self.generator.generate(prompt)
        elapsed = round(time.time() - start, 2)

        try:
            json_text = self._extract_json(response)
            data = json.loads(json_text)
        except Exception:
            data = {
                "claims": self._split_claims(answer),
                "claim_results": [],
                "verified_claims": [],
                "unverified_claims": [],
                "fact_check_score": 0.5
            }

        claims = data.get("claims", self._split_claims(answer)) or self._split_claims(answer)
        claim_results = data.get("claim_results", []) or []
        verified_claims = data.get("verified_claims", []) or []
        unverified_claims = data.get("unverified_claims", []) or []

        if not claim_results and claims:
            claim_results = [
                {"claim": claim, "status": "Cannot Verify", "confidence": 0.5}
                for claim in claims
            ]

        if claim_results:
            confidence_scores = [float(item.get("confidence", 0.5)) for item in claim_results if isinstance(item, dict)]
            if confidence_scores:
                fact_check_score = round(sum(confidence_scores) / len(confidence_scores), 4)
            else:
                fact_check_score = 0.5
        else:
            fact_check_score = float(data.get("fact_check_score", 0.5))

        print(f"[Fact Checker] Intermediate Score: {fact_check_score:.2f}")
        print(f"[Fact Checker] Finished in {elapsed:.2f}s")

        return FactCheckResult(
            question=question,
            answer=answer,
            claims=claims,
            claim_results=claim_results,
            verified_claims=verified_claims,
            unverified_claims=unverified_claims,
            fact_check_score=fact_check_score,
            fact_check_time=elapsed
        )
