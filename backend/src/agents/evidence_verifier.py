"""
evidence_verifier.py
--------------------
Verifies atomic claims from the answer against retrieved evidence chunks.
Now uses Gemini instead of Groq — interface unchanged.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import List, Optional

from src.models.gemini_client import GeminiClient
from src.knowledge.knowledge_base import SearchResult
from configs.settings import settings


@dataclass
class ClaimVerdict:
    claim:      str
    status:     str    # SUPPORTED | PARTIAL | UNSUPPORTED | UNKNOWN
    confidence: float
    evidence:   str
    source_ref: str


@dataclass
class VerificationResult:
    question:         str
    answer:           str
    verdicts:         List[ClaimVerdict]
    evidence_score:   float
    verified_count:   int
    unverified_count: int
    partial_count:    int
    verify_time:      float


_SYSTEM = (
    "You are a meticulous fact-verifier. "
    "Given a claim and evidence passages, classify the claim precisely. "
    "Return ONLY valid JSON — no other text."
)

_CLAIM_TEMPLATE = """Evidence passages:
{evidence}

Claim to verify: "{claim}"

Classify this claim against the evidence. Return JSON:
{{
  "status": "SUPPORTED" | "PARTIAL" | "UNSUPPORTED" | "UNKNOWN",
  "confidence": <float 0.0-1.0>,
  "evidence_excerpt": "<direct quote or paraphrase from evidence, max 80 words>",
  "reasoning": "<one sentence>"
}}

Rules:
- SUPPORTED: evidence clearly backs the claim
- PARTIAL: evidence partially supports but with caveats
- UNSUPPORTED: evidence contradicts or denies the claim
- UNKNOWN: evidence is silent on the claim
"""


def _extract_json(text: str) -> Optional[dict]:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return None


def _split_claims(answer: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
    claims = []
    for s in sentences:
        s_clean = re.sub(r"\[Source \d+\]", "", s).strip()
        if len(s_clean.split()) >= 4:
            claims.append(s_clean)
    return claims[:6]


class EvidenceVerifier:

    def __init__(self):
        self.llm = GeminiClient(
            model_name=settings.VERIFIER_MODEL,
            api_key=settings.require_gemini_key(),
        )

    def _build_evidence_context(self, sources: List[SearchResult], max_chars: int = 1500) -> str:
        blocks, total = [], 0
        for i, sr in enumerate(sources, start=1):
            block = sr.chunk.as_context_block(i)
            if total + len(block) > max_chars:
                break
            blocks.append(block)
            total += len(block)
        return "\n\n".join(blocks)

    def _verify_claim(self, claim: str, evidence_context: str, fallback_source: str = "") -> ClaimVerdict:
        prompt = _CLAIM_TEMPLATE.format(evidence=evidence_context, claim=claim)
        raw = self.llm.generate(prompt=prompt, system=_SYSTEM, temperature=0.1, max_tokens=300)

        data = _extract_json(raw) or {}
        status = data.get("status", "UNKNOWN")
        if status not in ("SUPPORTED", "PARTIAL", "UNSUPPORTED", "UNKNOWN"):
            status = "UNKNOWN"

        return ClaimVerdict(
            claim=claim,
            status=status,
            confidence=float(data.get("confidence", 0.5)),
            evidence=data.get("evidence_excerpt", ""),
            source_ref=fallback_source,
        )

    def run(self, question: str, answer: str, sources: List[SearchResult]) -> VerificationResult:
        t0 = time.time()

        if not sources:
            return VerificationResult(
                question=question, answer=answer, verdicts=[],
                evidence_score=0.5, verified_count=0, unverified_count=0,
                partial_count=0, verify_time=0.0,
            )

        claims = _split_claims(answer)
        if not claims:
            return VerificationResult(
                question=question, answer=answer, verdicts=[],
                evidence_score=0.5, verified_count=0, unverified_count=0,
                partial_count=0, verify_time=round(time.time() - t0, 2),
            )

        evidence_ctx = self._build_evidence_context(sources)
        fallback_ref = sources[0].chunk.citation_label() if sources else ""

        verdicts = [self._verify_claim(c, evidence_ctx, fallback_ref) for c in claims]

        score_map = {"SUPPORTED": 1.0, "PARTIAL": 0.6, "UNSUPPORTED": 0.1, "UNKNOWN": 0.4}
        scores = [score_map.get(v.status, 0.4) * v.confidence for v in verdicts]
        evidence_score = round(sum(scores) / len(scores), 4) if scores else 0.5

        return VerificationResult(
            question=question, answer=answer, verdicts=verdicts,
            evidence_score=evidence_score,
            verified_count=sum(1 for v in verdicts if v.status == "SUPPORTED"),
            unverified_count=sum(1 for v in verdicts if v.status in ("UNSUPPORTED", "UNKNOWN")),
            partial_count=sum(1 for v in verdicts if v.status == "PARTIAL"),
            verify_time=round(time.time() - t0, 2),
        )
