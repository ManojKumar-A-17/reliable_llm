"""
Confidence calibration.

Six signals, each in [0, 1], are combined with configurable weights that are
renormalised over whichever signals are available (fast mode, or a stage
that failed, simply drops out). Hard caps then encode rules no weighted
average should be able to override — e.g. a contradicted claim.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import List, Optional

from .stages import AnswerResult, ConsistencyResult, ReflectionResult, VerificationResult

COMPONENTS = [
    ("support", "Claim support", "Share of claims the auditor found backed by the sources"),
    ("grounding", "Semantic grounding", "How closely each claim matches a real source sentence (local embeddings)"),
    ("citation", "Citation accuracy", "Claims carry a citation, and it points at a source that backs them"),
    ("consistency", "Self-consistency", "Agreement between the answer and independent re-samples"),
    ("reflection", "Critical review", "Reviewer's score after penalising omissions and overclaims"),
    ("retrieval", "Retrieval relevance", "How relevant the cited passages are to the question"),
]


@dataclass
class Confidence:
    score: float
    level: str                   # HIGH | MEDIUM | LOW | ABSTAINED | NO_EVIDENCE
    headline: str
    explanation: str
    components: List[dict]
    caps: List[str] = field(default_factory=list)
    reliable: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def _level(score: float) -> str:
    return "HIGH" if score >= 0.80 else "MEDIUM" if score >= 0.60 else "LOW"


def retrieval_signal(relevances_all: List[float], cited: List[int]) -> Optional[float]:
    if not relevances_all:
        return None
    if cited:
        vals = [relevances_all[i - 1] for i in cited if 1 <= i <= len(relevances_all)]
    else:
        vals = sorted(relevances_all, reverse=True)[:3]
    return round(sum(vals) / len(vals), 4) if vals else None


def calibrate(settings, answer: Optional[AnswerResult], relevances: List[float],
              verification: Optional[VerificationResult],
              consistency: Optional[ConsistencyResult],
              reflection: Optional[ReflectionResult],
              threshold: float = 0.6) -> Confidence:

    weights = {
        "support": settings.W_SUPPORT, "grounding": settings.W_GROUNDING,
        "citation": settings.W_CITATION, "consistency": settings.W_CONSISTENCY,
        "reflection": settings.W_REFLECTION, "retrieval": settings.W_RETRIEVAL,
    }
    values = {
        "support": verification.support if verification else None,
        "grounding": verification.grounding if verification else None,
        "citation": verification.citation if verification else None,
        "consistency": consistency.score if consistency else None,
        "reflection": reflection.score if reflection else None,
        "retrieval": retrieval_signal(relevances, answer.citations if answer else []),
    }

    if not relevances:
        return Confidence(
            score=0.0, level="NO_EVIDENCE", headline="No supporting evidence",
            explanation="Nothing relevant was retrieved from your documents, so no answer could be grounded.",
            components=_components(values, weights), reliable=False,
        )

    present = {k: v for k, v in values.items() if v is not None and weights[k] > 0}
    total_w = sum(weights[k] for k in present) or 1.0
    score = sum(present[k] * weights[k] for k in present) / total_w

    caps: List[str] = []

    def cap(limit: float, reason: str):
        nonlocal score
        if score > limit:
            score = limit
            caps.append(reason)

    if verification and verification.counts.get("contradicted"):
        cap(0.50, "A claim contradicts the sources")
    if verification and verification.hallucination_rate is not None and verification.hallucination_rate > 0.34:
        cap(0.55, "Over a third of claims are not in the sources")
    if reflection and reflection.answers_question is False:
        cap(0.60, "Reviewer judged that the question was not actually answered")
    if answer and answer.invalid_citations:
        cap(0.70, "Answer cites sources that don't exist")
    if verification and not verification.judge_ok and verification.claims:
        cap(0.75, "Claims verified by local similarity only (auditor model unavailable)")

    score = round(max(0.0, min(1.0, score)), 4)

    if answer and answer.abstained:
        agree = consistency.abstention_agreement if consistency else None
        expl = "The documents don't appear to cover this question, so the system declined to guess."
        if agree is not None:
            expl += f" Re-samples agreed on declining {agree:.0%} of the time."
        return Confidence(score=score, level="ABSTAINED", headline="Declined — not in your documents",
                          explanation=expl, components=_components(values, weights), caps=caps,
                          reliable=True)

    level = _level(score)
    return Confidence(
        score=score, level=level, headline=_headline(level),
        explanation=_explain(values, verification, consistency, reflection, caps),
        components=_components(values, weights), caps=caps, reliable=score >= threshold,
    )


def _components(values: dict, weights: dict) -> List[dict]:
    present_w = sum(weights[k] for k, v in values.items() if v is not None) or 1.0
    out = []
    for key, label, desc in COMPONENTS:
        v = values.get(key)
        out.append({
            "key": key, "label": label, "description": desc,
            "value": None if v is None else round(v, 4),
            "weight": round(weights[key] / present_w, 3) if v is not None else 0.0,
        })
    return out


def _headline(level: str) -> str:
    return {
        "HIGH": "Well supported by your documents",
        "MEDIUM": "Partly supported — review flagged claims",
        "LOW": "Weakly supported — verify before relying on it",
    }[level]


def _explain(values, v: Optional[VerificationResult], c: Optional[ConsistencyResult],
             r: Optional[ReflectionResult], caps: List[str]) -> str:
    parts: List[str] = []
    if v and v.claims:
        n = len(v.claims)
        sup = v.counts.get("supported", 0)
        part = v.counts.get("partial", 0)
        bad = v.counts.get("not_found", 0) + v.counts.get("contradicted", 0)
        s = f"{sup} of {n} claim{'s' if n != 1 else ''} directly supported"
        if part:
            s += f", {part} partly"
        if bad:
            s += f", {bad} not backed by the sources"
        parts.append(s + ".")
    if c and c.score is not None:
        if c.score >= 0.75:
            parts.append(f"Independent re-samples agreed with the answer ({len(c.samples)} checked).")
        elif c.score >= 0.45:
            parts.append("Re-samples partly diverged from the answer.")
        else:
            parts.append("Re-samples gave noticeably different answers — a sign of uncertainty.")
    if r and r.issues:
        high = [i for i in r.issues if i["severity"] == "high"]
        if high:
            parts.append(f"Reviewer flagged: {high[0]['detail'].rstrip('.')}.")
        else:
            parts.append(f"Reviewer noted {len(r.issues)} minor issue{'s' if len(r.issues) != 1 else ''}.")
    if caps:
        parts.append(f"Capped: {caps[0].lower()}.")
    return " ".join(parts) or "Confidence is based on retrieval relevance only."
