"""
confidence_calibrator.py
------------------------
Combines evidence verification, consistency sampling, and reflection
into a single calibrated confidence score with a human-readable reason.

Score interpretation
--------------------
  >= 0.80  HIGH    — well-supported by evidence
  0.60-0.79 MEDIUM — partially supported, some uncertainty
  < 0.60   LOW     — weak or no evidence support; treat as uncertain
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from configs.settings import settings


@dataclass
class ConfidenceResult:
    score:       float   # 0.0 – 1.0
    level:       str     # "HIGH" | "MEDIUM" | "LOW"
    explanation: str     # one-sentence human reason
    breakdown:   dict    # component scores for transparency


def _level(score: float) -> str:
    if score >= 0.80:
        return "HIGH"
    if score >= 0.60:
        return "MEDIUM"
    return "LOW"


def _reason(score: float, evidence: float, consistency: float, reflection: float) -> str:
    level = _level(score)
    if level == "HIGH":
        return (
            f"The answer is well-supported by retrieved evidence "
            f"(evidence score {evidence:.0%}) and is consistent across "
            f"multiple samples."
        )
    if level == "MEDIUM":
        if evidence < 0.55:
            return (
                f"The answer is partially or weakly supported by the available "
                f"evidence (evidence score {evidence:.0%}). Some claims could not "
                "be verified against the knowledge base."
            )
        return (
            f"Evidence provides moderate support (evidence score {evidence:.0%}), "
            "but the answer shows some inconsistency across samples or limited "
            "self-reflection quality."
        )
    return (
        f"Evidence support is low (evidence score {evidence:.0%}). "
        "The answer may draw on the model's parametric knowledge rather than "
        "verified documents, or the sources may not address the question."
    )


class ConfidenceCalibrator:
    """
    Aggregates sub-scores into a final confidence score.

    Parameters passed via run():
        evidence_score   : from EvidenceVerifier  (0-1)
        consistency_score: from similarity/sampler (0-1); pass None to skip
        reflection_score : from ReflectionAgent   (0-1); pass None to skip
    """

    def run(
        self,
        evidence_score: float,
        consistency_score: Optional[float] = None,
        reflection_score: Optional[float] = None,
    ) -> ConfidenceResult:

        # Determine which components are available and renormalise weights
        weights: dict = {}
        scores:  dict = {}

        # Evidence is always present
        w_ev = settings.EVIDENCE_WEIGHT
        weights["evidence"] = w_ev
        scores["evidence"]  = max(0.0, min(1.0, evidence_score))

        if consistency_score is not None:
            w_con = settings.CONSISTENCY_WEIGHT
            weights["consistency"] = w_con
            scores["consistency"]  = max(0.0, min(1.0, consistency_score))

        if reflection_score is not None:
            w_ref = settings.REFLECTION_WEIGHT
            weights["reflection"] = w_ref
            scores["reflection"]  = max(0.0, min(1.0, reflection_score))

        total_weight = sum(weights.values())

        final = sum(
            scores[k] * weights[k] / total_weight
            for k in scores
        )
        final = round(max(0.0, min(1.0, final)), 4)

        ev   = scores.get("evidence", 0.5)
        con  = scores.get("consistency", 0.5)
        ref  = scores.get("reflection", 0.5)

        return ConfidenceResult(
            score=final,
            level=_level(final),
            explanation=_reason(final, ev, con, ref),
            breakdown={
                "evidence_score":     round(ev, 3),
                "consistency_score":  round(con, 3) if "consistency" in scores else None,
                "reflection_score":   round(ref, 3) if "reflection" in scores else None,
                "weights":            {k: round(v / total_weight, 3) for k, v in weights.items()},
            },
        )
