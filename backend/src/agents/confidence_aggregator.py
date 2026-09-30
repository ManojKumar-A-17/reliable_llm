from dataclasses import dataclass

from configs.config import (
    REFLECTION_WEIGHT,
    FACT_CHECK_WEIGHT,
    REASON_WEIGHT,
    CONSISTENCY_WEIGHT,
    SIMILARITY_WEIGHT,
)


@dataclass
class AggregatedConfidenceResult:
    """
    Output of the Confidence Aggregator Agent.
    """

    reflection_score: float
    fact_check_score: float
    reason_score: float
    consistency_score: float
    similarity_score: float

    final_confidence: float
    confidence_level: str
    confidence_explanation: str

    weighted_reflection: float
    weighted_fact_check: float
    weighted_reason: float
    weighted_consistency: float
    weighted_similarity: float
    weight_distribution: dict


class ConfidenceAggregatorAgent:
    """
    Evidence-based confidence aggregation combining multiple verification signals.
    """

    def __init__(self,
                 reflection_weight=None,
                 fact_check_weight=None,
                 reason_weight=None,
                 consistency_weight=None,
                 similarity_weight=None):

        self.reflection_weight = REFLECTION_WEIGHT if reflection_weight is None else reflection_weight
        self.fact_check_weight = FACT_CHECK_WEIGHT if fact_check_weight is None else fact_check_weight
        self.reason_weight = REASON_WEIGHT if reason_weight is None else reason_weight
        self.consistency_weight = CONSISTENCY_WEIGHT if consistency_weight is None else consistency_weight
        self.similarity_weight = SIMILARITY_WEIGHT if similarity_weight is None else similarity_weight

        total_weight = (
            self.reflection_weight + self.fact_check_weight + self.reason_weight +
            self.consistency_weight + self.similarity_weight
        )
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")

    def _normalize_score(self, score):
        return max(0.0, min(1.0, float(score)))

    def _get_confidence_level(self, confidence_score):
        if confidence_score >= 0.90:
            return "Very High"
        elif confidence_score >= 0.80:
            return "High"
        elif confidence_score >= 0.70:
            return "Medium"
        elif confidence_score >= 0.50:
            return "Low"
        return "Very Low"

    def _build_explanation(self, reflection_score, fact_check_score, reason_score, consistency_score, similarity_score):
        reasons = []
        if consistency_score >= 0.8:
            reasons.append("responses were highly consistent")
        if fact_check_score >= 0.8:
            reasons.append("no unsupported factual claims were detected")
        if reason_score >= 0.7:
            reasons.append("reasoning was coherent")
        if reflection_score >= 0.7 and similarity_score >= 0.7:
            reasons.append("only minor ambiguities were detected")
        if not reasons:
            reasons = ["the evidence was mixed and should be treated cautiously"]
        return "High confidence because " + "; ".join(reasons) + "."

    def run(self,
            reflection_score: float,
            fact_check_score: float,
            reason_score: float,
            consistency_score: float,
            similarity_score: float) -> AggregatedConfidenceResult:

        refl = self._normalize_score(reflection_score)
        fact = self._normalize_score(fact_check_score)
        reason = self._normalize_score(reason_score)
        cons = self._normalize_score(consistency_score)
        sim = self._normalize_score(similarity_score)

        weighted_refl = refl * self.reflection_weight
        weighted_fact = fact * self.fact_check_weight
        weighted_reason = reason * self.reason_weight
        weighted_cons = cons * self.consistency_weight
        weighted_sim = sim * self.similarity_weight

        final_confidence = (
            weighted_refl + weighted_fact + weighted_reason + weighted_cons + weighted_sim
        )

        confidence_level = self._get_confidence_level(final_confidence)
        confidence_explanation = self._build_explanation(
            refl, fact, reason, cons, sim
        )
        weight_distribution = {
            "reflection": self.reflection_weight,
            "fact_check": self.fact_check_weight,
            "reason": self.reason_weight,
            "consistency": self.consistency_weight,
            "similarity": self.similarity_weight,
        }

        return AggregatedConfidenceResult(
            reflection_score=round(refl, 4),
            fact_check_score=round(fact, 4),
            reason_score=round(reason, 4),
            consistency_score=round(cons, 4),
            similarity_score=round(sim, 4),
            final_confidence=round(final_confidence, 4),
            confidence_level=confidence_level,
            confidence_explanation=confidence_explanation,
            weighted_reflection=round(weighted_refl, 4),
            weighted_fact_check=round(weighted_fact, 4),
            weighted_reason=round(weighted_reason, 4),
            weighted_consistency=round(weighted_cons, 4),
            weighted_similarity=round(weighted_sim, 4),
            weight_distribution=weight_distribution,
        )
