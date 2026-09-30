from dataclasses import dataclass


@dataclass
class ConfidenceResult:
    """
    Stores the confidence estimation result.
    """

    similarity_score: float
    confidence_score: float
    confidence_level: str


class ConfidenceScorer:

    def __init__(self):
        """
        Initializes the confidence scorer.
        """
        pass

    def score(self, average_similarity: float) -> ConfidenceResult:
        """
        Converts average semantic similarity into a confidence score.

        Parameters
        ----------
        average_similarity : float
            Average cosine similarity between generated responses.

        Returns
        -------
        ConfidenceResult
        """

        # Ensure similarity stays between 0 and 1
        similarity = max(0.0, min(1.0, average_similarity))

        confidence = round(similarity * 100, 2)

        if confidence >= 90:
            level = "Very High"

        elif confidence >= 80:
            level = "High"

        elif confidence >= 70:
            level = "Moderate"

        elif confidence >= 50:
            level = "Low"

        else:
            level = "Very Low"

        return ConfidenceResult(
            similarity_score=round(similarity, 4),
            confidence_score=confidence,
            confidence_level=level
        )