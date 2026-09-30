from src.models.generator import QwenGenerator
from src.confidence.sampler import ResponseSampler
from src.confidence.similarity import SimilarityCalculator
from src.confidence.scorer import ConfidenceScorer

generator = QwenGenerator()

sampler = ResponseSampler(generator)

similarity = SimilarityCalculator()

scorer = ConfidenceScorer()

question = "Who is the father of computer? "

responses = sampler.sample(question)

matrix = similarity.calculate(responses)

avg_similarity = similarity.average_similarity(matrix)

result = scorer.score(avg_similarity)

print("\n" + "=" * 70)
print("Confidence Estimation")
print("=" * 70)

print(f"Average Similarity : {result.similarity_score}")
print(f"Confidence Score   : {result.confidence_score}%")
print(f"Confidence Level   : {result.confidence_level}")