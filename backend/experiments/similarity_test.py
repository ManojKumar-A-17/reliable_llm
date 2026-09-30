from src.models.generator import QwenGenerator
from src.confidence.sampler import ResponseSampler
from src.confidence.similarity import SimilarityCalculator

generator = QwenGenerator()

sampler = ResponseSampler(generator)

similarity = SimilarityCalculator()

question = "Who is the father of computer? "

responses = sampler.sample(question)

print("\nResponses\n")

for i, response in enumerate(responses):
    print(f"{i+1}. {response}\n")

matrix = similarity.calculate(responses)

print("=" * 80)
print("Similarity Matrix")
print("=" * 80)

print(matrix)

avg = similarity.average_similarity(matrix)

print("\nAverage Similarity:", avg)