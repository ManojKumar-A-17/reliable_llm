from src.models.generator import QwenGenerator
from src.confidence.sampler import ResponseSampler

generator = QwenGenerator()

sampler = ResponseSampler(generator)

question = "Who is the father of computer? "

responses = sampler.sample(question)

print("\n" + "="*80)

for i, response in enumerate(responses):

    print(f"\nResponse {i+1}")

    print(response)