from src.agents.response_generator import ResponseGeneratorAgent
from src.agents.reflection_agent import ReflectionAgent


question = "Who is the father of computer? "

generator = ResponseGeneratorAgent()

reflection = ReflectionAgent()

response = generator.run(question)

review = reflection.run(
    question,
    response.answer
)

print("=" * 80)
print("QUESTION")
print("=" * 80)

print(review.question)

print("\n")

print("=" * 80)
print("ORIGINAL ANSWER")
print("=" * 80)

print(review.original_answer)

print("\n")

print("=" * 80)
print("MISTAKES")
print("=" * 80)

for m in review.mistakes:
    print("-", m)

print("\n")

print("=" * 80)
print("REASONING")
print("=" * 80)

print(review.reasoning)

print("\n")

print("=" * 80)
print("IMPROVED ANSWER")
print("=" * 80)

print(review.improved_answer)

print("\n")

print("=" * 80)
print("REFLECTION TIME")
print("=" * 80)

print(review.reflection_time, "sec")