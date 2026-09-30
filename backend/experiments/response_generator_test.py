from src.agents.response_generator import ResponseGeneratorAgent

agent = ResponseGeneratorAgent()

result = agent.run(
    "Who invented the telephone?"
)

print("=" * 80)

print("Question")
print(result.question)

print()

print("Answer")
print(result.answer)

print()

print("Generation Time")
print(result.generation_time)

print()

print("Model")
print(result.model_name)