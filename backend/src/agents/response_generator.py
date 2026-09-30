import time
from dataclasses import dataclass

from src.models.groq_generator import GroqGenerator
from configs.config import GROQ_API_KEY, RESPONSE_GENERATOR_MODEL


@dataclass
class ResponseResult:
    """
    Output produced by the Response Generator Agent.
    """

    question: str
    answer: str
    generation_time: float
    model_name: str


class ResponseGeneratorAgent:

    def __init__(self):

        self.generator = GroqGenerator(
            model_name=RESPONSE_GENERATOR_MODEL,
            api_key=GROQ_API_KEY
        )

    def run(self, question: str) -> ResponseResult:

        start = time.time()

        answer = self.generator.generate(question)

        elapsed = round(time.time() - start, 2)

        return ResponseResult(
            question=question,
            answer=answer,
            generation_time=elapsed,
            model_name=self.generator.model_name
        )