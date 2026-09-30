import time
from dataclasses import dataclass

from src.models.groq_generator import GroqGenerator
from configs.config import GROQ_API_KEY, RESPONSE_IMPROVER_MODEL


@dataclass
class ImprovedResponseResult:
    """
    Output produced by the Response Improver Agent.
    """

    question: str
    original_answer: str
    improved_answer: str
    improvement_summary: str
    improvement_time: float
    model_name: str


class ResponseImproverAgent:
    """
    Refines low-confidence answers using a more powerful model.
    """

    def __init__(self):

        self.generator = GroqGenerator(
            model_name=RESPONSE_IMPROVER_MODEL,
            api_key=GROQ_API_KEY
        )

    def _build_prompt(self, question, original_answer):

        return f"""
You are an expert writer and fact-checker.

Your task is to improve the following answer by making it more accurate, clear, and comprehensive.

Question:
{question}

Original Answer:
{original_answer}

Please:
1. Verify the factual accuracy
2. Identify and correct any errors
3. Add missing important information
4. Improve clarity and structure
5. Ensure the answer is well-reasoned

Provide an improved version of the answer that is more reliable and comprehensive.
"""

    def run(self, question: str, original_answer: str) -> ImprovedResponseResult:

        prompt = self._build_prompt(question, original_answer)

        start = time.time()

        improved_answer = self.generator.generate(prompt, max_tokens=1024)

        elapsed = round(time.time() - start, 2)

        # Generate a brief summary of improvements
        improvement_summary = self._generate_improvement_summary(
            original_answer, 
            improved_answer
        )

        return ImprovedResponseResult(
            question=question,
            original_answer=original_answer,
            improved_answer=improved_answer,
            improvement_summary=improvement_summary,
            improvement_time=elapsed,
            model_name=self.generator.model_name
        )

    def _generate_improvement_summary(self, original, improved):
        """Generate a brief summary of what was improved."""
        
        # Simple heuristic: if improved answer is longer, information was added
        # In a real system, you'd use more sophisticated comparison
        
        original_len = len(original.split())
        improved_len = len(improved.split())
        
        if improved_len > original_len * 1.2:
            summary = f"Added more details and context ({improved_len - original_len} words added)"
        elif improved_len < original_len * 0.8:
            summary = f"Made more concise and focused ({original_len - improved_len} words removed)"
        else:
            summary = "Refined and improved for clarity"
        
        return summary
