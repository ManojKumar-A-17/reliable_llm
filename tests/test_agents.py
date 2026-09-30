import unittest
from unittest.mock import patch

from src.agents.fact_checker import FactCheckerAgent
from src.agents.reflection_agent import ReflectionAgent
from src.agents.consistency_checker import ConsistencyCheckerAgent
from src.agents.reason_checker import ReasonCheckerAgent
from src.agents.confidence_aggregator import ConfidenceAggregatorAgent
from src.agents.retrieval_agent import RetrievalAgent


class AgentScoringTests(unittest.TestCase):
    def test_fact_checker_splits_claims_and_averages_scores(self):
        agent = FactCheckerAgent()
        answer = (
            "Alexander Graham Bell invented the telephone. "
            "He patented it in 1876. "
            "He worked for Bell Telephone Company."
        )

        with patch.object(agent.generator, "generate", return_value='{"claims": ["Alexander Graham Bell invented the telephone.", "He patented it in 1876.", "He worked for Bell Telephone Company."], "claim_results": [{"claim": "Alexander Graham Bell invented the telephone.", "status": "Supported", "confidence": 0.9}, {"claim": "He patented it in 1876.", "status": "Partially Supported", "confidence": 0.6}, {"claim": "He worked for Bell Telephone Company.", "status": "Cannot Verify", "confidence": 0.3}], "fact_check_score": 0.6}'):
            result = agent.run("Who invented the telephone?", answer)

        self.assertEqual(len(result.claims), 3)
        self.assertLess(result.fact_check_score, 1.0)
        self.assertEqual(result.claims[0], "Alexander Graham Bell invented the telephone.")

    def test_reflection_agent_returns_measurable_score(self):
        agent = ReflectionAgent()
        answer = "The Eiffel Tower was built in 1900 and is in Rome."

        with patch.object(agent.generator, "generate", return_value='{"mistakes": ["factual mistake", "unsupported claim"], "unsupported_claims": ["unsupported claim"], "ambiguities": ["ambiguous wording"], "contradictions": ["conflict"], "missing_information": ["missing context"], "reasoning": "The answer contains several issues.", "improved_answer": "The Eiffel Tower is in Paris and was completed in 1889.", "reflection_score": 0.25}'):
            result = agent.run("Where is the Eiffel Tower?", answer)

        self.assertEqual(result.reflection_score, 0.25)
        self.assertGreaterEqual(len(result.mistakes), 1)

    def test_consistency_checker_detects_conflicting_facts(self):
        agent = ConsistencyCheckerAgent()
        responses = [
            "The capital of France is Paris.",
            "The capital of France is Lyon."
        ]

        with patch.object(agent.generator, "generate", return_value='{"agreement_score": 0.2, "conflicting_entities": ["capital city"], "conflicting_facts": ["Paris vs Lyon"], "summary": "Responses disagree on the capital city."}'):
            result = agent.run("What is the capital of France?", responses)

        self.assertLess(result.agreement_score, 1.0)
        self.assertGreater(len(result.conflicting_facts), 0)

    def test_reason_checker_scores_reasoning_problems(self):
        agent = ReasonCheckerAgent()
        answer = "The theory is true because it is true."

        with patch.object(agent.generator, "generate", return_value='{"reason_score": 0.35, "logical_contradictions": ["circular reasoning"], "unsupported_conclusions": ["unsupported conclusion"], "circular_reasoning": ["The claim is true because it is true"], "missing_justification": ["No evidence provided"], "summary": "The reasoning is circular."}'):
            result = agent.run("Explain why.", answer)

        self.assertLess(result.reason_score, 1.0)
        self.assertGreater(len(result.circular_reasoning), 0)

    def test_confidence_aggregator_returns_explanation(self):
        agent = ConfidenceAggregatorAgent()
        result = agent.run(
            reflection_score=0.8,
            fact_check_score=0.9,
            reason_score=0.7,
            consistency_score=0.85,
            similarity_score=0.95
        )

        self.assertIn("confidence_explanation", result.__dict__)
        self.assertIn(result.confidence_level, ["Very High", "High", "Medium", "Low", "Very Low"])

    def test_retrieval_agent_returns_grounded_context(self):
        agent = RetrievalAgent(knowledge_base_path="data/knowledge_base.txt")
        result = agent.run("Who invented the telephone?")

        self.assertGreater(len(result.documents), 0)
        self.assertIn("telephone", result.context_text.lower())


if __name__ == "__main__":
    unittest.main()
