"""
Quick test of the Reliable LLM Pipeline with hardcoded questions (no dataset download).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.response_generator import ResponseGeneratorAgent
from src.agents.reflection_agent import ReflectionAgent
from src.agents.fact_checker import FactCheckerAgent
from src.agents.consistency_checker import ConsistencyCheckerAgent
from src.agents.reason_checker import ReasonCheckerAgent
from src.agents.confidence_aggregator import ConfidenceAggregatorAgent
from src.agents.response_improver import ResponseImproverAgent
from src.confidence.sampler import ResponseSampler
from src.confidence.similarity import SimilarityCalculator

if __name__ == "__main__":

    print("=" * 80)
    print("TESTING RELIABLE LLM COMPONENTS")
    print("=" * 80)

    question = "Who is the father of computer? "

    # 1. Generate initial response
    print("\n[1] Testing Response Generator...")
    generator = ResponseGeneratorAgent()
    response = generator.run(question)
    print(f"✓ Initial answer: {response.answer[:100]}...")

    # 2. Test sampler
    print("\n[2] Testing Response Sampler...")
    sampler = ResponseSampler(generator.generator)
    responses = sampler.sample(question)
    print(f"✓ Generated {len(responses)} samples")

    # 3. Test similarity
    print("\n[3] Testing Similarity Calculator...")
    similarity_calc = SimilarityCalculator()
    sim_matrix = similarity_calc.calculate(responses)
    avg_sim = similarity_calc.average_similarity(sim_matrix)
    print(f"✓ Average similarity: {avg_sim:.4f}")

    # 4. Test reflection
    print("\n[4] Testing Reflection Agent...")
    reflection = ReflectionAgent()
    reflection_result = reflection.run(question, response.answer)
    print(f"✓ Found {len(reflection_result.mistakes)} mistakes")
    reflection_score = 1.0 - (len(reflection_result.mistakes) * 0.1)
    reflection_score = max(0.0, min(1.0, reflection_score))
    print(f"  Reflection score: {reflection_score:.4f}")

    # 5. Test fact checker
    print("\n[5] Testing Fact Checker Agent...")
    fact_checker = FactCheckerAgent()
    fact_result = fact_checker.run(question, response.answer)
    print(f"✓ Found {len(fact_result.unverified_claims)} unverified claims")
    print(f"  Fact check score: {fact_result.fact_check_score:.4f}")

    # 6. Test consistency checker
    print("\n[6] Testing Consistency Checker Agent...")
    consistency = ConsistencyCheckerAgent()
    consistency_result = consistency.run(question, responses)
    # field names: conflicting_facts, agreement_score
    print(f"✓ Found {len(consistency_result.conflicting_facts)} conflicting facts")
    print(f"  Consistency score (agreement): {consistency_result.agreement_score:.4f}")

    # 7. Test reason checker
    print("\n[7] Testing Reason Checker Agent...")
    reason_checker = ReasonCheckerAgent()
    reason_result = reason_checker.run(question, response.answer)
    print(f"✓ Reason score: {reason_result.reason_score:.4f}")

    # 8. Test confidence aggregator
    print("\n[8] Testing Confidence Aggregator...")
    aggregator = ConfidenceAggregatorAgent()
    agg_result = aggregator.run(
        reflection_score=reflection_score,
        fact_check_score=fact_result.fact_check_score,
        reason_score=reason_result.reason_score,
        consistency_score=consistency_result.agreement_score,
        similarity_score=avg_sim
    )
    print(f"✓ Final confidence: {agg_result.final_confidence:.4f}")
    print(f"  Confidence level: {agg_result.confidence_level}")
    print(f"  Breakdown:")
    print(f"    - Reflection  (25%): {agg_result.weighted_reflection:.4f}")
    print(f"    - Fact Check  (30%): {agg_result.weighted_fact_check:.4f}")
    print(f"    - Reason      (20%): {agg_result.weighted_reason:.4f}")
    print(f"    - Consistency (15%): {agg_result.weighted_consistency:.4f}")
    print(f"    - Similarity  (10%): {agg_result.weighted_similarity:.4f}")
    print(f"  Explanation: {agg_result.confidence_explanation}")

    # 9. Test response improver (only if low confidence)
    if agg_result.final_confidence < 0.70:
        print("\n[9] Testing Response Improver (low confidence — improving)...")
        improver = ResponseImproverAgent()
        improved = improver.run(question, response.answer)
        print(f"✓ Improved answer: {improved.improved_answer[:100]}...")
        print(f"  Improvement: {improved.improvement_summary}")
    else:
        print(f"\n[9] Skipping Response Improver (confidence {agg_result.final_confidence:.2%} >= 70%)")

    print("\n" + "=" * 80)
    print("✓ ALL COMPONENTS TESTED SUCCESSFULLY!")
    print("=" * 80)
