import pandas as pd
import time
from typing import Optional, List

from src.data.dataset import QADataset
from src.agents.response_generator import ResponseGeneratorAgent
from src.agents.retrieval_agent import RetrievalAgent
from src.agents.multi_llm_agent import MultiLLMAgent
from src.agents.reflection_agent import ReflectionAgent
from src.agents.fact_checker import FactCheckerAgent
from src.agents.consistency_checker import ConsistencyCheckerAgent
from src.agents.reason_checker import ReasonCheckerAgent
from src.agents.confidence_aggregator import ConfidenceAggregatorAgent
from src.agents.response_improver import ResponseImproverAgent
from src.confidence.sampler import ResponseSampler
from src.confidence.similarity import SimilarityCalculator
from src.evaluation.evaluation import EvaluationMetrics_Scorer

from configs.config import (
    NUM_EXAMPLES, RESULT_PATH, DATASET_NAME, DATASET_CONFIG
)


class ReliableLLMPipeline:
    """
    Complete end-to-end pipeline for reliable LLM response generation.

    Workflow:
    1.  Retrieve grounding context (FAISS RAG)
    2.  Query ALL configured LLMs in parallel (MultiLLMAgent)
    3.  Sample multiple responses for consistency signal
    4.  Calculate semantic similarity across samples
    5.  Reflect on primary answer (identify mistakes)
    6.  Fact-check primary answer
    7.  Check consistency across samples
    8.  Audit reasoning quality
    9.  Aggregate confidence from all signals
    10. Improve answer if confidence < threshold
    11. Return final answer + all model responses + all metrics
    """

    def __init__(self,
                 confidence_threshold: float = 0.70,
                 num_samples: int = 5):

        print("=" * 80)
        print("Initializing Reliable LLM Pipeline")
        print("=" * 80)

        self.confidence_threshold = confidence_threshold
        self.num_samples = num_samples
        self.eval_scorer = EvaluationMetrics_Scorer()

        # ── dataset ──────────────────────────────────────────────────────────
        print("\n[1/10] Loading dataset...")
        self.dataset = QADataset(DATASET_NAME, DATASET_CONFIG)

        # ── agents ───────────────────────────────────────────────────────────
        print("[2/10] Initializing Response Generator...")
        self.generator = ResponseGeneratorAgent()

        print("[3/10] Initializing Retrieval Agent (FAISS)...")
        self.retrieval_agent = RetrievalAgent()

        print("[4/10] Initializing Multi-LLM Agent...")
        self.multi_llm_agent = MultiLLMAgent()

        print("[5/10] Initializing Reflection Agent...")
        self.reflection_agent = ReflectionAgent()

        print("[6/10] Initializing Fact Checker Agent...")
        self.fact_checker = FactCheckerAgent()

        print("[7/10] Initializing Consistency Checker Agent...")
        self.consistency_checker = ConsistencyCheckerAgent()

        print("[8/10] Initializing Reason Checker...")
        self.reason_checker = ReasonCheckerAgent()

        print("[9/10] Initializing Confidence Aggregator...")
        self.confidence_aggregator = ConfidenceAggregatorAgent()

        print("[10/10] Initializing Response Improver...")
        self.response_improver = ResponseImproverAgent()

        print("[+] Initializing Sampler & Similarity Calculator...")
        self.sampler = ResponseSampler(self.generator.generator)
        self.similarity_calculator = SimilarityCalculator()

        print("[✓] Pipeline initialized successfully!\n")

    # ── single question ───────────────────────────────────────────────────────

    def process_question(self, question: str,
                         ground_truth: str = "") -> dict:
        """
        Process a single question through the full pipeline.

        Returns a dict with:
          - initial_answer / final_answer
          - llm_responses  : list of {model, answer, time} from all LLMs
          - confidence metrics
          - evaluation metrics (when ground_truth is provided)
          - timing breakdown
        """
        print(f"\nProcessing: {question}\n")
        start_total = time.time()

        # ── Step 1: RAG retrieval ─────────────────────────────────────────────
        print("  [1] Retrieving supporting context (FAISS)...")
        retrieval_result = self.retrieval_agent.run(question)
        context = retrieval_result.context_text

        # ── Step 2: Multi-LLM parallel query ─────────────────────────────────
        print("  [2] Querying all LLMs in parallel...")
        multi_result = self.multi_llm_agent.run(question, context)

        # Primary model's answer drives the confidence pipeline
        answer = multi_result.best_answer()

        # ── Step 3: Sample multiple responses ────────────────────────────────
        print(f"  [3] Sampling {self.num_samples} responses...")
        sampled_responses: List[str] = self.sampler.sample(question)

        # ── Step 4: Semantic similarity ───────────────────────────────────────
        print("  [4] Computing semantic similarity...")
        similarity_matrix = self.similarity_calculator.calculate(sampled_responses)
        similarity_score  = self.similarity_calculator.average_similarity(similarity_matrix)

        # ── Step 5: Reflect ───────────────────────────────────────────────────
        print("  [5] Reflecting on answer...")
        reflection_result = self.reflection_agent.run(question, answer)
        reflection_score  = max(0.0, min(1.0, 1.0 - len(reflection_result.mistakes) * 0.1))

        # ── Step 6: Fact-check ────────────────────────────────────────────────
        print("  [6] Fact-checking answer...")
        fact_check_result = self.fact_checker.run(question, answer)

        # ── Step 7: Consistency ───────────────────────────────────────────────
        print("  [7] Checking consistency across samples...")
        consistency_result = self.consistency_checker.run(question, sampled_responses)

        # ── Step 8: Reasoning quality ─────────────────────────────────────────
        print("  [8] Auditing reasoning quality...")
        reason_result = self.reason_checker.run(question, answer)

        # ── Step 9: Aggregate confidence ──────────────────────────────────────
        print("  [9] Aggregating confidence scores...")
        aggregated_confidence = self.confidence_aggregator.run(
            reflection_score=reflection_score,
            fact_check_score=fact_check_result.fact_check_score,
            reason_score=reason_result.reason_score,
            consistency_score=consistency_result.agreement_score,
            similarity_score=similarity_score,
        )
        final_confidence = aggregated_confidence.final_confidence

        # ── Step 10: Improve if needed ────────────────────────────────────────
        if final_confidence < self.confidence_threshold:
            print(f"  [10] Confidence {final_confidence:.2%} < threshold — improving answer...")
            improved_result = self.response_improver.run(question, answer)
            final_answer    = improved_result.improved_answer
            improvement_note = f"IMPROVED (was {final_confidence:.2%} confidence)"
        else:
            print(f"  [✓] Confidence {final_confidence:.2%} >= threshold — no improvement needed")
            final_answer    = answer
            improved_result = None
            improvement_note = "No improvement needed"

        elapsed_total = round(time.time() - start_total, 2)

        # ── Evaluation metrics ────────────────────────────────────────────────
        eval_metrics = None
        if ground_truth:
            eval_metrics = self.eval_scorer.evaluate(
                prediction=final_answer,
                ground_truth=ground_truth,
                context=context,
                samples=sampled_responses,
            )

        # ── Compile results ───────────────────────────────────────────────────
        results = {
            "question":           question,
            "initial_answer":     answer,
            "retrieved_context":  context,
            "final_answer":       final_answer,
            "improved":           improved_result is not None,
            "improvement_note":   improvement_note,

            # ── Multi-LLM responses (for display) ──────────────────────────
            "llm_responses":      multi_result.as_dict_list(),
            "multi_llm_time":     multi_result.total_time,

            # ── Confidence ─────────────────────────────────────────────────
            "confidence_score":       round(final_confidence, 4),
            "confidence_level":       aggregated_confidence.confidence_level,
            "confidence_explanation": aggregated_confidence.confidence_explanation,
            "reflection_score":       round(aggregated_confidence.reflection_score, 4),
            "fact_check_score":       round(aggregated_confidence.fact_check_score, 4),
            "reason_score":           round(aggregated_confidence.reason_score, 4),
            "consistency_score":      round(aggregated_confidence.consistency_score, 4),
            "similarity_score":       round(aggregated_confidence.similarity_score, 4),

            # ── Breakdown ──────────────────────────────────────────────────
            "reflection_mistakes":  reflection_result.mistakes,
            "unverified_claims":    fact_check_result.unverified_claims,
            "inconsistencies":      consistency_result.conflicting_facts,
            "reasoning_summary":    reason_result.summary,

            # ── Evaluation (when ground truth available) ───────────────────
            "eval_bleu":                  round(eval_metrics.bleu_score, 4)          if eval_metrics else None,
            "eval_rouge_1":               round(eval_metrics.rouge_1_f1, 4)          if eval_metrics else None,
            "eval_rouge_l":               round(eval_metrics.rouge_l_f1, 4)          if eval_metrics else None,
            "eval_precision":             round(eval_metrics.precision, 4)           if eval_metrics else None,
            "eval_recall":                round(eval_metrics.recall, 4)              if eval_metrics else None,
            "eval_f1":                    round(eval_metrics.f1, 4)                  if eval_metrics else None,
            "eval_factual_accuracy":      round(eval_metrics.factual_accuracy, 4)    if eval_metrics else None,
            "eval_hallucination_rate":    round(eval_metrics.hallucination_rate, 4)  if eval_metrics else None,
            "eval_response_consistency":  round(eval_metrics.response_consistency, 4) if eval_metrics else None,
            "eval_exact_match":           eval_metrics.exact_match                   if eval_metrics else None,

            # ── Timing ─────────────────────────────────────────────────────
            "retrieval_time":     retrieval_result.retrieval_time,
            "initial_gen_time":   multi_result.total_time,
            "reflection_time":    reflection_result.reflection_time,
            "fact_check_time":    fact_check_result.fact_check_time,
            "reasoning_time":     reason_result.reasoning_time,
            "consistency_time":   consistency_result.consistency_time,
            "improvement_time":   improved_result.improvement_time if improved_result else 0,
            "total_time":         elapsed_total,
        }

        return results

    # ── batch run ─────────────────────────────────────────────────────────────

    def run(self, num_examples: Optional[int] = None):
        """
        Run the pipeline on dataset examples and save results CSV.
        Also computes ECE across the batch.
        """
        if num_examples is None:
            num_examples = NUM_EXAMPLES

        print("\n" + "=" * 80)
        print(f"STARTING RELIABLE LLM PIPELINE — {num_examples} examples")
        print("=" * 80)

        validation   = self.dataset.get_validation()
        results      = []
        confidences  = []
        predictions  = []
        ground_truths = []

        num_examples = min(num_examples, len(validation))

        for i in range(num_examples):
            print(f"\n{'='*80}")
            print(f"Question {i+1}/{num_examples}")
            print(f"{'='*80}")

            sample       = self.dataset.preprocess(validation[i])
            question     = sample["question"]
            ground_truth = sample["answer"]

            result = self.process_question(question, ground_truth=ground_truth)
            result["ground_truth"] = ground_truth

            confidences.append(result["confidence_score"])
            predictions.append(result["final_answer"])
            ground_truths.append(ground_truth)

            print(f"\n{'─'*80}")
            print(f"FINAL ANSWER : {result['final_answer'][:100]}...")
            print(f"CONFIDENCE   : {result['confidence_level']} ({result['confidence_score']:.2%})")
            if result["eval_f1"] is not None:
                print(f"EVAL F1      : {result['eval_f1']:.4f}  |  "
                      f"Factual Acc: {result['eval_factual_accuracy']:.4f}  |  "
                      f"Halluc: {result['eval_hallucination_rate']:.4f}")
            print(f"TIME         : {result['total_time']:.2f}s")
            print(f"{'─'*80}")

            results.append(result)

        # ── Batch ECE ─────────────────────────────────────────────────────────
        batch = self.eval_scorer.batch_evaluate(
            predictions=predictions,
            ground_truths=ground_truths,
            confidences=confidences,
        )

        print(f"\n{'='*80}")
        print("BATCH SUMMARY")
        print(f"  ROUGE-L F1         : {batch.avg_rouge_l_f1:.4f}")
        print(f"  F1 (token-level)   : {batch.avg_f1:.4f}")
        print(f"  Factual Accuracy   : {batch.avg_factual_accuracy:.4f}")
        print(f"  Hallucination Rate : {batch.avg_hallucination_rate:.4f}")
        print(f"  Response Consist.  : {batch.avg_response_consistency:.4f}")
        print(f"  ECE                : {batch.ece:.4f}")
        print(f"{'='*80}\n")

        # Save results
        import os
        os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
        df = pd.DataFrame(results)
        # llm_responses is a list — serialise to string for CSV
        df["llm_responses"] = df["llm_responses"].apply(str)
        df.to_csv(RESULT_PATH, index=False, encoding="utf-8")

        print(f"✓ Saved {len(results)} results to {RESULT_PATH}")

        return results, batch
