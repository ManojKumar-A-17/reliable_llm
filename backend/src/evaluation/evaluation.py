"""
evaluation.py
-------------
Evaluation metrics for the Reliable LLM pipeline.

Metrics:
  Standard NLP
  ─────────────
  • BLEU score (bigram)
  • ROUGE-1 / ROUGE-2 / ROUGE-L  (P / R / F1)
  • Exact Match
  • Token-level Precision / Recall / F1

  Reliability metrics  (college slides 12-14)
  ────────────────────────────────────────────
  • Factual Accuracy   — fraction of answer tokens supported by retrieved context
  • Hallucination Rate — fraction of answer tokens NOT in context (1 − factual_accuracy)
  • Response Consistency — 1 − normalised std-dev of pairwise ROUGE-L F1 across samples
  • ECE               — Expected Calibration Error over a batch of confidence buckets
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class EvaluationMetrics:
    """All per-example metrics."""
    # NLP basics
    bleu_score: float
    rouge_1_f1: float
    rouge_2_f1: float
    rouge_l_f1: float
    exact_match: bool
    # Token-level
    precision: float
    recall: float
    f1: float
    # Reliability
    factual_accuracy: float       # how well answer is grounded in context
    hallucination_rate: float     # tokens not supported by context
    response_consistency: float   # across multiple sampled answers
    # Lengths
    answer_length: int
    prediction_length: int


@dataclass
class BatchMetrics:
    """Aggregate metrics over a batch, including ECE."""
    num_examples: int
    avg_bleu: float
    avg_rouge_1_f1: float
    avg_rouge_2_f1: float
    avg_rouge_l_f1: float
    exact_match_rate: float
    avg_precision: float
    avg_recall: float
    avg_f1: float
    avg_factual_accuracy: float
    avg_hallucination_rate: float
    avg_response_consistency: float
    ece: float                    # Expected Calibration Error (batch-level)
    individual_metrics: List[EvaluationMetrics] = field(default_factory=list)


# ── Scorer ────────────────────────────────────────────────────────────────────

class EvaluationMetrics_Scorer:
    """
    Computes all evaluation metrics.

    Usage — single example:
        scorer = EvaluationMetrics_Scorer()
        m = scorer.evaluate(prediction, ground_truth, context=ctx, samples=[s1,s2,s3])

    Usage — batch (also computes ECE):
        result = scorer.batch_evaluate(preds, refs, contexts, samples_list, confidences)
    """

    # ── helpers ───────────────────────────────────────────────────────────────

    def _normalize(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _tokens(self, text: str) -> List[str]:
        return self._normalize(text).split()

    def _ngrams(self, tokens: List[str], n: int) -> Counter:
        return Counter(" ".join(tokens[i : i + n]) for i in range(max(0, len(tokens) - n + 1)))

    def _lcs_len(self, a: List[str], b: List[str]) -> int:
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i - 1] == b[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[m][n]

    def _prf(self, pred_c: Counter, ref_c: Counter, pred_n: int, ref_n: int) -> tuple:
        overlap = sum((pred_c & ref_c).values())
        p = overlap / pred_n if pred_n else 0.0
        r = overlap / ref_n if ref_n else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        return round(p, 4), round(r, 4), round(f, 4)

    # ── standard NLP ─────────────────────────────────────────────────────────

    def exact_match(self, prediction: str, ground_truth: str) -> bool:
        return self._normalize(prediction) == self._normalize(ground_truth)

    def bleu_score(self, prediction: str, ground_truth: str) -> float:
        pt = self._tokens(prediction)
        rt = self._tokens(ground_truth)
        if not pt or not rt:
            return 0.0
        p1 = sum((self._ngrams(pt, 1) & self._ngrams(rt, 1)).values()) / len(pt)
        bg_pt = self._ngrams(pt, 2)
        bg_rt = self._ngrams(rt, 2)
        p2_denom = sum(bg_pt.values())
        p2 = sum((bg_pt & bg_rt).values()) / p2_denom if p2_denom else 0.0
        return round(min(1.0, (p1 * p2) ** 0.5 if p2 > 0 else p1), 4)

    def rouge_n(self, prediction: str, ground_truth: str, n: int = 1) -> dict:
        pt, rt = self._tokens(prediction), self._tokens(ground_truth)
        if not pt or not rt:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        pc, rc = self._ngrams(pt, n), self._ngrams(rt, n)
        p, r, f = self._prf(pc, rc, sum(pc.values()), sum(rc.values()))
        return {"precision": p, "recall": r, "f1": f}

    def rouge_l(self, prediction: str, ground_truth: str) -> dict:
        pt, rt = self._tokens(prediction), self._tokens(ground_truth)
        if not pt or not rt:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        lcs = self._lcs_len(pt, rt)
        p = lcs / len(pt) if pt else 0.0
        r = lcs / len(rt) if rt else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)}

    def token_prf(self, prediction: str, ground_truth: str) -> tuple:
        """Token-level precision, recall, F1."""
        pt, rt = self._tokens(prediction), self._tokens(ground_truth)
        if not pt or not rt:
            return 0.0, 0.0, 0.0
        pc, rc = Counter(pt), Counter(rt)
        p, r, f = self._prf(pc, rc, len(pt), len(rt))
        return p, r, f

    # ── reliability metrics ───────────────────────────────────────────────────

    def factual_accuracy(self, prediction: str, context: str) -> float:
        """
        Fraction of answer tokens that also appear in the retrieved context.
        Proxy for how well the answer is grounded in evidence.
        Returns 0.5 (neutral) when no context is available.
        """
        if not context or not context.strip():
            return 0.5
        pt = self._tokens(prediction)
        ct = set(self._tokens(context))
        if not pt:
            return 0.0
        supported = sum(1 for t in pt if t in ct)
        return round(supported / len(pt), 4)

    def hallucination_rate(self, prediction: str, context: str) -> float:
        """
        Fraction of answer tokens NOT supported by the context.
        1 − factual_accuracy.  Neutral (0.5) when no context.
        """
        return round(1.0 - self.factual_accuracy(prediction, context), 4)

    def response_consistency(self, samples: List[str]) -> float:
        """
        Consistency across multiple sampled responses (e.g. from sampler).
        Computed as 1 − normalised std-dev of pairwise ROUGE-L F1 scores.
        Returns 1.0 (perfectly consistent) when < 2 samples are given.
        """
        if len(samples) < 2:
            return 1.0

        scores: List[float] = []
        for i in range(len(samples)):
            for j in range(i + 1, len(samples)):
                scores.append(self.rouge_l(samples[i], samples[j])["f1"])

        mean = sum(scores) / len(scores)
        variance = sum((s - mean) ** 2 for s in scores) / len(scores)
        std = math.sqrt(variance)
        # Normalise std to [0,1] and invert so higher = more consistent
        consistency = max(0.0, 1.0 - std * 2)
        return round(consistency, 4)

    # ── ECE (batch-level) ─────────────────────────────────────────────────────

    @staticmethod
    def expected_calibration_error(
        confidences: List[float],
        accuracies: List[float],
        num_bins: int = 10,
    ) -> float:
        """
        Expected Calibration Error.

        confidences : model confidence score per example (0-1)
        accuracies  : binary correct/incorrect per example (0 or 1)
        num_bins    : number of equal-width calibration buckets

        Lower ECE = better calibrated model.
        """
        if not confidences or len(confidences) != len(accuracies):
            return 0.0

        bin_size = 1.0 / num_bins
        ece = 0.0
        n = len(confidences)

        for b in range(num_bins):
            lo, hi = b * bin_size, (b + 1) * bin_size
            bucket = [
                (c, a)
                for c, a in zip(confidences, accuracies)
                if lo <= c < hi
            ]
            if not bucket:
                continue
            avg_conf = sum(c for c, _ in bucket) / len(bucket)
            avg_acc  = sum(a for _, a in bucket) / len(bucket)
            ece += (len(bucket) / n) * abs(avg_conf - avg_acc)

        return round(ece, 4)

    # ── public evaluate ───────────────────────────────────────────────────────

    def evaluate(
        self,
        prediction: str,
        ground_truth: str,
        context: str = "",
        samples: Optional[List[str]] = None,
    ) -> EvaluationMetrics:
        """
        Compute all per-example metrics.

        prediction   : the model's final answer
        ground_truth : the reference answer
        context      : retrieved RAG context (for factual accuracy)
        samples      : list of sampled responses (for consistency)
        """
        samples = samples or []

        bleu   = self.bleu_score(prediction, ground_truth)
        r1     = self.rouge_n(prediction, ground_truth, n=1)
        r2     = self.rouge_n(prediction, ground_truth, n=2)
        rl     = self.rouge_l(prediction, ground_truth)
        em     = self.exact_match(prediction, ground_truth)
        p, r, f = self.token_prf(prediction, ground_truth)
        fa     = self.factual_accuracy(prediction, context)
        hr     = self.hallucination_rate(prediction, context)
        rc     = self.response_consistency(samples)

        return EvaluationMetrics(
            bleu_score=bleu,
            rouge_1_f1=r1["f1"],
            rouge_2_f1=r2["f1"],
            rouge_l_f1=rl["f1"],
            exact_match=em,
            precision=p,
            recall=r,
            f1=f,
            factual_accuracy=fa,
            hallucination_rate=hr,
            response_consistency=rc,
            answer_length=len(ground_truth.split()),
            prediction_length=len(prediction.split()),
        )

    def batch_evaluate(
        self,
        predictions: List[str],
        ground_truths: List[str],
        contexts: Optional[List[str]] = None,
        samples_list: Optional[List[List[str]]] = None,
        confidences: Optional[List[float]] = None,
    ) -> BatchMetrics:
        """
        Evaluate a batch of predictions. Also computes ECE.

        confidences : pipeline confidence score per example (for ECE)
        """
        if len(predictions) != len(ground_truths):
            raise ValueError("predictions and ground_truths must be the same length")

        n = len(predictions)
        contexts     = contexts     or [""] * n
        samples_list = samples_list or [[]] * n
        confidences  = confidences  or []

        metrics_list: List[EvaluationMetrics] = [
            self.evaluate(pred, ref, ctx, samps)
            for pred, ref, ctx, samps in zip(predictions, ground_truths, contexts, samples_list)
        ]

        def avg(attr: str) -> float:
            vals = [getattr(m, attr) for m in metrics_list]
            return round(sum(vals) / len(vals), 4)

        # Binary accuracy = ROUGE-L F1 > 0.5 threshold
        accuracies = [1.0 if m.rouge_l_f1 > 0.5 else 0.0 for m in metrics_list]
        ece = self.expected_calibration_error(confidences, accuracies) if confidences else 0.0

        return BatchMetrics(
            num_examples=n,
            avg_bleu=avg("bleu_score"),
            avg_rouge_1_f1=avg("rouge_1_f1"),
            avg_rouge_2_f1=avg("rouge_2_f1"),
            avg_rouge_l_f1=avg("rouge_l_f1"),
            exact_match_rate=round(sum(m.exact_match for m in metrics_list) / n, 4),
            avg_precision=avg("precision"),
            avg_recall=avg("recall"),
            avg_f1=avg("f1"),
            avg_factual_accuracy=avg("factual_accuracy"),
            avg_hallucination_rate=avg("hallucination_rate"),
            avg_response_consistency=avg("response_consistency"),
            ece=ece,
            individual_metrics=metrics_list,
        )
