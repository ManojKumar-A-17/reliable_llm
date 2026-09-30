"""
The evidence-first pipeline orchestrator.

    retrieve ─► answer ─┬─► verify ──────┐
                        ├─► consistency ─┼─► calibrate ─► result
                        └─► reflect ─────┘

Verification, consistency sampling and reflection are independent once the
answer exists, so they run concurrently; the rate limiter paces the actual
API calls. Every step reports progress through an ``emit(event, data)``
callback, which the API layer turns into Server-Sent Events.

LLM budget per question: thorough = 3 + NUM_SAMPLES calls (5 by default,
down from ~10 in v1); fast = 2 calls.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, List, Optional

from .. import telemetry
from ..knowledge.store import Hit, KnowledgeBase
from ..llm import LLMError, LLMRouter
from ..storage.history import HistoryStore
from . import prompts
from .calibrate import calibrate
from .stages import (AnswerResult, ConsistencyResult, ReflectionResult, VerificationResult,
                     check_consistency, generate_answer, reflect, verify_answer)

log = logging.getLogger("rllm.pipeline")

Emit = Callable[[str, dict], None]

STAGES = [
    ("retrieve", "Evidence retrieval", "Hybrid search + rerank over your library"),
    ("answer", "Cited answer", "Answer written only from retrieved passages"),
    ("verify", "Claim audit", "Each claim checked against the sources"),
    ("consistency", "Self-consistency", "Independent re-samples compared"),
    ("reflect", "Critical review", "Reviewer looks for omissions and overclaims"),
    ("calibrate", "Confidence", "Signals combined into a calibrated score"),
]


@dataclass
class QueryOptions:
    mode: str = "thorough"                 # "thorough" | "fast"
    top_k: Optional[int] = None
    doc_ids: Optional[List[str]] = None
    num_samples: Optional[int] = None


@dataclass
class _Timer:
    started: dict = field(default_factory=dict)
    durations: dict = field(default_factory=dict)

    def start(self, k):
        self.started[k] = time.perf_counter()

    def stop(self, k) -> int:
        ms = int((time.perf_counter() - self.started.get(k, time.perf_counter())) * 1000)
        self.durations[k] = ms
        return ms


class Engine:

    def __init__(self, kb: KnowledgeBase, router: LLMRouter, embedder, history: Optional[HistoryStore],
                 settings):
        self.kb = kb
        self.router = router
        self.embedder = embedder
        self.history = history
        self.settings = settings

    async def run(self, question: str, opts: QueryOptions, emit: Emit) -> dict:
        with telemetry.bind(emit):
            return await self._run(question, opts, emit)

    async def _run(self, question: str, opts: QueryOptions, emit: Emit) -> dict:
        s = self.settings
        run_id = uuid.uuid4().hex[:12]
        t_run = time.perf_counter()
        timer = _Timer()
        thorough = opts.mode != "fast"
        n_samples = opts.num_samples if opts.num_samples is not None else s.NUM_SAMPLES
        llm_calls: List[dict] = []
        notices: List[dict] = []

        def recorder(event: str, data: dict):
            if event == "llm_call":
                llm_calls.append(data)
            elif event == "notice":
                notices.append({**data, "t_ms": int((time.perf_counter() - t_run) * 1000)})
            emit(event, data)

        # Re-bind so the recorder sees events from deep layers too.
        with telemetry.bind(recorder):
            recorder("run_started", {
                "run_id": run_id, "question": question, "mode": "thorough" if thorough else "fast",
                "stages": [{"id": i, "name": n, "description": d} for i, n, d in STAGES],
            })

            stage_log: dict = {}

            def stage(sid: str, status: str, **data):
                payload = {"id": sid, "status": status, **data}
                stage_log[sid] = {"status": status, "summary": data.get("summary", "")}
                if status in ("done", "skipped", "error"):
                    payload.setdefault("duration_ms", timer.durations.get(sid, 0))
                recorder("stage", payload)

            # ── 1. Retrieve ────────────────────────────────────────────
            if not self.kb.ready:
                raise PipelineError("Your library is empty. Upload at least one document first.", "empty_kb")

            timer.start("retrieve")
            stage("retrieve", "running")
            top_k = opts.top_k or s.TOP_K
            hits: List[Hit] = await asyncio.to_thread(
                self.kb.search, question, top_k, s.CANDIDATE_POOL, opts.doc_ids)
            timer.stop("retrieve")
            sources = [h.to_dict(i) for i, h in enumerate(hits, start=1)]
            best = max((h.relevance for h in hits), default=0.0)
            stage("retrieve", "done", summary=f"{len(hits)} passages · best relevance {best:.0%}",
                  sources=sources)

            # ── 2. Answer ──────────────────────────────────────────────
            answer: Optional[AnswerResult] = None
            # With a reranker, relevance is a real probability: if every passage is
            # near zero, the library doesn't cover the question — decline instantly
            # instead of spending (slow, rate-limited) model calls on it.
            reranked = bool(hits) and all(h.rerank is not None for h in hits)
            off_topic = reranked and best < s.MIN_RELEVANCE
            skip_llm = not hits or off_topic
            if skip_llm:
                why = ("Nothing in your library matched the question." if not hits else
                       f"None of the retrieved passages look relevant (best match {best:.0%}).")
                answer = AnswerResult(text=prompts.ABSTAIN_PREFIX + " " + why,
                                      citations=[], invalid_citations=[], abstained=True)
                stage("answer", "skipped", summary="No relevant passages — declined without calling the model")
                recorder("token", {"text": answer.text})
            else:
                timer.start("answer")
                stage("answer", "running")
                try:
                    answer = await generate_answer(
                        self.router, question, hits,
                        on_token=lambda t: recorder("token", {"text": t}),
                        max_tokens=s.ANSWER_MAX_TOKENS)
                except LLMError as e:
                    timer.stop("answer")
                    stage("answer", "error", summary=e.short())
                    raise PipelineError(e.short(), e.kind)
                timer.stop("answer")
                stage("answer", "done",
                      summary=("Declined — documents don't cover it" if answer.abstained
                               else f"{len(answer.text.split())} words · cites {len(answer.citations)} source(s)"),
                      provider=answer.provider, model=answer.model)
            recorder("answer", {"text": answer.text, "citations": answer.citations,
                                "abstained": answer.abstained})

            # ── 3–5. Verify, consistency, reflect (concurrently) ───────
            verification: Optional[VerificationResult] = None
            consistency: Optional[ConsistencyResult] = None
            reflection: Optional[ReflectionResult] = None
            has_claims = not skip_llm and not (answer.abstained and not answer.citations)

            async def do_verify():
                nonlocal verification
                if not has_claims:
                    stage("verify", "skipped", summary="No factual claims to check")
                    return
                timer.start("verify")
                stage("verify", "running")
                try:
                    verification = await verify_answer(self.router, self.embedder, question, answer.text, hits)
                    timer.stop("verify")
                    c = verification.counts
                    summ = (f"{c['supported']} supported · {c['partial']} partial · "
                            f"{c['not_found'] + c['contradicted']} unsupported")
                    if not verification.judge_ok:
                        summ += " (local check only)"
                    stage("verify", "done", summary=summ, verification=verification.to_dict())
                except Exception as e:
                    timer.stop("verify")
                    log.exception("verify failed")
                    stage("verify", "error", summary=_short(e))

            async def do_consistency():
                nonlocal consistency
                if not thorough or n_samples <= 0 or skip_llm:
                    stage("consistency", "skipped",
                          summary="Fast mode" if not thorough else "Not applicable")
                    return
                timer.start("consistency")
                stage("consistency", "running")
                try:
                    consistency = await check_consistency(self.router, self.embedder, question, hits,
                                                          answer, n_samples, s.SAMPLE_TEMPERATURE)
                    timer.stop("consistency")
                    stage("consistency", "done",
                          summary=f"Agreement {consistency.score:.0%} across {len(consistency.samples)} re-samples",
                          consistency=consistency.to_dict())
                except Exception as e:
                    timer.stop("consistency")
                    stage("consistency", "error", summary=_short(e))

            async def do_reflect():
                nonlocal reflection
                if not thorough or not has_claims:
                    stage("reflect", "skipped", summary="Fast mode" if not thorough else "No claims to review")
                    return
                timer.start("reflect")
                stage("reflect", "running")
                try:
                    reflection = await reflect(self.router, question, answer.text, hits)
                    timer.stop("reflect")
                    stage("reflect", "done",
                          summary=f"Score {reflection.score:.0%} · {len(reflection.issues)} issue(s)",
                          reflection=reflection.to_dict())
                except Exception as e:
                    timer.stop("reflect")
                    stage("reflect", "error", summary=_short(e))

            # Verification first in the queue: it matters most for the score.
            await asyncio.gather(do_verify(), do_reflect(), do_consistency())

            # ── 6. Calibrate ───────────────────────────────────────────
            timer.start("calibrate")
            stage("calibrate", "running")
            confidence = calibrate(s, answer, [h.relevance for h in hits], verification,
                                   consistency, reflection, s.CONFIDENCE_THRESHOLD)
            timer.stop("calibrate")
            stage("calibrate", "done", summary=f"{confidence.level} · {confidence.score:.0%}",
                  confidence=confidence.to_dict())

            duration_ms = int((time.perf_counter() - t_run) * 1000)
            result = {
                "id": run_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "question": question,
                "mode": "thorough" if thorough else "fast",
                "answer": {
                    "text": answer.text, "citations": answer.citations,
                    "invalid_citations": answer.invalid_citations, "abstained": answer.abstained,
                    "provider": answer.provider, "model": answer.model,
                },
                "sources": sources,
                "verification": verification.to_dict() if verification else None,
                "consistency": consistency.to_dict() if consistency else None,
                "reflection": reflection.to_dict() if reflection else None,
                "confidence": confidence.to_dict(),
                "timings_ms": timer.durations,
                "stages": stage_log,
                "duration_ms": duration_ms,
                "llm_calls": llm_calls,
                "notices": notices,
                "prompt_version": prompts.PROMPT_VERSION,
            }
            if self.history is not None:
                try:
                    await asyncio.to_thread(self.history.save, result)
                except Exception:
                    log.exception("Could not save run history")
            recorder("complete", {"result": result})
            return result


class PipelineError(Exception):
    def __init__(self, message: str, kind: str = "error"):
        super().__init__(message)
        self.message = message
        self.kind = kind


def _short(e: Exception) -> str:
    if isinstance(e, LLMError):
        return e.short()
    return (str(e) or e.__class__.__name__)[:200]
