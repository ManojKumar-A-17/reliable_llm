"""
Pipeline stages. Each stage is an async function that takes plain inputs and
returns a plain dataclass, so it can be unit-tested without the API layer.

LLM-judged signals are always paired with a *local* check (embeddings,
string matching) so that one model grading itself is never the only
evidence of reliability.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..jsonutil import extract_json
from ..knowledge.chunking import split_sentences
from ..knowledge.store import Hit
from ..llm import LLMRequest, LLMRouter
from ..llm.base import TokenCallback
from . import prompts

_CITE_RE = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")
VERDICT_VALUE = {"supported": 1.0, "partial": 0.55, "not_found": 0.15, "contradicted": 0.0}


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))


def parse_citations(text: str) -> List[int]:
    nums: List[int] = []
    for m in _CITE_RE.finditer(text or ""):
        for n in re.split(r"[,;]", m.group(1)):
            n = n.strip()
            if n.isdigit():
                nums.append(int(n))
    return nums


def strip_citations(text: str) -> str:
    return re.sub(r"\s*" + _CITE_RE.pattern, "", text or "").strip()


def _norm(s: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", (s or "").lower()))


_NUM_WORDS = {w: str(i) for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve".split())}


def numbers_in(text: str) -> set:
    """Numeric facts in text, with small number-words normalised to digits."""
    out = set()
    for tok in re.findall(r"\d+(?:[.,]\d+)?|[a-z]+", (text or "").lower()):
        if tok[0].isdigit():
            out.add(tok.replace(",", ""))
        elif tok in _NUM_WORDS:
            out.add(_NUM_WORDS[tok])
    return out


def quote_match(quote: str, source_text: str) -> float:
    """1.0 if the quote appears verbatim (modulo punctuation), else token recall."""
    q, s = _norm(quote), _norm(source_text)
    if not q:
        return 0.0
    if q in s:
        return 1.0
    q_tokens = [t for t in q.split() if len(t) > 2]
    if not q_tokens:
        return 0.0
    s_tokens = set(s.split())
    return sum(t in s_tokens for t in q_tokens) / len(q_tokens)


# ════════════════════════════════════════════════════════════════════════════
# 1. Answer generation
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class AnswerResult:
    text: str
    citations: List[int]
    invalid_citations: List[int]
    abstained: bool
    provider: str = ""
    model: str = ""
    latency_ms: int = 0


def is_abstention(text: str) -> bool:
    head = _norm(text[:220])
    return _norm(prompts.ABSTAIN_PREFIX)[:40] in head or head.startswith(
        ("the provided documents do not contain", "the sources do not contain", "i couldn t find",
         "the documents do not contain"))


async def generate_answer(router: LLMRouter, question: str, hits: List[Hit],
                          on_token: Optional[TokenCallback] = None,
                          max_tokens: int = 1200) -> AnswerResult:
    resp = await router.complete(LLMRequest(
        prompt=prompts.answer_prompt(question, hits), system=prompts.ANSWER_SYSTEM,
        temperature=0.2, max_tokens=max_tokens, role="strong", purpose="answer",
    ), on_token=on_token)
    cites = parse_citations(resp.text)
    valid = sorted({c for c in cites if 1 <= c <= len(hits)})
    invalid = sorted({c for c in cites if not 1 <= c <= len(hits)})
    return AnswerResult(
        text=resp.text, citations=valid, invalid_citations=invalid,
        # The prompt lets a decline add one cited sentence about what the sources
        # do cover, so citations don't make it an answer.
        abstained=is_abstention(resp.text),
        provider=resp.provider, model=resp.model, latency_ms=resp.latency_ms,
    )


# ════════════════════════════════════════════════════════════════════════════
# 2. Claim verification (LLM judge + local grounding + quote check)
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class ClaimCheck:
    claim: str
    verdict: str                       # supported | partial | contradicted | not_found
    cited: List[int] = field(default_factory=list)
    source: Optional[int] = None       # best source ref (1-based)
    quote: str = ""
    quote_found: Optional[bool] = None # did the judge's quote really appear in that source?
    grounding: float = 0.0             # local semantic support 0–1
    best_sentence: str = ""            # closest source sentence (local)
    best_sentence_source: Optional[int] = None
    citation_ok: Optional[bool] = None
    note: str = ""
    judged_by: str = "llm"             # "llm" | "local"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class VerificationResult:
    claims: List[ClaimCheck]
    support: Optional[float]
    grounding: Optional[float]
    citation: Optional[float]
    counts: Dict[str, int]
    hallucination_rate: Optional[float]
    judge_ok: bool

    def to_dict(self) -> dict:
        d = asdict(self)
        d["claims"] = [c.to_dict() for c in self.claims]
        return d


class _SentenceIndex:
    """Embeds every sentence of the retrieved sources once per run."""

    def __init__(self, embedder, hits: List[Hit]):
        self.sentences: List[str] = []
        self.refs: List[int] = []
        for ref, h in enumerate(hits, start=1):
            for s in split_sentences(h.chunk.text) or [h.chunk.text]:
                if len(s.split()) >= 3:
                    self.sentences.append(s)
                    self.refs.append(ref)
        self.embedder = embedder
        self.vecs = embedder.encode(self.sentences) if self.sentences else None

    def best(self, texts: List[str]) -> List[tuple]:
        if self.vecs is None or not texts:
            return [(0.0, "", None) for _ in texts]
        q = self.embedder.encode(texts)
        sims = q @ self.vecs.T
        out = []
        for row in sims:
            j = int(np.argmax(row))
            out.append((float(row[j]), self.sentences[j], self.refs[j]))
        return out


def _grounding_from_cos(cos: float) -> float:
    # MiniLM: paraphrase ≈ 0.75–0.95, same topic ≈ 0.4–0.6, unrelated < 0.25
    return _clamp((cos - 0.30) / 0.50)


def _fallback_claims(answer: str) -> List[dict]:
    out = []
    for s in split_sentences(answer):
        clean = strip_citations(s)
        if len(clean.split()) >= 4 and not is_abstention(clean):
            out.append({"claim": clean, "cited": parse_citations(s), "verdict": None})
    return out[:10]


async def verify_answer(router: LLMRouter, embedder, question: str, answer: str,
                        hits: List[Hit]) -> VerificationResult:
    raw_claims: List[dict] = []
    judge_ok = False
    try:
        resp = await router.complete(LLMRequest(
            prompt=prompts.verify_prompt(question, answer, hits), system=prompts.VERIFY_SYSTEM,
            temperature=0.0, max_tokens=1600, json_mode=True, role="strong", purpose="verify",
        ))
        data = extract_json(resp.text)
        if isinstance(data, dict) and isinstance(data.get("claims"), list):
            raw_claims = [c for c in data["claims"] if isinstance(c, dict) and c.get("claim")]
            judge_ok = True
        elif isinstance(data, list):
            raw_claims = [c for c in data if isinstance(c, dict) and c.get("claim")]
            judge_ok = True
    except Exception:
        # judge unavailable → local-only verification below
        judge_ok = False

    if not raw_claims:
        raw_claims = _fallback_claims(answer)

    index = await asyncio.to_thread(_SentenceIndex, embedder, hits)
    locals_ = await asyncio.to_thread(index.best, [str(c["claim"]) for c in raw_claims])

    checks: List[ClaimCheck] = []
    for c, (cos, best_sent, best_ref) in zip(raw_claims, locals_):
        grounding = _grounding_from_cos(cos)
        verdict = str(c.get("verdict") or "").lower().replace(" ", "_").replace("-", "_")
        judged_by = "llm"
        if verdict not in VERDICT_VALUE:
            judged_by = "local"
            verdict = "supported" if grounding >= 0.75 else "partial" if grounding >= 0.45 else "not_found"

        cited = [int(x) for x in (c.get("cited") or []) if str(x).isdigit() and 1 <= int(x) <= len(hits)]
        src = c.get("source")
        src = int(src) if str(src).isdigit() and 1 <= int(src) <= len(hits) else None
        quote = str(c.get("quote") or "").strip()
        quote_found = None
        note = str(c.get("note") or "").strip()

        if quote and src:
            quote_found = quote_match(quote, hits[src - 1].chunk.text) >= 0.8
            if not quote_found:
                # the judge might have mis-numbered the source — check all
                for ref, h in enumerate(hits, start=1):
                    if quote_match(quote, h.chunk.text) >= 0.8:
                        quote_found, src = True, ref
                        break

        # Guard against a judge that "finds" evidence which isn't there.
        ref_text = hits[(src or best_ref) - 1].chunk.text if (src or best_ref) else ""
        missing_nums = numbers_in(str(c["claim"])) - numbers_in(ref_text)
        if verdict == "supported" and missing_nums:
            verdict = "partial"
            note = f"number(s) {', '.join(sorted(missing_nums))} not found in the source"
        if verdict == "supported" and quote_found is False and grounding < 0.45:
            verdict, note = "partial", "judge's quoted evidence was not found in the source"
        if verdict == "supported" and quote_found is None and grounding < 0.30:
            verdict, note = "partial", "weak semantic match to any source"

        best_src = src or best_ref
        citation_ok = None
        if cited:
            citation_ok = best_src in cited if best_src else None
            if citation_ok is False and verdict in ("supported", "partial"):
                # a cited source that also supports it is still acceptable
                citation_ok = any(quote_match(c["claim"], hits[r - 1].chunk.text) >= 0.5 for r in cited) \
                              or (best_ref in cited)
        elif verdict in ("supported", "partial"):
            citation_ok = False  # factual claim with no citation

        checks.append(ClaimCheck(
            claim=str(c["claim"]).strip(), verdict=verdict, cited=cited, source=src, quote=quote,
            quote_found=quote_found, grounding=round(grounding, 3), best_sentence=best_sent,
            best_sentence_source=best_ref, citation_ok=citation_ok, note=note, judged_by=judged_by,
        ))

    counts = {k: sum(1 for c in checks if c.verdict == k) for k in VERDICT_VALUE}
    n = len(checks)
    if n == 0:
        return VerificationResult([], None, None, None, counts, None, judge_ok)

    support = sum(VERDICT_VALUE[c.verdict] for c in checks) / n
    grounding = sum(c.grounding for c in checks) / n
    cit_rel = [c for c in checks if c.citation_ok is not None]
    citation = (sum(1 for c in cit_rel if c.citation_ok) / len(cit_rel)) if cit_rel else None
    halluc = (counts["not_found"] + counts["contradicted"]) / n
    return VerificationResult(checks, round(support, 4), round(grounding, 4),
                              None if citation is None else round(citation, 4),
                              counts, round(halluc, 4), judge_ok)


# ════════════════════════════════════════════════════════════════════════════
# 3. Consistency (self-agreement across re-samples)
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class ConsistencyResult:
    score: Optional[float]
    similarities: List[float]
    samples: List[str]
    abstention_agreement: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


async def check_consistency(router: LLMRouter, embedder, question: str, hits: List[Hit],
                            answer: AnswerResult, n: int = 2,
                            temperature: float = 0.8) -> ConsistencyResult:
    req = LLMRequest(prompt=prompts.answer_prompt(question, hits), system=prompts.ANSWER_SYSTEM,
                     temperature=temperature, max_tokens=600, role="fast", purpose="sample")
    results = await asyncio.gather(*[router.complete(req) for _ in range(n)], return_exceptions=True)
    samples = [r.text for r in results if not isinstance(r, BaseException) and r.text]
    if not samples:
        errs = [r for r in results if isinstance(r, BaseException)]
        raise errs[0] if errs else RuntimeError("No samples produced")

    texts = [strip_citations(answer.text)] + [strip_citations(s) for s in samples]
    vecs = await asyncio.to_thread(embedder.encode, texts)
    sims = [float(vecs[0] @ v) for v in vecs[1:]]
    # map cosine → agreement: paraphrases of the same answer sit ≳0.85
    mapped = [_clamp((s - 0.50) / 0.40) for s in sims]
    score = sum(mapped) / len(mapped)

    agree = sum(1 for s in samples if is_abstention(s) == answer.abstained) / len(samples)
    if agree < 1.0:
        score *= 0.5 + 0.5 * agree        # disagreement on *whether* to answer is serious
    return ConsistencyResult(round(score, 4), [round(s, 4) for s in sims], samples, round(agree, 3))


# ════════════════════════════════════════════════════════════════════════════
# 4. Self-reflection (critique)
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class ReflectionResult:
    score: Optional[float]
    judge_score: Optional[float]
    answers_question: Optional[bool]
    issues: List[dict]
    summary: str

    def to_dict(self) -> dict:
        return asdict(self)


_SEVERITY_PENALTY = {"high": 0.30, "medium": 0.15, "low": 0.05}


async def reflect(router: LLMRouter, question: str, answer: str, hits: List[Hit]) -> ReflectionResult:
    resp = await router.complete(LLMRequest(
        prompt=prompts.reflect_prompt(question, answer, hits), system=prompts.REFLECT_SYSTEM,
        temperature=0.1, max_tokens=900, json_mode=True, role="strong", purpose="reflect",
    ))
    data = extract_json(resp.text)
    if not isinstance(data, dict):
        raise ValueError("Reviewer returned unparseable output")

    issues = []
    for it in data.get("issues") or []:
        if isinstance(it, dict) and it.get("detail"):
            sev = str(it.get("severity", "low")).lower()
            issues.append({
                "type": str(it.get("type", "other")).lower(),
                "severity": sev if sev in _SEVERITY_PENALTY else "low",
                "detail": str(it["detail"]).strip(),
            })
    try:
        judge = _clamp(float(data.get("score")))
    except (TypeError, ValueError):
        judge = None
    penalty = min(0.9, sum(_SEVERITY_PENALTY[i["severity"]] for i in issues))
    rule = 1.0 - penalty
    # Small models sometimes return score 0 alongside only minor issues;
    # averaging with the severity-based score keeps one bad number from dominating.
    score = (judge + rule) / 2 if judge is not None else rule
    aq = data.get("answers_question")
    return ReflectionResult(
        score=round(score, 4), judge_score=judge,
        answers_question=aq if isinstance(aq, bool) else None,
        issues=issues, summary=str(data.get("summary") or "").strip(),
    )
