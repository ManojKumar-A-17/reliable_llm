import asyncio
import io
import json

import httpx
import pytest

from conftest import DOC_MD, FakeProvider
from rllm.jsonutil import extract_json
from rllm.knowledge import DuplicateDocument, KnowledgeBase, ParseError, chunk_blocks, parse_document
from rllm.llm import GeminiProvider, LLMError, LLMRequest, LLMRouter, OpenAICompatProvider
from rllm.llm.base import AUTH, DAILY_QUOTA, RATE_LIMIT, parse_duration
from rllm.pipeline.calibrate import calibrate
from rllm.pipeline.stages import (AnswerResult, VerificationResult, is_abstention, parse_citations,
                                  quote_match)


def run(coro):
    return asyncio.run(coro)


# ── JSON / parsing helpers ────────────────────────────────────────────────
def test_extract_json_variants():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! {"a": [1, 2,], } trailing') == {"a": [1, 2]}
    assert extract_json('noise [1, 2] noise') == [1, 2]
    assert extract_json('{"s": "brace } in string"}') == {"s": "brace } in string"}
    assert extract_json("nothing here") is None


def test_parse_duration_and_citations():
    assert parse_duration("37s") == 37
    assert parse_duration("2m59.5s") == pytest.approx(179.5)
    assert parse_duration("12") == 12
    assert parse_citations("A [1]. B [2, 3][4]. C [x]") == [1, 2, 3, 4]
    assert quote_match("tallest volcano known", "Olympus Mons is the tallest volcano known.") == 1.0
    assert is_abstention("The provided documents don't contain enough information to answer this.")
    assert not is_abstention("Mars has two moons [1].")


# ── Parsing & chunking ────────────────────────────────────────────────────
def test_markdown_sections_and_chunks():
    blocks = parse_document(DOC_MD, "space.md")
    sections = {b.section for b in blocks}
    assert {"Solar System Facts", "Mars", "Jupiter"} <= sections
    chunks = chunk_blocks(blocks, "d1", "space.md", target_words=20, overlap_words=8)
    assert all(c.words <= 40 for c in chunks)
    assert any("§ Mars" in c.label() for c in chunks)
    # chunks never mix sections
    for c in chunks:
        assert not ("Phobos" in c.text and "Great Red Spot" in c.text)


def test_html_docx_txt_and_errors():
    html = b"<html><head><title>x</title><script>bad()</script></head><body><h2>Intro</h2><p>Hello world here.</p><ul><li>Item one</li></ul></body></html>"
    blocks = parse_document(html, "p.html")
    assert blocks[0].section == "Intro" and "Hello" in blocks[0].text
    assert all("bad()" not in b.text for b in blocks)

    import docx
    d = docx.Document()
    d.add_heading("Results", level=1)
    d.add_paragraph("Accuracy improved by 12 percent.")
    buf = io.BytesIO()
    d.save(buf)
    blocks = parse_document(buf.getvalue(), "r.docx")
    assert blocks[0].section == "Results"

    assert parse_document(b"para one.\n\npara two.", "a.txt")[1].text == "para two."
    with pytest.raises(ParseError):
        parse_document(b"x", "a.exe")
    with pytest.raises(ParseError):
        parse_document(b"   ", "a.txt")


# ── Knowledge base ────────────────────────────────────────────────────────
def test_kb_add_search_persist_remove(tmp_path, embedder):
    kb = KnowledgeBase(tmp_path, embedder, None, chunk_words=30, overlap_words=8)
    info = kb.add_document(DOC_MD, "space.md")
    assert info.chunks >= 2
    with pytest.raises(DuplicateDocument):
        kb.add_document(DOC_MD, "copy.md")

    hits = kb.search("How many moons does Mars have? Phobos Deimos", top_k=3)
    assert hits and "Phobos" in hits[0].chunk.text
    assert 0 <= hits[0].relevance <= 1

    kb2 = KnowledgeBase(tmp_path, embedder, None)
    kb2.load()
    assert kb2.stats()["chunks"] == kb.stats()["chunks"]
    assert kb2.search("Great Red Spot storm", top_k=1)[0].chunk.section == "Jupiter"

    assert kb2.remove_document(info.id)
    assert not kb2.ready and kb2.search("Mars") == []


# ── Router: pacing, retries, failover ─────────────────────────────────────
def test_router_retries_rate_limit_then_succeeds():
    p = FakeProvider(fail=[LLMError(RATE_LIMIT, "slow down", retry_after=0.1)])
    router = LLMRouter([p], max_retries=3, max_wait=5)
    r = run(router.complete(LLMRequest("q", purpose="answer")))
    assert r.provider == "fake" and p.calls == ["answer", "answer"]


def test_router_fails_over_on_long_wait_daily_quota_and_auth():
    for err in (LLMError(RATE_LIMIT, "wait", retry_after=999),
                LLMError(DAILY_QUOTA, "per day"),
                LLMError(AUTH, "bad key")):
        a, b = FakeProvider("a", fail=[err]), FakeProvider("b")
        router = LLMRouter([a, b], max_retries=3, max_wait=5)
        events = []
        from rllm import telemetry
        async def go():
            with telemetry.bind(lambda e, d: events.append((e, d))):
                return await router.complete(LLMRequest("q", purpose="answer"))
        assert run(go()).provider == "b"
        assert any(e == "notice" and d.get("level") == "failover" for e, d in events)


def test_router_no_provider_and_exhausted():
    with pytest.raises(LLMError) as ei:
        run(LLMRouter([]).complete(LLMRequest("q")))
    assert ei.value.kind == "no_provider"
    p = FakeProvider(fail=[LLMError(AUTH, "bad")])
    with pytest.raises(LLMError) as ei:
        run(LLMRouter([p]).complete(LLMRequest("q")))
    assert ei.value.kind == "exhausted"


def test_rate_limiter_paces_calls():
    from rllm.llm import RateLimiter
    lim = RateLimiter(2)
    async def go():
        await lim.acquire(); await lim.acquire()
        return lim.predicted_wait()
    assert 50 < run(go()) <= 60.1


# ── Provider wire formats (mocked HTTP) ───────────────────────────────────
def test_gemini_parsing_retry_and_thinking_fallback():
    seen = []

    def handler(request: httpx.Request):
        body = json.loads(request.content)
        seen.append(body)
        assert request.headers["x-goog-api-key"] == "k"
        if len(seen) == 1:
            return httpx.Response(429, json={"error": {"code": 429, "message": "Quota exceeded per minute",
                                  "details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "0.1s"}]}})
        if "thinkingConfig" in body["generationConfig"]:
            return httpx.Response(400, json={"error": {"message": "thinking_level is not supported for this model"}})
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [
            {"text": "hidden", "thought": True}, {"text": "Hello [1]"}]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 2}})

    g = GeminiProvider("k", {"strong": "gm"}, {"strong": 0}, transport=httpx.MockTransport(handler))
    router = LLMRouter([g], max_retries=3, max_wait=5)
    r = run(router.complete(LLMRequest("q", system="sys")))
    assert r.text == "Hello [1]" and r.output_tokens == 2
    assert seen[-1]["systemInstruction"]["parts"][0]["text"] == "sys"
    assert "thinkingConfig" not in seen[-1]["generationConfig"]


def test_gemini_daily_quota_classification():
    def handler(request):
        return httpx.Response(429, json={"error": {"message": "Quota exceeded for metric: generate_content_free_tier_requests, quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier"}})
    g = GeminiProvider("k", {"strong": "gm"}, {"strong": 0}, transport=httpx.MockTransport(handler))
    with pytest.raises(LLMError) as ei:
        run(g.complete(LLMRequest("q"), "gm"))
    assert ei.value.kind == DAILY_QUOTA


def test_openai_compat_streaming_and_json_fallback():
    calls = []

    def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        if "response_format" in body:
            return httpx.Response(400, json={"error": {"message": "response_format not supported"}})
        sse = "".join(f'data: {json.dumps({"choices": [{"delta": {"content": t}}]})}\n\n' for t in ["He", "llo"])
        return httpx.Response(200, text=sse + "data: [DONE]\n\n", headers={"content-type": "text/event-stream"})

    p = OpenAICompatProvider("groq", "Groq", "https://x/v1", "k", {"strong": "m"}, {"strong": 0},
                             transport=httpx.MockTransport(handler))
    toks = []
    r = run(p.complete(LLMRequest("q", json_mode=True), "m", on_token=toks.append))
    assert r.text == "Hello" and toks == ["He", "llo"] and len(calls) == 2


# ── Calibration rules ─────────────────────────────────────────────────────
def test_calibration_caps_and_abstention(settings):
    ans = AnswerResult("x [1]", [1], [], False)
    good = VerificationResult([object()], 1.0, 0.9, 1.0,
                              {"supported": 3, "partial": 0, "not_found": 0, "contradicted": 0}, 0.0, True)
    c = calibrate(settings, ans, [0.9, 0.5], good, None, None)
    assert c.level == "HIGH" and c.reliable

    bad = VerificationResult([object()], 0.9, 0.9, 1.0,
                             {"supported": 3, "partial": 0, "not_found": 0, "contradicted": 1}, 0.25, True)
    c = calibrate(settings, ans, [0.9], bad, None, None)
    assert c.score <= 0.5 and c.caps

    c = calibrate(settings, AnswerResult("The provided documents don't contain…", [], [], True), [0.1], None, None, None)
    assert c.level == "ABSTAINED"
    assert calibrate(settings, ans, [], None, None, None).level == "NO_EVIDENCE"


def test_low_reranker_does_not_zero_out_relevant_passages(tmp_path, embedder):
    class Pessimist:
        name, available = "fake", True
        def score(self, q, texts): return [-11.0] * len(texts)
    kb = KnowledgeBase(tmp_path, embedder, Pessimist(), chunk_words=30, overlap_words=8)
    kb.add_document(DOC_MD, "space.md")
    hits = kb.search("Mars moons Phobos Deimos", top_k=3)
    assert hits[0].relevance > 0.2 and "Phobos" in hits[0].chunk.text


def test_local_provider_calls_are_serialised():
    import asyncio as aio
    active, peak = [0], [0]

    class Local(FakeProvider):
        max_concurrency = 1
        async def complete(self, req, model, on_token=None):
            active[0] += 1; peak[0] = max(peak[0], active[0])
            await aio.sleep(0.02)
            active[0] -= 1
            return await super().complete(req, model, on_token)

    router = LLMRouter([Local("ollama")])
    async def go():
        await aio.gather(*[router.complete(LLMRequest("q", purpose="sample")) for _ in range(3)])
    run(go())
    assert peak[0] == 1
