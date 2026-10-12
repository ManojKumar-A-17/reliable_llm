"""End-to-end API tests: upload → streamed query → history, with a fake LLM."""

import json

import pytest
from fastapi.testclient import TestClient

from conftest import DOC_MD, FakeProvider
from rllm.api import create_app


def parse_sse(text: str):
    events = []
    for block in text.split("\n\n"):
        ev, data = None, []
        for line in block.splitlines():
            if line.startswith("event:"):
                ev = line[6:].strip()
            elif line.startswith("data:"):
                data.append(line[5:].strip())
        if ev:
            events.append((ev, json.loads("\n".join(data))))
    return events


@pytest.fixture
def client(settings, embedder):
    provider = FakeProvider()
    app = create_app(settings, embedder=embedder, reranker=None, providers=[provider])
    with TestClient(app) as c:
        c.provider = provider
        yield c


def test_health_and_empty_library(client):
    h = client.get("/api/health").json()
    assert h["status"] == "ok" and h["llm_ready"] and h["library"]["documents"] == 0
    r = client.post("/api/query", json={"question": "Anything?"})
    ev = parse_sse(r.text)
    assert ev[-1][0] == "error" and "library is empty" in ev[-1][1]["message"]


def test_upload_query_history(client):
    r = client.post("/api/documents", files=[("files", ("space.md", DOC_MD, "text/markdown")),
                                             ("files", ("again.md", DOC_MD, "text/markdown")),
                                             ("files", ("bad.exe", b"MZ", "application/octet-stream"))])
    statuses = [x["status"] for x in r.json()["results"]]
    assert statuses == ["added", "duplicate", "error"]
    doc_id = r.json()["results"][0]["document"]["id"]
    assert client.get(f"/api/documents/{doc_id}/chunks").json()["chunks"]

    r = client.post("/api/query", json={"question": "How many moons does Mars have?", "mode": "thorough"})
    assert r.status_code == 200
    events = parse_sse(r.text)
    kinds = [e for e, _ in events]
    assert kinds[0] == "run_started" and kinds[-1] == "complete"
    assert "token" in kinds and "llm_call" in kinds

    done = {d["id"]: d for e, d in events if e == "stage" and d["status"] in ("done", "skipped", "error")}
    assert set(done) == {"retrieve", "answer", "verify", "consistency", "reflect", "calibrate"}
    assert all(d["status"] == "done" for d in done.values()), done

    result = events[-1][1]["result"]
    assert "Phobos" in result["answer"]["text"]
    assert result["verification"]["counts"]["supported"] == 2
    assert all(c["quote_found"] for c in result["verification"]["claims"])
    assert result["confidence"]["level"] in ("HIGH", "MEDIUM")
    assert sorted(client.provider.calls) == sorted(["answer", "verify", "reflect", "sample", "sample"])

    runs = client.get("/api/runs").json()["runs"]
    assert runs[0]["id"] == result["id"]
    assert client.get(f"/api/runs/{result['id']}").json()["question"].startswith("How many")
    assert client.get("/api/metrics").json()["runs"] == 1

    # fast mode = answer + verify only
    client.provider.calls.clear()
    res = client.post("/api/query/sync", json={"question": "Largest planet?", "mode": "fast"}).json()
    assert client.provider.calls == ["answer", "verify"] and res["mode"] == "fast"

    assert client.delete(f"/api/documents/{doc_id}").status_code == 200
    assert client.get("/api/documents").json()["stats"]["documents"] == 0


def test_hallucinating_judge_is_caught(settings, embedder):
    # Judge claims "supported" with a quote that isn't in any source.
    provider = FakeProvider(
        answer="Mars has four moons [1].",
        verify_json={"claims": [{"claim": "Mars has four moons.", "cited": [1], "verdict": "supported",
                                 "source": 1, "quote": "Mars has four large moons orbiting it closely"}]},
    )
    app = create_app(settings, embedder=embedder, reranker=None, providers=[provider])
    with TestClient(app) as c:
        c.post("/api/documents", files=[("files", ("space.md", DOC_MD, "text/markdown"))])
        res = c.post("/api/query/sync", json={"question": "How many moons does Mars have?", "mode": "fast"}).json()
    claim = res["verification"]["claims"][0]
    assert claim["quote_found"] is False
    assert claim["verdict"] == "partial" and "4" in claim["note"]
    assert res["confidence"]["level"] != "HIGH"


class _LowReranker:
    name = "fake-reranker"
    available = True

    def score(self, query, texts):
        return [-9.0 for _ in texts]      # sigmoid ≈ 0.0001 → nothing relevant

    def warmup(self):
        pass


def test_off_topic_question_declines_without_llm(settings, embedder):
    provider = FakeProvider()
    app = create_app(settings, embedder=embedder, reranker=_LowReranker(), providers=[provider])
    with TestClient(app) as c:
        c.post("/api/documents", files=[("files", ("space.md", DOC_MD, "text/markdown"))])
        res = c.post("/api/query/sync", json={"question": "Recipe for sourdough pancakes"}).json()
    assert provider.calls == []
    assert res["answer"]["abstained"] and res["confidence"]["level"] == "ABSTAINED"


def test_overview_question_uses_opening_passages(settings, embedder):
    # Off-topic for the reranker, but clearly about the document as a whole.
    provider = FakeProvider(answer="The document lists facts about the Sun, Mars and Jupiter [1].")
    app = create_app(settings, embedder=embedder, reranker=_LowReranker(), providers=[provider])
    with TestClient(app) as c:
        c.post("/api/documents", files=[("files", ("space.md", DOC_MD, "text/markdown"))])
        res = c.post("/api/query/sync", json={"question": "What is this file about?", "mode": "fast"}).json()
    assert provider.calls == ["answer", "verify"]
    assert not res["answer"]["abstained"] and res["sources"]
    assert res["sources"][0]["text"].startswith("The Sun")      # opening passage of the document


def test_decline_with_cited_context_counts_as_abstained(settings, embedder):
    provider = FakeProvider(answer="The provided documents don't contain enough information to answer this. "
                                   "They do describe the moons of Mars [1].")
    app = create_app(settings, embedder=embedder, reranker=None, providers=[provider])
    with TestClient(app) as c:
        c.post("/api/documents", files=[("files", ("space.md", DOC_MD, "text/markdown"))])
        res = c.post("/api/query/sync", json={"question": "What is the mass of Phobos?", "mode": "thorough"}).json()
    assert res["answer"]["abstained"] and res["confidence"]["level"] == "ABSTAINED"
    assert "verify" not in provider.calls and "reflect" not in provider.calls
