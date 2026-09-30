# Reliable LLM

Evidence-first question answering over your own documents. Every answer is written only from retrieved passages, every claim is audited against its source, and the result carries a calibrated confidence score — or an honest “the documents don't cover this”.

```
                       ┌──────────── runs concurrently ────────────┐
question ─► retrieve ─► cited answer ─┬─► claim audit (LLM + local checks) ─┐
            BM25 + dense              ├─► self-consistency (re-samples)     ├─► calibrate ─► answer + score
            RRF + cross-encoder       └─► critical review                   ┘   (6 signals, hard caps)
```

## Quick start

```bash
./setup.sh                # venv, Python + npm packages, creates backend/.env, runs tests
nano backend/.env         # add at least one API key (see below)
./start.sh                # backend :8000 + frontend :3000
```

Open http://localhost:3000, drop documents into the library, and ask.

To run the two halves separately (two terminals):

```bash
# terminal 1 — from the repo root
source venv/bin/activate
cd backend && uvicorn api.server:app --reload --port 8000

# terminal 2
cd frontend && npm run dev
```

The first start downloads the cross-encoder reranker (~90 MB, once). The embedding model is the same `all-MiniLM-L6-v2` v1 used, so it is likely already cached.

## API keys (all free)

You need at least one. Providers are tried in `LLM_PROVIDER_ORDER`; if the first is rate-limited for longer than `LLM_MAX_WAIT_SECS`, the next one takes over automatically.

| Provider | Get a key | Free tier (approx.) | Notes |
|---|---|---|---|
| Google Gemini | https://aistudio.google.com/apikey | ~5 req/min, ~20/day on flash models | Default primary. Tight limits — pair it with Groq. |
| Groq | https://console.groq.com/keys | 30 req/min, ~1,000/day (70B) | Fast. **Recommended as a second provider.** |
| OpenRouter | https://openrouter.ai/keys | ~20 req/min, ~50/day on `:free` models | Many open models. |
| Ollama | https://ollama.com (local) | unlimited | `ollama pull llama3.1:8b`, set `OLLAMA_ENABLED=true`. |

Free-tier limits change often; set `GEMINI_RPM` / `GROQ_RPM` to whatever your console shows and the limiter will pace calls to match.

> **Security:** keys belong only in `backend/.env`, which is git-ignored. If a key has ever appeared in a screenshot, chat or commit, revoke it in the provider console and create a new one.

## What changed from v1

v1 showed the right idea; v2 makes it dependable.

| Area | v1 | v2 |
|---|---|---|
| LLM calls per question | ~10 (one per claim) | **5** thorough / **2** fast; the claim audit is one batched call |
| Rate limits | reacted to 429s, stalled for minutes | proactive per-model pacing, honours retry hints, **fails over** across providers, live countdown in the UI |
| Consistency score | never worked (`SimilarityCalculator.calculate` called with the wrong arguments; error swallowed) | fixed; compares the answer with independent re-samples, and penalises disagreement on *whether* to answer |
| Claim verification | LLM judge only | LLM judge **plus** local checks: the judge's quote must really appear in the source, claim↔sentence embedding match, number consistency, citation correctness |
| Retrieval | min-max blend of BM25 and FAISS | Reciprocal Rank Fusion + cross-encoder rerank; chunks follow sentences, pages and headings |
| Confidence | 3-signal average | 6 signals, renormalised when a stage is skipped, plus hard caps (contradiction, many unsupported claims, fabricated citations) |
| Abstention | none | the model is told to decline when the sources don't answer; the UI shows it as “Declined”, not as a low score |
| Answer UX | blank until the end | tokens stream in; citations are clickable and highlight the source |
| Persistence | pickle index; no history | JSON + `.npy` index, SQLite run history, duplicate detection |
| SDK | deprecated `google-generativeai` | plain HTTPS (`httpx`) for every provider — no vendor SDKs |
| Tests | none runnable | 16 offline tests (fake LLM, mocked HTTP) |

## How confidence is computed

| Signal | Source | Default weight |
|---|---|---|
| Claim support | auditor verdicts: supported 1.0 · partial 0.55 · not found 0.15 · contradicted 0 | 0.35 |
| Semantic grounding | each claim vs. the closest source sentence (local embeddings) | 0.15 |
| Citation accuracy | factual claims carry a citation that points at a supporting source | 0.10 |
| Self-consistency | answer vs. re-samples (embedding similarity) | 0.15 |
| Critical review | reviewer score, reduced by issue severity | 0.15 |
| Retrieval relevance | cross-encoder relevance of the cited passages | 0.10 |

Weights are renormalised over the signals that ran. Caps then apply: a contradicted claim caps the score at 50%; more than a third of claims not in the sources caps it at 55%; citing a non-existent source caps it at 70%. HIGH ≥ 80%, MEDIUM ≥ 60%, otherwise LOW. All weights are configurable in `.env`.

## Project layout

```
backend/
  api/server.py            ASGI entry point (uvicorn api.server:app)
  rllm/
    config.py              all settings (env / .env)
    llm/                   providers (Gemini REST, OpenAI-compatible), router, rate limiter
    knowledge/             parsing (pdf/docx/md/html/txt), chunking, embeddings, reranker, store
    pipeline/              prompts, stages, calibration, engine (orchestrator)
    storage/history.py     SQLite run history + metrics
    api/app.py             FastAPI routes + SSE streaming
  scripts/evaluate.py      batch evaluation (accuracy, ECE, abstention, unsupported-claim rate)
  data/eval/               sample questions for research_paper_draft.md
  tests/                   offline test-suite
frontend/
  src/App.jsx              app shell and state wiring
  src/state.js             run reducer (SSE events → UI state)
  src/api.js               API client + SSE parser
  src/ui/                  Sidebar, Main (answer, claims, review), Inspector (confidence, pipeline, sources)
```

v1 code (`backend/src`, `backend/configs`, the old `frontend/src/components`, …) is no longer used. `./cleanup_legacy.sh` moves it into `_legacy/` without deleting anything. The v1 index isn't compatible, so re-upload your documents once.

## API

Interactive docs: http://localhost:8000/docs

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | status, providers, library stats |
| GET / POST | `/api/documents` | list / upload (multipart `files`, several at once) |
| DELETE | `/api/documents/{id}` | remove a document |
| GET | `/api/documents/{id}/chunks` | inspect how a document was chunked |
| POST | `/api/query` | **SSE stream**: `{"question": "...", "mode": "thorough" \| "fast"}` |
| POST | `/api/query/sync` | same, returns the final JSON |
| GET / DELETE | `/api/runs`, `/api/runs/{id}` | history |
| GET | `/api/metrics` | aggregate reliability metrics across saved runs |

SSE events: `run_started`, `stage` (running/done/skipped/error with data), `token`, `answer_reset`, `answer`, `notice` (rate-limit waits, failovers), `llm_call`, `complete`, `error`.

## Research evaluation

```bash
source venv/bin/activate && cd backend
python scripts/evaluate.py data/eval/sample_questions.jsonl --mode thorough --out data/eval/results.json
```

Each JSONL line has `question`, optional `expected` (enables accuracy and expected calibration error) and optional `answerable` (tests abstention). Upload the documents first. On Gemini's free tier, prefer `--mode fast` or add Groq, because the daily quota is small.

## Tests

```bash
source venv/bin/activate && cd backend && python -m pytest -q tests
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'api'` | Run uvicorn from inside `backend/`, or use `./start.sh`. |
| `startup.sh: command not found` | Use `./start.sh` (or `bash start.sh`) from the repo root. |
| “No LLM key configured” | Add a key to `backend/.env`, then restart the backend. |
| “model … is not available on this key” | Set `GEMINI_MODEL` to a model your key lists in AI Studio. |
| Long “waiting Ns” notices | That's the free-tier pacing. Add `GROQ_API_KEY` for failover, or use Fast mode. |
| “daily free quota used up” | Wait for the reset, or add another provider. |
| Reranker download fails (offline) | Set `RERANK_ENABLED=false`; retrieval falls back to hybrid scores. |
