"""
FastAPI application factory.

Run with:  uvicorn api.server:app --reload --port 8000   (from backend/)
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from .. import __version__
from ..config import Settings, get_settings
from ..knowledge import (SUPPORTED_EXTENSIONS, DuplicateDocument, KnowledgeBase, ParseError,
                         Reranker, SentenceEmbedder)
from ..llm import LLMRouter, build_providers
from ..logging_setup import configure_logging
from ..pipeline import STAGES, Engine, PipelineError, QueryOptions
from ..storage.history import HistoryStore

log = logging.getLogger("rllm.api")


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    mode: str = Field("thorough", pattern="^(thorough|fast)$")
    top_k: Optional[int] = Field(None, ge=1, le=20)
    doc_ids: Optional[List[str]] = None
    num_samples: Optional[int] = Field(None, ge=0, le=5)


def create_app(settings: Optional[Settings] = None, *, embedder=None, reranker=None,
               providers=None) -> FastAPI:
    """Build the app. Keyword overrides exist for tests (fake models / LLMs)."""
    settings = settings or get_settings()
    configure_logging(settings.LOG_LEVEL)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        emb = embedder or SentenceEmbedder(settings.EMBEDDING_MODEL)
        rr = reranker if reranker is not None else (
            Reranker(settings.RERANK_MODEL) if settings.RERANK_ENABLED else None)
        kb = KnowledgeBase(settings.data_path, emb, rr,
                           settings.CHUNK_WORDS, settings.CHUNK_OVERLAP_WORDS)
        await asyncio.to_thread(kb.load)
        provs = providers if providers is not None else build_providers(settings)
        router = LLMRouter(provs, settings.LLM_MAX_RETRIES, settings.LLM_MAX_WAIT_SECS)
        history = HistoryStore(settings.data_path / "rllm.db")

        app.state.settings = settings
        app.state.kb = kb
        app.state.router = router
        app.state.history = history
        app.state.engine = Engine(kb, router, emb, history, settings)
        app.state.models_ready = False

        if provs:
            log.info("LLM providers: %s", ", ".join(f"{p.name}({p.model_for('strong')})" for p in provs))
        else:
            log.warning("No LLM provider configured — add a key to backend/.env")

        async def warm():
            try:
                await asyncio.to_thread(emb.warmup)
                if rr is not None:
                    await asyncio.to_thread(rr.warmup)
            finally:
                app.state.models_ready = True
                log.info("Local models ready")
        warm_task = asyncio.create_task(warm())
        yield
        warm_task.cancel()

    app = FastAPI(title="Reliable LLM API", version=__version__, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware, allow_origins=settings.cors_origins, allow_credentials=False,
        allow_methods=["*"], allow_headers=["*"],
    )

    # ── Health / config ────────────────────────────────────────────────
    @app.get("/api/health")
    def health(request: Request):
        st = request.app.state
        return {
            "status": "ok",
            "version": __version__,
            "models_ready": st.models_ready,
            "library": st.kb.stats(),
            "llm_ready": st.router.ready,
            "providers": st.router.describe(),
            "retrieval": {
                "embedding_model": st.kb.embedder.name,
                "reranker": st.kb.reranker.name if st.kb.reranker and st.kb.reranker.available else None,
                "top_k": st.settings.TOP_K,
            },
            "pipeline": {
                "stages": [{"id": i, "name": n, "description": d} for i, n, d in STAGES],
                "num_samples": st.settings.NUM_SAMPLES,
                "threshold": st.settings.CONFIDENCE_THRESHOLD,
            },
            "supported_types": list(SUPPORTED_EXTENSIONS),
            "max_upload_mb": st.settings.MAX_UPLOAD_MB,
        }

    # ── Library ────────────────────────────────────────────────────────
    @app.get("/api/documents")
    def list_documents(request: Request):
        kb: KnowledgeBase = request.app.state.kb
        return {"documents": kb.list_documents(), "stats": kb.stats()}

    async def _ingest_one(request: Request, file: UploadFile) -> dict:
        kb: KnowledgeBase = request.app.state.kb
        limit = request.app.state.settings.MAX_UPLOAD_MB * 1024 * 1024
        data = await file.read(limit + 1)
        name = file.filename or "upload"
        if len(data) > limit:
            return {"status": "error", "name": name,
                    "error": f"File is larger than {request.app.state.settings.MAX_UPLOAD_MB} MB"}
        if not data:
            return {"status": "error", "name": name, "error": "File is empty"}
        try:
            info = await asyncio.to_thread(kb.add_document, data, name)
            return {"status": "added", "name": name, "document": info.to_dict()}
        except DuplicateDocument as e:
            return {"status": "duplicate", "name": name, "document": e.existing.to_dict(),
                    "error": str(e)}
        except ParseError as e:
            return {"status": "error", "name": name, "error": str(e)}
        except Exception as e:
            log.exception("Ingest failed for %s", name)
            return {"status": "error", "name": name, "error": f"Could not process file: {e}"}

    @app.post("/api/documents")
    async def upload_documents(request: Request, files: List[UploadFile] = File(...)):
        results = [await _ingest_one(request, f) for f in files]
        return {"results": results, "stats": request.app.state.kb.stats()}

    @app.post("/api/documents/ingest")   # v1-compatible single upload
    async def ingest_legacy(request: Request, file: UploadFile = File(...)):
        r = await _ingest_one(request, file)
        if r["status"] == "error":
            raise HTTPException(422, r["error"])
        return r

    @app.delete("/api/documents/{doc_id}")
    async def delete_document(request: Request, doc_id: str):
        ok = await asyncio.to_thread(request.app.state.kb.remove_document, doc_id)
        if not ok:
            raise HTTPException(404, "Document not found")
        return {"deleted": doc_id, "stats": request.app.state.kb.stats()}

    @app.get("/api/documents/{doc_id}/chunks")
    def document_chunks(request: Request, doc_id: str):
        kb: KnowledgeBase = request.app.state.kb
        if doc_id not in kb.documents:
            raise HTTPException(404, "Document not found")
        return {"document": kb.documents[doc_id].to_dict(), "chunks": kb.document_chunks(doc_id)}

    @app.get("/api/documents/{doc_id}/file")
    def document_file(request: Request, doc_id: str):
        kb: KnowledgeBase = request.app.state.kb
        path = kb.document_file(doc_id)
        if not path:
            raise HTTPException(404, "Original file not stored")
        return FileResponse(path, filename=kb.documents[doc_id].name)

    # ── Query ──────────────────────────────────────────────────────────
    def _opts(body: QueryRequest) -> QueryOptions:
        return QueryOptions(mode=body.mode, top_k=body.top_k, doc_ids=body.doc_ids or None,
                            num_samples=body.num_samples)

    @app.post("/api/query")
    async def query_stream(request: Request, body: QueryRequest):
        engine: Engine = request.app.state.engine
        question = body.question.strip()
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        DONE = object()

        def emit(event: str, data: dict):
            loop.call_soon_threadsafe(queue.put_nowait, (event, data))

        async def producer():
            try:
                await engine.run(question, _opts(body), emit)
            except PipelineError as e:
                emit("error", {"message": e.message, "kind": e.kind})
            except Exception as e:
                log.exception("Pipeline crashed")
                emit("error", {"message": f"Unexpected server error: {e}", "kind": "internal"})
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, DONE)

        async def stream():
            task = asyncio.create_task(producer())
            try:
                yield ": connected\n\n"
                while True:
                    try:
                        item = await asyncio.wait_for(queue.get(), timeout=15)
                    except asyncio.TimeoutError:
                        yield ": keep-alive\n\n"      # holds the connection during rate-limit waits
                        continue
                    if item is DONE:
                        break
                    event, data = item
                    yield f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
            finally:
                if not task.done():
                    task.cancel()            # client went away — stop spending quota

        return StreamingResponse(stream(), media_type="text/event-stream", headers={
            "Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive",
        })

    @app.post("/api/query/sync")
    async def query_sync(request: Request, body: QueryRequest):
        try:
            return await request.app.state.engine.run(body.question.strip(), _opts(body), lambda e, d: None)
        except PipelineError as e:
            raise HTTPException(400, e.message)

    # ── History ────────────────────────────────────────────────────────
    @app.get("/api/runs")
    def list_runs(request: Request, limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0)):
        return {"runs": request.app.state.history.list(limit, offset)}

    @app.get("/api/runs/{run_id}")
    def get_run(request: Request, run_id: str):
        run = request.app.state.history.get(run_id)
        if not run:
            raise HTTPException(404, "Run not found")
        return run

    @app.delete("/api/runs/{run_id}")
    def delete_run(request: Request, run_id: str):
        if not request.app.state.history.delete(run_id):
            raise HTTPException(404, "Run not found")
        return {"deleted": run_id}

    @app.delete("/api/runs")
    def clear_runs(request: Request):
        return {"deleted": request.app.state.history.clear()}

    @app.get("/api/metrics")
    def metrics(request: Request):
        return request.app.state.history.metrics()

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        log.exception("Unhandled error on %s", request.url.path)
        return JSONResponse(status_code=500, content={"detail": f"Internal error: {exc}"})

    return app
