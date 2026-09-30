"""
Knowledge base: hybrid lexical + semantic retrieval with reranking.

Retrieval pipeline
------------------
1. **Dense**   cosine similarity against normalised sentence embeddings.
2. **Sparse**  BM25 over lower-cased word tokens (catches exact names,
               numbers and jargon that embeddings blur).
3. **Fuse**    Reciprocal Rank Fusion — robust to the very different score
               scales of (1) and (2), unlike min-max blending.
4. **Rerank**  a cross-encoder reads (question, passage) pairs jointly and
               re-scores the fused candidate pool. Its sigmoid output is a
               usable *relevance probability*, which feeds the confidence
               calibrator.

Persistence is plain files (JSON + .npy) — no pickle, so the index is safe
to inspect and portable across code versions.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import shutil
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .chunking import Chunk, chunk_blocks
from .models import Embedder, Reranker
from .parsing import file_type_for, parse_document

log = logging.getLogger("rllm.kb")

INDEX_VERSION = 2
_STOP = set("""a an the and or of to in on for with by at from as is are was were be been being it its
this that these those which who whom what when where why how do does did not no can could should would
will may might must shall has have had i you he she we they them his her our your their""".split())


def _tokens(text: str) -> List[str]:
    return [t for t in re.findall(r"\w+", text.lower()) if t not in _STOP]


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


@dataclass
class DocumentInfo:
    id: str
    name: str
    type: str
    size_bytes: int
    sha256: str
    chunks: int
    words: int
    pages: int
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Hit:
    chunk: Chunk
    relevance: float                # 0–1, calibrated-ish
    dense: float = 0.0              # cosine
    sparse: float = 0.0             # bm25 raw
    rerank: Optional[float] = None  # cross-encoder logit
    rank: int = 0

    def to_dict(self, ref: int, full_text: bool = True) -> dict:
        c = self.chunk
        return {
            "ref": ref,
            "chunk_id": c.id,
            "doc_id": c.doc_id,
            "doc_name": c.doc_name,
            "label": c.label(),
            "page": c.page,
            "section": c.section,
            "text": c.text if full_text else c.text[:280],
            "relevance": round(self.relevance, 4),
            "dense": round(self.dense, 4),
            "sparse": round(self.sparse, 4),
            "rerank": None if self.rerank is None else round(self.rerank, 3),
        }


class DuplicateDocument(Exception):
    def __init__(self, existing: DocumentInfo):
        super().__init__(f"'{existing.name}' is already in the library")
        self.existing = existing


class KnowledgeBase:

    def __init__(self, root: Path, embedder: Embedder, reranker: Optional[Reranker] = None,
                 chunk_words: int = 200, overlap_words: int = 40):
        self.root = Path(root)
        self.index_dir = self.root / "index"
        self.upload_dir = self.root / "uploads"
        self.embedder = embedder
        self.reranker = reranker
        self.chunk_words = chunk_words
        self.overlap_words = overlap_words

        self._lock = threading.RLock()
        self.documents: Dict[str, DocumentInfo] = {}
        self.chunks: List[Chunk] = []
        self._emb: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self._bm25 = None

    # ── Persistence ────────────────────────────────────────────────────
    def load(self) -> None:
        with self._lock:
            meta_p = self.index_dir / "meta.json"
            if not meta_p.exists():
                log.info("No index yet at %s — starting empty", self.index_dir)
                return
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
            self.documents = {
                d["id"]: DocumentInfo(**d)
                for d in json.loads((self.index_dir / "documents.json").read_text(encoding="utf-8"))
            }
            self.chunks = [
                Chunk(**json.loads(line))
                for line in (self.index_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()
            ]
            emb_p = self.index_dir / "embeddings.npy"
            emb = np.load(emb_p) if emb_p.exists() else np.zeros((0, 0), dtype=np.float32)
            if meta.get("embedding_model") != self.embedder.name or len(emb) != len(self.chunks):
                log.warning("Embedding model changed (%s → %s); re-embedding %d chunks",
                            meta.get("embedding_model"), self.embedder.name, len(self.chunks))
                emb = self.embedder.encode([c.embed_text() for c in self.chunks]) if self.chunks \
                    else np.zeros((0, 0), dtype=np.float32)
                self._emb = emb
                self._rebuild_bm25()
                self.save()
            else:
                self._emb = emb.astype(np.float32)
                self._rebuild_bm25()
            log.info("Knowledge base loaded — %d documents, %d chunks", len(self.documents), len(self.chunks))

    def save(self) -> None:
        with self._lock:
            self.index_dir.mkdir(parents=True, exist_ok=True)
            tmp = self.index_dir / ".tmp"
            tmp.mkdir(exist_ok=True)
            (tmp / "documents.json").write_text(
                json.dumps([d.to_dict() for d in self.documents.values()], indent=1), encoding="utf-8")
            (tmp / "chunks.jsonl").write_text(
                "\n".join(json.dumps(c.to_dict(), ensure_ascii=False) for c in self.chunks), encoding="utf-8")
            np.save(tmp / "embeddings.npy", self._emb)
            (tmp / "meta.json").write_text(json.dumps({
                "version": INDEX_VERSION,
                "embedding_model": self.embedder.name,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }), encoding="utf-8")
            for f in tmp.iterdir():                 # atomic-ish swap per file
                f.replace(self.index_dir / f.name)
            tmp.rmdir()

    # ── Mutation ───────────────────────────────────────────────────────
    def _rebuild_bm25(self) -> None:
        if not self.chunks:
            self._bm25 = None
            return
        from rank_bm25 import BM25Okapi
        self._bm25 = BM25Okapi([_tokens(c.embed_text()) or ["∅"] for c in self.chunks])

    def add_document(self, data: bytes, filename: str) -> DocumentInfo:
        filename = Path(filename).name or "document.txt"
        sha = hashlib.sha256(data).hexdigest()
        with self._lock:
            for d in self.documents.values():
                if d.sha256 == sha:
                    raise DuplicateDocument(d)

        blocks = parse_document(data, filename)               # may raise ParseError
        doc_id = uuid.uuid4().hex[:12]
        chunks = chunk_blocks(blocks, doc_id, filename, self.chunk_words, self.overlap_words)
        if not chunks:
            from .parsing import ParseError
            raise ParseError("Document produced no text chunks")
        vecs = self.embedder.encode([c.embed_text() for c in chunks])   # slow part, outside lock

        info = DocumentInfo(
            id=doc_id, name=filename, type=file_type_for(filename), size_bytes=len(data),
            sha256=sha, chunks=len(chunks), words=sum(c.words for c in chunks),
            pages=max((b.page for b in blocks), default=0),
        )
        with self._lock:
            self.chunks.extend(chunks)
            self._emb = vecs if self._emb.size == 0 else np.vstack([self._emb, vecs])
            self.documents[doc_id] = info
            self._rebuild_bm25()
            self.upload_dir.mkdir(parents=True, exist_ok=True)
            (self.upload_dir / f"{doc_id}{Path(filename).suffix.lower()}").write_bytes(data)
            self.save()
        log.info("Indexed %s → %d chunks", filename, len(chunks))
        return info

    def remove_document(self, doc_id: str) -> bool:
        with self._lock:
            info = self.documents.pop(doc_id, None)
            if info is None:
                return False
            keep = [i for i, c in enumerate(self.chunks) if c.doc_id != doc_id]
            self.chunks = [self.chunks[i] for i in keep]
            self._emb = self._emb[keep] if keep else np.zeros((0, 0), dtype=np.float32)
            self._rebuild_bm25()
            for f in self.upload_dir.glob(f"{doc_id}.*"):
                f.unlink(missing_ok=True)
            self.save()
            return True

    def clear(self) -> None:
        with self._lock:
            self.documents.clear()
            self.chunks = []
            self._emb = np.zeros((0, 0), dtype=np.float32)
            self._bm25 = None
            shutil.rmtree(self.upload_dir, ignore_errors=True)
            self.save()

    # ── Read ───────────────────────────────────────────────────────────
    @property
    def ready(self) -> bool:
        return bool(self.chunks)

    def list_documents(self) -> List[dict]:
        docs = sorted(self.documents.values(), key=lambda d: d.created_at, reverse=True)
        return [d.to_dict() for d in docs]

    def document_file(self, doc_id: str) -> Optional[Path]:
        info = self.documents.get(doc_id)
        if not info:
            return None
        p = self.upload_dir / f"{doc_id}{Path(info.name).suffix.lower()}"
        return p if p.exists() else None

    def document_chunks(self, doc_id: str) -> List[dict]:
        return [c.to_dict() | {"label": c.label()} for c in self.chunks if c.doc_id == doc_id]

    def stats(self) -> dict:
        return {
            "documents": len(self.documents),
            "chunks": len(self.chunks),
            "words": sum(d.words for d in self.documents.values()),
        }

    def search(self, query: str, top_k: int = 6, pool: int = 24,
               doc_ids: Optional[List[str]] = None) -> List[Hit]:
        with self._lock:
            n = len(self.chunks)
            if n == 0 or not query.strip():
                return []
            chunks = self.chunks
            emb = self._emb
            bm25 = self._bm25

        allowed = np.ones(n, dtype=bool)
        if doc_ids:
            wanted = set(doc_ids)
            allowed = np.array([c.doc_id in wanted for c in chunks], dtype=bool)
            if not allowed.any():
                return []

        q = self.embedder.encode([query])[0]
        dense = emb @ q
        sparse = np.array(bm25.get_scores(_tokens(query)), dtype=np.float64) if bm25 else np.zeros(n)
        dense = np.where(allowed, dense, -np.inf)
        sparse = np.where(allowed, sparse, -np.inf)

        pool = max(top_k, min(pool, n))
        dense_rank = np.argsort(-dense)[:pool]
        sparse_rank = [i for i in np.argsort(-sparse)[:pool] if sparse[i] > 0]

        rrf: Dict[int, float] = {}
        for rank, i in enumerate(dense_rank):
            if np.isfinite(dense[i]):
                rrf[int(i)] = rrf.get(int(i), 0.0) + 1.0 / (60 + rank + 1)
        for rank, i in enumerate(sparse_rank):
            rrf[int(i)] = rrf.get(int(i), 0.0) + 1.0 / (60 + rank + 1)
        candidates = sorted(rrf, key=rrf.get, reverse=True)[:pool]
        if not candidates:
            return []

        rerank_scores = None
        if self.reranker is not None and self.reranker.available:
            try:
                rerank_scores = self.reranker.score(query, [chunks[i].text for i in candidates])
            except Exception as e:
                log.warning("Rerank failed: %s", e)

        # Newer sentence-transformers versions may already return probabilities.
        already_prob = rerank_scores is not None and all(0.0 <= x <= 1.0 for x in rerank_scores)

        hits: List[Hit] = []
        for j, i in enumerate(candidates):
            d = float(dense[i])
            # MiniLM cosine: ~0.15 unrelated … ~0.7 strong match
            dense_rel = max(0.0, min(1.0, (d - 0.15) / 0.55))
            if rerank_scores is not None:
                r = rerank_scores[j]
                rr_rel = r if already_prob else _sigmoid(r)
                # The ms-marco cross-encoder is trained on web queries and scores
                # short / acronym questions ("what is RLLM?") near zero even when
                # the passage is on-topic. Never let it drag relevance below what
                # the embedding match already shows.
                rel = max(rr_rel, dense_rel)
            else:
                rel = dense_rel
            hits.append(Hit(chunk=chunks[i], relevance=rel, dense=d,
                            sparse=float(sparse[i]) if np.isfinite(sparse[i]) else 0.0,
                            rerank=None if rerank_scores is None else rerank_scores[j]))

        if rerank_scores is not None:
            hits.sort(key=lambda h: (h.relevance, h.rerank), reverse=True)
            log.debug("rerank scores for %r: %s", query, [round(h.rerank, 2) for h in hits[:top_k]])
        # (without a reranker, RRF order is kept)

        hits = hits[:top_k]
        for r, h in enumerate(hits, start=1):
            h.rank = r
        return hits
