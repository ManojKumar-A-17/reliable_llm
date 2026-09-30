"""
knowledge_base.py
-----------------
Hybrid BM25 + FAISS vector store with document registry.

Features
--------
- Semantic search via FAISS (cosine, all-MiniLM-L6-v2)
- Keyword search via BM25 (rank-bm25)
- Hybrid scoring: alpha * vector + (1 - alpha) * BM25
- Document registry: doc_id → DocumentMeta
- Add / remove single documents without full rebuild
- Persists to two files: <path>.faiss + <path>.pkl
"""

from __future__ import annotations

import os
import pickle
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import faiss
except ImportError:
    raise ImportError("Run:  pip install faiss-cpu")

from sentence_transformers import SentenceTransformer
from src.knowledge.chunker import Chunk

EMBED_MODEL = "all-MiniLM-L6-v2"


@dataclass
class SearchResult:
    chunk:     Chunk
    score:     float    # hybrid 0–1, higher = more relevant
    bm25_score:   float = 0.0
    vector_score: float = 0.0
    rank:      int = 0  # 1-based result rank


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokeniser for BM25."""
    return re.findall(r"\b\w+\b", text.lower())


class KnowledgeBase:
    """
    Build once, query many times.  Add / remove documents incrementally.

    Typical usage
    -------------
    kb = KnowledgeBase()
    kb.add_document(meta, chunks)          # call for each uploaded file
    kb.save("backend/data/index/knowledge")

    # Later:
    kb = KnowledgeBase()
    kb.load("backend/data/index/knowledge")
    results = kb.search("Who invented the telephone?", top_k=5)
    """

    def __init__(self, model_name: str = EMBED_MODEL, hybrid_alpha: float = 0.5):
        """
        Parameters
        ----------
        hybrid_alpha : float  0 = BM25-only, 1 = vector-only, 0.5 = equal blend
        """
        print(f"[KnowledgeBase] Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.hybrid_alpha = hybrid_alpha

        self.index: Optional[faiss.Index] = None
        self.chunks: List[Chunk] = []
        self._embeddings: Optional[np.ndarray] = None   # kept for incremental adds

        # BM25 state
        self._bm25 = None           # rank_bm25.BM25Okapi instance
        self._tokenized_corpus: List[List[str]] = []

        # Document registry: doc_id → dict (metadata + chunk indices)
        self.registry: Dict[str, dict] = {}
        self._built = False

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _rebuild_faiss(self, embeddings: np.ndarray) -> None:
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(embeddings.astype(np.float32))

    def _rebuild_bm25(self) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("Run:  pip install rank-bm25")
        self._tokenized_corpus = [_tokenize(c.text) for c in self.chunks]
        self._bm25 = BM25Okapi(self._tokenized_corpus) if self._tokenized_corpus else None

    # ── Build (bulk, initial) ────────────────────────────────────────────────

    def build(self, chunks: List[Chunk], batch_size: int = 256) -> None:
        """Embed all chunks and build both indexes."""
        if not chunks:
            raise ValueError("chunks list is empty — nothing to index")
        print(f"[KnowledgeBase] Embedding {len(chunks):,} chunks …")
        t0 = time.time()
        texts = [c.text for c in chunks]
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        self.chunks = list(chunks)
        self._embeddings = embeddings
        self._rebuild_faiss(embeddings)
        self._rebuild_bm25()
        self._built = True
        print(f"[KnowledgeBase] Built in {time.time() - t0:.1f}s — {len(chunks):,} vectors")

    # ── Incremental add / remove ─────────────────────────────────────────────

    def add_document(self, meta, chunks: List[Chunk]) -> None:
        """
        Add one document's chunks to the live index.

        Parameters
        ----------
        meta   : DocumentMeta (or any object with .doc_id, .filename, .as_dict())
        chunks : list of Chunk from file_ingestor.ingest_file()
        """
        if not chunks:
            return

        doc_id = meta.doc_id
        if doc_id in self.registry:
            print(f"[KnowledgeBase] Document {doc_id} already indexed — skipping")
            return

        print(f"[KnowledgeBase] Embedding {len(chunks)} new chunks …")
        texts = [c.text for c in chunks]
        new_emb = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        # Track start index for the registry
        start_idx = len(self.chunks)

        # Append to state
        self.chunks.extend(chunks)
        self._tokenized_corpus.extend([_tokenize(c.text) for c in chunks])

        if self._embeddings is None:
            self._embeddings = new_emb
        else:
            self._embeddings = np.vstack([self._embeddings, new_emb])

        # Rebuild both indexes (FAISS add is fine; BM25 must rebuild)
        self._rebuild_faiss(self._embeddings)
        self._rebuild_bm25()
        self._built = True

        # Register the document
        self.registry[doc_id] = {
            **meta.as_dict(),
            "_chunk_start": start_idx,
            "_chunk_end":   len(self.chunks),
        }
        print(f"[KnowledgeBase] Added '{meta.filename}' — total {len(self.chunks):,} chunks")

    def remove_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to *doc_id*.  Returns True if found.
        Triggers a full index rebuild (unavoidable with FAISS IndexFlatIP).
        """
        if doc_id not in self.registry:
            return False

        keep_indices = [
            i for i, c in enumerate(self.chunks) if c.doc_id != doc_id
        ]
        self.chunks = [self.chunks[i] for i in keep_indices]
        self._embeddings = (
            self._embeddings[keep_indices] if self._embeddings is not None and len(keep_indices) > 0
            else np.zeros((0, self.dim), dtype=np.float32)
        )
        del self.registry[doc_id]

        if self.chunks:
            self._rebuild_faiss(self._embeddings)
            self._rebuild_bm25()
        else:
            self.index = None
            self._bm25 = None
            self._tokenized_corpus = []
            self._built = False

        print(f"[KnowledgeBase] Removed document {doc_id} — {len(self.chunks):,} chunks remain")
        return True

    # ── Search ───────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 6,
        alpha: Optional[float] = None,
    ) -> List[SearchResult]:
        """
        Hybrid BM25 + vector search.

        Parameters
        ----------
        query  : natural-language question
        top_k  : number of results
        alpha  : override instance-level hybrid_alpha for this call
        """
        if not self._built or not self.chunks:
            return []

        alpha = self.hybrid_alpha if alpha is None else alpha
        n = len(self.chunks)
        effective_k = min(top_k, n)

        # ── Vector scores ────────────────────────────────────────────
        q_vec = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)
        raw_scores, indices = self.index.search(q_vec, effective_k * 2)
        vec_scores_full = np.zeros(n)
        for score, idx in zip(raw_scores[0], indices[0]):
            if 0 <= idx < n:
                vec_scores_full[idx] = float(score)

        # ── BM25 scores ──────────────────────────────────────────────
        bm25_scores_full = np.zeros(n)
        if self._bm25 is not None:
            q_tokens = _tokenize(query)
            raw_bm25 = self._bm25.get_scores(q_tokens)
            bm25_scores_full = np.array(raw_bm25, dtype=np.float64)

        # ── Normalise both to [0, 1] ──────────────────────────────────
        def _norm(arr: np.ndarray) -> np.ndarray:
            lo, hi = arr.min(), arr.max()
            if hi == lo:
                return np.zeros_like(arr)
            return (arr - lo) / (hi - lo)

        vec_norm  = _norm(vec_scores_full)
        bm25_norm = _norm(bm25_scores_full)
        hybrid    = alpha * vec_norm + (1 - alpha) * bm25_norm

        # ── Top-k ────────────────────────────────────────────────────
        top_indices = np.argsort(hybrid)[::-1][:effective_k]

        results: List[SearchResult] = []
        for rank, idx in enumerate(top_indices, start=1):
            if hybrid[idx] == 0:
                continue
            results.append(SearchResult(
                chunk=self.chunks[idx],
                score=float(round(hybrid[idx], 4)),
                bm25_score=float(round(bm25_norm[idx], 4)),
                vector_score=float(round(vec_norm[idx], 4)),
                rank=rank,
            ))

        return results

    # ── Persist ──────────────────────────────────────────────────────────────

    def save(self, path_prefix: str) -> None:
        if not self.chunks:
            print("[KnowledgeBase] Nothing to save.")
            return
        os.makedirs(os.path.dirname(path_prefix) or ".", exist_ok=True)
        if self.index:
            faiss.write_index(self.index, f"{path_prefix}.faiss")
        state = {
            "chunks":             self.chunks,
            "embeddings":         self._embeddings,
            "tokenized_corpus":   self._tokenized_corpus,
            "registry":           self.registry,
            "hybrid_alpha":       self.hybrid_alpha,
        }
        with open(f"{path_prefix}.pkl", "wb") as f:
            pickle.dump(state, f)
        print(f"[KnowledgeBase] Saved → {path_prefix}.faiss / .pkl")

    def load(self, path_prefix: str) -> None:
        faiss_path = f"{path_prefix}.faiss"
        pkl_path   = f"{path_prefix}.pkl"
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(
                f"No index found at {pkl_path}\n"
                "Ingest at least one document first."
            )
        with open(pkl_path, "rb") as f:
            state = pickle.load(f)

        self.chunks             = state["chunks"]
        self._embeddings        = state.get("embeddings")
        self._tokenized_corpus  = state.get("tokenized_corpus", [])
        self.registry           = state.get("registry", {})
        self.hybrid_alpha       = state.get("hybrid_alpha", self.hybrid_alpha)

        if os.path.exists(faiss_path):
            self.index = faiss.read_index(faiss_path)
        elif self._embeddings is not None and len(self._embeddings) > 0:
            self._rebuild_faiss(self._embeddings)

        if self._tokenized_corpus:
            self._rebuild_bm25()

        self._built = bool(self.chunks)
        print(f"[KnowledgeBase] Loaded {len(self.chunks):,} chunks, "
              f"{len(self.registry)} documents")

    # ── Info ─────────────────────────────────────────────────────────────────

    def is_ready(self) -> bool:
        return self._built and bool(self.chunks)

    def size(self) -> int:
        return len(self.chunks)

    def list_documents(self) -> List[dict]:
        return [
            {k: v for k, v in meta.items() if not k.startswith("_")}
            for meta in self.registry.values()
        ]
