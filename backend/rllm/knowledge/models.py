"""
Local ML models: sentence embedder and cross-encoder reranker.

Both load lazily (first use) and run on CPU. ``HashEmbedder`` is a
dependency-free stand-in used by the test-suite.
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
from typing import List, Optional, Protocol

import numpy as np

log = logging.getLogger("rllm.models")


class Embedder(Protocol):
    name: str

    def encode(self, texts: List[str]) -> np.ndarray: ...


def _load_cached_first(cls, name: str):
    """Load from the local Hugging Face cache without touching the network;
    download only when the model isn't cached yet. Otherwise every start-up
    checks huggingface.co, which stalls for minutes on a slow or offline link."""
    try:
        return cls(name, device="cpu", local_files_only=True)
    except Exception:
        log.info("%s not in local cache — downloading", name)
        return cls(name, device="cpu")


class SentenceEmbedder:
    def __init__(self, model_name: str):
        self.name = model_name
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        with self._lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer
                log.info("Loading embedding model %s …", self.name)
                self._model = _load_cached_first(SentenceTransformer, self.name)
        return self._model

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        model = self._load()
        vecs = model.encode(texts, batch_size=64, convert_to_numpy=True,
                            normalize_embeddings=True, show_progress_bar=False)
        return vecs.astype(np.float32)

    def warmup(self) -> None:
        self._load()


class HashEmbedder:
    """Deterministic bag-of-words hashing embedder (tests / no-ML fallback)."""

    def __init__(self, dim: int = 256):
        self.name = f"hash-{dim}"
        self.dim = dim

    def encode(self, texts: List[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in re.findall(r"\w+", t.lower()):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0 if (h >> 8) & 1 else -1.0
            n = np.linalg.norm(out[i])
            if n:
                out[i] /= n
        return out

    def warmup(self) -> None:
        pass


class Reranker:
    """Cross-encoder reranker (ms-marco MiniLM). Returns raw relevance logits."""

    def __init__(self, model_name: str):
        self.name = model_name
        self._model = None
        self._failed = False
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return not self._failed

    def _load(self):
        with self._lock:
            if self._model is None and not self._failed:
                try:
                    from sentence_transformers import CrossEncoder
                    log.info("Loading reranker %s …", self.name)
                    self._model = _load_cached_first(CrossEncoder, self.name)
                except Exception as e:  # offline, disk, etc. — degrade gracefully
                    log.warning("Reranker unavailable (%s); using hybrid scores only", e)
                    self._failed = True
        return self._model

    def score(self, query: str, texts: List[str]) -> Optional[List[float]]:
        model = self._load()
        if model is None or not texts:
            return None
        scores = model.predict([(query, t) for t in texts], show_progress_bar=False)
        return [float(s) for s in scores]

    def warmup(self) -> None:
        self._load()
