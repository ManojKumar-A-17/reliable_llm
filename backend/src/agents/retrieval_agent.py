"""
retrieval_agent.py  (FAISS upgrade)
------------------------------------
Replaces keyword-based ranking with semantic vector search using FAISS.
Falls back to empty context if the index has not been built yet, so the
rest of the pipeline still runs.

Index location: backend/data/index/knowledge  (.faiss + .pkl)
Build it once:  python backend/scripts/build_index.py
"""

import os
import time
from dataclasses import dataclass
from typing import List

from src.knowledge.knowledge_base import KnowledgeBase

# Path resolved relative to the backend/ working directory
INDEX_PATH = os.path.join(
    os.path.dirname(__file__),          # backend/src/agents/
    "..", "..", "data", "index", "knowledge"
)
INDEX_PATH = os.path.normpath(INDEX_PATH)

# Singleton — loaded once, reused for all queries in a session
_kb: KnowledgeBase | None = None


def _get_kb() -> KnowledgeBase:
    global _kb
    if _kb is None:
        _kb = KnowledgeBase()
        try:
            _kb.load(INDEX_PATH)
        except FileNotFoundError as exc:
            print(f"[RetrievalAgent] WARNING: {exc}")
            print("[RetrievalAgent] Continuing without retrieval context.")
    return _kb


# ── Result dataclass (unchanged interface) ───────────────────────────────────

@dataclass
class RetrievalResult:
    question: str
    documents: List[str]
    context_text: str
    retrieval_time: float


# ── Agent ────────────────────────────────────────────────────────────────────

class RetrievalAgent:
    """
    Semantic retrieval using FAISS + all-MiniLM-L6-v2.

    Public interface is identical to the old keyword-based version so the
    pipeline code does not need to change.
    """

    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        # Warm up the KB on first instantiation
        _get_kb()

    def run(self, question: str) -> RetrievalResult:
        t0 = time.time()
        kb = _get_kb()

        if not kb.is_ready():
            # Index not built yet — return empty context (pipeline still runs)
            return RetrievalResult(
                question=question,
                documents=[],
                context_text="",
                retrieval_time=round(time.time() - t0, 3),
            )

        results = kb.search(question, top_k=self.top_k)

        # Build context: "Source: <title>\n<passage text>"
        docs: List[str] = []
        for r in results:
            header = f"[Source: {r.chunk.source}]"
            docs.append(f"{header}\n{r.chunk.text}")

        context_text = "\n\n---\n\n".join(docs)

        return RetrievalResult(
            question=question,
            documents=docs,
            context_text=context_text,
            retrieval_time=round(time.time() - t0, 3),
        )
