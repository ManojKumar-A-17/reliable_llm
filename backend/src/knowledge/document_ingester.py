"""
document_ingester.py
--------------------
Extracts Wikipedia passages from the SQuAD dataset and feeds them
into the KnowledgeBase.  Run via build_index.py — not called at inference time.
"""

from typing import List, Dict
from datasets import load_dataset

from src.knowledge.chunker import chunk_documents, Chunk


def load_squad_passages(
    num_examples: int = 5000,
    split: str = "train",
) -> List[Dict]:
    """
    Pull Wikipedia passages from SQuAD (Stanford QA Dataset).

    Each SQuAD example has a 'context' field — a Wikipedia paragraph —
    and a 'title' field. We extract unique contexts as our knowledge base.

    Returns
    -------
    List of {"text": str, "source": str} dicts.
    """
    print(f"[Ingester] Loading SQuAD — {split} split …")
    dataset = load_dataset("rajpurkar/squad", split=split)

    seen: set = set()
    documents: List[Dict] = []

    for example in dataset.select(range(min(num_examples, len(dataset)))):
        context = example.get("context", "").strip()
        title   = example.get("title", "Unknown")
        if not context or context in seen:
            continue
        seen.add(context)
        documents.append({"text": context, "source": title})

    print(f"[Ingester] Collected {len(documents):,} unique Wikipedia passages")
    return documents


def ingest(
    num_examples: int = 5000,
    chunk_size: int = 250,
    overlap: int = 50,
    split: str = "train",
) -> List[Chunk]:
    """
    Full pipeline: load → chunk → return chunks ready for KnowledgeBase.build().
    """
    documents = load_squad_passages(num_examples=num_examples, split=split)
    chunks    = chunk_documents(documents, chunk_size=chunk_size, overlap=overlap)
    print(f"[Ingester] Produced {len(chunks):,} chunks "
          f"(chunk_size={chunk_size}, overlap={overlap})")
    return chunks
