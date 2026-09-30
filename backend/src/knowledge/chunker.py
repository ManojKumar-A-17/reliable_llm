"""
chunker.py
----------
Splits documents into overlapping chunks with full citation metadata.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List


@dataclass
class Chunk:
    """A single text chunk with full provenance metadata."""
    text:        str
    source:      str           # display name / filename
    chunk_index: int           # 0-based position within the source document

    # Citation metadata
    doc_id:      str = ""      # UUID of the parent document
    page_num:    int = 0       # page number (PDFs); 0 for single-page formats
    char_offset: int = 0       # char position in the original full text
    created_at:  str = ""      # ISO-8601 UTC timestamp
    chunk_hash:  str = ""      # sha-256 of text for dedup / integrity

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
        if not self.chunk_hash:
            self.chunk_hash = hashlib.sha256(self.text.encode()).hexdigest()[:16]

    def citation_label(self) -> str:
        """Short label for inline citation, e.g. 'report.pdf p.3'."""
        base = self.source.split("/")[-1] if "/" in self.source else self.source
        return f"{base} §{self.chunk_index + 1}" + (f" p.{self.page_num}" if self.page_num else "")

    def as_context_block(self, ref_num: int) -> str:
        """Format for inclusion in an LLM prompt context window."""
        return f"[Source {ref_num}] ({self.citation_label()})\n{self.text}"


def chunk_text(
    text: str,
    source: str = "",
    doc_id: str = "",
    chunk_size: int = 250,
    overlap: int = 50,
    page_num: int = 0,
) -> List[Chunk]:
    """
    Split *text* into overlapping word-bounded chunks.

    Parameters
    ----------
    text       : raw document text
    source     : display label (filename, URL, title)
    doc_id     : parent document UUID
    chunk_size : target words per chunk
    overlap    : words shared between adjacent chunks
    page_num   : page number hint (for multi-page callers)
    """
    words = text.split()
    if not words:
        return []

    if not doc_id:
        doc_id = str(uuid.uuid4())

    chunks: List[Chunk] = []
    start = 0
    idx = 0
    char_pos = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunk_text_str = " ".join(chunk_words)
        chunks.append(Chunk(
            text=chunk_text_str,
            source=source,
            chunk_index=idx,
            doc_id=doc_id,
            page_num=page_num,
            char_offset=char_pos,
        ))
        char_pos += len(chunk_text_str) + 1
        if end == len(words):
            break
        start += chunk_size - overlap
        idx += 1

    return chunks


def chunk_documents(
    documents: List[dict],
    chunk_size: int = 250,
    overlap: int = 50,
) -> List[Chunk]:
    """
    Chunk a list of document dicts.

    Each dict may have:
        text       : required
        source     : display name
        doc_id     : optional UUID (generated if absent)
        page_num   : optional int
    """
    all_chunks: List[Chunk] = []
    for doc in documents:
        text    = doc.get("text", "")
        source  = doc.get("source", "unknown")
        doc_id  = doc.get("doc_id", "")
        page    = doc.get("page_num", 0)
        all_chunks.extend(
            chunk_text(text, source, doc_id, chunk_size, overlap, page)
        )
    return all_chunks
