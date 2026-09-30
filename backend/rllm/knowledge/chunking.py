"""
Sentence-aware chunking.

Chunks never cross a page or section boundary (so a citation points at one
place), end on sentence boundaries, and overlap by whole trailing sentences
so a fact split across a boundary is still retrievable.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from itertools import groupby
from typing import List

from .parsing import Block

_SENT_SPLIT = re.compile(r"(?<=[.!?])[\"')\]]?\s+(?=[A-Z0-9\"'(\[“‘])")


@dataclass
class Chunk:
    id: str
    doc_id: str
    doc_name: str
    index: int
    text: str
    page: int = 0
    section: str = ""
    words: int = 0

    def label(self) -> str:
        parts = [self.doc_name]
        if self.page:
            parts.append(f"p.{self.page}")
        if self.section:
            sec = self.section if len(self.section) <= 48 else self.section[:45] + "…"
            parts.append(f"§ {sec}")
        return " · ".join(parts)

    def embed_text(self) -> str:
        """Text used for embeddings: the section title adds useful context."""
        return f"{self.section}\n{self.text}" if self.section else self.text

    def to_dict(self) -> dict:
        return asdict(self)


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


def _hard_wrap(sentence: str, max_words: int) -> List[str]:
    words = sentence.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]


def chunk_blocks(blocks: List[Block], doc_id: str, doc_name: str,
                 target_words: int = 200, overlap_words: int = 40) -> List[Chunk]:
    chunks: List[Chunk] = []

    for (page, section), group in groupby(blocks, key=lambda b: (b.page, b.section)):
        sentences: List[str] = []
        for block in group:
            for s in split_sentences(block.text):
                sentences.extend(_hard_wrap(s, target_words) if len(s.split()) > target_words else [s])

        current: List[str] = []
        count = 0
        i = 0
        while i < len(sentences):
            s = sentences[i]
            n = len(s.split())
            if current and count + n > target_words:
                _emit(chunks, current, doc_id, doc_name, page, section)
                # carry trailing sentences as overlap
                carry, carried = [], 0
                for prev in reversed(current):
                    w = len(prev.split())
                    if carried + w > overlap_words:
                        break
                    carry.insert(0, prev)
                    carried += w
                if carried + n > target_words:
                    carry, carried = [], 0
                current, count = carry, carried
            current.append(s)
            count += n
            i += 1
        if current:
            # Avoid a tiny trailing chunk that is pure overlap.
            if chunks and chunks[-1].page == page and chunks[-1].section == section \
                    and " ".join(current) in chunks[-1].text:
                continue
            _emit(chunks, current, doc_id, doc_name, page, section)

    return chunks


def _emit(chunks: List[Chunk], sentences: List[str], doc_id: str, doc_name: str,
          page: int, section: str) -> None:
    text = " ".join(sentences).strip()
    if not text:
        return
    idx = len(chunks)
    chunks.append(Chunk(
        id=f"{doc_id}:{idx}", doc_id=doc_id, doc_name=doc_name, index=idx,
        text=text, page=page, section=section, words=len(text.split()),
    ))
