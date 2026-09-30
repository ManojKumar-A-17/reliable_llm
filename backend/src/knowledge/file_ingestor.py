"""
file_ingestor.py
----------------
Parses uploaded files (PDF, Markdown, HTML, plain text) into chunks
with full citation metadata.  Replaces the SQuAD-only document_ingester.

Supported formats
-----------------
    .pdf    — via pypdf (page-level granularity preserved)
    .md     — plain text after stripping Markdown syntax
    .html   — via BeautifulSoup (strips tags, preserves text)
    .txt    — raw text

Usage
-----
    from src.knowledge.file_ingestor import ingest_file, DocumentMeta

    meta, chunks = ingest_file("/path/to/report.pdf")
    kb.add_document(meta, chunks)
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

from src.knowledge.chunker import Chunk, chunk_text


@dataclass
class DocumentMeta:
    """Metadata for an ingested document (stored in the document registry)."""
    doc_id:      str
    filename:    str
    file_type:   str          # 'pdf', 'md', 'html', 'txt'
    page_count:  int
    chunk_count: int
    created_at:  str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    size_bytes:  int = 0

    def as_dict(self) -> dict:
        return {
            "doc_id":      self.doc_id,
            "filename":    self.filename,
            "file_type":   self.file_type,
            "page_count":  self.page_count,
            "chunk_count": self.chunk_count,
            "created_at":  self.created_at,
            "size_bytes":  self.size_bytes,
        }


# ── Parsers ──────────────────────────────────────────────────────────────────

def _parse_pdf(path: Path) -> List[Tuple[int, str]]:
    """Return list of (page_num, text) pairs."""
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError("Install pypdf:  pip install pypdf")

    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            pages.append((i, text))
    return pages


def _parse_markdown(path: Path) -> List[Tuple[int, str]]:
    """Strip Markdown syntax, return as single page."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    # Remove code blocks
    raw = re.sub(r"```.*?```", "", raw, flags=re.DOTALL)
    raw = re.sub(r"`[^`]+`", "", raw)
    # Remove images and links, keep link text
    raw = re.sub(r"!\[.*?\]\(.*?\)", "", raw)
    raw = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", raw)
    # Remove heading markers, bold/italic
    raw = re.sub(r"#{1,6}\s+", "", raw)
    raw = re.sub(r"[*_]{1,3}([^*_]+)[*_]{1,3}", r"\1", raw)
    # Remove HTML tags that sometimes appear in .md
    raw = re.sub(r"<[^>]+>", "", raw)
    text = " ".join(raw.split())
    return [(0, text)] if text else []


def _parse_html(path: Path) -> List[Tuple[int, str]]:
    """Extract visible text from HTML."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        raise ImportError("Install beautifulsoup4:  pip install beautifulsoup4")

    raw = path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(raw, "html.parser")
    # Remove script and style elements
    for tag in soup(["script", "style", "noscript", "head"]):
        tag.decompose()
    text = " ".join(soup.get_text(separator=" ").split())
    return [(0, text)] if text else []


def _parse_txt(path: Path) -> List[Tuple[int, str]]:
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    text = " ".join(text.split())
    return [(0, text)] if text else []


_PARSERS = {
    ".pdf":  _parse_pdf,
    ".md":   _parse_markdown,
    ".html": _parse_html,
    ".htm":  _parse_html,
    ".txt":  _parse_txt,
}


# ── Main entry point ─────────────────────────────────────────────────────────

def ingest_file(
    path: str | Path,
    chunk_size: int = 250,
    overlap: int = 50,
) -> Tuple[DocumentMeta, List[Chunk]]:
    """
    Parse *path* and return (DocumentMeta, chunks).

    Raises
    ------
    ValueError  : unsupported file type
    FileNotFoundError : file does not exist
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = path.suffix.lower()
    parser = _PARSERS.get(ext)
    if parser is None:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Supported: {', '.join(_PARSERS)}"
        )

    doc_id = str(uuid.uuid4())
    file_type = ext.lstrip(".")
    if ext in (".htm", ".html"):
        file_type = "html"

    print(f"[FileIngestor] Parsing {path.name} ({file_type}) …")
    pages = parser(path)

    if not pages:
        raise ValueError(f"No extractable text found in {path.name}")

    all_chunks: List[Chunk] = []
    for page_num, text in pages:
        chunks = chunk_text(
            text=text,
            source=path.name,
            doc_id=doc_id,
            chunk_size=chunk_size,
            overlap=overlap,
            page_num=page_num,
        )
        all_chunks.extend(chunks)

    meta = DocumentMeta(
        doc_id=doc_id,
        filename=path.name,
        file_type=file_type,
        page_count=len(pages),
        chunk_count=len(all_chunks),
        size_bytes=path.stat().st_size,
    )

    print(f"[FileIngestor] {path.name} → {len(pages)} page(s), {len(all_chunks)} chunks")
    return meta, all_chunks


def ingest_bytes(
    content: bytes,
    filename: str,
    chunk_size: int = 250,
    overlap: int = 50,
) -> Tuple[DocumentMeta, List[Chunk]]:
    """
    Ingest raw file bytes (from an HTTP upload).
    Writes to a temp file, calls ingest_file, cleans up.
    """
    import tempfile
    suffix = Path(filename).suffix or ".txt"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        meta, chunks = ingest_file(tmp_path, chunk_size, overlap)
        # Patch the filename back to the original
        meta.filename = filename
        for c in chunks:
            c.source = filename
        meta.size_bytes = len(content)
        return meta, chunks
    finally:
        tmp_path.unlink(missing_ok=True)
