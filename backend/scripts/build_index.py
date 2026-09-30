"""
build_index.py
--------------
One-time script to build the FAISS knowledge base from SQuAD Wikipedia
passages and save it to disk.

Run from the project root (reliable_llm/):
    python backend/scripts/build_index.py

Options (edit at the top of this file):
    NUM_EXAMPLES  — how many SQuAD training examples to pull passages from
    CHUNK_SIZE    — target words per chunk
    OVERLAP       — word overlap between adjacent chunks
    INDEX_PATH    — where to save the .faiss and .pkl files

After running this once you do NOT need to run it again unless you want to
rebuild with more documents.
"""

import sys
import os

# Make sure backend/ is on the path when run as a script
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))        # backend/scripts/
BACKEND_DIR = os.path.dirname(SCRIPT_DIR)                        # backend/
sys.path.insert(0, BACKEND_DIR)

# ── Config ────────────────────────────────────────────────────────────────────
NUM_EXAMPLES = 5000       # SQuAD training rows to pull Wikipedia passages from
CHUNK_SIZE   = 250        # words per chunk
OVERLAP      = 50         # overlapping words between chunks
INDEX_PATH   = os.path.join(BACKEND_DIR, "data", "index", "knowledge")
# ─────────────────────────────────────────────────────────────────────────────

from src.knowledge.document_ingester import ingest
from src.knowledge.knowledge_base import KnowledgeBase


def main():
    print("=" * 60)
    print("Building FAISS knowledge base")
    print(f"  Source   : SQuAD / train ({NUM_EXAMPLES} examples)")
    print(f"  Chunk    : {CHUNK_SIZE} words, {OVERLAP} overlap")
    print(f"  Save to  : {INDEX_PATH}")
    print("=" * 60)

    # 1. Load SQuAD Wikipedia passages and chunk them
    chunks = ingest(
        num_examples=NUM_EXAMPLES,
        chunk_size=CHUNK_SIZE,
        overlap=OVERLAP,
        split="train",
    )

    # 2. Embed and index
    kb = KnowledgeBase()
    kb.build(chunks)

    # 3. Persist
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    kb.save(INDEX_PATH)

    print("\n✓  Done.  Index is ready at:", INDEX_PATH)
    print("   Now run your pipeline — RetrievalAgent will use it automatically.")


if __name__ == "__main__":
    main()
