"""
Batch evaluation for research experiments.

Runs every question in a JSONL file through the full pipeline (in-process,
no server needed) and reports reliability metrics.

Input (one JSON object per line):
    {"question": "How many moons does Mars have?", "expected": "two", "answerable": true}
    - "expected"   optional reference answer (enables accuracy + calibration error)
    - "answerable" optional; false = the library should NOT answer it (tests abstention)

Usage (from backend/, venv active, documents already uploaded):
    python scripts/evaluate.py data/eval/questions.jsonl --mode thorough --out data/eval/results.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rllm.config import get_settings  # noqa: E402
from rllm.knowledge import KnowledgeBase, Reranker, SentenceEmbedder  # noqa: E402
from rllm.llm import LLMRouter, build_providers  # noqa: E402
from rllm.pipeline import Engine, PipelineError, QueryOptions  # noqa: E402
from rllm.storage.history import HistoryStore  # noqa: E402


def _tokens(s: str) -> set:
    return set(re.findall(r"[a-z0-9]+", (s or "").lower()))


def is_correct(answer: str, expected: str) -> bool:
    exp = _tokens(expected)
    return bool(exp) and len(exp & _tokens(answer)) / len(exp) >= 0.6


def ece(pairs, bins: int = 5) -> float:
    """Expected calibration error over (confidence, correct) pairs."""
    total, err = len(pairs), 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        bucket = [(p, c) for p, c in pairs if lo <= p < hi or (b == bins - 1 and p == 1.0)]
        if bucket:
            conf = sum(p for p, _ in bucket) / len(bucket)
            acc = sum(c for _, c in bucket) / len(bucket)
            err += len(bucket) / total * abs(conf - acc)
    return round(err, 4)


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("questions", type=Path)
    ap.add_argument("--mode", choices=["thorough", "fast"], default="thorough")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    s = get_settings()
    emb = SentenceEmbedder(s.EMBEDDING_MODEL)
    kb = KnowledgeBase(s.data_path, emb, Reranker(s.RERANK_MODEL) if s.RERANK_ENABLED else None,
                       s.CHUNK_WORDS, s.CHUNK_OVERLAP_WORDS)
    kb.load()
    if not kb.ready:
        sys.exit("Library is empty — upload documents in the app first.")
    engine = Engine(kb, LLMRouter(build_providers(s), s.LLM_MAX_RETRIES, s.LLM_MAX_WAIT_SECS), emb,
                    HistoryStore(s.data_path / "rllm.db"), s)

    items = [json.loads(l) for l in args.questions.read_text().splitlines() if l.strip()]
    rows = []
    for i, item in enumerate(items, 1):
        q = item["question"]
        t0 = time.time()
        try:
            res = await engine.run(q, QueryOptions(mode=args.mode), lambda e, d: None)
        except PipelineError as e:
            print(f"[{i}/{len(items)}] ERROR {e.message}")
            continue
        conf = res["confidence"]
        v = res.get("verification") or {}
        counts = v.get("counts") or {}
        row = {
            "question": q, "level": conf["level"], "confidence": conf["score"],
            "abstained": res["answer"]["abstained"], "claims": sum(counts.values()),
            "unsupported": counts.get("not_found", 0) + counts.get("contradicted", 0),
            "latency_s": round(time.time() - t0, 1), "answer": res["answer"]["text"],
        }
        if "expected" in item:
            row["correct"] = is_correct(res["answer"]["text"], item["expected"])
        if "answerable" in item:
            row["abstention_correct"] = row["abstained"] == (not item["answerable"])
        rows.append(row)
        print(f"[{i}/{len(items)}] {conf['level']:<10} {conf['score']:.2f}  {row['latency_s']:>5}s  {q[:70]}")

    claims = sum(r["claims"] for r in rows)
    summary = {
        "n": len(rows), "mode": args.mode,
        "mean_confidence": round(sum(r["confidence"] for r in rows) / len(rows), 4) if rows else None,
        "unsupported_claim_rate": round(sum(r["unsupported"] for r in rows) / claims, 4) if claims else None,
        "abstention_rate": round(sum(r["abstained"] for r in rows) / len(rows), 4) if rows else None,
        "mean_latency_s": round(sum(r["latency_s"] for r in rows) / len(rows), 2) if rows else None,
    }
    labelled = [r for r in rows if "correct" in r]
    if labelled:
        summary["accuracy"] = round(sum(r["correct"] for r in labelled) / len(labelled), 4)
        summary["ece"] = ece([(r["confidence"], r["correct"]) for r in labelled])
    abst = [r for r in rows if "abstention_correct" in r]
    if abst:
        summary["abstention_accuracy"] = round(sum(r["abstention_correct"] for r in abst) / len(abst), 4)

    print("\n" + json.dumps(summary, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
        print(f"Saved → {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
