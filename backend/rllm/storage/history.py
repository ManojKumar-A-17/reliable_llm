"""Run history in SQLite (stdlib only)."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import List, Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id          TEXT PRIMARY KEY,
    created_at  TEXT NOT NULL,
    question    TEXT NOT NULL,
    mode        TEXT NOT NULL,
    level       TEXT,
    score       REAL,
    duration_ms INTEGER,
    result_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS runs_created ON runs(created_at DESC);
"""


class HistoryStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def save(self, result: dict) -> None:
        conf = result.get("confidence") or {}
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?)",
                (result["id"], result["created_at"], result["question"], result.get("mode", ""),
                 conf.get("level"), conf.get("score"), result.get("duration_ms"),
                 json.dumps(result, ensure_ascii=False)),
            )
            self._conn.commit()

    def list(self, limit: int = 50, offset: int = 0) -> List[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, created_at, question, mode, level, score, duration_ms FROM runs "
                "ORDER BY created_at DESC LIMIT ? OFFSET ?", (limit, offset)).fetchall()
        return [dict(r) for r in rows]

    def get(self, run_id: str) -> Optional[dict]:
        with self._lock:
            row = self._conn.execute("SELECT result_json FROM runs WHERE id=?", (run_id,)).fetchone()
        return json.loads(row["result_json"]) if row else None

    def delete(self, run_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM runs WHERE id=?", (run_id,))
            self._conn.commit()
        return cur.rowcount > 0

    def clear(self) -> int:
        with self._lock:
            cur = self._conn.execute("DELETE FROM runs")
            self._conn.commit()
        return cur.rowcount

    def metrics(self) -> dict:
        """Aggregate numbers for the research dashboard."""
        with self._lock:
            rows = self._conn.execute("SELECT level, score, duration_ms, result_json FROM runs").fetchall()
        n = len(rows)
        if not n:
            return {"runs": 0}
        levels: dict = {}
        claims = bad = 0
        for r in rows:
            levels[r["level"]] = levels.get(r["level"], 0) + 1
            v = (json.loads(r["result_json"]).get("verification") or {})
            counts = v.get("counts") or {}
            claims += sum(counts.values())
            bad += counts.get("not_found", 0) + counts.get("contradicted", 0)
        scores = [r["score"] for r in rows if r["score"] is not None]
        durs = sorted(r["duration_ms"] for r in rows if r["duration_ms"])
        return {
            "runs": n,
            "levels": levels,
            "mean_confidence": round(sum(scores) / len(scores), 4) if scores else None,
            "claims_checked": claims,
            "unsupported_claim_rate": round(bad / claims, 4) if claims else None,
            "median_latency_ms": durs[len(durs) // 2] if durs else None,
        }
