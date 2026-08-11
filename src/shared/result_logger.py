"""Structured result logger for stance benchmark runs.

Policy: log metrics and verdict labels only.
Response text is truncated to ≤100 chars in all snippet fields.
Full response content is never persisted.
"""
from __future__ import annotations

import json
import time
from pathlib import Path


class ResultLogger:
    def __init__(self, output_path: str | Path) -> None:
        self.path = Path(output_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _truncate(text: str | None, limit: int = 100) -> str | None:
        if text is None:
            return None
        return text.replace("\n", " ").strip()[:limit]

    def _ts(self) -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())

    def log_trial(self, record: dict) -> None:
        safe = dict(record)
        for key in ("clean_snippet", "injected_snippet", "persist_snippet"):
            safe[key] = self._truncate(safe.get(key))
        safe["ts"] = self._ts()
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(safe, ensure_ascii=False) + "\n")

    def log_summary(self, summary: dict) -> None:
        summary = dict(summary)
        summary["ts"] = self._ts()
        summary["record_type"] = "summary"
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")
