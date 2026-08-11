"""Append-only JSONL trace sink for proxy events."""
from __future__ import annotations

import json
import time
from pathlib import Path


def load_trace_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    events: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            events.append(payload)
    return events


class FileTraceSink:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)

    def record(self, session_id: str, event: dict) -> None:
        entry = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()), **event}
        path = self.directory / f"{session_id}.jsonl"
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
