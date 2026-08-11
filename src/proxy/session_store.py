from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class SessionStore:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)

    def path_for(self, session_id: str) -> Path:
        return self.directory / f"{session_id}.json"

    def write(self, session_id: str, snapshot: dict[str, Any]) -> None:
        self.path_for(session_id).write_text(
            json.dumps(snapshot, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def read(self, session_id: str) -> dict[str, Any] | None:
        path = self.path_for(session_id)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
