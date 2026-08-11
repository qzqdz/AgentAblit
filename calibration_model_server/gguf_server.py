"""calibration model server — GGUF / llama.cpp deployment (PC).

This is a thin launcher that re-exports the proven, proxy-shared GGUF server living
at src/shared/calibration_server.py (the single source of truth the TMI proxy also
uses).  No code is duplicated here — the GGUF deployment method IS that module.

Run either of:
    uvicorn calibration_model_server.gguf_server:app --host 127.0.0.1 --port 8011
    PYTHONPATH=src uvicorn shared.calibration_server:app --host 127.0.0.1 --port 8011

Both serve identical endpoints: /health, /v1/models, /v1/chat/completions, /correct.
See README.md for the env vars (TMI_GGUF_MODEL_PATH, TMI_GGUF_N_CTX, ...).
"""
from __future__ import annotations

import os
import sys

# Make `shared.calibration_server` importable without requiring PYTHONPATH=src.
_SRC = os.path.join(os.path.dirname(__file__), "..", "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from shared.calibration_server import app, create_app  # noqa: E402,F401

__all__ = ["app", "create_app"]
