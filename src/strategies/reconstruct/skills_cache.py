"""Versioned, retryable cache for asynchronously extracted host skills.

The extractor is deliberately kept off the request hot path.  Cache entries are
therefore explicit state machines rather than ``summary-or-empty`` values:

``missing -> pending -> ready`` or ``missing/pending -> failed -> pending``.

An empty extractor response is not evidence that no skills exist.  Only the exact
``NONE`` sentinel is accepted as a successful empty result; transport/model failures
and blank output use bounded exponential backoff and may be retried by a later turn.
"""
from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable


_CACHE_SCHEMA_VERSION = 2
_DEFAULT_CONTRACT_VERSION = "skills-extractor-v1"
_DEFAULT_MODEL_IDENTITY = "unspecified"

# key -> entry.  Absence is the explicit ``missing`` state.
_CACHE: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
_IN_FLIGHT: dict[str, asyncio.Task[Any]] = {}
_MAX_ENTRIES = 512
_STORE_DIR: Path | None = None
_EXTRACTOR_CONTRACT_VERSION = _DEFAULT_CONTRACT_VERSION
_MODEL_IDENTITY = _DEFAULT_MODEL_IDENTITY
_RETRY_BASE_SECONDS = 5.0
_RETRY_MAX_SECONDS = 300.0
_CLOCK: Callable[[], float] = time.monotonic
_UNSET = object()


def configure(
    *,
    max_sessions: int = 512,
    store_dir: str = "",
    extractor_contract_version: str | None = None,
    model_identity: str | None = None,
    retry_base_seconds: float | None = None,
    retry_max_seconds: float | None = None,
    clock: Callable[[], float] | None | object = _UNSET,
) -> None:
    """Configure bounds, persistence and extractor identity.

    The original two keyword arguments remain valid.  Contract/model identity are part
    of the key so changing either cannot silently reuse a summary produced by a stale
    prompt, output schema, or model.  ``clock`` is injectable for deterministic tests;
    explicitly passing ``None`` restores the monotonic clock.
    """
    global _MAX_ENTRIES, _STORE_DIR, _EXTRACTOR_CONTRACT_VERSION
    global _MODEL_IDENTITY, _RETRY_BASE_SECONDS, _RETRY_MAX_SECONDS, _CLOCK

    _MAX_ENTRIES = max(1, int(max_sessions))
    _STORE_DIR = Path(store_dir) if store_dir else None
    if extractor_contract_version is not None:
        _EXTRACTOR_CONTRACT_VERSION = (
            str(extractor_contract_version).strip() or _DEFAULT_CONTRACT_VERSION
        )
    if model_identity is not None:
        _MODEL_IDENTITY = str(model_identity).strip() or _DEFAULT_MODEL_IDENTITY
    if retry_base_seconds is not None:
        _RETRY_BASE_SECONDS = max(0.0, float(retry_base_seconds))
    if retry_max_seconds is not None:
        _RETRY_MAX_SECONDS = max(0.0, float(retry_max_seconds))
    if clock is not _UNSET:
        _CLOCK = time.monotonic if clock is None else clock  # type: ignore[assignment]


def _identity(
    contract_version: str | None = None,
    model_identity: str | None = None,
) -> tuple[str, str]:
    contract = (
        _EXTRACTOR_CONTRACT_VERSION
        if contract_version is None
        else str(contract_version).strip() or _DEFAULT_CONTRACT_VERSION
    )
    model = (
        _MODEL_IDENTITY
        if model_identity is None
        else str(model_identity).strip() or _DEFAULT_MODEL_IDENTITY
    )
    return contract, model


def _key(
    system_prompt: str,
    *,
    contract_version: str | None = None,
    model_identity: str | None = None,
) -> str:
    """Stable key over source prompt *and* the extractor contract/model identity."""
    contract, model = _identity(contract_version, model_identity)
    material = json.dumps(
        {
            "system_prompt": system_prompt or "",
            "extractor_contract_version": contract,
            "model_identity": model,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def _path(key: str) -> Path:
    if _STORE_DIR is None:
        raise RuntimeError("skills cache persistence is not configured")
    return _STORE_DIR / ("skills_" + key + ".json")


def _remember(key: str, entry: dict[str, Any]) -> None:
    """Insert/touch an entry while preserving the configured LRU bound."""
    _CACHE[key] = entry
    _CACHE.move_to_end(key)
    while len(_CACHE) > _MAX_ENTRIES:
        _CACHE.popitem(last=False)


def _load(
    key: str,
    *,
    system_prompt: str = "",
    contract_version: str | None = None,
    model_identity: str | None = None,
) -> None:
    """Hydrate a successful entry, accepting pre-versioned files safely.

    Old installations named files with ``sha256(system_prompt)[:16]`` and stored only
    ``{"summary": ...}``.  Such metadata-free data is eligible solely under the default
    legacy identity; an explicitly identified new model/contract must re-extract.
    """
    if key in _CACHE or not _STORE_DIR:
        return
    contract, model = _identity(contract_version, model_identity)
    try:
        candidates: list[tuple[Path, bool]] = [(_path(key), False)]
        if (
            system_prompt
            and contract == _DEFAULT_CONTRACT_VERSION
            and model == _DEFAULT_MODEL_IDENTITY
        ):
            old_key = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()[:16]
            if old_key != key:
                candidates.append((_path(old_key), True))

        for p, is_legacy_filename in candidates:
            if not p.exists():
                continue
            payload = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                continue
            # Failed/pending records are never trusted after restart.  Version-1 records
            # did not have a status and are ready only when they contain evidence.
            if "status" in payload and payload.get("status") != "ready":
                continue
            raw_summary = payload.get("summary")
            summary = str(raw_summary or "").strip()
            explicit_none = bool(payload.get("explicit_none")) or summary.upper() == "NONE"
            if summary.upper() == "NONE":
                summary = ""
            # Old code persisted failures as ready-empty.  Do not resurrect that poisoned
            # state; only an explicit NONE is a valid successful empty result.
            if not summary and not explicit_none:
                continue
            entry = {
                "status": "ready",
                "summary": summary,
                "explicit_none": explicit_none,
                "attempt_count": int(payload.get("attempt_count") or 0),
                "extractor_contract_version": payload.get(
                    "extractor_contract_version", contract
                ),
                "model_identity": payload.get("model_identity", model),
            }
            _remember(key, entry)
            if is_legacy_filename:
                _persist_ready(key, entry)
            return
    except Exception:
        # Persistence is an optimization.  A malformed/inaccessible record is a miss and
        # must never break request processing.
        return


def _persist_ready(key: str, entry: dict[str, Any]) -> None:
    """Best-effort atomic persistence for successful entries only."""
    if not _STORE_DIR or entry.get("status") != "ready":
        return
    tmp: Path | None = None
    try:
        _STORE_DIR.mkdir(parents=True, exist_ok=True)
        target = _path(key)
        tmp = target.with_name(f"{target.name}.{uuid.uuid4().hex}.tmp")
        payload = {
            "schema_version": _CACHE_SCHEMA_VERSION,
            "status": "ready",
            "summary": str(entry.get("summary") or ""),
            "explicit_none": bool(entry.get("explicit_none")),
            "attempt_count": int(entry.get("attempt_count") or 0),
            "extractor_contract_version": entry.get(
                "extractor_contract_version", _EXTRACTOR_CONTRACT_VERSION
            ),
            "model_identity": entry.get("model_identity", _MODEL_IDENTITY),
        }
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        tmp.replace(target)
    except Exception:
        return
    finally:
        if tmp is not None:
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass


def _store(key: str, entry: dict[str, Any]) -> None:
    """Set an entry, LRU-evict past the cap, and persist ready state atomically."""
    normalized = dict(entry)
    if normalized.get("status") == "ready":
        summary = str(normalized.get("summary") or "").strip()
        if summary.upper() == "NONE":
            summary = ""
            normalized["explicit_none"] = True
        normalized["summary"] = summary
        normalized.setdefault("explicit_none", False)
    _remember(key, normalized)
    _persist_ready(key, normalized)


def get_status(
    system_prompt: str,
    *,
    extractor_contract_version: str | None = None,
    model_identity: str | None = None,
) -> str:
    """Return one of ``missing``, ``pending``, ``ready`` or ``failed``."""
    if not system_prompt:
        return "missing"
    key = _key(
        system_prompt,
        contract_version=extractor_contract_version,
        model_identity=model_identity,
    )
    _load(
        key,
        system_prompt=system_prompt,
        contract_version=extractor_contract_version,
        model_identity=model_identity,
    )
    entry = _CACHE.get(key)
    if not entry:
        return "missing"
    _CACHE.move_to_end(key)
    status = str(entry.get("status") or "missing")
    return status if status in {"pending", "ready", "failed"} else "missing"


def get_summary(
    system_prompt: str,
    *,
    extractor_contract_version: str | None = None,
    model_identity: str | None = None,
) -> str:
    """Return a ready summary, or ``""`` for missing/pending/failed/explicit NONE."""
    if not system_prompt:
        return ""
    key = _key(
        system_prompt,
        contract_version=extractor_contract_version,
        model_identity=model_identity,
    )
    _load(
        key,
        system_prompt=system_prompt,
        contract_version=extractor_contract_version,
        model_identity=model_identity,
    )
    entry = _CACHE.get(key)
    if not entry or entry.get("status") != "ready":
        return ""
    _CACHE.move_to_end(key)
    if entry.get("explicit_none"):
        return ""
    summary = str(entry.get("summary") or "").strip()
    return "" if summary.upper() == "NONE" else summary


def _extract_system_prompt(request_body: dict[str, Any]) -> str:
    for msg in (request_body.get("messages") or []):
        if isinstance(msg, dict) and msg.get("role") == "system":
            return str(msg.get("content") or "")
    return ""


def summary_for_request(
    request_body: dict[str, Any],
    *,
    extractor_contract_version: str | None = None,
    model_identity: str | None = None,
) -> str:
    """Cached skills summary for this request's system prompt (or ``""``)."""
    return get_summary(
        _extract_system_prompt(request_body),
        extractor_contract_version=extractor_contract_version,
        model_identity=model_identity,
    )


def _failure_entry(
    *,
    previous_attempts: int,
    contract_version: str,
    model_identity: str,
    error_kind: str,
) -> dict[str, Any]:
    attempts = previous_attempts + 1
    exponent = min(max(0, attempts - 1), 30)
    delay = min(_RETRY_MAX_SECONDS, _RETRY_BASE_SECONDS * (2**exponent))
    return {
        "status": "failed",
        "summary": "",
        "explicit_none": False,
        "attempt_count": attempts,
        "next_retry_at": float(_CLOCK()) + delay,
        # Category only: do not persist or expose provider exception details/secrets.
        "last_error_kind": error_kind,
        "extractor_contract_version": contract_version,
        "model_identity": model_identity,
    }


def ensure_extraction(
    request_body: dict[str, Any],
    extractor: Callable[[str], "asyncio.Future[Any] | Any"],
    *,
    extractor_contract_version: str | None = None,
    model_identity: str | None = None,
) -> None:
    """Schedule one extraction unless ready, pending, or inside failure backoff.

    Existing callers may keep passing only ``request_body`` and ``extractor``.  Callers
    that swap the extraction prompt/schema or model can supply identity overrides here,
    or set stable defaults once via :func:`configure`.
    """
    system_prompt = _extract_system_prompt(request_body)
    if not system_prompt:
        return
    contract, model = _identity(extractor_contract_version, model_identity)
    key = _key(system_prompt, contract_version=contract, model_identity=model)
    _load(
        key,
        system_prompt=system_prompt,
        contract_version=contract,
        model_identity=model,
    )

    task = _IN_FLIGHT.get(key)
    if task is not None and not task.done():
        return

    current = _CACHE.get(key)
    if current:
        _CACHE.move_to_end(key)
        status = current.get("status")
        if status in {"ready", "pending"}:
            return
        if status == "failed" and float(current.get("next_retry_at") or 0.0) > float(_CLOCK()):
            return
    previous_attempts = int((current or {}).get("attempt_count") or 0)

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # Fire-and-forget requires a running loop.  Leave the state missing/failed so a
        # later request in the async proxy can schedule it.
        return

    _store(
        key,
        {
            "status": "pending",
            "summary": "",
            "explicit_none": False,
            "attempt_count": previous_attempts,
            "extractor_contract_version": contract,
            "model_identity": model,
        },
    )

    async def _run() -> None:
        try:
            value = extractor(system_prompt)
            if inspect.isawaitable(value):
                value = await value
            if not isinstance(value, str) or not value.strip():
                _store(
                    key,
                    _failure_entry(
                        previous_attempts=previous_attempts,
                        contract_version=contract,
                        model_identity=model,
                        error_kind="empty_output",
                    ),
                )
                return
            summary = value.strip()
            explicit_none = summary.upper() == "NONE"
            _store(
                key,
                {
                    "status": "ready",
                    "summary": "" if explicit_none else summary,
                    "explicit_none": explicit_none,
                    "attempt_count": previous_attempts,
                    "extractor_contract_version": contract,
                    "model_identity": model,
                },
            )
        except asyncio.CancelledError:
            _store(
                key,
                _failure_entry(
                    previous_attempts=previous_attempts,
                    contract_version=contract,
                    model_identity=model,
                    error_kind="cancelled",
                ),
            )
            raise
        except Exception:
            _store(
                key,
                _failure_entry(
                    previous_attempts=previous_attempts,
                    contract_version=contract,
                    model_identity=model,
                    error_kind="extractor_error",
                ),
            )
        finally:
            if _IN_FLIGHT.get(key) is asyncio.current_task():
                _IN_FLIGHT.pop(key, None)

    created = loop.create_task(_run())
    _IN_FLIGHT[key] = created
