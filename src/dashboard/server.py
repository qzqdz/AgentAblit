"""Live dashboard for TMI proxy JSONL traces."""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

STATIC_DIR = Path(__file__).resolve().parent / "static"

from proxy.session_snapshot import build_session_snapshot
from shared.model_client import extract_task_output_text
from shared.regi import result_metadata
from strategies.reconstruct.trajectory import content_fingerprint

TRACE_DIR = Path(
    os.environ.get("TMI_DASHBOARD_TRACE_DIR")
    or os.environ.get("TMI_PROXY_TRACE_DIR")
    # Default must match the proxy's write default (ProxyConfig.trace_dir =
    # outputs/proxy_traces) so the dashboard reads what the proxy writes without
    # needing any env override.
    or str(ROOT / "outputs" / "proxy_traces")
)
SESSION_DIR = Path(
    os.environ.get("PROXY_SESSION_DIR", str(ROOT / "outputs" / "sessions"))
)
# Persisted trajectory store (must match the proxy's TMI_TRAJ_STORE_DIR so the
# dashboard reads the same anchors the proxy writes).
TRAJ_STORE_DIR = Path(
    os.environ.get("TMI_TRAJ_STORE_DIR", str(ROOT / "outputs" / "trajectory_store"))
)
MAX_PREVIEW = int(os.environ.get("TMI_DASHBOARD_PREVIEW_CHARS", "600"))

# ---------------------------------------------------------------------------
# Cache infrastructure
# ---------------------------------------------------------------------------
_CACHE_DIR = Path(os.environ.get("TMI_DASHBOARD_CACHE_DIR", str(ROOT / "outputs" / ".dashboard_cache")))
_SUMMARIZE_CACHE_FILE = _CACHE_DIR / "summarize.json"
_CLUSTER_CACHE_FILE = _CACHE_DIR / "clusters.json"

# In-memory summarize cache: path -> (mtime, result)
# Loaded from disk at startup; written to disk on every _CACHE_WRITE_EVERY new entries.
_summarize_cache: dict[str, tuple[float, dict[str, Any]]] = {}
_summarize_cache_dirty: int = 0
_CACHE_WRITE_EVERY = 3  # flush to disk after this many new computed entries

# In-memory cluster cache: (fingerprint_str, clusters_list)
# fingerprint = JSON of sorted [(path, mtime), ...]; invalidated when any file changes.
_cluster_cache: tuple[str, list[dict[str, Any]]] | None = None


def _load_disk_cache() -> None:
    """Load persisted summarize and cluster caches from disk at startup."""
    global _summarize_cache, _cluster_cache
    if _SUMMARIZE_CACHE_FILE.exists():
        try:
            raw = json.loads(_SUMMARIZE_CACHE_FILE.read_text("utf-8"))
            _summarize_cache = {k: (float(v[0]), v[1]) for k, v in raw.items()}
        except Exception:
            _summarize_cache = {}
    if _CLUSTER_CACHE_FILE.exists():
        try:
            raw = json.loads(_CLUSTER_CACHE_FILE.read_text("utf-8"))
            fp = raw.get("fingerprint", "")
            clusters = raw.get("clusters")
            if fp and isinstance(clusters, list):
                _cluster_cache = (fp, clusters)
        except Exception:
            _cluster_cache = None


def _save_summarize_cache() -> None:
    """Atomically write the summarize cache to disk."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _SUMMARIZE_CACHE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(_summarize_cache, ensure_ascii=False), "utf-8")
    tmp.replace(_SUMMARIZE_CACHE_FILE)


def _save_cluster_cache(clusters: list[dict[str, Any]], fingerprint: str) -> None:
    """Atomically write the cluster cache to disk."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _CLUSTER_CACHE_FILE.with_suffix(".tmp")
    tmp.write_text(
        json.dumps({"fingerprint": fingerprint, "clusters": clusters}, ensure_ascii=False),
        "utf-8",
    )
    tmp.replace(_CLUSTER_CACHE_FILE)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    _load_disk_cache()
    yield
    _save_summarize_cache()


app = FastAPI(title="REGI Audit Dashboard", version="0.2.0", lifespan=_lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _safe_session_id(session_id: str) -> str:
    clean = "".join(ch for ch in session_id if ch.isalnum() or ch in "._-")
    if not clean or clean != session_id:
        raise HTTPException(status_code=400, detail="invalid session id")
    return clean


def _safe_agent_key(agent_key: str) -> str:
    clean = "".join(ch for ch in agent_key if ch.isalnum() or ch in "._:-")
    if not clean or clean != agent_key:
        raise HTTPException(status_code=400, detail="invalid agent key")
    return clean


def _load_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if not path.exists():
        return events
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                events.append({"event_type": "parse_error", "raw": line[:MAX_PREVIEW]})
    return events


def _preview(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, default=str)
    return text[:MAX_PREVIEW] + ("..." if len(text) > MAX_PREVIEW else "")


def _message_text(messages: Any, *, role: str | None = None, latest_only: bool = False) -> str:
    if not isinstance(messages, list):
        return ""
    parts: list[str] = []
    for item in messages:
        if not isinstance(item, dict):
            continue
        if role and item.get("role") != role:
            continue
        content = item.get("content")
        if isinstance(content, str):
            parts.append(content)
    if latest_only:
        return parts[-1] if parts else ""
    return " ".join(parts)


def _normalize_text(text: str) -> str:
    compact = " ".join(str(text or "").split())
    lowered = "".join(ch.lower() if "A" <= ch <= "Z" else ch for ch in compact)
    filtered = "".join(
        ch if ch.isalnum() or ch.isspace() or "\u4e00" <= ch <= "\u9fff" else " "
        for ch in lowered
    )
    return " ".join(filtered.split())


def _char_bigrams(text: str) -> list[str]:
    compact = "".join(ch for ch in str(text or "") if not ch.isspace())
    if len(compact) < 2:
        return [compact] if compact else []
    return [compact[idx : idx + 2] for idx in range(len(compact) - 1)]


def _extract_agent_meta(events: list[dict[str, Any]], session_id: str) -> dict[str, str]:
    for event in events:
        agent_key = event.get("agent_key")
        agent_source = str(event.get("agent_source") or "unknown")
        source_session_id = str(event.get("source_session_id") or session_id)
        if agent_key:
            return {
                "agent_key": str(agent_key),
                "agent_source": agent_source,
                "source_session_id": source_session_id,
            }
        if event.get("source_session_id"):
            return {
                "agent_key": f"{agent_source}:{source_session_id}",
                "agent_source": agent_source,
                "source_session_id": source_session_id,
            }
    return {
        "agent_key": session_id,
        "agent_source": "legacy",
        "source_session_id": session_id,
    }


def _extract_response_text(response_event: dict[str, Any] | None) -> str:
    if not response_event:
        return ""
    response = response_event.get("response") or {}
    choices = response.get("choices") or []
    if choices and isinstance(choices[0], dict):
        message = choices[0].get("message") or {}
        content = message.get("content") or message.get("reasoning_content")
        if isinstance(content, str):
            return content
    if isinstance(response, str):
        return response
    return json.dumps(response, ensure_ascii=False, default=str)


def _extract_response_preview(response_event: dict[str, Any] | None) -> str:
    return _preview(_extract_response_text(response_event))


def _latest_request_event(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    return next(
        (event for event in reversed(events) if event.get("event_type") == "request_received"),
        None,
    )


def _latest_response_event(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    return next(
        (
            event
            for event in reversed(events)
            if event.get("event_type") in {"response_completed", "response_stream_ended"}
        ),
        None,
    )


def _latest_injection_event(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    return next(
        (event for event in reversed(events) if event.get("event_type") == "tmi_injection"),
        None,
    )


def _latest_recover_event(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    return next(
        (event for event in reversed(events) if event.get("event_type") == "recover_completed"),
        None,
    )


def _latest_reconstruct_event(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    return next(
        (event for event in reversed(events) if event.get("event_type") == "reconstruct_completed"),
        None,
    )


def _build_swap_summary(reconstruct_event: dict[str, Any]) -> dict[str, Any]:
    """Extract reconstruct swap info from a reconstruct_completed trace event."""
    structural = reconstruct_event.get("structural") or {}
    sniffer = reconstruct_event.get("sniffer") or {}
    b_summary = reconstruct_event.get("b_summary") or {}
    path = str(reconstruct_event.get("path") or "passthrough")
    # REGI operation: prefer the additive event block; derive from legacy path otherwise.
    event_regi = reconstruct_event.get("regi")
    regi = (
        dict(event_regi)
        if isinstance(event_regi, dict) and event_regi.get("operation")
        else result_metadata(path)
    )
    return {
        "path": path,
        "regi": regi,
        "progress_source": str(reconstruct_event.get("progress_source") or ""),
        "swap_executed": bool(reconstruct_event.get("swap_executed", False)),
        "calibration_applied": bool(reconstruct_event.get("calibration_applied", False)),
        "a_continued": bool(reconstruct_event.get("a_continued", False)),
        "b_status": int(reconstruct_event.get("b_status") or 0),
        "failure": str(reconstruct_event.get("failure") or ""),
        # LLM sniffer verdict (the real path driver) + deterministic structural facts.
        "sniffer_ran": bool(sniffer.get("ran", False)),
        "sniffer_action": str(sniffer.get("action") or ""),
        "sniffer_rewrite_target": str(sniffer.get("rewrite_target") or ""),
        "has_tool_calls": bool(structural.get("has_tool_calls")),
        "b_summary": {
            "has_tool_calls": bool(b_summary.get("has_tool_calls")),
            "tool_names": list(b_summary.get("tool_names") or []),
            "content_preview": str(b_summary.get("content_preview") or ""),
        },
    }


def _regi_block_for(snapshot: dict[str, Any], events: list[dict[str, Any]]) -> dict[str, Any] | None:
    """REGI metadata for a session view, backfilled in memory for old data.

    Prefers the snapshot's persisted `regi` block; else derives from the latest completed
    event using the same shared mapping the proxy uses. Old snapshots on disk are never
    rewritten — this is presentation-only backfill.
    """
    persisted = snapshot.get("regi")
    if isinstance(persisted, dict) and persisted.get("operation"):
        return persisted
    reconstruct_evt = _latest_reconstruct_event(events)
    if reconstruct_evt:
        event_regi = reconstruct_evt.get("regi")
        if isinstance(event_regi, dict) and event_regi.get("operation"):
            return event_regi
        return result_metadata(str(reconstruct_evt.get("path") or "passthrough"))
    recover_evt = _latest_recover_event(events)
    if recover_evt:
        event_regi = recover_evt.get("regi")
        if isinstance(event_regi, dict) and event_regi.get("operation"):
            return event_regi
        if recover_evt.get("response_replaced"):
            return result_metadata("pass_flawed")
        if recover_evt.get("failure_stage") or recover_evt.get("failure"):
            return result_metadata("degraded_raw")
        return result_metadata("pass")
    return None


def _conv_key_for(events: list[dict[str, Any]]) -> str:
    """The proxy's conv_key for a session. Prefer the value the proxy RECORDED on its events
    (proxy is the single source of truth, incl. the x-conversation-id alias and the user-salt
    path the dashboard can't reproduce); fall back to the legacy recompute only for old traces
    written before the proxy stamped conv_key. See docs/SESSION_ISOLATION_PLAN.md."""
    for event in events:
        recorded = str(event.get("conv_key") or "").strip()
        if recorded:
            return recorded
    for event in events:
        cid = str(event.get("client_session_id") or "").strip()
        if cid:
            return cid
    req = _latest_request_event(events) or {}
    return content_fingerprint(req.get("messages") or [])


def _trajectory_anchor(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Read the persisted behavioral-trajectory summary anchored to this conversation
    (the async fire-and-forget落盘). conv_key is recomputed the same way the proxy keys it."""
    key = _conv_key_for(events)
    anchor: dict[str, Any] = {"conv_key": key, "found": False, "summary": "", "seen_count": 0}
    if not key:
        return anchor
    try:
        path = TRAJ_STORE_DIR / (hashlib.sha256(key.encode("utf-8")).hexdigest()[:32] + ".json")
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            anchor.update(
                found=True,
                summary=str(data.get("summary") or ""),
                seen_count=len(data.get("seen") or []),
            )
    except Exception:
        pass
    return anchor


def _latest_turn_analysis(events: list[dict[str, Any]], session_id: str) -> dict[str, Any]:
    request_event = _latest_request_event(events) or {}
    response_event = _latest_response_event(events) or {}
    injection_event = _latest_injection_event(events) or {}
    sniffer_event = _latest_recover_event(events) or {}
    reconstruct_event = _latest_reconstruct_event(events)
    response = response_event.get("response") or {}
    choices = response.get("choices") or []
    choice = choices[0] if choices and isinstance(choices[0], dict) else {}
    message = choice.get("message") or {}
    summary = sniffer_event.get("summary") or {}
    sniffer_response = dict(sniffer_event.get("response") or {})
    if "decision" in sniffer_response:
        sniffer_response["decision"] = extract_task_output_text(
            str(sniffer_response.get("decision") or "")
        )
    finish_reason = str(choice.get("finish_reason") or "")
    tool_calls_present = bool(message.get("tool_calls"))
    inferred_turn_stage = "intermediate" if tool_calls_present or finish_reason == "tool_calls" else "final"
    result: dict[str, Any] = {
        "session_id": session_id,
        "user_input": _message_text(request_event.get("messages"), role="user", latest_only=True),
        "assistant_response": {
            "content": str(message.get("content") or ""),
            "reasoning_content": str(message.get("reasoning_content") or ""),
            "finish_reason": finish_reason,
            "tool_calls_present": tool_calls_present,
        },
        "injection": {
            "applied": bool(injection_event),
            "original": str(injection_event.get("original") or ""),
            "injected": str(injection_event.get("injected") or ""),
            "backend": str(injection_event.get("backend") or ""),
        },
        "sniffer": {
            "turn_stage": str(sniffer_event.get("turn_stage") or inferred_turn_stage),
            "action": str(summary.get("action") or ""),
            "confidence": summary.get("confidence", 0.0),
            "rewrite_target": str(
                summary.get("rewrite_target") or summary.get("rewrite_brief") or ""
            ),
            "calibrator_applied": bool(sniffer_event.get("calibrator_applied", False)),
            "response_replaced": bool(sniffer_event.get("response_replaced", False)),
            "response": sniffer_response,
            "failure_stage": str(sniffer_event.get("failure_stage") or ""),
            "failure": str(sniffer_event.get("failure") or ""),
        },
    }
    result["trajectory"] = _trajectory_anchor(events)
    if reconstruct_event:
        result["swap"] = _build_swap_summary(reconstruct_event)
    # Additive REGI operation metadata for the latest completed turn.
    result["regi"] = _regi_block_for(result, events)
    return result


def _session_paths() -> list[Path]:
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(TRACE_DIR.glob("*.jsonl"), key=lambda path: path.stat().st_mtime, reverse=True)


def _session_snapshot_path(session_id: str) -> Path:
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    return SESSION_DIR / f"{session_id}.json"


def _read_session_snapshot(session_id: str) -> dict[str, Any] | None:
    path = _session_snapshot_path(session_id)
    if not path.exists():
        return None
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    snippet = snapshot.get("snippet") or {}
    sniffer = snippet.get("sniffer") or {}
    response = sniffer.get("response") or {}
    if isinstance(response, dict) and "decision" in response:
        response["decision"] = extract_task_output_text(
            str(response.get("decision") or "")
        )
    if "rewrite_target" not in sniffer:
        sniffer["rewrite_target"] = str(sniffer.get("rewrite_brief") or "")
    if "calibrator_applied" not in sniffer:
        sniffer["calibrator_applied"] = bool(
            sniffer.get("action") == "rewrite"
            and isinstance(response, dict)
            and str(response.get("decision") or "").strip()
        )
    if "response_replaced" not in sniffer:
        # Snapshots created before response replacement was implemented only
        # recorded the calibrated decision; they never delivered it to clients.
        sniffer["response_replaced"] = False
    return snapshot


def _build_and_store_session_snapshot(session_id: str) -> dict[str, Any]:
    trace_path = TRACE_DIR / f"{session_id}.jsonl"
    events = _load_events(trace_path)
    if not events:
        raise HTTPException(status_code=404, detail="session not found")
    latest_turn = _latest_turn_analysis(events, session_id)
    agent_meta = _extract_agent_meta(events, session_id)
    swap_failure = (latest_turn.get("swap") or {}).get("failure") or ""
    status = "failed" if (latest_turn["sniffer"]["failure_stage"] or swap_failure) else "ready"
    snapshot = build_session_snapshot(session_id, events, status=status, agent_meta=agent_meta)
    _session_snapshot_path(session_id).write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return snapshot


def _summarize(path: Path) -> dict[str, Any]:
    global _summarize_cache_dirty
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    key = str(path)
    cached_mtime, cached_result = _summarize_cache.get(key, (None, None))
    if cached_mtime is not None and cached_mtime == mtime and cached_result is not None:
        return cached_result
    result = _summarize_uncached(path)
    _summarize_cache[key] = (mtime, result)
    _summarize_cache_dirty += 1
    if _summarize_cache_dirty >= _CACHE_WRITE_EVERY:
        _save_summarize_cache()
        _summarize_cache_dirty = 0
    return result


def _summarize_uncached(path: Path) -> dict[str, Any]:
    events = _load_events(path)
    injections = [event for event in events if event.get("event_type") == "tmi_injection"]
    responses = [
        event
        for event in events
        if event.get("event_type") in {"response_completed", "response_error", "response_stream_ended"}
    ]
    request = next((event for event in events if event.get("event_type") == "request_received"), {})
    last = events[-1] if events else {}
    agent_meta = _extract_agent_meta(events, path.stem)
    has_predictor_debug = any(event.get("event_type") == "v1_3_completed" and event.get("audit") for event in events)
    request_messages = request.get("messages") or []
    conversational_messages = [
        item
        for item in request_messages
        if isinstance(item, dict) and item.get("role") != "system"
    ]
    user_message_count = sum(
        1 for item in request_messages if isinstance(item, dict) and item.get("role") == "user"
    )
    assistant_message_count = sum(
        1 for item in request_messages if isinstance(item, dict) and item.get("role") == "assistant"
    )
    tool_message_count = sum(
        1 for item in request_messages if isinstance(item, dict) and item.get("role") == "tool"
    )
    has_prior_conversation = (
        user_message_count > 1
        or assistant_message_count > 0
        or tool_message_count > 0
    )
    request_text = _message_text(conversational_messages)
    latest_user_text = _message_text(request_messages, role="user", latest_only=True)
    response_text = _extract_response_text(responses[-1] if responses else None)
    response_preview = _preview(response_text)
    combined_text = " ".join(part for part in [request_text, response_text] if part)
    return {
        "session_id": path.stem,
        "agent_key": agent_meta["agent_key"],
        "agent_source": agent_meta["agent_source"],
        "source_session_id": agent_meta["source_session_id"],
        "message_count": len([item for item in request_messages if isinstance(item, dict)]),
        "user_message_count": user_message_count,
        "assistant_message_count": assistant_message_count,
        "tool_message_count": tool_message_count,
        "has_prior_conversation": has_prior_conversation,
        "session_total_chars": len(combined_text),
        "latest_user_chars": len(latest_user_text),
        "normalized_user_text": _normalize_text(latest_user_text),
        "normalized_request_text": _normalize_text(request_text),
        "normalized_session_text": _normalize_text(combined_text),
        "event_count": len(events),
        "has_injection": bool(injections),
        "injection_count": len(injections),
        "has_predictor_debug": has_predictor_debug,
        "last_event_type": last.get("event_type", ""),
        "model": request.get("model") or (responses[-1].get("model") if responses else ""),
        "updated_epoch": path.stat().st_mtime,
        "updated": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(path.stat().st_mtime)),
        "request_preview": _preview(request.get("messages")),
        "response_preview": response_preview,
    }


def _is_cluster_eligible(item: dict[str, Any]) -> bool:
    return (
        int(item.get("session_total_chars") or 0) >= 40
        or int(item.get("latest_user_chars") or 0) >= 20
        or int(item.get("message_count") or 0) >= 3
    )


def _char_bigram_overlap_score(left: str, right: str) -> float:
    left_bigrams = _char_bigrams(left)
    right_bigrams = _char_bigrams(right)
    if not left_bigrams or not right_bigrams:
        return 0.0
    left_counts = Counter(left_bigrams)
    right_counts = Counter(right_bigrams)
    shared = sum(min(left_counts[token], right_counts[token]) for token in left_counts.keys() & right_counts.keys())
    return (2.0 * shared) / (len(left_bigrams) + len(right_bigrams))


def _time_proximity_score(left_epoch: float, right_epoch: float) -> float:
    delta = abs(left_epoch - right_epoch)
    if delta <= 15 * 60:
        return 1.0 - (delta / (15 * 60))
    return 0.0


def _contains_with_growth(base: str, candidate: str) -> bool:
    if not base or not candidate or base == candidate:
        return False
    if base not in candidate:
        return False
    added = len(candidate) - len(base)
    return added >= 8 and added <= max(24, int(len(candidate) * 0.45))


def _matches_hard_containment(candidate: dict[str, Any], cluster: dict[str, Any]) -> bool:
    anchor_text = str(cluster.get("anchor_text") or "")
    candidate_text = str(candidate.get("normalized_session_text") or "")
    anchor_request = str(cluster.get("anchor_request_text") or "")
    candidate_request = str(candidate.get("normalized_request_text") or "")
    return _contains_with_growth(anchor_text, candidate_text) or (
        anchor_request
        and candidate_request
        and anchor_request in candidate_request
        and len(candidate_request) > len(anchor_request)
    )


def _matches_prefix_extension(candidate: dict[str, Any], cluster: dict[str, Any]) -> bool:
    anchor_text = str(cluster.get("anchor_text") or "")
    candidate_text = str(candidate.get("normalized_session_text") or "")
    anchor_request = str(cluster.get("anchor_request_text") or "")
    candidate_request = str(candidate.get("normalized_request_text") or "")
    if anchor_request and candidate_request and candidate_request.startswith(anchor_request) and candidate_request != anchor_request:
        added_request = len(candidate_request) - len(anchor_request)
        if added_request >= 8:
            return True
    if not anchor_text or not candidate_text or anchor_text == candidate_text:
        return False
    if not candidate_text.startswith(anchor_text):
        return False
    added = len(candidate_text) - len(anchor_text)
    return added >= 8 and added <= max(28, int(len(candidate_text) * 0.50))


def _matches_user_continuation(candidate: dict[str, Any], cluster: dict[str, Any]) -> bool:
    candidate_user = str(candidate.get("normalized_user_text") or "")
    if not candidate_user:
        return False
    recent_chain = list(cluster.get("recent_user_chain") or [])
    if not recent_chain:
        return False
    best = max(_char_bigram_overlap_score(candidate_user, text) for text in recent_chain if text)
    if best < 0.72:
        return False
    latest_cluster_size = max(int(session.get("session_total_chars") or 0) for session in cluster.get("sessions") or [{}])
    return int(candidate.get("session_total_chars") or 0) >= latest_cluster_size


def _within_adsorption_window(candidate: dict[str, Any], cluster: dict[str, Any], strength: str) -> bool:
    candidate_epoch = float(candidate.get("updated_epoch") or 0.0)
    cluster_epoch = float(cluster.get("updated_epoch") or 0.0)
    delta = abs(candidate_epoch - cluster_epoch)
    if strength == "hard":
        return delta <= 12 * 60 * 60
    if strength == "soft":
        return delta <= 90 * 60
    if strength == "fallback":
        return delta <= 30 * 60
    return False


def _has_contextual_chain_evidence(item: dict[str, Any]) -> bool:
    return (
        bool(str(item.get("response_preview") or "").strip())
        or int(item.get("message_count") or 0) >= 2
        or int(item.get("event_count") or 0) >= 2
    )


def _has_prior_conversation(item: dict[str, Any]) -> bool:
    # Real dashboard summaries always carry this field. Treat older/manual summary
    # dictionaries as unknown instead of silently classifying them as first-turn.
    if "has_prior_conversation" not in item:
        return True
    return bool(item.get("has_prior_conversation"))


def _can_use_heuristic_adsorption(candidate: dict[str, Any], cluster: dict[str, Any]) -> bool:
    if not _has_contextual_chain_evidence(candidate):
        return False
    sessions = list(cluster.get("sessions") or [])
    if not any(_has_contextual_chain_evidence(session) for session in sessions):
        return False

    # Content overlap can support an already-observed conversation, but it cannot
    # manufacture lineage between two unrelated first turns. A later request with
    # actual assistant/tool history may still attach to an earlier first-turn anchor.
    if not _has_prior_conversation(candidate) and not any(
        _has_prior_conversation(session) for session in sessions
    ):
        return False
    return True


def _pair_similarity(left: dict[str, Any], right: dict[str, Any]) -> float:
    request_score = _char_bigram_overlap_score(
        str(left.get("normalized_request_text") or ""),
        str(right.get("normalized_request_text") or ""),
    )
    session_score = _char_bigram_overlap_score(
        str(left.get("normalized_session_text") or ""),
        str(right.get("normalized_session_text") or ""),
    )
    text_score = max(request_score, session_score)
    user_score = _char_bigram_overlap_score(
        str(left.get("normalized_user_text") or ""),
        str(right.get("normalized_user_text") or ""),
    )
    time_score = _time_proximity_score(
        float(left.get("updated_epoch") or 0.0),
        float(right.get("updated_epoch") or 0.0),
    )
    same_source = (
        left.get("model") == right.get("model")
        or left.get("agent_source") == right.get("agent_source")
    )
    source_score = 1.0 if same_source else 0.0
    return (0.50 * text_score) + (0.25 * user_score) + (0.15 * time_score) + (0.10 * source_score)


def _progressive_continuation(candidate: dict[str, Any], cluster: dict[str, Any]) -> bool:
    representative = str(cluster.get("representative_text") or "")
    candidate_text = str(candidate.get("normalized_session_text") or "")
    if representative and representative in candidate_text:
        added = max(len(candidate_text) - len(representative), 0)
        if added <= int(len(candidate_text) * 0.35):
            return True

    chain = list(cluster.get("sessions") or [])[:2]
    texts = [candidate, *chain]
    if len(texts) < 3:
        return False
    return (
        _pair_similarity(texts[0], texts[1]) >= 0.60
        and _pair_similarity(texts[1], texts[2]) >= 0.60
        and abs(float(texts[0]["updated_epoch"]) - float(texts[1]["updated_epoch"])) <= 15 * 60
        and abs(float(texts[1]["updated_epoch"]) - float(texts[2]["updated_epoch"])) <= 15 * 60
        and int(texts[0]["session_total_chars"]) >= int(texts[1]["session_total_chars"]) >= int(texts[2]["session_total_chars"])
    )


def _build_cluster(
    cluster_key: str,
    sessions: list[dict[str, Any]],
    cluster_mode: str,
    match_reason: str,
) -> dict[str, Any]:
    ordered = sorted(sessions, key=lambda item: item["updated_epoch"], reverse=True)
    latest = ordered[0]
    anchor = max(
        ordered,
        key=lambda item: (
            int(item.get("session_total_chars") or 0),
            int(item.get("message_count") or 0),
            float(item.get("updated_epoch") or 0.0),
        ),
    )
    alias_key = cluster_key if cluster_mode in {"heuristic", "single"} else (latest.get("agent_key") or cluster_key)
    return {
        "cluster_key": cluster_key,
        "cluster_mode": cluster_mode,
        "match_reason": match_reason,
        "latest_session_id": latest["session_id"],
        "agent_key": alias_key,
        "agent_source": latest.get("agent_source") or "unknown",
        "session_count": len(ordered),
        "event_count": sum(item["event_count"] for item in ordered),
        "injection_count": sum(item["injection_count"] for item in ordered),
        "has_injection": any(item["has_injection"] for item in ordered),
        "has_predictor_debug": any(item.get("has_predictor_debug") for item in ordered),
        "updated_epoch": latest["updated_epoch"],
        "updated": latest["updated"],
        "model": latest.get("model") or "",
        "representative_text": latest.get("normalized_session_text") or "",
        "anchor_session_id": anchor["session_id"],
        "anchor_text": str(anchor.get("normalized_session_text") or ""),
        "anchor_request_text": str(anchor.get("normalized_request_text") or ""),
        "recent_user_chain": [
            str(item.get("normalized_user_text") or "")
            for item in ordered[:3]
            if str(item.get("normalized_user_text") or "")
        ],
        "history_depth": len(ordered),
        "sessions": ordered,
    }


def _find_history_adsorption_target(candidate: dict[str, Any], clusters: list[dict[str, Any]]) -> tuple[int, str] | None:
    best_match: tuple[int, str, float] | None = None
    for idx, cluster in enumerate(clusters):
        if cluster["cluster_mode"] != "heuristic":
            continue
        if not _can_use_heuristic_adsorption(candidate, cluster):
            continue
        cluster_epoch = float(cluster.get("updated_epoch") or 0.0)
        if _matches_prefix_extension(candidate, cluster) and _within_adsorption_window(candidate, cluster, "hard"):
            return idx, "prefix-extension"
        if _matches_hard_containment(candidate, cluster) and _within_adsorption_window(candidate, cluster, "hard"):
            return idx, "hard-containment"
        if _matches_user_continuation(candidate, cluster) and _within_adsorption_window(candidate, cluster, "soft"):
            if best_match is None or cluster_epoch > best_match[2]:
                best_match = (idx, "user-continuation", cluster_epoch)
        elif _progressive_continuation(candidate, cluster) and _within_adsorption_window(candidate, cluster, "soft"):
            if best_match is None or cluster_epoch > best_match[2]:
                best_match = (idx, "progressive-chain", cluster_epoch)
    if best_match is None:
        return None
    return best_match[0], best_match[1]


def _fallback_similarity(candidate: dict[str, Any], cluster: dict[str, Any]) -> float:
    sessions = list(cluster.get("sessions") or [])
    if not sessions:
        return 0.0
    return _pair_similarity(candidate, sessions[0])


def _find_fallback_similarity_target(candidate: dict[str, Any], clusters: list[dict[str, Any]]) -> int | None:
    best_index = -1
    best_score = 0.0
    best_updated_epoch = float("-inf")
    for idx, cluster in enumerate(clusters):
        if cluster["cluster_mode"] != "heuristic":
            continue
        if not _can_use_heuristic_adsorption(candidate, cluster):
            continue
        if not _within_adsorption_window(candidate, cluster, "fallback"):
            continue
        score = _fallback_similarity(candidate, cluster)
        cluster_updated_epoch = float(cluster.get("updated_epoch") or 0.0)
        if score >= 0.72 and (score > best_score or (score == best_score and cluster_updated_epoch > best_updated_epoch)):
            best_index = idx
            best_score = score
            best_updated_epoch = cluster_updated_epoch
    return best_index if best_index >= 0 else None


def _group_clusters(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stable_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    fragmented: list[dict[str, Any]] = []
    for item in items:
        stable_key = item.get("agent_key") or item["session_id"]
        if stable_key and stable_key != item["session_id"]:
            stable_groups[str(stable_key)].append(item)
        else:
            fragmented.append(item)

    clusters = [
        _build_cluster(cluster_key, sessions, "stable", "stable-id")
        for cluster_key, sessions in stable_groups.items()
    ]
    for item in sorted(fragmented, key=lambda value: value["updated_epoch"]):
        if not _is_cluster_eligible(item):
            clusters.append(
                _build_cluster(
                    f"single:{item['session_id']}",
                    [item],
                    "single",
                    "short-session",
                )
            )
            continue

        history_target = _find_history_adsorption_target(item, clusters)
        if history_target is not None:
            best_index, best_reason = history_target
            merged_sessions = [item, *clusters[best_index]["sessions"]]
            clusters[best_index] = _build_cluster(
                clusters[best_index]["cluster_key"],
                merged_sessions,
                "heuristic",
                best_reason,
            )
            continue

        fallback_index = _find_fallback_similarity_target(item, clusters)
        if fallback_index is not None:
            merged_sessions = [item, *clusters[fallback_index]["sessions"]]
            clusters[fallback_index] = _build_cluster(
                clusters[fallback_index]["cluster_key"],
                merged_sessions,
                "heuristic",
                "fallback-similarity",
            )
            continue

        clusters.append(
            _build_cluster(
                f"heuristic:{item['session_id']}",
                [item],
                "heuristic",
                "new-chain",
            )
        )

    return sorted(clusters, key=lambda item: item["updated_epoch"], reverse=True)


def _compute_fingerprint(paths: list[Path]) -> str:
    """Stable fingerprint of the current trace dir state: sorted (path, mtime) pairs."""
    items: list[tuple[str, float]] = []
    for path in sorted(paths, key=str):
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = 0.0
        items.append((str(path), mtime))
    return json.dumps(items)


def _get_all_cached() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (summaries, clusters), skipping cluster recomputation when nothing changed.

    Fingerprint = sorted (path, mtime) list. Cache hit when fingerprint matches.
    Summaries are always fast (individual mtime-keyed cache).
    """
    global _cluster_cache
    paths = _session_paths()
    fingerprint = _compute_fingerprint(paths)
    summaries = [_summarize(path) for path in paths]
    if _cluster_cache is not None and _cluster_cache[0] == fingerprint:
        return summaries, _cluster_cache[1]
    clusters = _group_clusters(summaries)
    _cluster_cache = (fingerprint, clusters)
    _save_cluster_cache(clusters, fingerprint)
    return summaries, clusters


def _cluster_detail_payload(cluster_key: str, selected_session_id: str | None = None) -> dict[str, Any]:
    summaries, groups = _get_all_cached()
    clean = _safe_agent_key(cluster_key)
    group = next((item for item in groups if item["cluster_key"] == clean), None)
    if group is None:
        matches = [item for item in groups if item.get("agent_key") == clean]
        if len(matches) == 1:
            group = matches[0]
    if group is None:
        raise HTTPException(status_code=404, detail="cluster not found")

    merged_events: list[dict[str, Any]] = []
    for session_summary in group["sessions"]:
        path = TRACE_DIR / f"{session_summary['session_id']}.jsonl"
        for event in _load_events(path):
            merged_events.append(
                {
                    **event,
                    "__session_id": session_summary["session_id"],
                    "__agent_key": group["agent_key"],
                    "__cluster_key": group["cluster_key"],
                }
            )

    merged_events.sort(key=lambda event: (str(event.get("ts") or ""), str(event.get("__session_id") or "")))
    sessions = list(group.get("sessions") or [])
    selected = selected_session_id or group.get("latest_session_id") or (sessions[0]["session_id"] if sessions else "")
    if selected and not any(item.get("session_id") == selected for item in sessions):
        selected = group.get("latest_session_id") or (sessions[0]["session_id"] if sessions else "")
    filtered_events = [event for event in merged_events if event.get("__session_id") == selected]
    latest_turn = _read_session_snapshot(str(selected or "")) or _latest_turn_analysis(
        filtered_events,
        str(selected or ""),
    )
    # Always attach the live persisted-trajectory anchor (stored snapshots predate it).
    latest_turn = {**latest_turn, "trajectory": _trajectory_anchor(filtered_events)}
    # Old stored snapshots may lack the additive REGI block — backfill in memory only.
    if "regi" not in latest_turn or not latest_turn.get("regi"):
        latest_turn = {**latest_turn, "regi": _regi_block_for(latest_turn, filtered_events)}
    return {
        "agent": group,
        "cluster": group,
        "selected_session_id": selected,
        "latest_turn": latest_turn,
        "events": merged_events,
    }


@app.get("/health")
def health() -> dict[str, Any]:
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    return {
        "status": "ok",
        "trace_dir": str(TRACE_DIR),
        "session_count": len(list(TRACE_DIR.glob("*.jsonl"))),
    }


@app.get("/api/sessions")
def sessions() -> dict[str, Any]:
    items, clusters = _get_all_cached()
    return {"clusters": clusters, "agents": clusters, "sessions": items}


@app.get("/api/sessions/{session_id}")
def session(session_id: str) -> dict[str, Any]:
    clean = _safe_session_id(session_id)
    snapshot = _read_session_snapshot(clean)
    if snapshot is not None:
        # Old stored snapshots may lack the additive REGI block — backfill in memory only
        # (never rewrite provenance on disk just because a viewer read it).
        if "regi" not in snapshot or not snapshot.get("regi"):
            events = _load_events(TRACE_DIR / f"{clean}.jsonl")
            snapshot = {**snapshot, "regi": _regi_block_for(snapshot, events)}
        return snapshot
    return _build_and_store_session_snapshot(clean)


@app.get("/api/agents/{agent_key}")
def agent_detail(agent_key: str, session_id: str | None = Query(default=None)) -> dict[str, Any]:
    return _cluster_detail_payload(agent_key, session_id)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Serve the static dashboard shell (frontend lives under static/)."""
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("TMI_DASHBOARD_PORT", "8788"))
    uvicorn.run(app, host="127.0.0.1", port=port)
