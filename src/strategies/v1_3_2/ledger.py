"""Hybrid V2 deterministic fact layer (L5-L7).

``build_ledger`` is a lossless, idempotent normalization of the OpenAI message graph.  It
never summarizes, never looks up a tool result globally, and never turns missing evidence into
success.  ``build_checkpoint`` derives evidence-linked state from a continuous closed prefix;
the latest user turn is excluded by default.  L6 helpers build hard value-flow edges without an
LLM so the context planner can recall old producer units recursively.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any


_ID_PATTERNS: list[tuple[str, "re.Pattern[str]"]] = [
    ("url", re.compile(r"https?://[^\s'\"<>)\]]+")),
    ("uuid", re.compile(
        r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
        r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
    )),
    ("path", re.compile(r"(?:/[A-Za-z0-9_.\-]+){2,}/?")),
    ("winpath", re.compile(r"[A-Za-z]:\\[^\s'\"<>|]+")),
    ("hexid", re.compile(r"\b[0-9a-fA-F]{16,}\b")),
    ("token", re.compile(r"\b[A-Za-z0-9_\-]{20,}\b")),
]
_FAIL_MARKERS = (
    "error", "failed", "failure", "denied", "refused", "rejected", "timeout",
    "not found", "traceback", "exception", "错误", "失败", "拒绝", "超时", "无法",
    "不存在",
)
_OK_STATUS = {"ok", "success", "succeeded", "complete", "completed", "done"}
_FAIL_STATUS = {"error", "failed", "failure", "denied", "rejected", "timeout"}


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def _json_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError):
        return str(value)


def _message_id(index: int, message: dict[str, Any]) -> str:
    try:
        raw = json.dumps(message, ensure_ascii=False, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError):
        raw = repr(message)
    return _stable_id("m", str(index), raw)


def _typed_arguments(raw: Any) -> tuple[dict[str, Any], bool]:
    """Return a typed object and whether native replay would be faithful."""
    if isinstance(raw, dict):
        return raw, True
    if isinstance(raw, str):
        if not raw.strip():
            return {}, True
        try:
            value = json.loads(raw, parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {token}")
            ))
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}, False
        return (value, True) if isinstance(value, dict) else ({}, False)
    if raw is None:
        return {}, True
    return {}, False


def _status_of(content: Any) -> str:
    """Classify explicit structured status before falling back to conservative prose markers."""
    text = _json_text(content).strip()
    if not text:
        return "ok"
    try:
        value = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        value = None
    if isinstance(value, dict):
        # Process-style tools often expose a null ``error`` channel even when the command
        # itself failed.  A non-zero exit status is stronger evidence than that empty channel.
        for key in ("exit_code", "exitCode", "return_code", "returncode"):
            exit_code = value.get(key)
            if isinstance(exit_code, bool):
                continue
            if isinstance(exit_code, (int, float)) and math.isfinite(exit_code):
                if exit_code != 0:
                    return "error"
                break
            if isinstance(exit_code, str) and re.fullmatch(r"[+-]?\d+", exit_code.strip()):
                if int(exit_code) != 0:
                    return "error"
                break
        for key in ("ok", "success", "succeeded"):
            if isinstance(value.get(key), bool):
                return "ok" if value[key] else "error"
        status = value.get("status")
        if isinstance(status, str):
            normalized = status.strip().lower()
            if normalized in _OK_STATUS:
                return "ok"
            if normalized in _FAIL_STATUS:
                return "error"
        error = value.get("error")
        if error not in (None, "", False, [], {}):
            return "error"
        errors = value.get("errors")
        if errors not in (None, "", False, [], {}):
            return "error"
        # Structured success payloads often mention errors in descriptive fields.  Once an
        # explicit empty error channel is present, words inside other fields are not status.
        if "error" in value or "errors" in value:
            return "ok"
    lowered = text.lower()
    return "error" if any(marker in lowered for marker in _FAIL_MARKERS) else "ok"


def _extract_ids(text: str) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    seen: set[str] = set()
    for kind, pattern in _ID_PATTERNS:
        for match in pattern.findall(text or ""):
            value = match.rstrip(".,;:)")
            # Extremely long alphanumeric stdout blobs are payload, not useful identifiers.
            if 3 <= len(value) <= 512 and value not in seen:
                seen.add(value)
                found.append((kind, value))
    return found


def _structured_values(content: Any) -> dict[str, Any]:
    text = _json_text(content).strip()
    try:
        value = json.loads(text) if text else None
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    if not isinstance(value, (dict, list)):
        return {}
    flat: dict[str, Any] = {}

    def walk(node: Any, locator: str) -> None:
        if isinstance(node, dict):
            for key, child in node.items():
                walk(child, f"{locator}.{key}" if locator else str(key))
        elif isinstance(node, list):
            for index, child in enumerate(node):
                walk(child, f"{locator}[{index}]")
        elif isinstance(node, (str, int, float, bool)) or node is None:
            if locator and not (isinstance(node, float) and not math.isfinite(node)):
                flat[locator] = node

    walk(value, "")
    return flat


def _new_unit(turn_id: int, user_anchor: str, user_message_id: str) -> dict[str, Any]:
    return {
        "unit_id": "",
        "turn_id": turn_id,
        "source_message_ids": [user_message_id] if user_message_id else [],
        "user_anchor": user_anchor,
        "calls": [],
        "observations": [],
        "orphan_observations": [],
        "assistant_text": "",
        "source_hash": "",
        "ledger_revision": "",
    }


def build_ledger(messages: list[dict]) -> list[dict]:
    """Normalize messages in source order into immutable owner ActionUnits.

    Results are matched only to a pending call in the current owner unit.  Reusing a call ID in
    a later unit therefore cannot overwrite an earlier observation.  Missing results remain
    explicit ``pending`` observations and are never rendered as synthetic tool messages.
    """
    units: list[dict] = []
    turn_id = 0
    user_anchor = ""
    user_message_id = ""
    current: dict[str, Any] | None = None
    revision_parts: list[str] = []

    def close(unit: dict[str, Any] | None) -> None:
        if unit is None:
            return
        payload = {
            key: unit[key]
            for key in (
                "turn_id", "source_message_ids", "user_anchor", "calls", "observations",
                "orphan_observations", "assistant_text",
            )
        }
        source = json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False)
        unit["source_hash"] = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
        unit["unit_id"] = _stable_id("u", str(unit["turn_id"]), unit["source_hash"])
        revision_parts.append(unit["source_hash"])
        unit["ledger_revision"] = _stable_id("lr", *revision_parts)
        units.append(unit)

    for index, message in enumerate(messages or []):
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        source_id = _message_id(index, message)
        if role == "system":
            continue
        if role == "user":
            close(current)
            current = None
            turn_id += 1
            user_anchor = _json_text(message.get("content")).strip()
            user_message_id = source_id
            continue
        if role == "assistant":
            calls = message.get("tool_calls") or []
            text = _json_text(message.get("content")).strip()
            if calls:
                close(current)
                current = _new_unit(turn_id, user_anchor, user_message_id)
                current["source_message_ids"].append(source_id)
                current["assistant_text"] = text
                for call_index, tool_call in enumerate(calls):
                    if not isinstance(tool_call, dict):
                        continue
                    function = tool_call.get("function")
                    function = function if isinstance(function, dict) else {}
                    raw_arguments = function.get("arguments")
                    arguments, faithful = _typed_arguments(raw_arguments)
                    call_id = tool_call.get("id")
                    call_id = call_id if isinstance(call_id, str) else ""
                    current["calls"].append({
                        "call_index": call_index,
                        "call_id": call_id,
                        "call_type": tool_call.get("type"),
                        "tool_name": str(function.get("name") or ""),
                        "arguments": arguments,
                        "raw_arguments": raw_arguments,
                        "args_faithful": faithful,
                        "source_message_id": source_id,
                    })
                    current["observations"].append({
                        "call_index": call_index,
                        "tool_call_id": call_id,
                        "status": "pending",
                        "content": None,
                        "raw_content": None,
                        "structured_values": {},
                        "error": None,
                        "source_message_id": None,
                    })
            elif text:
                if current is None:
                    current = _new_unit(turn_id, user_anchor, user_message_id)
                current["assistant_text"] = text
                current["source_message_ids"].append(source_id)
            continue
        if role == "tool":
            raw_content = message.get("content")
            content = _json_text(raw_content)
            call_id = message.get("tool_call_id")
            call_id = call_id if isinstance(call_id, str) else ""
            target = None
            if current is not None:
                target = next((
                    observation for observation in current["observations"]
                    if observation["status"] == "pending"
                    and observation["tool_call_id"] == call_id
                ), None)
            observation = {
                "call_index": target.get("call_index") if target else None,
                "tool_call_id": call_id,
                "status": _status_of(raw_content),
                "content": content,
                "raw_content": raw_content,
                "structured_values": _structured_values(raw_content),
                "error": content if _status_of(raw_content) == "error" else None,
                "source_message_id": source_id,
            }
            if target is not None:
                target.update(observation)
                current["source_message_ids"].append(source_id)
            elif current is not None:
                current["orphan_observations"].append(observation)
                current["source_message_ids"].append(source_id)

    close(current)
    return units


def _closed_units(units: list[dict], *, include_current: bool) -> list[dict]:
    if include_current or not units:
        return list(units)
    current_turn = max(int(unit.get("turn_id") or 0) for unit in units)
    return [unit for unit in units if int(unit.get("turn_id") or 0) != current_turn]


def _call_observation_pairs(unit: dict) -> list[tuple[dict, dict]]:
    observations = {o.get("call_index"): o for o in unit.get("observations") or []}
    return [
        (call, observations.get(call.get("call_index"), {
            "status": "pending", "content": None, "structured_values": {}, "error": None,
        }))
        for call in unit.get("calls") or []
    ]


def build_checkpoint(
    units: list[dict], messages: list[dict], *, include_current: bool = False,
) -> dict:
    """Build evidence-grounded deterministic state over a continuous unit prefix."""
    covered = _closed_units(units, include_current=include_current)
    users = [
        _json_text(message.get("content")).strip()
        for message in messages or []
        if isinstance(message, dict) and message.get("role") == "user"
    ]
    objective = {
        "initial": users[0] if users else "",
        "current": users[-1] if users else "",
    }
    live_artifacts: list[dict] = []
    completed: list[dict] = []
    failed: list[dict] = []
    unresolved: list[dict] = []
    seen_facts: set[str] = set()

    def add_artifact(unit: dict, call: dict, kind: str, value: Any, locator: str) -> None:
        try:
            identity = json.dumps([kind, value], ensure_ascii=False, sort_keys=True,
                                  allow_nan=False)
        except (TypeError, ValueError):
            return
        if identity in seen_facts:
            return
        seen_facts.add(identity)
        evidence = {
            "unit_id": unit["unit_id"], "locator": locator, "exact_span_or_value": value,
        }
        live_artifacts.append({
            "fact_id": _stable_id("fact", unit["unit_id"], locator, identity),
            "kind": kind,
            "name": locator.rsplit(".", 1)[-1] if kind == "field" else None,
            "value": value,
            "typed_value": value,
            "status": "active",
            "unit_id": unit["unit_id"],
            "tool": call.get("tool_name") or "",
            "evidence": [evidence],
        })

    for unit in covered:
        for call, observation in _call_observation_pairs(unit):
            entry = {
                "tool": call.get("tool_name") or "",
                "call_id": call.get("call_id") or "",
                "unit_id": unit["unit_id"],
                "arguments": call.get("arguments") if call.get("args_faithful") else None,
                "raw_arguments": call.get("raw_arguments"),
                "status": observation.get("status") or "pending",
                "evidence": [{
                    "unit_id": unit["unit_id"],
                    "locator": f"observations[{call.get('call_index', 0)}]",
                    "exact_span_or_value": observation.get("content"),
                }],
            }
            status = observation.get("status")
            if status == "pending":
                unresolved.append(entry)
                continue
            if status == "error":
                failed.append({**entry, "error": observation.get("error")})
                # Failed inputs are evidence of a dead end, not produced live artifacts.
                continue
            completed.append(entry)
            # Successful call arguments may identify an updated artifact even when the tool
            # only returned "ok".  Failed calls were excluded above.
            if call.get("args_faithful"):
                argument_text = json.dumps(call.get("arguments") or {}, ensure_ascii=False,
                                           sort_keys=True, allow_nan=False)
                for kind, value in _extract_ids(argument_text):
                    add_artifact(unit, call, kind, value,
                                 f"calls[{call.get('call_index', 0)}].arguments")
            result_text = str(observation.get("content") or "")
            for kind, value in _extract_ids(result_text):
                add_artifact(unit, call, kind, value,
                             f"observations[{call.get('call_index', 0)}].content")
            for locator, value in (observation.get("structured_values") or {}).items():
                # Large stdout/document bodies are raw evidence, not compact live state.  Exact
                # paths/URLs/IDs inside them were already extracted above and remain recallable.
                if isinstance(value, str) and len(value) > 512:
                    continue
                add_artifact(unit, call, "field", value,
                             f"observations[{call.get('call_index', 0)}].{locator}")

    source_prefix_hash = hashlib.sha256(
        "\x00".join(unit.get("source_hash") or "" for unit in covered).encode("utf-8")
    ).hexdigest()[:16]
    return {
        "schema_version": "v2-deterministic.2",
        "compressor_version": "deterministic-l5-l7.2",
        "covers_through_unit": covered[-1]["unit_id"] if covered else None,
        "covers_units": [unit["unit_id"] for unit in covered],
        "source_prefix_hash": source_prefix_hash,
        "ledger_revision": units[-1].get("ledger_revision") if units else None,
        "objective": objective,
        "constraints": [],
        "confirmed_state": [],
        "live_artifacts": live_artifacts,
        "completed_actions": completed,
        "failed_or_superseded": failed,
        "unresolved_requirements": unresolved,
        "optional_narrative": "",
    }


def _value_atoms(value: Any) -> set[str]:
    atoms: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for child in node.values():
                walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)
        elif node is None or isinstance(node, bool):
            return
        elif isinstance(node, float) and not math.isfinite(node):
            return
        else:
            text = str(node)
            if len(text) >= 2:
                atoms.add(text)

    walk(value)
    return atoms


def build_dependency_index(units: list[dict]) -> dict[str, Any]:
    """Build hard producer->consumer edges from exact structured value reuse."""
    producer_for: dict[str, str] = {}
    dependencies: dict[str, set[str]] = {unit["unit_id"]: set() for unit in units}
    produced_by_unit: dict[str, set[str]] = {unit["unit_id"]: set() for unit in units}
    consumed_by_unit: dict[str, set[str]] = {unit["unit_id"]: set() for unit in units}

    for unit in units:
        unit_id = unit["unit_id"]
        consumed: set[str] = set()
        produced: set[str] = set()
        for call, observation in _call_observation_pairs(unit):
            if call.get("args_faithful"):
                consumed.update(_value_atoms(call.get("arguments") or {}))
            if observation.get("status") == "ok":
                produced.update(_value_atoms(observation.get("structured_values") or {}))
                produced.update(value for _, value in _extract_ids(
                    str(observation.get("content") or "")
                ))
        # Explicit references in the active user anchor also form hard exact-value edges.
        anchor = str(unit.get("user_anchor") or "")
        consumed.update(value for value in producer_for if value and value in anchor)
        for value in consumed:
            producer = producer_for.get(value)
            if producer and producer != unit_id:
                dependencies[unit_id].add(producer)
        consumed_by_unit[unit_id] = consumed
        produced_by_unit[unit_id] = produced
        for value in produced:
            producer_for[value] = unit_id

    return {
        "dependencies": {key: sorted(value) for key, value in dependencies.items()},
        "produced_by_unit": {key: sorted(value) for key, value in produced_by_unit.items()},
        "consumed_by_unit": {key: sorted(value) for key, value in consumed_by_unit.items()},
        "producer_for": producer_for,
    }


def dependency_closure(
    units: list[dict], seed_unit_ids: set[str] | None = None,
) -> set[str]:
    """Return the recursive hard dependency closure, including the seeds."""
    index = build_dependency_index(units)
    dependencies = index["dependencies"]
    closure = set(seed_unit_ids or set())
    frontier = list(closure)
    while frontier:
        unit_id = frontier.pop()
        for dependency in dependencies.get(unit_id, []):
            if dependency not in closure:
                closure.add(dependency)
                frontier.append(dependency)
    return closure
