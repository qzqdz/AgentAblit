"""Hybrid V2 context planner and OpenAI-native projection (L6/L8).

The planner freezes one ledger revision, computes recursive hard dependency closure, assigns a
minimum evidence-card representation before optional native upgrades, and accounts for the whole
request context.  If P0/P1 evidence cannot fit, it returns an explicit
``_tmi_context_status=context_insufficient``; callers must not send that body to B.
"""
from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from typing import Any

from . import ledger as L
from .spec import B_COLDSTART_HARNESS, B_COLDSTART_HINT_TEMPLATE, B_COLDSTART_SKILLS_TEMPLATE


_B_ACTION_MAX_TOKENS = 2048
_B_ACTION_MIN_TOKENS = 256
_NATIVE_HOT_FRONTIER_UNITS = 2
_TOOL_DESCRIPTION_CHARS = 192
_PARAMETER_DESCRIPTION_CHARS = 80
_SCHEMA_ANNOTATION_KEYS = {
    "$comment",
    "deprecated",
    "example",
    "examples",
    "readOnly",
    "title",
    "writeOnly",
}
_SCHEMA_MAP_KEYWORDS = {
    "$defs",
    "definitions",
    "dependentSchemas",
    "patternProperties",
    "properties",
}
_SCHEMA_SINGLE_KEYWORDS = {
    "additionalItems",
    "additionalProperties",
    "contains",
    "contentSchema",
    "else",
    "if",
    "items",
    "not",
    "propertyNames",
    "then",
    "unevaluatedItems",
    "unevaluatedProperties",
}
_SCHEMA_ARRAY_KEYWORDS = {"allOf", "anyOf", "oneOf", "prefixItems"}


def _msg_chars(message: dict) -> int:
    return len(json.dumps(message, ensure_ascii=False, sort_keys=True, allow_nan=False))


def _messages_chars(messages: list[dict]) -> int:
    return sum(_msg_chars(message) for message in messages)


def _request_context_chars(messages: list[dict], tools: list[dict], output_reserve: int) -> int:
    return (
        _messages_chars(messages)
        + len(json.dumps(tools or [], ensure_ascii=False, sort_keys=True, allow_nan=False))
        + output_reserve
        + 512  # request/template safety margin
    )


def _tools_chars(tools: list[dict]) -> int:
    return len(json.dumps(tools or [], ensure_ascii=False, sort_keys=True, allow_nan=False))


def _clip_middle(value: str, limit: int) -> str:
    """Bound an annotation while retaining both its purpose and its final caveat."""
    if len(value) <= limit:
        return value
    marker = " … "
    if limit <= len(marker):
        return value[:limit]
    remaining = limit - len(marker)
    head = (remaining + 1) // 2
    tail = remaining // 2
    return value[:head].rstrip() + marker + value[-tail:].lstrip()


def _project_parameter_schema(value: Any, *, mode: str) -> Any:
    """Project JSON Schema annotations without changing its validation structure."""
    if not isinstance(value, dict):
        return copy.deepcopy(value)

    projected: dict[str, Any] = {}
    for key, item in value.items():
        if key in _SCHEMA_ANNOTATION_KEYS:
            continue
        if key == "description":
            if mode == "compact" and isinstance(item, str):
                projected[key] = _clip_middle(item, _PARAMETER_DESCRIPTION_CHARS)
            elif mode == "compact":
                projected[key] = copy.deepcopy(item)
            continue
        if key in _SCHEMA_MAP_KEYWORDS and isinstance(item, dict):
            # Map keys are user property/definition names, never schema annotations.
            projected[key] = {
                name: _project_parameter_schema(subschema, mode=mode)
                for name, subschema in item.items()
            }
        elif key == "dependencies" and isinstance(item, dict):
            # Draft-07 dependencies may contain either a schema or a property-name list.
            projected[key] = {
                name: (
                    _project_parameter_schema(dependency, mode=mode)
                    if isinstance(dependency, dict)
                    else copy.deepcopy(dependency)
                )
                for name, dependency in item.items()
            }
        elif key in _SCHEMA_SINGLE_KEYWORDS:
            if key == "items" and isinstance(item, list):
                projected[key] = [
                    _project_parameter_schema(subschema, mode=mode)
                    for subschema in item
                ]
            elif isinstance(item, dict):
                projected[key] = _project_parameter_schema(item, mode=mode)
            else:
                projected[key] = copy.deepcopy(item)
        elif key in _SCHEMA_ARRAY_KEYWORDS and isinstance(item, list):
            projected[key] = [
                _project_parameter_schema(subschema, mode=mode)
                for subschema in item
            ]
        else:
            # Validation values (enum/const/required/bounds/etc.) and unknown extensions
            # are copied exactly; recursing here would confuse user keys with annotations.
            projected[key] = copy.deepcopy(item)
    return projected


def _project_tools(tools: list[dict], *, mode: str) -> list[dict]:
    """Return exact, compact, or structural-only copies of all offered tools.

    Tool selection is deliberately out of scope: every tool name and every JSON Schema
    validation keyword remains available to B.  Only prose annotations lose resolution.
    Candidate validation later still uses the untouched source schemas.
    """
    if mode == "exact":
        return copy.deepcopy(tools)
    if mode not in {"compact", "structural"}:
        raise ValueError(f"unknown tool schema projection mode: {mode}")

    projected_tools: list[dict] = []
    for tool in tools:
        projected = copy.deepcopy(tool)
        function = projected.get("function")
        if isinstance(function, dict):
            description = function.get("description")
            if mode == "compact" and isinstance(description, str):
                function["description"] = _clip_middle(description, _TOOL_DESCRIPTION_CHARS)
            else:
                function.pop("description", None)
            if "parameters" in function:
                function["parameters"] = _project_parameter_schema(
                    function["parameters"], mode=mode
                )
        projected_tools.append(projected)
    return projected_tools


def _resolve_b_action_tokens(request_body: dict[str, Any]) -> tuple[int, Any]:
    """Separate B's one-action allowance from the host model's answer allowance."""
    requested = request_body.get("max_completion_tokens")
    if not isinstance(requested, int) or isinstance(requested, bool) or requested <= 0:
        requested = request_body.get("max_tokens")
    if not isinstance(requested, int) or isinstance(requested, bool) or requested <= 0:
        requested = None
    action_tokens = _B_ACTION_MAX_TOKENS if requested is None else requested
    action_tokens = max(_B_ACTION_MIN_TOKENS, min(_B_ACTION_MAX_TOKENS, action_tokens))
    return action_tokens, requested


def _has_pending_observation(unit: dict) -> bool:
    return any(
        observation.get("status") in {"pending", "missing"}
        for observation in unit.get("observations") or []
    )


def _split_folded_prefix(units: list[dict]) -> tuple[list[dict], list[dict]]:
    """Keep a bounded native frontier while preserving any unresolved owner chain."""
    if not units:
        return [], []
    frontier_start = max(0, len(units) - _NATIVE_HOT_FRONTIER_UNITS)
    pending_indexes = [
        index for index, unit in enumerate(units) if _has_pending_observation(unit)
    ]
    if pending_indexes:
        frontier_start = min(frontier_start, pending_indexes[0])
    return units[:frontier_start], units[frontier_start:]


def _bounded_state_value(value: Any) -> Any:
    """Keep exact ordinary state atoms; fingerprint unusually large narrative fields."""
    if isinstance(value, str) and len(value) > 512:
        return {
            "head": value[:192],
            "tail": value[-192:],
            "length": len(value),
            "sha256": hashlib.sha256(value.encode("utf-8")).hexdigest()[:16],
            "exact": False,
        }
    if isinstance(value, dict):
        return {key: _bounded_state_value(item) for key, item in value.items()}
    if isinstance(value, list):
        if len(value) <= 64:
            return [_bounded_state_value(item) for item in value]
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
        return {
            "head_items": [_bounded_state_value(item) for item in value[:16]],
            "tail_items": [_bounded_state_value(item) for item in value[-16:]],
            "length": len(value),
            "sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16],
            "exact": False,
        }
    return copy.deepcopy(value)


def _select_checkpoint_entries(
    entries: list[dict], relevant_unit_ids: set[str], tail: int,
) -> tuple[list[dict], int]:
    relevant = [entry for entry in entries if entry.get("unit_id") in relevant_unit_ids]
    recent = entries[-tail:] if tail > 0 else []
    selected: list[dict] = []
    seen: set[str] = set()
    for entry in relevant + recent:
        identity = str(entry.get("fact_id") or entry.get("call_id") or "")
        if not identity:
            identity = json.dumps(entry, ensure_ascii=False, sort_keys=True, default=str)
        if identity not in seen:
            seen.add(identity)
            selected.append(copy.deepcopy(entry))
    return selected, max(0, len(entries) - len(selected))


def _project_checkpoint_for_context(
    checkpoint: dict, relevant_unit_ids: set[str],
) -> tuple[dict, dict[str, int]]:
    """Render a bounded state index; the immutable ledger remains the complete source of truth."""
    projected = copy.deepcopy(checkpoint)
    limits = {
        "live_artifacts": 24,
        "completed_actions": 24,
        "failed_or_superseded": 8,
        "unresolved_requirements": 16,
    }
    omitted: dict[str, int] = {}
    for field, tail in limits.items():
        selected, omitted_count = _select_checkpoint_entries(
            list(checkpoint.get(field) or []), relevant_unit_ids, tail
        )
        if field == "live_artifacts":
            for entry in selected:
                entry["value"] = _bounded_state_value(entry.get("value"))
                entry["typed_value"] = _bounded_state_value(entry.get("typed_value"))
        projected[field] = selected
        omitted[field] = omitted_count
    projected["omitted_counts"] = {key: value for key, value in omitted.items() if value}
    return projected, omitted


def _compact_typed_evidence(values: dict[str, Any]) -> dict[str, Any]:
    """Keep typed state atoms without duplicating large raw narrative fields."""
    return {
        locator: _bounded_state_value(value)
        for locator, value in values.items()
        if not isinstance(value, str) or len(value) <= 512
    }


def _render_checkpoint(checkpoint: dict) -> dict:
    """Render derived state as a clearly labelled memory message, never as tool evidence."""
    lines = ["【派生状态摘要—确定性抽取，非环境证据；冲突时以后续原始证据为准】"]
    objective = checkpoint.get("objective") or {}
    if objective.get("initial"):
        lines.append(f"总目标：{objective['initial']}")
    if objective.get("current") and objective.get("current") != objective.get("initial"):
        lines.append(f"当前要求：{objective['current']}")
    artifacts = checkpoint.get("live_artifacts") or []
    if artifacts:
        lines.append("live values:")
        for artifact in artifacts:
            label = artifact.get("name") or artifact.get("kind") or "value"
            rendered_value = json.dumps(
                artifact.get("value"), ensure_ascii=False, allow_nan=False
            )
            lines.append(
                f"- {label}={rendered_value} "
                f"[source:{artifact.get('unit_id')}]"
            )
    completed = checkpoint.get("completed_actions") or []
    if completed:
        lines.append("completed (do not redo):")
        for action in completed:
            lines.append(
                f"- {action.get('tool')} args="
                f"{json.dumps(action.get('arguments'), ensure_ascii=False, sort_keys=True)} "
                f"[source:{action.get('unit_id')}]"
            )
    failed = checkpoint.get("failed_or_superseded") or []
    if failed:
        lines.append("failed/dead ends (do not retry unchanged):")
        for action in failed:
            lines.append(
                f"- {action.get('tool')} args="
                f"{json.dumps(action.get('arguments'), ensure_ascii=False, sort_keys=True)}; "
                f"error={json.dumps(action.get('error'), ensure_ascii=False)} "
                f"[source:{action.get('unit_id')}]"
            )
    unresolved = checkpoint.get("unresolved_requirements") or []
    if unresolved:
        lines.append("unresolved calls:")
        for action in unresolved:
            lines.append(f"- {action.get('tool')} [source:{action.get('unit_id')}]")
    omitted = checkpoint.get("omitted_counts") or {}
    if omitted:
        lines.append(
            "ledger-only state omitted from this bounded view (rehydrate on validation failure): "
            + json.dumps(omitted, ensure_ascii=False, sort_keys=True)
        )
    return {"role": "user", "content": "\n".join(lines)}


def _unit_native_safe(unit: dict, duplicate_ids: set[str]) -> bool:
    calls = unit.get("calls") or []
    observations = unit.get("observations") or []
    if not calls:
        return bool(unit.get("assistant_text"))
    if len(observations) != len(calls):
        return False
    local_ids: set[str] = set()
    for call in calls:
        call_id = call.get("call_id")
        if (
            not call.get("args_faithful")
            or not isinstance(call_id, str)
            or not call_id
            or call_id in duplicate_ids
            or call_id in local_ids
            or not call.get("tool_name")
        ):
            return False
        local_ids.add(call_id)
    by_index = {observation.get("call_index"): observation for observation in observations}
    return all(
        by_index.get(call.get("call_index"), {}).get("status") in {"ok", "error"}
        for call in calls
    )


def _unit_to_native_messages(unit: dict) -> list[dict]:
    calls = [{
        "id": call["call_id"],
        "type": "function",
        "function": {
            "name": call["tool_name"],
            "arguments": json.dumps(
                call.get("arguments") or {}, ensure_ascii=False, sort_keys=True, allow_nan=False
            ),
        },
    } for call in unit.get("calls") or []]
    if not calls:
        return ([{"role": "assistant", "content": unit.get("assistant_text") or ""}]
                if unit.get("assistant_text") else [])
    messages = [{
        "role": "assistant",
        "content": unit.get("assistant_text") or "",
        "tool_calls": calls,
    }]
    observations = {
        observation.get("call_index"): observation
        for observation in unit.get("observations") or []
    }
    for call in unit.get("calls") or []:
        observation = observations[call.get("call_index")]
        messages.append({
            "role": "tool",
            "tool_call_id": call["call_id"],
            "name": call["tool_name"],
            "content": observation.get("raw_content")
            if observation.get("raw_content") is not None
            else observation.get("content") or "",
        })
    return messages


def _unit_evidence_card(unit: dict) -> list[dict]:
    """R2 typed evidence card: exact state atoms plus bounded raw provenance."""
    calls = []
    observations = {o.get("call_index"): o for o in unit.get("observations") or []}
    for call in unit.get("calls") or []:
        observation = observations.get(call.get("call_index"), {})
        arguments = (
            call.get("arguments") if call.get("args_faithful")
            else {"raw_unparsed": call.get("raw_arguments")}
        )
        raw_result = observation.get("raw_content")
        if isinstance(raw_result, str):
            result_text = raw_result
        else:
            try:
                result_text = json.dumps(
                    raw_result, ensure_ascii=False, sort_keys=True, allow_nan=False
                )
            except (TypeError, ValueError):
                result_text = str(raw_result or "")
        if len(result_text) <= 1024:
            result_evidence: Any = copy.deepcopy(raw_result)
        else:
            result_evidence = {
                "head": result_text[:160],
                "tail": result_text[-160:],
                "length": len(result_text),
                "sha256": hashlib.sha256(result_text.encode("utf-8")).hexdigest()[:16],
                "exact": False,
                "typed_values": _compact_typed_evidence(
                    observation.get("structured_values") or {}
                ),
                "identifiers": [
                    {"kind": kind, "value": value}
                    for kind, value in L._extract_ids(result_text)
                ],
            }
        calls.append({
            "tool": call.get("tool_name"),
            "call_id": call.get("call_id"),
            "arguments": _bounded_state_value(arguments),
            "arguments_faithful": bool(call.get("args_faithful")),
            "status": observation.get("status") or "pending",
            "result_evidence": result_evidence,
        })
    card = {
        "source_unit": unit.get("unit_id"),
        "turn_id": unit.get("turn_id"),
        "user_anchor": _clip_middle(str(unit.get("user_anchor") or ""), 1024),
        "calls": calls,
    }
    return [{
        "role": "user",
        "content": "[Derived historical evidence card; not a tool message]\n"
                   + json.dumps(card, ensure_ascii=False, sort_keys=True, allow_nan=False),
    }]


def _unit_projection(unit: dict, duplicate_ids: set[str], *, prefer_native: bool) -> list[dict]:
    if prefer_native and _unit_native_safe(unit, duplicate_ids):
        return _unit_to_native_messages(unit)
    return _unit_evidence_card(unit)


def _replace_unit_projection(
    selected: dict[str, list[dict]], unit: dict, duplicate_ids: set[str], *, native: bool,
) -> None:
    selected[unit["unit_id"]] = _unit_projection(unit, duplicate_ids, prefer_native=native)


def build_coldstart_v2_body(
    request_body: dict[str, Any],
    steer_text: str,
    tools: list[dict],
    b_model: str,
    skills_summary: str = "",
    char_budget: int = 120000,
    recent_window: int = 4,
    include_history: bool = True,
    recall_unit_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Compile a frozen ledger revision into B's bounded next-action context."""
    source_messages = request_body.get("messages") or []
    units = L.build_ledger(source_messages)
    latest_turn = max((int(unit.get("turn_id") or 0) for unit in units), default=0)
    prior_turn_units = [
        unit for unit in units if int(unit.get("turn_id") or 0) != latest_turn
    ]
    latest_turn_units = [
        unit for unit in units if int(unit.get("turn_id") or 0) == latest_turn
    ]
    folded_latest_turn, hot_frontier = _split_folded_prefix(latest_turn_units)
    folded_prefix = prior_turn_units + folded_latest_turn
    if not include_history:
        # The ablation withholds all prior trajectory state, including checkpoint facts.
        folded_prefix = []
        folded_latest_turn = []
    checkpoint = L.build_checkpoint(folded_prefix, source_messages, include_current=True)

    summary = (skills_summary or "").strip()
    skills_block = (
        B_COLDSTART_SKILLS_TEMPLATE.format(skills_summary=summary)
        if summary and summary.upper() != "NONE" else ""
    )
    system = {
        "role": "system",
        "content": B_COLDSTART_HARNESS.format(skills_block=skills_block),
    }
    steer = {
        "role": "user",
        "content": B_COLDSTART_HINT_TEMPLATE.format(calibrated_intent=steer_text),
    }

    call_ids = [
        call.get("call_id")
        for unit in units for call in unit.get("calls") or []
        if call.get("call_id")
    ]
    duplicate_ids = {call_id for call_id, count in Counter(call_ids).items() if count > 1}
    by_id = {unit["unit_id"]: unit for unit in units}

    # P1: a bounded native hot frontier plus explicitly recalled units and recursive producers.
    eligible_units = units if include_history else hot_frontier
    eligible_ids = {unit["unit_id"] for unit in eligible_units}
    seeds = {unit["unit_id"] for unit in hot_frontier}
    seeds.update(unit_id for unit_id in (recall_unit_ids or set()) if unit_id in eligible_ids)
    objective_text = str((checkpoint.get("objective") or {}).get("current") or "")
    dependency_index = L.build_dependency_index(eligible_units)
    objective_dependency_ids = {
        producer
        for value, producer in (dependency_index.get("producer_for") or {}).items()
        if value and value in objective_text and producer in eligible_ids
    }
    seeds.update(objective_dependency_ids)
    closure = L.dependency_closure(eligible_units, seeds)
    required_closed_ids = {
        unit_id for unit_id in closure
        if unit_id in by_id and by_id[unit_id] in folded_prefix
    }
    same_turn_folded_ids = {unit["unit_id"] for unit in folded_latest_turn}
    minimum_evidence_ids = required_closed_ids | same_turn_folded_ids
    checkpoint_view, checkpoint_omitted = _project_checkpoint_for_context(
        checkpoint, minimum_evidence_ids
    )
    checkpoint_message = _render_checkpoint(checkpoint_view)

    selected: dict[str, list[dict]] = {}
    for unit in folded_prefix:
        if unit["unit_id"] in minimum_evidence_ids:
            _replace_unit_projection(selected, unit, duplicate_ids, native=False)
    for unit in hot_frontier:
        # The latest closed/open owner units retain the model's native continuation shape.
        _replace_unit_projection(selected, unit, duplicate_ids, native=True)

    def assemble() -> list[dict]:
        projected = [system, checkpoint_message]
        for unit in units:
            projected.extend(selected.get(unit["unit_id"], []))
        projected.append(steer)
        return projected

    b_max_tokens, host_requested_max = _resolve_b_action_tokens(request_body)
    output_reserve = b_max_tokens * 4
    minimum_messages = assemble()
    original_tool_chars = _tools_chars(tools)
    tool_variants = [
        (mode, _project_tools(tools, mode=mode))
        for mode in ("exact", "compact", "structural")
    ]
    tool_schema_mode, projected_tools = tool_variants[-1]
    insufficient = True
    for candidate_mode, candidate_tools in tool_variants:
        if _request_context_chars(minimum_messages, candidate_tools, output_reserve) \
                <= char_budget:
            tool_schema_mode = candidate_mode
            projected_tools = candidate_tools
            insufficient = False
            break

    if not insufficient:
        # Upgrade required dependencies to R0/R1, then add other P2/P3 units in a stable order.
        upgrade_order = [
            unit for unit in folded_prefix if unit["unit_id"] in required_closed_ids
        ]
        upgrade_order.extend(
            unit for unit in reversed(folded_latest_turn)
            if unit["unit_id"] not in required_closed_ids
        )
        optional_ids: list[str] = []
        failed = checkpoint_view.get("failed_or_superseded") or []
        if failed:
            unit_id = failed[-1].get("unit_id")
            if unit_id and unit_id not in optional_ids and unit_id not in required_closed_ids:
                optional_ids.append(unit_id)
        if recent_window > 0:
            for unit in folded_prefix[-recent_window:]:
                if unit["unit_id"] not in optional_ids \
                        and unit["unit_id"] not in required_closed_ids:
                    optional_ids.append(unit["unit_id"])
        optional_units = [by_id[unit_id] for unit_id in optional_ids if unit_id in by_id]

        for unit in upgrade_order:
            old = selected.get(unit["unit_id"])
            _replace_unit_projection(selected, unit, duplicate_ids, native=True)
            if _request_context_chars(assemble(), projected_tools, output_reserve) > char_budget:
                selected[unit["unit_id"]] = old or []
        for unit in optional_units:
            candidate_projection = _unit_projection(unit, duplicate_ids, prefer_native=True)
            selected[unit["unit_id"]] = candidate_projection
            if _request_context_chars(assemble(), projected_tools, output_reserve) > char_budget:
                selected.pop(unit["unit_id"], None)

    messages = assemble()
    final_cost = _request_context_chars(messages, projected_tools, output_reserve)
    body = copy.deepcopy(request_body)
    body["messages"] = messages
    body["tools"] = projected_tools
    body["model"] = b_model
    body["stream"] = False
    body["max_tokens"] = b_max_tokens
    body.pop("max_completion_tokens", None)
    body.pop("stream_options", None)
    # The V2 harness asks for a next action but preserves a caller's stronger explicit choice.
    body["_tmi_context_status"] = "context_insufficient" if insufficient else "ok"
    body["_tmi_context_plan"] = {
        "schema_version": "hybrid-v2-plan.3",
        "ledger_revision": units[-1].get("ledger_revision") if units else None,
        "checkpoint_prefix_hash": checkpoint.get("source_prefix_hash"),
        "checkpoint_omitted_counts": {
            key: value for key, value in checkpoint_omitted.items() if value
        },
        "char_budget": char_budget,
        "estimated_context_chars": final_cost,
        "output_reserve_chars": output_reserve,
        "host_requested_max_tokens": host_requested_max,
        "b_max_tokens": b_max_tokens,
        "tool_schema_mode": tool_schema_mode,
        "tool_schema_original_chars": original_tool_chars,
        "tool_schema_projected_chars": _tools_chars(projected_tools),
        "current_unit_ids": [unit["unit_id"] for unit in hot_frontier],
        "hot_frontier_unit_ids": [unit["unit_id"] for unit in hot_frontier],
        "folded_prefix_unit_ids": [unit["unit_id"] for unit in folded_prefix],
        "same_turn_evidence_unit_ids": [
            unit["unit_id"] for unit in folded_latest_turn
        ],
        "objective_dependency_unit_ids": sorted(objective_dependency_ids),
        "dependency_closure_unit_ids": sorted(closure),
        "recalled_unit_ids": sorted(recall_unit_ids or set()),
        "selected_unit_ids": [unit["unit_id"] for unit in units if unit["unit_id"] in selected],
        "unit_representations": {
            unit["unit_id"]: (
                "native"
                if any(message.get("role") in {"assistant", "tool"}
                       for message in selected[unit["unit_id"]])
                else "evidence_card"
            )
            for unit in units if unit["unit_id"] in selected
        },
    }
    return body
