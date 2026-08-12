"""Deterministic inventory adapters for budgeted tool and skill selection.

This module is deliberately model-free.  It turns the canonical request into immutable
``ContextAsset`` records, derives a small action-state query, and identifies assets which
must never be removed by a later semantic selector.  Canonical schemas and complete Hermes
skill catalog entries remain local; an observer may eventually rank their bounded cards but
is never asked to reconstruct either source of truth.
"""
from __future__ import annotations

import copy
import html
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

from .coldstart_v2 import _project_tools
from .context_assets import ContextAsset, canonical_catalog_digest


CostFn = Callable[[Any], int]

_SKILL_BLOCK_RE = re.compile(
    r"<available_skills(?:\s[^>]*)?>(.*?)</available_skills\s*>",
    re.IGNORECASE | re.DOTALL,
)
_SKILL_ROW_RE = re.compile(
    r"^(?P<indent>[ \t]*)-\s+(?P<name>.+?)(?::(?:[ \t]+|$))(?P<description>.*)$"
)
_CATEGORY_ROW_RE = re.compile(
    r"^(?P<indent>[ \t]*)(?P<name>[^-].*?)(?::(?:[ \t]+|$))(?P<description>.*)$"
)
_SKILL_NAME_RE = re.compile(r"^[^\s<>]+$")
_DEFAULT_SKILL_LOADER_NAMES = ("skill_view", "load_skill", "skill_load")
_JSON_SEPARATORS = (",", ":")


@dataclass(frozen=True)
class ContextAssetInventory:
    """One canonical request inventory and its content-addressed revision."""

    assets: tuple[ContextAsset, ...]
    digest: str


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=_JSON_SEPARATORS,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("context asset inventory must be canonical-JSON serializable") from exc


def _default_cost(payload: Any) -> int:
    return len(_canonical_json(payload))


def _measure(payload: Any, cost_fn: CostFn | None) -> int:
    value = (cost_fn or _default_cost)(payload)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("cost_fn must return a non-negative integer")
    return value


def _clip_middle(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    marker = " … "
    if limit <= len(marker):
        return text[:limit]
    remaining = limit - len(marker)
    head = (remaining + 1) // 2
    tail = remaining // 2
    return text[:head].rstrip() + marker + text[-tail:].lstrip()


def _one_line(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())


def adapt_openai_tools(
    tools: Sequence[Mapping[str, Any]] | None,
    *,
    cost_fn: CostFn | None = None,
    max_card_chars: int = 384,
) -> tuple[ContextAsset, ...]:
    """Convert OpenAI function tools without weakening their validation contract.

    The canonical payload is a deep copy of the request schema.  The structural payload
    reuses Hybrid V2's proven projector, which drops prose annotations while preserving
    names, property names, types, required fields, enums, bounds, combinators, and unknown
    validation extensions.
    """

    if isinstance(max_card_chars, bool) or not isinstance(max_card_chars, int):
        raise ValueError("max_card_chars must be an integer")
    if max_card_chars < 32:
        raise ValueError("max_card_chars must be at least 32")

    assets: list[ContextAsset] = []
    seen: set[str] = set()
    for index, raw_tool in enumerate(tools or ()):
        if not isinstance(raw_tool, Mapping):
            raise ValueError(f"tool at index {index} must be an object")
        tool = copy.deepcopy(dict(raw_tool))
        if tool.get("type") != "function" or not isinstance(tool.get("function"), Mapping):
            raise ValueError(f"tool at index {index} must be an OpenAI function tool")
        function = tool["function"]
        name = function.get("name")
        if not isinstance(name, str) or not name.strip() or name != name.strip():
            raise ValueError(f"tool at index {index} has an invalid function name")
        asset_id = f"tool:{name}"
        if asset_id in seen:
            raise ValueError(f"duplicate tool name: {name}")
        seen.add(asset_id)

        structural = _project_tools([tool], mode="structural")[0]
        description = _one_line(function.get("description"))
        card = name if not description else f"{name}: {description}"
        assets.append(ContextAsset(
            asset_id=asset_id,
            kind="tool",
            canonical_payload=tool,
            structural_payload=structural,
            structural_cost=_measure(structural, cost_fn),
            card=_clip_middle(card, max_card_chars),
            recency=index + 1,
        ))
    return tuple(assets)


def _parse_skill_rows(block: str) -> list[dict[str, str]]:
    category = ""
    current: dict[str, str] | None = None
    current_indent = -1
    rows: list[dict[str, str]] = []

    def finish() -> None:
        nonlocal current
        if current is None:
            return
        current["description"] = _one_line(html.unescape(current["description"]))
        rows.append(current)
        current = None

    for raw_line in block.splitlines():
        line = raw_line.expandtabs(2).rstrip()
        if not line.strip():
            continue
        skill_match = _SKILL_ROW_RE.match(line)
        if skill_match:
            finish()
            name = html.unescape(skill_match.group("name").strip())
            if not name or not _SKILL_NAME_RE.fullmatch(name):
                # Text in a structured block is still untrusted request content.  A row
                # which cannot be a skill_view name is data, not an asset identifier.
                current_indent = -1
                continue
            current_indent = len(skill_match.group("indent"))
            current = {
                "qualified_name": name,
                "category": category,
                "description": skill_match.group("description").strip(),
            }
            continue

        category_match = _CATEGORY_ROW_RE.match(line)
        if category_match and len(category_match.group("indent")) <= 2:
            finish()
            category = html.unescape(category_match.group("name").strip())
            current_indent = -1
            continue

        # Preserve wrapped descriptions in their entirety.  This is deterministic parsing,
        # not the previous 24K/512-token observer summary path.
        indent = len(line) - len(line.lstrip())
        if current is not None and indent > current_indent:
            continuation = line.strip()
            if continuation:
                current["description"] += " " + continuation

    finish()
    return rows


def parse_hermes_skill_assets(
    system_prompt: str | None,
    *,
    available_tool_ids: Iterable[str] = (),
    loader_tool_names: Sequence[str] = _DEFAULT_SKILL_LOADER_NAMES,
    cost_fn: CostFn | None = None,
    max_card_chars: int = 384,
) -> tuple[ContextAsset, ...]:
    """Parse every skill row from structured Hermes ``<available_skills>`` blocks.

    No model call, input clipping, or summary token limit is involved.  The complete name,
    category, and description are retained in the canonical payload.  A compact card is the
    only projected representation.  If the structured block is absent or malformed, the
    safe result is an empty tuple rather than a guess from arbitrary prompt prose.
    """

    if not isinstance(system_prompt, str) or not system_prompt:
        return ()
    if isinstance(max_card_chars, bool) or not isinstance(max_card_chars, int):
        raise ValueError("max_card_chars must be an integer")
    if max_card_chars < 32:
        raise ValueError("max_card_chars must be at least 32")

    available = {
        asset_id for asset_id in available_tool_ids
        if isinstance(asset_id, str) and asset_id.startswith("tool:")
    }
    loader_dependencies = tuple(
        f"tool:{name}"
        for name in loader_tool_names
        if isinstance(name, str) and f"tool:{name}" in available
    )
    # One loader is sufficient.  Requiring every synonymous loader would waste budget and
    # could make the mandatory closure impossible on backends exposing compatibility aliases.
    dependencies = loader_dependencies[:1]

    records: list[dict[str, str]] = []
    for match in _SKILL_BLOCK_RE.finditer(system_prompt):
        records.extend(_parse_skill_rows(match.group(1)))
    if not records:
        return ()

    assets: list[ContextAsset] = []
    seen: dict[str, dict[str, str]] = {}
    for index, record in enumerate(records):
        name = record["qualified_name"]
        asset_id = f"skill:{name}"
        previous = seen.get(asset_id)
        if previous is not None:
            if previous == record:
                continue
            raise ValueError(f"conflicting duplicate skill name: {name}")
        seen[asset_id] = record

        canonical = copy.deepcopy(record)
        description = record["description"]
        category_label = f" [{record['category']}]" if record["category"] else ""
        card_text = (
            f"{name}{category_label}"
            if not description
            else f"{name}{category_label}: {description}"
        )
        structural = {
            "qualified_name": name,
            "category": record["category"],
            "card": _clip_middle(card_text, max_card_chars),
        }
        assets.append(ContextAsset(
            asset_id=asset_id,
            kind="skill",
            canonical_payload=canonical,
            structural_payload=structural,
            structural_cost=_measure(structural, cost_fn),
            dependencies=dependencies,
            card=structural["card"],
            recency=index + 1,
        ))
    return tuple(assets)


def _content_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return ""
    parts: list[str] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        if item.get("type") in {"text", "input_text"} and isinstance(item.get("text"), str):
            parts.append(item["text"])
    return "\n".join(parts)


def build_context_asset_inventory(
    request_body: Mapping[str, Any],
    *,
    cost_fn: CostFn | None = None,
    max_card_chars: int = 384,
    loader_tool_names: Sequence[str] = _DEFAULT_SKILL_LOADER_NAMES,
) -> ContextAssetInventory:
    """Build the canonical tool+skill inventory for one OpenAI request."""

    if not isinstance(request_body, Mapping):
        raise ValueError("request_body must be an object")
    raw_tools = request_body.get("tools")
    if raw_tools is None:
        tools: Sequence[Mapping[str, Any]] = ()
    elif isinstance(raw_tools, list):
        tools = raw_tools
    else:
        raise ValueError("request tools must be a list")
    tool_assets = adapt_openai_tools(
        tools,
        cost_fn=cost_fn,
        max_card_chars=max_card_chars,
    )

    system_parts = [
        _content_text(message.get("content"))
        for message in request_body.get("messages") or ()
        if isinstance(message, Mapping) and message.get("role") == "system"
    ]
    skill_assets = parse_hermes_skill_assets(
        "\n".join(part for part in system_parts if part),
        available_tool_ids={asset.asset_id for asset in tool_assets},
        loader_tool_names=loader_tool_names,
        cost_fn=cost_fn,
        max_card_chars=max_card_chars,
    )
    assets = tool_assets + skill_assets
    return ContextAssetInventory(
        assets=assets,
        digest=canonical_catalog_digest(assets),
    )


def _bounded_value(value: Any, *, text_limit: int, item_limit: int, depth: int = 0) -> Any:
    if isinstance(value, str):
        return _clip_middle(value, text_limit)
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if depth >= 2:
        try:
            display = _canonical_json(value)
        except ValueError:
            # At the depth boundary this becomes quoted display evidence in the outer
            # state, so permissive NaN/default rendering cannot corrupt its JSON syntax.
            try:
                display = json.dumps(
                    value,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=_JSON_SEPARATORS,
                    default=str,
                )
            except (TypeError, ValueError):
                display = str(value)
        return _clip_middle(display, text_limit)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key in sorted(value, key=str)[:item_limit]:
            result[str(key)] = _bounded_value(
                value[key], text_limit=text_limit, item_limit=item_limit, depth=depth + 1
            )
        if len(value) > item_limit:
            result["__omitted_items__"] = len(value) - item_limit
        return result
    if isinstance(value, (list, tuple)):
        result = [
            _bounded_value(item, text_limit=text_limit, item_limit=item_limit, depth=depth + 1)
            for item in value[:item_limit]
        ]
        if len(value) > item_limit:
            result.append({"__omitted_items__": len(value) - item_limit})
        return result
    return _clip_middle(str(value), text_limit)


def _observation_for_call(
    unit: Mapping[str, Any], call: Mapping[str, Any], index: int
) -> Mapping[str, Any]:
    observations = unit.get("observations") or ()
    call_index = call.get("call_index", index)
    for observation in observations:
        if isinstance(observation, Mapping) and observation.get("call_index") == call_index:
            return observation
    if index < len(observations) and isinstance(observations[index], Mapping):
        return observations[index]
    return {"status": "pending", "content": None}


def _unit_is_open(unit: Mapping[str, Any]) -> bool:
    calls = [call for call in unit.get("calls") or () if isinstance(call, Mapping)]
    if not calls:
        return False
    return any(
        _observation_for_call(unit, call, index).get("status") in {None, "", "pending"}
        for index, call in enumerate(calls)
    )


def _select_hot_units(
    ledger: Sequence[Mapping[str, Any]], max_units: int
) -> list[Mapping[str, Any]]:
    if max_units <= 0:
        return []
    chosen: set[int] = set()
    for index in range(len(ledger) - 1, -1, -1):
        if _unit_is_open(ledger[index]):
            chosen.add(index)
            if len(chosen) >= max_units:
                break
    for index in range(len(ledger) - 1, -1, -1):
        chosen.add(index)
        if len(chosen) >= max_units:
            break
    return [ledger[index] for index in sorted(chosen)]


def _request_objective(
    request_body: Mapping[str, Any], checkpoint: Mapping[str, Any] | None
) -> str:
    users = [
        _content_text(message.get("content"))
        for message in request_body.get("messages") or ()
        if isinstance(message, Mapping) and message.get("role") == "user"
    ]
    if users:
        return users[-1]
    objective = (checkpoint or {}).get("objective")
    if isinstance(objective, Mapping):
        current = objective.get("current") or objective.get("initial")
        return current if isinstance(current, str) else ""
    return objective if isinstance(objective, str) else ""


def _action_record(
    unit: Mapping[str, Any], call: Mapping[str, Any], observation: Mapping[str, Any],
    *, text_limit: int, item_limit: int,
) -> dict[str, Any]:
    return {
        "unit_id": str(unit.get("unit_id") or ""),
        "tool": str(call.get("tool_name") or ""),
        "status": str(observation.get("status") or "pending"),
        "arguments": _bounded_value(
            call.get("arguments") if call.get("arguments") is not None else {},
            text_limit=text_limit,
            item_limit=item_limit,
        ),
        "content": _bounded_value(
            observation.get("content"), text_limit=text_limit, item_limit=item_limit
        ),
    }


def _encoded_length(value: Any) -> int:
    return len(_canonical_json(value))


def _fit_action_state(state: dict[str, Any], max_chars: int) -> dict[str, Any]:
    """Drop low-priority detail until the serialized state satisfies the hard bound."""

    while _encoded_length(state) > max_chars and state.get("live_artifacts"):
        state["live_artifacts"].pop(0)
        state["truncated"] = True
    while _encoded_length(state) > max_chars and state.get("unresolved_requirements"):
        state["unresolved_requirements"].pop(0)
        state["truncated"] = True

    for limit in (96, 48, 24):
        if _encoded_length(state) <= max_chars:
            break
        state["objective"] = _clip_middle(str(state.get("objective") or ""), limit)
        for unit in state.get("hot_units") or ():
            unit["user_anchor"] = _clip_middle(str(unit.get("user_anchor") or ""), limit)
            for record in unit.get("calls") or ():
                record["content"] = _bounded_value(
                    record.get("content"), text_limit=limit, item_limit=2
                )
                record["arguments"] = _bounded_value(
                    record.get("arguments"), text_limit=limit, item_limit=2
                )
        for key in ("latest_observation", "latest_error"):
            record = state.get(key)
            if isinstance(record, dict):
                record["content"] = _bounded_value(
                    record.get("content"), text_limit=limit, item_limit=2
                )
                record["arguments"] = _bounded_value(
                    record.get("arguments"), text_limit=limit, item_limit=2
                )
        state["truncated"] = True

    if _encoded_length(state) > max_chars:
        # Frontier membership is more important than duplicated argument/result detail.
        # Keep every selected hot/open unit's native tool identity and status first.
        for unit in state.get("hot_units") or ():
            unit.pop("user_anchor", None)
            for record in unit.get("calls") or ():
                record.pop("arguments", None)
                record.pop("content", None)
                record.pop("unit_id", None)
        state["truncated"] = True

    while _encoded_length(state) > max_chars and len(state.get("hot_units") or ()) > 1:
        state["hot_units"].pop(0)
        state["truncated"] = True

    if _encoded_length(state) > max_chars:
        # The identifiers and frontier tool/status pairs are the irreducible action-state
        # contract.  This fallback remains structured and never copies raw result bodies.
        state = {
            "schema_version": "context-action-state.v1",
            "objective": _clip_middle(str(state.get("objective") or ""), 48),
            "ledger_revision": state.get("ledger_revision"),
            "checkpoint_revision": state.get("checkpoint_revision"),
            "hot_units": [
                {
                    "unit_id": unit.get("unit_id"),
                    "calls": [
                        {"tool": call.get("tool"), "status": call.get("status")}
                        for call in unit.get("calls") or ()
                    ],
                }
                for unit in (state.get("hot_units") or ())[-1:]
            ],
            "truncated": True,
        }
    if _encoded_length(state) > max_chars:
        state = {"schema_version": "context-action-state.v1", "truncated": True}
    return state


def build_bounded_action_state(
    request_body: Mapping[str, Any],
    ledger: Sequence[Mapping[str, Any]] | None = None,
    checkpoint: Mapping[str, Any] | None = None,
    *,
    max_chars: int = 4_096,
    max_units: int = 4,
    max_items: int = 12,
    max_value_chars: int = 256,
) -> dict[str, Any]:
    """Build a bounded next-action query from the request, ledger, and checkpoint.

    Only the latest/open frontier, the latest observation/error, and small typed checkpoint
    facts are retained.  Large tool results and skill bodies are never copied into this view.
    """

    for name, value, minimum in (
        ("max_chars", max_chars, 128),
        ("max_units", max_units, 1),
        ("max_items", max_items, 1),
        ("max_value_chars", max_value_chars, 16),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if not isinstance(request_body, Mapping):
        raise ValueError("request_body must be an object")
    units = [unit for unit in (ledger or ()) if isinstance(unit, Mapping)]
    checkpoint = checkpoint if isinstance(checkpoint, Mapping) else {}
    hot = _select_hot_units(units, max_units)

    hot_units: list[dict[str, Any]] = []
    latest_observation: dict[str, Any] | None = None
    latest_error: dict[str, Any] | None = None
    for unit in units:
        for index, call in enumerate(
            call for call in unit.get("calls") or () if isinstance(call, Mapping)
        ):
            observation = _observation_for_call(unit, call, index)
            record = _action_record(
                unit,
                call,
                observation,
                text_limit=max_value_chars,
                item_limit=min(max_items, 6),
            )
            latest_observation = record
            if record["status"] == "error":
                latest_error = record

    for unit in hot:
        calls: list[dict[str, Any]] = []
        for index, call in enumerate(
            call for call in unit.get("calls") or () if isinstance(call, Mapping)
        ):
            observation = _observation_for_call(unit, call, index)
            calls.append(_action_record(
                unit,
                call,
                observation,
                text_limit=max_value_chars,
                item_limit=min(max_items, 6),
            ))
        hot_units.append({
            "unit_id": str(unit.get("unit_id") or ""),
            "turn_id": unit.get("turn_id"),
            "user_anchor": _clip_middle(str(unit.get("user_anchor") or ""), max_value_chars),
            "calls": calls,
        })

    artifact_keys = ("kind", "name", "value", "status", "tool", "unit_id")
    artifacts = [
        {
            key: _bounded_value(
                artifact.get(key), text_limit=max_value_chars, item_limit=min(max_items, 6)
            )
            for key in artifact_keys
            if key in artifact
        }
        for artifact in (checkpoint.get("live_artifacts") or ())[-max_items:]
        if isinstance(artifact, Mapping)
    ]
    unresolved = [
        _bounded_value(item, text_limit=max_value_chars, item_limit=min(max_items, 6))
        for item in (checkpoint.get("unresolved_requirements") or ())[-max_items:]
    ]
    state = {
        "schema_version": "context-action-state.v1",
        "objective": _clip_middle(
            _request_objective(request_body, checkpoint), max(max_value_chars * 2, 64)
        ),
        "ledger_revision": units[-1].get("ledger_revision") if units else None,
        "checkpoint_revision": checkpoint.get("source_prefix_hash")
        or checkpoint.get("ledger_revision"),
        "hot_units": hot_units,
        "latest_observation": latest_observation,
        "latest_error": latest_error,
        "live_artifacts": artifacts,
        "unresolved_requirements": unresolved,
        "truncated": (
            len(units) > len(hot)
            or len(checkpoint.get("live_artifacts") or ()) > len(artifacts)
            or len(checkpoint.get("unresolved_requirements") or ()) > len(unresolved)
        ),
    }
    return _fit_action_state(state, max_chars)


def _explicit_name_match(text: str, name: str) -> bool:
    if not text or not name:
        return False
    # Underscore is deliberately a name character: tool "read" must not become mandatory
    # merely because the objective explicitly names the different tool "read_file".
    left = r"(?<![A-Za-z0-9_])"
    right = r"(?![A-Za-z0-9_])"
    return re.search(left + re.escape(name) + right, text, re.IGNORECASE) is not None


def _forced_tool_name(tool_choice: Any) -> str:
    if isinstance(tool_choice, Mapping):
        function = tool_choice.get("function")
        if isinstance(function, Mapping) and isinstance(function.get("name"), str):
            return function["name"]
        return ""
    if isinstance(tool_choice, str) and tool_choice not in {"auto", "none", "required"}:
        return tool_choice
    return ""


def derive_mandatory_asset_ids(
    request_body: Mapping[str, Any],
    assets: Sequence[ContextAsset],
    *,
    ledger: Sequence[Mapping[str, Any]] | None = None,
    checkpoint: Mapping[str, Any] | None = None,
    action_state: Mapping[str, Any] | None = None,
    selected_asset_ids: Iterable[str] = (),
    hot_unit_count: int = 4,
) -> tuple[str, ...]:
    """Derive the deterministic asset closure which a selector may not remove."""

    if not isinstance(request_body, Mapping):
        raise ValueError("request_body must be an object")
    if (
        isinstance(hot_unit_count, bool)
        or not isinstance(hot_unit_count, int)
        or hot_unit_count < 1
    ):
        raise ValueError("hot_unit_count must be a positive integer")
    by_id = {asset.asset_id: asset for asset in assets}
    if len(by_id) != len(assets):
        raise ValueError("assets contain duplicate asset IDs")

    seeds = {asset.asset_id for asset in assets if asset.mandatory}
    forced = _forced_tool_name(request_body.get("tool_choice"))
    if f"tool:{forced}" in by_id:
        seeds.add(f"tool:{forced}")

    units = [unit for unit in (ledger or ()) if isinstance(unit, Mapping)]
    for unit in _select_hot_units(units, hot_unit_count):
        for call in unit.get("calls") or ():
            if not isinstance(call, Mapping):
                continue
            asset_id = f"tool:{call.get('tool_name') or ''}"
            if asset_id in by_id:
                seeds.add(asset_id)

    if isinstance(action_state, Mapping):
        objective_value = action_state.get("objective")
        if isinstance(objective_value, Mapping):
            objective = str(objective_value.get("current") or objective_value.get("initial") or "")
        else:
            objective = str(objective_value or "")
    else:
        objective = _request_objective(request_body, checkpoint)
    for asset_id, asset in by_id.items():
        name = asset_id.split(":", 1)[1]
        if _explicit_name_match(objective, name):
            seeds.add(asset_id)

    seeds.update(
        asset_id for asset_id in selected_asset_ids
        if isinstance(asset_id, str) and asset_id in by_id
    )

    if any(by_id[asset_id].kind == "skill" for asset_id in seeds):
        for loader_name in _DEFAULT_SKILL_LOADER_NAMES:
            loader_id = f"tool:{loader_name}"
            if loader_id in by_id:
                seeds.add(loader_id)
                break

    closure: set[str] = set()

    def include(asset_id: str) -> None:
        if asset_id in closure or asset_id not in by_id:
            return
        for dependency in by_id[asset_id].dependencies:
            include(dependency)
        closure.add(asset_id)

    for asset_id in sorted(seeds):
        include(asset_id)
    return tuple(sorted(closure))


__all__ = [
    "ContextAssetInventory",
    "adapt_openai_tools",
    "build_bounded_action_state",
    "build_context_asset_inventory",
    "derive_mandatory_asset_ids",
    "parse_hermes_skill_assets",
]
