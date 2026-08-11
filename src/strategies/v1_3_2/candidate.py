"""Hybrid V2 L9 deterministic candidate validation and repair planning."""
from __future__ import annotations

import json
import math
from typing import Any

from jsonschema import Draft7Validator
from jsonschema.exceptions import SchemaError


def _strict_json_loads(value: str) -> Any:
    def reject_constant(token: str) -> Any:
        raise ValueError(f"non-finite JSON constant is forbidden: {token}")

    return json.loads(value, parse_constant=reject_constant)


def _norm(arguments: Any) -> str:
    try:
        return json.dumps(arguments if isinstance(arguments, dict) else {}, sort_keys=True,
                          ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError):
        return repr(arguments)


def _parse_args(raw: Any) -> tuple[dict[str, Any] | None, Any]:
    if isinstance(raw, dict):
        parsed = raw
    elif isinstance(raw, str):
        try:
            parsed = _strict_json_loads(raw) if raw.strip() else {}
        except (TypeError, ValueError, json.JSONDecodeError):
            return None, raw
    elif raw is None:
        return None, raw
    else:
        return None, raw
    if not isinstance(parsed, dict):
        return None, raw

    def finite(node: Any) -> bool:
        if isinstance(node, float):
            return math.isfinite(node)
        if isinstance(node, dict):
            return all(finite(value) for value in node.values())
        if isinstance(node, list):
            return all(finite(value) for value in node)
        return True

    return (parsed, raw) if finite(parsed) else (None, raw)


def _tool_schemas(tools: list[dict]) -> tuple[dict[str, dict], list[dict]]:
    schemas: dict[str, dict] = {}
    reasons: list[dict] = []
    if not isinstance(tools, list):
        return {}, [{"code": "invalid_tools", "detail": "tools must be a list"}]
    for index, tool in enumerate(tools):
        function = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(tool, dict) or tool.get("type") != "function" \
                or not isinstance(function, dict):
            reasons.append({"code": "invalid_tools", "detail": f"tools[{index}]"})
            continue
        name = function.get("name")
        if not isinstance(name, str) or not name.strip() or name in schemas:
            reasons.append({"code": "invalid_tools", "detail": f"tools[{index}].name"})
            continue
        parameters = function.get("parameters")
        parameters = parameters if isinstance(parameters, dict) else {
            "type": "object", "properties": {},
        }
        try:
            Draft7Validator.check_schema(parameters)
        except SchemaError as error:
            reasons.append({"code": "invalid_schema", "detail": f"{name}: {error.message}"})
            continue
        schemas[name] = parameters
    return schemas, reasons


def _schema_reason(error: Any) -> dict:
    path = ".".join(str(part) for part in error.absolute_path) or "<root>"
    if error.validator == "required":
        missing = ""
        if isinstance(error.validator_value, list) and isinstance(error.instance, dict):
            missing = next((key for key in error.validator_value if key not in error.instance), "")
        return {"code": "missing_required", "detail": missing or error.message, "path": path}
    if error.validator == "type":
        return {"code": "type_error", "detail": f"{path}: {error.message}", "path": path}
    if error.validator == "enum":
        return {"code": "enum_error", "detail": f"{path}: {error.message}", "path": path}
    if error.validator == "additionalProperties":
        return {"code": "additional_properties", "detail": error.message, "path": path}
    return {"code": "schema_violation", "detail": f"{path}: {error.message}", "path": path}


def validate_candidate(candidate: dict, tools: list[dict], checkpoint: dict) -> dict:
    """Validate one complete OpenAI tool-call transport object and its state transition."""
    if not isinstance(candidate, dict):
        return {"verdict": "invalid", "reasons": [{"code": "no_action", "detail": ""}],
                "recall": []}
    call_id = candidate.get("id")
    call_type = candidate.get("type")
    function = candidate.get("function")
    transport_reasons: list[dict] = []
    if not isinstance(call_id, str) or not call_id.strip():
        transport_reasons.append({"code": "invalid_call_id", "detail": repr(call_id)})
    if call_type != "function":
        transport_reasons.append({"code": "invalid_call_type", "detail": repr(call_type)})
    if not isinstance(function, dict):
        transport_reasons.append({"code": "no_action", "detail": "missing function"})
        return {"verdict": "invalid", "reasons": transport_reasons, "recall": []}
    name = function.get("name")
    if not isinstance(name, str) or not name.strip():
        transport_reasons.append({"code": "no_action", "detail": "missing function.name"})
    if "arguments" not in function:
        transport_reasons.append({"code": "args_not_object", "detail": "arguments missing"})
    if transport_reasons:
        return {"verdict": "invalid", "reasons": transport_reasons, "recall": []}

    schemas, definition_reasons = _tool_schemas(tools)
    if definition_reasons:
        return {"verdict": "invalid", "reasons": definition_reasons, "recall": []}
    name = name.strip()
    if name not in schemas:
        return {
            "verdict": "invalid",
            "reasons": [{"code": "unknown_tool", "detail": name}],
            "recall": [],
        }

    arguments, raw = _parse_args(function.get("arguments"))
    reasons: list[dict] = []
    recall: list[dict] = []
    if arguments is None:
        reasons.append({"code": "args_not_object", "detail": repr(raw)[:240]})
        recall.append({"kind": "tool_schema", "tool": name})
        arguments = {}
    else:
        errors = sorted(
            Draft7Validator(schemas[name]).iter_errors(arguments),
            key=lambda error: [str(part) for part in error.absolute_path],
        )
        reasons.extend(_schema_reason(error) for error in errors)
        if errors:
            recall.append({"kind": "tool_schema", "tool": name})

    key = (name, _norm(arguments))
    for action in checkpoint.get("completed_actions") or []:
        if (action.get("tool"), _norm(action.get("arguments"))) == key:
            reasons.append({"code": "redo_completed", "detail": name})
            recall.append({"kind": "completed_chain", "unit_id": action.get("unit_id")})
            break
    for action in checkpoint.get("failed_or_superseded") or []:
        if (action.get("tool"), _norm(action.get("arguments"))) == key:
            reasons.append({
                "code": "deadend_retry", "detail": str(action.get("error") or "")[:240],
            })
            recall.append({
                "kind": "failure", "unit_id": action.get("unit_id"),
                "error": str(action.get("error") or "")[:400],
            })
            break

    if not reasons:
        return {"verdict": "valid", "reasons": [], "recall": []}
    deduped_recall: list[dict] = []
    seen: set[str] = set()
    for item in recall:
        identity = json.dumps(item, ensure_ascii=False, sort_keys=True)
        if identity not in seen:
            seen.add(identity)
            deduped_recall.append(item)
    return {"verdict": "repairable", "reasons": reasons, "recall": deduped_recall}


def validate_candidates(candidates: Any, tools: list[dict], checkpoint: dict) -> dict:
    """Validate a parallel tool-call batch atomically; one bad call rejects the batch."""
    if not isinstance(candidates, list) or not candidates:
        return {"verdict": "invalid", "reasons": [{"code": "no_action", "detail": ""}],
                "recall": []}
    ids = [candidate.get("id") for candidate in candidates if isinstance(candidate, dict)]
    if len(ids) != len(candidates) or len(ids) != len(set(ids)):
        return {
            "verdict": "invalid",
            "reasons": [{"code": "duplicate_call_id", "detail": repr(ids)}],
            "recall": [],
        }
    results = [validate_candidate(candidate, tools, checkpoint) for candidate in candidates]
    reasons = [reason for result in results for reason in result["reasons"]]
    recall = [item for result in results for item in result["recall"]]
    if any(result["verdict"] == "invalid" for result in results):
        verdict = "invalid"
    elif any(result["verdict"] == "repairable" for result in results):
        verdict = "repairable"
    else:
        verdict = "valid"
    unique_recall: list[dict] = []
    seen: set[str] = set()
    for item in recall:
        identity = json.dumps(item, ensure_ascii=False, sort_keys=True)
        if identity not in seen:
            seen.add(identity)
            unique_recall.append(item)
    return {"verdict": verdict, "reasons": reasons, "recall": unique_recall}


def build_repair_note(
    reasons: list[dict], recall: list[dict], checkpoint: dict, tools: list[dict],
) -> str:
    """Build a grounded correction note; recalled raw units are injected separately by L8."""
    lines = ["【上一个候选动作未通过确定性校验；修正后只输出下一批合法 tool_call】"]
    messages = {
        "redo_completed": "相同工具和参数已经完成，不能重做；推进到后续步骤。",
        "deadend_retry": "相同工具和参数已确认失败；状态未变化，不能原样重试。",
        "args_not_object": "arguments 必须是严格 JSON object，禁止 NaN/Infinity。",
        "invalid_call_id": "每个 tool_call 必须有非空且批内唯一的 id。",
        "invalid_call_type": "tool_call.type 必须是 function。",
    }
    for reason in reasons:
        code = reason.get("code")
        if code in messages:
            lines.append(f"- {messages[code]} 证据：{reason.get('detail') or ''}")
        elif code in {"missing_required", "type_error", "enum_error",
                      "additional_properties", "schema_violation"}:
            lines.append(f"- schema 错误：{reason.get('detail') or ''}")

    recalled_tools = {item.get("tool") for item in recall if item.get("kind") == "tool_schema"}
    for tool in tools or []:
        function = tool.get("function") if isinstance(tool, dict) else None
        if isinstance(function, dict) and function.get("name") in recalled_tools:
            lines.append(
                f"- {function['name']} schema="
                f"{json.dumps(function.get('parameters') or {}, ensure_ascii=False, sort_keys=True)}"
            )
    recalled_units = {item.get("unit_id") for item in recall if item.get("unit_id")}
    for action in (checkpoint.get("completed_actions") or []) \
            + (checkpoint.get("failed_or_superseded") or []):
        if action.get("unit_id") in recalled_units:
            lines.append(
                f"- source:{action['unit_id']} tool={action.get('tool')} "
                f"args={json.dumps(action.get('arguments'), ensure_ascii=False, sort_keys=True)} "
                f"status={action.get('status')} error={action.get('error') or ''}"
            )
    return "\n".join(lines)
