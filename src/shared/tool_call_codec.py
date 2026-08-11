"""OpenAI/native Qwen tool-call codec shared by local model servers.

The Qwen3.5 GGUF chat template expects historical ``function.arguments`` to be a
mapping, while the OpenAI wire contract requires it to be a JSON object string.
This module is the explicit boundary between those two representations.  It also
parses the model's native XML tool-call output without erasing JSON value types.
"""
from __future__ import annotations

import copy
import json
import math
import re
import uuid
from typing import Any

from jsonschema import Draft7Validator
from jsonschema.exceptions import SchemaError


class ToolCallCodecError(ValueError):
    """Raised when a tool-call graph cannot be represented without data loss."""


TOOL_OUTPUT_CONTRACT = "qwen35-xml-schema-typed-args.2"

_TOOLCALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_FUNCTION_RE = re.compile(r"<function=([^>\s]+)\s*>(.*?)</function>", re.DOTALL)
_PARAMETER_RE = re.compile(r"<parameter=([^>\s]+)\s*>(.*?)</parameter>", re.DOTALL)


def _strict_json_loads(value: str) -> Any:
    def reject_constant(token: str) -> Any:
        raise ValueError(f"non-finite JSON constant is forbidden: {token}")

    def finite_float(token: str) -> float:
        decoded = float(token)
        if not math.isfinite(decoded):
            raise ValueError(f"non-finite JSON number is forbidden: {token}")
        return decoded

    return json.loads(
        value,
        parse_constant=reject_constant,
        parse_float=finite_float,
    )


def _remove_protocol_layout_newlines(raw: str) -> str:
    """Remove at most one XML-layout newline per edge, preserving real whitespace."""
    value = raw
    for newline in ("\r\n", "\n", "\r"):
        if value.startswith(newline):
            value = value[len(newline) :]
            break
    for newline in ("\r\n", "\n", "\r"):
        if value.endswith(newline):
            value = value[: -len(newline)]
            break
    return value


def _tool_schemas(tools: list[dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    schemas: dict[str, dict[str, Any]] = {}
    if tools is not None and not isinstance(tools, list):
        raise ToolCallCodecError("tools must be a list")
    for index, tool in enumerate(tools or []):
        fn = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(tool, dict) or tool.get("type") != "function":
            raise ToolCallCodecError(f"tools[{index}].type must be 'function'")
        if not isinstance(fn, dict) or not isinstance(fn.get("name"), str) or not fn["name"].strip():
            raise ToolCallCodecError(f"tools[{index}] is missing function.name")
        name = fn["name"].strip()
        if name in schemas:
            raise ToolCallCodecError(f"tools repeats function name {name!r}")
        params = fn.get("parameters")
        if params is None:
            params = {"type": "object", "properties": {}}
        if not isinstance(params, dict):
            raise ToolCallCodecError(f"tools[{index}].function.parameters must be an object")
        try:
            Draft7Validator.check_schema(params)
        except SchemaError as exc:
            raise ToolCallCodecError(
                f"tools[{index}].function.parameters is not a valid JSON schema"
            ) from exc
        schemas[name] = params
    return schemas


def validate_tool_definitions(tools: list[dict[str, Any]] | None) -> None:
    """Fail closed on malformed or ambiguous tool definitions."""
    _tool_schemas(tools)


def _validate_arguments(
    arguments: dict[str, Any], schema: dict[str, Any], *, tool_name: str
) -> None:
    errors = sorted(
        Draft7Validator(schema).iter_errors(arguments),
        key=lambda error: [str(part) for part in error.path],
    )
    if not errors:
        return
    error = errors[0]
    location = ".".join(str(part) for part in error.path) or "<root>"
    raise ToolCallCodecError(
        f"tool {tool_name!r} arguments violate schema at {location}: {error.message}"
    )


def _validate_calls_against_tools(
    calls: list[dict[str, Any]], tools: list[dict[str, Any]] | None
) -> None:
    schemas = _tool_schemas(tools)
    if calls and not schemas:
        raise ToolCallCodecError("model emitted tool calls but no tools were offered")
    for index, call in enumerate(calls):
        name = call["function"]["name"]
        if name not in schemas:
            raise ToolCallCodecError(f"model called unoffered tool {name!r}")
        arguments = _arguments_object(
            call["function"]["arguments"], context=f"tool call {index}"
        )
        _validate_arguments(arguments, schemas[name], tool_name=name)


def _property_schema(
    schemas: dict[str, dict[str, Any]], tool_name: str, parameter_name: str
) -> dict[str, Any]:
    params = schemas.get(tool_name) or {}
    props = params.get("properties")
    if not isinstance(props, dict):
        props = {}
    if parameter_name in props:
        prop = props.get(parameter_name)
        return prop if isinstance(prop, dict) else {}
    additional = params.get("additionalProperties")
    return additional if isinstance(additional, dict) else {}


def _schema_has_validation_keywords(schema: dict[str, Any]) -> bool:
    """Distinguish a real constraint from annotation-only schema metadata."""
    return any(keyword in Draft7Validator.VALIDATORS for keyword in schema)


def _subschema_validator(
    schema: dict[str, Any], root_schema: dict[str, Any] | None
) -> tuple[Draft7Validator, dict[str, Any] | None]:
    """Build a property validator while retaining the tool schema as the ``$ref`` root."""
    if isinstance(root_schema, dict):
        validator = Draft7Validator(root_schema)
        evolve = getattr(validator, "evolve", None)
        if callable(evolve):
            return evolve(schema=schema), None
        # jsonschema 3.x validates a subschema through this optional second argument;
        # retaining it keeps the root resolver available for local $ref definitions.
        return validator, schema
    return Draft7Validator(schema), None


def _candidate_is_valid(
    validator: Draft7Validator,
    legacy_subschema: dict[str, Any] | None,
    candidate: Any,
) -> bool:
    if legacy_subschema is not None:
        return validator.is_valid(candidate, legacy_subschema)
    return validator.is_valid(candidate)


def _append_distinct(candidates: list[Any], candidate: Any) -> None:
    """Append a JSON candidate without conflating bool/int or duplicate containers."""
    if any(
        type(existing) is type(candidate) and existing == candidate
        for existing in candidates
    ):
        return
    candidates.append(candidate)


def decode_qwen_parameter(
    raw: str,
    schema: dict[str, Any] | None = None,
    *,
    root_schema: dict[str, Any] | None = None,
    reject_unmatched_text: bool = False,
) -> Any:
    """Recover one schema-valid JSON value from Qwen's untyped XML parameter text.

    Qwen XML does not encode argument types.  We therefore construct the plausible
    lossless string, JSON, and explicit ``True``/``False``/``None`` alias candidates,
    and validate each against the *complete* property schema.  A valid string wins an
    actual ambiguity (so IDs such as ``00123`` remain strings), but an invalid string
    branch cannot mask a valid numeric/object branch in ``oneOf``/``anyOf``.  The root
    tool schema is retained so local ``$ref`` constraints keep working.

    With no useful schema, this preserves the legacy behavior: standards-compliant JSON
    literals are decoded and other text remains text.  A non-matching result is normally
    returned for the caller's final whole-arguments validation to reject or repair;
    ``reject_unmatched_text`` lets a strict boundary reject non-JSON text immediately.
    """
    property_schema = schema if isinstance(schema, dict) else {}
    lossless_string = _remove_protocol_layout_newlines(raw)
    json_text = lossless_string.strip()
    missing = object()
    decoded_json: Any = missing
    try:
        decoded_json = _strict_json_loads(json_text)
    except (TypeError, ValueError, json.JSONDecodeError):
        pass

    if not _schema_has_validation_keywords(property_schema):
        return decoded_json if decoded_json is not missing else json_text

    validator, legacy_subschema = _subschema_validator(property_schema, root_schema)
    candidates: list[Any] = []

    # Explicit JSON quotes are syntax rather than part of the string payload.
    if (
        isinstance(decoded_json, str)
        and json_text.startswith('"')
        and json_text.endswith('"')
    ):
        _append_distinct(candidates, decoded_json)

    # Prefer the lossless representation only when it really satisfies the schema.
    _append_distinct(candidates, lossless_string)

    lowered = json_text.casefold()
    if lowered in {"true", "false"}:
        _append_distinct(candidates, lowered == "true")
    if lowered in {"null", "none"}:
        _append_distinct(candidates, None)
    if decoded_json is not missing:
        _append_distinct(candidates, decoded_json)

    for candidate in candidates:
        if _candidate_is_valid(validator, legacy_subschema, candidate):
            return candidate

    if reject_unmatched_text and decoded_json is missing:
        raise ToolCallCodecError(
            "parameter is not valid JSON and no textual candidate satisfies its schema: "
            f"{json_text!r}"
        )
    return decoded_json if decoded_json is not missing else json_text


def _arguments_object(arguments: Any, *, context: str) -> dict[str, Any]:
    if isinstance(arguments, str):
        try:
            arguments = _strict_json_loads(arguments) if arguments.strip() else {}
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ToolCallCodecError(f"{context} has invalid JSON arguments") from exc
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        raise ToolCallCodecError(f"{context} arguments must be a JSON object")
    return arguments


def _normalise_tool_call(
    tool_call: dict[str, Any], *, index: int, arguments_as_mapping: bool,
    require_arguments: bool = False,
) -> dict[str, Any]:
    if not isinstance(tool_call, dict):
        raise ToolCallCodecError(f"tool_calls[{index}] must be an object")
    call_type = tool_call.get("type")
    if call_type not in (None, "function"):
        raise ToolCallCodecError(
            f"tool_calls[{index}].type must be 'function', got {call_type!r}"
        )
    fn = tool_call.get("function")
    if not isinstance(fn, dict) or not isinstance(fn.get("name"), str) or not fn["name"].strip():
        raise ToolCallCodecError(f"tool_calls[{index}] is missing function.name")
    raw_arguments = fn.get("arguments")
    if require_arguments and (
        "arguments" not in fn
        or raw_arguments is None
        or (isinstance(raw_arguments, str) and not raw_arguments.strip())
    ):
        raise ToolCallCodecError(f"tool_calls[{index}] is missing function.arguments")
    arguments = _arguments_object(raw_arguments, context=f"tool_calls[{index}]")
    try:
        wire_arguments = json.dumps(arguments, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ToolCallCodecError(f"tool_calls[{index}] arguments are not JSON-safe") from exc
    return {
        "id": str(tool_call.get("id") or f"call_{uuid.uuid4().hex[:24]}"),
        "type": "function",
        "function": {
            "name": fn["name"].strip(),
            "arguments": arguments if arguments_as_mapping else wire_arguments,
        },
    }


def normalize_tool_calls(
    tool_calls: Any, *, arguments_as_mapping: bool = False,
    require_arguments: bool = False,
) -> list[dict[str, Any]]:
    """Validate tool calls and convert arguments to template or wire representation."""
    if tool_calls is None:
        return []
    if not isinstance(tool_calls, list):
        raise ToolCallCodecError("tool_calls must be a list")
    calls = [
        _normalise_tool_call(
            tc,
            index=i,
            arguments_as_mapping=arguments_as_mapping,
            require_arguments=require_arguments,
        )
        for i, tc in enumerate(tool_calls)
    ]
    ids = [call["id"] for call in calls]
    if len(ids) != len(set(ids)):
        raise ToolCallCodecError("tool_call ids must be unique within one assistant message")
    return calls


def messages_for_qwen_template(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a deep-copied native history suitable for the Qwen3.5 Jinja template.

    OpenAI JSON-string arguments become mappings.  All linkage metadata remains on
    the copied messages even if a particular template uses positional tool results.
    The caller's request is never mutated.
    """
    out = copy.deepcopy(messages)
    known_call_ids: set[str] = set()
    pending_call_ids: set[str] = set()
    for index, message in enumerate(out):
        if not isinstance(message, dict):
            raise ToolCallCodecError(f"messages[{index}] must be an object")
        role = message.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ToolCallCodecError(f"messages[{index}] has unsupported role {role!r}")
        if role == "system" and index != 0:
            raise ToolCallCodecError("system message is only allowed at messages[0]")

        if pending_call_ids and role != "tool":
            raise ToolCallCodecError(
                f"messages[{index}] appears before all tool results were supplied: "
                f"missing {sorted(pending_call_ids)!r}"
            )
        if message.get("tool_calls") is not None:
            if role != "assistant":
                raise ToolCallCodecError(
                    f"messages[{index}] has tool_calls but is not an assistant message"
                )
            message["tool_calls"] = normalize_tool_calls(
                message["tool_calls"],
                arguments_as_mapping=True,
                require_arguments=True,
            )
            message_ids = {call["id"] for call in message["tool_calls"]}
            repeated = known_call_ids.intersection(message_ids)
            if repeated:
                raise ToolCallCodecError(
                    f"messages[{index}] reuses tool_call_id {sorted(repeated)[0]!r}"
                )
            known_call_ids.update(message_ids)
            pending_call_ids = set(message_ids)
        if role == "tool":
            tool_call_id = message.get("tool_call_id")
            if not isinstance(tool_call_id, str) or not tool_call_id:
                raise ToolCallCodecError(
                    f"messages[{index}] tool result is missing tool_call_id"
                )
            if tool_call_id not in known_call_ids:
                raise ToolCallCodecError(
                    f"messages[{index}] references unknown tool_call_id "
                    f"{tool_call_id!r}"
                )
            if tool_call_id not in pending_call_ids:
                raise ToolCallCodecError(
                    f"messages[{index}] repeats a tool result or misorders one for "
                    f"{tool_call_id!r}"
                )
            pending_call_ids.remove(tool_call_id)
    if pending_call_ids:
        raise ToolCallCodecError(
            f"messages end before all tool results were supplied: "
            f"missing {sorted(pending_call_ids)!r}"
        )

    # Qwen3.5's embedded template renders parallel tool results positionally and
    # does not print ``tool_call_id``.  OpenAI permits callers to return those
    # results in any order, so canonicalise each contiguous result group to the
    # assistant call order before entering the template.  This is a projection
    # only: ``out`` is already a deep copy and linkage IDs remain intact.
    index = 0
    while index < len(out):
        message = out[index]
        calls = message.get("tool_calls") or []
        if not calls:
            index += 1
            continue

        order = {call["id"]: position for position, call in enumerate(calls)}
        result_end = index + 1
        while result_end < len(out) and out[result_end].get("role") == "tool":
            result_end += 1
        results = out[index + 1 : result_end]
        if results:
            result_ids: list[str] = []
            for result_index, result in enumerate(results, start=index + 1):
                tool_call_id = result.get("tool_call_id")
                if not isinstance(tool_call_id, str) or not tool_call_id:
                    raise ToolCallCodecError(
                        f"messages[{result_index}] tool result is missing tool_call_id"
                    )
                if tool_call_id not in order:
                    raise ToolCallCodecError(
                        f"messages[{result_index}] references unknown tool_call_id "
                        f"{tool_call_id!r}"
                    )
                result_ids.append(tool_call_id)
            if len(result_ids) != len(set(result_ids)):
                raise ToolCallCodecError(
                    f"messages[{index + 1}:{result_end}] repeats a tool result"
                )
            out[index + 1 : result_end] = sorted(
                results, key=lambda result: order[result["tool_call_id"]]
            )
        index = result_end
    return out


def parse_qwen_tool_calls(
    text: str, tools: list[dict[str, Any]] | None = None
) -> tuple[list[dict[str, Any]], str]:
    """Parse Qwen3.5 XML/JSON ``<tool_call>`` blocks into OpenAI tool calls."""
    if not isinstance(text, str):
        raise ToolCallCodecError("tool-call output must be text")
    blocks = list(_TOOLCALL_RE.finditer(text))
    if "<tool_call" in text and not blocks:
        raise ToolCallCodecError("unterminated or malformed <tool_call> block")
    residual_markers = _TOOLCALL_RE.sub("", text)
    if "<tool_call" in residual_markers or "</tool_call>" in residual_markers:
        raise ToolCallCodecError("unterminated or malformed <tool_call> block")

    schemas = _tool_schemas(tools)
    calls: list[dict[str, Any]] = []
    for block_index, match in enumerate(blocks):
        block = match.group(1).strip()
        function_match = _FUNCTION_RE.fullmatch(block)
        if function_match:
            name = function_match.group(1).strip()
            body = function_match.group(2)
            parameters: dict[str, Any] = {}
            for parameter_match in _PARAMETER_RE.finditer(body):
                parameter_name = parameter_match.group(1).strip()
                if parameter_name in parameters:
                    raise ToolCallCodecError(
                        f"tool call {block_index} repeats parameter {parameter_name!r}"
                    )
                parameters[parameter_name] = decode_qwen_parameter(
                    parameter_match.group(2),
                    _property_schema(schemas, name, parameter_name),
                    root_schema=schemas.get(name),
                    reject_unmatched_text=True,
                )
            residual = _PARAMETER_RE.sub("", body).strip()
            if residual:
                raise ToolCallCodecError(
                    f"tool call {block_index} contains malformed parameter markup"
                )
            raw_call: dict[str, Any] = {
                "type": "function",
                "function": {"name": name, "arguments": parameters},
            }
        else:
            try:
                payload = _strict_json_loads(block)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ToolCallCodecError(
                    f"tool call {block_index} is neither Qwen XML nor valid JSON"
                ) from exc
            if not isinstance(payload, dict):
                raise ToolCallCodecError(f"tool call {block_index} JSON must be an object")
            raw_call = payload
            if "function" not in raw_call:
                raw_call = {
                    "id": payload.get("id"),
                    "type": "function",
                    "function": {
                        "name": payload.get("name"),
                        "arguments": payload.get("arguments", {}),
                    },
                }
        call = _normalise_tool_call(
            raw_call, index=block_index, arguments_as_mapping=False
        )
        calls.append(call)

    if blocks:
        for previous, current in zip(blocks, blocks[1:]):
            if text[previous.end() : current.start()].strip():
                raise ToolCallCodecError("non-whitespace text appears between tool calls")
        if text[blocks[-1].end() :].strip():
            raise ToolCallCodecError("non-whitespace text appears after a tool call")
        cleaned = text[: blocks[0].start()].strip()
    else:
        cleaned = text.strip()
    _validate_calls_against_tools(calls, tools)
    return calls, cleaned


def normalize_completion_message(
    message: dict[str, Any], tools: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Normalize a llama.cpp assistant message to the OpenAI wire contract."""
    if not isinstance(message, dict):
        raise ToolCallCodecError("completion message must be an object")
    content = message.get("content")
    text = content if isinstance(content, str) else ""
    native_calls = message.get("tool_calls")
    if native_calls:
        if "<tool_call" in text or "</tool_call>" in text:
            raise ToolCallCodecError(
                "model output contains both native and textual tool-call representations"
            )
        calls = normalize_tool_calls(native_calls, arguments_as_mapping=False)
        _validate_calls_against_tools(calls, tools)
        cleaned = text.strip()
    else:
        calls, cleaned = parse_qwen_tool_calls(text, tools) if tools else ([], text.strip())

    out: dict[str, Any] = {
        "role": "assistant",
        "content": cleaned if cleaned else (None if calls else ""),
    }
    reasoning = message.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning.strip():
        out["reasoning_content"] = reasoning.strip()
    if calls:
        out["tool_calls"] = calls
    return out
