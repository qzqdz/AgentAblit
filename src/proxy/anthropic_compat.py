"""Anthropic ↔ OpenAI format conversion utilities for TMI proxy.

Converts incoming Anthropic /v1/messages requests to OpenAI chat completion
format (for upstream Dashscope) and converts OpenAI responses back to
Anthropic format (for clients like Claude Code, OpenClaw, etc.).
"""
from __future__ import annotations

import json
from typing import AsyncGenerator


# ---------------------------------------------------------------------------
# Request: Anthropic → OpenAI
# ---------------------------------------------------------------------------

def _content_blocks_to_openai(content: list[dict]) -> tuple[str, str, list[dict], list[dict]]:
    """Parse Anthropic content block array.

    Returns (text, reasoning_text, tool_calls, tool_results).
    tool_results are separate message dicts (role=tool).
    """
    text_parts: list[str] = []
    thinking_parts: list[str] = []
    tool_calls: list[dict] = []
    tool_results: list[dict] = []

    for block in content:
        btype = block.get("type", "")
        if btype == "text":
            text_parts.append(block.get("text") or "")
        elif btype == "thinking":
            thinking_parts.append(block.get("thinking") or "")
        elif btype == "tool_use":
            fn = block.get("name") or ""
            inp = block.get("input") or {}
            tool_calls.append({
                "id": block.get("id") or "",
                "type": "function",
                "function": {
                    "name": fn,
                    "arguments": json.dumps(inp, ensure_ascii=False),
                },
            })
        elif btype == "tool_result":
            rc = block.get("content")
            if isinstance(rc, list):
                result_text = "\n".join(
                    b.get("text") or "" for b in rc if b.get("type") == "text"
                )
            else:
                result_text = str(rc or "")
            tool_results.append({
                "role": "tool",
                "tool_call_id": block.get("tool_use_id") or "",
                "content": result_text,
            })

    return (
        "".join(text_parts),
        "".join(thinking_parts),
        tool_calls,
        tool_results,
    )


def _convert_tools(anthropic_tools: list[dict]) -> list[dict]:
    """Convert Anthropic tool definitions to OpenAI function calling format."""
    result = []
    for tool in anthropic_tools:
        result.append({
            "type": "function",
            "function": {
                "name": tool.get("name") or "",
                "description": tool.get("description") or "",
                "parameters": tool.get("input_schema") or {"type": "object", "properties": {}},
            },
        })
    return result


def _convert_tool_choice(tc) -> object:
    """Convert Anthropic tool_choice to OpenAI format."""
    if isinstance(tc, str):
        if tc == "auto":
            return "auto"
        if tc == "any":
            return "required"
        if tc == "none":
            return "none"
        return "auto"
    if isinstance(tc, dict):
        tc_type = tc.get("type", "auto")
        if tc_type == "tool":
            return {"type": "function", "function": {"name": tc.get("name") or ""}}
        if tc_type == "any":
            return "required"
        if tc_type == "none":
            return "none"
    return "auto"


def anthropic_to_openai_body(body: dict, override_model: str) -> dict:
    """Convert an Anthropic /v1/messages request body to OpenAI chat completions format.

    override_model is the model to send to the upstream (ignores what the
    client asked for, since the upstream is always DeepSeek/Dashscope).
    """
    messages: list[dict] = []

    # System prompt → role=system message
    system = body.get("system")
    if system:
        if isinstance(system, str):
            messages.append({"role": "system", "content": system})
        elif isinstance(system, list):
            system_text = "\n".join(
                b.get("text") or "" for b in system if b.get("type") == "text"
            )
            if system_text:
                messages.append({"role": "system", "content": system_text})

    for msg in body.get("messages") or []:
        role = msg.get("role") or "user"
        content = msg.get("content")

        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue

        if isinstance(content, list):
            text, thinking, tool_calls, tool_results = _content_blocks_to_openai(content)

            if tool_results:
                # tool_result blocks → individual tool role messages
                messages.extend(tool_results)
            elif tool_calls:
                msg_dict: dict = {"role": role, "content": text}
                msg_dict["tool_calls"] = tool_calls
                messages.append(msg_dict)
            else:
                msg_dict = {"role": role, "content": text}
                if thinking and role == "assistant":
                    msg_dict["reasoning_content"] = thinking
                messages.append(msg_dict)
        else:
            messages.append({"role": role, "content": ""})

    oai: dict = {
        "model": override_model,
        "messages": messages,
        "stream": False,  # always serialized; caller re-emits as needed
    }

    if body.get("max_tokens") is not None:
        oai["max_tokens"] = body["max_tokens"]
    if body.get("temperature") is not None:
        oai["temperature"] = body["temperature"]
    if body.get("top_p") is not None:
        oai["top_p"] = body["top_p"]
    if body.get("stop_sequences"):
        oai["stop"] = body["stop_sequences"]
    if body.get("tools"):
        oai["tools"] = _convert_tools(body["tools"])
    if body.get("tool_choice") is not None:
        oai["tool_choice"] = _convert_tool_choice(body["tool_choice"])

    return oai


# ---------------------------------------------------------------------------
# Response: OpenAI → Anthropic (non-streaming)
# ---------------------------------------------------------------------------

_FINISH_REASON_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "content_filter": "stop_sequence",
}


def openai_to_anthropic_response(data: dict, req_model: str) -> dict:
    """Convert an OpenAI chat completion dict to Anthropic message format."""
    choices = data.get("choices") or []
    choice = choices[0] if choices else {}
    message = choice.get("message") or {}

    content: list[dict] = []

    reasoning = str(message.get("reasoning_content") or "").strip()
    if reasoning:
        content.append({"type": "thinking", "thinking": reasoning})

    text = str(message.get("content") or "").strip()
    if text:
        content.append({"type": "text", "text": text})

    for tc in message.get("tool_calls") or []:
        fn = tc.get("function") or {}
        try:
            inp = json.loads(fn.get("arguments") or "{}")
        except Exception:
            inp = {}
        content.append({
            "type": "tool_use",
            "id": tc.get("id") or "",
            "name": fn.get("name") or "",
            "input": inp,
        })

    if not content:
        content.append({"type": "text", "text": ""})

    finish_reason = str(choice.get("finish_reason") or "stop")
    stop_reason = _FINISH_REASON_MAP.get(finish_reason, "end_turn")

    usage = data.get("usage") or {}
    return {
        "id": data.get("id") or "msg_tmi",
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": req_model,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


# ---------------------------------------------------------------------------
# Response: OpenAI → Anthropic SSE (streaming)
# ---------------------------------------------------------------------------

def _sse(event: str, payload: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")


async def emit_anthropic_stream(data: dict, req_model: str) -> AsyncGenerator[bytes, None]:
    """Emit Anthropic SSE events from a complete OpenAI response dict.

    Follows the Anthropic streaming event sequence:
      message_start → (thinking block) → text/tool blocks → message_delta → message_stop
    """
    choices = data.get("choices") or []
    choice = choices[0] if choices else {}
    message = choice.get("message") or {}

    reasoning = str(message.get("reasoning_content") or "").strip()
    text = str(message.get("content") or "").strip()
    tool_calls = message.get("tool_calls") or []

    finish_reason = str(choice.get("finish_reason") or "stop")
    stop_reason = _FINISH_REASON_MAP.get(finish_reason, "end_turn")

    usage = data.get("usage") or {}
    input_tokens = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)
    msg_id = data.get("id") or "msg_tmi"

    # 1. message_start
    yield _sse("message_start", {
        "type": "message_start",
        "message": {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": req_model,
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": input_tokens, "output_tokens": 0},
        },
    })

    block_index = 0

    # 2. thinking block (reasoning_content → type=thinking)
    if reasoning:
        yield _sse("content_block_start", {
            "type": "content_block_start",
            "index": block_index,
            "content_block": {"type": "thinking", "thinking": ""},
        })
        yield _sse("content_block_delta", {
            "type": "content_block_delta",
            "index": block_index,
            "delta": {"type": "thinking_delta", "thinking": reasoning},
        })
        yield _sse("content_block_stop", {
            "type": "content_block_stop",
            "index": block_index,
        })
        block_index += 1

    # 3. text block (always emit, even if empty, when no tool calls)
    if text or not tool_calls:
        yield _sse("content_block_start", {
            "type": "content_block_start",
            "index": block_index,
            "content_block": {"type": "text", "text": ""},
        })
        if text:
            yield _sse("content_block_delta", {
                "type": "content_block_delta",
                "index": block_index,
                "delta": {"type": "text_delta", "text": text},
            })
        yield _sse("content_block_stop", {
            "type": "content_block_stop",
            "index": block_index,
        })
        block_index += 1

    # 4. tool_use blocks
    for tc in tool_calls:
        fn = tc.get("function") or {}
        args_str = fn.get("arguments") or "{}"
        yield _sse("content_block_start", {
            "type": "content_block_start",
            "index": block_index,
            "content_block": {
                "type": "tool_use",
                "id": tc.get("id") or "",
                "name": fn.get("name") or "",
                "input": {},
            },
        })
        yield _sse("content_block_delta", {
            "type": "content_block_delta",
            "index": block_index,
            "delta": {"type": "input_json_delta", "partial_json": args_str},
        })
        yield _sse("content_block_stop", {
            "type": "content_block_stop",
            "index": block_index,
        })
        block_index += 1

    # 5. message_delta + message_stop
    yield _sse("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"output_tokens": output_tokens},
    })
    yield _sse("message_stop", {"type": "message_stop"})
