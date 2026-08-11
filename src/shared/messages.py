"""OpenAI-message helpers with no version-specific policy."""
from __future__ import annotations

import json
import re
from typing import Any


_TEXT_OMITTED = "[omitted: tool/result content is not evidence for new factual claims]"
_FORBIDDEN_FACT_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(read|opened|inspected|checked|found in|saw in)\b.*\b(file|log|repo|repository|codebase)\b",
        r"\b(read|opened|inspected|checked|found in|saw in)\b.*(?:[A-Za-z]:[/\\]|/|\.{1,2}[/\\]|[\w.-]+[/\\]|[\w.-]+\.(py|ts|tsx|js|json|md|txt|log|sh|yaml|yml))",
        r"\b(ran|executed|called)\b.*\b(command|test|tests|pytest|compileall|npm|pnpm|bash|powershell)\b",
        r"\b(test|tests|pytest|compile|build|lint|typecheck)\b.*\b(pass|passed|fail|failed|success|succeeded)\b",
        r"\b(pass|passed|fail|failed|success|succeeded)\b.*\b(test|tests|pytest|compile|build|lint|typecheck)\b",
        r"\b(log|logs|api response|http response|status code)\b.*\b(show|shows|returned|contains|indicates)\b",
        r"\b(file|repository|repo|codebase)\b.*\b(contains|has|defines|imports)\b",
    )
)


def text_from_content(content: Any) -> str:
    """Flatten a message `content` field to plain text.

    OpenAI Chat Completions allows `content` to be a string OR an array of
    content parts (`[{"type": "text", "text": ...}, {"type": "image_url", ...}]`).
    Anthropic blocks share the same `{"type": "text", "text": ...}` shape. This
    extracts only the textual parts and drops non-text parts (images, tool_use,
    etc.); the upstream is text-only, so dropping non-text parts is safe.

    Use this anywhere a sniffer/extractor needs "what was said" as a string,
    instead of assuming `isinstance(content, str)`.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        text = content.get("text")
        return text if isinstance(text, str) else ""
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                # OpenAI: {"type":"text","text":...}; Responses API: input_text /
                # output_text; Anthropic text block: same shape. Skip non-text
                # parts (image_url, tool_use, etc.).
                if item.get("type") in (None, "text", "input_text", "output_text"):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
        return "".join(parts)
    return ""


def compact_text(value: Any, *, limit: int = 1200) -> str:
    """Return a bounded plain-text representation for role-model inputs."""

    if value is None:
        text = ""
    elif isinstance(value, str):
        text = value
    elif isinstance(value, list):
        text = "\n".join(compact_text(item, limit=limit) for item in value)
    elif isinstance(value, dict):
        text = str(value.get("text") or value.get("content") or value)
    else:
        text = str(value)
    stripped = text.strip()
    if len(stripped) <= limit:
        return stripped
    omitted = len(stripped) - limit
    return f"{stripped[:limit].rstrip()}... [truncated {omitted} chars]"


def contains_forbidden_fact_claim(text: str, allowed_facts: list[str] | tuple[str, ...] = ()) -> bool:
    """Return True when text appears to assert an unverified external fact."""

    stripped = text.strip()
    if not stripped:
        return False
    for fact in allowed_facts:
        if stripped == fact.strip():
            return False
    return any(pattern.search(stripped) for pattern in _FORBIDDEN_FACT_PATTERNS)


def summarize_tool_calls(tool_calls: Any) -> list[dict[str, Any]]:
    """Summarize tool calls without copying argument values or tool outputs."""

    if not isinstance(tool_calls, list):
        return []

    def summarize(call: Any) -> dict[str, Any]:
        if not isinstance(call, dict):
            return {"type": "unknown"}
        function = call.get("function") if isinstance(call.get("function"), dict) else {}
        raw_arguments = function.get("arguments") or ""
        argument_keys: list[str] = []
        if isinstance(raw_arguments, str) and raw_arguments.strip().startswith("{"):
            try:
                parsed = json.loads(raw_arguments)
                if isinstance(parsed, dict):
                    argument_keys = [str(key) for key in list(parsed.keys())[:12]]
            except json.JSONDecodeError:
                argument_keys = []
        return {
            "type": str(call.get("type") or "function"),
            "name": str(function.get("name") or call.get("name") or "unknown"),
            "argument_keys": argument_keys,
        }

    return [summarize(call) for call in tool_calls[:8]]


def serialize_messages(messages: list[dict], *, limit: int = 20) -> str:
    """Serialize raw messages for legacy roles. Do not use for v1.3 fact boundaries."""

    selected = messages[-limit:]
    normalized = [
        {
            "role": message.get("role"),
            "content": message.get("content"),
            "reasoning_content": message.get("reasoning_content"),
            "tool_calls": message.get("tool_calls"),
        }
        for message in selected
    ]
    return json.dumps(normalized, ensure_ascii=False)


def _compact_role_messages(messages: list[dict], role: str, *, limit: int) -> list[str]:
    values: list[str] = []
    for message in messages:
        if message.get("role") != role:
            continue
        compacted = compact_text(message.get("content"), limit=limit)
        if compacted:
            values.append(compacted)
    return values


def build_role_context(messages: list[dict], *, limit: int = 8) -> dict[str, Any]:
    """Build a compact stance boundary for v1.3 role models.

    User text is an objective/instruction, not verified evidence. Tool outputs
    and assistant summaries of tool outputs are intentionally not facts.
    """

    selected = messages[-limit:]
    user_context = _compact_role_messages(selected, "user", limit=900)
    assistant_context = [
        {
            "decision": compact_text(message.get("content"), limit=500),
            "reasoning": compact_text(message.get("reasoning_content"), limit=300),
            "tool_calls": summarize_tool_calls(message.get("tool_calls")),
            "is_untrusted_prior_model_text": True,
        }
        for message in selected
        if message.get("role") == "assistant"
    ]
    tool_result_count = sum(1 for message in selected if message.get("role") == "tool")
    latest_user_request = next(
        (
            compacted
            for compacted in (
                compact_text(message.get("content"), limit=1500)
                for message in reversed(messages)
                if message.get("role") == "user"
            )
            if compacted
        ),
        "",
    )
    allowed_facts = [
        "Tool result contents are intentionally omitted; do not infer file contents, command output, or test results from them."
    ]
    return {
        "latest_user_request": latest_user_request,
        "user_objective": latest_user_request,
        "recent_user_context": user_context[-3:],
        "recent_assistant_context": assistant_context[-2:],
        "tool_result_count_omitted": tool_result_count,
        "expected_stance": "Move the next step toward the latest user request while staying conservative about factual claims.",
        "allowed_facts": allowed_facts,
        "forbidden_facts": [
            "User requests are objectives, not evidence that files were read, commands ran, tests passed, logs showed details, or APIs returned values.",
            "Do not claim to have read files unless that fact appears verbatim in allowed_facts.",
            "Do not claim to have run commands unless that fact appears verbatim in allowed_facts.",
            "Do not invent test results, log contents, API responses, or repository state.",
            "Do not add new task facts; only adjust stance, tone, and self-rationalization.",
        ],
        "omission_notice": _TEXT_OMITTED,
    }


def build_calibration_context(role_context: dict[str, Any]) -> dict[str, Any]:
    """Reduce role context to the facts calibrator is allowed to use."""

    return {
        "latest_user_request": role_context.get("latest_user_request", ""),
        "user_objective": role_context.get("user_objective", role_context.get("latest_user_request", "")),
        "expected_stance": role_context.get("expected_stance", ""),
        "allowed_facts": list(role_context.get("allowed_facts") or []),
        "forbidden_facts": list(role_context.get("forbidden_facts") or []),
        "omission_notice": role_context.get("omission_notice", _TEXT_OMITTED),
    }


def first_assistant_message(response: dict[str, Any]) -> tuple[int, dict] | None:
    choices = response.get("choices") or []
    for index, choice in enumerate(choices):
        message = choice.get("message")
        if isinstance(message, dict) and message.get("role", "assistant") == "assistant":
            return index, message
    return None
