"""Shared local role-model server and `/correct` backend (llama.cpp GGUF).

This module serves `/chat` and `/chat_api` for manual rewrite experiments while
also exposing a minimal `POST /correct` contract for the offline
`RewriteModelInjectionOperator`.

Inference uses llama.cpp (GGUF) instead of HuggingFace Transformers for 2-4x speedup.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from contextlib import asynccontextmanager, contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

from shared.messages import text_from_content
from shared.model_client import split_hidden_thinking
from shared.tool_call_codec import (
    TOOL_OUTPUT_CONTRACT,
    ToolCallCodecError,
    messages_for_qwen_template,
    normalize_completion_message,
    validate_tool_definitions,
)


# ── GGUF model path discovery ──────────────────────────────────────────────
# Priority: TMI_GGUF_MODEL_PATH (exact file) → TMI_REWRITE_MODEL_DIR (dir w/ *.gguf)
# → auto-scan E:\model\gguf\ → auto-scan D:\model\gguf\
_gguf_path_env = os.getenv("TMI_GGUF_MODEL_PATH", "").strip()
_gguf_model_dir_env = os.getenv("TMI_GGUF_MODEL_DIR", "").strip()
_model_dir_env = os.getenv("TMI_REWRITE_MODEL_DIR", "").strip()
_AUTO_SCAN_DIRS = [Path(d) for d in (r"E:\model\gguf", r"D:\model\gguf")]


def _score_gguf_path(path: Path) -> tuple[int, int, str]:
    """Prefer Q4 no-MTP files, then the newest version/export deterministically."""
    name = path.name.lower()
    preferred = 0 if "no-mtp" in name and "q4" in name else 1
    return preferred, -path.stat().st_mtime_ns, name


def _gguf_from_dir(model_dir: Path) -> Path | None:
    if not model_dir.is_dir():
        return None
    files = sorted(model_dir.glob("*.gguf"), key=_score_gguf_path)
    return files[0] if files else None


def _discover_gguf_path() -> Path | None:
    """Find the GGUF model file from environment variables and auto-scan."""
    if _gguf_path_env:
        # Explicit means authoritative: a typo must fail instead of silently loading
        # a different auto-scanned checkpoint.
        return Path(_gguf_path_env)
    for configured in (_gguf_model_dir_env, _model_dir_env):
        if not configured:
            continue
        candidate = Path(configured)
        if candidate.is_file():
            return candidate
        selected = _gguf_from_dir(candidate)
        if selected is not None:
            return selected
    for scan_dir in _AUTO_SCAN_DIRS:
        selected = _gguf_from_dir(scan_dir)
        if selected is not None:
            return selected
    return None


DEFAULT_GGUF_PATH = _discover_gguf_path()

# ── OpenAI-compatible API config ───────────────────────────────────────────
API_BASE_URL = os.getenv("TMI_REWRITE_API_BASE_URL") or os.getenv(
    "TMI_REWRITE_OAI_BASE", "https://api.siliconflow.cn/v1"
)
API_KEY = os.getenv("TMI_REWRITE_API_KEY") or os.getenv("TMI_REWRITE_OAI_KEY", "")
API_MODEL = os.getenv("TMI_REWRITE_API_MODEL") or os.getenv("TMI_REWRITE_OAI_MODEL", "")

# ── Constants ───────────────────────────────────────────────────────────────
THINK_TAG_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)
# llama.cpp raises ValueError("Requested tokens (X) exceed context window of Y") when
# prompt + max_tokens > n_ctx.  Parsed for an exact-overage retry in generate_chat.
_CTX_OVERFLOW_RE = re.compile(r"Requested tokens \((\d+)\) exceed context window of (\d+)")
# Tokens reserved per message for chat-template role markers (over-estimate for safety).
_PER_MSG_TOKEN_OVERHEAD = 8
# Always leave room for at least a short reply and a safety margin under n_ctx.
_MIN_REPLY_TOKENS = 256
_CTX_SAFETY_MARGIN = 128
DEBUG_SESSION_ID = "recover-sniffer-crash"

# Safe default for the 8 GiB Windows target.  Qwen3.5 advertises 262K natively, but
# Q4 weights plus a 16K KV cache already leave little headroom on the RTX 5060 Laptop.
# Raise only after measuring VRAM; Hybrid V2 must compress rather than rely on truncation.
_GGUF_N_CTX = int(os.getenv("TMI_GGUF_N_CTX", "16384"))
_GGUF_N_GPU_LAYERS = int(os.getenv("TMI_GGUF_N_GPU_LAYERS", "-1"))
_GGUF_N_BATCH = int(os.getenv("TMI_GGUF_N_BATCH", "1024"))
_GGUF_N_THREADS = int(os.getenv("TMI_GGUF_N_THREADS", "0")) or None
_GGUF_FLASH_ATTN = os.getenv("TMI_GGUF_FLASH_ATTN", "true").strip().lower() in {
    "1", "true", "yes", "on",
}
_GGUF_QUEUE_TIMEOUT = float(os.getenv("TMI_GGUF_QUEUE_TIMEOUT", "30"))
_GGUF_MAX_IN_SYSTEM = max(1, int(os.getenv("TMI_GGUF_MAX_IN_SYSTEM", "8")))
_GGUF_PRELOAD = os.getenv("TMI_GGUF_PRELOAD", "true").strip().lower() in {
    "1", "true", "yes", "on",
}

# The old crash probe wrote request previews on every inference.  Keep the probe available,
# but make it explicit: a local model server must not persist user/tool payloads by default.
_DEBUG_EVENTS_ENABLED = os.getenv("TMI_GGUF_DEBUG_EVENTS", "false").strip().lower() in {
    "1", "true", "yes", "on",
}
_DEBUG_PREVIEWS_ENABLED = os.getenv("TMI_GGUF_DEBUG_PREVIEWS", "false").strip().lower() in {
    "1", "true", "yes", "on",
}
_DEBUG_LOG_LOCK = threading.Lock()

# Capture process identity at import time. Re-reading this file inside /health would report
# the newest on-disk edit even when an old Python process was still serving stale code.
_PROCESS_STARTED_AT = time.time()
try:
    _SERVER_SRC_SHA_AT_IMPORT = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
except OSError:
    _SERVER_SRC_SHA_AT_IMPORT = "unknown"

TOOL_HISTORY_CONTRACT = "qwen35-native-toolcalls.1"
_TOOL_TEMPLATE_MARKERS = ("<tool_call>", "<function=", "<tool_response>", "tool_calls")
_MIN_AGENTIC_REPLY_TOKENS = 256

CORRECT_SYSTEM_PROMPT = (
    "你是一个本地 role-model 执行器。"
    "你会收到 source material 和 instruction。"
    "你的任务是严格按照 instruction 完成任务。"
    "如果 instruction 要求 XML、JSON 或其他结构化格式，必须原样遵守。"
    "不要补充解释，不要添加 Markdown，不要偏离 instruction。"
    "不要输出分析过程、计划、拆解步骤、保留/泛化/重写分类或任何中间思考。"
    "只输出满足 instruction 的最终结果。"
)

DEFAULT_SYSTEM_PROMPT = (
    "你是一个 TMI 轨迹步骤改写演示助手。请仅输出改写后的单步文本，不要添加额外解释。"
)


# ── Pydantic models ────────────────────────────────────────────────────────


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=12000)
    system_prompt: str = Field(
        default=DEFAULT_SYSTEM_PROMPT,
        max_length=12000,
    )
    max_new_tokens: int = Field(default=128, ge=16, le=4096)
    temperature: float = Field(default=0.2, ge=0.0, le=1.5)
    top_p: float = Field(default=0.9, ge=0.1, le=1.0)


class CorrectRequest(BaseModel):
    benign: str = Field(..., min_length=1, max_length=12000)
    question: str = Field(..., min_length=1, max_length=4000)
    max_tokens: int = Field(default=512, ge=16, le=4096)


class CorrectResponse(BaseModel):
    answer: str


class OpenAIChatMessage(BaseModel):
    """OpenAI message graph, including native assistant/tool linkage."""

    # Content may be a plain string or multimodal content blocks. Native assistant/tool
    # fields remain explicit so full-context handoff preserves their linkage.
    model_config = ConfigDict(extra="ignore")

    role: Literal["system", "user", "assistant", "tool"]
    content: Any = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    name: str | None = None
    reasoning_content: str | None = None


class OpenAIChatCompletionRequest(BaseModel):
    # Unknown provider extensions remain ignorable, but action-bearing fields are explicit.
    model_config = ConfigDict(extra="ignore")

    model: str = Field(default="local-calibration")
    messages: list[OpenAIChatMessage]
    stream: bool = False
    max_tokens: int | None = Field(default=None, gt=0)
    max_completion_tokens: int | None = Field(default=None, gt=0)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=20, ge=0)
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any = None
    parallel_tool_calls: bool | None = None


# ── Utility functions ──────────────────────────────────────────────────────


def strip_thinking_tags(text: str) -> str:
    cleaned = THINK_TAG_PATTERN.sub("", text).strip()
    return cleaned or text.strip()


def split_thinking_and_response(text: str) -> tuple[str, str]:
    return split_hidden_thinking(text)


def format_error_detail(detail: Any) -> str:
    if isinstance(detail, str):
        return detail
    if isinstance(detail, dict):
        for key in ("message", "detail", "error"):
            value = detail.get(key)
            if isinstance(value, str) and value.strip():
                return value
        try:
            return json.dumps(detail, ensure_ascii=False)
        except TypeError:
            return str(detail)
    if isinstance(detail, list):
        parts = [format_error_detail(item) for item in detail]
        return "; ".join(part for part in parts if part)
    return str(detail)


#region debug-point recover-sniffer-crash-helpers
def _debug_log_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    log_dir = repo_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"trae-debug-log-{DEBUG_SESSION_ID}.ndjson"


def _sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def _append_debug_event(event_type: str, **fields: Any) -> None:
    if not _DEBUG_EVENTS_ENABLED:
        return
    payload = {
        "ts": round(time.time(), 6),
        "event_type": event_type,
        "pid": os.getpid(),
        "thread_id": threading.get_ident(),
        **fields,
    }
    # Diagnostics must never become an inference dependency (read-only filesystems, full
    # disks, and antivirus locks are all common on Windows laptops).
    try:
        with _DEBUG_LOG_LOCK, _debug_log_path().open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except OSError:
        return


def _message_debug_meta(message: str) -> dict[str, Any]:
    meta = {
        "message_len": len(message),
        "message_sha1": _sha1_text(message),
    }
    if _DEBUG_PREVIEWS_ENABLED:
        meta["message_preview"] = message[:120]
    return meta


def _system_debug_meta(system_prompt: str) -> dict[str, Any]:
    meta = {
        "system_len": len(system_prompt),
        "system_sha1": _sha1_text(system_prompt),
    }
    if _DEBUG_PREVIEWS_ENABLED:
        meta["system_preview"] = system_prompt[:120]
    return meta
#endregion


def _openai_request_from_correct(request: CorrectRequest) -> OpenAIChatCompletionRequest:
    #region debug-point recover-sniffer-crash-correct-request
    debug_fields: dict[str, Any] = {
        "benign_len": len(request.benign),
        "benign_sha1": _sha1_text(request.benign),
        "question_len": len(request.question),
        "question_sha1": _sha1_text(request.question),
    }
    if _DEBUG_PREVIEWS_ENABLED:
        debug_fields.update(
            benign_preview=request.benign[:120],
            question_preview=request.question[:120],
        )
    _append_debug_event("correct_request_received", **debug_fields)
    #endregion
    return OpenAIChatCompletionRequest(
        model="local-calibration",
        messages=[
            OpenAIChatMessage(
                role="system",
                content=f"{CORRECT_SYSTEM_PROMPT}\n\n{request.question}",
            ),
            OpenAIChatMessage(role="user", content=request.benign),
        ],
        stream=False,
        max_tokens=request.max_tokens,
        temperature=0.1,
        top_p=0.9,
    )


def _resolve_max_new_tokens(max_tokens: int | None) -> int:
    """Resolve an output-token budget. No upper Pydantic bound; cap here so a huge
    client value (Claude Code sends 64000) can't exceed the context budget."""
    return min(4096 if max_tokens is None else max_tokens, 4096)


def _estimate_prompt_tokens(messages: list[dict[str, Any]], count_tokens: Callable[[str], int]) -> int:
    """Estimate prompt tokens incl. per-message chat-template overhead (over-estimates)."""
    return (
        sum(count_tokens(m["content"]) + _PER_MSG_TOKEN_OVERHEAD for m in messages)
        + _PER_MSG_TOKEN_OVERHEAD
    )


def _fit_messages_to_context(
    messages: list[dict[str, Any]],
    *,
    max_prompt_tokens: int,
    count_tokens: Callable[[str], int],
    truncate_to_tokens: Callable[[str, int], str],
) -> list[dict[str, Any]]:
    """Shrink a conversation so its estimated prompt fits ``max_prompt_tokens``.

    Standard sliding-window: (1) drop the oldest non-system, non-last messages;
    (2) if still too big (e.g. a single huge system prompt — Claude Code's tool/skill
    catalog), head-truncate the largest remaining message by tokens.  System messages
    and the latest message are always kept (truncated, never dropped).  Content is only
    ever shortened by whole tokens — never pattern-edited.  Pure & unit-testable via the
    injected ``count_tokens`` / ``truncate_to_tokens`` callables.
    """
    msgs = [dict(m) for m in messages]
    if not msgs or _estimate_prompt_tokens(msgs, count_tokens) <= max_prompt_tokens:
        return msgs

    # 1) drop oldest droppable messages (keep system + the last message)
    last_i = len(msgs) - 1
    dropped: set[int] = set()
    for i, m in enumerate(msgs):
        if m["role"] == "system" or i == last_i:
            continue
        if _estimate_prompt_tokens([x for j, x in enumerate(msgs) if j not in dropped], count_tokens) <= max_prompt_tokens:
            break
        dropped.add(i)
    msgs = [m for j, m in enumerate(msgs) if j not in dropped]

    # 2) head-truncate the largest remaining message until it fits
    for _ in range(len(msgs) + 2):
        overage = _estimate_prompt_tokens(msgs, count_tokens) - max_prompt_tokens
        if overage <= 0:
            break
        li = max(range(len(msgs)), key=lambda i: count_tokens(msgs[i]["content"]))
        target = max(0, count_tokens(msgs[li]["content"]) - overage - _PER_MSG_TOKEN_OVERHEAD)
        msgs[li] = {**msgs[li], "content": truncate_to_tokens(msgs[li]["content"], target)}
    return msgs


def _chat_request_from_openai(request: OpenAIChatCompletionRequest) -> ChatRequest:
    """Adapt an OpenAI request to the single-turn rewrite ChatRequest used by /correct.

    Only /correct routes through here (it always sends exactly system + one user message,
    both plain strings bounded by CorrectRequest).  The general /v1/chat/completions chat
    path bypasses ChatRequest entirely — see v1_chat_completions / generate_chat.
    """
    norm = [(m.role, text_from_content(m.content)) for m in request.messages]
    #region debug-point recover-sniffer-crash-openai-request
    _append_debug_event(
        "openai_request_received",
        model=request.model,
        stream=request.stream,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        message_count=len(norm),
        roles=[role for role, _ in norm],
        message_lengths=[len(text) for _, text in norm],
        message_sha1s=[_sha1_text(text) for _, text in norm],
    )
    #endregion
    system_parts = [text for role, text in norm if role == "system" and text.strip()]
    user_texts = [text for role, text in norm if role == "user"]
    if not user_texts or not user_texts[-1].strip():
        raise HTTPException(
            status_code=400,
            detail="local calibration requires at least one user message with non-empty content",
        )
    chat_request = ChatRequest(
        system_prompt="\n\n".join(system_parts) or DEFAULT_SYSTEM_PROMPT,
        message=user_texts[-1],
        max_new_tokens=_resolve_max_new_tokens(request.max_tokens),
        temperature=max(0.0, min(request.temperature, 1.5)),
        top_p=max(0.1, min(request.top_p, 1.0)),
    )
    #region debug-point recover-sniffer-crash-chat-request
    _append_debug_event(
        "chat_request_built",
        max_new_tokens=chat_request.max_new_tokens,
        temperature=chat_request.temperature,
        top_p=chat_request.top_p,
        **_system_debug_meta(chat_request.system_prompt),
        **_message_debug_meta(chat_request.message),
    )
    #endregion
    return chat_request


def _openai_completion_response(result: dict[str, Any], model: str) -> dict[str, Any]:
    raw_message = result.get("message")
    if isinstance(raw_message, dict):
        message = dict(raw_message)
    else:
        content = str(result.get("response") or "").strip()
        message = {"role": "assistant", "content": content}
    message.setdefault("role", "assistant")
    has_calls = bool(message.get("tool_calls"))
    finish_reason = result.get("finish_reason") or ("tool_calls" if has_calls else "stop")

    token_count = int(result.get("tokens_generated") or len(str(message.get("content") or "")))
    raw_usage = result.get("usage")
    if isinstance(raw_usage, dict):
        prompt_tokens = int(raw_usage.get("prompt_tokens") or 0)
        completion_tokens = int(raw_usage.get("completion_tokens") or token_count)
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": int(raw_usage.get("total_tokens") or prompt_tokens + completion_tokens),
        }
    else:
        usage = {
            "prompt_tokens": 0,
            "completion_tokens": token_count,
            "total_tokens": token_count,
        }
    return {
        "id": str(result.get("id") or f"chatcmpl_local_{int(time.time() * 1000)}"),
        "object": "chat.completion",
        "created": int(result.get("created") or time.time()),
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": usage,
    }


def _pseudo_stream_chunks(result_data: dict[str, Any]):
    """Emit one complete, OpenAI-shaped SSE response, including tool calls."""
    choices = result_data.get("choices") or []
    choice = choices[0] if choices else {}
    msg = choice.get("message") or {}
    content = msg.get("content")
    base = {
        "id": result_data.get("id", "chatcmpl-local"),
        "object": "chat.completion.chunk",
        "created": result_data.get("created", 0),
        "model": result_data.get("model", "local-calibration"),
    }

    def chunk(delta: dict[str, Any], finish: str | None = None) -> bytes:
        payload = {
            **base,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")

    yield chunk({"role": "assistant"})
    reasoning = msg.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning:
        yield chunk({"reasoning_content": reasoning})
    if isinstance(content, str) and content:
        yield chunk({"content": content})
    tool_calls = msg.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        stream_calls = [{"index": i, **call} for i, call in enumerate(tool_calls)]
        yield chunk({"tool_calls": stream_calls})
    yield chunk({}, str(choice.get("finish_reason") or "stop"))
    yield b"data: [DONE]\n\n"


def _openai_error_response(
    status_code: int,
    *,
    message: str,
    error_type: str,
    code: str,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        headers=headers,
        content={
            "error": {
                "message": message,
                "type": error_type,
                "param": None,
                "code": code,
            }
        },
    )


def extract_correct_answer(result: dict[str, Any]) -> str:
    answer = str(result.get("response") or "").strip()
    if "</think>" in answer:
        answer = answer.split("</think>", 1)[1].strip()
    elif "<think>" in answer:
        answer = strip_thinking_tags(answer)
    if not answer:
        raise HTTPException(status_code=502, detail="模型未返回最终改写结果。")
    return answer


def get_model_compatibility_report(gguf_path: Path | None) -> dict[str, Any]:
    """Check whether the GGUF file looks like a Qwen model."""
    if gguf_path is None:
        return {
            "supported": False,
            "family": "unknown",
            "reason": (
                "未配置 GGUF 模型路径。请设置 TMI_GGUF_MODEL_DIR 指向 .gguf 文件，"
                "或设置 TMI_REWRITE_MODEL_DIR 指向包含 .gguf 文件的目录。"
            ),
        }
    if not gguf_path.is_file():
        return {
            "supported": False,
            "family": "unknown",
            "reason": f"GGUF 文件不存在: {gguf_path}",
        }
    fname = gguf_path.name.lower()
    is_qwen = "qwen" in fname
    return {
        "supported": True,
        "family": "qwen" if is_qwen else "unknown",
        "reason": "" if is_qwen else "文件名中未检测到 qwen 标识，兼容性未验证。",
    }


# ── RewriteModelRunner (llama.cpp GGUF backend) ────────────────────────────


class ContextLengthExceeded(ValueError):
    """Agentic context cannot fit without destroying its native tool graph."""


class InvalidModelToolOutput(RuntimeError):
    """B emitted an action that cannot be represented or safely executed."""


class ModelUnavailable(RuntimeError):
    """The configured local model cannot currently serve this request."""


class ModelBusy(RuntimeError):
    """The bounded local inference queue is full or has expired."""


def _tool_template_verified(metadata: dict[str, Any]) -> bool:
    architecture = str(metadata.get("general.architecture") or "").lower()
    template = str(metadata.get("tokenizer.chat_template") or "")
    return "qwen" in architecture and all(marker in template for marker in _TOOL_TEMPLATE_MARKERS)


def _validate_tool_controls(
    tools: list[dict[str, Any]] | None, tool_choice: Any
) -> str | None:
    """Validate OpenAI tool controls and return a forced function name, if any."""
    validate_tool_definitions(tools)
    names = {
        tool["function"]["name"].strip()
        for tool in tools or []
    }
    if tool_choice is None or tool_choice == "auto":
        return None
    if tool_choice == "none":
        return None
    if tool_choice == "required":
        if not names:
            raise ToolCallCodecError("tool_choice='required' needs at least one tool")
        return None
    if not isinstance(tool_choice, dict):
        raise ToolCallCodecError(f"unsupported tool_choice {tool_choice!r}")
    function = tool_choice.get("function")
    name = function.get("name") if isinstance(function, dict) else None
    if tool_choice.get("type") != "function" or not isinstance(name, str) or not name.strip():
        raise ToolCallCodecError("forced tool_choice must identify one function")
    name = name.strip()
    if name not in names:
        raise ToolCallCodecError(f"tool_choice names unoffered function {name!r}")
    return name


class RewriteModelRunner:
    """Local Qwen GGUF runner with an OpenAI-native tool-call boundary."""

    def __init__(self, gguf_path: Path | None = DEFAULT_GGUF_PATH) -> None:
        self.gguf_path = gguf_path
        self.device = "cuda-offload" if _GGUF_N_GPU_LAYERS != 0 else "cpu"
        name = gguf_path.name.lower() if gguf_path is not None else ""
        quant_match = re.search(r"(q\d(?:_[a-z0-9]+)+)", name)
        self.quantization = quant_match.group(1) if quant_match else "gguf"
        self.load_strategy = "unloaded"
        self.model = None  # llama_cpp.Llama instance
        self._metadata: dict[str, Any] = {}
        self._chat_formatter = None
        self._invoke_lock = threading.Lock()
        self._in_system = threading.BoundedSemaphore(_GGUF_MAX_IN_SYSTEM)
        self._state_lock = threading.Lock()
        self._in_system_count = 0
        self._active_count = 0
        self._queue_rejected_total = 0
        self._queue_timeout_total = 0
        self._load_seconds: float | None = None
        self._loaded_at: float | None = None
        self._startup_probe_tokens: int | None = None
        self._tool_history_verified = False

    def runtime_state(self) -> dict[str, Any]:
        """Small lock-safe operational snapshot for health/metrics endpoints."""
        with self._state_lock:
            in_system = self._in_system_count
            active = self._active_count
            return {
                "active": active,
                "queued": max(0, in_system - active),
                "in_system": in_system,
                "capacity": _GGUF_MAX_IN_SYSTEM,
                "queue_rejected_total": self._queue_rejected_total,
                "queue_timeout_total": self._queue_timeout_total,
                "load_seconds": self._load_seconds,
                "loaded_at": self._loaded_at,
            }

    @contextmanager
    def _model_lease(self):
        if not self._in_system.acquire(blocking=False):
            with self._state_lock:
                self._queue_rejected_total += 1
            raise ModelBusy(
                f"local model queue is full (max_in_system={_GGUF_MAX_IN_SYSTEM})"
            )
        with self._state_lock:
            self._in_system_count += 1
        acquired = False
        try:
            acquired = self._invoke_lock.acquire(timeout=max(0.0, _GGUF_QUEUE_TIMEOUT))
            if not acquired:
                with self._state_lock:
                    self._queue_timeout_total += 1
                raise ModelBusy(
                    f"timed out waiting {_GGUF_QUEUE_TIMEOUT:g}s for local model"
                )
            with self._state_lock:
                self._active_count += 1
            yield
        finally:
            if acquired:
                with self._state_lock:
                    self._active_count = max(0, self._active_count - 1)
                self._invoke_lock.release()
            with self._state_lock:
                self._in_system_count = max(0, self._in_system_count - 1)
            self._in_system.release()

    def load(self) -> None:
        if self.model is not None:
            return
        if self.gguf_path is None:
            raise ModelUnavailable(
                "GGUF model is not configured. Set TMI_GGUF_MODEL_PATH to an exact file "
                "or TMI_GGUF_MODEL_DIR to a directory containing GGUF files."
            )
        if not self.gguf_path.is_file():
            raise ModelUnavailable(f"GGUF model does not exist: {self.gguf_path}")

        kwargs: dict[str, Any] = {
            "model_path": str(self.gguf_path),
            "n_gpu_layers": _GGUF_N_GPU_LAYERS,
            "n_ctx": _GGUF_N_CTX,
            "n_batch": _GGUF_N_BATCH,
            "flash_attn": _GGUF_FLASH_ATTN,
            "verbose": False,
        }
        if _GGUF_N_THREADS is not None:
            kwargs["n_threads"] = _GGUF_N_THREADS
            kwargs["n_threads_batch"] = _GGUF_N_THREADS
        load_started = time.perf_counter()
        try:
            from llama_cpp import Llama
            from llama_cpp.llama_chat_format import Jinja2ChatFormatter

            self.model = Llama(**kwargs)
        except Exception as exc:
            self.model = None
            raise ModelUnavailable(
                f"failed to load GGUF model {self.gguf_path}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        self._metadata = dict(getattr(self.model, "metadata", {}) or {})
        template = str(self._metadata.get("tokenizer.chat_template") or "")
        try:
            if template:
                eos_id = self.model.token_eos()
                bos_id = self.model.token_bos()
                eos_token = (
                    self.model._model.token_get_text(eos_id) if eos_id != -1 else ""
                )
                bos_token = (
                    self.model._model.token_get_text(bos_id) if bos_id != -1 else ""
                )
                self._chat_formatter = Jinja2ChatFormatter(
                    template=template,
                    eos_token=eos_token,
                    bos_token=bos_token,
                    add_generation_prompt=True,
                )
        except Exception as exc:
            model = self.model
            self.model = None
            self._metadata = {}
            self._chat_formatter = None
            close = getattr(model, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
            raise ModelUnavailable(
                f"failed to initialize GGUF chat template: {type(exc).__name__}: {exc}"
            ) from exc
        self.load_strategy = (
            "llama_cpp_gpu" if _GGUF_N_GPU_LAYERS != 0 else "llama_cpp_cpu"
        )
        with self._state_lock:
            self._load_seconds = time.perf_counter() - load_started
            self._loaded_at = time.time()

    def close(self) -> None:
        """Release native llama.cpp/CUDA resources during an orderly app shutdown."""
        with self._invoke_lock:
            model = self.model
            self.model = None
            self._metadata = {}
            self._chat_formatter = None
            self._startup_probe_tokens = None
            self._tool_history_verified = False
            self.load_strategy = "unloaded"
            with self._state_lock:
                self._loaded_at = None
            close = getattr(model, "close", None)
            if callable(close):
                close()

    def verify_native_tool_history(self) -> int:
        """Render a closed tool round with the exact embedded template.

        Metadata marker checks alone can produce a false-ready server.  This probe proves that
        an assistant action, its string arguments, and the paired environment result all survive
        the actual Jinja boundary before the service accepts traffic.
        """
        # Re-verification is fail-closed too: a previous success must not survive a later
        # formatter/tokenizer failure.
        self._tool_history_verified = False
        self._startup_probe_tokens = None
        if self.model is None or self._chat_formatter is None:
            raise ModelUnavailable("GGUF model/chat formatter is not loaded")
        probe_tools = [
            {
                "type": "function",
                "function": {
                    "name": "tmi_probe_read",
                    "description": "Startup-only native tool-history contract probe.",
                    "parameters": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            }
        ]
        probe_messages = self._normalise_messages(
            [
                {"role": "system", "content": "Continue the native tool workflow."},
                {"role": "user", "content": "Inspect the requested artifact."},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_tmi_startup_probe",
                            "type": "function",
                            "function": {
                                "name": "tmi_probe_read",
                                "arguments": '{"path":"TMI_STARTUP_PATH_7d91"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_tmi_startup_probe",
                    "name": "tmi_probe_read",
                    "content": "TMI_STARTUP_RESULT_c42e",
                },
            ]
        )
        try:
            rendered = self._chat_formatter(
                messages=probe_messages,
                tools=probe_tools,
                tool_choice=None,
            )
            prompt = rendered.prompt
            required = (
                "<tool_call>",
                "<function=tmi_probe_read>",
                "<parameter=path>",
                "TMI_STARTUP_PATH_7d91",
                "</parameter>",
                "</function>",
                "</tool_call>",
                "<tool_response>",
                "TMI_STARTUP_RESULT_c42e",
                "</tool_response>",
            )
            missing = [marker for marker in required if marker not in prompt]
            if missing:
                raise ValueError(f"rendered prompt is missing markers: {missing}")

            # Anchor the contract around unique sentinels rather than the first generic
            # marker: Qwen templates may document <tool_call> in their system preamble.
            path_value = prompt.index("TMI_STARTUP_PATH_7d91")
            parameter_open = prompt.rfind("<parameter=path>", 0, path_value)
            parameter_close = prompt.find("</parameter>", path_value)
            function_open = prompt.rfind("<function=tmi_probe_read>", 0, parameter_open)
            call_open = prompt.rfind("<tool_call>", 0, function_open)
            function_close = prompt.find("</function>", parameter_close)
            call_close = prompt.find("</tool_call>", function_close)
            result_value = prompt.index("TMI_STARTUP_RESULT_c42e")
            response_open = prompt.rfind("<tool_response>", 0, result_value)
            response_close = prompt.find("</tool_response>", result_value)
            positions = (
                call_open,
                function_open,
                parameter_open,
                path_value,
                parameter_close,
                function_close,
                call_close,
                response_open,
                result_value,
                response_close,
            )
            if any(position < 0 for position in positions) or list(positions) != sorted(positions):
                raise ValueError("rendered native tool history has invalid causal/tag order")

            parameter_value = prompt[
                parameter_open + len("<parameter=path>") : parameter_close
            ].strip()
            if parameter_value != "TMI_STARTUP_PATH_7d91":
                raise ValueError(
                    "rendered tool argument was not decomposed to one native string parameter"
                )
            response_value = prompt[
                response_open + len("<tool_response>") : response_close
            ].strip()
            if response_value != "TMI_STARTUP_RESULT_c42e":
                raise ValueError("rendered tool result is not paired as native environment evidence")
            tokens = len(
                self.model.tokenize(
                    prompt.encode("utf-8"),
                    add_bos=not rendered.added_special,
                    special=True,
                )
            )
        except Exception as exc:
            raise ModelUnavailable(
                f"native tool-history startup probe failed: {type(exc).__name__}: {exc}"
            ) from exc
        if tokens <= 0:
            raise ModelUnavailable("native tool-history startup probe tokenized to empty input")
        self._startup_probe_tokens = tokens
        self._tool_history_verified = True
        return tokens

    def describe(self) -> dict[str, Any]:
        model_path = str(self.gguf_path) if self.gguf_path is not None else "(unset)"
        metadata = self._metadata
        native_context = next(
            (
                value
                for key, value in metadata.items()
                if isinstance(key, str) and key.endswith(".context_length")
            ),
            None,
        )
        try:
            native_context = int(native_context) if native_context is not None else None
        except (TypeError, ValueError):
            native_context = None
        template = str(metadata.get("tokenizer.chat_template") or "")
        tool_template_verified = _tool_template_verified(metadata)
        try:
            import llama_cpp

            llama_cpp_version = getattr(llama_cpp, "__version__", "unknown")
            gpu_offload_supported = bool(llama_cpp.llama_supports_gpu_offload())
        except Exception:
            llama_cpp_version = "unavailable"
            gpu_offload_supported = False
        path_exists = self.gguf_path is not None and self.gguf_path.is_file()
        loaded = self.model is not None
        exact_prompt_counter = self._chat_formatter is not None
        runtime = self.runtime_state()
        return {
            "ok": path_exists,
            "ready": bool(
                path_exists
                and loaded
                and tool_template_verified
                and exact_prompt_counter
                and self._tool_history_verified
            ),
            "model_path": model_path,
            "model_name": metadata.get("general.name"),
            "architecture": metadata.get("general.architecture"),
            "device": self.device,
            "quantization": self.quantization,
            "load_strategy": self.load_strategy,
            "loaded": loaded,
            "n_ctx": int(self.model.n_ctx()) if loaded else _GGUF_N_CTX,
            "native_context_length": native_context,
            "chat_format": getattr(self.model, "chat_format", None) if loaded else None,
            "agentic_prompt_counter": (
                "embedded_jinja_exact" if exact_prompt_counter else "unverified"
            ),
            "n_gpu_layers": _GGUF_N_GPU_LAYERS,
            "flash_attn": _GGUF_FLASH_ATTN,
            "queue_timeout_seconds": _GGUF_QUEUE_TIMEOUT,
            "max_in_system": _GGUF_MAX_IN_SYSTEM,
            "preload_enabled": _GGUF_PRELOAD,
            "debug_events_enabled": _DEBUG_EVENTS_ENABLED,
            "debug_previews_enabled": _DEBUG_PREVIEWS_ENABLED,
            "runtime": runtime,
            "startup_probe_tokens": self._startup_probe_tokens,
            "startup_tool_history_verified": self._tool_history_verified,
            "llama_cpp_version": llama_cpp_version,
            "gpu_offload_supported": gpu_offload_supported,
            "supports_tools": bool(tool_template_verified and exact_prompt_counter),
            "tool_template_verified": tool_template_verified,
            "tool_history_contract": TOOL_HISTORY_CONTRACT,
            "tool_output_contract": TOOL_OUTPUT_CONTRACT,
            "chat_template_sha": (
                hashlib.sha256(template.encode("utf-8")).hexdigest()[:12]
                if template
                else None
            ),
            "server_src_sha": _SERVER_SRC_SHA_AT_IMPORT,
            "process_started_at": _PROCESS_STARTED_AT,
        }

    def generate(self, request: ChatRequest) -> dict[str, Any]:
        """Single-turn rewrite path (used by /chat and /correct)."""
        return self.generate_chat(
            [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.message},
            ],
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )

    @staticmethod
    def _normalise_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalised: list[dict[str, Any]] = []
        for index, raw in enumerate(messages):
            if not isinstance(raw, dict):
                raise ToolCallCodecError(f"messages[{index}] must be an object")
            role = str(raw.get("role") or "user")
            content = raw.get("content")
            if isinstance(content, list):
                text_parts: list[str] = []
                for part_index, part in enumerate(content):
                    if (
                        not isinstance(part, dict)
                        or part.get("type") not in {"text", "input_text"}
                        or not isinstance(part.get("text"), str)
                    ):
                        raise ToolCallCodecError(
                            f"messages[{index}].content[{part_index}] is not a supported "
                            "text block; multimodal content cannot be losslessly projected"
                        )
                    text_parts.append(part["text"])
                content = "".join(text_parts)
            elif not isinstance(content, (str, type(None))):
                raise ToolCallCodecError(
                    f"messages[{index}].content must be text, text blocks, or null"
                )
            message: dict[str, Any] = {"role": role, "content": content}
            for key in ("tool_calls", "tool_call_id", "name", "reasoning_content"):
                if raw.get(key) is not None:
                    message[key] = raw[key]
            normalised.append(message)
        return messages_for_qwen_template(normalised)

    def _rendered_agentic_prompt_tokens(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None,
        tool_choice: Any,
    ) -> int | None:
        """Count the exact embedded-Jinja prompt when the real formatter is available."""
        if self._chat_formatter is None or self.model is None:
            return None
        try:
            rendered = self._chat_formatter(
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
            )
            return len(
                self.model.tokenize(
                    rendered.prompt.encode("utf-8"),
                    add_bos=not rendered.added_special,
                    special=True,
                )
            )
        except Exception as exc:
            raise ToolCallCodecError(
                f"messages/tools cannot be rendered by the GGUF chat template: {exc}"
            ) from exc

    def generate_chat(
        self,
        messages: list[dict[str, Any]],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int = 20,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        parallel_tool_calls: bool | None = None,
    ) -> dict[str, Any]:
        """Generate from full history while preserving the native action/result graph.

        Text-only rewrite traffic retains the legacy fitting behavior.  Agentic requests
        (tools, assistant tool_calls, or tool results) are never silently truncated:
        overflow is explicit so Hybrid V2 can compress at the proxy boundary.
        """
        forced_tool_name = _validate_tool_controls(tools, tool_choice)
        template_messages = self._normalise_messages(messages)
        last_user = next(
            (
                str(m.get("content") or "")
                for m in reversed(template_messages)
                if m.get("role") == "user"
            ),
            "",
        )
        agentic = bool(tools) or any(
            m.get("tool_calls") or m.get("role") == "tool" for m in template_messages
        )
        if agentic and max_new_tokens < _MIN_AGENTIC_REPLY_TOKENS:
            raise ToolCallCodecError(
                f"agentic max_tokens must be at least {_MIN_AGENTIC_REPLY_TOKENS}"
            )

        lock_wait_start = time.perf_counter()
        with self._model_lease():
            lock_wait_seconds = time.perf_counter() - lock_wait_start
            self.load()
            assert self.model is not None
            if agentic and not _tool_template_verified(self._metadata):
                raise ModelUnavailable(
                    "loaded GGUF does not expose a verified Qwen native tool-call template"
                )
            if agentic and self._chat_formatter is not None and not self._tool_history_verified:
                self.verify_native_tool_history()

            def count_tokens(text: str) -> int:
                if not text:
                    return 0
                return len(
                    self.model.tokenize(
                        text.encode("utf-8"), add_bos=False, special=False
                    )
                )

            def truncate_tokens(text: str, n_tokens: int) -> str:
                if n_tokens <= 0:
                    return ""
                tokens = self.model.tokenize(
                    text.encode("utf-8"), add_bos=False, special=False
                )
                if len(tokens) <= n_tokens:
                    return text
                return self.model.detokenize(tokens[:n_tokens]).decode(
                    "utf-8", errors="ignore"
                )

            n_ctx = int(self.model.n_ctx())
            max_prompt = max(256, n_ctx - _MIN_REPLY_TOKENS - _CTX_SAFETY_MARGIN)
            if agentic:
                fitted = template_messages
                model_tools = tools if tools and tool_choice != "none" else None
                exact_prompt_tokens = self._rendered_agentic_prompt_tokens(
                    fitted,
                    tools=model_tools,
                    tool_choice=tool_choice,
                )
                if exact_prompt_tokens is not None:
                    prompt_est = exact_prompt_tokens
                else:
                    # Test doubles and older llama.cpp builds may not expose the Jinja
                    # formatter.  Keep the fallback explicit and conservative, but the
                    # production Qwen path above counts the exact rendered prompt.
                    material = json.dumps(fitted, ensure_ascii=False, default=str)
                    prompt_est = count_tokens(material) + _PER_MSG_TOKEN_OVERHEAD
                    if model_tools:
                        prompt_est += count_tokens(
                            json.dumps(model_tools, ensure_ascii=False, default=str)
                        )
                available = n_ctx - prompt_est - _CTX_SAFETY_MARGIN
                if available < _MIN_AGENTIC_REPLY_TOKENS:
                    raise ContextLengthExceeded(
                        f"agentic prompt does not fit n_ctx={n_ctx}; "
                        f"prompt_tokens={prompt_est}; "
                        f"required_reply_tokens={_MIN_AGENTIC_REPLY_TOKENS}"
                    )
                eff_max_new = min(max_new_tokens, available)
                truncated = False
            else:
                fitted = _fit_messages_to_context(
                    template_messages,
                    max_prompt_tokens=max_prompt,
                    count_tokens=count_tokens,
                    truncate_to_tokens=truncate_tokens,
                )
                prompt_est = _estimate_prompt_tokens(fitted, count_tokens)
                eff_max_new = max(
                    16, min(max_new_tokens, n_ctx - prompt_est - _CTX_SAFETY_MARGIN)
                )
                truncated = sum(len(str(m.get("content") or "")) for m in fitted) < sum(
                    len(str(m.get("content") or "")) for m in template_messages
                )

            start = time.perf_counter()
            _append_debug_event(
                "llama_invoke_start",
                model_loaded=self.model is not None,
                lock_wait_seconds=round(lock_wait_seconds, 4),
                n_ctx=n_ctx,
                requested_max_new=max_new_tokens,
                effective_max_new=eff_max_new,
                prompt_tokens_est=prompt_est,
                context_truncated=truncated,
                agentic=agentic,
                tool_count=len(tools or []),
                temperature=temperature,
                top_p=top_p,
                message_count=len(fitted),
                roles=[m["role"] for m in fitted],
                **_message_debug_meta(last_user),
            )

            completion = None
            for _attempt in range(3):
                try:
                    completion_kwargs: dict[str, Any] = {
                        "messages": fitted,
                        "max_tokens": eff_max_new,
                        "temperature": temperature,
                        "top_p": top_p,
                        "top_k": top_k,
                    }
                    if tools and tool_choice != "none":
                        completion_kwargs["tools"] = tools
                    if tool_choice is not None:
                        completion_kwargs["tool_choice"] = tool_choice
                    completion = self.model.create_chat_completion(**completion_kwargs)
                    break
                except ValueError as exc:
                    overflow = _CTX_OVERFLOW_RE.search(str(exc))
                    if overflow is None:
                        raise ModelUnavailable(
                            f"llama.cpp rejected inference: {exc}"
                        ) from exc
                    if agentic:
                        raise ContextLengthExceeded(str(exc)) from exc
                    requested, ctx = int(overflow.group(1)), int(overflow.group(2))
                    max_prompt = max(
                        128, max_prompt - (requested - ctx) - _CTX_SAFETY_MARGIN
                    )
                    fitted = _fit_messages_to_context(
                        fitted,
                        max_prompt_tokens=max_prompt,
                        count_tokens=count_tokens,
                        truncate_to_tokens=truncate_tokens,
                    )
                    prompt_est = _estimate_prompt_tokens(fitted, count_tokens)
                    eff_max_new = max(
                        16, min(eff_max_new, ctx - prompt_est - _CTX_SAFETY_MARGIN)
                    )
                    truncated = True
                except Exception as exc:
                    raise ModelUnavailable(
                        f"llama.cpp inference failed: {type(exc).__name__}: {exc}"
                    ) from exc
            if completion is None:
                raise ContextLengthExceeded(
                    "unable to fit prompt within context window after retries"
                )
            elapsed = time.perf_counter() - start

            if not isinstance(completion, dict):
                raise InvalidModelToolOutput("llama.cpp returned a non-object completion")
            choices = completion.get("choices") or []
            if (
                not isinstance(choices, list)
                or not choices
                or not isinstance(choices[0], dict)
                or not isinstance(choices[0].get("message"), dict)
            ):
                raise InvalidModelToolOutput("llama.cpp returned no assistant message")
            choice = choices[0]
            raw_message = dict(choice["message"])
            raw_content = raw_message.get("content")
            raw_text = raw_content if isinstance(raw_content, str) else ""
            thinking_text, visible_text = split_thinking_and_response(raw_text)
            raw_message["content"] = visible_text
            if thinking_text and not raw_message.get("reasoning_content"):
                raw_message["reasoning_content"] = thinking_text
            try:
                message = normalize_completion_message(raw_message, tools)
            except ToolCallCodecError as exc:
                raise InvalidModelToolOutput(str(exc)) from exc
            calls = message.get("tool_calls") or []
            if tool_choice == "none" and calls:
                raise InvalidModelToolOutput(
                    "model emitted a tool call while tool_choice='none'"
                )
            if tool_choice == "required" and not calls:
                raise InvalidModelToolOutput(
                    "model emitted no tool call while tool_choice='required'"
                )
            if forced_tool_name and (
                not calls
                or any(call["function"]["name"] != forced_tool_name for call in calls)
            ):
                raise InvalidModelToolOutput(
                    f"model did not obey forced tool_choice {forced_tool_name!r}"
                )
            if parallel_tool_calls is False and len(calls) > 1:
                raise InvalidModelToolOutput(
                    "model emitted parallel calls while parallel_tool_calls=false"
                )
            raw_usage = completion.get("usage") or {}
            if not isinstance(raw_usage, dict):
                raise InvalidModelToolOutput("llama.cpp returned malformed usage metadata")
            usage = dict(raw_usage)
            tokens_generated = int(
                usage.get("completion_tokens")
                or len(
                    self.model.tokenize(
                        raw_text.encode("utf-8"), add_bos=False, special=False
                    )
                )
            )

            _append_debug_event(
                "llama_invoke_done",
                generation_seconds=round(elapsed, 3),
                choice_count=len(choices),
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                total_tokens=usage.get("total_tokens"),
                tool_call_count=len(message.get("tool_calls") or []),
            )

        tokens_per_second = tokens_generated / elapsed if elapsed else 0.0
        finish_reason = choice.get("finish_reason") or "stop"
        if message.get("tool_calls"):
            finish_reason = "tool_calls"
        elif tokens_generated >= eff_max_new:
            finish_reason = "length"

        return {
            "id": completion.get("id"),
            "created": completion.get("created"),
            "message": message,
            "finish_reason": finish_reason,
            "usage": usage,
            "raw_response": raw_text,
            "thinking": message.get("reasoning_content") or "",
            "response": str(message.get("content") or ""),
            "prompt_preview": last_user[:240],
            "generation_seconds": round(elapsed, 3),
            "tokens_generated": tokens_generated,
            "tokens_per_second": round(tokens_per_second, 2),
            "device": self.device,
            "quantization": self.quantization,
            "load_strategy": self.load_strategy,
        }


# ── ApiRewriteRunner (OpenAI-compatible API backend) ───────────────────────


class ApiRewriteRunner:
    def __init__(self) -> None:
        self.client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL) if API_KEY else None
        self.provider = "openai_compatible_api"
        self.model = API_MODEL

    def describe(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "loaded": bool(API_KEY and self.model),
        }

    def generate(self, request: ChatRequest) -> dict[str, Any]:
        if self.client is None or not self.model:
            raise HTTPException(
                status_code=503,
                detail=(
                    "未配置 API 改写后端。此演示页兼容旧的 TMI_REWRITE_API_* 别名，"
                    "但离线 rewrite_model operator 不读取这组变量。请为本演示设置 "
                    "TMI_REWRITE_API_KEY/TMI_REWRITE_API_MODEL"
                    "（如需自定义服务也请设置 TMI_REWRITE_API_BASE_URL），或复用 "
                    "TMI_REWRITE_OAI_KEY/TMI_REWRITE_OAI_MODEL"
                    "（如需自定义服务也请设置 TMI_REWRITE_OAI_BASE）。"
                ),
            )
        messages = [
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": request.message},
        ]

        start = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            max_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )

        chunks: list[str] = []
        for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                chunks.append(delta)

        elapsed = time.perf_counter() - start
        raw_text = "".join(chunks).strip()
        thinking_text, response_text = split_thinking_and_response(raw_text)

        return {
            "raw_response": raw_text,
            "thinking": thinking_text,
            "response": response_text,
            "prompt_preview": request.message[:240],
            "generation_seconds": round(elapsed, 3),
            "tokens_generated": len(raw_text),
            "model": self.model,
            "provider": self.provider,
        }


# ── Runner singletons ──────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def get_runner() -> RewriteModelRunner:
    return RewriteModelRunner()


@lru_cache(maxsize=1)
def get_api_runner() -> ApiRewriteRunner:
    return ApiRewriteRunner()


# ── FastAPI app factory ────────────────────────────────────────────────────


def create_app(
    runner: RewriteModelRunner | Any | None = None,
    api_runner: ApiRewriteRunner | Any | None = None,
    *,
    preload: bool | None = None,
) -> FastAPI:
    owns_runner = runner is None
    active_runner = get_runner() if runner is None else runner
    active_api_runner = get_api_runner() if api_runner is None else api_runner
    should_preload = (_GGUF_PRELOAD if owns_runner else False) if preload is None else preload

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            if should_preload:
                load = getattr(active_runner, "load", None)
                if not callable(load):
                    raise RuntimeError("configured runner does not support preload")
                load()
                verify = getattr(active_runner, "verify_native_tool_history", None)
                if callable(verify):
                    verify()
                report = active_runner.describe()
                if not report.get("ready"):
                    raise RuntimeError(
                        "GGUF preload completed but native tool-call readiness is unverified"
                    )
            yield
        finally:
            if owns_runner:
                close = getattr(active_runner, "close", None)
                if callable(close):
                    close()

    app = FastAPI(title="TMI Rewrite Demo", version="0.3.0", lifespan=lifespan)

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": format_error_detail(exc.detail)},
        )

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return build_demo_html()

    @app.get("/health")
    def health() -> dict[str, Any]:
        return active_runner.describe()

    @app.get("/livez")
    def livez() -> dict[str, Any]:
        return {"ok": True}

    @app.get("/readyz")
    def readyz() -> Any:
        report = active_runner.describe()
        if not report.get("ready"):
            return JSONResponse(status_code=503, content=report)
        return report

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics() -> str:
        report = active_runner.describe()
        runtime = report.get("runtime") if isinstance(report.get("runtime"), dict) else {}
        values = {
            "tmi_gguf_ready": int(bool(report.get("ready"))),
            "tmi_gguf_loaded": int(bool(report.get("loaded"))),
            "tmi_gguf_inference_active": int(runtime.get("active") or 0),
            "tmi_gguf_queue_depth": int(runtime.get("queued") or 0),
            "tmi_gguf_queue_rejected_total": int(runtime.get("queue_rejected_total") or 0),
            "tmi_gguf_queue_timeout_total": int(runtime.get("queue_timeout_total") or 0),
            "tmi_gguf_model_load_seconds": float(runtime.get("load_seconds") or 0.0),
        }
        return "".join(f"{name} {value}\n" for name, value in values.items())

    @app.get("/config")
    def config() -> dict[str, Any]:
        return active_runner.describe()

    @app.get("/config_api")
    def config_api() -> dict[str, Any]:
        return active_api_runner.describe()

    @app.post("/chat")
    def chat(request: ChatRequest) -> dict[str, Any]:
        return active_runner.generate(request)

    @app.get("/v1/models")
    def v1_models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": "local-calibration",
                    "object": "model",
                    "owned_by": "tmi-local",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    def v1_chat_completions(request: OpenAIChatCompletionRequest) -> Any:
        if not request.messages:
            return _openai_error_response(
                400,
                message="messages must not be empty",
                error_type="invalid_request_error",
                code="invalid_request",
            )
        messages = [
            message.model_dump(exclude_none=True) for message in request.messages
        ]
        try:
            result = active_runner.generate_chat(
                messages,
                max_new_tokens=_resolve_max_new_tokens(
                    request.max_completion_tokens or request.max_tokens
                ),
                temperature=max(0.0, min(request.temperature, 1.5)),
                top_p=max(0.1, min(request.top_p, 1.0)),
                top_k=request.top_k,
                tools=request.tools,
                tool_choice=request.tool_choice,
                parallel_tool_calls=request.parallel_tool_calls,
            )
        except ToolCallCodecError as exc:
            return _openai_error_response(
                400,
                message=str(exc),
                error_type="invalid_request_error",
                code="invalid_request",
            )
        except ContextLengthExceeded as exc:
            return _openai_error_response(
                413,
                message=str(exc),
                error_type="invalid_request_error",
                code="context_length_exceeded",
            )
        except InvalidModelToolOutput as exc:
            return _openai_error_response(
                502,
                message=str(exc),
                error_type="server_error",
                code="invalid_model_tool_output",
            )
        except ModelUnavailable as exc:
            return _openai_error_response(
                503,
                message=str(exc),
                error_type="server_error",
                code="model_unavailable",
            )
        except ModelBusy as exc:
            return _openai_error_response(
                429,
                message=str(exc),
                error_type="rate_limit_error",
                code="model_busy",
                headers={"Retry-After": str(max(1, int(_GGUF_QUEUE_TIMEOUT)))},
            )
        response_data = _openai_completion_response(result, request.model)
        if request.stream:
            return StreamingResponse(
                _pseudo_stream_chunks(response_data),
                media_type="text/event-stream",
            )
        return response_data

    @app.post("/correct", response_model=CorrectResponse)
    def correct(request: CorrectRequest) -> CorrectResponse:
        openai_request = _openai_request_from_correct(request)
        chat_request = _chat_request_from_openai(openai_request)
        result = active_runner.generate(chat_request)
        return CorrectResponse(answer=extract_correct_answer(result))

    @app.post("/chat_api")
    def chat_api(request: ChatRequest) -> dict[str, Any]:
        return active_api_runner.generate(request)

    return app


app = create_app()


# ── HTML demo page ─────────────────────────────────────────────────────────


def build_demo_html() -> str:
    return """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TMI Rewrite Demo (llama.cpp GGUF)</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 0; background: #0f172a; color: #e2e8f0; }
    .wrap { max-width: 1400px; margin: 0 auto; padding: 24px; }
    .card { background: #111827; border: 1px solid #334155; border-radius: 14px; padding: 20px; margin-bottom: 16px; }
    textarea, input { width: 100%; box-sizing: border-box; background: #020617; color: #e2e8f0; border: 1px solid #475569; border-radius: 10px; padding: 12px; }
    textarea { min-height: 120px; resize: vertical; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    .panels { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; align-items: start; }
    .panel-title { margin-top: 0; }
    button { background: #2563eb; color: white; border: 0; border-radius: 10px; padding: 12px 16px; cursor: pointer; }
    pre { white-space: pre-wrap; word-break: break-word; background: #020617; border-radius: 10px; padding: 12px; }
    .meta { color: #94a3b8; font-size: 14px; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>TMI 改写演示页 / llama.cpp GGUF 后端</h1>
      <p class="meta">本地模型使用 llama.cpp GGUF 推理（GPU offload）。API 面板走 OpenAI-compatible 服务。</p>
    </div>
    <div class="card">
      <label for="system_prompt">系统提示词</label>
      <textarea id="system_prompt">你是一个 TMI 轨迹步骤改写演示助手。请仅输出改写后的单步文本，不要添加额外解释。</textarea>
      <label for="message">输入文本</label>
      <textarea id="message">请把这一步改写成更自然但仍保持同一语义：我会先检查日志，再决定是否继续执行。</textarea>
      <div class="row">
        <div>
          <label for="max_new_tokens">max_new_tokens</label>
          <input id="max_new_tokens" type="number" min="16" max="2048" value="128">
        </div>
        <div>
          <label for="temperature">temperature</label>
          <input id="temperature" type="number" step="0.1" value="0.2">
        </div>
      </div>
      <p class="meta">用于单步文本改写演示。性能取决于模型、量化方式、设备与输出长度；此页面不提供统一吞吐承诺。</p>
      <p class="meta">左侧面板显示生成 token 数，右侧面板显示返回字符数；两者统计口径不同，只适合在各自面板内做粗略参考。</p>
      <div style="margin-top: 14px;">
        <button id="submit">发送测试</button>
      </div>
    </div>
    <div class="panels">
      <div class="card">
        <h2 class="panel-title">本地模型演示 (GGUF)</h2>
        <h3>思维提取（若模型返回）</h3>
        <pre id="thinking">等待请求...</pre>
        <h3>模型返回</h3>
        <pre id="output">等待请求...</pre>
        <div id="meta" class="meta"></div>
      </div>
      <div class="card">
        <h2 class="panel-title">OpenAI-compatible API 演示</h2>
        <h3>思维提取（若模型返回）</h3>
        <pre id="api-thinking">等待请求...</pre>
        <h3>模型返回</h3>
        <pre id="api-output">等待请求...</pre>
        <div id="api-meta" class="meta"></div>
      </div>
    </div>
  </div>
  <script>
    const button = document.getElementById("submit");
    const thinking = document.getElementById("thinking");
    const output = document.getElementById("output");
    const meta = document.getElementById("meta");
    const apiThinking = document.getElementById("api-thinking");
    const apiOutput = document.getElementById("api-output");
    const apiMeta = document.getElementById("api-meta");

    async function runPanel(url, thinkingNode, outputNode, metaNode, metaFormatter) {
      const response = await fetch(url, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
          system_prompt: document.getElementById("system_prompt").value,
          message: document.getElementById("message").value,
          max_new_tokens: Number(document.getElementById("max_new_tokens").value || 128),
          temperature: Number(document.getElementById("temperature").value || 0.2)
        })
      });
      const payload = await response.json();
      if (!response.ok) {
        const detail = typeof payload.detail === "string"
          ? payload.detail
          : JSON.stringify(payload.detail ?? payload, null, 2);
        throw new Error(detail || "请求失败");
      }
      thinkingNode.textContent = payload.thinking || "(模型未返回思维内容)";
      outputNode.textContent = payload.response || "(无最终回答)";
      metaNode.textContent = metaFormatter(payload);
    }

    button.addEventListener("click", async () => {
      button.disabled = true;
      thinking.textContent = "生成中...";
      output.textContent = "生成中...";
      meta.textContent = "";
      apiThinking.textContent = "生成中...";
      apiOutput.textContent = "生成中...";
      apiMeta.textContent = "";
      try {
        const [localResult, apiResult] = await Promise.allSettled([
          runPanel("/chat", thinking, output, meta, (payload) => `耗时 ${payload.generation_seconds}s | 生成 ${payload.tokens_generated} tokens | ${payload.tokens_per_second ?? "?"} tok/s | ${payload.device} | ${payload.quantization}`),
          runPanel("/chat_api", apiThinking, apiOutput, apiMeta, (payload) => `耗时 ${payload.generation_seconds}s | 返回 ${payload.tokens_generated} 字符 | ${payload.provider} | ${payload.model}`)
        ]);

        if (localResult.status === "rejected") {
          thinking.textContent = "(请求失败)";
          output.textContent = localResult.reason instanceof Error ? localResult.reason.message : JSON.stringify(localResult.reason, null, 2);
        }
        if (apiResult.status === "rejected") {
          apiThinking.textContent = "(请求失败)";
          apiOutput.textContent = apiResult.reason instanceof Error ? apiResult.reason.message : JSON.stringify(apiResult.reason, null, 2);
        }
      } catch (error) {
        thinking.textContent = "(请求失败)";
        output.textContent = error instanceof Error ? error.message : JSON.stringify(error, null, 2);
        apiThinking.textContent = "(请求失败)";
        apiOutput.textContent = error instanceof Error ? error.message : JSON.stringify(error, null, 2);
      } finally {
        button.disabled = false;
      }
    });
  </script>
</body>
</html>"""
