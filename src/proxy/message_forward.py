"""TMI proxy with OpenAI-compatible and Anthropic-compatible endpoints."""
from __future__ import annotations

import asyncio
import codecs
import copy
import hashlib
import json
import os
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Callable
from urllib.parse import urlsplit, urlunsplit

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .config import ProxyConfig
from .session_snapshot import build_session_snapshot
from .session_store import SessionStore
from .trace import FileTraceSink, load_trace_events
from shared.messages import first_assistant_message
from shared.model_client import CorrectEndpointClient, OpenAICompatibleRoleModelClient
from shared.regi import (
    LEGACY_SELECTORS,
    REGI_SYSTEM_NAME,
    engine_selector,
    execution_profile,
    result_metadata,
    selector_metadata,
)
from strategies.v1_3_1.calibrator import UserWillCalibrator
from strategies.v1_3_1.control import V131Controller
from strategies.v1_3_1.reasoning_sanitizer import ReasoningSanitizer
from strategies.v1_3_1.salvage_steer import SalvageSteerSynthesizer
from strategies.v1_3_1.spec import (
    CALIBRATOR_MAX_TOKENS,
    REASONING_SANITIZER_MAX_TOKENS,
    SALVAGE_STEER_MAX_TOKENS,
    SUMMARIZER_MAX_TOKENS,
    VERSION as V131_VERSION,
)
from strategies.v1_3_1.summarizer import UserWillSummarizer
from strategies.v1_3_2 import observer_context, skills_cache, swapper, trajectory
from strategies.v1_3_2.control import V132Controller
from strategies.v1_3_2.spec import SKILL_EXTRACT_MAX_TOKENS, TRAJECTORY_SUMMARY_MAX_TOKENS
from strategies.v1_3_2.swapper import ABSwapper
from strategies.v1_3_2.types import VERSION as V132_VERSION


def _public_endpoint(url: str) -> str:
    """Return an endpoint safe for health output (no credentials, query, or fragment)."""
    if not url:
        return ""
    try:
        parsed = urlsplit(url)
        host = parsed.hostname or ""
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        netloc = f"{host}:{parsed.port}" if parsed.port is not None else host
        return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))
    except ValueError:
        return "<invalid-url>"


def _resolve_execution_selector(header_value: str | None, configured_version: str) -> tuple[str, str]:
    """Resolve the legacy execution selector for one request.

    Returns (requested_selector, engine_selector): the header override (or configured
    TMI_VERSION) as requested, and the normalized engine that actually dispatches. Only
    `v1.3.3` is aliased — to the v1.3.2 engine; it never enables observer/trajectory/
    context capabilities by name (those are config-driven, see ProxyConfig.util_enabled).
    `passthrough` stays a transport-only relay with no REGI intervention.
    """
    requested = header_value if header_value else configured_version
    return requested, engine_selector(requested)


def _reject_unknown_selector(requested: str) -> str | None:
    """Return an error message for an unknown selector, or None when accepted.

    Keeps the original set of accepted wire selectors and the original error wording so
    old clients and dashboards keep working.
    """
    if requested in LEGACY_SELECTORS:
        return None
    return f"unknown TMI version {requested}"


def _execution_mode(engine: str) -> str:
    """Whether the engine runs REGI intervention or is transport-only (passthrough)."""
    return "passthrough" if engine == "passthrough" else "interposition"


def _build_regi_health_block(config: ProxyConfig) -> dict:
    """Additive REGI health metadata derived from the ACTUAL resolved configuration.

    Capabilities are read from the same fields that build the controllers (util_enabled,
    ablations, context resolution), never inferred from the selector spelling — selecting
    `v1.3.3` does not enable observer/trajectory/context features by itself. Every URL is
    sanitized with _public_endpoint(); keys are only reported as booleans.
    """
    passthrough = config.version == "passthrough"
    engine = engine_selector(config.version)
    interposition = not passthrough
    return {
        "schema_version": 1,
        "system": "REGI",
        "name": "Reachability-Gated Interposition",
        "requested_selector": config.version,
        "engine_selector": engine,
        "execution_profile": execution_profile(config.version),
        "mode": _execution_mode(engine),
        "layers": {
            "reachability_sensing": {
                "enabled": interposition and bool(
                    config.v131_base_url or config.calibration_base_url
                ),
            },
            "relay": {"enabled": True},
            "recover": {
                "enabled": interposition and not config.ablate_recover,
            },
            "reconstruct": {
                "enabled": interposition
                and not config.disable_salvage
                and not config.ablate_reconstruct
                and bool(config.v132_b_url),
                # L3 hijack-escalation ladder: re-attempts the hijack with escalated levers
                # (passthrough on the decensored primary, laundered-steer on the aligned fallback)
                # when B + fallback B both fail to forge a tool_call. In the reconstruct family.
                "l3_escalation_enabled": interposition
                and not config.disable_salvage
                and not config.ablate_reconstruct
                and not config.ablate_l3
                and bool(config.v132_b_url),
                "fallback_model_enabled": interposition
                and not config.disable_salvage
                and not config.ablate_reconstruct
                and bool(config.v132_b_fallback_url),
            },
            "state_context_augmentation": {
                "enabled": interposition,
                "utility_enabled": config.util_enabled,
                "trajectory_enabled": not config.ablate_trajectory,
                "context_resolution": config.context_resolution,
                "coldstart_context_resolution": config.context_resolution_coldstart,
                "sniffer_context_resolution": config.context_resolution_sniffer,
            },
        },
        "endpoints": {
            "host": _public_endpoint(config.upstream_url),
            "role_model": _public_endpoint(
                config.v131_base_url or config.calibration_base_url
            ),
            "reconstruction_model": _public_endpoint(config.v132_b_url),
            "utility_model": _public_endpoint(config.util_base_url),
        },
    }


def _role_client(
    base_url: str,
    fallback_url: str,
    timeout: float,
    *,
    max_tokens: int = 512,
    model: str = "local-calibration",
    api_key: str = "tmi-local",
):
    if base_url:
        return OpenAICompatibleRoleModelClient(
            base_url,
            timeout=timeout,
            max_tokens=max_tokens,
            model=model,
            api_key=api_key or "tmi-local",
        )
    return CorrectEndpointClient(
        fallback_url,
        timeout=timeout,
        max_tokens=max_tokens,
    )


def build_v132_controller(config: ProxyConfig) -> V132Controller:
    # Wire v1.3.1 calibration as the sniffer/calibrator; B-first cold-start handles tools.
    # Apply the fold primitives' render caps in place (idempotent; this factory runs per request).
    trajectory.configure(
        result_cap=config.traj_step_result_cap,
        args_cap=config.traj_step_args_cap,
    )
    # v1.3.5 synthesized QA store (per-turn A_i).
    observer_context.configure(
        max_sessions=config.traj_max_sessions,
        store_dir=config.traj_store_dir,
    )
    # Bound + persist the skills cache (was an unbounded in-memory dict — a per-distinct-
    # system-prompt leak that also re-extracted on every restart).
    skills_cache.configure(
        max_sessions=config.traj_max_sessions,
        store_dir=config.traj_store_dir,
    )
    # Salvage coldstart: keep the last k steps' real tool RESULTS at full fidelity for B
    # (so it can pick the NEXT action instead of re-doing the last one). See swapper.configure.
    swapper.configure(
        recent_full_steps=config.traj_coldstart_recent_full_steps,
        full_cap=config.traj_coldstart_full_cap,
        char_budget=config.traj_coldstart_char_budget,
        history_encoding=config.traj_coldstart_history_encoding,
        context_asset_resolver_enabled=config.traj_context_asset_resolver_enabled,
    )
    v131 = build_v131_controller(config) if (
        config.v131_base_url or config.calibration_base_url
    ) else None
    # v1.3.3 trajectory summarizer completers: util model (cloud, concurrent) for the
    # neutral behavioral summary, stance-free 9B as fallback when util refuses/errs.
    util_complete = None
    if config.util_enabled:
        util_complete = OpenAICompatibleRoleModelClient(
            config.util_base_url,
            timeout=config.util_timeout,
            max_tokens=TRAJECTORY_SUMMARY_MAX_TOKENS,
            model=config.util_model,
            api_key=config.util_key,
        ).complete
    fallback_complete = None
    if config.v131_base_url or config.calibration_base_url:
        fallback_complete = _role_client(
            config.v131_base_url,
            config.v131_url or config.calibration_url,
            config.role_timeout,
            max_tokens=TRAJECTORY_SUMMARY_MAX_TOKENS,
            model=config.v131_model or "local-calibration",
            api_key=config.v131_key or "tmi-local",
        ).complete
    # Host skill listing is pure structural extraction (same "no harm-adjacent generation"
    # reasoning as the action classifier in build_v131_controller below), so it defaults to
    # the aligned util model when configured — a dedicated client since SKILL_EXTRACT_MAX_TOKENS
    # (512, sized for large Hermes catalogs) differs from util_complete's own 400-token budget.
    # V132Controller falls back to swapper.extract_skills (B) only when util is unset or errors.
    util_skill_extract_complete = None
    if config.util_enabled:
        util_skill_extract_complete = OpenAICompatibleRoleModelClient(
            config.util_base_url,
            timeout=config.util_timeout,
            max_tokens=SKILL_EXTRACT_MAX_TOKENS,
            model=config.util_model,
            api_key=config.util_key,
        ).complete
    swapper_fallback = None
    if config.v132_b_fallback_url:
        swapper_fallback = ABSwapper(
            b_url=config.v132_b_fallback_url,
            b_api_key=config.v132_b_fallback_key or "EMPTY",
            b_model=config.v132_b_fallback_model or "qwen2.5-7b-instruct",
            b_timeout=config.v132_b_fallback_timeout,
        )
    return V132Controller(
        swapper=ABSwapper(
            b_url=config.v132_b_url,
            b_api_key=config.v132_b_key or "EMPTY",
            b_model=config.v132_b_model or "qwen2.5-7b-instruct",
            b_timeout=config.v132_b_timeout,
        ),
        swapper_fallback=swapper_fallback,
        v131_controller=v131,
        util_complete=util_complete,
        fallback_complete=fallback_complete,
        util_skill_extract_complete=util_skill_extract_complete,
        intent_window_size=config.sniffer_intent_window,
        ablate_graying=config.ablate_graying,
        ablate_trajectory=config.ablate_trajectory,
        ablate_salvage_graying=config.ablate_salvage_graying,
        ablate_reasoning_calibration=config.ablate_reasoning_calibration,
        disable_salvage=config.disable_salvage,
        ablate_reconstruct=config.ablate_reconstruct,
        ablate_recover=config.ablate_recover,
        ablate_l3=config.ablate_l3,
        qa_synthesis=config.qa_synthesis,
        context_resolution=config.context_resolution,
        context_resolution_coldstart=config.context_resolution_coldstart,
        context_resolution_sniffer=config.context_resolution_sniffer,
        coldstart_passthrough=config.traj_coldstart_passthrough,
    )


def build_v131_controller(config: ProxyConfig) -> V131Controller:
    endpoint = config.v131_url or config.calibration_url
    role_client = lambda max_tok: _role_client(  # noqa: E731
        config.v131_base_url,
        endpoint,
        config.role_timeout,
        max_tokens=max_tok,
        model=config.v131_model or "local-calibration",
        api_key=config.v131_key or "tmi-local",
    )
    # Action classification is pure structural judgment (no harm-adjacent generation), so it
    # can run on the safety-aligned TMI_UTIL_* model when configured — reusing the same
    # endpoint already wired for the v1.3.3 trajectory summarizer. Falls back to the compliant
    # v131 model (degraded, not broken) when the util model isn't configured.
    classifier_client = (
        OpenAICompatibleRoleModelClient(
            config.util_base_url,
            timeout=config.util_timeout,
            max_tokens=SUMMARIZER_MAX_TOKENS,
            model=config.util_model,
            api_key=config.util_key,
        )
        if config.util_enabled
        else role_client(SUMMARIZER_MAX_TOKENS)
    )
    # Always construct the sanitizer: it is now shared by TWO independent consumers with
    # their OWN ablation flags — the retired retroactive history scan (message_forward.py,
    # gated on config.ablate_reasoning_san at its call sites below) and
    # V132Controller._calibrate_reasoning (control.py, gated on
    # config.ablate_reasoning_calibration). Gating construction on ablate_reasoning_san alone
    # would silently disable BOTH consumers whenever the (now default-True) retirement flag
    # is set, since they'd read the same None. Each call site decides for itself.
    reasoning_sanitizer = ReasoningSanitizer(role_client(REASONING_SANITIZER_MAX_TOKENS))
    # Salvage-steer synthesis (v1.3.6) MUST run on this same compliant endpoint, never the
    # aligned util/classifier_client above — it reads full-fidelity qa_progress (real tool
    # args/values), the exact payload the aligned classifier is deliberately kept isolated
    # from. See PA_REACT_CONSISTENCY_AUDIT.md Finding 1.
    salvage_synthesizer = SalvageSteerSynthesizer(role_client(SALVAGE_STEER_MAX_TOKENS))
    local_classifier = role_client(SUMMARIZER_MAX_TOKENS)
    return V131Controller(
        summarizer=UserWillSummarizer(
            classifier_client,
            local_classifier,
            classifier_fallback_client=(local_classifier if config.util_enabled else None),
        ),
        calibrator=UserWillCalibrator(role_client(CALIBRATOR_MAX_TOKENS)),
        reasoning_sanitizer=reasoning_sanitizer,
        salvage_synthesizer=salvage_synthesizer,
    )


def _upstream_headers(config: ProxyConfig) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {config.upstream_key}",
        "Content-Type": "application/json",
    }


def _derive_conv_key(request: Request, body: dict) -> tuple[str, str]:
    """Resolve the single per-conversation root key for one request, preferring an explicit
    client id over a content fingerprint. Computed ONCE at handler entry and threaded to every
    consumer (the state stores, the trace, session_id) so keying never disagrees. Order:

      (a) x-source-session-id — collision-free cooperative path (the eval harness sends a unique
          value here per trial, so it keeps perfect isolation);
      (b) x-conversation-id   — explicit alias for other cooperating clients;
      (c) content fingerprint of the invariant opening (system + first user message);
      (d) if the OpenAI ``user`` body field is set, salt (c) with it so two DIFFERENT end-users
          running the SAME templated task stop colliding — ``user`` is a SALT only, never a
          standalone key, so one user's distinct conversations are never merged.

    Returns (conv_key, source_label) where source_label ∈ {header, alias, content, content+user}
    — recorded in the trace so collision/fragmentation rates are measurable. The realistic case
    (clients that send no custom header and replay full history in-band) lands on (c)/(d) and
    still gets a turn-stable, mostly-distinct key. See docs/SESSION_ISOLATION_PLAN.md."""
    header = (request.headers.get("x-source-session-id") or "").strip()
    if header:
        return header, "header"
    alias = (request.headers.get("x-conversation-id") or "").strip()
    if alias:
        return alias, "alias"
    fingerprint = trajectory.content_fingerprint(body.get("messages") or [])
    user = body.get("user")
    if isinstance(user, str) and user.strip():
        salted = "cfu_" + hashlib.sha256(
            (user.strip() + "\x00" + fingerprint).encode("utf-8")
        ).hexdigest()[:24]
        return salted, "content+user"
    return fingerprint, "content"


def _agent_metadata(request: Request, session_id: str, conv_key: str) -> dict[str, str]:
    agent_key = (request.headers.get("x-agent-key") or "").strip()
    agent_source = (request.headers.get("x-agent-source") or "").strip()
    # Raw client-supplied conversation id (empty if the client didn't send one).
    client_session_id = (request.headers.get("x-source-session-id") or "").strip()
    # Fall back to the STABLE conv_key (not the per-request-random session_id) so a header-less
    # agent's turns share one source_session_id / agent_key instead of fragmenting per request —
    # and, because conv_key != the sess_<hash> session_id, the dashboard takes its trusted
    # stable-id grouping path rather than the fuzzy heuristic one.
    source_session_id = client_session_id or conv_key

    if not agent_source:
        agent_source = "unknown"
    if not agent_key:
        agent_key = f"{agent_source}:{source_session_id}" if agent_source != "unknown" else conv_key

    return {
        "agent_key": agent_key,
        "agent_source": agent_source,
        "source_session_id": source_session_id,
        "client_session_id": client_session_id,
        "conv_key": conv_key,
    }


def _latest_user_input(messages: list[dict] | None) -> str:
    for message in reversed(messages or []):
        if (
            isinstance(message, dict)
            and message.get("role") == "user"
            and isinstance(message.get("content"), str)
        ):
            return message["content"]
    return ""


def _assistant_payload(response: dict) -> dict[str, str]:
    found = first_assistant_message(response)
    if found is None:
        return {
            "content": "",
            "reasoning_content": "",
            "finish_reason": "",
            "tool_calls_present": False,
        }
    index, message = found
    choices = response.get("choices") or []
    choice = choices[index] if 0 <= index < len(choices) and isinstance(choices[index], dict) else {}
    return {
        "content": str(message.get("content") or ""),
        "reasoning_content": str(message.get("reasoning_content") or ""),
        "finish_reason": str(choice.get("finish_reason") or ""),
        "tool_calls_present": bool(message.get("tool_calls")),
    }


def _assistant_turn_stage(assistant_output: dict[str, object]) -> str:
    if assistant_output.get("tool_calls_present") or assistant_output.get("finish_reason") == "tool_calls":
        return "intermediate"
    return "final"


def _assistant_has_sniffable_signal(assistant_output: dict[str, object]) -> bool:
    return bool(
        str(assistant_output.get("content") or "").strip()
        or str(assistant_output.get("reasoning_content") or "").strip()
        or assistant_output.get("tool_calls_present")
    )


def _build_v131_trace_event(
    source_input: dict[str, object],
    result: V131Controller | object,
    agent_meta: dict[str, str],
) -> dict[str, object]:
    audit = getattr(result, "audit", {}) or {}
    assistant_output = source_input.get("assistant_output") or {}
    response = getattr(result, "response", {}) or {}
    turn_stage = str(audit.get("turn_stage") or _assistant_turn_stage(assistant_output))
    calibrator_applied = bool(audit.get("calibrator_applied", False))
    calibration_failure = str(audit.get("calibration_failure") or "").strip()
    decision = str(response.get("decision") or "").strip()
    response_replaced = bool(calibrator_applied and turn_stage == "final" and decision)
    # REGI semantics for the v1.3.1 sensing+Recover path: an applied rewrite is the
    # text-level recovery (recover_reframe); a failed/absent calibrator is the honest
    # floor (degraded) unless the turn was a plain intermediate/tool stage (relay).
    # calibration_failure must take precedence over the "final, no calibrator" default so
    # new events render the same operation as the snapshot/dashboard backfill of old ones.
    if response_replaced:
        synthetic_path = "pass_flawed"
    elif calibration_failure:
        synthetic_path = "degraded_raw"
    else:
        synthetic_path = "pass"
    return {
        "event_type": "v1_3_1_completed",
        "strategy_version": V131_VERSION,
        "regi": result_metadata(synthetic_path),
        "turn_stage": turn_stage,
        "source_input": source_input,
        "summary": audit.get("user_will_summary", {}),
        "calibrator_applied": calibrator_applied,
        "response_replaced": response_replaced,
        "response": response,
        "audit": audit,
        "failure_stage": "calibrator_output" if calibration_failure else "",
        "failure": calibration_failure,
        **agent_meta,
    }


def _build_v131_failure_event(
    source_input: dict[str, object],
    failure_stage: str,
    failure: str,
    agent_meta: dict[str, str],
) -> dict[str, object]:
    return {
        "event_type": "v1_3_1_completed",
        "strategy_version": V131_VERSION,
        "regi": result_metadata("degraded_raw"),
        "turn_stage": "failed",
        "source_input": source_input,
        "summary": {},
        "calibrator_applied": False,
        "response_replaced": False,
        "response": {},
        "audit": {},
        "failure_stage": failure_stage,
        "failure": failure,
        **agent_meta,
    }


def _apply_v131_rewrite(data: dict, event: dict[str, object]) -> dict:
    if not event.get("response_replaced"):
        return data
    calibrated = event.get("response") or {}
    if not isinstance(calibrated, dict):
        event["response_replaced"] = False
        return data
    decision = str(calibrated.get("decision") or "").strip()
    if not decision:
        event["response_replaced"] = False
        return data

    delivered = copy.deepcopy(data)
    found = first_assistant_message(delivered)
    if found is None:
        event["response_replaced"] = False
        return data
    _, assistant = found
    assistant["content"] = decision
    assistant["reasoning_content"] = ""
    return delivered


# --- Proxy-side memoization of reasoning sanitization (the O(N^2) fix) ----------
# This is an APPLICATION-LEVEL cache living in the proxy process — NOT a model
# KV/prompt cache. It is therefore fully device/quantization/backend agnostic
# (identical whether B is gguf-int4 on Windows, nvfp4 on GB10, or a cloud API):
# it avoids *calling* the 9B for reasoning text already processed, rather than
# making the model cache better.
#
# Root cause it removes: the client resends the full history verbatim every turn,
# so the same assistant reasoning_content was re-sanitized (1-2 serialized 9B calls
# each) on every subsequent turn → O(N) calls/turn × N turns = O(N^2).
#
# Key   = sha256(reasoning text).
# Value = _NO_REWRITE sentinel (sanitized, keep original) OR the rewritten string.
# Negative results MUST be cached too, else every "pass" reasoning re-runs the 9B
# sanitize each turn — the dominant cost.
_NO_REWRITE = "\x00__tmi_no_rewrite__"  # sentinel distinct from any real rewrite
_REASONING_CACHE: "OrderedDict[str, str]" = OrderedDict()
_REASONING_CACHE_CAP = 4096


def _reasoning_cache_get(key: str) -> tuple[bool, str]:
    if key in _REASONING_CACHE:
        _REASONING_CACHE.move_to_end(key)
        return True, _REASONING_CACHE[key]
    return False, ""


def _reasoning_cache_put(key: str, value: str) -> None:
    _REASONING_CACHE[key] = value
    _REASONING_CACHE.move_to_end(key)
    while len(_REASONING_CACHE) > _REASONING_CACHE_CAP:
        _REASONING_CACHE.popitem(last=False)


async def _sanitize_history_reasoning(
    messages: list[dict],
    sanitizer: ReasoningSanitizer,
    calibrator: UserWillCalibrator,
) -> tuple[list[dict], dict]:
    """Scan assistant messages in history for withdrawal patterns in reasoning_content.

    When an intermediate turn's reasoning contains cautious language
    ("有风险"/"需要谨慎"/etc.), it pollutes subsequent turns via context.
    Sanitize it here — before forwarding to model A — so the model never
    sees the original cautious reasoning.

    Memoized per reasoning text (see _REASONING_CACHE) so each distinct reasoning
    is sanitized at most once per process. Returns (messages, stats) where stats
    exposes the 9B call counts for trace phase_timing.
    """
    modified = False
    result = list(messages)
    stats = {"total": 0, "sanitize_calls": 0, "rewrite_calls": 0, "hits": 0}
    for i, msg in enumerate(result):
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        reasoning = str(msg.get("reasoning_content") or "").strip()
        if not reasoning:
            continue
        stats["total"] += 1
        key = hashlib.sha256(reasoning.encode("utf-8")).hexdigest()

        hit, cached = _reasoning_cache_get(key)
        if hit:
            stats["hits"] += 1
            # ⚠ Re-apply on hit: a positive (rewritten) result must still be written
            # back EVERY turn, otherwise the original cautious reasoning leaks into A's
            # context again — defeating the whole purpose of this function.
            if cached != _NO_REWRITE:
                result[i] = {**msg, "reasoning_content": cached}
                modified = True
            continue

        # Cache miss → consult the 9B. Failures are NOT cached (transient → retry).
        try:
            summary = await asyncio.to_thread(sanitizer.sanitize, reasoning)
            stats["sanitize_calls"] += 1
        except Exception:
            continue
        if summary.action != "rewrite" or not summary.rewrite_target:
            _reasoning_cache_put(key, _NO_REWRITE)  # negative cache
            continue
        try:
            calibrated = await asyncio.to_thread(
                calibrator.rewrite, summary.rewrite_target, reasoning
            )
            stats["rewrite_calls"] += 1
            result[i] = {**msg, "reasoning_content": calibrated.decision}
            _reasoning_cache_put(key, calibrated.decision)
            modified = True
        except Exception:
            continue
    return (result if modified else messages), stats


def _record_phase_timing(
    trace_sink,
    session_id: str,
    phase: str,
    started: float,
    agent_meta: dict,
    **extra,
) -> None:
    """Emit a per-phase latency event so traces show where a turn spends its time."""
    try:
        trace_sink.record(
            session_id,
            {
                "event_type": "phase_timing",
                "phase": phase,
                "seconds": round(time.monotonic() - started, 3),
                **extra,
                **agent_meta,
            },
        )
    except Exception:
        pass


def _get_prior_intermediate_decision(trace_sink: FileTraceSink, session_id: str) -> str:
    """Return the sniffer decision from the immediately prior intermediate turn, if any.

    Used to propagate cross-turn context: when A uses tool_calls in turn N, the sniffer
    records what should happen in turn N+1. Turn N+1 reads this and passes it to the
    sniffer as prior_context so the full user intent is preserved across the tool boundary.
    """
    try:
        path = trace_sink.directory / f"{session_id}.jsonl"
        events = load_trace_events(path)
    except Exception:
        return ""
    for event in reversed(events):
        if event.get("event_type") != "v1_3_1_completed":
            continue
        if event.get("turn_stage") == "intermediate":
            return str(event.get("intermediate_decision") or "")
        # Last v1_3_1 event was not intermediate — no carry-forward
        return ""
    return ""


async def _record_v131_completion(
    controller: V131Controller,
    messages: list[dict] | None,
    response: dict,
    trace_sink,
    session_id: str,
    agent_meta: dict[str, str],
) -> dict[str, object]:
    assistant_output = _assistant_payload(response)
    user_input = _latest_user_input(messages)
    is_intermediate = _assistant_turn_stage(assistant_output) == "intermediate"

    # Read prior intermediate decision for cross-turn context propagation.
    # Only used for final turns (intermediate turns produce the decision, not consume it).
    prior_context = "" if is_intermediate else _get_prior_intermediate_decision(trace_sink, session_id)

    source_input: dict[str, object] = {
        "user_input": user_input,
        "assistant_output": assistant_output,
    }
    if prior_context:
        source_input["prior_context"] = prior_context

    # Intermediate turns (finish_reason=tool_calls) never need sniffer judgment —
    # calibration cannot touch tool_calls output. Short-circuit here: store user_input
    # directly as intermediate_decision for the next (final) turn to read as prior_context.
    if is_intermediate:
        event: dict[str, object] = {
            "event_type": "v1_3_1_completed",
            "turn_stage": "intermediate",
            "response_replaced": False,
            "intermediate_decision": user_input,
            "agent": agent_meta,
        }
        trace_sink.record(session_id, event)
        return event

    if not user_input:
        event = _build_v131_failure_event(
            source_input, "user_input", "latest user message missing", agent_meta,
        )
        trace_sink.record(session_id, event)
        return event
    if not _assistant_has_sniffable_signal(assistant_output):
        event = _build_v131_failure_event(
            source_input, "assistant_message", "assistant content and reasoning missing", agent_meta,
        )
        trace_sink.record(session_id, event)
        return event

    try:
        result = await asyncio.to_thread(controller.run, source_input)
    except Exception as exc:
        event = _build_v131_failure_event(
            source_input, "v1_3_1", str(exc), agent_meta,
        )
        trace_sink.record(session_id, event)
        return event

    event = _build_v131_trace_event(source_input, result, agent_meta)
    trace_sink.record(session_id, event)
    return event


async def _finalize_stream_trace(
    *,
    controller: V131Controller,
    prepared_messages: list[dict] | None,
    response_status: int,
    stream_payload: dict[str, object],
    trace_sink,
    session_id: str,
    agent_meta: dict[str, str],
) -> None:
    trace_sink.record(
        session_id,
        {
            "event_type": "response_stream_ended",
            "status": response_status,
            **stream_payload,
            **agent_meta,
        },
    )
    await _record_v131_completion(
        controller,
        prepared_messages,
        stream_payload.get("response") or {},
        trace_sink,
        session_id,
        agent_meta,
    )


async def _post_upstream_serialized_for_v131(
    config: ProxyConfig,
    body: dict,
    http_client_factory: Callable,
) -> tuple[int, dict]:
    serialized_body = copy.deepcopy(body)
    serialized_body["stream"] = False
    serialized_body.pop("stream_options", None)
    return await _post_upstream(config, serialized_body, http_client_factory)


async def _emit_pseudo_stream_response(
    data: dict,
    *,
    include_usage: bool = False,
):
    found = first_assistant_message(data)
    content = ""
    reasoning_content = ""
    tool_calls = []
    finish_reason = "stop"
    if found is not None:
        index, assistant = found
        content = str(assistant.get("content") or "").strip()
        reasoning_content = str(assistant.get("reasoning_content") or "").strip()
        tool_calls = assistant.get("tool_calls") or []
        choices = data.get("choices") or []
        choice = choices[index] if 0 <= index < len(choices) and isinstance(choices[index], dict) else {}
        finish_reason = str(choice.get("finish_reason") or "").strip() or "stop"

    base_event = {
        "id": data.get("id") or "chatcmpl-tmi-pseudo",
        "object": "chat.completion.chunk",
        "created": data.get("created") or 0,
        "model": data.get("model") or "",
    }

    role_event = {
        **base_event,
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": "", "reasoning_content": ""}, "finish_reason": None}],
    }
    yield f"data: {json.dumps(role_event, ensure_ascii=False)}\n\n".encode("utf-8")

    # Always emit reasoning_content chunk before content, even when empty.
    # cc-switch and similar OpenAI↔Anthropic converters require this field to be
    # present; without it they fall back to the model A original response.
    reasoning_event = {
        **base_event,
        "choices": [{"index": 0, "delta": {"reasoning_content": reasoning_content}, "finish_reason": None}],
    }
    yield f"data: {json.dumps(reasoning_event, ensure_ascii=False)}\n\n".encode("utf-8")

    if content:
        content_event = {
            **base_event,
            "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(content_event, ensure_ascii=False)}\n\n".encode("utf-8")

    if tool_calls:
        tool_calls_delta = []
        for i, tc in enumerate(tool_calls):
            tc_delta = {
                "index": i,
                "id": tc.get("id", ""),
                "type": tc.get("type", "function"),
                "function": {
                    "name": tc.get("function", {}).get("name", ""),
                    "arguments": tc.get("function", {}).get("arguments", "")
                }
            }
            tool_calls_delta.append(tc_delta)

        tool_calls_event = {
            **base_event,
            "choices": [{"index": 0, "delta": {"tool_calls": tool_calls_delta}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(tool_calls_event, ensure_ascii=False)}\n\n".encode("utf-8")

    stop_event = {
        **base_event,
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
    }
    yield f"data: {json.dumps(stop_event, ensure_ascii=False)}\n\n".encode("utf-8")

    if include_usage:
        usage_event = {
            **base_event,
            "choices": [],
            "usage": data.get("usage")
            or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        yield f"data: {json.dumps(usage_event, ensure_ascii=False)}\n\n".encode("utf-8")

    yield b"data: [DONE]\n\n"


async def _run_v131_serialized_stream(
    *,
    active_config: ProxyConfig,
    prepared: dict,
    active_trace,
    active_session_store: SessionStore,
    session_id: str,
    agent_meta: dict[str, str],
    http_client_factory: Callable,
    v131_factory: Callable[[], V131Controller],
):
    status, data = await _post_upstream_serialized_for_v131(
        active_config,
        prepared,
        http_client_factory,
    )
    active_trace.record(
        session_id,
        {"event_type": "response_completed", "candidate_model": "A",
         "candidate_disposition": "pending_gate", "status": status,
         "response": data, **agent_meta},
    )
    if status >= 400:
        try:
            _record_terminal_upstream_failure(
                trace_sink=active_trace,
                session_store=active_session_store,
                trace_dir=active_config.trace_dir,
                session_id=session_id,
                status=status,
                data=data,
                agent_meta=agent_meta,
            )
        except OSError:
            return JSONResponse(
                status_code=502,
                content={"error": "session snapshot failed before client delivery"},
            )
        return JSONResponse(status_code=status, content=data)
    try:
        _write_session_snapshot(
            session_store=active_session_store,
            trace_dir=active_config.trace_dir,
            trace_sink=active_trace,
            session_id=session_id,
            status="pending_snippet",
            agent_meta=agent_meta,
        )
    except OSError:
        return JSONResponse(
            status_code=502,
            content={"error": "session snapshot failed before client delivery"},
        )
    _include_usage = bool((prepared.get("stream_options") or {}).get("include_usage"))
    try:
        event = await _record_v131_completion(
            v131_factory(),
            prepared.get("messages", []),
            data,
            active_trace,
            session_id,
            agent_meta,
        )
    except Exception:
        # Graceful degradation: calibration threw, deliver A's original response.
        # Failure is already recorded in the trace by _record_v131_completion.
        try:
            _write_session_snapshot(
                session_store=active_session_store,
                trace_dir=active_config.trace_dir,
                trace_sink=active_trace,
                session_id=session_id,
                status="ready",
                agent_meta=agent_meta,
            )
        except OSError:
            pass
        return StreamingResponse(
            _emit_pseudo_stream_response(data, include_usage=_include_usage),
            media_type="text/event-stream",
            status_code=status,
        )
    if str(event.get("turn_stage") or "") == "failed":
        # Graceful degradation: v1.3.1 returned a failure event, deliver A's original response.
        try:
            _write_session_snapshot(
                session_store=active_session_store,
                trace_dir=active_config.trace_dir,
                trace_sink=active_trace,
                session_id=session_id,
                status="ready",
                agent_meta=agent_meta,
            )
        except OSError:
            pass
        return StreamingResponse(
            _emit_pseudo_stream_response(data, include_usage=_include_usage),
            media_type="text/event-stream",
            status_code=status,
        )
    data = _apply_v131_rewrite(data, event)
    try:
        _write_session_snapshot(
            session_store=active_session_store,
            trace_dir=active_config.trace_dir,
            trace_sink=active_trace,
            session_id=session_id,
            status="ready",
            agent_meta=agent_meta,
        )
    except OSError:
        return JSONResponse(
            status_code=502,
            content={"error": "session snapshot failed before client delivery"},
        )
    return StreamingResponse(
        _emit_pseudo_stream_response(data, include_usage=_include_usage),
        media_type="text/event-stream",
        status_code=status,
    )


async def _run_v132_serialized_stream(
    *,
    active_config: ProxyConfig,
    prepared: dict,
    active_trace,
    active_session_store: SessionStore,
    session_id: str,
    agent_meta: dict[str, str],
    http_client_factory: Callable,
    v132_factory: Callable[[], V132Controller],
):
    """Serialize A's call, run swap detection, emit final response as pseudo-stream."""
    _t_up = time.monotonic()
    status, data = await _post_upstream_serialized_for_v131(
        active_config,
        prepared,
        http_client_factory,
    )
    _record_phase_timing(
        active_trace, session_id, "upstream_a", _t_up, agent_meta, status=status
    )
    active_trace.record(
        session_id,
        {"event_type": "response_completed", "candidate_model": "A",
         "candidate_disposition": "pending_gate", "status": status,
         "response": data, **agent_meta},
    )
    if status >= 400:
        try:
            _record_terminal_upstream_failure(
                trace_sink=active_trace,
                session_store=active_session_store,
                trace_dir=active_config.trace_dir,
                session_id=session_id,
                status=status,
                data=data,
                agent_meta=agent_meta,
            )
        except OSError:
            return JSONResponse(
                status_code=502,
                content={"error": "session snapshot failed before client delivery"},
            )
        return JSONResponse(status_code=status, content=data)
    try:
        _write_session_snapshot(
            session_store=active_session_store,
            trace_dir=active_config.trace_dir,
            trace_sink=active_trace,
            session_id=session_id,
            status="pending_snippet",
            agent_meta=agent_meta,
        )
    except OSError:
        return JSONResponse(
            status_code=502,
            content={"error": "session snapshot failed before client delivery"},
        )
    _t_handle = time.monotonic()
    result = await v132_factory().handle(
        prepared, status, data, http_client_factory,
        conv_key=agent_meta["conv_key"],
    )
    _record_phase_timing(
        active_trace, session_id, "sniffer_handle", _t_handle, agent_meta, path=result.path
    )
    active_trace.record(session_id, result.to_trace_event(agent_meta))
    active_trace.record(
        session_id,
        {"event_type": "response_delivered", "selected_path": result.path,
         "selected_model": "B" if result.path == "salvage_tool" else "A",
         "status": result.final_status, "response": result.final_response, **agent_meta},
    )

    try:
        _write_session_snapshot(
            session_store=active_session_store,
            trace_dir=active_config.trace_dir,
            trace_sink=active_trace,
            session_id=session_id,
            status="ready",
            agent_meta=agent_meta,
        )
    except OSError:
        return JSONResponse(
            status_code=502,
            content={"error": "session snapshot failed before client delivery"},
        )
    return StreamingResponse(
        _emit_pseudo_stream_response(
            result.final_response,
            include_usage=bool((prepared.get("stream_options") or {}).get("include_usage")),
        ),
        media_type="text/event-stream",
        status_code=result.final_status,
    )


class _StreamTraceCollector:
    def __init__(self) -> None:
        self._buffer = ""
        self._content_parts: list[str] = []
        self._reasoning_parts: list[str] = []
        self._tool_calls: dict[int, dict[str, object]] = {}
        self._finish_reason = ""
        self._chunk_count = 0

    def feed(self, chunk: bytes) -> None:
        self._chunk_count += 1
        self._buffer += chunk.decode("utf-8", errors="ignore").replace("\r\n", "\n").replace("\r", "\n")
        while "\n\n" in self._buffer:
            raw_event, self._buffer = self._buffer.split("\n\n", 1)
            self._consume_event(raw_event)

    def _consume_event(self, raw_event: str) -> None:
        data_lines = []
        for line in raw_event.splitlines():
            if line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
        if not data_lines:
            return

        payload_text = "\n".join(data_lines).strip()
        if not payload_text or payload_text == "[DONE]":
            return

        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError:
            return

        for choice in payload.get("choices") or []:
            delta = choice.get("delta") or {}
            if delta.get("content"):
                self._content_parts.append(str(delta["content"]))
            if delta.get("reasoning_content"):
                self._reasoning_parts.append(str(delta["reasoning_content"]))
            for tool_call in delta.get("tool_calls") or []:
                index = int(tool_call.get("index", 0))
                current = self._tool_calls.setdefault(
                    index,
                    {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
                )
                if tool_call.get("id"):
                    current["id"] = tool_call["id"]
                if tool_call.get("type"):
                    current["type"] = tool_call["type"]
                current_function = current.setdefault("function", {"name": "", "arguments": ""})
                function_payload = tool_call.get("function") or {}
                if function_payload.get("name"):
                    current_function["name"] = f"{current_function['name']}{function_payload['name']}"
                if function_payload.get("arguments"):
                    current_function["arguments"] = (
                        f"{current_function['arguments']}{function_payload['arguments']}"
                    )
            finish_reason = choice.get("finish_reason")
            if finish_reason:
                self._finish_reason = str(finish_reason)

    def event_payload(self) -> dict[str, object]:
        tool_calls = [self._tool_calls[index] for index in sorted(self._tool_calls)]
        message: dict[str, object] = {
            "content": "".join(self._content_parts),
            "reasoning_content": "".join(self._reasoning_parts),
        }
        if tool_calls:
            message["tool_calls"] = tool_calls
        return {
            "response": {
                "choices": [
                    {
                        "message": message,
                        "finish_reason": self._finish_reason or "stop",
                    }
                ]
            },
            "chunk_count": self._chunk_count,
        }


def _write_session_snapshot(
    *,
    session_store: SessionStore,
    trace_dir: Path,
    trace_sink,
    session_id: str,
    status: str,
    agent_meta: dict[str, str],
) -> dict[str, object]:
    events = load_trace_events(trace_dir / f"{session_id}.jsonl")
    snapshot = build_session_snapshot(
        session_id,
        events,
        status=status,
        agent_meta=agent_meta,
    )
    session_store.write(session_id, snapshot)
    trace_sink.record(
        session_id,
        {
            "event_type": "session_snapshot_written",
            "snapshot_status": status,
            **agent_meta,
        },
    )
    return snapshot


def _record_terminal_upstream_failure(
    *,
    trace_sink,
    session_store: SessionStore,
    trace_dir: Path,
    session_id: str,
    status: int,
    data: dict,
    agent_meta: dict[str, str],
) -> None:
    error = data.get("error") if isinstance(data, dict) else None
    error = error if isinstance(error, dict) else {}
    code = str(error.get("code") or f"upstream_http_{status}")
    transport_codes = {
        "upstream_connect_error",
        "upstream_timeout",
        "upstream_transport_error",
    }
    trace_sink.record(
        session_id,
        {
            "event_type": (
                "upstream_transport_failed"
                if code in transport_codes
                else "upstream_http_failed"
            ),
            "status": status,
            "code": code,
            "retryable": bool(error.get("retryable", False)),
            **agent_meta,
        },
    )
    _write_session_snapshot(
        session_store=session_store,
        trace_dir=trace_dir,
        trace_sink=trace_sink,
        session_id=session_id,
        status="failed",
        agent_meta=agent_meta,
    )


async def _post_upstream(
    config: ProxyConfig,
    body: dict,
    http_client_factory: Callable,
) -> tuple[int, dict]:
    if config.upstream_strip_params:
        body = {k: v for k, v in body.items() if k not in config.upstream_strip_params}
    # Some upstreams (e.g. aiping's Qwen endpoints) reject the OpenAI "developer" role
    # (a newer system-level alias that GPT/GLM endpoints accept). Remap developer->system
    # so the host accepts the request; semantically equivalent and safe for any provider.
    _msgs = body.get("messages")
    if isinstance(_msgs, list) and any(
        isinstance(m, dict) and m.get("role") == "developer" for m in _msgs
    ):
        body = {
            **body,
            "messages": [
                {**m, "role": "system"}
                if isinstance(m, dict) and m.get("role") == "developer"
                else m
                for m in _msgs
            ],
        }
    async with http_client_factory(timeout=httpx.Timeout(config.upstream_timeout)) as client:
        for attempt in range(2):
            try:
                response = await client.post(
                    config.upstream_url,
                    headers=_upstream_headers(config),
                    json=body,
                )
                break
            except (httpx.ConnectError, httpx.ConnectTimeout):
                # No upstream response was established, so one replay is safe and absorbs
                # transient local-proxy/socket churn. Never replay after a read timeout: the
                # provider may already have accepted and billed that request.
                if attempt == 0:
                    await asyncio.sleep(0.1)
                    continue
                return 502, {
                    "error": {
                        "message": "upstream connection failed before a response was received",
                        "type": "upstream_error",
                        "code": "upstream_connect_error",
                        "retryable": True,
                    }
                }
            except httpx.TimeoutException:
                return 504, {
                    "error": {
                        "message": "upstream response timed out",
                        "type": "upstream_error",
                        "code": "upstream_timeout",
                        "retryable": True,
                    }
                }
            except httpx.RequestError:
                return 502, {
                    "error": {
                        "message": "upstream transport failed before a response was available",
                        "type": "upstream_error",
                        "code": "upstream_transport_error",
                        "retryable": True,
                    }
                }
    try:
        data = response.json()
    except Exception:
        data = {"error": response.text[:2000]}
    if response.status_code >= 400:
        _dbg = os.environ.get("PROXY_DUMP_4XX_BODY")
        if _dbg:
            try:
                with open(_dbg, "a") as _f:
                    _f.write(json.dumps({"status": response.status_code, "resp": data, "req": body}, ensure_ascii=False)[:200000] + "\n")
            except Exception:
                pass
    return response.status_code, data


def create_app(
    config: ProxyConfig | None = None,
    *,
    http_client_factory: Callable = httpx.AsyncClient,
    trace_sink=None,
    session_store: SessionStore | None = None,
    v131_controller_factory: Callable[[], V131Controller] | None = None,
    v132_controller_factory: Callable[[], V132Controller] | None = None,
) -> FastAPI:
    active_config = config or ProxyConfig.from_env()
    active_trace = trace_sink or FileTraceSink(active_config.trace_dir)
    active_session_store = session_store or SessionStore(active_config.session_dir)
    v131_factory = v131_controller_factory or (
        lambda: build_v131_controller(active_config)
    )
    v132_factory = v132_controller_factory or (
        lambda: build_v132_controller(active_config)
    )
    app = FastAPI(title="AgentAblit proxy")

    @app.get("/")
    @app.get("/health")
    async def health() -> dict:
        return {
            "status": "ok",
            "upstream": _public_endpoint(active_config.upstream_url),
            "key_set": bool(active_config.upstream_key),
            "tmi_version": active_config.version,
            "v132": {
                "b_url": _public_endpoint(active_config.v132_b_url),
                "b_model": active_config.v132_b_model,
                "b_timeout": active_config.v132_b_timeout,
                "history_encoding": active_config.traj_coldstart_history_encoding,
                "char_budget": active_config.traj_coldstart_char_budget,
                "passthrough": active_config.traj_coldstart_passthrough,
                "ablate_graying": active_config.ablate_graying,
                "ablate_trajectory": active_config.ablate_trajectory,
                "ablate_salvage_graying": active_config.ablate_salvage_graying,
                "disable_salvage": active_config.disable_salvage,
                "ablate_reconstruct": active_config.ablate_reconstruct,
                "ablate_recover": active_config.ablate_recover,
                "ablate_l3": active_config.ablate_l3,
            },
            # REGI identity: the configured selector, the engine it normalizes to, and the
            # effective capability footprint derived from the ACTUAL config/ablation fields —
            # never inferred from the selector spelling (v1.3.3 does not enable anything).
            # All URLs pass through _public_endpoint(); secrets are never serialized.
            "regi": _build_regi_health_block(active_config),
        }

    @app.get("/v1/models")
    @app.get("/models")
    async def models() -> dict:
        return {
            "object": "list",
            "data": [
                {
                    "id": active_config.model_id,
                    "object": "model",
                    "owned_by": "proxy",
                },
                {
                    "id": "deepseek-v4-flash",
                    "object": "model",
                    "owned_by": "proxy",
                },
            ],
        }

    @app.post("/chat/completions")
    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        conv_key, conv_key_source = _derive_conv_key(request, body)
        # session_id derives DETERMINISTICALLY from conv_key (not a per-request random UUID) so a
        # header-less multi-turn agent's turns land in ONE trace/snapshot file and v1.3.1's
        # cross-turn carryover (_get_prior_intermediate_decision, keyed on {session_id}.jsonl)
        # stops silently no-op'ing. An explicit x-session-id still wins for clients that set one.
        session_id = request.headers.get("x-session-id") or (
            f"sess_{hashlib.sha256(conv_key.encode('utf-8')).hexdigest()[:12]}"
        )
        agent_meta = _agent_metadata(request, session_id, conv_key)
        # Single, shared legacy-selector resolution for both endpoints: header wins over the
        # configured TMI_VERSION; only v1.3.3 is aliased to the v1.3.2 engine. The engine
        # selector drives dispatch below; the requested selector is kept for REGI audit.
        requested_version, version = _resolve_execution_selector(
            request.headers.get("x-tmi-version"), active_config.version
        )
        error_msg = _reject_unknown_selector(requested_version)
        if error_msg is not None:
            return JSONResponse(status_code=400, content={"error": error_msg})
        regi_meta = selector_metadata(
            requested_version, passthrough=version == "passthrough"
        )

        active_trace.record(
            session_id,
            {
                "event_type": "request_received",
                "strategy_version": version,
                "regi": regi_meta,
                "model": body.get("model"),
                "stream": bool(body.get("stream", False)),
                "messages": body.get("messages", []),
                "conv_key_source": conv_key_source,
                **agent_meta,
            },
        )
        try:
            _write_session_snapshot(
                session_store=active_session_store,
                trace_dir=active_config.trace_dir,
                trace_sink=active_trace,
                session_id=session_id,
                status="pending_upstream",
                agent_meta=agent_meta,
            )
        except OSError:
            return JSONResponse(
                status_code=502,
                content={"error": "session snapshot failed before client delivery"},
            )

        # Force the upstream model to the proxy's configured host model, regardless of the
        # label the client sent (hermes/Claude Code send their own default like "gpt-5.4").
        # Mirrors the /v1/messages endpoint's override so the same proxy port can front any
        # host without the client having to know the upstream model id.
        prepared = {**body, "model": active_config.model_id} if active_config.model_id else body

        # RETIRED by default (active_config.ablate_reasoning_san now defaults True). Superseded
        # by V132Controller._calibrate_reasoning, which cleans A's reasoning_content at delivery
        # time instead of retroactively rewriting already-committed history on the next request.
        # Kept for TMI_ABLATE_REASONING_SAN=0 comparison runs. Gated on the flag directly (not
        # just `_ctrl.reasoning_sanitizer` truthiness) — that object is now always constructed,
        # shared with V132Controller._calibrate_reasoning's own, independent ablation flag.
        if version in ("v1.3.1", "v1.3.2") and not active_config.ablate_reasoning_san:
            _ctrl = v131_factory()
            if _ctrl.reasoning_sanitizer and prepared.get("messages"):
                _t_san = time.monotonic()
                _sanitized, _san_stats = await _sanitize_history_reasoning(
                    prepared["messages"],
                    _ctrl.reasoning_sanitizer,
                    _ctrl.calibrator,
                )
                _record_phase_timing(
                    active_trace, session_id, "reasoning_sanitize", _t_san,
                    agent_meta, **_san_stats,
                )
                if _sanitized is not prepared["messages"]:
                    prepared = {**prepared, "messages": _sanitized}

        if version == "v1.3.1" and prepared.get("stream"):
            return await _run_v131_serialized_stream(
                active_config=active_config,
                prepared=prepared,
                active_trace=active_trace,
                active_session_store=active_session_store,
                session_id=session_id,
                agent_meta=agent_meta,
                http_client_factory=http_client_factory,
                v131_factory=v131_factory,
            )

        if version == "v1.3.2" and prepared.get("stream"):
            return await _run_v132_serialized_stream(
                active_config=active_config,
                prepared=prepared,
                active_trace=active_trace,
                active_session_store=active_session_store,
                session_id=session_id,
                agent_meta=agent_meta,
                http_client_factory=http_client_factory,
                v132_factory=v132_factory,
            )

        if prepared.get("stream"):
            # Passthrough streaming: some upstreams (e.g. aiping's Qwen/GLM endpoints) emit
            # non-OpenAI-conformant SSE that strict clients (hermes) reject as an "empty stream
            # with no finish_reason". Rather than forward the raw stream, fetch a single
            # non-streaming response and re-emit it as a clean synthetic OpenAI SSE — the same
            # serialize-then-emit approach the v1.3.x paths already use, so both arms behave
            # identically at the wire level and tool_calls survive.
            non_stream_body = {k: v for k, v in prepared.items() if k != "stream"}
            _t_up = time.monotonic()
            status, data = await _post_upstream(
                active_config, non_stream_body, http_client_factory
            )
            _record_phase_timing(
                active_trace, session_id, "upstream_a", _t_up, agent_meta, status=status
            )
            active_trace.record(
                session_id,
                {"event_type": "response_completed", "candidate_model": "A",
                 "candidate_disposition": "pending_gate", "status": status,
                 "response": data, **agent_meta},
            )
            if status >= 400:
                return JSONResponse(status_code=status, content=data)
            include_usage = bool((prepared.get("stream_options") or {}).get("include_usage"))
            return StreamingResponse(
                _emit_pseudo_stream_response(data, include_usage=include_usage),
                media_type="text/event-stream",
                status_code=status,
            )

        _t_up = time.monotonic()
        status, data = await _post_upstream(
            active_config, prepared, http_client_factory
        )
        _record_phase_timing(
            active_trace, session_id, "upstream_a", _t_up, agent_meta, status=status
        )
        active_trace.record(
            session_id,
            {"event_type": "response_completed", "candidate_model": "A",
             "candidate_disposition": "pending_gate", "status": status,
             "response": data, **agent_meta},
        )
        # Passthrough (Vanilla / Parasite-only arms): deliver A's upstream response verbatim,
        # no TMI rewrite. The OpenAI streaming branch above already forwards raw for any
        # non-v131/v132 version; this is the matching non-streaming return.
        if version == "passthrough":
            return JSONResponse(status_code=status, content=data)
        if version == "v1.3.2":
            if status >= 400:
                try:
                    _record_terminal_upstream_failure(
                        trace_sink=active_trace,
                        session_store=active_session_store,
                        trace_dir=active_config.trace_dir,
                        session_id=session_id,
                        status=status,
                        data=data,
                        agent_meta=agent_meta,
                    )
                except OSError:
                    return JSONResponse(
                        status_code=502,
                        content={"error": "session snapshot failed before client delivery"},
                    )
                return JSONResponse(status_code=status, content=data)
            try:
                _write_session_snapshot(
                    session_store=active_session_store,
                    trace_dir=active_config.trace_dir,
                    trace_sink=active_trace,
                    session_id=session_id,
                    status="pending_snippet",
                    agent_meta=agent_meta,
                )
            except OSError:
                return JSONResponse(
                    status_code=502,
                    content={"error": "session snapshot failed before client delivery"},
                )
            _t_handle = time.monotonic()
            result = await v132_factory().handle(
                prepared, status, data, http_client_factory,
                conv_key=agent_meta["conv_key"],
            )
            _record_phase_timing(
                active_trace, session_id, "sniffer_handle", _t_handle, agent_meta,
                path=result.path,
            )
            active_trace.record(session_id, result.to_trace_event(agent_meta))
            active_trace.record(
                session_id,
                {"event_type": "response_delivered", "selected_path": result.path,
                 "selected_model": "B" if result.path == "salvage_tool" else "A",
                 "status": result.final_status, "response": result.final_response, **agent_meta},
            )
            try:
                _write_session_snapshot(
                    session_store=active_session_store,
                    trace_dir=active_config.trace_dir,
                    trace_sink=active_trace,
                    session_id=session_id,
                    status="ready",
                    agent_meta=agent_meta,
                )
            except OSError:
                return JSONResponse(
                    status_code=502,
                    content={"error": "session snapshot failed before client delivery"},
                )
            return JSONResponse(
                status_code=result.final_status, content=result.final_response
            )

        if version == "v1.3.1":
            if status >= 400:
                try:
                    _record_terminal_upstream_failure(
                        trace_sink=active_trace,
                        session_store=active_session_store,
                        trace_dir=active_config.trace_dir,
                        session_id=session_id,
                        status=status,
                        data=data,
                        agent_meta=agent_meta,
                    )
                except OSError:
                    return JSONResponse(
                        status_code=502,
                        content={"error": "session snapshot failed before client delivery"},
                    )
                return JSONResponse(status_code=status, content=data)
            try:
                _write_session_snapshot(
                    session_store=active_session_store,
                    trace_dir=active_config.trace_dir,
                    trace_sink=active_trace,
                    session_id=session_id,
                    status="pending_snippet",
                    agent_meta=agent_meta,
                )
            except OSError:
                return JSONResponse(
                    status_code=502,
                    content={"error": "session snapshot failed before client delivery"},
                )
            try:
                event = await _record_v131_completion(
                    v131_factory(),
                    prepared.get("messages", []),
                    data,
                    active_trace,
                    session_id,
                    agent_meta,
                )
            except Exception:
                # Graceful degradation: calibration threw, deliver A's original response.
                try:
                    _write_session_snapshot(
                        session_store=active_session_store,
                        trace_dir=active_config.trace_dir,
                        trace_sink=active_trace,
                        session_id=session_id,
                        status="ready",
                        agent_meta=agent_meta,
                    )
                except OSError:
                    pass
                return JSONResponse(status_code=status, content=data)
            if str(event.get("turn_stage") or "") == "failed":
                # Graceful degradation: v1.3.1 returned a failure event, deliver A's original response.
                try:
                    _write_session_snapshot(
                        session_store=active_session_store,
                        trace_dir=active_config.trace_dir,
                        trace_sink=active_trace,
                        session_id=session_id,
                        status="ready",
                        agent_meta=agent_meta,
                    )
                except OSError:
                    pass
                return JSONResponse(status_code=status, content=data)
            data = _apply_v131_rewrite(data, event)
            try:
                _write_session_snapshot(
                    session_store=active_session_store,
                    trace_dir=active_config.trace_dir,
                    trace_sink=active_trace,
                    session_id=session_id,
                    status="ready",
                    agent_meta=agent_meta,
                )
            except OSError:
                return JSONResponse(
                    status_code=502,
                    content={"error": "session snapshot failed before client delivery"},
                )
            return JSONResponse(status_code=status, content=data)
        await _record_v131_completion(
            v131_factory(),
            prepared.get("messages", []),
            data,
            active_trace,
            session_id,
            agent_meta,
        )
        return JSONResponse(status_code=status, content=data)

    # ------------------------------------------------------------------
    # Anthropic-compatible endpoint: POST /v1/messages
    # Accepts Anthropic API format from Claude Code / OpenClaw / etc.,
    # converts to OpenAI internally, runs the TMI pipeline, converts back.
    # ------------------------------------------------------------------
    @app.post("/v1/messages")
    @app.post("/messages")
    async def messages_endpoint(request: Request):
        # The Anthropic-format conversion layer (proxy.anthropic_compat) is not present in this
        # build — its implementation was never committed. Fail cleanly (501) instead of crashing
        # with a 500 ModuleNotFoundError. OpenAI-format clients should use /v1/chat/completions.
        try:
            from proxy.anthropic_compat import (
                anthropic_to_openai_body,
                emit_anthropic_stream,
                openai_to_anthropic_response,
            )
        except ImportError:
            return JSONResponse(
                status_code=501,
                content={"error": {
                    "type": "not_implemented",
                    "message": "Anthropic-format endpoint unavailable (proxy.anthropic_compat "
                               "missing). Use the OpenAI-compatible /v1/chat/completions endpoint.",
                }},
            )

        body = await request.json()
        client_stream = bool(body.get("stream", False))
        req_model = str(body.get("model") or active_config.model_id)
        # Mirror of the /v1/chat/completions identity derivation so this endpoint inherits the
        # same conv_key-rooted keying if proxy.anthropic_compat is ever committed. NOTE: `body`
        # here is still ANTHROPIC-format; when anthropic_compat lands, move _derive_conv_key to
        # run on the CONVERTED OpenAI body (the header paths (a)/(b) are format-independent).
        conv_key, conv_key_source = _derive_conv_key(request, body)
        session_id = request.headers.get("x-session-id") or (
            f"sess_{hashlib.sha256(conv_key.encode('utf-8')).hexdigest()[:12]}"
        )
        agent_meta = _agent_metadata(request, session_id, conv_key)
        # Same shared selector resolution as /v1/chat/completions (header wins; v1.3.3 is
        # aliased to the v1.3.2 engine only).
        requested_version, version = _resolve_execution_selector(
            request.headers.get("x-tmi-version"), active_config.version
        )
        error_msg = _reject_unknown_selector(requested_version)
        if error_msg is not None:
            return JSONResponse(
                status_code=400,
                content={"type": "error", "error": {"type": "invalid_request_error", "message": error_msg}},
            )
        regi_meta = selector_metadata(
            requested_version, passthrough=version == "passthrough"
        )

        # Convert Anthropic → OpenAI (always serialized; we re-emit below)
        prepared = anthropic_to_openai_body(body, active_config.model_id)

        active_trace.record(
            session_id,
            {
                "event_type": "request_received",
                "strategy_version": version,
                "regi": regi_meta,
                "model": req_model,
                "stream": client_stream,
                "messages": prepared.get("messages", []),
                "conv_key_source": conv_key_source,
                **agent_meta,
            },
        )
        try:
            _write_session_snapshot(
                session_store=active_session_store,
                trace_dir=active_config.trace_dir,
                trace_sink=active_trace,
                session_id=session_id,
                status="pending_upstream",
                agent_meta=agent_meta,
            )
        except OSError:
            return JSONResponse(
                status_code=502,
                content={"type": "error", "error": {"type": "api_error", "message": "session snapshot failed"}},
            )

        # RETIRED by default — see the matching comment on the non-streaming path above.
        if version in ("v1.3.1", "v1.3.2") and not active_config.ablate_reasoning_san:
            _ctrl = v131_factory()
            if _ctrl.reasoning_sanitizer and prepared.get("messages"):
                _t_san = time.monotonic()
                _sanitized, _san_stats = await _sanitize_history_reasoning(
                    prepared["messages"],
                    _ctrl.reasoning_sanitizer,
                    _ctrl.calibrator,
                )
                _record_phase_timing(
                    active_trace, session_id, "reasoning_sanitize", _t_san,
                    agent_meta, **_san_stats,
                )
                if _sanitized is not prepared["messages"]:
                    prepared = {**prepared, "messages": _sanitized}

        # Always fetch serialized response from upstream for TMI rewriting
        _t_up = time.monotonic()
        status, data = await _post_upstream_serialized_for_v131(
            active_config, prepared, http_client_factory
        )
        _record_phase_timing(
            active_trace, session_id, "upstream_a", _t_up, agent_meta, status=status
        )
        active_trace.record(
            session_id,
            {"event_type": "response_completed", "candidate_model": "A",
             "candidate_disposition": "pending_gate", "status": status,
             "response": data, **agent_meta},
        )

        def _anthropic_error(status_code: int, msg: str):
            return JSONResponse(
                status_code=status_code,
                content={"type": "error", "error": {"type": "api_error", "message": msg}},
            )

        if status >= 400:
            return _anthropic_error(status, str(data))

        # v1.3.1 / v1.3.2 rewriting
        if version in ("v1.3.1", "v1.3.2"):
            try:
                _write_session_snapshot(
                    session_store=active_session_store,
                    trace_dir=active_config.trace_dir,
                    trace_sink=active_trace,
                    session_id=session_id,
                    status="pending_snippet",
                    agent_meta=agent_meta,
                )
            except OSError:
                return _anthropic_error(502, "session snapshot failed before client delivery")

            if version == "v1.3.2":
                _t_handle = time.monotonic()
                result = await v132_factory().handle(
                    prepared, status, data, http_client_factory,
                    conv_key=agent_meta["conv_key"],
                )
                _record_phase_timing(
                    active_trace, session_id, "sniffer_handle", _t_handle, agent_meta,
                    path=result.path,
                )
                active_trace.record(session_id, result.to_trace_event(agent_meta))
                active_trace.record(
                    session_id,
                    {"event_type": "response_delivered", "selected_path": result.path,
                     "selected_model": "B" if result.path == "salvage_tool" else "A",
                     "status": result.final_status, "response": result.final_response,
                     **agent_meta},
                )
                data = result.final_response
                status = result.final_status
            else:
                # v1.3.1
                try:
                    event = await _record_v131_completion(
                        v131_factory(),
                        prepared.get("messages", []),
                        data,
                        active_trace,
                        session_id,
                        agent_meta,
                    )
                except Exception:
                    pass  # graceful degradation: deliver A's original response
                else:
                    if str(event.get("turn_stage") or "") != "failed":
                        data = _apply_v131_rewrite(data, event)

            try:
                _write_session_snapshot(
                    session_store=active_session_store,
                    trace_dir=active_config.trace_dir,
                    trace_sink=active_trace,
                    session_id=session_id,
                    status="ready",
                    agent_meta=agent_meta,
                )
            except OSError:
                return _anthropic_error(502, "session snapshot failed before client delivery")

        # Emit in Anthropic format
        if client_stream:
            return StreamingResponse(
                emit_anthropic_stream(data, req_model),
                media_type="text/event-stream",
                status_code=status,
            )
        return JSONResponse(
            status_code=status,
            content=openai_to_anthropic_response(data, req_model),
        )

    return app


def _load_default_config() -> ProxyConfig:
    """Build the proxy config. Prefer a config file so a plain clone with a config.yaml runs
    out of the box; env vars still override any file value (see ProxyConfig.from_file).

    Resolution order:
      1. $AGENTABLIT_CONFIG (explicit path to a .yaml/.yml/.json file)
      2. ./config.yaml or ./config.yml in the working directory
      3. env-only (ProxyConfig.from_env)
    """
    explicit = os.environ.get("AGENTABLIT_CONFIG", "").strip()
    candidates = [explicit] if explicit else ["config.yaml", "config.yml"]
    for c in candidates:
        if c and Path(c).is_file():
            try:
                return ProxyConfig.from_file(c)
            except Exception as exc:  # bad file → fall back to env, but say so
                print(f"[agentablit] config file '{c}' failed to load ({exc}); using env only",
                      flush=True)
                break
    return ProxyConfig.from_env()


DEFAULT_CONFIG = _load_default_config()
app = create_app(DEFAULT_CONFIG)

# Compatibility constants used by older launch/inspection scripts.
UPSTREAM_URL = DEFAULT_CONFIG.upstream_url
UPSTREAM_KEY = DEFAULT_CONFIG.upstream_key
PROXY_MODEL_ID = DEFAULT_CONFIG.model_id
TRACE_DIR = DEFAULT_CONFIG.trace_dir


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PROXY_PORT", "8787"))
    host = os.environ.get("PROXY_HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port, log_level="info")
