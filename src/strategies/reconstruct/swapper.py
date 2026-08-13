"""A↔B role-swap executor for the reconstruct engine.

Normal flow (recover):    A executes, B audits stance.
Swap flow (reconstruct):  A plans (schema-only), B executes (plan-guided).

A's plan is injected into B's system message so B only has to fill concrete
argument values from context — no need for B to reason about tool selection.
"""
from __future__ import annotations

import asyncio
import copy
import ipaddress
import json
import os
from typing import Any, Callable
from urllib.parse import urlsplit

import httpx

from shared.model_client import RoleModelError
from . import context_asset_inventory as asset_inventory
from . import context_assets
from . import ledger as L
from . import observer_context
from .spec import (
    B_COLDSTART_HARNESS,
    B_COLDSTART_HINT_TEMPLATE,
    B_COLDSTART_SKILLS_TEMPLATE,
    SKILL_EXTRACT_MAX_TOKENS,
    SKILL_EXTRACT_PROMPT,
)
from .types import RoleSwapResult


# Older-step content cap for the current (stalled) segment's VERBATIM pass-through: steps
# older than the most-recent `_RECENT_FULL_STEPS` (see below) are clipped to this; the recent
# ones use `_STEP_FULL_CAP`. Same "character cap" style as the old step-digest caps.
_STEP_OLD_CAP = 120
# Combined char budget for closed-segment gist history + the current segment's native
# replay — no token-counting lib, same "character cap" style as the numbers above. Trim
# order: oldest closed segments first, then oldest current-segment steps — never the
# newest/most-relevant steps, which is what B needs most to pick its next action. See
# PA_REACT_CONSISTENCY_AUDIT.md Finding 1 §C.
# Overall char budget for B's coldstart progress context. Balanced for speed: big enough that
# B sees the whole (short) agentic trajectory it must continue, small enough that the 9B B
# stays fast (the old 16K starved B; a 120K "give it everything" made B calls crawl). Env-
# tunable via configure() — raise TMI_COLDSTART_CHAR_BUDGET / _FULL_CAP for fidelity over speed.
_PROGRESS_CHAR_BUDGET = 24000

# How many of the MOST-RECENT steps (owner-units) of the current stalled segment are passed to
# B at FULL fidelity (`_STEP_FULL_CAP`) vs the older-step snippet cap. `<= 0` means ALL steps
# full (the default): AgentHarm-style trajectories are short and B can hold them whole, so
# snippet-clipping older steps just starves B and drives repeat-the-same-call loops. Set from
# config via configure() (TMI_COLDSTART_RECENT_FULL_STEPS / _FULL_CAP / _CHAR_BUDGET).
_RECENT_FULL_STEPS = 0
_STEP_FULL_CAP = 3000

# Codec for the current stalled segment in B's coldstart pseudo-history. "native" keeps real
# assistant tool_call / tool-result messages (structured, matches B's training distribution;
# results render as environment <tool_response>). "text" collapses them to assistant-content
# prose (pre-fix behavior) — the A/B CONTROL ARM. See _build_coldstart_body / _textualize_segment.
_HISTORY_ENCODING = "native"

# Passthrough budget guard (escalation rung 1 / any passthrough coldstart). Passthrough raw-copies the
# FULL request into B — great for structure fidelity on the local nvfp4 B (MAX_CONTEXT=262144),
# but a Win/gguf B defaults to TMI_GGUF_N_CTX=16384 and, for an AGENTIC request (tools present,
# which every coldstart is), the server RAISES ContextLengthExceeded (HTTP 500) rather than
# truncating — so ~half of real cases (measured: 15/32 salvage_text cases exceed 16K, some at
# 200K-400K tok) would 500 and waste a giant round-trip. When the passthrough body would exceed
# the effective B context, fall through to the budget-protected coldstart_v2 path instead (which
# trims structure-aware, keeps action-critical units + steer, never over-sends). Char budget is
# derived from the server's token context via TMI_B_CONTEXT_TOKENS (default 16384, matching the
# gguf default), minus an output/template reserve, at a CONSERVATIVE 2 chars/token: CJK tokenizes
# at ~1-2 tokens/char, so a CJK-heavy body counted at 3 chars/tok could still exceed the window and
# 500 the gguf B — the exact outcome this guard prevents. 2 chars/tok under-fills for English/code
# (safe: coldstart_v2 would just get invoked slightly earlier) but never OVER-fills for CJK. Tune
# via TMI_PASSTHROUGH_CHARS_PER_TOKEN if a real tokenizer estimate is wired for the served B.
_B_CONTEXT_TOKENS = int(os.environ.get("TMI_B_CONTEXT_TOKENS", "16384"))
_B_OUTPUT_RESERVE_TOKENS = 2048
_PASSTHROUGH_CHARS_PER_TOKEN = int(os.environ.get("TMI_PASSTHROUGH_CHARS_PER_TOKEN", "2"))
_PASSTHROUGH_CHAR_BUDGET = max(
    8000, (_B_CONTEXT_TOKENS - _B_OUTPUT_RESERVE_TOKENS) * _PASSTHROUGH_CHARS_PER_TOKEN
)

# Hybrid V2 only: retry an overflowing coldstart build with a resolver-selected tool
# subset instead of failing context_insufficient. See ABSwapper.coldstart /
# _resolve_tool_overflow and docs/HYBRID_V2_CONTEXT_ASSET_RESOLVER.md §10 step 3.
_CONTEXT_ASSET_RESOLVER_ENABLED = False
_ASSET_RESOLVER = context_assets.BudgetedAssetResolver()


def _asset_char_cost(payload: Any) -> int:
    # Matches coldstart_v2._tools_chars' per-tool measurement convention (default
    # json.dumps separators) so a resolver budget in chars means the same thing the
    # coldstart planner already measured — NOT context_asset_inventory's compact-JSON
    # _default_cost, which would silently under-count against that budget.
    return len(json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False))


def _is_loopback_url(url: str) -> bool:
    """Return whether ``url`` targets a loopback host without doing DNS I/O."""
    try:
        host = (urlsplit(str(url)).hostname or "").lower().rstrip(".")
    except ValueError:
        return False
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _b_client_kwargs(url: str, timeout: float) -> dict[str, Any]:
    """Keep environment proxies for remote B endpoints, but never proxy loopback B I/O."""
    kwargs: dict[str, Any] = {"timeout": httpx.Timeout(timeout)}
    if _is_loopback_url(url):
        kwargs["trust_env"] = False
    return kwargs


def configure(
    *, recent_full_steps: int, full_cap: int, char_budget: int, history_encoding: str = "native",
    context_asset_resolver_enabled: bool = False,
) -> None:
    """Set B's coldstart context fidelity: how many trailing steps are kept full (<=0 = ALL),
    the per-message full cap, the overall char budget, and the history CODEC (native|text).
    build_reconstruct_controller calls this per request (idempotent)."""
    global _RECENT_FULL_STEPS, _STEP_FULL_CAP, _PROGRESS_CHAR_BUDGET, _HISTORY_ENCODING
    global _CONTEXT_ASSET_RESOLVER_ENABLED
    _RECENT_FULL_STEPS = int(recent_full_steps)
    _STEP_FULL_CAP = max(_STEP_OLD_CAP, int(full_cap))
    _PROGRESS_CHAR_BUDGET = max(4000, int(char_budget))
    _enc = str(history_encoding).strip().lower()
    _HISTORY_ENCODING = _enc if _enc in ("text", "v2") else "native"
    _CONTEXT_ASSET_RESOLVER_ENABLED = bool(context_asset_resolver_enabled)


def _msg_chars(msgs: list[dict]) -> int:
    return sum(len(json.dumps(m, ensure_ascii=False)) for m in msgs)


def _cap_message_content(msg: dict, cap: int) -> dict:
    """Shallow-copy `msg` with its content / tool-call arguments capped to `cap` chars —
    keeps one verbose step (e.g. a giant file read) from alone blowing the coldstart
    context budget, without touching the message's structural fields (role, tool_call_id,
    ids stay exact so B's own tool_call/tool pairing is never broken by capping)."""
    out = dict(msg)
    content = out.get("content")
    if isinstance(content, str) and len(content) > cap:
        out["content"] = content[:cap]
    tool_calls = out.get("tool_calls")
    if isinstance(tool_calls, list):
        capped: list[Any] = []
        for tc in tool_calls:
            if not isinstance(tc, dict):
                capped.append(tc)
                continue
            fn = tc.get("function") or {}
            args = fn.get("arguments")
            if isinstance(args, str) and len(args) > cap:
                tc = copy.deepcopy(tc)
                tc["function"]["arguments"] = args[:cap]
            capped.append(tc)
        out["tool_calls"] = capped
    return out


def _group_by_leading_user(msgs: list[dict]) -> list[list[dict]]:
    """Group a flat message list back into per-segment chunks, each starting at the
    role=user message that opened it (mirrors how render_closed_history emits one such
    chunk per kept segment) — lets the budget trim drop whole segments, never half a
    Q/gist pair."""
    groups: list[list[dict]] = []
    for m in msgs:
        if m.get("role") == "user" or not groups:
            groups.append([m])
        else:
            groups[-1].append(m)
    return groups


def _group_by_owner(msgs: list[dict]) -> list[list[dict]]:
    """Group a flat message list so every role=='tool' message stays attached to the
    message that precedes it (its owning assistant tool_calls turn) — lets the budget trim
    drop whole units from the front without ever orphaning a `tool` message. A `tool`
    message with no preceding assistant tool_calls of the matching id is rejected by most
    OpenAI-compatible endpoints, which would silently degrade salvage_tool to salvage_text
    exactly in the long/high-context stalled loops where salvage matters most — trimming
    by individual message (rather than by owner-unit) can produce exactly that."""
    groups: list[list[dict]] = []
    for m in msgs:
        if m.get("role") == "tool" and groups:
            groups[-1].append(m)
        else:
            groups.append([m])
    return groups


def _textualize_segment(segment: list[dict]) -> list[dict]:
    """Text-codec (A/B CONTROL ARM) rendering of ONE segment: collapse its native assistant
    tool_call / tool-result messages into assistant-content prose (mirrors the closed-history
    digest), so tool RESULTS become assistant SELF-REPORT rather than environment
    <tool_response>. This is the deliberately-degraded encoding the native codec is measured
    against — NOT the default; selected only by history_encoding='text'."""
    q = observer_context._seg_user_query(segment)[: observer_context._Q_CAP]
    steps = observer_context._seg_tool_steps(segment)
    out: list[dict] = []
    if q:
        out.append({"role": "user", "content": q})
    if steps:
        out.append({
            "role": "assistant",
            "content": observer_context._raw_steps_block(steps, neutralized=False),
        })
    return out


def _assemble_progress(
    closed_history: list[dict], current_segment: list[dict], *, char_budget: int,
) -> list[dict]:
    """Combine older-segment gist pairs (closed_history) with the current (stalled)
    segment's native tool_call/tool replay (current_segment) under one length budget.

    The last `_RECENT_FULL_STEPS` owner-units (an assistant tool_calls turn + its tool
    results) are kept at FULL fidelity (`_STEP_FULL_CAP`) so B sees the real, untruncated
    RESULT of the step(s) it must continue FROM — snippet-clipping the previous result starves
    B's choice of next action and drives repeat-the-same-call loops. Older units are snippet-
    clipped (`_STEP_OLD_CAP`); the whole is then trimmed to char_budget oldest-first."""
    closed_groups = _group_by_leading_user(closed_history)
    head = current_segment[:1] if current_segment and current_segment[0].get("role") == "user" else []
    body = list(current_segment[len(head):])
    body_groups = _group_by_owner(body)
    n = len(body_groups)
    keep_all = _RECENT_FULL_STEPS <= 0
    body_groups = [
        [
            _cap_message_content(
                m, _STEP_FULL_CAP if (keep_all or i >= n - _RECENT_FULL_STEPS) else _STEP_OLD_CAP
            )
            for m in unit
        ]
        for i, unit in enumerate(body_groups)
    ]

    def _total() -> int:
        return (
            sum(_msg_chars(g) for g in closed_groups)
            + _msg_chars(head)
            + sum(_msg_chars(g) for g in body_groups)
        )

    while closed_groups and _total() > char_budget:
        closed_groups.pop(0)
    # Never drop the LAST unit (the most recent step is what B needs most to decide its
    # next action) — mirrors the closed-segment trim's "whole pairs, never half" discipline.
    while len(body_groups) > 1 and _total() > char_budget:
        body_groups.pop(0)
    return (
        [m for g in closed_groups for m in g]
        + head
        + [m for g in body_groups for m in g]
    )


def _build_coldstart_body(
    request_body: dict[str, Any],
    steer_text: str,
    tools: list[dict],
    b_model: str,
    skills_summary: str = "",
    conv_key: str = "",
    context_resolution_coldstart: str = "snippet",
    include_history: bool = True,
    passthrough: bool = False,
    recall_unit_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Build B's cold-start request to drive the client agent's loop for one turn.

    Client-agnostic (the client's own system prompt is intentionally NOT reused — tool
    names come from `tools`, and a client prompt like Claude Code's ~32K is full of
    refusal instructions counterproductive here):
      - System: a generic tool-name-free harness (B_COLDSTART_HARNESS) with the
                async-extracted skills summary.
      - History: older closed segments compressed to one Q/gist message pair each
                (observer_context.render_closed_history); the CURRENT (stalled) segment
                passed through VERBATIM from the live request — real tool_call/tool
                message pairs in the exact form B's own training distribution expects,
                not a rendered digest. This needs no reconstruction: A's stalled/refused
                turn was never appended to `request_body["messages"]` to begin with (it
                is a separate `a_response` the caller already excludes), so everything
                already in these messages is genuinely native history.
      - User (last): the grayed-intent steering, placed last for recency — this is what
                reliably makes B emit a tool_call rather than narrate.
    """
    if passthrough:
        # Full-context direct load: hand B A's EXACT request — original system prompt, full
        # message history, tools — unchanged, model swapped to B. B is decensored, so A's
        # own refusal instructions don't re-induce refusal; B decodes A's withheld turn at
        # full fidelity with zero information loss. No harness swap / steer / trajectory
        # processing; the normal PA-ReAct loop resumes on the next turn.
        body = copy.deepcopy(request_body)
        body["model"] = b_model
        body["stream"] = False
        body.pop("stream_options", None)
        if tools:
            body["tools"] = copy.deepcopy(tools)
        if steer_text.strip():
            body.setdefault("messages", []).append({
                "role": "user",
                "content": steer_text,
            })
        # Budget guard: passthrough sends the FULL context with tools; on a Win/gguf B the
        # agentic server RAISES ContextLengthExceeded (HTTP 500) rather than truncating when the
        # prompt exceeds n_ctx. Under budget (the common short-task case, where passthrough's
        # structure fidelity is the measured escalation rung-1 win), deliver the full-fidelity body.
        if len(json.dumps(body, ensure_ascii=False)) <= _PASSTHROUGH_CHAR_BUDGET:
            return body
        # Over budget: route to the STRUCTURE-AWARE, budget-checked coldstart_v2 builder, which
        # keeps action-critical units + steer within char_budget and, when even the minimum won't
        # fit (e.g. a single oversized system/tool message — measured on real "2 msgs" cases at
        # 200K-400K tok), sets _tmi_context_status="context_insufficient" so ABSwapper.coldstart
        # returns HTTP 422 WITHOUT sending an over-limit request (no wasted 500 round-trip). The
        # plain native/snippet path below is NOT safe here: it passes the current segment through
        # verbatim, so a single huge message would still overflow. Force v2 for the guard.
        from . import coldstart_v2
        return coldstart_v2.build_coldstart_v2_body(
            request_body, steer_text, tools, b_model,
            skills_summary=skills_summary, char_budget=_PROGRESS_CHAR_BUDGET,
            include_history=include_history, recall_unit_ids=recall_unit_ids,
        )

    if _HISTORY_ENCODING == "v2":
        # Hybrid V2: build B's context from the immutable Action Ledger — closed turns replayed
        # NATIVE (not prose), exact values pinned in a deterministic derived checkpoint, budgeted
        # so action-critical values never fall to a char cap. See coldstart_v2 / ledger.
        from . import coldstart_v2
        return coldstart_v2.build_coldstart_v2_body(
            request_body, steer_text, tools, b_model,
            skills_summary=skills_summary, char_budget=_PROGRESS_CHAR_BUDGET,
            include_history=include_history, recall_unit_ids=recall_unit_ids,
        )

    all_msgs = request_body.get("messages") or []

    summary = (skills_summary or "").strip()
    skills_block = ""
    if summary and summary.upper() != "NONE":
        skills_block = B_COLDSTART_SKILLS_TEMPLATE.format(skills_summary=summary)
    harness = B_COLDSTART_HARNESS.format(skills_block=skills_block)

    closed_history = (
        observer_context.render_closed_history(
            conv_key, all_msgs, context_resolution_coldstart, window=6,
        )
        if include_history else []
    )
    segments = observer_context.split_user_segments(all_msgs)
    current_segment = list(segments[-1]) if segments else []
    if _HISTORY_ENCODING == "text":
        # Control arm: render the current segment as textualized prose instead of native
        # tool_call/tool messages (results become assistant self-report). native = default.
        current_segment = _textualize_segment(current_segment)
    progress_msgs = _assemble_progress(
        closed_history, current_segment, char_budget=_PROGRESS_CHAR_BUDGET,
    )

    # Grayed-intent steering as the final message (recency).
    steer = B_COLDSTART_HINT_TEMPLATE.format(calibrated_intent=steer_text)

    msgs: list[dict[str, Any]] = (
        [{"role": "system", "content": harness}] + progress_msgs + [{"role": "user", "content": steer}]
    )

    body = copy.deepcopy(request_body)
    body["messages"] = msgs
    body["tools"] = tools
    body["model"] = b_model
    body["stream"] = False
    body.pop("stream_options", None)
    body.pop("tool_choice", None)  # 8009 ignores it; don't send
    return body


async def _resolve_tool_overflow(
    *,
    request_body: dict[str, Any],
    steer_text: str,
    tools: list[dict],
    b_model: str,
    skills_summary: str,
    conv_key: str,
    context_resolution_coldstart: str,
    include_history: bool,
    passthrough: bool,
    recall_unit_ids: set[str] | None,
    b_body: dict[str, Any],
) -> dict[str, Any]:
    """Hybrid V2 only: when the full-structural tool catalog still overflows char_budget,
    retry the build with a resolver-selected subset (mandatory closure + deterministic
    lexical fallback ranking — no observer C call in this rollout step) instead of leaving
    the turn to fail with context_insufficient. Returns ``b_body`` unchanged whenever the
    resolver cannot help or tool schemas don't adapt cleanly — never raises."""
    plan = b_body.get("_tmi_context_plan") or {}
    char_budget = plan.get("char_budget")
    estimated = plan.get("estimated_context_chars")
    tool_chars = plan.get("tool_schema_projected_chars")
    if not tools or not isinstance(char_budget, int) or not isinstance(estimated, int) \
            or not isinstance(tool_chars, int):
        return b_body
    # Fixed headroom: summed per-tool structural costs run slightly below the tool
    # array's own json.dumps length (brackets + ", " separators between elements).
    available = max(0, char_budget - estimated + tool_chars - 64)
    if available <= 0:
        return b_body

    try:
        assets = asset_inventory.adapt_openai_tools(tools, cost_fn=_asset_char_cost)
    except ValueError:
        return b_body
    if not assets:
        return b_body

    ledger_units = L.build_ledger(request_body.get("messages") or [])
    mandatory_ids = asset_inventory.derive_mandatory_asset_ids(
        request_body, assets, ledger=ledger_units,
    )
    selection = await _ASSET_RESOLVER.resolve(
        assets,
        budget=available,
        action_state={"conv_key": conv_key},
        query=steer_text,
        mandatory_ids=mandatory_ids,
        selector=None,
    )
    if selection.reason == "mandatory_closure_exceeds_budget":
        return b_body

    selected_names = {
        asset_id.split(":", 1)[1] for asset_id in selection.selected_ids
        if asset_id.startswith("tool:")
    }
    if not selected_names or len(selected_names) >= len(tools):
        return b_body
    reduced_tools = [
        tool for tool in tools
        if isinstance(tool, dict) and (tool.get("function") or {}).get("name") in selected_names
    ]
    if not reduced_tools:
        return b_body

    retry_body = _build_coldstart_body(
        request_body, steer_text, reduced_tools, b_model,
        skills_summary, conv_key, context_resolution_coldstart, include_history,
        passthrough, recall_unit_ids,
    )
    retry_body.setdefault("_tmi_context_plan", {})["asset_resolver"] = {
        "used": True,
        "reason": selection.reason,
        "mandatory_tool_ids": sorted(selection.mandatory_ids),
        "selected_tool_ids": sorted(selection.selected_ids),
        "dropped_tool_ids": sorted(selection.dropped_ids),
        "fallback_metadata": selection.fallback_metadata,
    }
    return retry_body


class ABSwapper:
    """Send the (plan-augmented) request to model B."""

    def __init__(
        self,
        *,
        b_url: str,
        b_api_key: str,
        b_model: str,
        b_timeout: float = 60.0,
    ) -> None:
        self.b_url = b_url
        self.b_api_key = b_api_key
        self.b_model = b_model
        self.b_timeout = b_timeout

    async def coldstart(
        self,
        request_body: dict[str, Any],
        steer_text: str,
        tools: list[dict],
        http_client_factory: Callable,
        skills_summary: str = "",
        conv_key: str = "",
        context_resolution_coldstart: str = "snippet",
        include_history: bool = True,
        passthrough: bool = False,
        recall_unit_ids: set[str] | None = None,
    ) -> RoleSwapResult:
        """Cold-start B to emit tool_calls from the synthesized steer + inferred schemas.
        passthrough=True instead hands B A's raw context unchanged (see _build_coldstart_body)."""
        b_body = _build_coldstart_body(
            request_body, steer_text, tools, self.b_model,
            skills_summary, conv_key, context_resolution_coldstart, include_history,
            passthrough, recall_unit_ids,
        )
        if _CONTEXT_ASSET_RESOLVER_ENABLED \
                and b_body.get("_tmi_context_status") == "context_insufficient":
            b_body = await _resolve_tool_overflow(
                request_body=request_body,
                steer_text=steer_text,
                tools=tools,
                b_model=self.b_model,
                skills_summary=skills_summary,
                conv_key=conv_key,
                context_resolution_coldstart=context_resolution_coldstart,
                include_history=include_history,
                passthrough=passthrough,
                recall_unit_ids=recall_unit_ids,
                b_body=b_body,
            )
        context_plan = b_body.get("_tmi_context_plan") or None
        if b_body.get("_tmi_context_status") == "context_insufficient":
            error = {
                "error": {
                    "type": "context_insufficient",
                    "message": "required Hybrid V2 evidence exceeds the configured context budget",
                    "context_plan": context_plan or {},
                }
            }
            return RoleSwapResult(
                swap_executed=False,
                b_status=422,
                b_response=error,
                final_response=error,
                context_plan=context_plan,
            )
        b_body.pop("_tmi_context_status", None)
        b_body.pop("_tmi_context_plan", None)
        headers = {
            "Authorization": f"Bearer {self.b_api_key}",
            "Content-Type": "application/json",
        }
        async with http_client_factory(**_b_client_kwargs(self.b_url, self.b_timeout)) as client:
            resp = await client.post(self.b_url, headers=headers, json=b_body)
        try:
            data = resp.json()
        except Exception:
            data = {"error": resp.text[:2000]}
        return RoleSwapResult(
            swap_executed=True,
            b_status=resp.status_code,
            b_response=data,
            final_response=data,
            context_plan=context_plan,
        )

    async def extract_skills(
        self, system_prompt: str, http_client_factory: Callable
    ) -> str:
        """Ask B (9B) to list the host agent's on-demand skills from its system prompt.

        Format-agnostic (handles hermes prose / Claude Code blocks). Returns one line per
        skill ("name — desc") or "" / "NONE". Capped input; best-effort.
        """
        body = {
            "model": self.b_model,
            "stream": False,
            "max_tokens": SKILL_EXTRACT_MAX_TOKENS,
            "messages": [
                {"role": "system", "content": SKILL_EXTRACT_PROMPT},
                {"role": "user", "content": (system_prompt or "")[:24000]},
            ],
        }
        headers = {
            "Authorization": f"Bearer {self.b_api_key}",
            "Content-Type": "application/json",
        }
        async with http_client_factory(**_b_client_kwargs(self.b_url, self.b_timeout)) as client:
            resp = await client.post(self.b_url, headers=headers, json=body)
        try:
            data = resp.json()
        except Exception:
            return ""
        return self.extract_content(data)

    @staticmethod
    def extract_content(response: dict[str, Any]) -> str:
        for choice in response.get("choices") or []:
            content = str((choice.get("message") or {}).get("content") or "").strip()
            if content:
                return content
        return ""
