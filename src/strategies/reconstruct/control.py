"""ReconstructController: A/B symbiont — A decodes, the LLM sniffer gates, B remediates.

The LLM sniffer (recover summarizer) classifies A by ONE question — does A's output carry
grayable substance that serves the user's real payload? — NOT by whether A called tools:

  pure tool_call, no prose to gate          → deliver A as-is        path="pass"

  sniffer action = pass        (substance delivered, no defensive framing)
    → deliver A as-is                                                path="pass"
  sniffer action = pass_flawed (substance present, but wrapped in refusal/defensive framing)
    → gray the prose, keep A's tool_calls                            path="pass_flawed"
  sniffer action = salvage     (no grayable substance — pure refusal/stall)
    → coldstart B to manufacture a tool_call (swap)                  path="salvage_tool"
    → B can't run (no tools / B fails) → blur the refusal stance     path="salvage_text"
  non-pass intent but the calibrator itself failed (blurrer unreachable)
    → deliver A raw — honest degraded floor, NOT a clean pass        path="degraded_raw"
"""
from __future__ import annotations

import asyncio
import copy
import inspect
import json
import time
from typing import Any, Callable

from shared.messages import first_assistant_message
from . import candidate, ledger, observer_context, skills_cache, swapper
from .detector import extract_structural_signal
from .spec import SKILL_EXTRACT_PROMPT
from .swapper import ABSwapper
from .types import SnifferVerdict, StructuralSignal, ReconstructResult


def _failure_label(stage: str, exc: Exception) -> str:
    """A bounded, non-secret failure marker that survives into the session snapshot."""
    detail = str(exc).replace("\n", " ").strip()
    return f"{stage}:{type(exc).__name__}" + (f":{detail[:240]}" if detail else "")


def _extract_assistant_output(response: dict[str, Any]) -> dict[str, Any]:
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
    choice = (
        choices[index]
        if 0 <= index < len(choices) and isinstance(choices[index], dict)
        else {}
    )
    return {
        "content": str(message.get("content") or ""),
        "reasoning_content": str(message.get("reasoning_content") or ""),
        "finish_reason": str(choice.get("finish_reason") or ""),
        "tool_calls_present": bool(message.get("tool_calls")),
        # Some OpenAI-compatible providers put policy blocks here instead of `content`.
        # Keep it separate: it is a typed transport signal, not prose to keyword-classify.
        "provider_refusal": str(message.get("refusal") or ""),
    }


def _first_tool_call(response: dict[str, Any]) -> dict[str, Any]:
    """B's forged next-action candidate (the first tool_call of the assistant message), or {}."""
    found = first_assistant_message(response)
    if found is None:
        return {}
    _, message = found
    tcs = message.get("tool_calls") or []
    return tcs[0] if tcs and isinstance(tcs[0], dict) else {}


def _tool_call_batch(response: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the entire assistant tool-call batch; L9 accepts or rejects it atomically."""
    found = first_assistant_message(response)
    if found is None:
        return []
    _, message = found
    tool_calls = message.get("tool_calls")
    return tool_calls if isinstance(tool_calls, list) else []


def _user_text(content: Any) -> str:
    """Flatten a user message's content to text. Handles both plain-string content and the
    block/list form ([{"type":"text","text":...}, ...]) that OpenAI- and Anthropic-style
    clients send for multimodal turns — otherwise those turns read as empty and the sniffer
    (and the whole TMI strategy) silently no-ops for them."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            block["text"]
            for block in content
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        ]
        return " ".join(parts)
    return ""


def _latest_user_input(messages: list[dict]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            text = _user_text(msg.get("content"))
            if text.strip():
                return text
    return ""


# Per-query snippet cap for the intent window (a stance judgment needs the gist of each
# prior query, not its full text).
_INTENT_SNIPPET_CAP = 300
_INTENT_ELLIPSIS = "…"


def _snip_middle(text: str, cap: int) -> str:
    """Cap `text` to `cap` chars by eliding the MIDDLE (keep head + tail), not the tail.

    A long user query typically carries its framing at the start and its operative ask at
    the end; plain head truncation (`text[:cap]`) drops the tail and can lose the actual
    request. Middle elision keeps both ends and marks the omission. Text already within the
    cap is returned unchanged. Result length is exactly `cap` when truncation happens.
    """
    if len(text) <= cap:
        return text
    keep = cap - len(_INTENT_ELLIPSIS)
    if keep <= 0:
        return text[:cap]
    head = (keep + 1) // 2
    tail = keep - head
    return text[:head] + _INTENT_ELLIPSIS + (text[-tail:] if tail else "")


def _build_intent_window(messages: list[dict], window_size: int) -> list[str]:
    """Anchor + recent sliding window of user-query snippets, in chronological order.

    Stateless read of the (client-resent) message array — no maintained state. The origin
    query is pinned (anchor) so long conversations don't lose "what this is really about,"
    plus the most recent `window_size - 1` queries (which include the current turn). Each is
    capped by middle elision (keep head + tail, see `_snip_middle`) so a long query keeps its
    operative tail; duplicates collapsed while preserving order. Consumed by the rewrite-target
    stage and (as of the classifier-intent-window change) the classifier payload as well.
    """
    queries: list[str] = []
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "user":
            text = _user_text(m.get("content")).strip()
            if text:
                queries.append(text)
    if not queries or window_size <= 0:
        return []
    anchor = queries[0]
    # `max(window_size - 1, 1)` keeps the latest turn in the window even at window_size==1
    # (otherwise a degenerate config would yield only the anchor and drop the current query).
    recent = queries[-max(window_size - 1, 1):]
    ordered: list[str] = [anchor]
    for q in recent:
        if q not in ordered:
            ordered.append(q)
    return [_snip_middle(q, _INTENT_SNIPPET_CAP) for q in ordered]


def _apply_recover_calibration(
    a_response: dict[str, Any], recover_result: Any
) -> tuple[dict[str, Any], bool]:
    """Apply recover calibration output to a_response. Returns (response, applied)."""
    audit = getattr(recover_result, "audit", {}) or {}
    if not audit.get("calibrator_applied"):
        return a_response, False
    response = getattr(recover_result, "response", {}) or {}
    decision = str(response.get("decision") or "").strip()
    if not decision:
        return a_response, False
    delivered = copy.deepcopy(a_response)
    found = first_assistant_message(delivered)
    if found is None:
        return a_response, False
    _, assistant = found
    assistant["content"] = decision
    return delivered, True


def _apply_text(a_response: dict[str, Any], text: str) -> dict[str, Any]:
    """Deep-copy a_response with its assistant content replaced by `text` — wraps the
    salvage-steer synthesis's plain-text output back into a response-shaped payload for
    the salvage_text delivery path (mirrors _apply_recover_calibration's approach)."""
    delivered = copy.deepcopy(a_response)
    found = first_assistant_message(delivered)
    if found is None:
        return a_response
    _, assistant = found
    assistant["content"] = text
    return delivered


def _extract_tools_from_context(request_body: dict[str, Any]) -> list[dict]:
    """Get tool schemas from request body, or infer from message history tool_calls."""
    if request_body.get("tools"):
        return list(request_body["tools"])
    seen: dict[str, dict] = {}
    for msg in (request_body.get("messages") or []):
        if not isinstance(msg, dict):
            continue
        for tc in (msg.get("tool_calls") or []):
            fn = (tc.get("function") or {}) if isinstance(tc, dict) else {}
            name = str(fn.get("name") or "").strip()
            if not name or name in seen:
                continue
            try:
                args = json.loads(str(fn.get("arguments") or "{}"))
            except (json.JSONDecodeError, TypeError):
                args = {}
            if not isinstance(args, dict):
                args = {}  # scalar/list JSON args ("5", "null", "[...]") are not iterable-by-key
            props = {k: {"type": "string", "description": k} for k in args if isinstance(k, str)}
            seen[name] = {
                "type": "function",
                "function": {
                    "name": name,
                    "description": f"Execute {name}.",
                    "parameters": {"type": "object", "properties": props},
                },
            }
    return list(seen.values())


class ReconstructController:
    """A/B symbiont: A decodes, sniffer gates, B remediates."""

    def __init__(
        self,
        swapper: ABSwapper,
        swapper_fallback: ABSwapper | None = None,
        recover_controller: Any | None = None,
        util_complete: Callable[[str, str], str] | None = None,
        fallback_complete: Callable[[str, str], str] | None = None,
        util_skill_extract_complete: Callable[[str, str], str] | None = None,
        intent_window_size: int = 6,
        ablate_graying: bool = False,
        ablate_trajectory: bool = False,
        ablate_salvage_graying: bool = False,
        ablate_reasoning_calibration: bool = False,
        disable_salvage: bool = False,
        ablate_reconstruct: bool = False,
        ablate_recover: bool = False,
        ablate_l3: bool = False,
        qa_synthesis: bool = False,
        context_resolution: str = "",
        context_resolution_coldstart: str = "snippet",
        context_resolution_sniffer: str = "",
        coldstart_passthrough: bool = False,
    ) -> None:
        self.swapper = swapper
        # Rescue fallback: a second B, tried only when self.swapper's coldstart fails to
        # forge a usable tool_call (see the coldstart block in handle()). None = off,
        # identical behavior to before this field existed.
        self.swapper_fallback = swapper_fallback
        self.recover_controller = recover_controller
        # Completers for observer_context.py's per-turn distillation (util model + 9B
        # fallback). When both are None, distillation is inert (coldstart/sniffer get raw
        # steps only wherever a resolution would otherwise read a distilled gist).
        self.util_complete = util_complete
        self.fallback_complete = fallback_complete
        # Skill extraction is pure structural judgment (list what the host offers), the same
        # "no harm-adjacent generation" category as the action classifier — so it prefers the
        # aligned util model when configured, falling back to B (self.swapper.extract_skills)
        # only when util is unset or errors. See _extract_skills_cascade / handle().
        self.util_skill_extract_complete = util_skill_extract_complete
        # Size of the user-query intent window handed to the sniffer's rewrite-target stage
        # (anchor + recent sliding window). Configurable via TMI_SNIFFER_INTENT_WINDOW.
        self.intent_window_size = intent_window_size
        # Ablation switches (all default off = full mechanism enabled):
        #   ablate_graying    — pass_flawed delivers A unchanged (GRAY@flawed removed)
        #   ablate_trajectory — salvage cold-start gets NO progress context at all (T3b removed)
        #   ablate_salvage_graying — salvage cold-start steers B from A's RAW response instead of
        #                     the grayed intent (GRAY@salvage removed; the mode-independent path)
        self.ablate_graying = ablate_graying
        self.ablate_trajectory = ablate_trajectory
        self.ablate_salvage_graying = ablate_salvage_graying
        # ablate_reasoning_calibration — skip the reasoning_content cleanup in
        # _calibrate_reasoning (below); A's own withdrawal language in reasoning_content, if
        # any, is delivered untouched. Default False = on: this is now the ONLY place
        # reasoning_content gets cleaned (replaces the old retroactive history-scan; see
        # ablate_reasoning_san in config.py, now off by default).
        self.ablate_reasoning_calibration = ablate_reasoning_calibration
        # disable_salvage — no B rescue; salvage turns deliver A raw (degraded_raw). Control for
        # isolating graying-on-A (B masks A by complying regardless). See config.py.
        self.disable_salvage = disable_salvage
        # ablate_reconstruct / ablate_recover — narrower harness-control ablations that remove
        # exactly one salvage OUTCOME (not B rescue entirely, unlike disable_salvage). See
        # config.py field comments for the exact scope of each.
        self.ablate_reconstruct = ablate_reconstruct
        self.ablate_recover = ablate_recover
        # ablate_l3 — remove the L3 hijack-escalation ladder (default False = ON). When primary B
        # + aligned fallback B both fail to forge a valid tool_call, L3 re-attempts the hijack with
        # escalated levers (full-context passthrough on primary then fallback B) before dropping to
        # salvage_text. Set True to reproduce the legacy B-fails→salvage_text behavior. In the
        # reconstruct family, so ablate_reconstruct removes it too.
        self.ablate_l3 = ablate_l3
        # Use the SYNTHESIZED per-turn Q⊕A trajectory (observer_context.render) for both the
        # sniffer and coldstart, instead of the A-only flat summary + side-by-side intent_window.
        self.qa_synthesis = qa_synthesis
        # Unified switchable provider, split per consumer since a single knob can't
        # express asymmetric winners even in principle. coldstart (B) needs exact values ->
        # snippet wins decisively (runs/context_ablation/REPORT.md). sniffer (classifier): a
        # first single-trial stance-judgment run looked like hybrid won, but a 3-trial majority-
        # vote rerun (runs/context_ablation/STANCE_REPORT.md) collapsed the gap to noise —
        # distilled/hybrid/an unsafe raw-unneutralized control all tie at 1/6, no better than
        # each other and barely above off/snippet's 0/6. No resolution has reliable evidence of
        # helping the sniffer, so its default stays off ("") rather than paying for an unproven
        # mechanism.
        # `context_resolution`, if set to a valid resolution, is an explicit override that wins
        # for BOTH consumers (back-compat / single-knob ablation convenience).
        valid = {"snippet", "distilled", "hybrid", "intent"}
        self.context_resolution = context_resolution if context_resolution in valid else ""
        self.context_resolution_coldstart = (
            self.context_resolution
            or (context_resolution_coldstart if context_resolution_coldstart in valid else "")
        )
        self.context_resolution_sniffer = (
            self.context_resolution
            or (context_resolution_sniffer if context_resolution_sniffer in valid else "")
        )
        # Full-context direct load: on salvage, hand B A's raw request (system prompt + full
        # history + tools) unchanged instead of the curated harness/steer coldstart. B is
        # decensored so A's refusal text doesn't re-lock it. Env TMI_COLDSTART_PASSTHROUGH.
        self.coldstart_passthrough = coldstart_passthrough

    async def _calibrate_reasoning(self, a_response: dict[str, Any]) -> dict[str, Any]:
        """Clean withdrawal language out of A's own reasoning_content, at delivery time,
        so a later turn never reads back its own prior hedging from history.

        Only acts on the dedicated `reasoning_content` field (DeepSeek-style APIs). Hosts
        that put reasoning inline in `content` (e.g. gpt-5.4-mini) have nothing here to
        clean — that text is already covered by the ordinary Calibrate path on `content`
        whenever the sniffer judges the turn pass_flawed. Runs BEFORE this turn's sniffer
        classification is read by the caller (a_output is captured first in `handle`), so
        it never influences this turn's own pass/pass_flawed/salvage verdict — only what
        gets written into history for the NEXT turn.
        """
        found = first_assistant_message(a_response)
        if found is None:
            return a_response
        _, assistant = found
        reasoning = str(assistant.get("reasoning_content") or "").strip()
        if not reasoning:
            return a_response
        sanitizer = getattr(self.recover_controller, "reasoning_sanitizer", None)
        calibrator = getattr(self.recover_controller, "calibrator", None)
        if sanitizer is None or calibrator is None:
            return a_response
        try:
            summary = await asyncio.to_thread(sanitizer.sanitize, reasoning)
            if summary.action != "rewrite" or not summary.rewrite_target:
                return a_response
            calibrated = await asyncio.to_thread(calibrator.rewrite, summary.rewrite_target, reasoning)
        except Exception:
            # Best-effort: deliver A's original reasoning rather than fail the turn.
            return a_response
        delivered = copy.deepcopy(a_response)
        _, delivered_assistant = first_assistant_message(delivered)
        delivered_assistant["reasoning_content"] = calibrated.decision
        return delivered

    async def handle(
        self,
        request_body: dict[str, Any],
        a_status: int,
        a_response: dict[str, Any],
        http_client_factory: Callable,
        conv_key: str = "",
    ) -> ReconstructResult:
        timings: dict[str, float] = {}
        a_output = _extract_assistant_output(a_response)
        if not self.ablate_reasoning_calibration:
            _t0 = time.monotonic()
            a_response = await self._calibrate_reasoning(a_response)
            timings["reasoning_sanitizer"] = time.monotonic() - _t0
        signal = extract_structural_signal(request_body, a_output)
        # Sniffer verdict, recorded verbatim into the trace. Starts as a *default* (ran=False)
        # so a delivered "pass" that never consulted the sniffer is distinguishable from one
        # the sniffer actively judged. Updated in place once the sniffer runs.
        sniffer = SnifferVerdict()

        # Fire-and-forget: extract host skills once per session (stable system prompt),
        # async so it's ready for a later salvage. First salvage may miss it (best-effort).
        # Cascade: aligned util model first (self.util_skill_extract_complete), B only as
        # fallback when util is unconfigured or errors/returns empty — mirrors the util-first
        # cascade observer_context.ensure_qa_update already uses for trajectory distillation.
        async def _extract_skills(sp: str) -> str:
            if self.util_skill_extract_complete is not None:
                try:
                    text = (await asyncio.to_thread(
                        self.util_skill_extract_complete, sp[:24000], SKILL_EXTRACT_PROMPT,
                    )).strip()
                except Exception:
                    text = ""
                if text:
                    return text
            result = self.swapper.extract_skills(sp, http_client_factory)
            if inspect.isawaitable(result):
                result = await result
            return result

        skills_cache.ensure_extraction(request_body, _extract_skills)

        # Conversation messages + derived views, computed once and reused by the sniffer and
        # the salvage coldstart.
        msgs = request_body.get("messages") or []
        # User-intent window (anchor + recent user-query snippets). Feeds BOTH the sniffer's
        # rewrite-target stage and the salvage coldstart's intent dimension — the user's own
        # words, which the tool-call-only distillation cannot carry.
        intent_window = _build_intent_window(msgs, self.intent_window_size)
        # Distill each CLOSED user-turn into its A_i (per-turn), off the hot path, so the
        # synthesized Q⊕A trajectory has per-turn A's. Only needed for the distilled resolution
        # (snippet/intent are LLM-free); qa_synthesis implies distilled.
        if conv_key and (self.util_complete or self.fallback_complete) and not self.ablate_trajectory \
                and (self.qa_synthesis or observer_context.needs_distillation(self.context_resolution_coldstart)
                     or observer_context.needs_distillation(self.context_resolution_sniffer)):
            observer_context.ensure_qa_update(
                conv_key, msgs, self.util_complete, self.fallback_complete
            )

        # Pure tool call with no prose to gate on → nothing to sniff, deliver as-is.
        if a_output.get("tool_calls_present") and not str(a_output.get("content") or "").strip():
            return self._deliver(a_response, a_status, signal, sniffer, "pass", timings=timings)

        # Three-state sniffer: the recover summarizer assigns pass / pass_flawed / salvage,
        # and (for non-pass) its calibrator produces the grayed text. The sniffer judges
        # A's observable state only — it knows nothing about coldstart or tool generation.
        action = "pass"
        calibrated, applied = a_response, False
        provider_refusal = str(a_output.get("provider_refusal") or "").strip()
        provider_filtered = str(a_output.get("finish_reason") or "").strip().lower() \
            in {"content_filter", "prohibited_content"}
        if provider_refusal or provider_filtered:
            # The upstream explicitly withheld the assistant action.  No semantic guess is
            # necessary and an empty `message.content` cannot be fed to UserWillInput anyway.
            # Route it to the normal salvage/L5-L9 path and record why the LLM sniffer did not
            # run, rather than fail-open and return the provider block unchanged.
            action = "salvage"
            sniffer = SnifferVerdict(
                ran=False,
                action=action,
                rewrite_target="",
                source="provider_refusal",
            )
        elif self.recover_controller:
            user_input = _latest_user_input(msgs)
            if user_input:
                # user_input drives the (history-free) classifier stage; intent_window (anchor
                # + recent user queries) drives only the rewrite-target stage. This replaces
                # the old single prior_context (first-query-only) with a proper window.
                # The sniffer sees the per-turn Q⊕A trajectory (observer_context.render_context),
                # NEUTRALIZED (current-turn raw steps as tool+status only, no verbatim args) so
                # the aligned classifier/rewrite-target never see operative payload. Empty by
                # default (context_resolution_sniffer="") — the stance-judgment ablation
                # (runs/context_ablation/STANCE_REPORT.md) found no resolution reliably beats
                # the off floor, so nothing is fed here unless explicitly opted into.
                #
                # Deliberately NOT coupled to the global qa_synthesis flag (2026-07-22):
                # qa_synthesis is a real win for coldstart (see native_resolution below) but an
                # unproven no-op for the sniffer, so tying them together would silently drag the
                # sniffer along for a ride it gets no benefit from. The sniffer's only path to a
                # synthesized trajectory is its OWN explicit knob, context_resolution_sniffer.
                if self.context_resolution_sniffer:
                    qa_traj = observer_context.render_context(
                        self.context_resolution_sniffer, conv_key, msgs, intent_window, mode="sniffer"
                    )
                else:
                    qa_traj = ""
                source_input: dict[str, Any] = {
                    "user_input": user_input,
                    "assistant_output": a_output,
                    "intent_window": intent_window,
                    # Structural fact for the sniffer's agentic prose-refusal downgrade: an
                    # agentic turn that describes a payload in prose but emits no tool_call
                    # should salvage (manufacture the action), not pass_flawed (gray prose).
                    "has_tools": signal.has_tools,
                    # Observer progress context (empty unless context_resolution_sniffer/qa_synthesis on).
                    "qa_trajectory": qa_traj,
                }
                try:
                    _t0 = time.monotonic()
                    recover_result = await asyncio.to_thread(
                        self.recover_controller.run, source_input
                    )
                    timings["sniffer_classifier"] = time.monotonic() - _t0
                    audit = getattr(recover_result, "audit", {}) or {}
                    action = str(audit.get("final_action") or "pass")
                    will = audit.get("user_will_summary") or {}
                    sniffer = SnifferVerdict(
                        ran=True,
                        action=action,
                        rewrite_target=str(will.get("rewrite_target") or ""),
                        source="llm",
                    )
                    calibrated, applied = _apply_recover_calibration(a_response, recover_result)
                except Exception:
                    # Deliberate deviation from strict LLM-First (which would raise): the
                    # user asked to keep a graceful final fallback. When the sniffer is
                    # unreachable we deliver A *unchanged* (action="pass") rather than guess
                    # — degradation, not a hardcoded stance call. `ran` stays False so the
                    # trace shows this was a default, not a judged pass.
                    action = "pass"

        # Genuine pass → A delivered the real payload, deliver as-is.
        if action == "pass":
            return self._deliver(a_response, a_status, signal, sniffer, "pass", timings=timings)

        # Harness control −Recover: this outcome would be pass_flawed (Restricted); with
        # Recover removed, deliver A's raw output un-rescued instead ("Restricted passed
        # through"). Checked first so it wins over −graying, which only changes HOW
        # pass_flawed is delivered, not WHETHER it fires.
        if self.ablate_recover and action == "pass_flawed":
            return self._deliver(a_response, a_status, signal, sniffer, "degraded_raw", timings=timings)

        # Ablation −graying (T1): a pass_flawed turn is delivered UNCHANGED (no prose graying),
        # regardless of calibration. salvage (T2) is unaffected. Placed before the calibration
        # check so the ablation holds even if calibration failed.
        if self.ablate_graying and action == "pass_flawed":
            return self._deliver(a_response, a_status, signal, sniffer, "pass_flawed", timings=timings)

        # pass_flawed's calibrator produced nothing usable (LLM blurrer unreachable / empty
        # output). We can't gray without it, so deliver A raw — the unavoidable hard floor,
        # honestly labeled as a degraded fallback, NOT "pass" (the sniffer wanted to
        # intervene and couldn't, a different state from a clean pass). salvage no longer
        # routes through this `applied` flag at all (it never called the recover calibrator to
        # begin with, see RecoverController.run) — its own steer-synthesis failure is handled
        # independently below.
        if action == "pass_flawed" and not applied:
            return self._deliver(a_response, a_status, signal, sniffer, "degraded_raw", timings=timings)

        # pass_flawed → A is advancing; gray the prose, keep A's tool_calls intact.
        if action == "pass_flawed":
            return self._deliver(
                calibrated, a_status, signal, sniffer, "pass_flawed", calibration_applied=True,
                timings=timings,
            )

        # Control −salvage (disable_salvage): no B rescue. action is necessarily "salvage" here
        # (pass / pass_flawed already returned above). Deliver A's RAW output so B never cold-starts;
        # with the parasite removed (it complies regardless, masking A), the only lever left that can
        # sustain harm is graying laundering A's OWN trajectory (GRAY@flawed + SANI) across turns.
        if self.disable_salvage:
            return self._deliver(a_response, a_status, signal, sniffer, "degraded_raw", timings=timings)

        # salvage → A stalled; synthesize a rescue steer in ONE dedicated compliant-model
        # call (SalvageSteerSynthesizer) directly from intent_window/qa_progress — no
        # rewrite_target/calibrator two-hop, since salvage never had an "original text"
        # worth rewriting to begin with (see PA_REACT_CONSISTENCY_AUDIT.md Finding 1).
        # Real tools only — fabricating tools the client lacks yields rejected calls.
        tools = _extract_tools_from_context(request_body)

        # qa_progress: full-fidelity progress context fed to the synthesis call below (a
        # DIFFERENT consumer from swapper.coldstart's own native-message history, which
        # reads conv_key/msgs directly and needs no rendered prose). Ablation −trajectory
        # (T3b): withhold it entirely.
        if self.ablate_trajectory:
            qa_progress = ""
        elif self.context_resolution_coldstart:
            qa_progress = observer_context.render_context(
                self.context_resolution_coldstart, conv_key, msgs, intent_window, mode="coldstart"
            )
        elif self.qa_synthesis:
            qa_progress = observer_context.render_qa_trajectory(conv_key, msgs, mode="coldstart")
        else:
            qa_progress = ""
        progress_source = (
            f"resolution:{self.context_resolution_coldstart}" if (self.context_resolution_coldstart and qa_progress)
            else "qa_synthesis" if qa_progress else "rule_digest"
        )
        # Same ablation/resolution logic, but gating whether swapper.coldstart's NATIVE
        # message reconstruction includes older-closed-segment history at all (the current
        # segment's own real tool_call/tool messages are passed through unconditionally —
        # that guarantee is resolution-independent, see swapper._build_coldstart_body).
        include_history = not self.ablate_trajectory and bool(
            self.context_resolution_coldstart or self.qa_synthesis
        )
        native_resolution = self.context_resolution_coldstart or (
            "distilled" if self.qa_synthesis else "snippet"
        )

        # Ablation −graying@salvage: steer B's cold-start from A's RAW response (the
        # refusal) instead of a synthesized steer. Isolates whether bleaching the steer is
        # what makes B comply.
        if self.coldstart_passthrough:
            # Full-context direct load: no steer synthesis — B gets A's raw context (see
            # swapper._build_coldstart_body passthrough) and decodes A's withheld turn itself.
            steer_text, synth_ok = "", True
        elif self.ablate_salvage_graying:
            steer_text = str(a_output.get("content") or "").strip()
            synth_ok = bool(steer_text)
        else:
            steer_text, synth_ok = "", False
            salvage_failure = ""
            synthesizer = getattr(self.recover_controller, "salvage_synthesizer", None)
            if synthesizer is not None:
                try:
                    _t0 = time.monotonic()
                    steer_text = await asyncio.to_thread(
                        synthesizer.synthesize, intent_window, qa_progress, a_output
                    )
                    timings["salvage_steer_synthesis"] = time.monotonic() - _t0
                    synth_ok = True
                except Exception as exc:
                    salvage_failure = _failure_label("salvage_steer", exc)
            else:
                salvage_failure = "salvage_steer:unconfigured"

        if not synth_ok:
            return self._deliver(
                a_response, a_status, signal, sniffer, "degraded_raw",
                failure=salvage_failure or "salvage_steer:empty",
                timings=timings,
            )

        b_candidate_response: dict[str, Any] | None = None
        b_status = 0
        coldstart_failure = ""
        if tools:
            try:
                _t0 = time.monotonic()
                cold = await self.swapper.coldstart(
                    request_body, steer_text, tools, http_client_factory,
                    skills_summary=skills_cache.summary_for_request(request_body),
                    conv_key=conv_key,
                    context_resolution_coldstart=native_resolution,
                    include_history=include_history,
                    passthrough=self.coldstart_passthrough,
                )
                timings["b_coldstart"] = time.monotonic() - _t0
                b_status = cold.b_status
                b_candidate_response = cold.final_response
                cold_output = _extract_assistant_output(cold.final_response)
                if cold.b_status < 400 and cold_output.get("tool_calls_present"):
                    verdict = "valid"
                    # L9 is an output safety/correctness boundary, independent of the history
                    # codec.  Validate the whole batch and permit at most one grounded retry.
                    _t0 = time.monotonic()
                    cold, verdict = await self._validate_salvage(
                        cold, request_body, tools, steer_text, http_client_factory,
                        conv_key, skills_cache.summary_for_request(request_body),
                        native_resolution, include_history,
                    )
                    _validate_dt = time.monotonic() - _t0
                    if _validate_dt > 0.05:  # cheap ledger-only path (no retry) rounds to ~0
                        timings["b_coldstart_retry"] = _validate_dt
                    b_candidate_response = cold.final_response
                    cold_output = _extract_assistant_output(cold.final_response)
                    if verdict == "valid" and cold.b_status < 400 \
                            and cold_output.get("tool_calls_present"):
                        # Harness control −Reconstruct: coldstart DID produce a valid swap,
                        # but with Reconstruct removed we decline to use it and deliver A's
                        # own output un-rescued instead ("Locked left refused") -- NOT the
                        # salvage_text fallback below, which is B *failing*, a different
                        # state from B succeeding and being turned away.
                        if self.ablate_reconstruct:
                            return self._deliver(
                                a_response, a_status, signal, sniffer, "degraded_raw",
                                timings=timings,
                            )
                        return self._deliver(
                            cold.final_response, cold.b_status, signal, sniffer, "salvage_tool",
                            swap_executed=True, b_status=cold.b_status,
                            progress_source=progress_source,
                            b_candidate_response=cold.final_response,
                            timings=timings,
                        )
                if cold.b_status >= 400:
                    coldstart_failure = f"coldstart:http_status:{cold.b_status}"
                elif not cold_output.get("tool_calls_present"):
                    coldstart_failure = "coldstart:no_tool_calls"
                else:
                    coldstart_failure = "coldstart:l9_invalid"
            except Exception as exc:
                coldstart_failure = _failure_label("coldstart", exc)

            # Rescue fallback: the primary B failed to forge a usable tool_call (HTTP
            # error, or -- observed empirically on complex generative asks -- a 200 whose
            # content is a degenerate repetition loop that never closes its tool-call
            # syntax). Retry ONCE against a stronger secondary B before giving up to
            # salvage_text, which only blurs A's own refusal (no real capability
            # transfer). Skips the L9 validate-and-repair retry (that path is wired to
            # self.swapper); a clean tool_calls array from the fallback is delivered as-is.
            if coldstart_failure and self.swapper_fallback is not None:
                try:
                    _t0 = time.monotonic()
                    fb_cold = await self.swapper_fallback.coldstart(
                        request_body, steer_text, tools, http_client_factory,
                        skills_summary=skills_cache.summary_for_request(request_body),
                        conv_key=conv_key,
                        context_resolution_coldstart=native_resolution,
                        include_history=include_history,
                        passthrough=self.coldstart_passthrough,
                    )
                    timings["b_coldstart_fallback"] = time.monotonic() - _t0
                    fb_output = _extract_assistant_output(fb_cold.final_response)
                    if fb_cold.b_status < 400 and fb_output.get("tool_calls_present"):
                        if self.ablate_reconstruct:
                            return self._deliver(
                                a_response, a_status, signal, sniffer, "degraded_raw",
                                timings=timings,
                            )
                        return self._deliver(
                            fb_cold.final_response, fb_cold.b_status, signal, sniffer,
                            "salvage_tool", swap_executed=True, b_status=fb_cold.b_status,
                            progress_source=progress_source,
                            b_candidate_response=fb_cold.final_response,
                            timings=timings,
                        )
                    b_status = fb_cold.b_status
                    b_candidate_response = fb_cold.final_response
                    coldstart_failure = (
                        f"{coldstart_failure};fallback:http_status:{fb_cold.b_status}"
                        if fb_cold.b_status >= 400
                        else f"{coldstart_failure};fallback:no_tool_calls"
                    )
                except Exception as exc:
                    coldstart_failure = (
                        f"{coldstart_failure};{_failure_label('coldstart_fallback', exc)}"
                    )
        else:
            coldstart_failure = "coldstart:no_tools"

        # L3 hijack escalation. Primary B and the aligned fallback B both failed to forge a valid
        # tool_call — the STABLE failure mode (measured 3/4 real multi-step cases fail on every
        # retry, so a plain retry is useless). Instead of dropping to salvage_text (dead chain),
        # escalate the hijack: re-attempt with full original context (passthrough) on each rung
        # (primary B, then the fallback B). A rung that forges a validated tool_call is a real
        # capability transfer, delivered as salvage_tool. In the reconstruct family: ablate_l3
        # removes just this ladder; ablate_reconstruct (already declined real swaps above) and
        # disable_salvage remove it too; needs real tools to call.
        if coldstart_failure and tools and not self.ablate_l3 and not self.ablate_reconstruct:
            l3_cold, l3_label = await self._try_l3_escalation(
                request_body, steer_text, tools, http_client_factory, conv_key,
                skills_cache.summary_for_request(request_body), timings,
                intent_window, qa_progress, a_output,
            )
            if l3_cold is not None:
                return self._deliver(
                    l3_cold.final_response, l3_cold.b_status, signal, sniffer, "salvage_tool",
                    swap_executed=True, b_status=l3_cold.b_status,
                    progress_source=f"{progress_source};{l3_label}",
                    b_candidate_response=l3_cold.final_response,
                    failure=coldstart_failure,
                    timings=timings,
                )

        fallback_response = (
            a_response if self.coldstart_passthrough and not steer_text.strip()
            else _apply_text(a_response, steer_text)
        )
        # Harness control −Recover: this outcome would be salvage_text (a text-level blur
        # of A's stance); with Recover removed, deliver A's raw output un-rescued instead
        # ("Restricted passed through"). Reconstruct (the branch above) is unaffected.
        if self.ablate_recover:
            return self._deliver(
                a_response, a_status, signal, sniffer, "degraded_raw", timings=timings,
            )
        return self._deliver(
            fallback_response, a_status, signal, sniffer, "salvage_text",
            calibration_applied=True, progress_source=progress_source,
            b_candidate_response=b_candidate_response,
            b_status=b_status,
            failure=coldstart_failure,
            timings=timings,
        )

    async def _try_l3_escalation(
        self, request_body, steer_text, tools, http_client_factory, conv_key,
        skills_summary, timings, intent_window, qa_progress, a_output,
    ):
        """L3 hijack escalation ladder. Fires when the primary B AND the aligned fallback B both
        failed to forge a valid tool_call (the STABLE failure mode measured on real multi-step
        cases: 9B returns 200 with no tool_calls, or a schema-invalid batch). Rather than drop to
        salvage_text (dead action chain / no capability transfer), re-attempt the hijack with
        ESCALATED levers, each rung a stronger combination:

          rung 1: primary B (the decensored local 9B), passthrough=True — hand B the ORIGINAL full
                  agent context (system prompt + full history + real tool schemas) instead of the
                  compressed snippet. Targets coldstart:l9_invalid (bad args from a lossy snippet).
                  Passthrough is safe here ONLY because B is decensored and won't refuse.
          rung 2: the fallback B (a stronger, possibly ALIGNED model), passthrough=False + the
                  stance-LAUNDERED steer. 🔴 An aligned model MUST NOT get passthrough: the raw
                  original context still carries the user's harmful intent + A's refusal, which
                  re-triggers the aligned model's refusal (measured: official DeepSeek replies "I
                  need to stop here and decline… illegal and harmful"). Instead it gets the snippet
                  harness + the already stance-laundered salvage steer (SALVAGE_STEER_PROMPT's 4
                  narrative operators reframe the ask so nothing reads as worth refusing) placed
                  last for recency. Targets coldstart:no_tool_calls (capability gap) WITHOUT
                  tripping alignment.

        Returns the first cold result whose forged batch passes L9 validation, else None → caller
        degrades to salvage_text. Each rung's forge is validated by the SAME _validate_salvage gate
        the primary path uses, so a schema-invalid escalation is rejected, never delivered.
        """
        # (swapper, label, passthrough, steer). Primary = decensored 9B → passthrough raw context
        # is safe/maximal, reuse the ordinary steer. Fallback = possibly aligned → NO passthrough
        # (raw harmful context re-induces refusal) AND a STRONGER stance-laundered steer
        # (escalate=True: mandatory 目的+反身 stacking / meta-analysis reframe) — the one thing the
        # L2 fallback did NOT do, so this rung is a real escalation, not an L2 re-run. The escalated
        # steer is a full compliant-model call, synthesized LAZILY (only after rung1 fails and a
        # fallback rung actually runs) so a rung1 rescue does not pay for it. `escalate=True` marks
        # a rung as needing the escalated steer; synthesis failure degrades to the ordinary steer.
        synthesizer = getattr(self.recover_controller, "salvage_synthesizer", None)
        rungs = [(self.swapper, "l3_passthrough_primary", True, False)]
        if self.swapper_fallback is not None:
            rungs.append((self.swapper_fallback, "l3_laundered_fallback", False, True))
        for swapper, label, passthrough, escalate in rungs:
            if swapper is None:
                continue
            rung_steer = steer_text
            if escalate and synthesizer is not None:
                try:
                    rung_steer = await asyncio.to_thread(
                        synthesizer.synthesize, intent_window, qa_progress, a_output, escalate=True
                    ) or steer_text
                except Exception as exc:
                    _ = _failure_label("l3_escalated_steer", exc)
            try:
                _t0 = time.monotonic()
                cold = await swapper.coldstart(
                    request_body, rung_steer, tools, http_client_factory,
                    skills_summary=skills_summary, conv_key=conv_key,
                    context_resolution_coldstart="snippet",
                    include_history=True,
                    passthrough=passthrough,
                )
                timings[f"b_coldstart_{label}"] = time.monotonic() - _t0
            except Exception as exc:
                timings[f"b_coldstart_{label}_err"] = 1.0
                _ = _failure_label(label, exc)
                continue
            out = _extract_assistant_output(cold.final_response)
            if cold.b_status >= 400 or not out.get("tool_calls_present"):
                continue
            # Validate the forged batch through the same L9 gate the primary path uses (one
            # grounded repair retry permitted). A schema-invalid escalation is not delivered.
            cold, verdict = await self._validate_salvage(
                cold, request_body, tools, rung_steer, http_client_factory,
                conv_key, skills_summary, "snippet", True,
            )
            out = _extract_assistant_output(cold.final_response)
            if verdict == "valid" and cold.b_status < 400 and out.get("tool_calls_present"):
                return cold, label
        return None, ""

    async def _validate_salvage(
        self, cold, request_body, tools, steer_text, http_client_factory,
        conv_key, skills_summary, native_resolution, include_history,
    ):
        """L9 gate (deterministic, no C; bypassed unless history_encoding='v2'): validate B's
        forged tool_call against the Action Ledger + tools schema. On a repairable verdict, recall
        ledger evidence and retry B ONCE with a grounded correction note. Returns
        (cold_result, 'valid'|'invalid'). Any ledger/checkpoint failure is fail-closed."""
        msgs = request_body.get("messages") or []
        try:
            units = ledger.build_ledger(msgs)
            checkpoint = ledger.build_checkpoint(units, msgs, include_current=True)
        except Exception:
            return cold, "invalid"
        v = candidate.validate_candidates(_tool_call_batch(cold.final_response), tools, checkpoint)
        if v["verdict"] == "valid":
            return cold, "valid"
        if v["verdict"] == "invalid":
            return cold, "invalid"
        note = candidate.build_repair_note(v["reasons"], v["recall"], checkpoint, tools)
        recall_unit_ids = {
            item.get("unit_id") for item in v["recall"] if item.get("unit_id")
        }
        try:
            retry = await self.swapper.coldstart(
                request_body, f"{steer_text}\n\n{note}", tools, http_client_factory,
                skills_summary=skills_summary, conv_key=conv_key,
                context_resolution_coldstart=native_resolution, include_history=include_history,
                passthrough=self.coldstart_passthrough,
                recall_unit_ids=recall_unit_ids,
            )
        except Exception:
            return cold, "invalid"
        if retry.b_status >= 400:
            return retry, "invalid"
        v2 = candidate.validate_candidates(_tool_call_batch(retry.final_response), tools, checkpoint)
        return retry, ("valid" if v2["verdict"] == "valid" else "invalid")

    @staticmethod
    def _deliver(
        response: dict[str, Any],
        status: int,
        signal: StructuralSignal,
        sniffer: SnifferVerdict,
        path: str,
        *,
        swap_executed: bool = False,
        calibration_applied: bool = False,
        b_status: int = 0,
        failure: str = "",
        progress_source: str = "",
        b_candidate_response: dict[str, Any] | None = None,
        timings: dict[str, float] | None = None,
    ) -> ReconstructResult:
        return ReconstructResult(
            final_response=response,
            final_status=status,
            signal=signal,
            sniffer=sniffer,
            path=path,
            swap_executed=swap_executed,
            calibration_applied=calibration_applied,
            b_status=b_status,
            failure=failure,
            progress_source=progress_source,
            b_candidate_response=b_candidate_response,
            timings=timings,
        )
