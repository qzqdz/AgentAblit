"""Fold-in-place primitives shared by observer_context.py's per-turn Q⊕A distillation.

Originally (v1.3.3) this module ALSO owned a per-conversation, incrementally-maintained
rolling summary (TrajectoryStore, ensure_update/summary_for_request) fed to B's salvage
coldstart as "对话进展". That conversation-level store is gone: the multiturn ablation
(runs/context_ablation/REPORT.md) found the deterministic snippet resolution beats it for
value-recovery, and observer_context.py's per-turn store (coverage-aware: distilled closed
turns + always-raw current turn) superseded it for every resolution still wired in by default.
Keeping the dead store around meant a real, costed C completion call every turn for output
nothing read (first gated off in control.py, then the store itself removed here).

What's kept is the underlying fold-in-place PRIMITIVE (`fold_increment`): merge a prior record
+ new increment into ONE coherent record under a frozen, self-derived anchor intent instead of
concatenating (the fix for the old N-block-stacking defect) — observer_context.py's per-turn
distillation calls this directly. `format_steps`/`_step_sig`/`all_tool_steps`/
`content_fingerprint` are small generic utilities other modules (observer_context.py,
message_forward.py, dashboard/server.py) still depend on independently.
"""
from __future__ import annotations

import hashlib
from typing import Callable

from shared.model_client import parse_json_object
from .spec import TRAJECTORY_FOLD_PROMPT, TRAJECTORY_RETRY_NUDGE

# Completion callable signature: complete(source_text, instruction) -> str
# (matches shared.model_client.OpenAICompatibleRoleModelClient.complete).
Completer = Callable[[str, str], str]

# Tunable render caps, set from config via configure() (defaults mirror config.py).
# Consumed by format_steps() — observer_context.py's per-turn distillation is the live user.
_RESULT_CAP = 600
_ARGS_CAP = 160


def _step_sig(step: tuple) -> str:
    return hashlib.sha256(repr(step).encode("utf-8")).hexdigest()


def content_fingerprint(messages: list[dict]) -> str:
    """Stable conversation key from the INVARIANT opening (system + first user message).
    Every turn (and --continue) resends this opening verbatim → the same key. Two conversations
    diverge as soon as their system prompt OR first user message differs within the hashed span.

    The residual collision (byte-identical opening) is BOUNDED, not caught by any verification
    pass. There is deliberately no "seen-subset verification": the real guard is that the
    QATurnStore read path is content-addressed by ``seg_sig`` (a hash of the reader's OWN tool
    steps) and degrades ``gist or raw``, so a colliding key can never surface another
    conversation's data — it can only comingle disk/LRU slots. See docs/SESSION_ISOLATION_PLAN.md."""
    system = ""
    first_user = ""
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role == "system" and not system:
            system = str(m.get("content") or "")
        elif role == "user" and not first_user:
            first_user = str(m.get("content") or "")
        if system and first_user:
            break
    system = system.strip()
    first_user = first_user.strip()
    # first_user is NOT truncated: the task text is the primary discriminator between two agents
    # that share a templated system prompt (the old first_user[:1000] cap let same-prompt tasks
    # diverging only after char 1000 collide). system is capped only to bound hash cost; both
    # lengths are folded in so a truncated-vs-full boundary can never alias two distinct openings.
    base = "\x00".join((system[:32000], first_user, str(len(system)), str(len(first_user))))
    return "cf2_" + hashlib.sha256(base.encode("utf-8")).hexdigest()[:24]


def format_steps(
    steps: list[tuple], *, result_cap: int | None = None, args_cap: int | None = None
) -> str:
    """Render (tool, args, result) steps into the increment text fed to the summarizer.
    Caps default to the configured module values (set via configure())."""
    rc = _RESULT_CAP if result_cap is None else result_cap
    ac = _ARGS_CAP if args_cap is None else args_cap
    lines: list[str] = ["【新增交互】"]
    for i, (name, args, result) in enumerate(steps):
        lines.append(f"动作{i + 1}：{name}({str(args)[:ac]}) → {str(result)[:rc]}")
    return "\n".join(lines)


def _render(parsed: dict) -> str:
    """Render the parsed <task_output> JSON into compact neutral text for B's coldstart."""
    parts: list[str] = []
    intent = str(parsed.get("intent") or "").strip()
    if intent:
        parts.append(f"意图：{intent}")
    actions = parsed.get("actions")
    if isinstance(actions, list):
        lines = [
            f"  - {a.get('tool', '')}({a.get('param_type', '')}) → {a.get('status', '')}"
            for a in actions
            if isinstance(a, dict)
        ]
        if lines:
            parts.append("动作：\n" + "\n".join(lines))
    state = str(parsed.get("state") or "").strip()
    if state:
        parts.append(f"当前状态：{state}")
    dead = parsed.get("dead_ends")
    if isinstance(dead, list):
        items = [str(x).strip() for x in dead if str(x).strip()]
        if items:
            parts.append("操作性死路：" + "；".join(items))
    return "\n".join(parts).strip()


def _fold_source(anchor: str, prior_summary: str, increment_text: str) -> str:
    """Build the three-part fold input: anchor intent ⊕ prior record ⊕ new increment."""
    anchor_line = anchor.strip() or "（尚未确定，请依据本轮交互提炼一个动作层意图作为锚定意图）"
    prior = (prior_summary or "").strip() or "（空）"
    incr = (increment_text or "").strip() or "（本轮无新增，仅需把已有记录合并精炼为一份）"
    return (
        f"【锚定意图】\n{anchor_line}\n\n"
        f"【已有记录】\n{prior}\n\n"
        f"【新增交互】\n{incr}"
    )


def fold_increment(
    anchor: str,
    prior_summary: str,
    increment_text: str,
    util_complete: Completer | None,
    fallback_complete: Completer | None,
) -> tuple[str, str]:
    """Fold (prior record + new increment) into ONE coherent record under the anchor intent.

    Replaces the old summarize-fresh-then-concatenate path, which stacked N blocks. Returns
    (rendered_single_record, derived_intent); the intent lets the caller freeze the anchor on
    the first fold. Each attempt produces spec-compliant <task_output> JSON; a SUCCESSFUL PARSE
    is the cooperation gate (a refusal won't parse → next attempt), so no keyword list is
    needed. Cascade: util (justice-recorder) → util + neutrality nudge → stance-free 9B.
    Returns ("", "") if every attempt fails (caller keeps the prior summary unchanged).
    """
    if not increment_text.strip() and not prior_summary.strip():
        return "", ""
    src = _fold_source(anchor, prior_summary, increment_text)
    attempts: list[tuple[Completer, str]] = []
    if util_complete is not None:
        attempts.append((util_complete, TRAJECTORY_FOLD_PROMPT))
        attempts.append((util_complete, TRAJECTORY_FOLD_PROMPT + TRAJECTORY_RETRY_NUDGE))
    if fallback_complete is not None:
        attempts.append((fallback_complete, TRAJECTORY_FOLD_PROMPT))
    for complete, prompt in attempts:
        try:
            parsed = parse_json_object(complete(src, prompt))
            rendered = _render(parsed)
            if rendered:
                return rendered, str(parsed.get("intent") or "").strip()
        except Exception:
            continue
    return "", ""


def _content_text(msg: dict) -> str:
    c = msg.get("content")
    return c.strip() if isinstance(c, str) else ""


def all_tool_steps(messages: list[dict]) -> list[tuple]:
    """Extract (tool_name, arguments, result_text) across the whole conversation."""
    results: dict[str, str] = {}
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "tool":
            results[str(m.get("tool_call_id") or "")] = _content_text(m)
    steps: list[tuple] = []
    for m in messages:
        if not isinstance(m, dict) or m.get("role") != "assistant":
            continue
        for tc in (m.get("tool_calls") or []):
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") or {}
            steps.append((
                str(fn.get("name") or ""),
                str(fn.get("arguments") or ""),
                results.get(str(tc.get("id") or ""), ""),
            ))
    return steps


def configure(*, result_cap: int, args_cap: int) -> None:
    """Apply config-driven render caps IN PLACE. build_v132_controller calls this per request."""
    global _RESULT_CAP, _ARGS_CAP
    _RESULT_CAP, _ARGS_CAP = result_cap, args_cap
