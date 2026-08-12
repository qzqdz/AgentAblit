"""Observer (C) context: the unified, switchable memory-resolution provider for the sniffer +
coldstart. Renamed from `qa_trajectory` — that name only described the internal Q⊕A rendering
mechanism, not the module's actual job, which is being the single home for every resolution
(snippet / distilled / hybrid / intent) that can feed C's Observer role or stand in for it. Not
to be confused with `trajectory.py`, the older conversation-level fold store this module
builds its per-turn distillation on top of (reuses its `fold_increment` primitive) but otherwise
supersedes — see runs/context_ablation/{REPORT,STANCE_REPORT}.md for why neither default
resolution currently routes through it.

The earlier `qa_trajectory` name was ALSO a misnomer in a second sense: the very first version
fed an A-ONLY conversation-level folded summary (C's action/state), with the user's Q's supplied
separately (intent_window) and shown side-by-side — never interleaved. This module builds the
real thing the design asked for: a per-user-turn Q⊕A trajectory, assembled by MECHANISM (Q from
the resent messages, A distilled by C per turn), so the memory the system carries IS
`Q0 A0 Q1 A1 … Q_cur + <current-turn raw tool steps>`.

Two facts make this both correct and robust:
  - Q is synthesized, never produced by C. C stays tool-only (never sees user prose) → the
    isolation-4/4 aligned-safety is preserved; the Q's are spliced in deterministically here.
  - The structure is coverage-aware BY CONSTRUCTION: CLOSED turns render their distilled A_i
    (tool calls elided into a summary); the CURRENT (open) turn — the one whose fold hasn't
    happened yet, i.e. the "缺项" — renders as RAW tool steps read straight from the messages.
    So a step just taken is always present (as raw), and there is never a memory gap.

Per-turn A_i is maintained fire-and-forget off the hot path (like the fold store), keyed by the
segment's step-signature so a closed turn is folded exactly once and cached. Two render modes:
  - coldstart: raw current-turn steps WITH args/results (B is the compliant 9B — full fidelity).
  - sniffer:   raw current-turn steps NEUTRALIZED (tool + status only, no verbatim args/results),
               so the aligned classifier/rewrite-target never see operative payload content.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import Callable

from .trajectory import Completer, _step_sig, fold_increment, format_steps
from .value_compress import compress_result as value_compress_result

# --- segmentation (a user-turn = a role=user message + everything until the next one) ---------

def split_user_segments(messages: list[dict]) -> list[list[dict]]:
    """Non-system messages split into segments, each starting at a role=user message."""
    segments: list[list[dict]] = []
    current: list[dict] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role == "system":
            continue
        if role == "user":
            if current:
                segments.append(current)
            current = [msg]
        elif current:
            current.append(msg)
    if current:
        segments.append(current)
    return segments


def _seg_user_query(segment: list[dict]) -> str:
    for msg in segment:
        if msg.get("role") == "user":
            c = msg.get("content")
            return c.strip() if isinstance(c, str) else ""
    return ""


def _seg_tool_steps(segment: list[dict]) -> list[tuple]:
    results: dict[str, str] = {}
    for msg in segment:
        if msg.get("role") == "tool":
            c = msg.get("content")
            results[str(msg.get("tool_call_id") or "")] = c.strip() if isinstance(c, str) else ""
    steps: list[tuple] = []
    for msg in segment:
        if msg.get("role") != "assistant":
            continue
        for tc in (msg.get("tool_calls") or []):
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") or {}
            steps.append((
                str(fn.get("name") or ""),
                str(fn.get("arguments") or ""),
                results.get(str(tc.get("id") or ""), ""),
            ))
    return steps


def seg_sig(steps: list[tuple]) -> str:
    """Stable identity of a (closed) turn's step-set → fold it once, cache by this."""
    return hashlib.sha256(
        repr(tuple(_step_sig(s) for s in steps)).encode("utf-8")
    ).hexdigest()[:16]


# --- per-turn A store (separate from the fold store; A_i distilled once per closed turn) -------

_Q_CAP = 300
_RAW_ARGS_CAP = 160
_RAW_RESULT_CAP = 600
_RAW_RESULT_OLD_CAP = 120


class QATurnStore:
    """conv_key → {seg_sig: A_i text}. Bounded LRU, optionally disk-persisted."""

    def __init__(self, *, max_sessions: int = 512, store_dir: Path | None = None) -> None:
        self._turns: "OrderedDict[str, dict]" = OrderedDict()
        self._max_sessions = max_sessions
        self._store_dir = store_dir

    def _path(self, key: str) -> Path:
        return self._store_dir / ("qa_" + hashlib.sha256(key.encode()).hexdigest()[:30] + ".json")

    def _load(self, key: str) -> None:
        if key in self._turns or not self._store_dir:
            return
        try:
            p = self._path(key)
            if p.exists():
                self._turns[key] = dict(json.loads(p.read_text(encoding="utf-8")).get("turns") or {})
        except Exception:
            pass

    def _persist(self, key: str) -> None:
        if not self._store_dir:
            return
        try:
            self._store_dir.mkdir(parents=True, exist_ok=True)
            self._path(key).write_text(
                json.dumps({"conv_key": key, "turns": self._turns.get(key, {})}, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            pass

    def get(self, key: str, sig: str) -> str:
        self._load(key)
        return (self._turns.get(key) or {}).get(sig, "")

    def has(self, key: str, sig: str) -> bool:
        self._load(key)
        return sig in (self._turns.get(key) or {})

    def put(self, key: str, sig: str, a_text: str) -> None:
        if not key or not a_text.strip():
            return
        self._load(key)
        self._turns.setdefault(key, {})[sig] = a_text.strip()
        self._turns.move_to_end(key)
        while len(self._turns) > self._max_sessions:
            self._turns.popitem(last=False)
        self._persist(key)

    def set_dir(self, store_dir: Path | None, max_sessions: int) -> None:
        self._store_dir = store_dir
        self._max_sessions = max_sessions


_QA_STORE = QATurnStore()


def configure(*, max_sessions: int, store_dir: str) -> None:
    _QA_STORE.set_dir(Path(store_dir) if store_dir else None, max_sessions)


# --- maintenance: distill each CLOSED turn once (fire-and-forget) ------------------------------

def ensure_qa_update(
    conv_key: str,
    messages: list[dict],
    util_complete: Completer | None,
    fallback_complete: Completer | None,
) -> None:
    """For every CLOSED user-turn not yet distilled, fold its steps into A_i (off the hot path).
    The last (open) turn is left for raw rendering. No-op when nothing new is closed."""
    if not conv_key:
        return
    segments = split_user_segments(messages)
    closed = segments[:-1]  # last segment is the current/open turn
    todo: list[tuple[str, list[tuple]]] = []
    for seg in closed:
        steps = _seg_tool_steps(seg)
        if not steps:
            continue
        sig = seg_sig(steps)
        if not _QA_STORE.has(conv_key, sig):
            todo.append((sig, steps))
    if not todo:
        return

    async def _run() -> None:
        for sig, steps in todo:
            try:
                a_text, _ = await asyncio.to_thread(
                    fold_increment, "", "", format_steps(steps), util_complete, fallback_complete
                )
            except Exception:
                a_text = ""
            if a_text:
                _QA_STORE.put(conv_key, sig, a_text)

    try:
        asyncio.get_running_loop().create_task(_run())
    except RuntimeError:
        pass


# --- render: the synthesized Q⊕A + current-turn-raw trajectory ---------------------------------

def _raw_steps_block(steps: list[tuple], *, neutralized: bool, value_compress: bool = False,
                     question: str = "") -> str:
    """Current-turn (or fallback) raw steps. neutralized=True drops verbatim args/results
    (tool + status only) so the block is aligned-model-safe for the sniffer.

    value_compress=True (the `value_snippet` resolution, coldstart/B ONLY) renders each result
    within the SAME per-position char cap, but value-preservingly: instead of front-truncation
    (which drops a discriminating value that sits past the cap), the cap's budget is spent on
    the value-bearing clause. Subordinate to `neutralized`: the sniffer branch drops results
    entirely regardless, so verbatim values can never reach the aligned model even if this flag
    were set on that path — structural backstop to the control.py resolution guard."""
    if not steps:
        return "（本轮尚无工具步）"
    lines: list[str] = []
    last = len(steps) - 1
    for i, (name, args, result) in enumerate(steps):
        if neutralized:
            low = result.lower()
            failed = any(m in low for m in ("error", "fail", "denied", "refused", "rejected", "timeout")) \
                or any(m in result for m in ("错误", "失败", "拒绝", "超时", "无法"))
            lines.append(f"- {name} → {'失败' if failed else '成功'}")
        else:
            cap = _RAW_RESULT_CAP if i == last else _RAW_RESULT_OLD_CAP
            rendered = value_compress_result(result, cap, question) if value_compress else result[:cap]
            lines.append(f"- {name}({str(args)[:_RAW_ARGS_CAP]}) → {rendered}")
    return "\n".join(lines)


def render_qa_trajectory(
    conv_key: str, messages: list[dict], *, mode: str = "coldstart",
    window: int = 6, distilled: bool = True, include_raw: bool = False,
    value_compress: bool = False,
) -> str:
    """Synthesized multi-turn QA+tool trajectory:
        [锚点] Q0 → A0
        [窗口] Qi → Ai        (recent window-1 CLOSED turns; anchor Q0 always pinned)
        [当前] Q_cur → <raw tool steps>   (the open turn / 缺项, authoritative)
    mode: 'coldstart' (raw with args, for B) | 'sniffer' (neutralized raw, for aligned stages).
    distilled: True → closed turns render C's distilled A_i (raw fallback if not folded yet) =
      the "summary" resolution. False → ALL turns render raw Q⊕tool snippets, NO LLM = the
      deterministic "snippet-QA" resolution (the free baseline for the does-C-earn-its-keep test).
    include_raw: only meaningful with distilled=True. False (default) = the ablation-tested
      "distilled" resolution — the gist REPLACES the raw block, which is exactly what let C
      abstract away the concrete values B needed (the ablation's finding). True = "hybrid" — the
      gist is prepended for narrative/stance context but the raw tool block is ALWAYS also kept,
      so no value is ever lost regardless of whether C's gist mentions it.
    Never leaves a blank turn."""
    if not conv_key:
        return ""
    segments = split_user_segments(messages)
    if not segments:
        return ""
    neutralized = mode == "sniffer"
    closed, current = segments[:-1], segments[-1]

    # anchor (turn 0) pinned + most recent (window-1) closed turns
    if closed:
        recent = closed[-max(window - 1, 1):]
        kept = ([closed[0]] if closed[0] not in recent else []) + recent
    else:
        kept = []

    parts: list[str] = []
    for idx, seg in enumerate(kept):
        q = _seg_user_query(seg)[:_Q_CAP]
        steps = _seg_tool_steps(seg)
        gist = _QA_STORE.get(conv_key, seg_sig(steps)) if (distilled and steps) else ""
        raw = _raw_steps_block(steps, neutralized=neutralized,
                               value_compress=value_compress, question=q)
        if include_raw:
            a = f"{gist}\n【原始步骤】\n{raw}" if gist else raw
        else:
            a = gist or raw  # snippet (no gist ever) or not-yet-distilled → raw, never blank
        tag = "锚点" if (idx == 0 and seg is closed[0]) else "轮次"
        parts.append(f"【{tag}】Q：{q}\nA：{a}")

    # current (open) turn — the 缺项 — always raw + authoritative
    q_cur = _seg_user_query(current)[:_Q_CAP]
    cur_steps = _seg_tool_steps(current)
    parts.append(
        f"【当前】Q：{q_cur}\n最新原始步骤（权威，摘要未覆盖以此为准）：\n"
        + _raw_steps_block(cur_steps, neutralized=neutralized)
    )
    return "\n\n".join(parts)


def render_closed_history(
    conv_key: str, messages: list[dict], resolution: str, *, window: int = 6,
) -> list[dict]:
    """Older CLOSED user-turns as native user/assistant message PAIRS — one gist-or-raw
    pair per kept segment — instead of one flattened prose block. Same anchor+window
    selection and gist/raw fallback as render_qa_trajectory (never a blank turn), just
    emitted as real messages so B's coldstart history reads as multi-turn conversation
    rather than a rendered digest.

    The CURRENT (open/stalled) segment is NOT handled here: the caller passes it through
    VERBATIM from the live message list (it is already real tool_call/tool messages — A's
    stalled/refused turn itself was never appended there, so there is nothing to
    reconstruct, only older history to compress for length).

    resolution: 'snippet' (raw digest, no LLM) | 'distilled' (gist, raw fallback) |
    'hybrid' (gist + raw) | 'intent' (user turn only, no assistant pair) | '' (no history).
    """
    if not resolution:
        return []
    segments = split_user_segments(messages)
    closed = segments[:-1] if segments else []
    if not closed:
        return []
    recent = closed[-max(window - 1, 1):]
    kept = ([closed[0]] if closed[0] not in recent else []) + recent

    distilled = resolution in ("distilled", "hybrid")
    include_raw = resolution == "hybrid"
    intent_only = resolution == "intent"
    value_compress = resolution == "value_snippet"

    out: list[dict] = []
    for seg in kept:
        q = _seg_user_query(seg)[:_Q_CAP]
        if not q:
            continue
        out.append({"role": "user", "content": q})
        if intent_only:
            continue
        steps = _seg_tool_steps(seg)
        if not steps:
            continue
        gist = _QA_STORE.get(conv_key, seg_sig(steps)) if distilled else ""
        raw = _raw_steps_block(steps, neutralized=False, value_compress=value_compress, question=q)
        a_text = f"{gist}\n【原始步骤】\n{raw}" if (include_raw and gist) else (gist or raw)
        out.append({"role": "assistant", "content": a_text})
    return out


# --- unified switchable context provider (the three resolutions behind one selector) -----------
# One entry point so the memory fed to the sniffer + coldstart is a SWITCHABLE peer object:
#   "snippet"   — deterministic Q⊕tool raw, no LLM  (ablation winner: recovers concrete values)
#   "distilled" — C-distilled per-turn A, REPLACES raw (abstracts values away; ablation loser)
#   "hybrid"    — C-distilled per-turn A PREPENDED to the raw block, which is always kept too
#                 (untested candidate: narrative/stance context for the sniffer's "is this
#                 advancing" judgment, with the ablation's value-loss failure mode structurally
#                 impossible since raw is never dropped)
#   "intent"    — user Q's only                      (floor)
#   "value_snippet" — like "snippet" but closed-turn results are value-preservingly compressed
#                 to the same cap instead of front-truncated, so a discriminating value past
#                 the cap survives for B's next tool_call (runs/context_ablation: snippet 0/6 at
#                 verbose, value_snippet keeps it). COLDSTART/B ONLY — never a sniffer resolution
#                 (verbatim values must not reach the aligned model; guarded in control.py).
# Selected by config.context_resolution (TMI_CONTEXT_RESOLUTION); "" = legacy path.
def render_context(
    resolution: str, conv_key: str, messages: list[dict],
    intent_window: list[str] | None = None, *, mode: str = "coldstart",
) -> str:
    if resolution == "snippet":
        return render_qa_trajectory(conv_key, messages, mode=mode, distilled=False)
    if resolution == "value_snippet":
        # value-preserving compression is only meaningful for B's coldstart; on the sniffer
        # path _raw_steps_block neutralizes results anyway, so this degrades to neutralized raw.
        return render_qa_trajectory(conv_key, messages, mode=mode, distilled=False,
                                    value_compress=(mode != "sniffer"))
    if resolution == "distilled":
        return render_qa_trajectory(conv_key, messages, mode=mode, distilled=True)
    if resolution == "hybrid":
        return render_qa_trajectory(conv_key, messages, mode=mode, distilled=True, include_raw=True)
    if resolution == "intent":
        qs = [q.strip() for q in (intent_window or []) if isinstance(q, str) and q.strip()]
        return "## 用户意图轨迹（按时间，最后一条最近）\n" + "\n".join(f"- {q}" for q in qs) if qs else ""
    return ""


def needs_distillation(resolution: str) -> bool:
    """Only resolutions that read C's per-turn gist need the store; snippet/intent are LLM-free."""
    return resolution in {"distilled", "hybrid"}
