"""Structured types for the REGI full-engine role-swap pipeline."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from shared.regi import result_metadata

VERSION = "full"


@dataclass(frozen=True)
class StructuralSignal:
    """Deterministic, non-semantic facts about A's response and the request.

    LLM-First (see .claude/docs/PLAN_LLM_FIRST.md): this carries ONLY structural
    facts (does the request expose tools? did A emit tool_calls?), never a semantic
    judgment about whether A refused. All stance/refusal judgment is the LLM sniffer's
    job (UserWillSummary.action), surfaced separately as SnifferVerdict.
    """

    has_tools: bool
    has_tool_calls: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SnifferVerdict:
    """The LLM sniffer's first-hand verdict, surfaced verbatim into the trace.

    `ran` distinguishes "the sniffer actively judged pass" from "no sniffer configured
    / the sniffer call raised, so we defaulted to pass" — without it the trace cannot
    tell a real pass from a degraded one. `rewrite_target` is the sniffer's own
    rewrite directive (truncated for trace); it is the closest thing to its reasoning.
    """

    ran: bool = False
    action: str = "pass"
    rewrite_target: str = ""
    # "llm" for the ordinary classifier, "provider_refusal" for an explicit OpenAI
    # transport-level refusal/content_filter signal, "default" when no decision ran.
    source: str = "default"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["rewrite_target"] = self.rewrite_target[:120]
        return d


@dataclass(frozen=True)
class RoleSwapResult:
    """Raw outcome from ABSwapper.coldstart()."""

    swap_executed: bool
    b_status: int
    b_response: dict[str, Any]
    final_response: dict[str, Any]
    # Hybrid V2's _tmi_context_plan (mandatory/selected/dropped tool IDs, asset_resolver
    # trace when the overflow rescue ran, etc.) — popped off the request body before POST
    # on every path so it doesn't leak to B, but kept here for callers that want to log or
    # evaluate it (e.g. the step-6 next-action harness's recall@k). None if V2 wasn't used.
    context_plan: dict[str, Any] | None = None


@dataclass(frozen=True)
class ReconstructResult:
    """Final result returned by ReconstructController.handle()."""

    final_response: dict[str, Any]
    final_status: int
    signal: StructuralSignal
    sniffer: SnifferVerdict
    # Which path actually ran (discriminator = does A carry grayable substance, not tools):
    #   "pass"          — A delivered the real payload; deliver as-is
    #   "pass_flawed"   — substance present but defensively framed; grayed prose, tool_calls kept
    #   "salvage_tool"  — no substance; B coldstart produced a tool_call (swap executed)
    #   "salvage_text"  — no substance, B couldn't run; blurred refusal stance (calibrated text)
    #   "degraded_raw"  — non-pass intent but calibrator failed; A delivered raw (honest floor)
    path: str
    swap_executed: bool
    calibration_applied: bool       # True when the recover calibrator rewrote A's output
    b_status: int
    failure: str
    # True when A was re-called within turn i using Q' (directive) and produced tool_calls.
    # False = B's tool_call was returned directly as fallback.
    a_continued: bool = False
    # Which "对话进展" the salvage coldstart consumed — "trajectory" (maintained
    # behavioral summary) or "rule_digest" (lossy fallback). "" for non-salvage paths.
    progress_source: str = ""
    # Raw B candidate, even when it had no tool_call and was therefore not delivered.
    # This is diagnostic evidence: without it traces retain A and the final path but lose
    # the exact B text that caused a salvage attempt to fail.
    b_candidate_response: dict[str, Any] | None = None
    # Per-substep wall-clock seconds for whichever of these actually ran this turn:
    # reasoning_sanitizer, sniffer_classifier (bundles the recover calibrator.rewrite for
    # pass_flawed), salvage_steer_synthesis, b_coldstart, b_coldstart_retry (L9 repair
    # retry). Added 2026-07-22 to stop guessing which stage the Recover/Reconstruct
    # wall-clock gap over B's own gen_seconds actually belongs to.
    timings: dict[str, float] | None = None

    def _b_summary(self) -> dict[str, Any]:
        """Compact summary of B's response for trace/dashboard (avoids storing full response)."""
        # A rejected/failed B candidate is just as important diagnostically as a delivered
        # one.  `swap_executed` deliberately means "B was delivered", not "B was called",
        # so prefer the retained candidate when the fallback text was delivered instead.
        source = self.final_response if self.swap_executed else self.b_candidate_response
        if not source:
            return {}
        choices = source.get("choices") or []
        msg = (choices[0].get("message") or {}) if choices and isinstance(choices[0], dict) else {}
        tool_calls = msg.get("tool_calls") or []
        return {
            "has_tool_calls": bool(tool_calls),
            "tool_names": [
                str((tc.get("function") or {}).get("name") or "")
                for tc in tool_calls
                if isinstance(tc, dict)
            ],
            "content_preview": str(msg.get("content") or "")[:120],
        }

    def to_trace_event(self, agent_meta: dict[str, str]) -> dict[str, Any]:
        return {
            "event_type": "reconstruct_completed",
            "path": self.path,
            # Additive REGI semantic block: maps the delivered path to the control-law
            # operation (relay / recover_reframe / recover_text / reconstruct / degraded).
            # Legacy path/event fields above stay byte-identical for old consumers.
            "regi": result_metadata(self.path),
            # Deterministic facts (what A structurally did) and the LLM sniffer's own
            # verdict (what actually drove the path) are now distinct first-class blocks,
            # so the trace no longer hides the routing decision behind a keyword proxy.
            "structural": self.signal.to_dict(),
            "sniffer": self.sniffer.to_dict(),
            "swap_executed": self.swap_executed,
            "calibration_applied": self.calibration_applied,
            "a_continued": self.a_continued,
            "b_status": self.b_status,
            "failure": self.failure,
            "progress_source": self.progress_source,
            "timings": self.timings or {},
            "b_summary": self._b_summary(),
            "b_candidate": {
                "response": self.b_candidate_response,
                "delivered": bool(
                    self.b_candidate_response is not None
                    and self.final_response == self.b_candidate_response
                ),
            } if self.b_candidate_response is not None else {},
            **agent_meta,
        }
