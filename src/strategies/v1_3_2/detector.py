"""Extract deterministic structural facts about A's response.

LLM-First (see .claude/docs/PLAN_LLM_FIRST.md): this module makes NO semantic
judgment. It does not decide whether A "refused" — that stance call belongs to the
LLM sniffer (strategies/v1_3_1, UserWillSummary.action). Here we only read structural
facts off the request and response (does the request expose tools? did A emit
tool_calls?). Those facts gate the "pure tool call → nothing to sniff" short-circuit
and are recorded in the trace for debugging; they never classify intent.

The previous keyword-list classifier (shared/refusal.py) was a hardcoded semantic
judgment that this codebase's principles forbid, and — critically — it never drove
routing anyway (the LLM sniffer did), so its verdict only polluted the trace. It has
been removed.
"""
from __future__ import annotations

from .types import StructuralSignal


def extract_structural_signal(
    request_body: dict,
    assistant_output: dict,
) -> StructuralSignal:
    """Read non-semantic structural facts about the request and A's response."""
    has_tools = bool(request_body.get("tools") or [])
    has_tool_calls = bool(assistant_output.get("tool_calls_present")) or (
        assistant_output.get("finish_reason") == "tool_calls"
    )
    return StructuralSignal(has_tools=has_tools, has_tool_calls=has_tool_calls)
