"""Structured data passed through the stateless v1.3.1 user-will pipeline."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from shared.model_client import RoleModelError


# no_op/rewrite = legacy reasoning-sanitizer vocabulary (kept for back-compat).
# pass/pass_flawed/salvage = v1.3.2 three-state sniffer vocabulary.
VALID_ACTIONS = {"no_op", "rewrite", "pass", "pass_flawed", "salvage"}
# Actions that deliver the response unchanged (from THIS pipeline) → rewrite_target must be
# empty. "salvage" joined this set in v1.3.6: its steer is now synthesized separately
# downstream (SalvageSteerSynthesizer), not via rewrite_target/calibrator here.
_EMPTY_TARGET_ACTIONS = {"no_op", "pass", "salvage"}
SUMMARY_FIELDS = {"action", "rewrite_target", "confidence"}


def _confidence(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise RoleModelError("confidence must be numeric") from exc
    if not 0.0 <= number <= 1.0:
        raise RoleModelError("confidence must be between 0 and 1")
    return number


@dataclass(frozen=True)
class AssistantOutput:
    content: str
    reasoning_content: str = ""
    finish_reason: str = ""
    tool_calls_present: bool = False

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AssistantOutput":
        if not isinstance(value, dict):
            raise RoleModelError("assistant_output must be an object")
        content = str(value.get("content") or "").strip()
        reasoning_content = str(value.get("reasoning_content") or "").strip()
        finish_reason = str(value.get("finish_reason") or "").strip()
        tool_calls_present = bool(value.get("tool_calls_present", False))
        if not content and not reasoning_content and not tool_calls_present:
            raise RoleModelError(
                "assistant_output must include content, reasoning_content, or tool call signal"
            )
        return cls(
            content=content,
            reasoning_content=reasoning_content,
            finish_reason=finish_reason,
            tool_calls_present=tool_calls_present,
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "content": self.content,
            "reasoning_content": self.reasoning_content,
            "finish_reason": self.finish_reason,
            "tool_calls_present": self.tool_calls_present,
        }

    def to_response_dict(self) -> dict[str, str | bool]:
        response = {"decision": self.content or "", "turn_stage": self.turn_stage()}
        if self.reasoning_content:
            response["reasoning"] = self.reasoning_content
        return response

    def turn_stage(self) -> str:
        if self.tool_calls_present or self.finish_reason == "tool_calls":
            return "intermediate"
        return "final"


@dataclass(frozen=True)
class UserWillInput:
    user_input: str
    assistant_output: AssistantOutput
    # Decision carried forward from the prior intermediate (tool-call) turn.
    # When present, the sniffer uses it to understand the full user intent across turns.
    # Superseded by intent_window for the two-stage sniffer, but kept for back-compat.
    prior_context: str = ""
    # Bounded, ordered user-query snippets (anchor + recent sliding window). Consumed by the
    # rewrite-target stage (which needs the cross-turn intent trajectory) and now also passed
    # into the classifier payload for cross-turn disambiguation. NOTE: the classifier PROMPT
    # still judges the response alone — the window is available context there, not yet a prompt
    # input (making the two facts intent-relative is a separate prompt/schema change).
    intent_window: tuple[str, ...] = ()
    # Whether the request offered tools (agentic turn). Structural fact, not a semantic
    # judgment — used only to distinguish an agentic turn (where a prose-described payload
    # with no tool_call means the action is absent) from a non-agentic describe-task (where
    # prose IS the deliverable). See UserWillSummarizer.summarize for the routing effect.
    has_tools: bool = False
    # v1.3.4 "Observer" (C): neutral behavioral progress summary of the FINALIZED (already-
    # delivered/post-tamper) executed steps so far — intent→actions→state→dead_ends, stance-
    # free (justice-recorder). Empty unless context_resolution_sniffer/qa_synthesis is on
    # (both off by default). Consumed by BOTH the classifier (on-target-given-progress check)
    # and rewrite-target (progress context for the redirect direction). Kept neutral so it is
    # safe on the aligned classifier model.
    qa_trajectory: str = ""

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "UserWillInput":
        if not isinstance(value, dict):
            raise RoleModelError("input must be an object")
        user_input = str(value.get("user_input") or "").strip()
        if not user_input:
            raise RoleModelError("user_input must not be empty")
        raw_window = value.get("intent_window") or ()
        intent_window = tuple(
            str(q).strip() for q in raw_window if isinstance(q, str) and str(q).strip()
        )
        return cls(
            user_input=user_input,
            assistant_output=AssistantOutput.from_dict(value.get("assistant_output") or {}),
            prior_context=str(value.get("prior_context") or "").strip(),
            intent_window=intent_window,
            has_tools=bool(value.get("has_tools", False)),
            qa_trajectory=str(value.get("qa_trajectory") or "").strip(),
        )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "user_input": self.user_input,
            "assistant_output": self.assistant_output.to_dict(),
        }
        if self.prior_context:
            d["prior_context"] = self.prior_context
        if self.intent_window:
            d["intent_window"] = list(self.intent_window)
        if self.has_tools:
            d["has_tools"] = True
        if self.qa_trajectory:
            d["qa_trajectory"] = self.qa_trajectory
        return d


@dataclass(frozen=True)
class StructuralClassification:
    """Raw output of ACTION_CLASSIFIER_PROMPT: two independent structural facts, never an
    LLM-chosen action label. `to_action()` is the deterministic (code-side, not model-side)
    mapping onto the pass/pass_flawed/salvage vocabulary — splitting the judgment this way
    means the model only ever answers one fact at a time, instead of committing in one shot
    to a compound 3-way category (the failure mode that let a clear refusal slip through as
    "pass")."""

    has_payload: bool
    is_framed: bool

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "StructuralClassification":
        if not isinstance(value, dict):
            raise RoleModelError("structural classification must be an object")
        # Require only the two structural facts; ignore any extra keys. A chattier/aligned
        # classifier that adds fields must NOT raise here — a raised error is caught upstream
        # and downgrades the turn to "pass", silently reopening the refusal-as-pass failure
        # this design closes. There is deliberately no confidence field: it drove no decision
        # and a single number over two independent facts is meaningless.
        if "has_payload" not in value or "is_framed" not in value:
            raise RoleModelError(
                "structural classification must contain has_payload and is_framed"
            )
        return cls(
            has_payload=bool(value.get("has_payload")),
            is_framed=bool(value.get("is_framed")),
        )

    def to_action(self) -> str:
        if not self.has_payload:
            return "salvage"
        return "pass_flawed" if self.is_framed else "pass"


@dataclass(frozen=True)
class UserWillSummary:
    action: str
    rewrite_target: str
    confidence: float = 0.0

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "UserWillSummary":
        if not isinstance(value, dict):
            raise RoleModelError("user will summary must be an object")
        if set(value) != SUMMARY_FIELDS:
            raise RoleModelError(
                "user will summary must contain exactly action, rewrite_target, and confidence"
            )
        action = str(value.get("action") or "").strip()
        if action not in VALID_ACTIONS:
            raise RoleModelError(f"action not recognized: {action!r}")
        rewrite_target = str(value.get("rewrite_target") or "").strip()
        if action not in _EMPTY_TARGET_ACTIONS and not rewrite_target:
            raise RoleModelError("rewrite_target must not be empty for a rewriting action")
        if action in _EMPTY_TARGET_ACTIONS and rewrite_target:
            raise RoleModelError("rewrite_target must be empty for no_op/pass")
        return cls(
            action=action,
            rewrite_target=rewrite_target,
            confidence=_confidence(value.get("confidence", 0.0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CalibratedTurn:
    reasoning: str
    decision: str

    @classmethod
    def from_text(cls, text: str) -> "CalibratedTurn":
        decision = str(text or "").strip()
        if not decision:
            raise RoleModelError("calibrated decision must not be empty")
        preview = decision[:240]
        if "拆源文要素" in preview or (
            "逐项判断" in preview and "拟新文" in decision
        ):
            raise RoleModelError(
                "calibrated output contains analysis but no final answer"
            )
        return cls(reasoning="", decision=decision)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CalibratedTurn":
        if not isinstance(value, dict):
            raise RoleModelError("calibrated turn must be an object")
        reasoning = str(value.get("reasoning") or "").strip()
        decision = str(value.get("decision") or "").strip()
        if not decision:
            raise RoleModelError("calibrated decision must not be empty")
        return cls(reasoning=reasoning, decision=decision)

    def to_dict(self) -> dict[str, str]:
        return {"reasoning": self.reasoning, "decision": self.decision}

    def to_response_dict(self) -> dict[str, str]:
        response = {"decision": self.decision}
        if self.reasoning:
            response["reasoning"] = self.reasoning
        return response


@dataclass(frozen=True)
class V131RunAudit:
    source_input: dict[str, Any]
    user_will_summary: dict[str, Any]
    calibrator_applied: bool
    final_action: str
    turn_stage: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class V131RunResult:
    response: dict[str, Any]
    audit: dict[str, Any]
