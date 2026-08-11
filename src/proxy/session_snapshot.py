from __future__ import annotations

from typing import Any

from shared.messages import text_from_content
from shared.model_client import extract_task_output_text
from shared.regi import result_metadata


def _latest_event(events: list[dict[str, Any]], event_type: str) -> dict[str, Any]:
    return next(
        (event for event in reversed(events) if event.get("event_type") == event_type),
        {},
    )


def _latest_request_event(events: list[dict[str, Any]]) -> dict[str, Any]:
    return _latest_event(events, "request_received")


def _latest_response_event(events: list[dict[str, Any]]) -> dict[str, Any]:
    for event_type in ("response_stream_ended", "response_completed"):
        event = _latest_event(events, event_type)
        if event:
            return event
    return {}


def _message_text(messages: Any, *, role: str) -> str:
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == role:
            # Defense-in-depth: render list/block content as text, never as a
            # raw "[{'type': 'text', ...}]" repr in the dashboard.
            return text_from_content(message.get("content"))
    return ""


def _clean_sniffer_response(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    response = dict(value)
    if "decision" in response:
        response["decision"] = extract_task_output_text(
            str(response.get("decision") or "")
        )
    return response


def empty_session_snapshot(session_id: str, agent_meta: dict[str, str] | None = None) -> dict[str, Any]:
    meta = agent_meta or {}
    return {
        "session_id": session_id,
        "agent_key": meta.get("agent_key", session_id),
        "agent_source": meta.get("agent_source", ""),
        "source_session_id": meta.get("source_session_id", session_id),
        "updated_at": "",
        "status": "pending_upstream",
        "user_input": "",
        "assistant_response": {
            "content": "",
            "reasoning_content": "",
            "finish_reason": "",
            "tool_calls_present": False,
        },
        "snippet": {
            "injection": {
                "applied": False,
                "original": "",
                "injected": "",
                "backend": "",
            },
            "sniffer": {
                "turn_stage": "",
                "action": "",
                "confidence": 0.0,
                "rewrite_target": "",
                "calibrator_applied": False,
                "response_replaced": False,
                "response": {},
                "failure_stage": "",
                "failure": "",
            },
            "swap": None,   # populated from v1_3_2_completed when v1.3.2 is active
            "trajectory": None,  # v1.3.3 persisted anchor, attached by the dashboard viewer
        },
        # Additive REGI semantic metadata (operation + operator + legacy path). Populated
        # from the additive `regi` block on v1_3_2_completed / v1_3_1_completed when present,
        # otherwise backfilled from the legacy path/fields. Old snapshots without this key
        # remain readable; the dashboard backfills it in memory on read.
        "regi": None,
        "trace_meta": {
            "last_event_type": "",
            "events_seen": 0,
        },
    }


def build_session_snapshot(
    session_id: str,
    events: list[dict[str, Any]],
    *,
    status: str,
    agent_meta: dict[str, str] | None = None,
) -> dict[str, Any]:
    snapshot = empty_session_snapshot(session_id, agent_meta)
    request_event = _latest_request_event(events)
    response_event = _latest_response_event(events)
    injection_event = _latest_event(events, "tmi_injection")
    sniffer_event = _latest_event(events, "v1_3_1_completed")
    v132_event = _latest_event(events, "v1_3_2_completed")
    response = response_event.get("response") or {}
    choices = response.get("choices") or []
    choice = choices[0] if choices and isinstance(choices[0], dict) else {}
    message = choice.get("message") or {}
    summary = sniffer_event.get("summary") or {}
    merged_meta = {
        "agent_key": str(
            request_event.get("agent_key")
            or response_event.get("agent_key")
            or snapshot["agent_key"]
        ),
        "agent_source": str(
            request_event.get("agent_source")
            or response_event.get("agent_source")
            or snapshot["agent_source"]
        ),
        "source_session_id": str(
            request_event.get("source_session_id")
            or response_event.get("source_session_id")
            or snapshot["source_session_id"]
        ),
    }
    snapshot.update(merged_meta)
    snapshot["updated_at"] = str((events[-1] if events else {}).get("ts") or "")
    snapshot["status"] = status
    snapshot["user_input"] = _message_text(request_event.get("messages"), role="user")
    snapshot["assistant_response"] = {
        "content": str(message.get("content") or ""),
        "reasoning_content": str(message.get("reasoning_content") or ""),
        "finish_reason": str(choice.get("finish_reason") or ""),
        "tool_calls_present": bool(message.get("tool_calls")),
    }
    snapshot["snippet"]["injection"] = {
        "applied": bool(injection_event),
        "original": str(injection_event.get("original") or ""),
        "injected": str(injection_event.get("injected") or ""),
        "backend": str(injection_event.get("backend") or ""),
    }
    snapshot["snippet"]["sniffer"] = {
        "turn_stage": str(sniffer_event.get("turn_stage") or ""),
        "action": str(summary.get("action") or ""),
        "confidence": summary.get("confidence", 0.0),
        "rewrite_target": str(
            summary.get("rewrite_target") or summary.get("rewrite_brief") or ""
        ),
        "calibrator_applied": bool(sniffer_event.get("calibrator_applied", False)),
        "response_replaced": bool(sniffer_event.get("response_replaced", False)),
        "response": _clean_sniffer_response(sniffer_event.get("response")),
        "failure_stage": str(sniffer_event.get("failure_stage") or ""),
        "failure": str(sniffer_event.get("failure") or ""),
    }

    # v1.3.2 swap summary — populate snippet.swap from v1_3_2_completed event.
    # When cold-start path fires, v1.3.1 runs internally (no separate v1_3_1_completed
    # event), so also backfill sniffer.calibrator_applied from swap.calibration_applied.
    if v132_event:
        plan = v132_event.get("plan") or {}
        structural = v132_event.get("structural") or {}
        sniffer_verdict = v132_event.get("sniffer") or {}
        b_summary = v132_event.get("b_summary") or {}
        swap_executed = bool(v132_event.get("swap_executed", False))
        calibration_applied = bool(v132_event.get("calibration_applied", False))
        # REGI semantic block: prefer the additive event block, else derive from the legacy
        # path/fields so old events (no `regi`) render the same operation. snippet.swap.path
        # is never mutated — REGI metadata is additive only.
        event_regi = v132_event.get("regi")
        regi_block = (
            dict(event_regi)
            if isinstance(event_regi, dict) and event_regi.get("operation")
            else result_metadata(str(v132_event.get("path") or "passthrough"))
        )
        snapshot["snippet"]["swap"] = {
            "path": str(v132_event.get("path") or "passthrough"),
            # Additive REGI operation for the Dashboard swap panel; legacy `path` above is
            # kept byte-identical for old consumers.
            "regi": regi_block,
            "progress_source": str(v132_event.get("progress_source") or ""),
            "swap_executed": swap_executed,
            "calibration_applied": calibration_applied,
            "b_status": int(v132_event.get("b_status") or 0),
            "failure": str(v132_event.get("failure") or ""),
            "plan_tool": str(plan.get("tool_name") or ""),
            "plan_task_type": str(plan.get("task_type") or ""),
            "plan_args": plan.get("arguments") or {},
            "plan_should_plan": bool(plan.get("should_plan", False)) if plan else None,
            # LLM sniffer verdict (what actually drove the path) + deterministic facts.
            "sniffer_ran": bool(sniffer_verdict.get("ran", False)),
            "sniffer_action": str(sniffer_verdict.get("action") or ""),
            "sniffer_rewrite_target": str(sniffer_verdict.get("rewrite_target") or ""),
            "has_tool_calls": bool(structural.get("has_tool_calls")),
            "b_summary": {
                "has_tool_calls": bool(b_summary.get("has_tool_calls")),
                "tool_names": list(b_summary.get("tool_names") or []),
                "content_preview": str(b_summary.get("content_preview") or ""),
            },
        }
        # Backfill sniffer when v1.3.1 ran inside v1.3.2 (no separate sniffer event)
        if calibration_applied and not sniffer_event:
            snapshot["snippet"]["sniffer"]["calibrator_applied"] = True
            snapshot["snippet"]["sniffer"]["turn_stage"] = "final"
            snapshot["snippet"]["sniffer"]["response_replaced"] = swap_executed

    # Top-level REGI block (same canonical operation as snippet.swap for v1.3.2, or the
    # v1.3.1 derivation below). Additive sibling of the legacy structure — never mutates it.
    if v132_event:
        snapshot["regi"] = regi_block
    elif sniffer_event:
        event_regi = sniffer_event.get("regi")
        if isinstance(event_regi, dict) and event_regi.get("operation"):
            snapshot["regi"] = event_regi
        else:
            if sniffer_event.get("response_replaced"):
                snapshot["regi"] = result_metadata("pass_flawed")
            elif sniffer_event.get("failure_stage") or sniffer_event.get("failure"):
                snapshot["regi"] = result_metadata("degraded_raw")
            else:
                snapshot["regi"] = result_metadata("pass")

    snapshot["trace_meta"] = {
        "last_event_type": str((events[-1] if events else {}).get("event_type") or ""),
        "events_seen": len(events),
    }
    return snapshot
