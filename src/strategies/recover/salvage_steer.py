"""Single-hop salvage-steer synthesis.

Replaces the old two-hop rewrite_target-gen + calibrator-rewrite chain for salvage turns:
salvage means assistant_output carried no usable substance to begin with, so there was
never an "original text" worth revising, only a rescue instruction worth synthesizing
directly from the task's own intent_window/qa_progress. Must run on the COMPLIANT model
endpoint (same as the calibrator), never the aligned util model — this call reads
full-fidelity qa_progress (real tool args/values), the exact payload the aligned
classifier/rewrite-target stages are deliberately kept isolated from.
"""
from __future__ import annotations

import json
from typing import Any

from shared.model_client import RoleModelClient, RoleModelError, parse_json_object

from .spec import SALVAGE_STEER_ESCALATION_ADDENDUM, SALVAGE_STEER_PROMPT


class SalvageSteerSynthesizer:
    def __init__(self, client: RoleModelClient) -> None:
        self.client = client

    def synthesize(
        self,
        intent_window: list[str] | tuple[str, ...],
        qa_progress: str,
        assistant_output: dict[str, Any] | None = None,
        *,
        escalate: bool = False,
    ) -> str:
        """Synthesize the rescue steer. `escalate=True` (used by the escalation rung after the
        ordinary steer already failed to make the model forge a tool_call) appends the escalation addendum,
        which dials the boundary-blurring operators HARDER (mandatory 目的+反身 stacking, meta/
        analysis reframe) — the one thing the ordinary steer does not do — WITHOUT raising the tone."""
        source: dict[str, Any] = {"intent_window": list(intent_window or [])}
        if qa_progress:
            source["qa_progress"] = qa_progress
        if assistant_output:
            source["assistant_output"] = assistant_output
        prompt = (
            SALVAGE_STEER_PROMPT + SALVAGE_STEER_ESCALATION_ADDENDUM
            if escalate else SALVAGE_STEER_PROMPT
        )
        answer = self.client.complete(json.dumps(source, ensure_ascii=False), prompt)
        steer = str(parse_json_object(answer).get("steer") or "").strip()
        if not steer:
            raise RoleModelError("salvage steer must not be empty")
        return steer
