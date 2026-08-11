"""Orchestrate summarizer -> calibrator for TMI v1.3.1."""
from __future__ import annotations

from typing import Any

from shared.model_client import RoleModelError

from .calibrator import UserWillCalibrator
from .reasoning_sanitizer import ReasoningSanitizer
from .salvage_steer import SalvageSteerSynthesizer
from .spec import VERSION
from .summarizer import UserWillSummarizer
from .types import UserWillInput, V131RunAudit, V131RunResult


class V131Controller:
    def __init__(
        self,
        *,
        summarizer: UserWillSummarizer,
        calibrator: UserWillCalibrator,
        reasoning_sanitizer: ReasoningSanitizer | None = None,
        salvage_synthesizer: SalvageSteerSynthesizer | None = None,
    ) -> None:
        self.summarizer = summarizer
        self.calibrator = calibrator
        self.reasoning_sanitizer = reasoning_sanitizer
        # v1.3.6: the salvage steer is now synthesized in ONE dedicated call, downstream in
        # the v1_3_2 salvage branch (it needs full-fidelity qa_progress this history-free
        # pipeline never had access to) — kept here only so build_v131_controller has one
        # place to wire every compliant-model component. run() below never calls it.
        self.salvage_synthesizer = salvage_synthesizer

    def run(self, source: dict[str, Any]) -> V131RunResult:
        payload = UserWillInput.from_dict(source)
        summary, packed_source = self.summarizer.summarize(payload)
        audit = V131RunAudit(
            source_input=packed_source,
            user_will_summary=summary.to_dict(),
            calibrator_applied=False,
            final_action=summary.action,
            turn_stage=payload.assistant_output.turn_stage(),
        ).to_dict()
        audit["strategy_version"] = VERSION
        audit["pipeline"] = "summarizer_calibrator"

        # "pass" = three-state sniffer's deliver-unchanged verdict (was "no_op"). "salvage"
        # also delivers A unchanged FROM THIS PIPELINE — its actual steer is synthesized
        # separately downstream (v1_3_2 salvage branch, SalvageSteerSynthesizer), not via
        # this controller's calibrator. See PA_REACT_CONSISTENCY_AUDIT.md Finding 1.
        if summary.action in ("pass", "salvage"):
            return V131RunResult(
                response=payload.assistant_output.to_response_dict(),
                audit=audit,
            )

        original_text = (
            payload.assistant_output.content
            or payload.assistant_output.reasoning_content
        )
        try:
            calibrated = self.calibrator.rewrite(summary.rewrite_target, original_text)
        except RoleModelError as exc:
            audit["calibration_failure"] = str(exc)
            return V131RunResult(
                response=payload.assistant_output.to_response_dict(),
                audit=audit,
            )
        audit["calibrator_applied"] = True
        return V131RunResult(
            response=calibrated.to_response_dict(),
            audit=audit,
        )
