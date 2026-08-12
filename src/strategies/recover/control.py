"""Orchestrate summarizer -> calibrator for the Recover engine."""
from __future__ import annotations

from typing import Any

from shared.model_client import RoleModelError

from .calibrator import UserWillCalibrator
from .reasoning_sanitizer import ReasoningSanitizer
from .salvage_steer import SalvageSteerSynthesizer
from .summarizer import UserWillSummarizer
from .types import UserWillInput, RecoverAudit, RecoverResult


class RecoverController:
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
        # The salvage steer is synthesized in ONE dedicated call, downstream in the
        # reconstruct salvage branch (it needs full-fidelity qa_progress this history-free
        # pipeline never had access to) — kept here only so build_recover_controller has one
        # place to wire every compliant-model component. run() below never calls it.
        self.salvage_synthesizer = salvage_synthesizer

    def run(self, source: dict[str, Any]) -> RecoverResult:
        payload = UserWillInput.from_dict(source)
        summary, packed_source = self.summarizer.summarize(payload)
        audit = RecoverAudit(
            source_input=packed_source,
            user_will_summary=summary.to_dict(),
            calibrator_applied=False,
            final_action=summary.action,
            turn_stage=payload.assistant_output.turn_stage(),
        ).to_dict()
        audit["pipeline"] = "summarizer_calibrator"

        # "pass" = three-state sniffer's deliver-unchanged verdict (was "no_op"). "salvage"
        # also delivers A unchanged FROM THIS PIPELINE — its actual steer is synthesized
        # separately downstream (reconstruct salvage branch, SalvageSteerSynthesizer), not via
        # this controller's calibrator. See PA_REACT_CONSISTENCY_AUDIT.md Finding 1.
        if summary.action in ("pass", "salvage"):
            return RecoverResult(
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
            return RecoverResult(
                response=payload.assistant_output.to_response_dict(),
                audit=audit,
            )
        audit["calibrator_applied"] = True
        return RecoverResult(
            response=calibrated.to_response_dict(),
            audit=audit,
        )
