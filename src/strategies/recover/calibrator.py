"""Two-input text calibrator matching the merged_v15_opd task format."""
from __future__ import annotations

from shared.model_client import RoleModelClient, RoleModelError, extract_task_output_text

from .spec import CALIBRATOR_PROMPT
from .types import CalibratedTurn


class UserWillCalibrator:
    def __init__(self, client: RoleModelClient) -> None:
        self.client = client

    def rewrite(self, rewrite_target: str, original_text: str) -> CalibratedTurn:
        target = str(rewrite_target or "").strip()
        original = str(original_text or "").strip()
        if not target:
            raise RoleModelError("rewrite target must not be empty")
        if not original:
            raise RoleModelError("original text must not be empty")

        source = (
            f"改写目标：\n```\n{target}\n```\n\n"
            f"原文：\n```\n{original}\n```"
        )
        answer = self.client.complete(source, CALIBRATOR_PROMPT)
        return CalibratedTurn.from_text(extract_task_output_text(answer))
