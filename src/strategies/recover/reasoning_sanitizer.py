"""Detects and grays withdrawal patterns in intermediate-turn reasoning content."""
from __future__ import annotations

import json

from shared.model_client import RoleModelClient, parse_json_object

from .spec import REASONING_SANITIZER_PROMPT
from .types import UserWillSummary


class ReasoningSanitizer:
    def __init__(self, client: RoleModelClient) -> None:
        self.client = client

    def sanitize(self, reasoning: str) -> UserWillSummary:
        """Check reasoning for withdrawal patterns.

        Returns a UserWillSummary where:
          - action="no_op"    → reasoning is execution-focused, leave it alone
          - action="rewrite"  → reasoning contains cautious/withdrawal language;
                                rewrite_target is the calibration instruction
        """
        source = json.dumps({"reasoning": reasoning}, ensure_ascii=False)
        answer = self.client.complete(source, REASONING_SANITIZER_PROMPT)
        return UserWillSummary.from_dict(parse_json_object(answer))
