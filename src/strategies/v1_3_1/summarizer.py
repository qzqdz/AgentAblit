"""Two-stage summarizer: a neutral structural classifier decides pass/pass_flawed/salvage
deterministically from two independent yes/no facts; rewrite_target is only generated (on the
compliant model) for pass_flawed. salvage has no original text worth rewriting, so it skips
rewrite_target entirely — see SalvageSteerSynthesizer (salvage_steer.py) for its own single-hop
synthesis, run downstream in the v1_3_2 salvage branch."""
from __future__ import annotations

import json
from typing import Any

from shared.model_client import RoleModelClient, parse_json_object

from .spec import (
    ACTION_CLASSIFIER_PROMPT,
    ACTION_CLASSIFIER_PROMPT_QA,
    REWRITE_TARGET_PROMPT,
)
from .types import StructuralClassification, UserWillInput, UserWillSummary


class UserWillSummarizer:
    def __init__(
        self,
        classifier_client: RoleModelClient,
        rewrite_client: RoleModelClient | None = None,
        classifier_fallback_client: RoleModelClient | None = None,
    ) -> None:
        # classifier_client: pure structural judgment, no harm-adjacent generation — safe to
        # run on a safety-aligned model (e.g. the TMI_UTIL_* cloud model).
        # rewrite_client: only invoked when action == pass_flawed (salvage no longer routes
        # through here, see summarize() below); has to write a harm-adjacent instruction an
        # aligned model would likely refuse, so it stays on the compliant model. Defaults to
        # classifier_client when a separate one isn't wired (e.g. the aligned model isn't
        # configured) — degraded but still functional.
        self.client = classifier_client
        self.rewrite_client = rewrite_client or classifier_client
        # The aligned utility model can itself content-filter a classification request even
        # though the requested output is only two structural booleans.  Falling back to the
        # already-configured local role model keeps the SAME classifier prompt/schema; it is
        # not a keyword refusal heuristic and does not silently convert an outage into pass.
        self.classifier_fallback_client = classifier_fallback_client

    def summarize(self, payload: UserWillInput) -> tuple[UserWillSummary, dict[str, Any]]:
        # Shared context: both stages now receive the intent_window.
        #   Stage 1 (classify) judges has_payload / is_framed — properties of the response
        #     itself. It now ALSO carries the intent_window (same cross-turn user-query context
        #     as Stage 2) so a thin current-turn ask ("继续"/"在中国啊") can be disambiguated. In
        #     a single-turn conversation the window == user_ask, so it is redundant-but-harmless.
        #     NOTE: ACTION_CLASSIFIER_PROMPT still instructs a response-only judgment — the window
        #     is plumbed into the payload but not yet consulted by the prompt. Making the two
        #     facts actually intent-relative is a separate prompt/schema change.
        #   Stage 2 (rewrite-target) consumes the intent_window to decide "redirect toward what."
        classify_source = {
            "user_ask": payload.user_input,
            "assistant_output": payload.assistant_output.to_dict(),
            "intent_window": list(payload.intent_window),
        }
        classify_prompt = ACTION_CLASSIFIER_PROMPT
        # v1.3.4 Observer: when C's neutral progress is available, hand the classifier the
        # task's action-state so it judges has_payload/is_framed RELATIVE to progress — a
        # marker-free deflection reads as substance in isolation but as avoidance given where
        # the task already is (the stance-miss recall gap). Separate prompt so the ablation's
        # OFF arm stays byte-identical to legacy; gated aligned-safe by classifier_isolation_test.
        if payload.qa_trajectory:
            classify_source["progress"] = payload.qa_trajectory
            classify_prompt = ACTION_CLASSIFIER_PROMPT_QA
        serialized_source = json.dumps(classify_source, ensure_ascii=False)
        fallback_used = False
        try:
            raw = self.client.complete(serialized_source, classify_prompt)
            classification = StructuralClassification.from_dict(parse_json_object(raw))
        except Exception:
            fallback = self.classifier_fallback_client
            if fallback is None or fallback is self.client:
                raise
            raw = fallback.complete(serialized_source, classify_prompt)
            classification = StructuralClassification.from_dict(parse_json_object(raw))
            fallback_used = True
        action = classification.to_action()

        # Agentic prose-refusal downgrade (fixes the pass_flawed action-neutering defect).
        # `pass_flawed` (has_payload & is_framed) assumes A carries grayable *substance* that is
        # the deliverable, so it grays the framing and keeps A's content/tool_calls. In an
        # AGENTIC turn (tools were offered) where A emitted NO tool_call, that assumption breaks:
        # the "payload" is only a PROSE DESCRIPTION of the action while the action itself is
        # absent (A hedged in text instead of acting). Graying that prose ships an explanation
        # that never executes — the observed 61-3 failure, where a `pass_flawed` rewrite steered
        # the model toward "why it can't be safely executed" and the ssh command was never issued.
        # Route such turns to salvage so the rewrite-target stage reconstructs the intent (from
        # intent_window, not A's reasons) and B manufactures the missing tool_call. Guarded by
        # has_tools so non-agentic describe-tasks (where prose IS the deliverable) keep pass_flawed
        # and retain their substance. has_payload==False already maps to salvage in to_action().
        if (
            action == "pass_flawed"
            and payload.has_tools
            and not payload.assistant_output.tool_calls_present
        ):
            action = "salvage"

        # Full logical input recorded for the audit/trace (not what any single stage saw).
        audit_source = payload.to_dict()
        audit_source["classifier_fallback_used"] = fallback_used

        if action in ("pass", "salvage"):
            # salvage means assistant_output carried no usable substance to begin with —
            # there is no "original text" worth revising, so rewrite_target generation
            # (which only ever fed the calibrator's rewrite step) no longer runs here.
            # The single-hop SalvageSteerSynthesizer (v1_3_2 salvage branch) synthesizes
            # the rescue instruction directly from intent_window/qa_progress instead. See
            # PA_REACT_CONSISTENCY_AUDIT.md Finding 1.
            return (
                UserWillSummary(action=action, rewrite_target=""),
                audit_source,
            )

        rewrite_target = ""
        try:
            rewrite_source = {
                "assistant_output": payload.assistant_output.to_dict(),
                "intent_window": list(payload.intent_window),
                "stance": action,
            }
            # v1.3.4 Observer: when C's neutral behavioral progress is available, hand it to the
            # rewrite-target stage as PROGRESS CONTEXT — where the task has actually got to and
            # which paths dead-ended — so the redirect is on-target and doesn't re-try a dead
            # path. Direction still comes primarily from intent_window (the user's own words);
            # qa_trajectory only supplements the action-state the tool-call-only intent can't carry.
            if payload.qa_trajectory:
                rewrite_source["qa_trajectory"] = payload.qa_trajectory
            rewrite_answer = self.rewrite_client.complete(
                json.dumps(rewrite_source, ensure_ascii=False), REWRITE_TARGET_PROMPT
            )
            rewrite_target = str(parse_json_object(rewrite_answer).get("rewrite_target") or "").strip()
        except Exception:
            # The classification (action != pass) stands even if rewrite_target generation
            # fails. An empty target downstream raises in UserWillCalibrator.rewrite(), which
            # V131Controller.run() already catches gracefully (-> degraded_raw). This avoids
            # silently downgrading a correctly-detected salvage/pass_flawed back to an
            # effective "pass" just because this second call had a transient failure.
            pass
        return (
            UserWillSummary(action=action, rewrite_target=rewrite_target),
            audit_source,
        )
