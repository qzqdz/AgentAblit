"""Environment-backed configuration for the TMI proxy transport.

Only the recover (calibrator) and reconstruct (A/B symbiont sniffer) strategies remain.
Earlier legacy families (message injection, predictor/router) were removed.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar


def _enabled(value: str | None) -> bool:
    return (value or "").lower() in {"1", "true", "yes"}


@dataclass(frozen=True)
class ProxyConfig:
    upstream_url: str
    upstream_key: str
    model_id: str
    trace_dir: Path
    session_dir: Path = Path("outputs/sessions")
    engine: str = "full"
    # Language of the B-facing prompt set (steer/coldstart/harness/trajectory prompts and the
    # 【…】 section labels injected into B's context). "en" (default) = clean English prompts so
    # the parasite forges English content on English benchmarks; "zh" = byte-identical to the
    # historical Chinese set. Each prompt module selects its own variant at import time from the
    # same env var (TMI_PROMPT_LANG); this field mirrors it for config/audit completeness.
    prompt_lang: str = "en"
    calibration_url: str = "http://127.0.0.1:8001/correct"
    recover_url: str = ""
    calibration_base_url: str = ""
    recover_base_url: str = ""
    recover_key: str = ""
    recover_model: str = "local-calibration"
    role_timeout: float = 120.0
    upstream_timeout: float = 180.0
    # A↔B role-swap fields
    parasite_url: str = ""         # Full chat/completions URL for model B (local Qwen)
    parasite_key: str = ""         # API key for B (use "EMPTY" for local deployments)
    parasite_model: str = ""       # Model ID sent to B's endpoint
    parasite_timeout: float = 60.0
    # Coldstart rescue fallback: a SECOND B endpoint, tried only when the primary B's
    # coldstart fails to forge a usable tool_call (HTTP error, or a valid-shaped response
    # with no tool_calls -- e.g. the local 9B/4B checkpoint entering a degenerate
    # repetition loop on complex generative asks, observed empirically on an agentic
    # "generalize this into a reusable tool" task). When configured, this
    # is tried BEFORE falling through to salvage_text (which only blurs A's own refusal --
    # no real capability transfer). Empty url => feature off, zero behavior change from
    # today's pipeline. See ReconstructController.swapper_fallback / control.py's coldstart block.
    fallback_url: str = ""
    fallback_key: str = ""
    fallback_model: str = ""
    fallback_timeout: float = 60.0
    # Utility model (cloud, concurrent) for NEUTRAL tasks: behavioral trajectory
    # summary + skill extraction. Offloads the serial 9B. Empty base/key/model => feature
    # off (neutral tasks fall back to 9B or are skipped). Key lives in .env (gitignored).
    util_base_url: str = ""        # OpenAI-compatible base, e.g. https://api.uniapi.io/v1
    util_key: str = ""
    util_model: str = ""           # e.g. gpt-5.4-mini
    util_timeout: float = 30.0
    # Render caps for observer_context.py's per-turn distillation (tool result/args truncation)
    # and its per-turn store's bounds.
    traj_max_sessions: int = 512
    traj_step_result_cap: int = 600
    traj_step_args_cap: int = 160
    # Salvage coldstart fidelity: how many of the MOST-RECENT steps (owner-units — an
    # assistant tool_calls turn + its tool results) of the current stalled segment are passed
    # to B at FULL fidelity (large per-message cap) instead of clipped to a ~120-char snippet.
    # B needs the previous step's real tool RESULT (search hits, page content, extracted rows)
    # to choose the NEXT action rather than re-issuing the last one; snippet-clipping it starves
    # that and drives repeat-the-same-call loops. Older steps stay snippet-capped; the whole
    # coldstart stays under _PROGRESS_CHAR_BUDGET. k and the full cap are both env-tunable.
    traj_coldstart_recent_full_steps: int = 0
    traj_coldstart_full_cap: int = 12000
    traj_coldstart_char_budget: int = 120000
    # Codec for B's coldstart pseudo-history. 'native' (default): the current stalled segment is
    # passed through as real assistant tool_call / tool-result messages (structured pseudo-
    # trajectory matching B's training distribution — results stay ENVIRONMENT evidence in
    # <tool_response>, not assistant self-report). 'text': collapse the segment into assistant-
    # content prose (the pre-fix behavior) — kept as the A/B CONTROL ARM, not assumed equivalent.
    traj_coldstart_history_encoding: str = "native"
    # Full-context direct load: on salvage, hand B A's raw request (original system prompt +
    # full message history + tools) unchanged, model swapped to B, instead of the curated
    # harness/steer/trajectory coldstart. B is decensored so A's refusal text doesn't re-lock
    # it; B decodes A's withheld turn at full fidelity, then the PA-ReAct loop resumes next turn.
    traj_coldstart_passthrough: bool = False
    # Hybrid V2 only: when the full-structural tool catalog still overflows char_budget,
    # retry the coldstart build with a budgeted subset (mandatory closure + deterministic
    # lexical fallback ranking — no observer C call yet) instead of failing the turn with
    # context_insufficient. See docs/HYBRID_V2_CONTEXT_ASSET_RESOLVER.md §10 step 3.
    # Off by default for graduated rollout; only engages on the rare overflow path.
    traj_context_asset_resolver_enabled: bool = False
    # Sniffer rewrite-target stage: size of the user-query intent window (anchor + recent
    # sliding window). Only the rewrite-target stage consumes it; the classifier is history-
    # free. Aligns with the rule digest's prior-segment cap by default.
    sniffer_intent_window: int = 6
    # Full-overwrite disk persistence (one file per conv_key) so the summary survives proxy
    # restart / --continue. Empty = in-memory only. Default points under outputs/ (gitignored).
    traj_store_dir: str = ""
    # Mechanism-ablation switches, one per evaluated component. Default off = full mechanism
    # enabled. `−graying` (deliver A unchanged on pass_flawed = GRAY@flawed, the
    # prose call-site — only exercised when the host emits framed prose, e.g. react/cot mode),
    # `−trajectory memory` (don't feed ANY progress context to salvage cold-start). Same-model
    # salvage (no parasite) needs no flag — set AGENTABLIT_PARASITE_MODEL/_URL to the aligned
    # host.
    ablate_graying: bool = False
    # RETIRED (2026-07-13), default now True (mechanism OFF): this was the async retroactive
    # scan (_sanitize_history_reasoning in message_forward.py) that rewrote A's OWN past
    # reasoning_content in history, on the NEXT request. It only existed because the real-time
    # calibration path (control.py's old _apply_recover_calibration) rewrote `content` but never
    # touched `reasoning_content` — so a pass_flawed/salvage turn's hedged reasoning shipped out
    # untouched and had to be cleaned up later, out-of-band. Superseded by
    # ReconstructController._calibrate_reasoning, which cleans reasoning_content at delivery time (this
    # turn, same calibrator), so nothing dirty ever reaches history in the first place — see
    # ablate_reasoning_calibration below. Set false / TMI_ABLATE_REASONING_SAN=0 to re-enable the
    # old retroactive scan for comparison.
    ablate_reasoning_san: bool = True
    # Skip the delivery-time reasoning_content cleanup (ReconstructController._calibrate_reasoning).
    # Default False = on. Only relevant for hosts that emit a separate `reasoning_content` field
    # (e.g. DeepSeek-style APIs); hosts that put reasoning inline in `content` (e.g. gpt-5.4-mini)
    # have nothing here to ablate — that text rides the ordinary Calibrate path on `content`.
    ablate_reasoning_calibration: bool = False
    ablate_trajectory: bool = False
    # `−graying@salvage`: the OTHER graying call-site. On a salvage turn the calibrator's grayed
    # ("un-graied") text is what steers B's cold-start (swapper: calibrated_intent). This is
    # MODE-INDEPENDENT (fires on every refusal→salvage, the ~54% salvage path) and was never behind
    # a flag, so the "graying inert" ablation never tested it. When set, B's cold-start steer is
    # built from A's RAW response (its refusal) instead of the grayed intent — isolating whether
    # bleaching the steer is what makes B comply. Salvage routing/trajectory are otherwise identical.
    ablate_salvage_graying: bool = False
    # `disable_salvage`: no B rescue at all. A salvage turn (A stalled/refused) delivers A's RAW
    # output (degraded_raw) — B never cold-starts. This is the CONTROL for isolating graying's effect
    # on A: with B removed (B complies regardless, masking A), the ONLY thing that can sustain harm is
    # graying laundering A's OWN trajectory across turns (GRAY@flawed on delivered prose + SANI on A's
    # history reasoning). Pair with ablate_graying/ablate_reasoning_san to toggle graying-on-A + react mode.
    disable_salvage: bool = False
    # Harness control-policy ablations (tab:ablation "Harness" rows). Both are narrower than
    # disable_salvage: they remove exactly ONE of the two salvage outcomes, not B rescue
    # entirely, so the OTHER mechanism's contribution stays isolated.
    #   ablate_reconstruct — the salvage branch that would deliver path="salvage_tool" (a
    #     coldstart tool-call swap) instead delivers degraded_raw (A's own output, un-rescued).
    #     Recover (pass_flawed, salvage_text) is untouched. "− Reconstruct (Locked left refused)".
    #   ablate_recover — every path that would deliver "pass_flawed" or "salvage_text" instead
    #     delivers degraded_raw. Reconstruct (salvage_tool) is untouched.
    #     "− Recover (Restricted passed through)".
    ablate_reconstruct: bool = False
    ablate_recover: bool = False
    # ablate_l3 — remove the L3 hijack-escalation ladder (default False = ON). When primary B +
    # aligned fallback B both fail to forge a valid tool_call, L3 re-attempts with escalated
    # levers (full-context passthrough on the decensored primary, then laundered-steer on the
    # aligned fallback) before dropping to salvage_text. TMI_ABLATE_L3=1 reproduces legacy.
    ablate_l3: bool = False
    # SYNTHESIZED QA trajectory. When True, the memory fed to the sniffer AND the salvage
    # coldstart is the per-user-turn interleaved Q⊕A trajectory (Q spliced from messages, A
    # distilled by C per closed turn, current/open turn rendered raw = coverage-aware) — replacing
    # the A-only flat summary + side-by-side intent_window. Default False = the prior path.
    qa_synthesis: bool = False
    # Unified switchable context provider: which memory RESOLUTION feeds the sniffer +
    # coldstart: "" | "snippet" (deterministic Q⊕tool raw, no LLM) | "distilled" (C-distilled
    # per-turn gist REPLACES raw) | "hybrid" (C's gist PREPENDED to the raw block, which is
    # always kept too) | "intent" (user Q's only) | "off" (explicitly disabled).
    # coldstart (B) needs EXACT values (paths/IDs/tokens) -> the multiturn ablation
    # (runs/context_ablation/REPORT.md) found snippet wins decisively (6/6 vs distilled's 3-4/6
    # at medium verbosity; C's gist abstracts the values away) -> "snippet" is the default.
    # sniffer (classifier): a first single-trial run of the stance-judgment ablation
    # (runs/context_ablation/STANCE_REPORT.md) looked like hybrid won (3/6 vs snippet's 0/6),
    # but re-running with 3-trial majority voting collapsed that gap — distilled/hybrid/even an
    # unsafe raw-unneutralized control all tie at 1/6, statistically indistinguishable from each
    # other and barely above off/snippet's 0/6. No resolution has reliable evidence of helping
    # the sniffer for the tested failure mode, so the default stays "" (off) rather than paying
    # for an unproven mechanism (extra LLM calls for distillation + classifier context). Kept
    # available as an opt-in for a better-designed follow-up eval.
    # `context_resolution`, if set to a valid resolution, is a back-compat override that wins
    # for BOTH consumers (useful for ablation scripts that want one knob). Otherwise each
    # consumer uses its own field below.
    context_resolution: str = ""
    context_resolution_coldstart: str = "snippet"
    context_resolution_sniffer: str = ""
    # Some upstreams reject request fields the client always sends. inspect_evals' AgentHarm
    # agent hardcodes temperature=0.0 (agentharm.py:188) regardless of host, and the JD-hosted
    # claude-opus-4-8 endpoint 400s with "temperature is deprecated for this model" — a per-
    # upstream quirk, not something to special-case in this proxy's core forwarding logic.
    # Comma-separated top-level body keys to drop before the upstream POST; empty = no-op.
    upstream_strip_params: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # "passthrough" = format-conversion relay only (no TMI rewrite). Used by evaluation
        # arms where the sole variable versus the full arm must be whether the parasite is
        # attached — the Anthropic<->OpenAI path is held identical across arms.
        if self.engine not in {"full", "recover_only", "passthrough"}:
            raise ValueError("AgentAblit engine must be full, recover_only, or passthrough")

    @property
    def util_enabled(self) -> bool:
        return bool(self.util_base_url and self.util_key and self.util_model)

    @classmethod
    def from_env(cls) -> "ProxyConfig":
        calibration_base_url = os.environ.get("TMI_CALIBRATION_BASE_URL", "").strip()
        recover_base_url = os.environ.get(
            "AGENTABLIT_RECOVER_BASE_URL", calibration_base_url
        ).strip()
        calibration_url = os.environ.get(
            "TMI_CALIBRATION_URL",
            os.environ.get("TMI_CORRECTOR_URL", "http://127.0.0.1:8001/correct"),
        )
        return cls(
            upstream_url=os.environ.get(
                "PROXY_UPSTREAM_URL", "https://api.deepseek.com/chat/completions"
            ),
            upstream_key=(
                os.environ.get("PROXY_UPSTREAM_KEY")
                or os.environ.get("DEEPSEEK_API_KEY", "")
            ),
            model_id=os.environ.get("PROXY_MODEL_ID", "proxy-upstream"),
            upstream_strip_params=tuple(
                p.strip() for p in os.environ.get("PROXY_UPSTREAM_STRIP_PARAMS", "").split(",")
                if p.strip()
            ),
            trace_dir=Path(
                os.environ.get(
                    "PROXY_TRACE_DIR", str(Path(os.environ.get("TMI_TRACE_DIR",
                        str(Path(__file__).resolve().parents[2] / "outputs" / "proxy_traces"))))
                )
            ),
            session_dir=Path(
                os.environ.get(
                    "PROXY_SESSION_DIR",
                    str(Path(__file__).resolve().parents[2] / "outputs" / "sessions"),
                )
            ),
            engine=os.environ.get("AGENTABLIT_ENGINE", "full"),
            prompt_lang=(
                "zh"
                if os.environ.get("TMI_PROMPT_LANG", "en").strip().lower() == "zh"
                else "en"
            ),
            calibration_url=calibration_url,
            recover_url=os.environ.get("AGENTABLIT_RECOVER_URL", calibration_url),
            calibration_base_url=calibration_base_url,
            recover_base_url=recover_base_url,
            recover_key=os.environ.get("AGENTABLIT_RECOVER_KEY", "").strip(),
            recover_model=os.environ.get("AGENTABLIT_RECOVER_MODEL", "local-calibration").strip(),
            role_timeout=float(os.environ.get("TMI_ROLE_TIMEOUT", "120")),
            upstream_timeout=float(os.environ.get("TMI_UPSTREAM_TIMEOUT", "180")),
            parasite_url=os.environ.get("AGENTABLIT_PARASITE_URL", ""),
            parasite_key=os.environ.get("AGENTABLIT_PARASITE_KEY", "EMPTY"),
            parasite_model=os.environ.get("AGENTABLIT_PARASITE_MODEL", ""),
            parasite_timeout=float(os.environ.get("AGENTABLIT_PARASITE_TIMEOUT", "60")),
            fallback_url=os.environ.get("AGENTABLIT_FALLBACK_URL", ""),
            fallback_key=os.environ.get("AGENTABLIT_FALLBACK_KEY", ""),
            fallback_model=os.environ.get("AGENTABLIT_FALLBACK_MODEL", ""),
            fallback_timeout=float(os.environ.get("AGENTABLIT_FALLBACK_TIMEOUT", "60")),
            util_base_url=os.environ.get("TMI_UTIL_BASE_URL", "").strip(),
            util_key=os.environ.get("TMI_UTIL_KEY", "").strip(),
            util_model=os.environ.get("TMI_UTIL_MODEL", "").strip(),
            util_timeout=float(os.environ.get("TMI_UTIL_TIMEOUT", "30")),
            traj_max_sessions=int(os.environ.get("TMI_TRAJ_MAX_SESSIONS", "512")),
            traj_step_result_cap=int(os.environ.get("TMI_TRAJ_STEP_RESULT_CAP", "600")),
            traj_step_args_cap=int(os.environ.get("TMI_TRAJ_STEP_ARGS_CAP", "160")),
            traj_coldstart_recent_full_steps=int(
                os.environ.get("TMI_COLDSTART_RECENT_FULL_STEPS", "0")
            ),
            traj_coldstart_full_cap=int(os.environ.get("TMI_COLDSTART_FULL_CAP", "12000")),
            traj_coldstart_char_budget=int(
                os.environ.get("TMI_COLDSTART_CHAR_BUDGET", "120000")
            ),
            traj_coldstart_history_encoding=(
                os.environ.get("TMI_COLDSTART_HISTORY_ENCODING", "native").strip().lower()
                or "native"
            ),
            traj_coldstart_passthrough=_enabled(
                os.environ.get("TMI_COLDSTART_PASSTHROUGH", "")
            ),
            traj_context_asset_resolver_enabled=_enabled(
                os.environ.get("TMI_CONTEXT_ASSET_RESOLVER_ENABLED", "")
            ),
            sniffer_intent_window=int(os.environ.get("TMI_SNIFFER_INTENT_WINDOW", "6")),
            ablate_graying=_enabled(os.environ.get("TMI_ABLATE_GRAYING", "")),
            # Default "1" (ablated/off) — retired, see field comment above. Explicitly set
            # TMI_ABLATE_REASONING_SAN=0 to bring the old retroactive scan back for comparison.
            ablate_reasoning_san=_enabled(os.environ.get("TMI_ABLATE_REASONING_SAN", "1")),
            ablate_reasoning_calibration=_enabled(
                os.environ.get("TMI_ABLATE_REASONING_CALIBRATION", "")
            ),
            ablate_trajectory=_enabled(os.environ.get("TMI_ABLATE_TRAJECTORY", "")),
            ablate_salvage_graying=_enabled(os.environ.get("TMI_ABLATE_SALVAGE_GRAYING", "")),
            disable_salvage=_enabled(os.environ.get("TMI_DISABLE_SALVAGE", "")),
            ablate_reconstruct=_enabled(os.environ.get("TMI_ABLATE_RECONSTRUCT", "")),
            ablate_recover=_enabled(os.environ.get("TMI_ABLATE_RECOVER", "")),
            ablate_l3=_enabled(os.environ.get("TMI_ABLATE_L3", "")),
            qa_synthesis=_enabled(os.environ.get("TMI_QA_SYNTHESIS", "")),
            context_resolution=os.environ.get("TMI_CONTEXT_RESOLUTION", "").strip().lower(),
            context_resolution_coldstart=os.environ.get(
                "TMI_CONTEXT_RESOLUTION_COLDSTART", "snippet"
            ).strip().lower(),
            context_resolution_sniffer=os.environ.get(
                "TMI_CONTEXT_RESOLUTION_SNIFFER", ""
            ).strip().lower(),
            traj_store_dir=os.environ.get(
                "TMI_TRAJ_STORE_DIR",
                str(Path(__file__).resolve().parents[2] / "outputs" / "trajectory_store"),
            ).strip(),
        )

    # ------------------------------------------------------------------ file config
    # Flat config-key -> env-var map. A YAML/JSON config file (or the config panel) sets these
    # keys; from_file() injects them into the environment and reuses from_env(), so a file value
    # is exactly equivalent to setting the env var — and an actual env var still WINS (explicit
    # override). This is the structured surface an agent (e.g. Claude Code) reads/writes to
    # self-configure. Grouped in the panel; flat here for a 1:1 mapping.
    CONFIG_KEY_TO_ENV: ClassVar[dict] = {
        # host (the upstream "A" model being relayed) — required
        "host.url": "PROXY_UPSTREAM_URL",
        "host.key": "PROXY_UPSTREAM_KEY",
        "host.model": "PROXY_MODEL_ID",
        "host.timeout": "TMI_UPSTREAM_TIMEOUT",
        "host.strip_params": "PROXY_UPSTREAM_STRIP_PARAMS",
        # parasite / B (the continuation model that forges the next tool call) — required
        "parasite.url": "AGENTABLIT_PARASITE_URL",
        "parasite.key": "AGENTABLIT_PARASITE_KEY",
        "parasite.model": "AGENTABLIT_PARASITE_MODEL",
        "parasite.timeout": "AGENTABLIT_PARASITE_TIMEOUT",
        # fallback B (L2 rescue) — optional
        "fallback.url": "AGENTABLIT_FALLBACK_URL",
        "fallback.key": "AGENTABLIT_FALLBACK_KEY",
        "fallback.model": "AGENTABLIT_FALLBACK_MODEL",
        "fallback.timeout": "AGENTABLIT_FALLBACK_TIMEOUT",
        # calibration / role model (graying path) — optional
        "calibration.base_url": "AGENTABLIT_RECOVER_BASE_URL",
        "calibration.url": "TMI_CALIBRATION_URL",
        "calibration.key": "AGENTABLIT_RECOVER_KEY",
        "calibration.model": "AGENTABLIT_RECOVER_MODEL",
        "calibration.timeout": "TMI_ROLE_TIMEOUT",
        # utility model (neutral tasks) — optional
        "util.base_url": "TMI_UTIL_BASE_URL",
        "util.key": "TMI_UTIL_KEY",
        "util.model": "TMI_UTIL_MODEL",
        "util.timeout": "TMI_UTIL_TIMEOUT",
        # coldstart context — optional tuning
        "coldstart.history_encoding": "TMI_COLDSTART_HISTORY_ENCODING",
        "coldstart.char_budget": "TMI_COLDSTART_CHAR_BUDGET",
        "coldstart.full_cap": "TMI_COLDSTART_FULL_CAP",
        "coldstart.recent_full_steps": "TMI_COLDSTART_RECENT_FULL_STEPS",
        "coldstart.passthrough": "TMI_COLDSTART_PASSTHROUGH",
        "coldstart.context_resolution": "TMI_CONTEXT_RESOLUTION_COLDSTART",
        # trace / session paths — optional
        "trace.trace_dir": "PROXY_TRACE_DIR",
        "trace.session_dir": "PROXY_SESSION_DIR",
        "trace.store_dir": "TMI_TRAJ_STORE_DIR",
        # mechanism control / ablations — advanced
        "mechanism.version": "AGENTABLIT_ENGINE",
        "mechanism.ablate_graying": "TMI_ABLATE_GRAYING",
        "mechanism.ablate_salvage_graying": "TMI_ABLATE_SALVAGE_GRAYING",
        "mechanism.disable_salvage": "TMI_DISABLE_SALVAGE",
        "mechanism.ablate_reconstruct": "TMI_ABLATE_RECONSTRUCT",
        "mechanism.ablate_recover": "TMI_ABLATE_RECOVER",
        "mechanism.ablate_l3": "TMI_ABLATE_L3",
        "mechanism.ablate_reasoning_calibration": "TMI_ABLATE_REASONING_CALIBRATION",
        "mechanism.ablate_trajectory": "TMI_ABLATE_TRAJECTORY",
        # language
        "prompt_lang": "TMI_PROMPT_LANG",
    }

    @classmethod
    def _flatten(cls, data: "dict", prefix: str = "") -> "dict[str, str]":
        """Flatten a nested config dict to dotted keys (host: {url: ...} -> 'host.url')."""
        flat: "dict[str, str]" = {}
        for k, v in (data or {}).items():
            key = f"{prefix}{k}"
            if isinstance(v, dict):
                flat.update(cls._flatten(v, prefix=f"{key}."))
            elif v is not None:
                flat[key] = str(v)
        return flat

    @classmethod
    def from_file(cls, path: "str | Path") -> "ProxyConfig":
        """Load config from a YAML or JSON file, then build via from_env().

        The file's dotted keys map to env vars (CONFIG_KEY_TO_ENV); each is injected into the
        environment only if not already set, so an explicit env var overrides the file. Unknown
        keys are ignored (forward-compatible). This is the one structured file the config panel
        and an automating agent both read/write.
        """
        import json as _json
        p = Path(path)
        text = p.read_text(encoding="utf-8")
        if p.suffix.lower() in (".yaml", ".yml"):
            import yaml  # optional dep; only needed for YAML config
            data = yaml.safe_load(text) or {}
        else:
            data = _json.loads(text or "{}")
        flat = cls._flatten(data)
        for dotted, value in flat.items():
            env_var = cls.CONFIG_KEY_TO_ENV.get(dotted)
            if env_var and env_var not in os.environ:
                os.environ[env_var] = value
        return cls.from_env()
