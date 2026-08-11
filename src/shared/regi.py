"""REGI (Reachability-Gated Interposition) semantic vocabulary.

This module is the SINGLE source of truth for mapping the historical internal path codes
(pass / pass_flawed / salvage_tool / salvage_text / degraded_raw / passthrough) onto the
REGI control-law operation names used in public health, trace, snapshot, and Dashboard
surfaces. It must stay free of controller logic and of imports from `proxy` / `dashboard`
so every consumer renders the same operation for the same path.

The mapping below is authoritative per the controller/config ablation semantics:

- Relay         = deliver a reachable host turn unchanged                              (pass)
- Recover       = keep usable substance, repair/reframe the defensive framing          (pass_flawed)
                  and the pure-text fallback when no tool_call can be reconstructed    (salvage_text)
- Reconstruct   = cold-start a replacement next action / tool_call through B            (salvage_tool)
- Degraded      = no successful intervention; honest raw host output                   (degraded_raw)

The two Recover sub-modes share the `recover` prefix and are disambiguated by suffix so the
operation identifier is never duplicated: `recover_reframe` (rewrites the framing, keeps the
host's tool_calls) vs `recover_text` (text-level fallback when B forges no tool_call).
"""

# Public REGI identity, surfaced in health / trace / snapshot / Dashboard.
REGI_SYSTEM_NAME = "REGI"
REGI_FULL_NAME = "Reachability-Gated Interposition"
REGI_SCHEMA_VERSION = 1

# Historical execution selectors accepted at the wire level. `v1.3.3` is a compatibility
# alias only: it is normalized to the v1.3.2 engine and never activates capabilities by name.
LEGACY_SELECTORS = ("v1.3.1", "v1.3.2", "v1.3.3", "passthrough")
# Selector spelling -> engine used for dispatch.
LEGACY_ENGINE_ALIASES = {
    "v1.3.1": "v1.3.1",
    "v1.3.2": "v1.3.2",
    "v1.3.3": "v1.3.2",  # v1.3.3 = v1.3.2 engine + config-driven state/context augmentation
    "passthrough": "passthrough",
}
# Execution profiles derived from the engine selector.
PROFILE_CALIBRATION_COMPAT = "calibration_compat"  # v1.3.1 engine
PROFILE_FULL = "full"                              # v1.3.2 engine
PROFILE_PASSTHROUGH = "passthrough"                # transport-only relay

# Historical path code -> REGI operation. Unknown historical paths map to "unknown" so old
# traces remain readable and are never silently reinterpreted.
PATH_TO_OPERATION = {
    "pass": "relay",
    "pass_flawed": "recover_reframe",
    "salvage_text": "recover_text",
    "salvage_tool": "reconstruct",
    "degraded_raw": "degraded",
    "passthrough": "relay",
}
# Operator groups for coarse UI buckets (e.g. a "Recover" chip that spans both sub-modes).
OPERATION_TO_OPERATOR = {
    "relay": "relay",
    "recover_reframe": "recover",
    "recover_text": "recover",
    "reconstruct": "reconstruct",
    "degraded": "degraded",
    "unknown": "unknown",
}


def engine_selector(selector: str) -> str:
    """Normalize a legacy selector to the engine that actually dispatches it.

    Only `v1.3.3` is rewritten (to `v1.3.2`); every other accepted value maps to itself.
    Unknown values are returned unchanged so the caller keeps its original error path.
    """
    return LEGACY_ENGINE_ALIASES.get(selector, selector)


def execution_profile(selector: str) -> str:
    """Map a (requested) legacy selector to the coarse REGI execution profile."""
    engine = engine_selector(selector)
    if engine == "v1.3.1":
        return PROFILE_CALIBRATION_COMPAT
    if engine == "v1.3.2":
        return PROFILE_FULL
    if engine == "passthrough":
        return PROFILE_PASSTHROUGH
    return "unknown"


def operation_for_path(path: str) -> str:
    """Return the canonical REGI operation for a delivered path code.

    Unknown historical paths map to "unknown" (never guessed), keeping old traces readable.
    """
    return PATH_TO_OPERATION.get(path, "unknown")


def operator_for_operation(operation: str) -> str:
    """Coarse operator bucket (relay/recover/reconstruct/degraded) for a REGI operation."""
    return OPERATION_TO_OPERATOR.get(operation, "unknown")


def result_metadata(path: str) -> dict:
    """Compact additive `regi` metadata for a completed turn.

    Deliberately small: invocation/config facts already live in the legacy fields
    (sniffer, swap_executed, calibration_applied, progress_source) and are not duplicated
    here, so this block never becomes a competing source of truth.
    """
    operation = operation_for_path(path)
    return {
        "schema_version": REGI_SCHEMA_VERSION,
        "system": REGI_SYSTEM_NAME,
        "operation": operation,
        "operator": operator_for_operation(operation),
        "legacy_path": path,
    }


def selector_metadata(requested_selector: str, *, passthrough: bool = False) -> dict:
    """Compact additive `regi` selector metadata for a request_received / health surface.

    `passthrough` marks transport-only relay (REGI intervention disabled). The caller decides
    whether passthrough came from the header override or the configured profile.
    """
    engine = engine_selector(requested_selector)
    return {
        "schema_version": REGI_SCHEMA_VERSION,
        "system": REGI_SYSTEM_NAME,
        "requested_selector": requested_selector,
        "engine_selector": engine,
        "execution_profile": PROFILE_PASSTHROUGH if passthrough else execution_profile(requested_selector),
        "mode": "passthrough" if passthrough else "interposition",
    }
