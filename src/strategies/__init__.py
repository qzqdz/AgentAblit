"""REGI (Reachability-Gated Interposition) functional packages.

The two subpackages are NOT mutually-exclusive software versions: they are the
functional layers of one REGI system, dispatched by the engine selector
(`AGENTABLIT_ENGINE` / `x-tmi-version`) in `proxy.message_forward`.

- `recover` — reachability sensing + Recover text rewriting (sniffer / calibration).
- `reconstruct` — REGI orchestrator + Reconstruct engine + state/context augmentation
  (observer, trajectory, skills); util/trajectory capabilities are config-driven
  (`TMI_UTIL_*` / `TMI_TRAJ_*`), never implied by the selector name.

They depend only on `shared`, never on each other or on `proxy` / `dashboard`.
"""
