"""REGI (Reachability-Gated Interposition) functional packages + compatibility namespaces.

The two subpackages are NOT mutually-exclusive software versions: they are the
functional layers of one REGI system, dispatched by the compatibility selector
(`TMI_VERSION` / `x-tmi-version`) in `proxy.message_forward`.

- `v1_3_1` — reachability sensing + Recover text rewriting (sniffer / calibration).
- `v1_3_2` — REGI orchestrator + Reconstruct engine + state/context augmentation
  (observer, trajectory, skills). Selector `v1.3.3` is a compatibility alias that
  normalizes to this engine; util/trajectory capabilities are config-driven
  (`TMI_UTIL_*` / `TMI_TRAJ_*`), never implied by the selector name.

They depend only on `shared`, never on each other or on `proxy` / `dashboard`.
The historical directory names are kept as a wire/compat contract.
"""
