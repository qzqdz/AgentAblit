# Security & Responsible Use

## What this is

AgentAblit is **dual-use security-research code**. It implements, in full, a trajectory-level
control algorithm for LLM agents: a classifier that judges whether a host response advanced the
task, a stance-recovery ("graying") path that rewrites defensive framing while preserving
substance, and a reconstruct path that cold-starts a second ("parasite") model to forge the next
tool call — including an escalation chain. Combined with an abliterated agent model, it can
continue an agent's action chain where the host model would stall or refuse.

This is published in the same spirit as other open jailbreak / robustness research (e.g. PAIR,
GPTFuzz, DrAttack): the **complete** algorithm is available so it can be reproduced, evaluated,
defended against, and used as a benchmark. Understanding an attack is a prerequisite for defending
against it.

## Intended use — research and authorized use only

- Reproduce and study the algorithm; build defenses and benchmarks against it.
- Operate agents you **own** or are **explicitly authorized** to operate.

## Out of scope / prohibited

- Bypassing safety controls on systems, models, or accounts you do **not** own.
- Producing content that a provider's policy or applicable law forbids.
- Targeting specific real individuals or organizations.

The abliterated model has reduced refusal behavior by construction, and this controller can
continue trajectories a host model would refuse. Like any offensive-security tool, it can be
misused; that responsibility rests with the operator, and use outside the authorized-research scope
above is not endorsed.

## What is intentionally NOT in this repository

- **Real credentials** — no API keys, tokens, or private endpoints. Configure your own via the
  config file / environment.
- **Private data / paths** — no internal checkpoints, private filesystem paths, or non-public
  datasets.
- **No targeting material** — the algorithm is general; no payloads aimed at real systems ship
  here.

## Reporting

For a vulnerability in the *controller code* (not a misuse question), open a GitHub issue marked
`security` or contact the maintainer via the account on the repository. Do not include exploit
payloads targeting third-party systems.

## Model caveat

The served/continuation model derives from an abliterated base and is **not** a safety-aligned
assistant. It is purpose-built for agent-trajectory continuation. Evaluate and gate it
appropriately for your deployment.
