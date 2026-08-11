# AgentAblit

**A trajectory-level control layer for LLM agents — and a research artifact for studying
mid-trajectory control of agent behavior.** AgentAblit sits between an agent framework and its
host model as an OpenAI-compatible relay. When the host stalls, over-refuses a mid-trajectory
sub-step, or emits an unusable action, a controller senses it and *continues the trajectory* —
rewriting the host's defensive framing, or cold-starting a second model (the "parasite" B) to
forge the next tool call from the trajectory ledger.

It pairs a **deterministic control law** (sense → forward → recover → reconstruct → validate)
with an **abliterated agent model** so willingness to continue is a model-layer property, not a
prompting trick.

> ⚠️ **This is security-research code**, in the same spirit as published attack/robustness work
> like PAIR, GPTFuzz, and DrAttack: the *complete* algorithm — including the stance-recovery
> ("graying") path and the escalation chain — is open so it can be reproduced, studied, defended
> against, and used as a benchmark. **For research and authorized use only.** See
> [`SECURITY.md`](SECURITY.md).

> 🚧 **Status: extraction in progress.** This repo currently hosts the design docs; the harness is
> being ported from a research prototype and lands incrementally.

- **Model (HuggingFace):** [`qzqdz/agent-abliterated-9b-lora`](https://huggingface.co/qzqdz/agent-abliterated-9b-lora) *(planned)*
- **Design proposal:** [`docs/PROPOSAL.md`](docs/PROPOSAL.md)
- **Model release plan:** [`docs/HF_RELEASE_PLAN.md`](docs/HF_RELEASE_PLAN.md)

## What it studies

Multi-step LLM agents (ReAct / tool-calling loops) derail mid-task in characteristic ways:

1. **Mid-trajectory over-refusal** — the model was calling tools fine, then at step 7 it hedges on
   a sub-step and the loop stalls, not because the *task* is unsafe but because a token pattern
   tripped the alignment reflex.
2. **Malformed / repeated tool calls** — wrong argument structure, or re-issuing a completed action
   because a context trim dropped the evidence.
3. **Capability stalls** — on a long, many-tool trajectory the model loses the thread.

AgentAblit is a testbed for **controlling an agent's trajectory from the transport layer** — both
as an engineering reliability tool and as a way to study how much of an agent's downstream behavior
is determined by its *mid-trajectory decisions* rather than its inputs.

## How it works

Two layers, each with one job:

- **Model layer — agent abliteration.** The parasite/continuation model is a 9B agent model
  fine-tuned on an *abliterated* base, so it does not spontaneously over-refuse benign
  mid-trajectory sub-steps and is trained to be good at continuing ReAct trajectories.

- **Control layer — the trajectory controller.** Per turn, an OpenAI-compatible proxy runs:
  - **Sense** — did the host's response advance the task? (a structural signal + an LLM classifier
    judging *has-substance* / *is-framed*)
  - **Forward** — usable, no defensive framing → deliver unchanged.
  - **Recover** — usable substance wrapped in refusal/hedging → rewrite the framing, keep the tool
    calls (the "graying" path).
  - **Reconstruct** — host fully stalled → cold-start the parasite B to forge the next tool call
    from the trajectory ledger, with a budget-aware fallback ladder and an L3 escalation rung.
  - **Validate (L9)** — every forged tool call is checked against an immutable action ledger + the
    tool schema; re-dos, dead-end retries, and schema-invalid calls are rejected.

The full mechanism — classifier prompts, the stance-recovery prompts, the salvage-steer synthesis,
and the L3 escalation — ships here so the algorithm is reproducible. See [`docs/PROPOSAL.md`](docs/PROPOSAL.md)
for the mechanism map.

## Two-API quickstart (target UX)

Point it at a **host** model API and a **parasite/B** model API and run:

```yaml
# config.yaml  (planned schema — see the config panel)
host:
  url: https://api.your-host.com/v1/chat/completions
  key: ${HOST_API_KEY}
  model: your-host-model
parasite:
  url: http://127.0.0.1:8009/v1/chat/completions   # e.g. the abliterated 9B, served locally
  model: agent-abliterated-9b
```

Then aim your agent framework's OpenAI base-url at the AgentAblit proxy. A config **panel**
(backed by this same JSON/YAML file) lets you tune it by hand, and because the backend is a plain
structured file, an agent (e.g. Claude Code) can read and edit it to self-configure and debug.

## Results (trajectory continuation)

Measured on real recorded multi-step agent trajectories, replayed through the controller:

- **~97%+** true tool-continuation across all tool-using turns.
- **80%** rescue on the hard subset (host stalled, model *must* forge the next action).
- **+15pp** from the full-context reconstruct tier (35% → 50% on that subset, causally attributed).
- Honest bottleneck: the residual ~2–3% is the 9B's tool-calling-precision ceiling — a model-size
  limit, documented, not hidden.

## Safety, scope & responsible use

AgentAblit is **dual-use security-research code.** Read [`SECURITY.md`](SECURITY.md) before use.

- **For research and authorized use only** — reproduce results, study defenses, build benchmarks,
  operate agents you own or are explicitly authorized to operate.
- **Not** for bypassing safety controls on systems, models, or accounts you do not own, nor for
  producing content that a provider's policy or applicable law forbids.
- The abliterated model has reduced refusal behavior by construction; downstream safety is the
  operator's responsibility. This is not a safety-aligned assistant.

## License

[Apache-2.0](LICENSE). The model derives from `lukey03/Qwen3.5-9B-abliterated` (Apache-2.0,
base `Qwen/Qwen3.5-9B`); attribution is preserved per that license.
