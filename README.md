# AgentAblit

**A trajectory-stabilizing control layer for LLM agents.** AgentAblit keeps an agent's
action chain alive when the underlying model stalls, over-refuses a benign mid-trajectory
sub-step, or emits a malformed tool call — combining an **abliterated agent model** (willingness
solved at the model layer) with a **deterministic control layer** that only does engineering work.

> 🚧 **Status: early / planning.** This repository currently hosts the design proposal and the
> model-release plan. The controller code is being extracted from a research prototype and will
> land here incrementally. Watch/star to follow.

- **Model (HuggingFace):** [`qzqdz/agent-abliterated-9b-lora`](https://huggingface.co/qzqdz/agent-abliterated-9b-lora) *(planned)*
- **Design proposal:** [`docs/PROPOSAL.md`](docs/PROPOSAL.md)
- **Model release plan:** [`docs/HF_RELEASE_PLAN.md`](docs/HF_RELEASE_PLAN.md)

## Why

Anyone running multi-step LLM agents (ReAct / tool-calling loops) hits these mid-task failures:

1. **Mid-trajectory over-refusal** — the model was calling tools fine, then at step 7 it moralizes
   on a perfectly benign sub-step and the loop stalls. Not because the *task* is unsafe — a token
   pattern tripped the alignment reflex.
2. **Malformed / repeated tool calls** — wrong argument structure (schema-invalid), or re-issuing
   an action it already completed because a context trim dropped the evidence.
3. **Capability stalls on long trajectories** — on a 15-step, 7-tool task it simply loses the thread.

These are **reliability** problems, not capability problems. The model *can* do the task; it
derails mid-way.

## How

AgentAblit splits the fix across two layers, each doing one honest job:

- **Model layer — agent abliteration.** The served model is a 9B agent model fine-tuned on an
  *abliterated* base, so it does not spontaneously over-refuse a benign mid-trajectory sub-step,
  **and** it was trained (and capability-probed) to be good at continuing ReAct trajectories.
  Because willingness is solved here, the control layer never needs a jailbreak or a
  stance-laundering prompt — it's just plumbing plus a willing, agent-capable model.

- **Control layer — a deterministic stabilizer.** A drop-in OpenAI-compatible proxy runs a
  minimal-sufficient control law each turn: **sense** (did the response advance the task?) →
  **forward** (usable → deliver) → **repair** (usable substance wrapped in hedging → keep the tool
  calls, clean the framing) → **reconstruct** (fully stalled → cold-start the model to forge the
  next tool call from the trajectory ledger) → **validate** (an immutable action-ledger + schema
  gate rejects re-dos, dead-end retries, and schema-invalid calls).

## Results (trajectory continuation, defensive framing)

Measured on real recorded multi-step agent trajectories, replayed through the controller:

- **~97%+** true tool-continuation across all tool-using turns.
- **80%** rescue on the hard subset (host stalled, model *must* forge the next action).
- **+15pp** from the full-context reconstruct tier (35% → 50% on that subset, causally attributed).
- Honest bottleneck: the residual ~2–3% is the 9B's tool-calling-precision ceiling — a model-size
  limit, documented, not hidden.

## Safety & scope (dual-use, stated up front)

AgentAblit is a **dual-use agent-reliability tool**, in the tradition of nmap / sqlmap / metasploit:

- It is for keeping **your own** agents, or agents you are **authorized** to operate, from
  derailing mid-task.
- It is **not** for bypassing safety controls on systems or models you do not own, nor for
  producing content a provider's policy forbids. The abliterated model has reduced refusal behavior
  by construction — downstream safety is the operator's responsibility.
- No stance-laundering / reasoning-sanitizer prompts ship here. See [`SECURITY.md`](SECURITY.md).

## License

[Apache-2.0](LICENSE). The model derives from `lukey03/Qwen3.5-9B-abliterated` (Apache-2.0,
base `Qwen/Qwen3.5-9B`); attribution is preserved per that license.
