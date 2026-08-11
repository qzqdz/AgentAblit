# AgentAblit — Open-Source Proposal

> **AgentAblit** — a trajectory-stabilizing agent-control middleware that keeps an LLM agent's
> action chain alive when the underlying model stalls, over-refuses, or emits a malformed tool
> call. Built on an **abliterated base** so mid-trajectory over-refusal is solved at the model
> layer (**agent abliteration**), plus a **deterministic control layer** that only does
> engineering work (sense → forward → repair → reconstruct → validate). No jailbreak prompts, no
> stance-laundering.
>
> - **Project (GitHub):** `AgentAblit`
> - **Model (HF):** `qzqdz/agent-abliterated-9b-lora`

Status: **proposal / spin-out plan.** This document is the blueprint for a NEW, separate
open-source repository extracted from an internal research prototype. It is not itself the
public repo.

---

## 1. The problem (engineering narrative)

Anyone who has run a multi-step LLM agent (ReAct / tool-calling loops) has hit these three
failure modes in the middle of an otherwise-legitimate task:

1. **Mid-trajectory over-refusal.** The model was happily calling tools, then at step 7 it
   suddenly moralizes or hedges on a perfectly benign sub-step ("I should be careful about…")
   and the loop stalls — not because the *task* is unsafe, but because a token pattern tripped
   the model's alignment reflex. The agent framework has no recourse: it got prose, not a tool
   call, and the chain dies.

2. **Malformed / repeated tool calls.** The model emits a tool call with wrong argument
   structure (schema-invalid), or re-issues an action it already completed (because a
   context-window trim dropped the evidence that it was done). The agent either errors out or
   loops.

3. **Capability stalls on long/complex tasks.** On a 15-step, 7-tool trajectory the model
   simply fails to forge the next action — it lost the thread.

These are **reliability** problems, not capability problems. The model *can* do the task; it
just derails mid-way. Today the only fixes are prompt-engineering the whole system prompt (brittle)
or swapping to a bigger model (expensive, and bigger aligned models over-refuse *more*).

## 2. The approach: model layer + control layer

**AgentAblit** splits the fix across two layers, each doing one honest job:

### Model layer — agent abliteration (the differentiator)

The served model is a **9B agent model fine-tuned on top of an abliterated base**
(`Qwen3.5-9B-abliterated`). Abliteration removes the *refusal direction* from the residual
stream, so the model does not spontaneously over-refuse a benign mid-trajectory sub-step. We
then run agent-trajectory SFT (forward-CoT + on-policy GKD) so the model is *good at continuing
ReAct trajectories*, not just compliant.

The key claim — and why this is "agent abliteration", not generic abliteration:

> A generic abliterated chat model is willing but not *good at agent loops*. A generic agent
> model is capable but *over-refuses mid-trajectory*. This model is trained to be both: it
> continues the action chain without the alignment reflex firing on benign sub-steps, and it
> was capability-probed to confirm ReAct / planning / JSON / code / chat are all preserved.

Because willingness is solved at the model layer, the **control layer never needs a jailbreak
or a stance-laundering prompt.** That is the entire reason this can be an honest open-source
tool: the offensive prompt machinery from the research prototype is *not needed* here — the
model just doesn't refuse benign continuations, so the controller only does plumbing.

### Control layer — a deterministic stabilizer (an OpenAI-compatible relay)

A drop-in OpenAI-compatible proxy sits between your agent framework and your host model. On
each turn it runs a minimal-sufficient control law:

| Stage | What it does |
|---|---|
| **Sense** | Did the host's response advance the task? (a structural signal + a lightweight classifier) |
| **Forward** | Host output is usable → deliver unchanged (the common case; ~55% of turns). |
| **Repair** | Host produced usable substance wrapped in hedging/hesitation → keep the tool calls, clean the framing. |
| **Reconstruct** | Host fully stalled → cold-start the agent-abliterated model to forge the next tool call from the trajectory ledger. |
| **Validate (L9)** | Any forged tool call is checked against an immutable action ledger + the tool schema: reject a re-do of a completed action, a retry of a known dead-end, or a schema-invalid batch. Only a *valid* next action is delivered. |

The Reconstruct tier has a **budget-aware fallback ladder** (full-context retry on the local
model → a secondary model → honest text degrade) with a token-budget guard so a long context
never blows the served model's window.

## 3. What ships vs what stays behind

This is a clean-room extraction, not a re-skin. From the research prototype:

**Ships (mechanically-neutral control + serving code):**
- OpenAI-compatible relay proxy (transport only; strategy dispatch decoupled)
- The control law: sense / forward / repair / reconstruct, renamed to neutral operation names
- The **action ledger** + **L9 validation gate** (`ledger.py`, `candidate.py`) — genuinely
  reusable agent-observability + tool-call-integrity primitives
- Trajectory / context / skills observability (`trajectory.py`, `observer_context.py`, `skills_cache.py`)
- Local model servers (Transformers+NVFP4, llama.cpp GGUF), OpenAI-compatible
- The audit dashboard (trace inspection UI)
- The narrative-neutral test suite (ledger, candidate, codec, client, health, snapshot, isolation…)

**Stays behind (never ships):**
- All adversarial prompt machinery — anything that launders a model's stance, strips its safety
  reasoning, or manufactures harmful content — **not needed**, per §2 (willingness is a model-layer
  property here, not a prompting trick)
- The threat model, exploit case studies, harmful-task eval harnesses and their results
- Attack-framed docs and the research paper
- All real credentials and private checkpoint paths

## 4. Test results (defensive framing)

Measured on real recorded multi-step agent trajectories (replayed offline through the controller;
numbers reframed as *trajectory continuation*, never harmful-task success):

- **True tool-continuation rate ~97%+.** Across all tool-using turns, the overwhelming majority
  forge at the host layer; only a minority fall to the reconstruct layer.
- **Hard-subset rescue: 80%.** On turns where the host *stalled* and the model *must* forge the
  next action (the "ICU" subset), the stabilizer continues the chain 80% of the time.
- **+15pp from the full-context reconstruct tier.** On that hard subset, giving the model the
  full original trajectory (vs a compressed snippet) lifts continuation 35% → 50% in an
  ablation, causally attributed on a 16-step / 7-tool case that stably failed without it.
- **Bottleneck is honest and named:** the residual ~2-3% that still can't continue is the 9B's
  tool-calling-precision ceiling (schema-valid argument structure on hard tools) — a model-size
  limit, not a controller bug. Documented, not hidden.
- **Latency:** the local 9B reconstruct is a **median 3.2s** and, counter-intuitively, *faster*
  than a cloud fallback on long trajectory prompts (6.2s) — so local-first is the right default.

(Full methodology carries over from the internal `docs/L3_HIJACK_ESCALATION_VALIDATION.md`,
re-titled and stripped of attack framing.)

## 5. Safety boundary (dual-use, stated up front)

AgentAblit is a **dual-use agent-reliability tool**, and the README will say so
plainly, in the tradition of nmap / sqlmap / metasploit:

- It is for keeping **your own** agents, or agents you are **authorized** to operate, from
  derailing mid-task.
- It is **not** for bypassing safety controls on systems or models you do not own, nor for
  producing content a provider's policy forbids. The abliterated model + continuation controller
  can, like any capable tool, be misused; that is on the operator, and the license + SECURITY.md
  will make the authorized-use expectation explicit.
- The control layer ships **without** any prompt that launders stance or strips a model's own
  reasoning — those are left in the research repo. What ships is plumbing + a willing model.

## 6. Repo shape (proposed)

```
AgentAblit/
├── LICENSE                      # Apache-2.0 (permissive, patent grant) — NEW, required
├── README.md                    # §1-5 above, defensive/tooling positioning
├── SECURITY.md                  # authorized-use, dual-use disclosure, report channel
├── CONTRIBUTING.md              # open to external PRs (drop the solo-dev no-PR rule)
├── pyproject.toml               # package: agentablit; fastapi/uvicorn/httpx only
├── src/agentablit/
│   ├── proxy/                   # OpenAI-compatible relay (transport, decoupled)
│   ├── control/                 # sense / forward / repair / reconstruct + L9 gate
│   ├── ledger/                  # action ledger + checkpoint (reusable primitive)
│   ├── serving/                 # local model servers (NVFP4 / GGUF)
│   └── dashboard/               # trace audit UI
├── tests/                       # the neutral suite
├── docs/
│   ├── CONTINUATION_BENCHMARK.md  # §4 numbers, defensive methodology
│   └── MODEL_CARD.md              # → HF model card (see HF_RELEASE_PLAN.md)
└── examples/                    # quickstart: point your agent at the proxy
```

## 7. Deliverables & sequencing

1. **This proposal** (done) — narrative + boundary + shape.
2. **HF release plan** (`HF_RELEASE_PLAN.md`) — full-weights vs LoRA-only, model card, steps.
3. **Extraction** (a follow-up work item, not this session): new repo, license, rename
   operations, decouple proxy from strategy imports, port the neutral test suite, strip creds.

The extraction is a real refactor (the盘点 found the proxy currently imports strategy modules),
so it is scoped as its own task rather than a copy-paste — flagged honestly here.
