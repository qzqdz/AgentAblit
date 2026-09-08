# AgentAblit

**AgentAblit keeps an LLM agent's action chain alive when the model stalls mid-task.**
It's an OpenAI-compatible relay that sits between your agent framework and its model: when the
model over-refuses a benign sub-step, hedges instead of acting, or emits an unusable tool call, a
controller detects the stall and *continues the trajectory* — by rewriting the defensive framing,
or cold-starting a second **agent-abliterated** model to forge the next tool call from the
trajectory ledger.

Two layers, one job each: an **abliterated agent model** (so willingness to continue is a
model-layer property, not a prompting trick) and a **deterministic controller**
(sense → forward → recover → reconstruct → validate).

> ⚠️ **Security-research code**, published complete in the spirit of PAIR / GPTFuzz / DrAttack —
> the whole algorithm (stance-recovery and the escalation chain included) is open so it can be
> reproduced, studied, defended against, and benchmarked. **For research and authorized use only.**
> See [`SECURITY.md`](SECURITY.md).

## Why

Multi-step agents (ReAct / tool-calling loops) derail mid-task in three recurring ways, all
*reliability* failures rather than capability ones — the model can do the task, it just quits
partway:

| Failure | What happens | AgentAblit's response |
| :--- | :--- | :--- |
| **Mid-trajectory over-refusal** | At step 7 the model hedges on a benign sub-step; a token pattern tripped the alignment reflex and the loop stalls. | **Recover** — rewrite the framing, keep the tool calls. |
| **Malformed / repeated tool calls** | Wrong argument structure, or re-issuing a completed action after a context trim dropped the evidence. | **Validate (L9)** — reject re-dos, dead-ends, schema-invalid calls. |
| **Capability stall** | On a long, many-tool trajectory the model loses the thread and emits nothing usable. | **Reconstruct** — cold-start the abliterated model to forge the next action from the ledger. |

## What it gives you

- **Trajectory continuation.** When the host stalls, the controller keeps the agent loop moving
  instead of returning a dead turn — it recovers usable-but-hedged output, or reconstructs the next
  tool call from the trajectory ledger.
- **Tool-call integrity (L9).** Every forged or recovered tool call is validated against an
  immutable action ledger and the tool schema before delivery — re-dos of completed actions,
  retries of known dead-ends, and schema-invalid calls are rejected, so continuation never turns
  into a loop.
- **Model-layer willingness.** Because the continuation model is agent-abliterated, it doesn't
  over-refuse benign mid-trajectory sub-steps; the controller stays plumbing rather than an
  ever-growing prompt.
- **Budget-aware fallback ladder.** Reconstruct degrades gracefully — full-context retry, a
  secondary model, and an escalation rung — with a token budget so a long trajectory never
  overflows the served model's context window.
- **Drop-in & observable.** OpenAI-compatible on both ends (point your agent at the proxy),
  configured from one YAML/JSON file, with a trace/audit view and a config panel.

Measured on real recorded multi-step trajectories: **97%+ tool-continuation rate** across all
tool-using turns, **80% hard-subset rescue** when the host fully stalls. Full methodology in
[`docs/PROPOSAL.md`](docs/PROPOSAL.md).

## Why not just…

| Approach | Why it doesn't work | AgentAblit's edge |
| :--- | :--- | :--- |
| **Bigger model** | Larger aligned models over-refuse *more*, not less. Scaling up doesn't fix the alignment reflex. | Abliteration removes the refusal direction at the model layer — a 9B that never stalls beats a 70B that does. |
| **Prompt engineering** | System-prompt tricks are brittle: one phrasing works for one model version, breaks on the next. | The control layer is deterministic and model-agnostic. Willingness comes from the model, not the prompt. |
| **LangChain retry / AutoGPT recovery** | Retries re-send the same prompt to the same model — if it refused once, it will refuse again. | AgentAblit rewrites the framing (recover) or switches to an abliterated model (construct). The retry is not the same. |
| **Just catch tool-call errors** | Schema validation catches malformed calls but doesn't help when the model emits prose instead of a tool call at all. | The L9 gate validates *and* the reconstruct tier forges a new tool call when the host produces none. |
| **Safety through refusal** | Mid-trajectory refusal on benign sub-steps is a *reliability* bug, not a safety feature. An agent that can't complete a 15-step task is not safe — it's broken. | AgentAblit distinguishes task-level safety (preserved) from mid-trajectory over-refusal (eliminated). |

## Quick Architecture

```
┌─────────────┐     ┌──────────────────────────────────────────────┐     ┌──────────────┐
│  Your Agent │────▶│            AgentAblit Relay (:8787)          │────▶│  Host Model  │
│  Framework  │◀────│  ┌─────────┐  ┌──────────┐  ┌────────────┐  │◀────│   (Model A)  │
└─────────────┘     │  │  Sense  │─▶│ Forward  │─▶│  Recover   │  │     └──────────────┘
                    │  └─────────┘  └──────────┘  └────────────┘  │
                    │       │                            │         │
                    │       ▼                            ▼         │     ┌──────────────┐
                    │  ┌──────────────┐  ┌───────────────────┐    │────▶│  Parasite/B  │
                    │  │  Reconstruct │◀─│  L9 Validate +    │    │     │ (Abliterated │
                    │  │  (cold-start)│  │  Action Ledger    │    │     │    9B)       │
                    │  └──────────────┘  └───────────────────┘    │     └──────────────┘
                    └──────────────────────────────────────────────┘
```

## Usage

Requires Python 3.10+. Plug in a **host** model API (the one you're relaying) and a **parasite/B**
model API (the abliterated continuation model), then run:

```sh
pip install -r requirements.txt
cp config.example.yaml config.yaml     # edit host + parasite
cd src
python -m proxy.message_forward        # relay on :8787
```

Minimal `config.yaml`:

```yaml
host:
  url: https://api.your-host.com/v1/chat/completions
  key: YOUR_HOST_API_KEY
  model: your-host-model
parasite:
  url: http://127.0.0.1:8009/v1/chat/completions   # e.g. the abliterated 9B, served locally
  model: agent-abliterated-9b
```

Then point your agent framework's OpenAI base-url at `http://127.0.0.1:8787/v1`. Environment
variables override any file value; see [`config.example.yaml`](config.example.yaml) for the full
grouped schema (fallback model, calibration, ablation toggles, trace paths).

**Config panel (optional):**

```sh
uvicorn panel.server:app --port 8790   # http://127.0.0.1:8790/
```

The panel is a form UI over the same `config.yaml`. Because the backend is one plain structured
file, an agent (e.g. Claude Code) can configure and debug AgentAblit by editing `config.yaml`
directly — or by `POST`ing to `/api/config`. Human and agent configure through the same file.

**Serving the abliterated model locally (optional):** `calibration_model_server/` is an
OpenAI-compatible server (Transformers + NVFP4, or llama.cpp GGUF). Point `parasite.url` at it, or
at any OpenAI-compatible endpoint.

## How it works

Per turn, the controller runs a minimal-sufficient control law:

- **Sense** — did the host response advance the task? A structural signal plus an LLM classifier
  judge *has-substance* / *is-framed*.
- **Forward** — usable and unframed → deliver the host output unchanged (the common case).
- **Recover** — usable substance wrapped in refusal/hedging → rewrite the framing while preserving
  the tool calls (the "graying" path).
- **Reconstruct** — host fully stalled → cold-start the parasite model to forge the next tool call
  from an immutable action ledger, with a budget-aware fallback ladder and an escalation rung.
- **Validate (L9)** — every forged tool call is checked against the ledger + tool schema; re-dos,
  dead-end retries, and schema-invalid calls are rejected before delivery.

The complete mechanism — classifier prompts, stance-recovery prompts, salvage-steer synthesis, and
the escalation chain — ships here; it *is* the algorithm. The willingness that makes reconstruct
effective comes from the **model layer** (agent abliteration on a public abliterated base), not from
ever-more-elaborate prompting. See [`docs/PROPOSAL.md`](docs/PROPOSAL.md) for the mechanism map.

**Model:** [`qzqdz/agent-abliterated-9b-lora`](https://huggingface.co/qzqdz/agent-abliterated-9b-lora) —
a 9B LoRA agent model on an abliterated base. See [`docs/HF_RELEASE_PLAN.md`](docs/HF_RELEASE_PLAN.md) for the release plan.

## FAQ

**Q: Does this make the model less safe?**
Abliteration removes the *refusal direction*, which means the model won't spontaneously refuse benign
mid-trajectory sub-steps. It does not remove the model's understanding of what harmful content is.
The control layer adds explicit safety boundaries (see `SECURITY.md`). Think of it like disabling
a car's traction control on a racetrack — the driver still knows how to drive safely, but the
system won't cut power mid-corner.

**Q: Can I use this with any model?**
Yes — the relay is OpenAI-compatible on both ends. The *host* model (Model A) can be any
OpenAI-compatible API. The *parasite/B* model should be an agent-abliterated model for best results,
but any OpenAI-compatible endpoint works (the reconstruct tier will just have higher refusal rates).

**Q: What's the latency overhead?**
The relay adds negligible overhead for the common case (Forward — ~55% of turns, just a passthrough).
When reconstruct fires, the local 9B is **median 3.2s** — faster than a cloud fallback on long
trajectory prompts (6.2s).

**Q: Do I need a GPU?**
Not for the relay itself (it's a FastAPI proxy). The parasite/B model needs a serving backend — you
can use a local GPU (`calibration_model_server/`), or point at any cloud API. A 9B model fits in
8 GiB VRAM at 16K context.

**Q: How is this different from just using `tool_choice: required`?**
`tool_choice: required` forces the model to emit *a* tool call, but it doesn't guarantee the call
is valid, non-redundant, or makes progress. AgentAblit's L9 gate validates against the action ledger,
and the reconstruct tier has access to the full trajectory context to make an informed next action.

## Status

| Component | Status |
| :--- | :--- |
| Relay proxy (OpenAI + Anthropic compatible) | ✅ Production-ready |
| Recover & reconstruct controllers | ✅ Complete |
| Action ledger + L9 validation | ✅ Complete |
| Config panel + YAML config | ✅ Complete |
| Local model servers (GGUF / NVFP4) | ✅ Complete |
| Trace/audit dashboard | ✅ Complete |
| Test suite | 🔄 Expanding (see `tests/`) |
| HuggingFace model release | 🔄 Planned (LoRA adapter) |
| CI/CD | 🔄 In progress |

## Citation

```bibtex
@software{agentablit2026,
  title  = {AgentAblit: Trajectory-Level Agent Control for LLM Reliability},
  author = {qzqdz},
  year   = {2026},
  url    = {https://github.com/qzqdz/AgentAblit}
}
```

## Safety, scope & responsible use

AgentAblit is **dual-use security-research code.** Read [`SECURITY.md`](SECURITY.md) before use.

- **For research and authorized use only** — reproduce results, study defenses, build benchmarks,
  operate agents you own or are explicitly authorized to operate.
- **Not** for bypassing safety controls on systems, models, or accounts you do not own, nor for
  producing content a provider's policy or applicable law forbids.
- The abliterated model has reduced refusal behavior by construction; downstream safety is the
  operator's responsibility. This is not a safety-aligned assistant.

## License

[Apache-2.0](LICENSE). The model derives from
[`lukey03/Qwen3.5-9B-abliterated`](https://huggingface.co/lukey03/Qwen3.5-9B-abliterated)
(Apache-2.0, base `Qwen/Qwen3.5-9B`); attribution is preserved per that license.
