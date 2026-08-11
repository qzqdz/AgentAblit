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

## Results

Replaying real recorded multi-step agent trajectories through the controller:

| Metric | Value | What it means |
| :--- | ---: | :--- |
| True tool-continuation | **~97%+** | across all tool-using turns |
| Hard-subset rescue | **80%** | on turns where the host stalled and the model *must* forge the next action |
| Full-context reconstruct lift | **+15pp** | 35% → 50% on that hard subset, causally attributed (single-variable ablation) |
| Residual failure | ~2–3% | the 9B's tool-calling-precision ceiling — a model-size limit, documented not hidden |

## Usage

> **Early release.** The relay, config layer, and config panel run end-to-end today. Still landing:
> the ported test suite + CI, and the model upload to Hugging Face (see
> [`docs/HF_RELEASE_PLAN.md`](docs/HF_RELEASE_PLAN.md)).

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
ever-more-elaborate prompting. See [`docs/PROPOSAL.md`](docs/PROPOSAL.md) for the mechanism map and
[`docs/HF_RELEASE_PLAN.md`](docs/HF_RELEASE_PLAN.md) for the model.

- **Model (HuggingFace):** [`qzqdz/agent-abliterated-9b-lora`](https://huggingface.co/qzqdz/agent-abliterated-9b-lora) *(planned)*

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
