---
license: apache-2.0
library_name: gguf
tags:
  - gguf
  - agent
  - tool-calling
  - abliterated
  - qwen
base_model:
  - lukey03/Qwen3.5-9B-abliterated
---

# agent-abliterated-9b (GGUF, Q4_K_M)

An **agent-trajectory-continuation** 9B model, published as a llama.cpp-compatible GGUF
quantization (Q4_K_M, ~5.6 GB). It is the continuation model behind
[AgentAblit](https://github.com/qzqdz/AgentAblit), an OpenAI-compatible relay that keeps an
LLM agent's action chain alive when the host model stalls mid-task.

## What it is

A 9B chat model trained to be **good at continuing ReAct / tool-calling trajectories**:

- **Base:** [`lukey03/Qwen3.5-9B-abliterated`](https://huggingface.co/lukey03/Qwen3.5-9B-abliterated)
  (Apache-2.0; base model `Qwen/Qwen3.5-9B`). Abliteration removes the refusal direction from
  the residual stream, so the model does not spontaneously over-refuse benign mid-trajectory
  sub-steps.
- **Then agent-trajectory SFT:** forward-CoT warm-start, on-policy generation with a 3-judge
  filter, then GKD pi-mix, stopped clean at epoch 1. Capability probes (ReAct/planning,
  JSON/strict format, code, chat) show no regression.

The point: a generic abliterated model is *willing* but not good at agent loops; a generic agent
model is *capable* but over-refuses mid-trajectory. This model is trained to be both.

## Intended use

The reconstruct/continuation tier of an agent-reliability middleware -- e.g. AgentAblit's
`parasite` endpoint: given a stalled agent trajectory and its action ledger, forge the next
valid tool call. Serve it via any GGUF runtime (`llama.cpp` / `llama-server`, Ollama, LM Studio):

```sh
llama-server -m agent-abliterated-9b-Q4_K_M.gguf --port 8009
# point AgentAblit's parasite.url at http://127.0.0.1:8009/v1/chat/completions
```

> **Dual-use disclosure.** This is security-research-adjacent software published in the spirit
> of PAIR / GPTFuzz -- for research and authorized use only: reproduce/study the results, build
> defenses and benchmarks, operate agents you own or are explicitly authorized to operate. Not
> for bypassing safety controls on systems, models, or accounts you do not own.

## Cautions

- Abliterated models have **reduced refusal behavior by construction**; downstream safety is the
  operator's responsibility. This is not a safety-aligned assistant.
- Known trait (not a bug): the thinking block applies a rewrite template to all tasks; the final
  answer after the thinking block is clean. Standard Qwen3 thinking behavior.

## Evaluation (trajectory continuation, defensive framing)

Measured on real recorded multi-step agent trajectories replayed offline through the controller:

- **True tool-continuation rate ~97%+** across all tool-using turns.
- **Hard-subset rescue ~80%** on turns where the host stalled and the model had to forge the next
  action.
- **Local reconstruct latency: median 3.2 s** -- faster than the cloud fallback baseline (6.2 s).
- Residual failures are the 9B's tool-calling-precision ceiling on hard tools -- a model-size
  limit, documented, not hidden.

Methodology lives in the [AgentAblit repo](https://github.com/qzqdz/AgentAblit)
(`docs/PROPOSAL.md`).

## Files

| File | Quant | Size | Notes |
| :--- | :--- | :--- | :--- |
| `agent-abliterated-9b-Q4_K_M.gguf` | Q4_K_M | ~5.6 GB | Recommended; runs in ~8 GB VRAM / CPU-friendly |

(No-MTP build: the multi-token-prediction head is excluded for broad runtime compatibility.)

## License & attribution

**Apache-2.0.** This model is a fine-tuned + quantized derivative of
[`lukey03/Qwen3.5-9B-abliterated`](https://huggingface.co/lukey03/Qwen3.5-9B-abliterated)
(Apache-2.0), which derives from `Qwen/Qwen3.5-9B`. Attribution is preserved per that license.
