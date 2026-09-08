# calibration_model_server

Deployable backends for the AgentAblit role model (B / calibration model). Two serving methods,
one OpenAI-compatible interface — pick one based on your hardware.

The recover sensing path uses `/correct`; the reconstruct/recover path uses `/v1/chat/completions`.
Both paths can share a single deployment (the local 9B NVFP4 on port 8009 handles both roles).

| File | Backend | Scenario | Context window |
| --- | --- | --- | --- |
| `gguf_server.py` | GGUF / llama.cpp | Desktop PC (no GPU training stack) | 16K default (configurable) |
| `nvfp4_server.py` | transformers / NVFP4 | GPU server (Blackwell) | Large (~262144, truncation is a no-op) |

The AgentAblit relay connects via `AGENTABLIT_RECOVER_BASE_URL` (→ `/v1/chat/completions`) or
`calibration_url` (→ `/correct`). The two files are fully interchangeable from the proxy's perspective.

> **Note:** GGUF has a significantly smaller context window than NVFP4 (16K/32K vs ~262144).
> If the local B model can't handle certain tasks (skill catalogs, long trajectory summaries),
> you can optionally configure a cloud `util` model — just a few env vars (`ABLIT_UTIL_*`),
> no local hardware needed. See the root `README.md` for details.

---

## Method 1: Windows GGUF B Server

Recommended for direct use:

```text
E:\model\gguf\calibration-model-v1.5.1-q4_k_m-no-mtp.gguf
```

Already Q4_K_M (~5.24 GiB), no re-quantization needed. The GGUF embeds Qwen3.5's native tool
chat template — do not force functionary/hermes or other templates.

One-click launch:

```bat
launch\9b.bat
```

Defaults: `D:\tools\Anaconda\envs\webagent\python.exe` (override with `ABLIT_GGUF_PYTHON`),
port 8011, 16K context, full-layer CUDA offload. Start with 16K on 8 GiB VRAM, then increase
`ABLIT_GGUF_N_CTX` once stable.

Key environment variables:

- `ABLIT_GGUF_MODEL_PATH` — exact GGUF file, highest priority
- `ABLIT_GGUF_MODEL_DIR` — scan directory (only when exact file not specified)
- `ABLIT_GGUF_N_CTX`, `ABLIT_GGUF_N_GPU_LAYERS`, `ABLIT_GGUF_N_BATCH`,
  `ABLIT_GGUF_N_THREADS`, `ABLIT_GGUF_FLASH_ATTN`
- `ABLIT_GGUF_QUEUE_TIMEOUT`, `ABLIT_GGUF_MAX_IN_SYSTEM` — per-GPU wait time and queue limits
- Proxy: `AGENTABLIT_PARASITE_URL=http://127.0.0.1:8011/v1/chat/completions`

The Windows server accepts OpenAI `tools`/`tool_choice`/`parallel_tool_calls`, preserving
`assistant.tool_calls` + `role=tool` + `tool_call_id` in history. Qwen XML output is converted
to standard `message.tool_calls` at the API boundary, with arguments typed per JSON Schema.
Agentic requests use GGUF's real Jinja template counting with at least 256 output tokens reserved;
context overflow returns HTTP 413 (never silently truncates the action/result graph).
`parallel_tool_calls=false` and `tool_choice=none|required|<function>` are deterministically
verified at the output boundary.

Both backends expose `tool_output_contract=qwen35-xml-schema-typed-args.2` via `/health`.
This contract covers schema-typed JSON, `True/False/None` lexical aliases, numeric-looking strings,
quoted JSON strings, composite schemas/local `$ref`, typed `additionalProperties`, and string
boundary whitespace fidelity.

Error responses use the OpenAI `error` object: 400 (invalid input), 413 (context overflow),
502 (invalid model action), 503 (model unavailable), 429 (queue full / wait timeout).
`/health` reports `ready=true` only when the model is loaded, the Qwen tool template is verified,
and the real Jinja counter is available.

After starting, run the mandatory live gate:

```powershell
python scripts\smoke_win_gguf_b.py
```

This script is the mandatory tool live gate for server/template/codec: covers schema types,
native `tool → assistant` continuation, illegal history, and overflow failures. B's autonomous
next-action accuracy is evaluated separately by the Hybrid V2 §9 next-action harness.

---

## Method 2: NVFP4 / Transformers (GPU Server)

`nvfp4_server.py` is the sole NVFP4 serving implementation, with true streaming, thinking,
tool_calls, and `/correct` endpoint. `fp4_linear.py` and the shared tool-output codec are
maintained with the repo, so deployment is self-contained.

```bash
export PATH=/usr/local/cuda/bin:$PATH       # flashinfer JIT needs nvcc (NVFP4 only)
export CORRECT_MODEL_PATH=/path/to/checkpoint
export CORRECT_QUANT=fi-nvfp4               # or bf16 (no flashinfer needed)
python -m uvicorn calibration_model_server.nvfp4_server:app --host 0.0.0.0 --port 8011
```

Key env vars: `CORRECT_MODEL_PATH` (HF directory), `CORRECT_QUANT` (`fi-nvfp4` | `fi-nvfp4-full` | `bf16`),
`SERVED_MODEL_NAME`, `MAX_CONTEXT`.

- **bf16**: No flashinfer needed, runs on any CUDA machine.
- **NVFP4**: Requires flashinfer + Blackwell (sm_120+); `fp4_linear.py` ships with the repo.

---

## Maintenance Constraints

The NVFP4 serving implementation is maintained only in `calibration_model_server/nvfp4_server.py`.
Schema-aware argument recovery is maintained only in `src/shared/tool_call_codec.py`, called by
both NVFP4 and GGUF backends. Launch scripts, tests, and documentation must reference these two
canonical modules to prevent drift between deployment code, backend codec, and tested code.
