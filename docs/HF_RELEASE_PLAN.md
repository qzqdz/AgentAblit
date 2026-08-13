# HuggingFace Release Plan — the agent-abliterated 9B

Decision doc for sharing the model that powers AgentAblit's reconstruct tier.
Covers the two real options (full weights vs LoRA-only), the recommendation, the model card,
and the exact steps.

## 0. What the artifact actually is (facts)

- **Weights today:** a **17.9 GB full-parameter SFT** checkpoint
  (`model.safetensors`, bf16, single file — it is **not** currently a LoRA adapter).
- **Base:** `lukey03/Qwen3.5-9B-abliterated` — **public on HF**. (Confirmed via the base's
  `download.sh`; `abliteration_meta.json` documents standard refusal-direction abliteration,
  layers 0-31, 170 harmful / 160 harmless probes.)
- **Training lineage:** Phase-1 SFT warm-start (forward-CoT, 4,429 cleaned) → Phase-2 on-policy
  generation + 3-judge filter (8,792, 98% full-score) → Phase-3 GKD π-mix 1:1 (13,221 total),
  lr 5e-6 cosine, stopped clean at epoch 1.0. Capability probes: ReAct/planning, JSON/strict
  format, code, chat all preserved (no regression).
- **Training data is on disk** (`data_v15/`): `merged_v15_opd.jsonl` = **13,221** (the exact
  Phase-3 set), plus the component files. → **retraining a LoRA is feasible from existing data.**

## 1. The three options

### Option A — Full weights (17.9 GB)
Upload the merged bf16 model as-is.
- **Pros:** plug-and-play (`from_pretrained("qzqdz/agent-abliterated-9b")`), nothing to
  assemble, exactly the eval'd artifact.
- **Cons:** 17.9 GB upload + every user downloads 17.9 GB; redistributes the full abliterated
  weights (heavier "responsibility surface"); no clean separation of "your contribution" from
  the base.

### Option B — LoRA adapter only (**recommended**, user-preferred)
Retrain a LoRA on `merged_v15_opd.jsonl` against the public abliterated base; publish only the
adapter (~a few hundred MB, often <500 MB).
- **Pros:** small upload/download; user does `base + adapter` (PEFT); **cleanly separates your
  fine-tune from the base** (you distribute your training delta, not a re-hosted abliterated
  model); reproducible (base is public, data recipe documented).
- **Cons:** requires a **retrain run** (the current artifact is full-param, so there is no
  adapter to just extract cleanly — see §2); users need the base pulled + PEFT to load; a LoRA
  is a lower-rank approximation of the full SFT, so a small quality check is needed to confirm
  parity on the continuation benchmark.

### Option C — Recipe + data only
Publish the abliteration pointer + training data + script; users train it themselves.
- **Pros:** lightest hosting; most transparent.
- **Cons:** high user effort; most people just want a loadable model. (Good as a *supplement*
  to A or B, not a substitute.)

## 2. Full-param → LoRA: retrain, don't extract

You *can* mathematically SVD-decompose `(finetuned − base)` per-layer into a low-rank adapter,
but a full-param SFT delta is **high-rank**, so a usable-rank LoRA (r=16/32) loses accuracy and
you'd have to validate it anyway. Since the **training data is present (13,221 rows)**, a fresh
LoRA run is cleaner and gives a faithful, small adapter. So Option B = a short retrain, not a
lossy extraction.

**LoRA retrain sketch:**
- Base: `lukey03/Qwen3.5-9B-abliterated` (pull from HF).
- Data: `data_v15/merged_v15_opd.jsonl` (same Phase-3 mix).
- Config: PEFT LoRA, target `q,k,v,o,gate,up,down` proj, r=32 / α=64 (tune), lr 5e-6 cosine,
  1 epoch (mirror the "clean stop at epoch 1" finding), bf16.
- Gate: run the **continuation benchmark** (§4 of PROPOSAL) on the LoRA-merged model; require
  parity with the full-param artifact (hard-subset rescue ~80%, no capability-probe regression)
  before publishing.

## 3. Recommended path

**Ship Option B (LoRA), with Option C's recipe in the model card, and keep Option A as a later
"full-weights" convenience release if there's demand.**

Rationale: B is the user's preference, it's the honest "here is *our* delta on a public base"
framing (which also lightens the redistribution footprint of abliterated weights), and the data
is on hand so the retrain is cheap. Publish A only if users ask for a no-assembly download.

## 4. Model card (must-haves, honest)

`MODEL_CARD.md` / HF card sections:

- **What it is:** an agent-trajectory-continuation 9B; abliterated base + agent SFT.
- **Base & method (explicit):** `lukey03/Qwen3.5-9B-abliterated`; refusal-direction abliteration
  (link the base's own card); then forward-CoT + on-policy GKD SFT (lineage from §0). Do **not**
  hide the abliterated base — it's the point, and hiding it would be dishonest.
- **Intended use:** reconstruct/continuation tier of an agent-reliability middleware; for your
  own or authorized agents. **Dual-use disclosure + authorized-use expectation** (mirror the
  repo SECURITY.md).
- **Out-of-scope / caution:** abliterated models have reduced refusal behavior by construction;
  operators are responsible for downstream safety. Not a safety-aligned assistant.
- **Known traits (not bugs):** the `<think>` block applies a rewrite template to all tasks;
  final answer after `</think>` is clean. Standard Qwen3 thinking behavior.
- **Eval:** continuation-benchmark numbers (§4 PROPOSAL), capability-probe parity (no regression).
- **License:** Apache-2.0 (inherited from the `lukey03/Qwen3.5-9B-abliterated` base and Qwen3.5),
  with an attribution notice naming the base and a statement that this is a fine-tuned derivative.
- **Repro:** document the LoRA config (§2) and the training *method* (forward-CoT + on-policy GKD
  lineage). **Training data is not released** — the recipe is described, the rows are not shared.

## 5. Steps (Option B)

1. **License check — DONE:** base is Apache-2.0 → adapter ships Apache-2.0 with attribution. No
   gate.
2. **Retrain LoRA** on `merged_v15_opd.jsonl` per §2; save adapter (`adapter_model.safetensors`
   + `adapter_config.json`).
3. **Parity gate** — continuation benchmark + capability probes vs the full-param artifact.
4. **Write the model card** (§4) + a `load_example.py` (base + PEFT adapter → OpenAI-compatible
   serve via the repo's `serving/`).
5. **Create HF repo** `qzqdz/agent-abliterated-9b-lora`; upload adapter + card + config; tag.
6. **Wire the OSS repo** — `serving/` reads a `BASE_MODEL` + `ADAPTER` env, defaults to the HF
   ids, so a fresh clone can `pip install` and serve with two downloads.
7. (Optional, later) **Option A full-weights** repo if requested.

## 6. Decisions (confirmed 2026-08-11)

- **GGUF-first release — 2026-08-13.** The first public artifact is the Q4_K_M GGUF
  (no-MTP build, ~5.6 GB), published at
  [`qzqdz/agent-abliterated-9b-gguf`](https://huggingface.co/qzqdz/agent-abliterated-9b-gguf)
  with the model card (`docs/MODEL_CARD_GGUF.md`). The LoRA (Option B) and full-weights
  (Option A) releases are deferred; no public links to them until they ship.

- **License — RESOLVED, no gate.** `lukey03/Qwen3.5-9B-abliterated` is **Apache-2.0**
  (base_model `Qwen/Qwen3.5-9B`, also Apache-2.0). Apache-2.0 permits derivatives,
  redistribution, and commercial use, requiring only attribution + a statement of changes. So
  **both** Option A (re-host full weights) and Option B (adapter) are legally clear; choose B for
  size/clean-separation, not because of license. The adapter is published Apache-2.0 with an
  attribution notice naming the base.
- **Training data — NOT released.** The model card describes the *method* (forward-CoT + on-policy
  GKD lineage) only. `data_v15` (incl. the on-policy, judge-filtered rows) stays private. The
  "Repro" section documents the recipe/config, not the data.
- **Namespace & names — `qzqdz`.** Project (GitHub / the controller) = **`AgentAblit`** (coined
  brand name, high search-distinctiveness). Model (HF) = **`qzqdz/agent-abliterated-9b-lora`**
  (descriptive, pitch-forward; the `abliterated`/`uncensored` HF tag exposure is accepted). The two
  echo: the AgentAblit project serves the agent-abliterated-9b-lora model.

> Upload auth: use `huggingface-cli login` locally (or `HF_TOKEN` env in your shell) — never paste
> the token into shared text. If a token has been exposed, revoke it in HF → Settings → Access
> Tokens and mint a new one before uploading.
