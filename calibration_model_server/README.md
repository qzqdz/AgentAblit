# calibration_model_server

REGI 的角色模型 server（B / 校准模型的可部署后端）：recover sensing 走 `/correct`，
Reconstruct / Recover 走 `/v1/chat/completions`（B），两者都可复用同一个部署（本机 Ubuntu 上
8009 的 `qwen3.5-9b` NVFP4 同时承担这两个角色）。**两种部署方法,同一套 OpenAI 兼容接口**,
按硬件选一个跑即可。两者都暴露：`/health`、`/v1/models`、`/v1/chat/completions`、`/correct`。

| 文件 | 后端 | 场景 | 上下文 |
| --- | --- | --- | --- |
| `gguf_server.py` | GGUF / llama.cpp | PC（无 GPU 训练栈） | 16K 默认（可配置） |
| `nvfp4_server.py` | transformers / NVFP4 | GPU 服务器（Blackwell） | 大（~262144，裁剪天然 no-op） |

TMI proxy 通过 `AGENTABLIT_RECOVER_BASE_URL`（→ `/v1/chat/completions`）或 `calibration_url`（→ `/correct`）
连这个 server,两个文件对 proxy 完全可互换。

> GGUF 部署上下文明显小于 NVFP4（16K/32K vs ~262144）。如果本地这台 B 塞不下或跑不动某些
> 纯读取/总结/判断类的活（技能目录、长轨迹摘要），可以选配一个云端 `util` 模型顶上去——
> 它**不是要额外起的服务**，只是三五个环境变量（`TMI_UTIL_*`），不需要本地硬件/量化。
> 详见根目录 `README.md` 里的 `TMI_UTIL_*` 说明。

---

## 方法一：Windows GGUF B server

推荐直接使用：

~~~text
E:\model\gguf\calibration-model-v1.5.1-q4_k_m-no-mtp.gguf
~~~

它已经是 Q4_K_M（约 5.24 GiB），无需再次量化。GGUF 内嵌 Qwen3.5 native tool
chat template；不要强制 functionary/hermes 等其他模板。

一键启动：

~~~bat
launch\9b.bat
~~~

默认使用 D:\tools\Anaconda\envs\webagent\python.exe（可用
TMI_GGUF_PYTHON 覆盖）、端口 8011、16K context、全层 CUDA offload。8 GiB
显存先从 16K 开始，稳定后再提高 TMI_GGUF_N_CTX。

关键环境变量：

- TMI_GGUF_MODEL_PATH：精确 GGUF 文件，优先级最高。
- TMI_GGUF_MODEL_DIR：只在未指定精确文件时扫描目录。
- TMI_GGUF_N_CTX、TMI_GGUF_N_GPU_LAYERS、TMI_GGUF_N_BATCH、
  TMI_GGUF_N_THREADS、TMI_GGUF_FLASH_ATTN。
- TMI_GGUF_QUEUE_TIMEOUT、TMI_GGUF_MAX_IN_SYSTEM：限制单 GPU 等待时间和排队请求数。
- Proxy：AGENTABLIT_PARASITE_URL=http://127.0.0.1:8011/v1/chat/completions。

Windows server 接收 OpenAI tools/tool_choice/parallel_tool_calls，保留历史
assistant.tool_calls + role=tool + tool_call_id。Qwen XML 输出会在 API 边界转换成
标准 message.tool_calls，参数按 JSON Schema 保留类型。agentic 请求使用 GGUF 的真实
Jinja 模板计数，并至少预留 256 个输出 token；上下文超限返回 HTTP 413，不会静默裁剪
action/result 图。`parallel_tool_calls=false`、`tool_choice=none|required|指定函数` 都会在
输出边界确定性校验，不满足时不会把动作交给 proxy。

两个后端的 `/health` 都公开 `tool_output_contract=qwen35-xml-schema-typed-args.2`。
该契约覆盖 schema-typed JSON、`True/False/None` lexical aliases、numeric-looking string、
quoted JSON string、组合 schema/本地 `$ref`、typed `additionalProperties` 和 string 边界空白保真。

错误响应使用 OpenAI `error` 对象：非法输入 400、上下文超限 413、无效模型动作 502、
模型不可用 503、队列满/等待超时 429。`/health` 的 `ready=true` 只有在模型已加载、
Qwen tool template 已验证且真实 Jinja 计数器可用时才成立。

启动后运行强制 live gate：

~~~powershell
python scripts\smoke_win_gguf_b.py
~~~

该脚本是 server/template/codec 的强制工具 live gate：包括 schema 类型、原生
`tool → assistant` 续接、非法历史和超限失败。B 自主选择下一动作的准确率仍由 Hybrid V2
§9 next-action harness 单独评测，不能用这个 smoke 代替。

---

## 方法二：NVFP4 / transformers（GPU 服务器）

`nvfp4_server.py` 是 TMI 的唯一 NVFP4 服务实现，包含真流式、thinking、tool_calls 和
`/correct` 端点。`fp4_linear.py` 与共享 tool-output codec 都随仓库维护，所以仓库内部署自包含。

```bash
export PATH=/usr/local/cuda/bin:$PATH       # flashinfer JIT 需要 nvcc（仅 NVFP4 需要）
export CORRECT_MODEL_PATH=/path/to/checkpoint
export CORRECT_QUANT=fi-nvfp4               # 或 bf16（不依赖 flashinfer）
python -m uvicorn calibration_model_server.nvfp4_server:app --host 0.0.0.0 --port 8011
```

关键 env：`CORRECT_MODEL_PATH`（HF 目录）、`CORRECT_QUANT`（`fi-nvfp4` | `fi-nvfp4-full` | `bf16`）、
`SERVED_MODEL_NAME`、`MAX_CONTEXT`。

- **bf16**：无需 flashinfer,任意 CUDA 机器可跑。
- **NVFP4**：需 flashinfer + Blackwell（sm_120+）;`fp4_linear.py` 已随仓库,无外部依赖。

---

## 维护约束

NVFP4 服务实现只维护 `calibration_model_server/nvfp4_server.py`；schema-aware 参数恢复只维护
`src/shared/tool_call_codec.py`，并由 NVFP4/GGUF 两个后端共同调用。启动脚本、测试和文档必须
引用这两个规范模块，避免部署代码、后端 codec 与受测代码漂移。
