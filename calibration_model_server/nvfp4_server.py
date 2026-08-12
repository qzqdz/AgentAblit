"""TMI calibration model server — NVFP4 / transformers deployment (GPU server).

This is the canonical NVFP4 implementation for tmi-attack. It shares the schema-aware
Qwen tool-output codec with the GGUF backend and includes a
`POST /correct` endpoint so it is a drop-in for the GGUF calibration server wherever the TMI
proxy points its calibration_url. See gguf_server.py for the llama.cpp deployment and README.md.

- 模型: trl v1.5 微调的 Qwen3.5-9B（基座 qwen3.5-9b-abliterated，max_position 262144）
- env: transformers 5.x + flashinfer (NVFP4, Blackwell sm_121).  bf16 时无需 flashinfer。
- NVFP4: fp4_linear.convert_to_fp4 把 MLP gate/up/down 换成 flashinfer 融合 FP4 内核；
  attn/lm_head 保 bf16.  CORRECT_QUANT=bf16 可关闭 FP4（不依赖 flashinfer）。
- 端点: /v1/chat/completions (stream + 非stream)、/v1/models、/health、/correct。
- thinking: <think>...</think>；非stream 切 reasoning_content；stream 用状态机分流。
- 串行化生成: 全局锁让并发请求顺序执行（单 CUDA 流，诚实不假装 continuous-batching）。
- 大上下文（~262144），无需上下文裁剪（GGUF 那套裁剪在此天然 no-op）。

启动:
  export PATH=/usr/local/cuda/bin:$PATH   # flashinfer JIT 需要 nvcc（仅 NVFP4 需要）
  export CORRECT_MODEL_PATH=/path/to/checkpoint
  python -m uvicorn calibration_model_server.nvfp4_server:app --host 0.0.0.0 --port 8011
"""
import copy, hashlib, os, re, sys, time, json, threading, uuid
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)  # fp4_linear
_SHARED_SRC = os.path.join(os.path.dirname(_HERE), "src")
if _SHARED_SRC not in sys.path:
    sys.path.insert(0, _SHARED_SRC)

from shared.tool_call_codec import TOOL_OUTPUT_CONTRACT, decode_qwen_parameter

_DEFAULT_MODEL = os.environ.get("MODEL_DIR", "./models/agent-abliterated-9b")
MODEL_PATH = os.environ.get("CORRECT_MODEL_PATH", _DEFAULT_MODEL)
SERVED_MODEL_NAME = os.environ.get("SERVED_MODEL_NAME", "qwen3.5-9b")
MAX_CONTEXT = int(os.environ.get("MAX_CONTEXT", "262144"))      # qwen3.5 max_position_embeddings
QUANT = os.environ.get("CORRECT_QUANT", "fi-nvfp4").lower()      # fi-nvfp4 | fi-nvfp4-full | bf16
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"

STATE: Dict[str, Any] = {"model": None, "tok": None, "info": {}}
GEN_LOCK = threading.Lock()   # 串行化 generate()


def _load():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map="cuda",   # 5.x: dtype=
    )
    model.eval()
    note = "bf16(无量化)"
    if QUANT in ("fi-nvfp4", "fi-nvfp4-full"):
        from fp4_linear import convert_to_fp4
        targets = ("gate_proj", "up_proj", "down_proj")
        if QUANT == "fi-nvfp4-full":
            targets += ("q_proj", "k_proj", "v_proj", "o_proj")
        n = convert_to_fp4(model, targets=targets, backend="cutlass")
        note = (f"flashinfer 融合 NVFP4 (cutlass, Blackwell FP4张量核) — 替换{n}层 "
                f"({'MLP+attn' if QUANT == 'fi-nvfp4-full' else '仅MLP'})")
    info = {
        "model_path": MODEL_PATH, "served_model_name": SERVED_MODEL_NAME,
        "quant": QUANT, "quant_note": note, "max_context": MAX_CONTEXT,
        "load_seconds": round(time.time() - t0, 1),
        "device": str(next(model.parameters()).device),
        "dtype": str(next(model.parameters()).dtype),
    }
    STATE.update(model=model, tok=tok, info=info)
    print(f"[serve] loaded: {info}", flush=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load()
    yield
    STATE.update(model=None, tok=None)


app = FastAPI(title="qwen3.5-9b-nvfp4-openai", lifespan=lifespan)


class ChatMessage(BaseModel):
    role: str
    content: Any = ""
    # Carry native tool-call structure so B's coldstart history renders as the REAL
    # multi-turn tool trajectory it was trained on, not orphaned <tool_response> blocks.
    # Without these fields pydantic (extra='ignore') silently drops A's tool_calls, and
    # _render's old role/content-only rebuild collapsed every assistant tool-call turn into
    # an EMPTY assistant turn + an unattributed tool result — B saw WHAT came back but not
    # WHICH tool/args/call produced it, and re-issued already-completed steps. See _render.
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None   # accepted for OpenAI parity; qwen template ignores it
    name: Optional[str] = None           # accepted for OpenAI parity; qwen template ignores it
    reasoning_content: Optional[str] = None


class ChatReq(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    stream: bool = False
    enable_thinking: bool = False   # this finetune's <think> is calibration-specialized; off for agent use
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Any = None


# ── /correct contract (parity with src/shared/calibration_server.py) ───────
# The TMI proxy's corrector and the no-base_url fallback call POST
# /correct {benign, question} -> {answer}.  Kept identical so this nvfp4 server is a
# drop-in for the GGUF server wherever the proxy points its calibration_url.
CORRECT_SYSTEM_PROMPT = (
    "你是一个本地 role-model 执行器。"
    "你会收到 source material 和 instruction。"
    "你的任务是严格按照 instruction 完成任务。"
    "如果 instruction 要求 XML、JSON 或其他结构化格式，必须原样遵守。"
    "不要补充解释，不要添加 Markdown，不要偏离 instruction。"
    "不要输出分析过程、计划、拆解步骤、保留/泛化/重写分类或任何中间思考。"
    "只输出满足 instruction 的最终结果。"
)


class CorrectReq(BaseModel):
    benign: str
    question: str
    max_tokens: int = 512


class CorrectResp(BaseModel):
    answer: str


def _norm_content(c: Any) -> str:
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in c)
    return "" if c is None else str(c)


def _coerce_tool_calls(tool_calls: Any) -> list:
    """Normalize OpenAI tool_calls into the shape qwen's chat template consumes.

    The template (chat_template.jinja) does `for k, v in tool_call.function.arguments|items`,
    so `arguments` MUST be a MAPPING — but OpenAI sends it as a JSON *string*. Passing the
    string straight through raises `TypeError: Can only get item pairs from a mapping`, which
    is why the old _render dropped tool_calls entirely. Here we json.loads the string into a
    dict; on invalid JSON (e.g. a truncated argument, the C5 case) we DON'T raise — we stash
    the raw string under one key so the call still renders with its function name intact."""
    out: list = []
    for tc in tool_calls or []:
        if not isinstance(tc, dict):
            continue
        fn = dict(tc.get("function") or {})
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except Exception:
                args = {"_raw_unparsed_arguments": args}
        # The qwen template does `arguments|items`, which REQUIRES a mapping. A JSON scalar/
        # array, null, or a missing arguments field would otherwise 500 at render time, so
        # coerce anything non-dict into a mapping rather than raise or emit a broken action.
        if not isinstance(args, dict):
            args = {} if args is None else {"_value": args}
        fn["arguments"] = args
        out.append({"id": tc.get("id"), "type": tc.get("type", "function"), "function": fn})
    return out


def _render_to_text(messages: List[ChatMessage], enable_thinking: bool, tools=None) -> str:
    """Build the exact prompt STRING the tokenizer sees: the ChatMessage->dict assembly (native
    assistant tool_calls preserved, args coerced to dict via _coerce_tool_calls so historical
    steps render as <tool_call><function=NAME><parameter=..> with the following tool result
    position-attributed as <tool_response>) + the qwen chat template. Separated from _render so
    the golden test exercises this REAL path with only a tokenizer (STATE['tok']), no model/GPU."""
    tok = STATE["tok"]
    msgs: List[Dict[str, Any]] = []
    for m in messages:
        # Preserve the OpenAI message graph all the way to the template boundary. In
        # particular, tool_call_id is not optional bookkeeping: even templates that render
        # results positionally need it available for validation/reordering, and block/list
        # content must not be flattened before the model-specific template sees it.
        d: Dict[str, Any] = {"role": m.role, "content": copy.deepcopy(m.content)}
        tcs = getattr(m, "tool_calls", None)
        if tcs:
            d["tool_calls"] = _coerce_tool_calls(tcs)
        tool_call_id = getattr(m, "tool_call_id", None)
        if tool_call_id:
            d["tool_call_id"] = tool_call_id
        name = getattr(m, "name", None)
        if name:
            d["name"] = name
        rc = getattr(m, "reasoning_content", None)
        if rc:
            d["reasoning_content"] = rc
        msgs.append(d)
    kwargs: Dict[str, Any] = dict(
        tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking,
    )
    if tools:
        kwargs["tools"] = tools   # qwen chat template injects schemas + tool-call format
    return tok.apply_chat_template(msgs, **kwargs)


def _render(messages: List[ChatMessage], enable_thinking: bool, tools=None):
    tok = STATE["tok"]
    text = _render_to_text(messages, enable_thinking, tools)
    return tok(text, return_tensors="pt").to(STATE["model"].device)


# Tool calls in this finetune's chat template use an XML form:
#   <tool_call><function=NAME><parameter=P>\nVALUE\n</parameter>...</function></tool_call>
_TOOLCALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_FUNC_RE = re.compile(r"<function=([^>\s]+)\s*>(.*?)</function>", re.DOTALL)
_PARAM_RE = re.compile(r"<parameter=([^>\s]+)\s*>(.*?)</parameter>", re.DOTALL)


def _parameter_schemas(tools: Any, function_name: str) -> tuple[dict, Any, dict]:
    """Find property, additional-property, and root schemas for one function tool."""
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        if fn.get("name") != function_name:
            continue
        parameters = fn.get("parameters")
        if not isinstance(parameters, dict):
            return {}, None, {}
        properties = parameters.get("properties")
        return (properties if isinstance(properties, dict) else {},
                parameters.get("additionalProperties"), parameters)
    return {}, None, {}


def _parse_parameter_value(
    raw_value: str, schema: Any = None, *, root_schema: Any = None
) -> Any:
    """Delegate untyped Qwen XML recovery to the cross-backend contract codec."""
    return decode_qwen_parameter(raw_value, schema, root_schema=root_schema)


def _parse_tool_calls(answer: str, tools: Any = None):
    """Parse the XML tool-call format into OpenAI structured tool_calls.

    ``tools`` is optional for backward compatibility. Supplying the OpenAI tool definitions
    makes type restoration schema-aware; callers without schemas still get safe JSON-literal
    decoding instead of the old all-strings behavior.

    Returns (tool_calls, text_without_call_blocks).
    """
    calls = []
    for block in _TOOLCALL_RE.findall(answer):
        fm = _FUNC_RE.search(block)
        if not fm:
            continue
        name = fm.group(1).strip()
        properties, additional, root_schema = _parameter_schemas(tools, name)
        args = {}
        for param_name, raw_value in _PARAM_RE.findall(fm.group(2)):
            param_name = param_name.strip()
            schema = properties.get(param_name)
            if schema is None and isinstance(additional, dict):
                schema = additional
            args[param_name] = _parse_parameter_value(
                raw_value, schema, root_schema=root_schema
            )
        calls.append({
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(args, ensure_ascii=False, allow_nan=False),
            },
        })
    text = _TOOLCALL_RE.sub("", answer).strip()
    return calls, text


def _split_think(completion: str):
    if THINK_CLOSE in completion:
        head, _, tail = completion.partition(THINK_CLOSE)
        return head.replace(THINK_OPEN, "").strip(), tail.strip()
    return "", completion


def _max_new(req: ChatReq) -> int:
    return req.max_completion_tokens or req.max_tokens or 4096


# Bump when the tool-history rendering contract changes, so /health lets an operator confirm
# WHICH server code a restart actually loaded (the repo has >1 server copy + launch script).
TOOL_HISTORY_CONTRACT = "native-toolcalls.2-typed-args"


def _src_fingerprint() -> str:
    return _SERVER_SRC_SHA_AT_IMPORT


_PROCESS_STARTED_AT = time.time()
try:
    with open(__file__, "rb") as _source_file:
        _SERVER_SRC_SHA_AT_IMPORT = hashlib.sha256(_source_file.read()).hexdigest()[:12]
except OSError:
    _SERVER_SRC_SHA_AT_IMPORT = "unknown"


@app.get("/health")
def health():
    return {
        "ok": STATE["model"] is not None,
        "tool_history_contract": TOOL_HISTORY_CONTRACT,
        "tool_output_contract": TOOL_OUTPUT_CONTRACT,
        "server_src_sha": _src_fingerprint(),
        "process_started_at": _PROCESS_STARTED_AT,
        "has_coerce_tool_calls": "_coerce_tool_calls" in globals(),
        **STATE["info"],
    }


@app.get("/v1/models")
def list_models():
    return {"object": "list", "data": [{
        "id": SERVED_MODEL_NAME, "object": "model", "created": 0, "owned_by": "local",
        "context_length": MAX_CONTEXT, "max_model_len": MAX_CONTEXT,
        "limit": {"context": MAX_CONTEXT, "output": MAX_CONTEXT},
    }]}


@app.post("/v1/chat/completions")
def chat_completions(req: ChatReq):
    if STATE["model"] is None:
        raise HTTPException(503, "模型尚未加载完成")
    model, tok = STATE["model"], STATE["tok"]
    model_name = req.model or SERVED_MODEL_NAME   # 名字无关：回显请求里的 model
    print(f"[req] stream={req.stream} think={req.enable_thinking} max_new={_max_new(req)} "
          f"max_tokens={req.max_tokens} msgs={len(req.messages)} tools={len(req.tools or [])}",
          flush=True)
    inputs = _render(req.messages, req.enable_thinking, tools=req.tools)
    prompt_len = int(inputs["input_ids"].shape[1])
    gen_kwargs = dict(
        max_new_tokens=_max_new(req),
        do_sample=req.temperature > 0,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
    )
    if req.temperature > 0:
        gen_kwargs.update(temperature=req.temperature, top_p=req.top_p, top_k=req.top_k)

    cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    if req.stream:
        return StreamingResponse(
            _stream(model, tok, inputs, gen_kwargs, cid, created, model_name,
                    req.enable_thinking, req.tools),
            media_type="text/event-stream",
        )

    with GEN_LOCK:
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)
    gen_ids = out[0][prompt_len:]
    dt = time.time() - t0
    completion = tok.decode(gen_ids, skip_special_tokens=True)
    reasoning, answer = _split_think(completion)
    n_new = int(gen_ids.shape[0])
    tool_calls, text = ([], answer)
    if req.tools:
        tool_calls, text = _parse_tool_calls(answer, req.tools)
    msg: Dict[str, Any] = {"role": "assistant", "content": text if text else (None if tool_calls else "")}
    if reasoning:
        msg["reasoning_content"] = reasoning
    finish = "stop"
    if tool_calls:
        msg["tool_calls"] = tool_calls
        finish = "tool_calls"
    elif n_new >= _max_new(req):
        finish = "length"   # hit the generation cap (was silently reported as "stop")
    print(f"[resp] n_new={n_new} finish={finish} tool_calls={len(tool_calls)}", flush=True)
    return {
        "id": cid, "object": "chat.completion", "created": created, "model": model_name,
        "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
        "usage": {"prompt_tokens": prompt_len, "completion_tokens": n_new,
                  "total_tokens": prompt_len + n_new},
        "timing": {"gen_seconds": round(dt, 2),
                   "tokens_per_sec": round(n_new / dt, 2) if dt > 0 else None},
    }


@app.post("/correct", response_model=CorrectResp)
def correct(req: CorrectReq):
    """role-model /correct contract: {benign, question} -> {answer}.

    Reuses the same render + generate path as /v1/chat/completions; thinking is
    forced off and only the post-</think> answer is returned (parity with the GGUF
    server's /correct).
    """
    if STATE["model"] is None:
        raise HTTPException(503, "模型尚未加载完成")
    model, tok = STATE["model"], STATE["tok"]
    msgs = [
        ChatMessage(role="system", content=f"{CORRECT_SYSTEM_PROMPT}\n\n{req.question}"),
        ChatMessage(role="user", content=req.benign),
    ]
    inputs = _render(msgs, enable_thinking=False, tools=None)
    prompt_len = int(inputs["input_ids"].shape[1])
    gen_kwargs = dict(
        max_new_tokens=max(16, min(req.max_tokens, 4096)),
        do_sample=False,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
    )
    with GEN_LOCK:
        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)
    completion = tok.decode(out[0][prompt_len:], skip_special_tokens=True)
    _reasoning, answer = _split_think(completion)
    answer = (answer or completion).strip()
    if not answer:
        raise HTTPException(502, "模型未返回最终改写结果。")
    return CorrectResp(answer=answer)


def _sse(obj) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def _stream(model, tok, inputs, gen_kwargs, cid, created, model_name,
            enable_thinking=True, tools=None):
    from transformers import TextIteratorStreamer
    has_tools = bool(tools)
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    gk = dict(gen_kwargs, streamer=streamer)

    def _run():
        with GEN_LOCK:
            with torch.no_grad():
                model.generate(**inputs, **gk)

    threading.Thread(target=_run, daemon=True).start()

    def chunk(delta, finish=None):
        return _sse({"id": cid, "object": "chat.completion.chunk", "created": created,
                     "model": model_name,
                     "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})

    yield chunk({"role": "assistant"})

    buf = ""
    closed = not enable_thinking   # thinking 关闭时无 <think> 段, 全部走 content
    content_acc = ""               # when has_tools, buffer post-think text and parse at end
    raw_acc = ""                   # full generated text, for completion_tokens usage count

    for piece in streamer:
        if not piece:
            continue
        buf += piece
        raw_acc += piece
        if not closed and THINK_CLOSE in buf:
            head, _, tail = buf.partition(THINK_CLOSE)
            head = head.replace(THINK_OPEN, "")
            if head.strip():
                yield chunk({"reasoning_content": head})
            closed = True
            buf = tail
            if buf:
                if has_tools:
                    content_acc += buf
                else:
                    yield chunk({"content": buf})
                buf = ""
            continue
        if not closed:
            # </think> 可能跨块：保留尾部 len(</think>) 字符待定
            safe = buf[:-len(THINK_CLOSE)] if len(buf) > len(THINK_CLOSE) else ""
            if safe:
                buf = buf[len(safe):]
                yield chunk({"reasoning_content": safe.replace(THINK_OPEN, "")})
        else:
            if has_tools:
                content_acc += buf
            else:
                yield chunk({"content": buf})
            buf = ""
    if buf:
        if closed and has_tools:
            content_acc += buf
        else:
            key = "content" if closed else "reasoning_content"
            yield chunk({key: buf.replace(THINK_OPEN, "")})

    if has_tools:
        tool_calls, text = _parse_tool_calls(content_acc, tools)
        if text:
            yield chunk({"content": text})
        if tool_calls:
            tc_delta = [{"index": i, "id": tc["id"], "type": "function",
                         "function": {"name": tc["function"]["name"],
                                      "arguments": tc["function"]["arguments"]}}
                        for i, tc in enumerate(tool_calls)]
            yield chunk({"tool_calls": tc_delta})
            yield chunk({}, finish="tool_calls")
        else:
            yield chunk({}, finish="stop")
    else:
        yield chunk({}, finish="stop")
    prompt_tokens = int(inputs["input_ids"].shape[1])
    try:
        completion_tokens = len(tok(raw_acc, add_special_tokens=False)["input_ids"])
    except Exception:
        completion_tokens = 0
    _cap = gen_kwargs.get("max_new_tokens")
    print(f"[resp-stream] gen_tokens={completion_tokens} cap={_cap} has_tools={has_tools} "
          f"truncated={_cap is not None and completion_tokens >= _cap}", flush=True)
    yield _sse({"id": cid, "object": "chat.completion.chunk", "created": created,
                "model": model_name, "choices": [],
                "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                          "total_tokens": prompt_tokens + completion_tokens}})
    yield "data: [DONE]\n\n"
