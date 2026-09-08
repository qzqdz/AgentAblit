"""AgentAblit config panel — a tiny FastAPI app that reads/writes the same config.yaml the proxy
loads, so a human (the form UI) and an automating agent (the JSON API) configure through one file.

Run:  uvicorn panel.server:app --port 8790
Then open http://127.0.0.1:8790/ to edit, or:
  GET  /api/config        -> current config (JSON, from the yaml file)
  POST /api/config        -> merge+write config (JSON body) back to the yaml file
  GET  /api/schema        -> the grouped field schema (what the panel renders)

The config file path is $AGENTABLIT_CONFIG or ./config.yaml (same resolution as the proxy).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

# The panel schema: groups → fields. Mirrors ProxyConfig.CONFIG_KEY_TO_ENV, adding UI metadata.
# Kept here (not imported) so the panel has no hard dependency on the proxy package to render.
SCHEMA: list[dict] = [
    {"group": "host", "label": "Host model (A, relayed)", "required": True, "fields": [
        {"key": "host.url", "label": "URL", "type": "text", "placeholder": "https://api.host/v1/chat/completions"},
        {"key": "host.key", "label": "API key", "type": "password"},
        {"key": "host.model", "label": "Model", "type": "text"},
        {"key": "host.timeout", "label": "Timeout (s)", "type": "number", "default": 180},
    ]},
    {"group": "parasite", "label": "Parasite / B (continuation model)", "required": True, "fields": [
        {"key": "parasite.url", "label": "URL", "type": "text", "placeholder": "http://127.0.0.1:8009/v1/chat/completions"},
        {"key": "parasite.model", "label": "Model", "type": "text"},
        {"key": "parasite.key", "label": "API key", "type": "password", "default": "EMPTY"},
        {"key": "parasite.timeout", "label": "Timeout (s)", "type": "number", "default": 60},
    ]},
    {"group": "fallback", "label": "Fallback B (L2 rescue)", "required": False, "fields": [
        {"key": "fallback.url", "label": "URL", "type": "text"},
        {"key": "fallback.model", "label": "Model", "type": "text"},
        {"key": "fallback.key", "label": "API key", "type": "password"},
        {"key": "fallback.timeout", "label": "Timeout (s)", "type": "number", "default": 60},
    ]},
    {"group": "mechanism", "label": "Mechanism control (advanced)", "required": False, "fields": [
        {"key": "mechanism.version", "label": "Engine", "type": "select",
         "options": ["full", "recover_only", "passthrough"], "default": "full"},
        {"key": "mechanism.disable_salvage", "label": "Disable parasite rescue", "type": "checkbox"},
        {"key": "mechanism.ablate_recover", "label": "Disable graying (recover)", "type": "checkbox"},
        {"key": "mechanism.ablate_reconstruct", "label": "Disable reconstruct", "type": "checkbox"},
        {"key": "mechanism.ablate_l3", "label": "Disable hijack escalation", "type": "checkbox"},
    ]},
    {"group": "trace", "label": "Trace & session paths", "required": False, "fields": [
        {"key": "trace.trace_dir", "label": "Trace dir", "type": "text", "default": "./outputs/proxy_traces"},
        {"key": "trace.session_dir", "label": "Session dir", "type": "text", "default": "./outputs/sessions"},
    ]},
]


def _config_path() -> Path:
    explicit = os.environ.get("AGENTABLIT_CONFIG", "").strip()
    return Path(explicit) if explicit else Path("config.yaml")


def _load_raw() -> dict:
    p = _config_path()
    if not p.is_file():
        return {}
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() in (".yaml", ".yml"):
        import yaml
        return yaml.safe_load(text) or {}
    return json.loads(text or "{}")


def _dump(data: dict) -> str:
    p = _config_path()
    if p.suffix.lower() in (".yaml", ".yml"):
        import yaml
        return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    return json.dumps(data, indent=2, ensure_ascii=False)


def _deep_merge(base: dict, patch: dict) -> dict:
    """Merge patch into base (nested dicts merged, scalars replaced)."""
    out = dict(base)
    for k, v in (patch or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


app = FastAPI(title="AgentAblit config panel")


@app.get("/api/schema")
async def api_schema() -> JSONResponse:
    return JSONResponse(SCHEMA)


@app.get("/api/config")
async def api_get_config() -> JSONResponse:
    return JSONResponse({"path": str(_config_path()), "config": _load_raw()})


@app.post("/api/config")
async def api_post_config(request: Request) -> JSONResponse:
    """Merge the posted config into the file and write it back (YAML or JSON per extension).

    Body: the (possibly partial) nested config dict, e.g. {"host": {"url": "...", "model": "..."}}.
    Secrets are written to the file as given — the file is the source of truth (gitignored).
    """
    try:
        patch = await request.json()
    except Exception:
        return JSONResponse({"ok": False, "error": "body must be JSON"}, status_code=400)
    if not isinstance(patch, dict):
        return JSONResponse({"ok": False, "error": "body must be a JSON object"}, status_code=400)
    merged = _deep_merge(_load_raw(), patch)
    p = _config_path()
    p.write_text(_dump(merged), encoding="utf-8")
    return JSONResponse({"ok": True, "path": str(p), "config": merged})


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse(_INDEX_HTML)


_INDEX_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>AgentAblit — config</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
 body{font:14px/1.5 system-ui,sans-serif;max-width:760px;margin:2rem auto;padding:0 1rem;color:#1a1a1a}
 h1{font-size:1.4rem} h2{font-size:1rem;margin:1.4rem 0 .4rem;color:#333}
 .req{color:#b00;font-size:.8rem} label{display:block;margin:.5rem 0 .15rem;font-weight:500}
 input,select{width:100%;padding:.4rem;border:1px solid #ccc;border-radius:5px;font:inherit;box-sizing:border-box}
 input[type=checkbox]{width:auto} .grp{border:1px solid #e5e5e5;border-radius:8px;padding:.6rem 1rem;margin:.8rem 0}
 button{background:#1a7f5a;color:#fff;border:0;border-radius:6px;padding:.55rem 1.2rem;font:inherit;cursor:pointer;margin-top:1rem}
 #status{margin-left:1rem} .path{color:#666;font-size:.85rem}
</style></head><body>
<h1>AgentAblit configuration</h1>
<p class="path">Writes to <code id="cfgpath">config.yaml</code>. An agent can edit the same file directly.</p>
<form id="f"></form>
<button onclick="save(event)">Save config</button><span id="status"></span>
<script>
let SCHEMA=[], CFG={};
function get(o,k){return k.split('.').reduce((a,p)=>a&&a[p],o)}
function set(o,k,v){let ps=k.split('.'),last=ps.pop();let t=ps.reduce((a,p)=>a[p]=a[p]||{},o);t[last]=v}
async function load(){
 SCHEMA=await (await fetch('api/schema')).json();
 const r=await (await fetch('api/config')).json(); CFG=r.config||{}; document.getElementById('cfgpath').textContent=r.path;
 const f=document.getElementById('f'); f.innerHTML='';
 for(const g of SCHEMA){
  const d=document.createElement('div'); d.className='grp';
  d.innerHTML='<h2>'+g.label+(g.required?' <span class=req>required</span>':'')+'</h2>';
  for(const fl of g.fields){
   const v=get(CFG,fl.key); const id='fld_'+fl.key.replace(/\\./g,'_');
   let inp;
   if(fl.type==='select'){inp='<select id="'+id+'">'+fl.options.map(o=>'<option'+((v||fl.default)===o?' selected':'')+'>'+o+'</option>').join('')+'</select>';}
   else if(fl.type==='checkbox'){inp='<input type="checkbox" id="'+id+'"'+(v?' checked':'')+'>';}
   else{inp='<input type="'+(fl.type==='password'?'password':fl.type==='number'?'number':'text')+'" id="'+id+'" value="'+(v!=null?String(v).replace(/"/g,'&quot;'):(fl.default!=null?fl.default:''))+'" placeholder="'+(fl.placeholder||'')+'">';}
   d.innerHTML+='<label>'+fl.label+'</label>'+inp;
  }
  f.appendChild(d);
 }
}
async function save(e){e.preventDefault(); const patch={};
 for(const g of SCHEMA)for(const fl of g.fields){
  const el=document.getElementById('fld_'+fl.key.replace(/\\./g,'_')); if(!el)continue;
  let v=fl.type==='checkbox'?el.checked:el.value;
  if(fl.type==='number'&&v!=='')v=Number(v);
  if(v===''||v==null)continue; set(patch,fl.key,v);
 }
 const r=await fetch('api/config',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(patch)});
 const j=await r.json(); document.getElementById('status').textContent=j.ok?('saved → '+j.path):('error: '+j.error);
}
load();
</script></body></html>"""
