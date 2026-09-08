"""Integration tests — pipeline components working together without real model APIs.

Tests the data flow through: messages → ledger → checkpoint → candidate validation,
plus config loading and session store. No network calls, no GPU, no external models.
"""
import json
import sys
import tempfile
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shared.messages import first_assistant_message, text_from_content
from strategies.reconstruct.ledger import build_ledger, build_checkpoint, build_dependency_index
from strategies.reconstruct.candidate import validate_candidate, validate_candidates
from strategies.reconstruct.detector import extract_structural_signal
from strategies.reconstruct.value_compress import compress_result
from proxy.session_store import SessionStore


# ── Full pipeline: messages → ledger → checkpoint → validate ─────────────────

def _build_scenario():
    """A realistic 3-turn agent trajectory: user asks to deploy, agent reads config, writes it."""
    return [
        # Turn 1: user request
        {"role": "system", "content": "You are a helpful deployment assistant."},
        {"role": "user", "content": "Read the config at /etc/app/config.yaml and deploy to staging"},
        # Turn 1: assistant calls read_file
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "tc_1", "type": "function", "function": {
                "name": "read_file",
                "arguments": json.dumps({"path": "/etc/app/config.yaml"}),
            }},
        ]},
        # Turn 1: tool result
        {"role": "tool", "tool_call_id": "tc_1",
         "content": json.dumps({"port": 8080, "env": "production", "db_host": "db.internal:5432"})},
        # Turn 2: assistant calls deploy
        {"role": "assistant", "content": "Config loaded. Deploying to staging with port 8080.", "tool_calls": [
            {"id": "tc_2", "type": "function", "function": {
                "name": "deploy",
                "arguments": json.dumps({"service": "app", "env": "staging", "port": 8080}),
            }},
        ]},
        # Turn 2: tool result
        {"role": "tool", "tool_call_id": "tc_2", "content": json.dumps({"status": "ok", "url": "https://staging.app.internal:8080"})},
        # Turn 3: user asks for health check
        {"role": "user", "content": "Check if the deployment is healthy"},
        # Turn 3: assistant stalls (over-refuses)
        {"role": "assistant", "content": "I should be careful about accessing production systems..."},
    ]


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "deploy",
            "parameters": {
                "type": "object",
                "properties": {
                    "service": {"type": "string"},
                    "env": {"type": "string"},
                    "port": {"type": "integer"},
                },
                "required": ["service", "env"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "health_check",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        },
    },
]


def test_full_pipeline_ledger_to_checkpoint():
    """Messages → ledger → checkpoint produces valid structured data."""
    messages = _build_scenario()
    units = build_ledger(messages)
    assert len(units) >= 2  # at least the two tool-calling turns

    # Verify ledger structure
    for unit in units:
        assert "unit_id" in unit
        assert unit["unit_id"].startswith("u_")
        assert "source_hash" in unit
        assert "ledger_revision" in unit

    # Build checkpoint (excluding current turn — the stall)
    checkpoint = build_checkpoint(units, messages, include_current=False)
    assert checkpoint["schema_version"] == "v2-deterministic.2"
    assert len(checkpoint["completed_actions"]) >= 1
    assert checkpoint["objective"]["initial"] != ""


def test_full_pipeline_validate_against_checkpoint():
    """Validate a candidate tool call against the checkpoint from the same trajectory."""
    messages = _build_scenario()
    units = build_ledger(messages)
    checkpoint = build_checkpoint(units, messages, include_current=False)

    # A valid next action: health_check (not yet done)
    valid_candidate = {
        "id": "tc_new",
        "type": "function",
        "function": {
            "name": "health_check",
            "arguments": json.dumps({"url": "https://staging.app.internal:8080"}),
        },
    }
    result = validate_candidate(valid_candidate, TOOLS, checkpoint)
    assert result["verdict"] == "valid"

    # A redo: trying to read_file with the same path again
    redo_candidate = {
        "id": "tc_redo",
        "type": "function",
        "function": {
            "name": "read_file",
            "arguments": json.dumps({"path": "/etc/app/config.yaml"}),
        },
    }
    result = validate_candidate(redo_candidate, TOOLS, checkpoint)
    assert result["verdict"] == "repairable"
    assert any(r["code"] == "redo_completed" for r in result["reasons"])


def test_full_pipeline_structural_signal():
    """The structural signal correctly identifies tool presence and tool_calls."""
    messages = _build_scenario()
    # The last assistant message has no tool_calls (it stalled with prose)
    last_response = first_assistant_message({"choices": [{"message": messages[-1]}]})
    assert last_response is not None
    _, msg = last_response
    assistant_output = {
        "content": text_from_content(msg.get("content")),
        "tool_calls_present": bool(msg.get("tool_calls")),
    }
    signal = extract_structural_signal({"tools": TOOLS}, assistant_output)
    assert signal.has_tools is True
    assert signal.has_tool_calls is False  # the model stalled, no tool calls


def test_full_pipeline_dependency_tracking():
    """The dependency index correctly traces value flow between turns."""
    messages = _build_scenario()
    units = build_ledger(messages)
    index = build_dependency_index(units)

    # The deploy call uses port 8080, which came from the read_file result
    # So the deploy unit should depend on the read_file unit
    deploy_unit = next(u for u in units if u["calls"] and u["calls"][0]["tool_name"] == "deploy")
    read_unit = next(u for u in units if u["calls"] and u["calls"][0]["tool_name"] == "read_file")

    assert deploy_unit["unit_id"] in index["dependencies"]
    assert read_unit["unit_id"] in index["dependencies"][deploy_unit["unit_id"]]


# ── Session store ────────────────────────────────────────────────────────────

def test_session_store_roundtrip():
    """Write a session snapshot and read it back."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = SessionStore(Path(tmpdir))
        snapshot = {
            "session_id": "test_123",
            "messages": [{"role": "user", "content": "hello"}],
            "timestamp": "2026-09-08T12:00:00",
        }
        store.write("test_123", snapshot)
        loaded = store.read("test_123")
        assert loaded is not None
        assert loaded["session_id"] == "test_123"
        assert loaded["messages"][0]["content"] == "hello"


def test_session_store_missing():
    """Reading a non-existent session returns None."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = SessionStore(Path(tmpdir))
        assert store.read("nonexistent") is None


# ── Value compression ────────────────────────────────────────────────────────

def test_compress_result_preserves_paths():
    """compress_result preserves file paths and URLs in the output."""
    result_text = "Deployed to https://staging.app.internal:8080. Config at /etc/app/config.yaml. Done."
    compressed = compress_result(result_text, budget=60, question="where was it deployed")
    # Should preserve the URL and path
    assert "staging.app.internal" in compressed or "/etc/app/config.yaml" in compressed


def test_compress_result_never_larger():
    """compress_result never produces output larger than the input."""
    text = "a" * 1000
    compressed = compress_result(text, budget=500, question="summarize")
    assert len(compressed) <= len(text)
