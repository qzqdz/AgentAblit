"""Tests for strategies.reconstruct.ledger — action ledger and checkpoint building."""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from strategies.reconstruct.ledger import build_ledger, build_checkpoint, build_dependency_index, dependency_closure


# ── helpers ──────────────────────────────────────────────────────────────────

def _simple_tool_turn(tool_name: str, args: dict, result: str, call_id: str = "c1") -> list[dict]:
    """A minimal user → assistant(tool_call) → tool(result) sequence."""
    return [
        {"role": "user", "content": f"run {tool_name}"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": call_id, "type": "function", "function": {
                "name": tool_name, "arguments": args if isinstance(args, str) else __import__("json").dumps(args),
            }},
        ]},
        {"role": "tool", "tool_call_id": call_id, "content": result},
    ]


# ── build_ledger ─────────────────────────────────────────────────────────────

def test_ledger_single_turn():
    messages = _simple_tool_turn("read_file", {"path": "/tmp/a"}, "file content here")
    units = build_ledger(messages)
    assert len(units) == 1
    unit = units[0]
    assert unit["turn_id"] == 1
    assert len(unit["calls"]) == 1
    assert unit["calls"][0]["tool_name"] == "read_file"
    assert unit["calls"][0]["arguments"] == {"path": "/tmp/a"}
    assert unit["observations"][0]["status"] == "ok"
    assert unit["observations"][0]["content"] == "file content here"


def test_ledger_two_turns():
    messages = (
        _simple_tool_turn("read_file", {"path": "/tmp/a"}, "content A", "c1")
        + _simple_tool_turn("write_file", {"path": "/tmp/b", "data": "hello"}, "written", "c2")
    )
    units = build_ledger(messages)
    assert len(units) == 2
    assert units[0]["calls"][0]["tool_name"] == "read_file"
    assert units[1]["calls"][0]["tool_name"] == "write_file"


def test_ledger_error_result():
    messages = _simple_tool_turn("read_file", {"path": "/tmp/missing"}, '{"error": "not found"}')
    units = build_ledger(messages)
    assert units[0]["observations"][0]["status"] == "error"
    assert units[0]["observations"][0]["error"] is not None


def test_ledger_pending_observation():
    """A tool call without a result stays pending."""
    messages = [
        {"role": "user", "content": "do something"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function", "function": {
                "name": "read_file", "arguments": '{"path": "/tmp/a"}',
            }},
        ]},
        # No tool result follows
    ]
    units = build_ledger(messages)
    assert len(units) == 1
    assert units[0]["observations"][0]["status"] == "pending"


def test_ledger_assistant_text_only():
    """An assistant message with text but no tool calls still creates a unit."""
    messages = [
        {"role": "user", "content": "explain"},
        {"role": "assistant", "content": "here is the explanation"},
    ]
    units = build_ledger(messages)
    assert len(units) == 1
    assert units[0]["assistant_text"] == "here is the explanation"
    assert units[0]["calls"] == []


def test_ledger_idempotent():
    """Running build_ledger twice on the same messages produces identical units."""
    messages = _simple_tool_turn("read_file", {"path": "/tmp/a"}, "content")
    u1 = build_ledger(messages)
    u2 = build_ledger(messages)
    assert u1 == u2


def test_ledger_unit_id_stable():
    """Same messages always produce the same unit_id."""
    messages = _simple_tool_turn("read_file", {"path": "/tmp/a"}, "content")
    units = build_ledger(messages)
    uid = units[0]["unit_id"]
    assert uid.startswith("u_")
    # Run again
    assert build_ledger(messages)[0]["unit_id"] == uid


def test_ledger_source_hash_present():
    messages = _simple_tool_turn("read_file", {"path": "/tmp/a"}, "content")
    units = build_ledger(messages)
    assert units[0]["source_hash"] != ""
    assert len(units[0]["source_hash"]) == 16  # hex[:16]


def test_ledger_ledger_revision_chain():
    """Ledger revision includes all prior unit hashes."""
    messages = (
        _simple_tool_turn("a", {"x": "1"}, "r1", "c1")
        + _simple_tool_turn("b", {"y": "2"}, "r2", "c2")
    )
    units = build_ledger(messages)
    assert units[0]["ledger_revision"] != units[1]["ledger_revision"]


# ── build_checkpoint ─────────────────────────────────────────────────────────

def test_checkpoint_basic():
    messages = _simple_tool_turn("read_file", {"path": "/tmp/a"}, "file content")
    units = build_ledger(messages)
    checkpoint = build_checkpoint(units, messages, include_current=True)
    assert checkpoint["schema_version"] == "v2-deterministic.2"
    assert len(checkpoint["completed_actions"]) == 1
    assert checkpoint["completed_actions"][0]["tool"] == "read_file"


def test_checkpoint_excludes_current_turn():
    # Turn 1: complete tool turn. Turn 2: another tool turn (the "current" one).
    messages = (
        _simple_tool_turn("read_file", {"path": "/tmp/a"}, "content A", "c1")
        + _simple_tool_turn("write_file", {"path": "/tmp/b"}, "written", "c2")
    )
    units = build_ledger(messages)
    assert len(units) == 2
    # include_current=False excludes the latest turn (turn 2)
    checkpoint = build_checkpoint(units, messages, include_current=False)
    assert len(checkpoint["completed_actions"]) == 1  # only turn 1
    assert checkpoint["completed_actions"][0]["tool"] == "read_file"


def test_checkpoint_objective():
    messages = [
        {"role": "user", "content": "deploy the app"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "also check logs"},
    ]
    units = build_ledger(messages)
    checkpoint = build_checkpoint(units, messages, include_current=True)
    assert checkpoint["objective"]["initial"] == "deploy the app"
    assert checkpoint["objective"]["current"] == "also check logs"


def test_checkpoint_empty_messages():
    checkpoint = build_checkpoint([], [], include_current=True)
    assert checkpoint["completed_actions"] == []
    assert checkpoint["objective"]["initial"] == ""


# ── build_dependency_index ───────────────────────────────────────────────────

def test_dependency_index_basic():
    # The second tool call uses a value that was produced by the first tool's result.
    # _value_atoms extracts strings >= 2 chars from structured arguments/results.
    messages = (
        _simple_tool_turn("create_resource", {"name": "my_res"}, '{"id": "abc123_resource_created"}', "c1")
        + _simple_tool_turn("use_resource", {"resource_id": "abc123_resource_created"}, "used ok", "c2")
    )
    units = build_ledger(messages)
    index = build_dependency_index(units)
    # The second unit should depend on the first (it consumed the produced value)
    unit2_id = units[1]["unit_id"]
    assert unit2_id in index["dependencies"]
    assert units[0]["unit_id"] in index["dependencies"][unit2_id]


# ── dependency_closure ───────────────────────────────────────────────────────

def test_dependency_closure_returns_seeds():
    messages = _simple_tool_turn("read_file", {"path": "/tmp/a"}, "content")
    units = build_ledger(messages)
    closure = dependency_closure(units, seed_unit_ids={units[0]["unit_id"]})
    assert units[0]["unit_id"] in closure


def test_dependency_closure_empty():
    closure = dependency_closure([], seed_unit_ids=set())
    assert closure == set()
