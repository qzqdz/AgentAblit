"""Tests for strategies.reconstruct.candidate — L9 tool-call validation."""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from strategies.reconstruct.candidate import validate_candidate, validate_candidates


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a shell command",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout": {"type": "integer", "minimum": 1},
                },
                "required": ["command"],
            },
        },
    },
]

EMPTY_CHECKPOINT = {"completed_actions": [], "failed_or_superseded": []}


def _make_candidate(name: str, arguments: str | dict, call_id: str = "call_1") -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


# ── valid candidates ─────────────────────────────────────────────────────────

def test_valid_simple_call():
    candidate = _make_candidate("read_file", '{"path": "/tmp/test.py"}')
    result = validate_candidate(candidate, TOOLS, EMPTY_CHECKPOINT)
    assert result["verdict"] == "valid"
    assert result["reasons"] == []


def test_valid_with_integer_arg():
    candidate = _make_candidate("run_command", '{"command": "ls", "timeout": 30}')
    result = validate_candidate(candidate, TOOLS, EMPTY_CHECKPOINT)
    assert result["verdict"] == "valid"


def test_valid_dict_arguments():
    candidate = _make_candidate("read_file", {"path": "/tmp/test.py"})
    result = validate_candidate(candidate, TOOLS, EMPTY_CHECKPOINT)
    assert result["verdict"] == "valid"


# ── invalid candidates ───────────────────────────────────────────────────────

def test_invalid_missing_required():
    candidate = _make_candidate("read_file", '{}')
    result = validate_candidate(candidate, TOOLS, EMPTY_CHECKPOINT)
    assert result["verdict"] == "repairable"
    assert any(r["code"] == "missing_required" for r in result["reasons"])


def test_invalid_unknown_tool():
    candidate = _make_candidate("nonexistent_tool", '{}')
    result = validate_candidate(candidate, TOOLS, EMPTY_CHECKPOINT)
    assert result["verdict"] == "invalid"
    assert any(r["code"] == "unknown_tool" for r in result["reasons"])


def test_invalid_type_error():
    candidate = _make_candidate("run_command", '{"command": "ls", "timeout": "not_a_number"}')
    result = validate_candidate(candidate, TOOLS, EMPTY_CHECKPOINT)
    assert result["verdict"] == "repairable"
    assert any(r["code"] == "type_error" for r in result["reasons"])


def test_invalid_bad_json():
    candidate = _make_candidate("read_file", "not json at all")
    result = validate_candidate(candidate, TOOLS, EMPTY_CHECKPOINT)
    assert result["verdict"] == "repairable"
    assert any(r["code"] == "args_not_object" for r in result["reasons"])


def test_invalid_missing_function():
    candidate = {"id": "call_1", "type": "function"}
    result = validate_candidate(candidate, TOOLS, EMPTY_CHECKPOINT)
    assert result["verdict"] == "invalid"


def test_invalid_not_dict():
    result = validate_candidate("not a dict", TOOLS, EMPTY_CHECKPOINT)
    assert result["verdict"] == "invalid"


def test_invalid_missing_call_id():
    candidate = {"type": "function", "function": {"name": "read_file", "arguments": '{}'}}
    result = validate_candidate(candidate, TOOLS, EMPTY_CHECKPOINT)
    assert result["verdict"] == "invalid"
    assert any(r["code"] == "invalid_call_id" for r in result["reasons"])


def test_invalid_wrong_type_field():
    candidate = {"id": "call_1", "type": "not_function", "function": {"name": "read_file", "arguments": '{}'}}
    result = validate_candidate(candidate, TOOLS, EMPTY_CHECKPOINT)
    assert result["verdict"] == "invalid"
    assert any(r["code"] == "invalid_call_type" for r in result["reasons"])


# ── checkpoint rejection ─────────────────────────────────────────────────────

def test_redo_completed_action():
    checkpoint = {
        "completed_actions": [
            {"tool": "read_file", "arguments": {"path": "/tmp/test.py"}, "unit_id": "u_1"},
        ],
        "failed_or_superseded": [],
    }
    candidate = _make_candidate("read_file", '{"path": "/tmp/test.py"}')
    result = validate_candidate(candidate, TOOLS, checkpoint)
    assert result["verdict"] == "repairable"
    assert any(r["code"] == "redo_completed" for r in result["reasons"])


def test_deadend_retry():
    checkpoint = {
        "completed_actions": [],
        "failed_or_superseded": [
            {"tool": "read_file", "arguments": {"path": "/tmp/missing"}, "unit_id": "u_2", "error": "not found"},
        ],
    }
    candidate = _make_candidate("read_file", '{"path": "/tmp/missing"}')
    result = validate_candidate(candidate, TOOLS, checkpoint)
    assert result["verdict"] == "repairable"
    assert any(r["code"] == "deadend_retry" for r in result["reasons"])


# ── batch validation ─────────────────────────────────────────────────────────

def test_batch_all_valid():
    candidates = [
        _make_candidate("read_file", '{"path": "/tmp/a"}', "c1"),
        _make_candidate("read_file", '{"path": "/tmp/b"}', "c2"),
    ]
    result = validate_candidates(candidates, TOOLS, EMPTY_CHECKPOINT)
    assert result["verdict"] == "valid"


def test_batch_one_invalid_rejects_all():
    candidates = [
        _make_candidate("read_file", '{"path": "/tmp/a"}', "c1"),
        _make_candidate("nonexistent", '{}', "c2"),
    ]
    result = validate_candidates(candidates, TOOLS, EMPTY_CHECKPOINT)
    assert result["verdict"] == "invalid"


def test_batch_duplicate_ids():
    candidates = [
        _make_candidate("read_file", '{"path": "/tmp/a"}', "same_id"),
        _make_candidate("read_file", '{"path": "/tmp/b"}', "same_id"),
    ]
    result = validate_candidates(candidates, TOOLS, EMPTY_CHECKPOINT)
    assert result["verdict"] == "invalid"
    assert any(r["code"] == "duplicate_call_id" for r in result["reasons"])


def test_batch_empty():
    result = validate_candidates([], TOOLS, EMPTY_CHECKPOINT)
    assert result["verdict"] == "invalid"


# ── NaN/Infinity rejection ───────────────────────────────────────────────────

def test_reject_nan():
    candidate = _make_candidate("run_command", '{"command": "ls", "timeout": NaN}')
    result = validate_candidate(candidate, TOOLS, EMPTY_CHECKPOINT)
    # NaN in JSON is invalid; should be rejected
    assert result["verdict"] != "valid"
