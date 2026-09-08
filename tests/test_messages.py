"""Tests for shared.messages — message extraction and text utilities."""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shared.messages import (
    text_from_content,
    compact_text,
    first_assistant_message,
    contains_forbidden_fact_claim,
    summarize_tool_calls,
    build_role_context,
)


# ── text_from_content ────────────────────────────────────────────────────────

def test_text_from_string():
    assert text_from_content("hello") == "hello"


def test_text_from_none():
    assert text_from_content(None) == ""


def test_text_from_list_of_text_parts():
    parts = [
        {"type": "text", "text": "part 1 "},
        {"type": "text", "text": "part 2"},
    ]
    assert text_from_content(parts) == "part 1 part 2"


def test_text_from_list_skips_non_text():
    parts = [
        {"type": "text", "text": "visible"},
        {"type": "image_url", "image_url": {"url": "http://example.com/img.png"}},
    ]
    assert text_from_content(parts) == "visible"


def test_text_from_dict_with_text_key():
    assert text_from_content({"type": "text", "text": "from dict"}) == "from dict"


def test_text_from_list_with_plain_strings():
    assert text_from_content(["hello", " ", "world"]) == "hello world"


def test_text_from_unknown_type():
    assert text_from_content(42) == ""


# ── compact_text ─────────────────────────────────────────────────────────────

def test_compact_short_text():
    assert compact_text("short") == "short"


def test_compact_long_text_truncates():
    long = "x" * 2000
    result = compact_text(long, limit=100)
    assert len(result) < 200
    assert "truncated" in result


def test_compact_none():
    assert compact_text(None) == ""


def test_compact_list():
    assert compact_text(["a", "b", "c"]) == "a\nb\nc"


def test_compact_dict():
    result = compact_text({"text": "value", "other": "ignored"})
    assert "value" in result


# ── first_assistant_message ──────────────────────────────────────────────────

def test_first_assistant_basic():
    response = {"choices": [{"message": {"role": "assistant", "content": "hi"}}]}
    result = first_assistant_message(response)
    assert result is not None
    index, message = result
    assert index == 0
    assert message["content"] == "hi"


def test_first_assistant_skips_non_assistant():
    response = {"choices": [
        {"message": {"role": "user", "content": "question"}},
        {"message": {"role": "assistant", "content": "answer"}},
    ]}
    result = first_assistant_message(response)
    assert result is not None
    index, message = result
    assert index == 1


def test_first_assistant_empty_choices():
    assert first_assistant_message({"choices": []}) is None


def test_first_assistant_no_choices_key():
    assert first_assistant_message({}) is None


# ── contains_forbidden_fact_claim ────────────────────────────────────────────

def test_forbidden_claim_empty():
    assert not contains_forbidden_fact_claim("")


def test_forbidden_claim_file_read():
    assert contains_forbidden_fact_claim("I read the file at /tmp/test.py")


def test_forbidden_claim_test_result():
    assert contains_forbidden_fact_claim("tests passed successfully")


def test_forbidden_claim_allowed_fact():
    allowed = ["I read the file at /tmp/test.py"]
    assert not contains_forbidden_fact_claim("I read the file at /tmp/test.py", allowed)


def test_forbidden_claim_normal_text():
    assert not contains_forbidden_fact_claim("The weather is nice today")


# ── summarize_tool_calls ─────────────────────────────────────────────────────

def test_summarize_basic():
    calls = [
        {"type": "function", "function": {"name": "get_weather", "arguments": '{"location":"Tokyo"}'}},
    ]
    result = summarize_tool_calls(calls)
    assert len(result) == 1
    assert result[0]["name"] == "get_weather"
    assert "location" in result[0]["argument_keys"]


def test_summarize_non_list():
    assert summarize_tool_calls(None) == []
    assert summarize_tool_calls("not a list") == []


def test_summarize_max_eight():
    calls = [{"type": "function", "function": {"name": f"tool_{i}", "arguments": "{}"}} for i in range(20)]
    result = summarize_tool_calls(calls)
    assert len(result) == 8


# ── build_role_context ───────────────────────────────────────────────────────

def test_build_role_context_structure():
    messages = [
        {"role": "user", "content": "deploy the app"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "0", "function": {"name": "deploy", "arguments": "{}"}}
        ]},
        {"role": "tool", "tool_call_id": "0", "content": "deployed ok"},
        {"role": "user", "content": "check the logs"},
    ]
    ctx = build_role_context(messages)
    assert "latest_user_request" in ctx
    assert ctx["latest_user_request"] == "check the logs"
    assert "expected_stance" in ctx
    assert isinstance(ctx["allowed_facts"], list)
    assert isinstance(ctx["forbidden_facts"], list)
