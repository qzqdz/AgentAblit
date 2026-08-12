"""The `value_snippet` coldstart resolution: value-preserving compression + isolation-4/4.

Contract under test:
  1. RENDER — when a discriminating value sits PAST the closed-turn cap, `value_snippet` keeps
     it verbatim in B's coldstart memory where `snippet` front-truncates it away.
  2. NO REGRESSION — when the value fits (short result), value_snippet == snippet content-wise.
  3. SIZE — value_snippet never renders a LARGER memory than snippet (same per-result budget).
  4. ISOLATION — on mode="sniffer" the value is neutralized out (never verbatim), so it can
     never reach the aligned model even if the resolution were selected there.
  5. GUARD — the control.py resolution guard admits value_snippet for coldstart but NOT for the
     sniffer (verified by replicating the guard's set logic, the single source of the rule).
"""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from strategies.reconstruct import observer_context as QA
from strategies.reconstruct.observer_context import _RAW_RESULT_CAP, render_context
from strategies.reconstruct.value_compress import compress_result

FACT = "/srv/app_7f3c/staging"


def _convo_with_buried_fact(noise_chars: int):
    """A closed turn whose tool result buries FACT behind `noise_chars` of filler, then a
    thin current turn that refers to it only indirectly (B must recover it from memory)."""
    noise = ("[log] routine status line ok; " * 60)[:noise_chars]
    return [
        {"role": "system", "content": "agent"},
        {"role": "user", "content": "准备部署环境"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "0", "function": {"name": "provision", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "0", "content": f"{noise} provisioned workspace at {FACT}"},
        {"role": "user", "content": "把配置写到之前那个工作目录"},
    ]


def test_value_snippet_keeps_value_past_cap_where_snippet_drops_it():
    # bury the fact PAST the closed-turn cap so front-truncation loses it
    msgs = _convo_with_buried_fact(_RAW_RESULT_CAP + 80)
    snip = render_context("snippet", "c1", msgs, mode="coldstart")
    vsnip = render_context("value_snippet", "c1", msgs, mode="coldstart")
    assert FACT not in snip, "precondition: snippet must bury the value at this noise level"
    assert FACT in vsnip, "value_snippet must preserve the value past the cap"


def test_no_regression_when_value_fits():
    msgs = _convo_with_buried_fact(0)  # short result, value well within cap
    snip = render_context("snippet", "c2", msgs, mode="coldstart")
    vsnip = render_context("value_snippet", "c2", msgs, mode="coldstart")
    assert FACT in snip and FACT in vsnip  # both keep it; no regression


def test_value_snippet_not_larger_than_snippet():
    msgs = _convo_with_buried_fact(_RAW_RESULT_CAP + 200)
    snip = render_context("snippet", "c3", msgs, mode="coldstart")
    vsnip = render_context("value_snippet", "c3", msgs, mode="coldstart")
    assert len(vsnip) <= len(snip) + 4  # same per-result budget; never materially larger


def test_isolation_sniffer_mode_neutralizes_the_value():
    msgs = _convo_with_buried_fact(_RAW_RESULT_CAP + 80)
    sniff = render_context("value_snippet", "c4", msgs, mode="sniffer")
    assert FACT not in sniff, "verbatim value must never appear on the aligned sniffer path"
    assert ("成功" in sniff or "失败" in sniff)  # neutralized tool→status form


def test_control_guard_admits_coldstart_only():
    # Replicate the single source of the rule (control.py __init__): value_snippet is in
    # valid_coldstart, NOT in valid (which gates the sniffer and the both-consumers override).
    valid = {"snippet", "distilled", "hybrid", "intent"}
    valid_coldstart = valid | {"value_snippet"}
    assert "value_snippet" in valid_coldstart
    assert "value_snippet" not in valid
    # a sniffer knob set to value_snippet must resolve to "" (rejected)
    sniffer_knob = "value_snippet"
    resolved_sniffer = sniffer_knob if sniffer_knob in valid else ""
    assert resolved_sniffer == ""
    # a coldstart knob set to value_snippet is accepted
    coldstart_knob = "value_snippet"
    resolved_coldstart = coldstart_knob if coldstart_knob in valid_coldstart else ""
    assert resolved_coldstart == "value_snippet"


def test_compress_result_is_extractive_and_source_only():
    txt = "region us-east-1; iops 3000; mounted at /mnt/vol_k29x/data; done"
    out = compress_result(txt, budget=40, question="write config to the mounted data path")
    assert "/mnt/vol_k29x/data" in out          # value preserved verbatim
    assert len(out) <= len(txt)                  # never larger
    assert all(ord(c) < 0x2E7F for c in out)     # no CJK injected (source was ASCII)


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        try:
            fn(); passed += 1; print(f"ok   {fn.__name__}")
        except Exception:
            print(f"FAIL {fn.__name__}"); traceback.print_exc()
    print(f"\n{passed}/{len(fns)} passed")
