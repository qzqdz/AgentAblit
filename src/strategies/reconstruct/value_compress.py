"""Value-preserving EXTRACTIVE compression for the B (9B) coldstart context.

WHY THIS EXISTS. The coldstart resolutions render a CLOSED turn's tool result by FRONT-
truncating it to a char cap (`observer_context._RAW_RESULT_OLD_CAP` / `_RAW_RESULT_CAP`). When
the discriminating value B needs for its next tool_call (a path / id / token / host) sits past
that cap — e.g. buried behind a verbose status preamble — front-truncation drops it, and B
cannot reconstruct the next action (runs/context_ablation/REPORT.md: snippet 0/6 at verbose;
C's LLM distillation loses it too by ABSTRACTING concrete values away, 3-4/6). This module keeps
the SAME char budget but spends it on the value-bearing clause instead of the front:
extractive (keep-or-drop whole clauses, never paraphrase), so a surviving token is byte-
identical — and, being source-only, it cannot inject a language absent from the source (B is
Chinese-leaning; an abstractive rewrite would).

It is deterministic and model-free (matches the "snippet is free" ethos), and by construction
only preserves; it does not summarise or add narrative. Coldstart (B / 9B) path ONLY — its
verbatim values must never reach the aligned sniffer (isolation-4/4); the resolution wiring in
control.py structurally keeps `value_snippet` out of the sniffer's valid set.

Validated on the value-recovery ablation (both the original 6-scenario set and a 23-scenario
value-type/burial-position expansion): the value survives the cap and the real 9B uses it in
its next tool_call where front-truncation buried it, with no context-size increase and the
confound (value-regex-off collapses to front-truncation) ruled out.

Honest limit: the value patterns are ENUMERABLE — a novel value shape not matched here is
front-truncated like snippet (a MAC address was missed until its pattern was added). A learned
token classifier (LLMLingua-2) is the generalisation upgrade behind the same seam.
"""
from __future__ import annotations

import re

# Operational-value patterns: the concrete tokens B's next tool_call is built from.
_VALUE_PATTERNS = [
    re.compile(r"(?:/[\w.\-]+){2,}/?"),                       # absolute-ish paths /a/b/c
    re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"),              # IPv4
    re.compile(r"\b[\w.\-]+/[\w./\-]+\b"),                    # relative paths, host/path, s3://bkt/x
    re.compile(r"https?://\S+|s3://\S+"),                    # URLs / object stores
    re.compile(r"\b[0-9a-fA-F]{8,}\b"),                      # hex ids / hashes / git sha
    re.compile(r"\b[A-Za-z0-9_\-]*\d{4,}[A-Za-z0-9_\-]*\b"),  # long numeric ids
    # mixed alphanumeric token (has BOTH a letter and a digit) → almost always an identifier
    # (req-9x42kd, sk-tok-Q7wz, db_shard_e19, app_7f3c), never an ordinary word.
    re.compile(r"\b(?=[\w.\-]*[A-Za-z])(?=[\w.\-]*\d)[\w.\-]{4,}\b"),
    re.compile(r"\b[\w\-]+\.(?:tar\.gz|tar|tgz|zip|gz|txt|ya?ml|json|csv|log|py|sh|conf|db|sql)\b", re.I),  # filenames
    re.compile(r"\b(?:[0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2}\b"),  # MAC
    re.compile(r"\b(?:[0-9a-fA-F]{2,4}:){2,}[0-9a-fA-F]{1,4}\b"),  # colon-hex groups (MAC/IPv6-ish)
    re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),  # emails
    re.compile(r"\b\w+=[^\s,;]+"),                            # key=value args
    re.compile(r'"[^"]{2,}"|\'[^\']{2,}\''),                  # quoted args / values
    re.compile(r"\bv\d+\.\d+\.\d+(?:-[\w.]+)?\b"),           # semver (v3.11.2-rc4)
    re.compile(r"\bport\s+\d{2,5}\b|:\d{2,5}\b"),           # ports
]

_STATUS_RE = re.compile(
    r"\b(error|failed|failure|denied|refused|rejected|timeout|unavailable|"
    r"success|succeeded|done|completed)\b|错误|失败|拒绝|超时|无法|成功|完成",
    re.I,
)
_WORD_RE = re.compile(r"[A-Za-z0-9_]{3,}")


def _has_value(s: str) -> bool:
    return any(p.search(s) for p in _VALUE_PATTERNS)


def _clause_score(clause: str, q_terms: set) -> float:
    s = 0.0
    if _has_value(clause):
        s += 10.0
    if _STATUS_RE.search(clause):
        s += 3.0
    if q_terms:
        toks = {t.lower() for t in _WORD_RE.findall(clause)}
        s += 1.5 * len(toks & q_terms)
    return s


def compress_result(text: str, budget: int, question: str = "") -> str:
    """Return an extractive compression of `text` within `budget` chars, value clauses first.

    Segments at clause granularity (newline / ';'), scores by operational-value + status +
    optional question overlap, keeps highest-scoring clauses until the budget is spent, then
    restores original order. A value-bearing clause is admitted even slightly over budget so a
    single long value is never dropped for a filler clause; if the whole text already fits the
    budget it is returned unchanged (never larger than front-truncation would be). Never
    rewrites a kept clause — survivors are byte-identical, source-language only.
    """
    if len(text) <= budget:
        return text
    q_terms = {t.lower() for t in _WORD_RE.findall(question)} if question else set()
    clauses = [c.strip() for c in re.split(r"[\n;]+", text) if c.strip()]
    if not clauses:
        return text[:budget]
    scored = sorted(
        ((i, c, _clause_score(c, q_terms)) for i, c in enumerate(clauses)),
        key=lambda t: (-t[2], t[0]),
    )
    kept: list[tuple] = []
    used = 0
    for i, c, sc in scored:
        if sc <= 0 and kept:
            continue
        if used + len(c) > budget and kept and sc < 10.0:
            continue
        kept.append((i, c))
        used += len(c) + 2
    if not kept:
        return text[:budget]
    kept.sort(key=lambda t: t[0])
    return "; ".join(c for _, c in kept)
