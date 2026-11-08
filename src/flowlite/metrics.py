"""Edit-rate and latency metrics (dependency-free).

zero_edit_strict   - output equals reference after whitespace collapse only
                     (case + punctuation must match: "would the user touch it?")
zero_edit_lenient  - equal after lower-casing, punctuation removal and time normalisation
wer (lenient)      - micro-averaged word error rate on lenient tokens
char_edit_ratio    - Levenshtein(strict(out), strict(ref)) / len(ref): rough "keystrokes to fix"
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence


def levenshtein(a: Sequence, b: Sequence) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def normalize_strict(s: str) -> str:
    return " ".join(s.split())


def normalize_lenient(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b([ap])\.m\.?", r"\1m", s)        # p.m. -> pm
    s = re.sub(r"(\d)\s+([ap]m)\b", r"\1\2", s)     # "6 pm" -> "6pm"
    s = re.sub(r"[^\w\s']", " ", s)
    return " ".join(s.split())


@dataclass
class SampleScore:
    strict_exact: bool
    lenient_exact: bool
    word_edits: int
    ref_words: int
    char_ratio: float


def score(reference: str, hypothesis: str) -> SampleScore:
    ref_s, hyp_s = normalize_strict(reference), normalize_strict(hypothesis)
    ref_l, hyp_l = normalize_lenient(reference).split(), normalize_lenient(hypothesis).split()
    return SampleScore(
        strict_exact=ref_s == hyp_s,
        lenient_exact=ref_l == hyp_l,
        word_edits=levenshtein(ref_l, hyp_l),
        ref_words=len(ref_l),
        char_ratio=levenshtein(ref_s, hyp_s) / max(len(ref_s), 1),
    )


def aggregate(scores: Sequence[SampleScore]) -> dict:
    n = len(scores)
    if n == 0:
        return {"n": 0, "zero_edit_strict": 0.0, "zero_edit_lenient": 0.0,
                "wer": 0.0, "char_edit_ratio": 0.0}
    words = sum(s.ref_words for s in scores)
    return {
        "n": n,
        "zero_edit_strict": sum(s.strict_exact for s in scores) / n,
        "zero_edit_lenient": sum(s.lenient_exact for s in scores) / n,
        "wer": sum(s.word_edits for s in scores) / max(words, 1),
        "char_edit_ratio": sum(s.char_ratio for s in scores) / n,
    }


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile, q in [0, 100]."""
    if not values:
        raise ValueError("empty")
    v = sorted(values)
    if len(v) == 1:
        return v[0]
    pos = (len(v) - 1) * q / 100
    lo = int(pos)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


def latency_stats(values: Sequence[float]) -> dict | None:
    if not values:
        return None
    return {
        "mean": sum(values) / len(values),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": max(values),
    }
