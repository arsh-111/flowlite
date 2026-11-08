import pytest

from flowlite.metrics import (aggregate, latency_stats, levenshtein, normalize_lenient,
                              percentile, score)


def test_levenshtein():
    assert levenshtein("kitten", "sitting") == 3
    assert levenshtein([], ["a"]) == 1
    assert levenshtein("abc", "abc") == 0


def test_lenient_normalisation_handles_time_and_punct():
    assert normalize_lenient("Let's meet at 6 p.m.!") == normalize_lenient("let's meet at 6pm")


def test_score_strict_vs_lenient():
    s = score("Hey, can you send it?", "hey can you send it")
    assert not s.strict_exact and s.lenient_exact and s.word_edits == 0
    s = score("Hello world.", "Hello world.")
    assert s.strict_exact and s.char_ratio == 0


def test_wer_micro_average():
    scores = [score("a b c d", "a b c d"), score("a b", "a x")]
    agg = aggregate(scores)
    assert agg["wer"] == pytest.approx(1 / 6)
    assert agg["zero_edit_lenient"] == 0.5


def test_percentiles():
    vals = list(range(1, 101))
    assert percentile(vals, 50) == pytest.approx(50.5)
    assert percentile(vals, 95) == pytest.approx(95.05)
    assert latency_stats([])is None
    assert latency_stats([5])["p99"] == 5
