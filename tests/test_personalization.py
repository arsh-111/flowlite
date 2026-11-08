from flowlite.personalization import CorrectionMemory


def test_correct_twice_never_again(tmp_path):
    m = CorrectionMemory(tmp_path / "c.json", threshold=2)
    assert m.learn("sa hodge", "Sahaj") is False
    assert m.apply("ask sa hodge") == "ask sa hodge"
    assert m.learn("Sa Hodge", "Sahaj") is True
    assert m.apply("Ask sa hodge to review") == "Ask Sahaj to review"
    assert m.terms() == ["Sahaj"]


def test_persistence(tmp_path):
    p = tmp_path / "c.json"
    m = CorrectionMemory(p, threshold=1)
    m.learn("cooper netties", "Kubernetes")
    assert CorrectionMemory(p).apply("cooper netties config") == "Kubernetes config"


def test_ignores_noops_and_corrupt_file(tmp_path):
    p = tmp_path / "c.json"
    p.write_text("{not json")
    m = CorrectionMemory(p)
    assert m.learn("same", "same") is False
    assert m.active == {}
