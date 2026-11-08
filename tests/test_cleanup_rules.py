import pytest

from flowlite.cleanup import RuleBasedCleaner

r = RuleBasedCleaner()


@pytest.mark.parametrize("raw,expected", [
    ("um can you review my pull request", "Can you review my pull request?"),
    ("so um the deployment finished and uh everything looks healthy",
     "So the deployment finished and everything looks healthy."),
    ("we need to to update the the documentation", "We need to update the documentation."),
    ("let's meet at five actually six pm", "Let's meet at six pm."),
    ("the meeting is on Tuesday no wait Wednesday", "The meeting is on Wednesday."),
    ("it should take three actually four days", "It should take four days."),
    ("i think that's fine", "I think that's fine."),
    ("Hey, um, can you help", "Hey, can you help?"),
])
def test_rules(raw, expected):
    assert r.clean(raw).text == expected


def test_does_not_touch_legit_actually():
    assert "actually" in r.clean("it actually works well").text


def test_keeps_legit_double_had():
    assert r.clean("he had had enough").text.lower().startswith("he had had")


def test_dictionary_casing():
    assert r.clean("deploy to kubernetes", dictionary=["Kubernetes"]).text == "Deploy to Kubernetes."
