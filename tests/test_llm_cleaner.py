import json

import httpx

from flowlite.cleanup import LLMCleaner, build_messages, validate_output


def sse(*pieces):
    body = "".join("data: " + json.dumps({"choices": [{"delta": {"content": p}}]}) + "\n\n"
                   for p in pieces) + "data: [DONE]\n\n"
    return body.encode()


def make(handler):
    return LLMCleaner(client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_streams_and_measures_ttft():
    seen = {}

    def handler(req: httpx.Request):
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, content=sse("Let's meet ", "at 6 pm."),
                              headers={"content-type": "text/event-stream"})

    res = make(handler).clean("uh let's meet at five actually six pm", dictionary=["Sahaj"])
    assert res.text == "Let's meet at 6 pm."
    assert not res.used_fallback and res.ttft_ms is not None and res.ttft_ms <= res.total_ms
    assert seen["body"]["stream"] is True and seen["body"]["temperature"] == 0
    assert "Sahaj" in seen["body"]["messages"][0]["content"]
    assert seen["body"]["messages"][-1]["content"].startswith("<transcript>uh let's")


def test_falls_back_on_hallucination():
    def handler(req):
        return httpx.Response(200, content=sse(
            "Sure! Here is a long poem about the ocean and the moonlit sky tonight."))

    res = make(handler).clean("um can you send the report")
    assert res.used_fallback and res.fallback_reason in {"novel_content", "too_long"}
    assert res.text == "Can you send the report?"


def test_falls_back_on_http_error():
    res = make(lambda req: httpx.Response(500)).clean("um hello there")
    assert res.used_fallback and res.fallback_reason.startswith("error:")
    assert res.text == "Hello there."


def test_strips_tags_and_quotes():
    res = make(lambda req: httpx.Response(200, content=sse('"<transcript>Hello there.</transcript>"'))
               ).clean("um hello there")
    assert res.text == "Hello there."


def test_validate_output():
    assert validate_output("a b c", "") == "empty"
    assert validate_output("what is the capital of france", "What is the capital of France?") is None
    assert validate_output("ask sa hodge to review the config",
                           "Ask Sahaj to review the Kubernetes config.", ["Sahaj", "Kubernetes"]) is None


def test_prompt_contains_few_shot_and_style():
    msgs = build_messages("x", style="formal")
    assert msgs[0]["role"] == "system" and "professional" in msgs[0]["content"]
    assert len(msgs) == 1 + 2 * 3 + 1
