import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from conftest import FakeBackend, silence, tone  # noqa: E402
import numpy as np  # noqa: E402

from flowlite.audio_io import float32_to_pcm16  # noqa: E402
from flowlite.cleanup import RuleBasedCleaner  # noqa: E402
from flowlite.server import create_app  # noqa: E402


def make_client():
    app = create_app(FakeBackend(["um hello world"]), RuleBasedCleaner(), partial_interval_s=None)
    return TestClient(app)


def test_health_and_clean_endpoint():
    c = make_client()
    page = c.get("/")
    assert page.status_code == 200 and "Flowlite" in page.text
    assert c.get("/healthz").json()["ok"] is True
    r = c.post("/clean", json={"text": "um can you help"}).json()
    assert r["text"] == "Can you help?"


def test_websocket_stream_to_clean():
    c = make_client()
    pcm = float32_to_pcm16(np.concatenate([tone(0.8), silence(0.9)]))
    with c.websocket_connect("/ws") as ws:
        for i in range(0, len(pcm), 3200):
            ws.send_bytes(pcm[i:i + 3200])
        final = ws.receive_json()
        clean = ws.receive_json()
        ws.send_text("flush")
        flushed = ws.receive_json()
    assert final["type"] == "final" and final["text"] == "um hello world"
    assert clean["type"] == "clean" and clean["text"] == "Hello world."
    assert clean["after_endpoint_ms"] >= 0 and clean["fallback"] is False
    assert flushed["type"] == "flushed"
