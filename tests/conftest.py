import numpy as np
import pytest

from flowlite.vad import SAMPLE_RATE


def tone(seconds: float, amp: float = 0.3, freq: float = 220.0) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(SAMPLE_RATE * seconds), dtype=np.float32)


class FakeBackend:
    """Scripted ASR: returns outputs in order, repeating the last one."""

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = 0
        self.name = "fake"

    def transcribe(self, audio, prompt=None):
        out = self.outputs[min(self.calls, len(self.outputs) - 1)]
        self.calls += 1
        return out


@pytest.fixture
def audio():
    return {"tone": tone, "silence": silence}
