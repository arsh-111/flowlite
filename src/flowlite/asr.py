"""ASR backends. faster-whisper is imported lazily so the rest of the package
(and the unit tests) work without the heavy dependency installed."""
from __future__ import annotations

import threading
from typing import Protocol

import numpy as np


class ASRBackend(Protocol):
    name: str

    def transcribe(self, audio: np.ndarray, prompt: str | None = None) -> str:
        """audio: mono float32 at 16 kHz in [-1, 1]. Returns plain text."""
        ...


class FasterWhisperBackend:
    def __init__(
        self,
        model_size: str = "base.en",
        device: str = "auto",
        compute_type: str = "int8",
        language: str | None = "en",
        beam_size: int = 1,
        cpu_threads: int = 0,
    ) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "faster-whisper is not installed. Run: pip install -e '.[asr]'"
            ) from e
        self.name = f"faster-whisper:{model_size}:{device}:{compute_type}"
        self.language = language
        self.beam_size = beam_size
        self.model = WhisperModel(
            model_size, device=device, compute_type=compute_type, cpu_threads=cpu_threads
        )

    def transcribe(self, audio: np.ndarray, prompt: str | None = None) -> str:
        segments, _info = self.model.transcribe(
            audio.astype(np.float32, copy=False),
            language=self.language,
            beam_size=self.beam_size,
            temperature=0.0,
            condition_on_previous_text=False,
            vad_filter=False,  # we endpoint ourselves
            without_timestamps=True,
            initial_prompt=prompt,
        )
        return " ".join(s.text.strip() for s in segments).strip()


class LockedBackend:
    """Serialises access to a backend shared by several threads/connections."""

    def __init__(self, inner: ASRBackend) -> None:
        self.inner = inner
        self.name = inner.name
        self._lock = threading.Lock()

    def transcribe(self, audio: np.ndarray, prompt: str | None = None) -> str:
        with self._lock:
            return self.inner.transcribe(audio, prompt)


def warmup(backend: ASRBackend) -> None:
    """Run one dummy decode so model load / kernel init doesn't pollute latency numbers."""
    backend.transcribe(np.zeros(16_000, dtype=np.float32))
