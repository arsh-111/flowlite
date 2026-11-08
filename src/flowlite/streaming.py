"""Streaming transcription: VAD endpointing + Local-Agreement partials.

Design
------
* Audio arrives in arbitrary chunks and is re-framed into 30 ms frames.
* An energy VAD opens an utterance (with ~240 ms pre-roll) and closes it after
  `endpoint_silence_ms` of silence (or `max_utterance_s`).
* While speaking, every `partial_interval_s` of audio the whole utterance buffer is
  re-decoded. Words that two consecutive hypotheses agree on (Local Agreement-2, the
  policy popularised by whisper_streaming) are reported as *stable*, the rest as
  *tentative*. This gives live feedback without committing to unstable words.
* At the endpoint the utterance is decoded once more -> the `final` event.
  `Event.endpoint_at` is stamped *before* that decode so callers can measure true
  post-speech latency (final ASR + cleanup).
"""
from __future__ import annotations

import math
import re
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from .asr import ASRBackend
from .vad import FRAME_MS, FRAME_SAMPLES, SAMPLE_RATE, EnergyVAD


@dataclass
class Event:
    kind: str  # "partial" | "final"
    utt_id: int
    text: str
    stable: str = ""
    tentative: str = ""
    audio_s: float = 0.0
    asr_ms: float = 0.0
    endpoint_at: float = 0.0  # time.perf_counter() when the endpoint fired (final only)


def _norm_tok(tok: str) -> str:
    return re.sub(r"[^\w']+", "", tok.lower())


def common_prefix_len(a: list[str], b: list[str]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


class StreamingTranscriber:
    def __init__(
        self,
        backend: ASRBackend,
        *,
        vad: EnergyVAD | None = None,
        endpoint_silence_ms: int = 500,
        partial_interval_s: float | None = 1.0,
        min_speech_s: float = 0.25,
        max_utterance_s: float = 25.0,
        preroll_ms: int = 240,
        prompt: str | None = None,
    ) -> None:
        self.backend = backend
        self.vad = vad or EnergyVAD()
        self.endpoint_silence_ms = endpoint_silence_ms
        self.min_speech_s = min_speech_s
        self.max_utterance_s = max_utterance_s
        self.prompt = prompt
        self._endpoint_frames = math.ceil(endpoint_silence_ms / FRAME_MS)
        self._partial_frames = (
            math.ceil(partial_interval_s * 1000 / FRAME_MS) if partial_interval_s else None
        )
        self._preroll: deque[np.ndarray] = deque(maxlen=max(1, preroll_ms // FRAME_MS))
        self._remainder = np.zeros(0, dtype=np.float32)
        self._utt_id = 0
        self._reset_utterance()

    # ------------------------------------------------------------------ state
    def _reset_utterance(self) -> None:
        self._in_speech = False
        self._frames: list[np.ndarray] = []
        self._speech_frames = 0
        self._silence_run = 0
        self._since_partial = 0
        self._prev_norm: list[str] = []

    # -------------------------------------------------------------------- API
    def feed(self, samples: np.ndarray) -> list[Event]:
        """Feed mono float32 16 kHz samples (any length). Returns new events."""
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        if self._remainder.size:
            samples = np.concatenate([self._remainder, samples])
        n = (samples.size // FRAME_SAMPLES) * FRAME_SAMPLES
        self._remainder = samples[n:]
        events: list[Event] = []
        for i in range(0, n, FRAME_SAMPLES):
            events.extend(self._on_frame(samples[i : i + FRAME_SAMPLES]))
        return events

    def flush(self) -> list[Event]:
        """Force-finalise any utterance in progress (end of stream)."""
        if self._in_speech:
            return self._finalize()
        return []

    # --------------------------------------------------------------- internals
    def _on_frame(self, frame: np.ndarray) -> list[Event]:
        is_speech = self.vad.is_speech(frame)
        if not self._in_speech:
            if is_speech:
                self._in_speech = True
                self._frames = list(self._preroll) + [frame]
                self._preroll.clear()
                self._speech_frames = 1
            else:
                self._preroll.append(frame)
            return []

        self._frames.append(frame)
        self._since_partial += 1
        if is_speech:
            self._speech_frames += 1
            self._silence_run = 0
        else:
            self._silence_run += 1

        duration_s = len(self._frames) * FRAME_MS / 1000
        if self._silence_run >= self._endpoint_frames or duration_s >= self.max_utterance_s:
            return self._finalize()
        if (
            self._partial_frames is not None
            and is_speech
            and self._since_partial >= self._partial_frames
            and self._speech_frames * FRAME_MS / 1000 >= self.min_speech_s
        ):
            return [self._partial()]
        return []

    def _decode(self) -> tuple[str, float, float]:
        audio = np.concatenate(self._frames)
        t0 = time.perf_counter()
        text = self.backend.transcribe(audio, self.prompt)
        return text.strip(), (time.perf_counter() - t0) * 1000, len(audio) / SAMPLE_RATE

    def _partial(self) -> Event:
        text, ms, audio_s = self._decode()
        toks = text.split()
        norm = [_norm_tok(t) for t in toks]
        k = common_prefix_len(norm, self._prev_norm) if self._prev_norm else 0
        self._prev_norm = norm
        self._since_partial = 0
        return Event(
            "partial", self._utt_id, text,
            stable=" ".join(toks[:k]), tentative=" ".join(toks[k:]),
            audio_s=audio_s, asr_ms=ms,
        )

    def _finalize(self) -> list[Event]:
        endpoint_at = time.perf_counter()
        speech_s = self._speech_frames * FRAME_MS / 1000
        utt_id = self._utt_id
        if speech_s < self.min_speech_s:  # blip / click: drop it
            self._reset_utterance()
            return []
        text, ms, audio_s = self._decode()
        self._reset_utterance()
        self._utt_id += 1
        if not text:
            return []
        return [Event("final", utt_id, text, stable=text, audio_s=audio_s,
                      asr_ms=ms, endpoint_at=endpoint_at)]
