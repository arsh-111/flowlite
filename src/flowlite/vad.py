"""Tiny energy-based voice activity detector with an adaptive noise floor.

Deliberately dependency-free so endpointing is deterministic and unit-testable.
Swap in Silero/WebRTC VAD for production (see README roadmap).
"""
from __future__ import annotations

import math

import numpy as np

SAMPLE_RATE = 16_000
FRAME_MS = 30
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000  # 480


def rms_dbfs(frame: np.ndarray) -> float:
    if frame.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float64))))
    return 20.0 * math.log10(max(rms, 1e-6))


class EnergyVAD:
    """Speech if frame energy exceeds max(absolute floor, noise floor + margin)."""

    def __init__(
        self,
        min_db: float = -45.0,
        margin_db: float = 9.0,
        noise_alpha: float = 0.05,
        init_noise_db: float = -60.0,
    ) -> None:
        self.min_db = min_db
        self.margin_db = margin_db
        self.noise_alpha = noise_alpha
        self.noise_db = init_noise_db

    def is_speech(self, frame: np.ndarray) -> bool:
        db = rms_dbfs(frame)
        threshold = max(self.min_db, self.noise_db + self.margin_db)
        speech = db > threshold
        if not speech:  # only adapt the noise floor on non-speech frames
            a = self.noise_alpha
            self.noise_db = (1 - a) * self.noise_db + a * db
        return speech
