"""WAV helpers (stdlib `wave` + numpy). Expects 16-bit PCM; convert others with ffmpeg."""
from __future__ import annotations

import wave
from pathlib import Path
from typing import Iterator

import numpy as np

from .vad import SAMPLE_RATE


def read_wav(path: str | Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError(f"{path}: need 16-bit PCM WAV "
                             f"(try: ffmpeg -i in.wav -ar 16000 -ac 1 -c:a pcm_s16le out.wav)")
        rate, channels = w.getframerate(), w.getnchannels()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    audio = data.astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    if rate != SAMPLE_RATE:  # linear resample: fine for a demo, use soxr/ffmpeg for prod
        n_out = int(round(len(audio) * SAMPLE_RATE / rate))
        audio = np.interp(np.linspace(0, len(audio) - 1, n_out),
                          np.arange(len(audio)), audio).astype(np.float32)
    return audio


def write_wav(path: str | Path, audio: np.ndarray, rate: int = SAMPLE_RATE) -> None:
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())


def pcm16_to_float32(raw: bytes) -> np.ndarray:
    usable = len(raw) - (len(raw) % 2)
    return np.frombuffer(raw[:usable], dtype="<i2").astype(np.float32) / 32768.0


def float32_to_pcm16(audio: np.ndarray) -> bytes:
    return (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()


def iter_chunks(audio: np.ndarray, seconds: float = 0.1) -> Iterator[np.ndarray]:
    step = int(SAMPLE_RATE * seconds)
    for i in range(0, len(audio), step):
        yield audio[i : i + step]
