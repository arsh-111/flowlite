"""FastAPI server: POST /clean and a WebSocket /ws that streams PCM16 audio in and
JSON events out (partial -> final -> clean). One shared model, serialised by a lock;
decode runs in a worker thread so the event loop stays responsive.

Protocol (client -> server): binary frames of 16 kHz mono little-endian int16 PCM;
text frame "flush" ends the current utterance.
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .asr import ASRBackend, LockedBackend
from .audio_io import pcm16_to_float32
from .cleanup import Cleaner
from .config import Settings, build_asr, build_cleaner
from .personalization import CorrectionMemory
from .streaming import StreamingTranscriber


class CleanRequest(BaseModel):
    text: str
    style: str = "neutral"
    dictionary: list[str] = []


def create_app(backend: ASRBackend, cleaner: Cleaner, *, memory: CorrectionMemory | None = None,
               endpoint_silence_ms: int = 500, partial_interval_s: float | None = 1.0,
               dictionary: list[str] | None = None) -> FastAPI:
    app = FastAPI(title="flowlite", version="0.1.0")
    backend = LockedBackend(backend)
    base_terms = list(dictionary or [])

    @app.get("/", include_in_schema=False)
    async def demo():
        return FileResponse(Path(__file__).parent / "static" / "index.html")

    @app.get("/healthz")
    def healthz():
        return {"ok": True, "asr": backend.name, "cleaner": cleaner.name}

    @app.post("/clean")
    async def clean(req: CleanRequest):
        res = await asyncio.to_thread(cleaner.clean, req.text, style=req.style,
                                      dictionary=req.dictionary or base_terms)
        return res.__dict__

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket, style: str = "neutral"):
        await ws.accept()
        tr = StreamingTranscriber(backend, endpoint_silence_ms=endpoint_silence_ms,
                                  partial_interval_s=partial_interval_s,
                                  prompt=", ".join(base_terms) or None)
        try:
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if msg.get("bytes") is not None:
                    events = await asyncio.to_thread(tr.feed, pcm16_to_float32(msg["bytes"]))
                elif msg.get("text") == "flush":
                    events = await asyncio.to_thread(tr.flush)
                else:
                    continue
                for ev in events:
                    await _emit(ws, ev, cleaner, memory, style, base_terms)
                if msg.get("text") == "flush":
                    await ws.send_json({"type": "flushed"})
        except WebSocketDisconnect:
            pass

    return app


async def _emit(ws, ev, cleaner, memory, style, terms) -> None:
    if ev.kind == "partial":
        await ws.send_json({"type": "partial", "utt_id": ev.utt_id, "stable": ev.stable,
                            "tentative": ev.tentative, "asr_ms": round(ev.asr_ms, 1)})
        return
    await ws.send_json({"type": "final", "utt_id": ev.utt_id, "text": ev.text,
                        "asr_ms": round(ev.asr_ms, 1), "audio_s": round(ev.audio_s, 2)})
    raw = memory.apply(ev.text) if memory else ev.text
    all_terms = terms + (memory.terms() if memory else [])
    res = await asyncio.to_thread(cleaner.clean, raw, style=style, dictionary=all_terms)
    await ws.send_json({
        "type": "clean", "utt_id": ev.utt_id, "text": res.text, "raw": raw,
        "cleanup_ms": round(res.total_ms, 1),
        "ttft_ms": None if res.ttft_ms is None else round(res.ttft_ms, 1),
        "fallback": res.used_fallback, "fallback_reason": res.fallback_reason,
        "after_endpoint_ms": round((time.perf_counter() - ev.endpoint_at) * 1000, 1),
    })


def app_from_env() -> FastAPI:  # uvicorn flowlite.server:app_from_env --factory
    s = Settings()
    from .cleanup import load_dictionary
    import os
    terms = load_dictionary(os.getenv("FLOWLITE_DICTIONARY", "data/dictionary.txt"))
    return create_app(build_asr(s), build_cleaner(s), memory=CorrectionMemory(),
                      endpoint_silence_ms=s.endpoint_ms, partial_interval_s=s.partial_s,
                      dictionary=terms)
