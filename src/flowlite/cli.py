"""flowlite command line: live | transcribe | clean | teach | bench-cleanup | bench-e2e | serve | doctor"""
from __future__ import annotations

import argparse
import os
import queue
import sys
import time
from pathlib import Path

from . import __version__
from .bench import (evaluate, format_category_table, format_cleanup_table, format_e2e_table,
                    load_samples, run_e2e, save_results, show_failures)
from .cleanup import PassthroughCleaner, load_dictionary
from .config import Settings, build_asr, build_cleaner
from .personalization import CorrectionMemory
from .vad import SAMPLE_RATE

DEFAULT_DICT = "data/dictionary.txt"


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--cleaner", choices=["llm", "rules", "none"], help="cleanup backend")
    p.add_argument("--llm-url", help="OpenAI-compatible base URL (default: Ollama)")
    p.add_argument("--llm-model")
    p.add_argument("--style", choices=["neutral", "formal", "casual"])
    p.add_argument("--dictionary", help="file with preferred spellings, one per line")
    p.add_argument("--asr-model", help="faster-whisper model: tiny.en, base.en, small, ...")
    p.add_argument("--device", help="auto | cpu | cuda")
    p.add_argument("--compute-type", help="int8 | float16 | ...")
    p.add_argument("--language", help="en | hi | auto ...")
    p.add_argument("--endpoint-ms", type=int, help="silence needed to end an utterance")
    p.add_argument("--partial-s", type=float, help="seconds between partial decodes (0=off)")


def _settings(args) -> Settings:
    s = Settings()
    for src, dst in [("cleaner", "cleaner"), ("llm_url", "llm_url"), ("llm_model", "llm_model"),
                     ("style", "style"), ("asr_model", "asr_model"), ("device", "device"),
                     ("compute_type", "compute_type"), ("language", "language"),
                     ("endpoint_ms", "endpoint_ms"), ("partial_s", "partial_s")]:
        v = getattr(args, src, None)
        if v is not None:
            setattr(s, dst, v)
    return s


def _dictionary(args, default: str | None = None) -> list[str]:
    path = getattr(args, "dictionary", None) or default
    return load_dictionary(path) if path else []


def _print_final(raw, res, ev, hangover_ms) -> None:
    after = (time.perf_counter() - ev.endpoint_at) * 1000
    ttft = f", first token {res.ttft_ms:.0f} ms" if res.ttft_ms is not None else ""
    fb = f"  [fallback: {res.fallback_reason}]" if res.used_fallback else ""
    print(f"\r\033[2K  raw    : {ev.text}")
    print(f"  clean  : {res.text}{fb}")
    print(f"  timing : speech {ev.audio_s:.1f}s | ASR {ev.asr_ms:.0f} ms | cleanup "
          f"{res.total_ms:.0f} ms{ttft} | after-endpoint {after:.0f} ms "
          f"(+{hangover_ms} ms silence to detect end)\n")


# --------------------------------------------------------------------- commands
def cmd_live(args) -> None:
    try:
        import sounddevice as sd
    except (ImportError, OSError) as e:
        raise SystemExit(f"sounddevice unavailable ({e}). pip install -e '.[asr]' "
                         "(Linux: apt install libportaudio2)")
    from .streaming import StreamingTranscriber
    from .vad import FRAME_SAMPLES
    s = _settings(args)
    memory = CorrectionMemory()
    terms = _dictionary(args, DEFAULT_DICT if Path(DEFAULT_DICT).exists() else None)
    print(f"Loading {s.asr_model} ...")
    from .asr import warmup
    backend = build_asr(s)
    warmup(backend)
    cleaner = build_cleaner(s)
    if hasattr(cleaner, "healthcheck"):
        ok, msg = cleaner.healthcheck()
        if not ok:
            print(f"! LLM endpoint not reachable ({msg}); will fall back to rules.\n"
                  f"  Start one with: ollama pull {s.llm_model}")
    cleaner.clean("um warm up the model", dictionary=terms)  # warm LLM + prefix cache
    tr = StreamingTranscriber(backend, endpoint_silence_ms=s.endpoint_ms,
                              partial_interval_s=s.partial_s or None,
                              prompt=", ".join(terms + memory.terms()) or None)
    q: queue.Queue = queue.Queue()

    def callback(indata, frames, t, status):
        q.put(indata[:, 0].copy())

    print("Listening. Speak, then pause. Ctrl+C to quit.\n")
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                        blocksize=FRAME_SAMPLES, callback=callback):
        try:
            while True:
                for ev in tr.feed(q.get()):
                    if ev.kind == "partial":
                        print(f"\r\033[2K  ... {ev.stable} \033[2m{ev.tentative}\033[0m",
                              end="", flush=True)
                    else:
                        raw = memory.apply(ev.text)
                        res = cleaner.clean(raw, style=s.style,
                                            dictionary=terms + memory.terms())
                        _print_final(raw, res, ev, s.endpoint_ms)
        except KeyboardInterrupt:
            for ev in tr.flush():
                raw = memory.apply(ev.text)
                _print_final(raw, cleaner.clean(raw, style=s.style, dictionary=terms), ev,
                             s.endpoint_ms)


def cmd_transcribe(args) -> None:
    from .asr import warmup
    from .bench import transcribe_file
    s = _settings(args)
    terms = _dictionary(args, DEFAULT_DICT if Path(DEFAULT_DICT).exists() else None)
    backend = build_asr(s)
    warmup(backend)
    cleaner = build_cleaner(s)
    memory = CorrectionMemory()
    text, asr_ms, audio_s = transcribe_file(backend, args.file, endpoint_silence_ms=s.endpoint_ms,
                                            prompt=", ".join(terms) or None)
    raw = memory.apply(text)
    res = cleaner.clean(raw, style=s.style, dictionary=terms + memory.terms())
    print(f"raw   : {raw}\nclean : {res.text}")
    print(f"audio {audio_s:.1f}s | ASR {asr_ms:.0f} ms (RTF {asr_ms / 1000 / audio_s:.2f}) | "
          f"cleanup {res.total_ms:.0f} ms" + (f" | fallback: {res.fallback_reason}"
                                              if res.used_fallback else ""))


def cmd_clean(args) -> None:
    s = _settings(args)
    terms = _dictionary(args, DEFAULT_DICT if Path(DEFAULT_DICT).exists() else None)
    memory = CorrectionMemory()
    cleaner = build_cleaner(s)
    raw = memory.apply(args.text)
    res = cleaner.clean(raw, style=s.style, dictionary=terms + memory.terms())
    print(res.text)
    print(f"[{res.backend} | {res.total_ms:.0f} ms"
          + (f" | ttft {res.ttft_ms:.0f} ms" if res.ttft_ms else "")
          + (f" | fallback: {res.fallback_reason}" if res.used_fallback else "") + "]",
          file=sys.stderr)


def cmd_teach(args) -> None:
    mem = CorrectionMemory()
    active = mem.learn(args.wrong, args.right)
    n = mem.counts.get(f"{' '.join(args.wrong.lower().split())}=>{args.right}", 0)
    print(f"{'ACTIVE' if active else 'noted'}: '{args.wrong}' -> '{args.right}' "
          f"(seen {n}/{mem.threshold})")


def _build_backends(names: list[str], s: Settings):
    out = []
    for n in names:
        out.append(PassthroughCleaner() if n == "raw" else build_cleaner(s, n))
    return out


def cmd_bench_cleanup(args) -> None:
    s = _settings(args)
    terms = _dictionary(args, args.dictionary or (DEFAULT_DICT if Path(DEFAULT_DICT).exists()
                                                  else None))
    samples = load_samples(args.data)[: args.limit or None]
    items = [{"id": x["id"], "category": x.get("category", "all"), "input": x["raw"],
              "reference": x["reference"]} for x in samples]
    summaries, results = [], {}
    for cleaner in _build_backends(args.backends.split(","), s):
        if hasattr(cleaner, "healthcheck"):
            ok, msg = cleaner.healthcheck()
            if not ok:
                print(f"! skipping {cleaner.name}: endpoint not reachable ({msg})", file=sys.stderr)
                continue
        print(f"running {cleaner.name} ...", file=sys.stderr)
        res = evaluate(cleaner, items, repeats=args.repeats, warmup=args.warmup,
                       style=s.style, dictionary=terms)
        summaries.append(res["summary"])
        results[cleaner.name] = res
        if args.failures:
            print(f"\n{cleaner.name}: first {args.failures} lenient misses", file=sys.stderr)
            print(show_failures(res, args.failures), file=sys.stderr)
    md = ("## Cleanup benchmark\n\n" + format_cleanup_table(summaries) +
          "\n\nZero-edit rate by category (lenient):\n\n" + format_category_table(summaries))
    print("\n" + md)
    path = save_results(args.out, "cleanup", {"summaries": summaries, "results": results}, md)
    print(f"\nsaved -> {path}", file=sys.stderr)


def cmd_bench_e2e(args) -> None:
    s = _settings(args)
    terms = _dictionary(args, DEFAULT_DICT if Path(DEFAULT_DICT).exists() else None)
    samples = load_samples(args.data)
    items = [{"id": x["id"], "category": x.get("category", "all"), "reference": x["reference"]}
             for x in samples]
    from .asr import warmup
    backend = build_asr(s)
    warmup(backend)
    cleaners = _build_backends(args.backends.split(","), s)
    rows, details = run_e2e(backend, cleaners, items, args.wav_dir,
                            endpoint_silence_ms=s.endpoint_ms, style=s.style, dictionary=terms)
    md = (f"## End-to-end benchmark ({backend.name}, endpoint {s.endpoint_ms} ms)\n\n"
          + format_e2e_table(rows)
          + "\n\npost-speech = endpoint silence + final ASR decode + cleanup "
            "(time from the user stopping to text being ready)")
    print("\n" + md)
    path = save_results(args.out, "e2e", {"rows": rows, "details": details}, md)
    print(f"\nsaved -> {path}", file=sys.stderr)


def cmd_serve(args) -> None:
    import uvicorn
    s = _settings(args)
    os.environ.update({
        "FLOWLITE_ASR_MODEL": s.asr_model, "FLOWLITE_DEVICE": s.device,
        "FLOWLITE_COMPUTE_TYPE": s.compute_type, "FLOWLITE_LANGUAGE": s.language,
        "FLOWLITE_CLEANER": s.cleaner, "FLOWLITE_LLM_BASE_URL": s.llm_url,
        "FLOWLITE_LLM_MODEL": s.llm_model,
    })
    uvicorn.run("flowlite.server:app_from_env", factory=True, host=args.host, port=args.port)


def cmd_doctor(args) -> None:
    s = _settings(args)
    print(f"flowlite {__version__} | python {sys.version.split()[0]}")
    for mod in ("numpy", "httpx", "faster_whisper", "sounddevice", "fastapi"):
        try:
            __import__(mod)
            print(f"  [ok]      {mod}")
        except (ImportError, OSError) as e:
            print(f"  [missing] {mod} ({type(e).__name__})")
    from .cleanup import LLMCleaner
    ok, msg = LLMCleaner(s.llm_url, s.llm_model, s.llm_key).healthcheck()
    print(f"  [{'ok' if ok else 'down'}]    LLM endpoint {s.llm_url} ({msg})")
    if not ok:
        print(f"            -> ollama serve & ollama pull {s.llm_model}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="flowlite", description=__doc__)
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("live", help="mic -> streaming ASR -> cleanup"); _add_common(p)
    p.set_defaults(fn=cmd_live)
    p = sub.add_parser("transcribe", help="run one WAV through the pipeline"); _add_common(p)
    p.add_argument("file"); p.set_defaults(fn=cmd_transcribe)
    p = sub.add_parser("clean", help="clean up a piece of text"); _add_common(p)
    p.add_argument("text"); p.set_defaults(fn=cmd_clean)
    p = sub.add_parser("teach", help="teach a correction (active after 2 repeats)")
    p.add_argument("wrong"); p.add_argument("right"); p.set_defaults(fn=cmd_teach)

    p = sub.add_parser("bench-cleanup", help="text-only cleanup benchmark"); _add_common(p)
    p.add_argument("--data", default="data/samples.jsonl")
    p.add_argument("--backends", default="raw,rules,llm")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--failures", type=int, default=0, help="print N misses per backend")
    p.add_argument("--out", default="results"); p.set_defaults(fn=cmd_bench_cleanup)

    p = sub.add_parser("bench-e2e", help="audio -> ASR -> cleanup benchmark"); _add_common(p)
    p.add_argument("--data", default="data/samples.jsonl")
    p.add_argument("--wav-dir", default="data/audio")
    p.add_argument("--backends", default="raw,rules,llm")
    p.add_argument("--out", default="results"); p.set_defaults(fn=cmd_bench_e2e)

    p = sub.add_parser("serve", help="FastAPI + WebSocket server"); _add_common(p)
    p.add_argument("--host", default="127.0.0.1"); p.add_argument("--port", type=int, default=8000)
    p.set_defaults(fn=cmd_serve)
    p = sub.add_parser("doctor", help="check environment"); _add_common(p)
    p.set_defaults(fn=cmd_doctor)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
