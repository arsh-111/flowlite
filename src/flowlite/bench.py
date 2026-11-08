"""Benchmarks: text-only cleanup (bench-cleanup) and audio end-to-end (bench-e2e)."""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

from .audio_io import iter_chunks, read_wav
from .metrics import aggregate, latency_stats, score
from .streaming import StreamingTranscriber
from .vad import SAMPLE_RATE


def load_samples(path: str | Path) -> list[dict]:
    items = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            items.append(json.loads(line))
    return items


def evaluate(cleaner, items: Sequence[dict], *, repeats: int = 1, warmup: int = 0,
             style: str = "neutral", dictionary: Sequence[str] = ()) -> dict:
    """items: dicts with id, input (text to clean), reference, category."""
    for it in list(items)[:warmup]:
        cleaner.clean(it["input"], style=style, dictionary=dictionary)

    scored, records = [], []
    lat, ttft, fallbacks, calls = [], [], 0, 0
    for it in items:
        first = None
        per_sample = []
        for _ in range(max(1, repeats)):
            res = cleaner.clean(it["input"], style=style, dictionary=dictionary)
            calls += 1
            lat.append(res.total_ms)
            per_sample.append(res.total_ms)
            if res.ttft_ms is not None:
                ttft.append(res.ttft_ms)
            fallbacks += int(res.used_fallback)
            first = first or res
        sc = score(it["reference"], first.text)
        scored.append((it.get("category", "all"), sc))
        records.append({
            "id": it["id"], "category": it.get("category", "all"),
            "input": it["input"], "output": first.text, "reference": it["reference"],
            "strict_exact": sc.strict_exact, "lenient_exact": sc.lenient_exact,
            "latency_ms": per_sample, "fallback": first.used_fallback,
            "fallback_reason": first.fallback_reason,
        })

    by_cat: dict[str, list] = defaultdict(list)
    for cat, sc in scored:
        by_cat[cat].append(sc)
    summary = aggregate([sc for _, sc in scored])
    summary.update({
        "backend": cleaner.name,
        "latency_ms": latency_stats(lat),
        "ttft_ms": latency_stats(ttft),
        "fallback_rate": fallbacks / max(calls, 1),
        "by_category": {c: {"n": len(v), "zero_edit_lenient": aggregate(v)["zero_edit_lenient"]}
                        for c, v in by_cat.items()},
    })
    return {"summary": summary, "records": records}


# ------------------------------------------------------------------ formatting
def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def _ms(stats: dict | None, key: str) -> str:
    return "-" if not stats else f"{stats[key]:.0f}"


def format_cleanup_table(summaries: Sequence[dict]) -> str:
    head = ("| backend | n | zero-edit (strict) | zero-edit (lenient) | WER | char-edit | "
            "p50 ms | p95 ms | TTFT p50 ms | fallback |\n"
            "|---|---|---|---|---|---|---|---|---|---|")
    rows = [
        f"| {s['backend']} | {s['n']} | {_pct(s['zero_edit_strict'])} | "
        f"{_pct(s['zero_edit_lenient'])} | {_pct(s['wer'])} | {_pct(s['char_edit_ratio'])} | "
        f"{_ms(s['latency_ms'], 'p50')} | {_ms(s['latency_ms'], 'p95')} | "
        f"{_ms(s['ttft_ms'], 'p50')} | {_pct(s['fallback_rate'])} |"
        for s in summaries
    ]
    return "\n".join([head, *rows])


def format_category_table(summaries: Sequence[dict]) -> str:
    cats = sorted({c for s in summaries for c in s["by_category"]})
    head = "| category | " + " | ".join(s["backend"] for s in summaries) + " |\n"
    head += "|---|" + "---|" * len(summaries)
    rows = []
    for c in cats:
        cells = [_pct(s["by_category"][c]["zero_edit_lenient"]) if c in s["by_category"]
                 else "-" for s in summaries]
        rows.append(f"| {c} | " + " | ".join(cells) + " |")
    return "\n".join([head, *rows])


def format_e2e_table(rows: Sequence[dict]) -> str:
    head = ("| pipeline | n | zero-edit (strict) | zero-edit (lenient) | WER | "
            "ASR p50 ms | ASR p95 ms | RTF p50 | cleanup p50 ms | post-speech p50 ms | "
            "post-speech p95 ms |\n|---|---|---|---|---|---|---|---|---|---|---|")
    out = [
        f"| {r['pipeline']} | {r['n']} | {_pct(r['zero_edit_strict'])} | "
        f"{_pct(r['zero_edit_lenient'])} | {_pct(r['wer'])} | {_ms(r['asr_ms'], 'p50')} | "
        f"{_ms(r['asr_ms'], 'p95')} | {r['rtf_p50']:.2f} | {_ms(r['latency_ms'], 'p50')} | "
        f"{_ms(r['post_speech_ms'], 'p50')} | {_ms(r['post_speech_ms'], 'p95')} |"
        for r in rows
    ]
    return "\n".join([head, *out])


def show_failures(result: dict, limit: int) -> str:
    bad = [r for r in result["records"] if not r["lenient_exact"]][:limit]
    lines = []
    for r in bad:
        lines.append(f"  [{r['id']}/{r['category']}] in : {r['input']}\n"
                     f"{' ' * (len(r['id']) + len(r['category']) + 5)} out: {r['output']}\n"
                     f"{' ' * (len(r['id']) + len(r['category']) + 5)} ref: {r['reference']}"
                     + (f"   (fallback: {r['fallback_reason']})" if r["fallback"] else ""))
    return "\n".join(lines)


def save_results(out_dir: str | Path, kind: str, payload: dict, markdown: str) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    (out / f"{kind}_{stamp}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                                              encoding="utf-8")
    md = out / f"latest_{kind}.md"
    md.write_text(markdown + "\n", encoding="utf-8")
    return md


# ------------------------------------------------------------------------ e2e
def transcribe_file(backend, wav_path: str | Path, *, endpoint_silence_ms: int,
                    prompt: str | None) -> tuple[str, float, float]:
    """Replay a WAV through the streaming path. Returns (text, asr_ms, audio_s)."""
    audio = read_wav(wav_path)
    tail = np.zeros(int(SAMPLE_RATE * (endpoint_silence_ms / 1000 + 0.3)), dtype=np.float32)
    tr = StreamingTranscriber(backend, endpoint_silence_ms=endpoint_silence_ms,
                              partial_interval_s=None, prompt=prompt)
    events = []
    for chunk in iter_chunks(np.concatenate([audio, tail]), 0.1):
        events += tr.feed(chunk)
    events += tr.flush()
    finals = [e for e in events if e.kind == "final"]
    return (" ".join(e.text for e in finals).strip(),
            sum(e.asr_ms for e in finals), len(audio) / SAMPLE_RATE)


def run_e2e(backend, cleaners: Sequence, items: Sequence[dict], wav_dir: str | Path, *,
            endpoint_silence_ms: int, style: str, dictionary: Sequence[str], warmup: int = 1
            ) -> tuple[list[dict], dict]:
    wav_dir = Path(wav_dir)
    prompt = ", ".join(dictionary) if dictionary else None
    usable = [it for it in items if (wav_dir / f"{it['id']}.wav").exists()]
    if not usable:
        raise SystemExit(f"No WAVs found in {wav_dir}. Record some with scripts/record_dataset.py")

    asr_items, asr_ms, rtfs = [], [], []
    for it in usable:
        text, ms, audio_s = transcribe_file(backend, wav_dir / f"{it['id']}.wav",
                                            endpoint_silence_ms=endpoint_silence_ms, prompt=prompt)
        asr_items.append({"id": it["id"], "category": it.get("category", "all"),
                          "input": text, "reference": it["reference"]})
        asr_ms.append(ms)
        rtfs.append(ms / 1000 / max(audio_s, 1e-6))

    rows, details = [], {}
    for cleaner in cleaners:
        res = evaluate(cleaner, asr_items, repeats=1, warmup=warmup, style=style,
                       dictionary=dictionary)
        s = res["summary"]
        clean_ms = [r["latency_ms"][0] for r in res["records"]]
        post = [endpoint_silence_ms + a + c for a, c in zip(asr_ms, clean_ms)]
        rows.append({
            "pipeline": f"ASR + {cleaner.name}" if cleaner.name != "raw" else "ASR only",
            "n": s["n"], "zero_edit_strict": s["zero_edit_strict"],
            "zero_edit_lenient": s["zero_edit_lenient"], "wer": s["wer"],
            "asr_ms": latency_stats(asr_ms), "rtf_p50": latency_stats(rtfs)["p50"],
            "latency_ms": latency_stats(clean_ms), "post_speech_ms": latency_stats(post),
            "fallback_rate": s["fallback_rate"],
        })
        details[cleaner.name] = res
    return rows, details
