#!/usr/bin/env python
"""Record your own voice for the end-to-end benchmark.

For each sample it shows the script to read (speak it NATURALLY, including the
um/uh and self-corrections), records until you press Enter, and writes
data/audio/<id>.wav (16 kHz mono). Then run:  flowlite bench-e2e

  python scripts/record_dataset.py                 # all samples
  python scripts/record_dataset.py --only f1,c1    # a subset
  python scripts/record_dataset.py --overwrite     # re-record existing files
"""
from __future__ import annotations

import argparse
import queue
from pathlib import Path

import numpy as np
import sounddevice as sd

from flowlite.audio_io import write_wav
from flowlite.bench import load_samples
from flowlite.vad import SAMPLE_RATE


def record() -> np.ndarray:
    q: queue.Queue = queue.Queue()
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                        callback=lambda indata, *_: q.put(indata[:, 0].copy())):
        input("   ● recording... press Enter to stop ")
    chunks = []
    while not q.empty():
        chunks.append(q.get())
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/samples.jsonl")
    ap.add_argument("--out", default="data/audio")
    ap.add_argument("--only", default="")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    only = {x for x in args.only.split(",") if x}
    for it in load_samples(args.data):
        if only and it["id"] not in only:
            continue
        path = Path(args.out) / f"{it['id']}.wav"
        if path.exists() and not args.overwrite:
            print(f"skip {it['id']} (exists)")
            continue
        while True:
            print(f"\n[{it['id']} / {it['category']}]  SAY:  {it.get('speak', it['raw'])}")
            input("   press Enter to start ")
            audio = record()
            if len(audio) < SAMPLE_RATE // 2:
                print("   too short, try again")
                continue
            write_wav(path, audio)
            print(f"   saved {path} ({len(audio) / SAMPLE_RATE:.1f}s)")
            if input("   keep? [Y/n] ").strip().lower() != "n":
                break


if __name__ == "__main__":
    main()
