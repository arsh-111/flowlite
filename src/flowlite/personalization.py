"""'Correct it twice, never again' - a tiny correction memory.

Every time the user fixes a mistake (wrong -> right) we count it. Once the same
correction has been seen `threshold` times it becomes active and is applied to the raw
ASR text *before* the LLM sees it. Stored locally as JSON; nothing leaves the machine.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

DEFAULT_PATH = Path.home() / ".flowlite" / "corrections.json"


class CorrectionMemory:
    def __init__(self, path: Path | str | None = DEFAULT_PATH, threshold: int = 2) -> None:
        self.path = Path(path) if path else None
        self.threshold = threshold
        self.counts: dict[str, int] = {}
        self.active: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if not self.path or not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.counts = {str(k): int(v) for k, v in data.get("counts", {}).items()}
            self.active = {str(k): str(v) for k, v in data.get("active", {}).items()}
        except (OSError, ValueError, TypeError):
            self.counts, self.active = {}, {}

    def _save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"counts": self.counts, "active": self.active},
                                  indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)

    def learn(self, wrong: str, right: str) -> bool:
        """Record one user correction. Returns True if it is now active."""
        wrong = " ".join(wrong.lower().split())
        right = " ".join(right.split())
        if not wrong or not right or wrong == right.lower():
            return False
        key = f"{wrong}=>{right}"
        self.counts[key] = self.counts.get(key, 0) + 1
        if self.counts[key] >= self.threshold:
            self.active[wrong] = right
        self._save()
        return wrong in self.active

    def apply(self, text: str) -> str:
        # longest phrases first so multi-word corrections win
        for wrong in sorted(self.active, key=len, reverse=True):
            text = re.sub(rf"\b{re.escape(wrong)}\b", self.active[wrong], text, flags=re.I)
        return text

    def terms(self) -> list[str]:
        return sorted(set(self.active.values()))
