"""Settings: environment variables first, CLI flags override."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass
class Settings:
    asr_model: str = field(default_factory=lambda: _env("FLOWLITE_ASR_MODEL", "base.en"))
    device: str = field(default_factory=lambda: _env("FLOWLITE_DEVICE", "auto"))
    compute_type: str = field(default_factory=lambda: _env("FLOWLITE_COMPUTE_TYPE", "int8"))
    language: str = field(default_factory=lambda: _env("FLOWLITE_LANGUAGE", "en"))
    cleaner: str = field(default_factory=lambda: _env("FLOWLITE_CLEANER", "llm"))
    llm_url: str = field(default_factory=lambda: _env("FLOWLITE_LLM_BASE_URL",
                                                       "http://localhost:11434/v1"))
    llm_model: str = field(default_factory=lambda: _env("FLOWLITE_LLM_MODEL", "qwen2.5:1.5b"))
    llm_key: str = field(default_factory=lambda: _env("FLOWLITE_LLM_API_KEY", "ollama"))
    style: str = "neutral"
    endpoint_ms: int = 500
    partial_s: float = 1.0


def build_asr(s: Settings):
    from .asr import FasterWhisperBackend
    lang = None if s.language.lower() in ("", "auto") else s.language
    return FasterWhisperBackend(s.asr_model, s.device, s.compute_type, lang)


def build_cleaner(s: Settings, kind: str | None = None):
    from .cleanup import make_cleaner
    return make_cleaner(kind or s.cleaner, base_url=s.llm_url, model=s.llm_model,
                        api_key=s.llm_key)
