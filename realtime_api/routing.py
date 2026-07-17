"""Language -> endpoint routing for the standalone realtime voice API.

Each requested BCP 47 language resolves to a promoted STT/LLM/TTS endpoint,
falling back to the configured defaults when no language-specific route is
promoted yet. Route data is supplied by the caller (from the ``realtime_voice:``
config block); this module stays free of any config-file coupling.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EndpointSet:
    stt_endpoint: str
    llm_endpoint: str
    tts_endpoint: str


@dataclass(frozen=True)
class RouteTable:
    """Resolves per-language endpoints with a guaranteed default."""

    default: EndpointSet
    by_language: dict[str, EndpointSet]

    def resolve(self, language: str) -> EndpointSet:
        if language in self.by_language:
            return self.by_language[language]
        base = language.split("-")[0]
        for tag, endpoints in self.by_language.items():
            if tag.split("-")[0] == base:
                return endpoints
        return self.default

    @classmethod
    def from_dict(cls, raw: dict[str, Any], *, default: EndpointSet) -> "RouteTable":
        routes = raw.get("routes") or {}
        by_language: dict[str, EndpointSet] = {}
        for language, entry in routes.items():
            entry = entry if isinstance(entry, dict) else {}
            by_language[str(language)] = EndpointSet(
                stt_endpoint=str(entry.get("stt_endpoint") or default.stt_endpoint),
                llm_endpoint=str(entry.get("llm_endpoint") or default.llm_endpoint),
                tts_endpoint=str(entry.get("tts_endpoint") or default.tts_endpoint),
            )
        return cls(default=default, by_language=by_language)

    @classmethod
    def from_yaml(cls, path: str | Path, *, default: EndpointSet) -> "RouteTable":
        import yaml

        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.from_dict(raw, default=default)
