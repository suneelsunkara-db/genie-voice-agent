"""Provider registry / factory - fully vendor-agnostic.

The core code never imports a vendor adapter. Instead, config declares an
`adapters` map of {logical_name: "module.path:ClassName"} and an `active` name.
This resolver dynamically imports the active adapter at runtime.

Consequences:
  - No vendor (Deepgram, ElevenLabs, ...) is referenced anywhere in core code.
  - Swapping/adding a vendor = drop an adapter file + add one config line.
  - Tests can inject fakes via `register_stt`/`register_tts` (takes precedence).
"""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Type

from .base import STTProvider, TTSProvider

if TYPE_CHECKING:  # avoid importing config (yaml/pydantic) just to import an adapter
    from genie_voice.config import Settings
    from genie_voice.config.settings import ProviderSlot

# Optional in-process overrides (e.g. tests). Empty by default; config-driven
# dynamic import is the normal path.
_STT_OVERRIDES: dict[str, Type[STTProvider]] = {}
_TTS_OVERRIDES: dict[str, Type[TTSProvider]] = {}


def register_stt(name: str, cls: Type[STTProvider]) -> None:
    _STT_OVERRIDES[name] = cls


def register_tts(name: str, cls: Type[TTSProvider]) -> None:
    _TTS_OVERRIDES[name] = cls


def _load_class(dotted: str) -> type:
    module_path, _, class_name = dotted.partition(":")
    if not module_path or not class_name:
        raise ValueError(
            f"Adapter path must be 'module.path:ClassName', got '{dotted}'"
        )
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _build(slot: "ProviderSlot", base_cls: type, kind: str, overrides: dict):
    name = slot.active
    if name in overrides:
        cls = overrides[name]
    else:
        if name not in slot.adapters:
            raise ValueError(
                f"Active {kind} provider '{name}' is not declared in "
                f"providers.{kind.lower()}.adapters ({list(slot.adapters)})"
            )
        cls = _load_class(slot.adapters[name])
    if not (isinstance(cls, type) and issubclass(cls, base_cls)):
        raise TypeError(f"Adapter '{name}' ({cls}) is not a {base_cls.__name__}")
    return cls(slot.active_options())


def get_stt_provider(settings: "Settings") -> STTProvider:
    return _build(settings.providers.stt, STTProvider, "STT", _STT_OVERRIDES)


def get_tts_provider(settings: "Settings") -> TTSProvider:
    return _build(settings.providers.tts, TTSProvider, "TTS", _TTS_OVERRIDES)
