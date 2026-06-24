"""Shared API dependencies (config-driven, no vendor coupling)."""
from __future__ import annotations

from functools import lru_cache

from genie_voice.config import get_settings
from genie_voice.genie import GenieClient
from genie_voice.serve import LakebaseServing


def settings_dep():
    return get_settings()


@lru_cache(maxsize=1)
def serving() -> LakebaseServing:
    return LakebaseServing(get_settings())


@lru_cache(maxsize=1)
def genie() -> GenieClient:
    return GenieClient(get_settings())
