"""Lightweight structured observability for the realtime voice API.

Emits one JSON log line per lifecycle/turn event with a stable schema so the new
service can be traced independently of the existing application's logging.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

_logger = logging.getLogger("realtime_voice")


def new_session_id() -> str:
    return uuid.uuid4().hex


@dataclass
class TurnTimings:
    stages: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.stages[name] = round((time.perf_counter() - start) * 1000, 2)

    def as_dict(self) -> dict[str, float]:
        return dict(self.stages)


def log_event(
    event: str,
    *,
    session_id: str,
    turn_id: int | None = None,
    language: str | None = None,
    **fields: object,
) -> None:
    record = {"event": event, "session_id": session_id}
    if turn_id is not None:
        record["turn_id"] = turn_id
    if language is not None:
        record["language"] = language
    record.update({key: value for key, value in fields.items() if value is not None})
    _logger.info(json.dumps(record, ensure_ascii=False))
