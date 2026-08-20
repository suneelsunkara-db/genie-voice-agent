"""Cooperative cancellation for progressive turns."""
from __future__ import annotations

import asyncio


class CancellationToken:
    """Checked by long-running work; cancel drops later events for the turn."""

    def __init__(self) -> None:
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def raise_if_cancelled(self) -> None:
        if self._cancelled:
            raise asyncio.CancelledError()
