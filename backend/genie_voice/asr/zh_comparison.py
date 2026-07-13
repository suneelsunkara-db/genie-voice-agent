"""Background zh-CN ASR model comparison for live mic prompts."""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from genie_voice.databricks.client import get_workspace_client
from genie_voice.i18n import asr_model_language, normalize_language

log = logging.getLogger(__name__)

_ZH_COMPARE_MODELS: tuple[dict[str, str], ...] = (
    {"key": "sensevoice", "label": "SenseVoice-Small", "route": "zh-CN-sensevoice"},
    {"key": "paraformer_8k", "label": "Paraformer-8k", "route": "zh-CN-paraformer"},
    {"key": "qwen3", "label": "Qwen3-ASR-0.6B", "route": "zh-CN"},
)

_RECENT: deque[dict[str, Any]] = deque(maxlen=50)
_LOCK = threading.Lock()
_LOG_DIR = Path(".run/zh_asr_comparison")


@dataclass
class ZhAsrModelResult:
    key: str
    label: str
    endpoint: str
    transcript: str = ""
    error: str | None = None
    elapsed_ms: int | None = None
    selected: bool = False


@dataclass
class ZhAsrComparison:
    comparison_id: str
    call_id: str
    created_at: str
    selected_language: str
    primary_transcript: str
    browser_caption: str | None = None
    mime_type: str | None = None
    status: str = "pending"
    models: list[ZhAsrModelResult] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["models"] = [asdict(item) for item in self.models]
        return payload


def _endpoint_for_route(settings: Any, route_language: str) -> str:
    from genie_voice.i18n import stt_options_for_language

    options = stt_options_for_language(settings, route_language)
    return str(options.get("endpoint") or "").strip()


def _query_endpoint(
    settings: Any,
    *,
    endpoint: str,
    audio_b64: str,
    mime_type: str,
    speaker: int,
    language: str,
) -> tuple[str, str | None]:
    client = get_workspace_client(settings)
    response = client.serving_endpoints.query(
        name=endpoint,
        dataframe_records=[
            {
                "audio_b64": audio_b64,
                "mime_type": mime_type,
                "speaker": speaker,
                "language": asr_model_language(language),
                "task": "transcribe",
            }
        ],
    )
    payload = response.as_dict() if hasattr(response, "as_dict") else dict(response)
    predictions = payload.get("predictions") or []
    first = predictions[0] if predictions else {}
    transcript = str(first.get("raw_transcript") or first.get("transcript") or "").strip()
    if not transcript:
        return "", "empty transcript"
    return transcript, None


def _append_log(record: dict[str, Any]) -> None:
    try:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        path = _LOG_DIR / "comparisons.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:  # noqa: BLE001
        log.warning("zh asr comparison log write failed: %s", exc)


def _store(record: dict[str, Any]) -> None:
    with _LOCK:
        _RECENT.appendleft(record)
    _append_log(record)


def list_recent_comparisons(*, call_id: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
    with _LOCK:
        items = list(_RECENT)
    if call_id:
        items = [item for item in items if item.get("call_id") == call_id]
    return items[: max(1, min(limit, 50))]


def schedule_zh_asr_comparison(
    *,
    call_id: str,
    audio_b64: str,
    mime_type: str,
    speaker: int,
    selected_language: str,
    primary_transcript: str,
    settings: Any,
    browser_caption: str | None = None,
) -> str:
    """Run all zh ASR endpoints in the background and store side-by-side results."""
    comparison_id = str(uuid.uuid4())
    selected = normalize_language(selected_language)
    record = ZhAsrComparison(
        comparison_id=comparison_id,
        call_id=call_id,
        created_at=datetime.now(UTC).isoformat(),
        selected_language=selected,
        primary_transcript=primary_transcript,
        browser_caption=(browser_caption or "").strip() or None,
        mime_type=mime_type,
        status="running",
    )
    _store(record.to_dict())

    def _run() -> None:
        models: list[ZhAsrModelResult] = []
        notes: list[str] = []
        for spec in _ZH_COMPARE_MODELS:
            endpoint = _endpoint_for_route(settings, spec["route"])
            if not endpoint:
                models.append(
                    ZhAsrModelResult(
                        key=spec["key"],
                        label=spec["label"],
                        endpoint="",
                        error="endpoint not configured",
                        selected=spec["route"] == selected,
                    )
                )
                continue
            started = time.perf_counter()
            try:
                transcript, error = _query_endpoint(
                    settings,
                    endpoint=endpoint,
                    audio_b64=audio_b64,
                    mime_type=mime_type,
                    speaker=speaker,
                    language=spec["route"],
                )
                models.append(
                    ZhAsrModelResult(
                        key=spec["key"],
                        label=spec["label"],
                        endpoint=endpoint,
                        transcript=transcript,
                        error=error,
                        elapsed_ms=round((time.perf_counter() - started) * 1000),
                        selected=spec["route"] == selected,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                models.append(
                    ZhAsrModelResult(
                        key=spec["key"],
                        label=spec["label"],
                        endpoint=endpoint,
                        error=str(exc),
                        elapsed_ms=round((time.perf_counter() - started) * 1000),
                        selected=spec["route"] == selected,
                    )
                )

        transcripts = [m.transcript for m in models if m.transcript]
        unique = {t for t in transcripts if t}
        if len(unique) <= 1:
            notes.append("all models agree" if unique else "no transcripts returned")
        else:
            notes.append(f"{len(unique)} distinct transcripts")

        selected_model = next((m for m in models if m.selected), None)
        if selected_model and primary_transcript and selected_model.transcript:
            if selected_model.transcript != primary_transcript:
                notes.append("primary path differed from shadow replay")

        finished = ZhAsrComparison(
            comparison_id=comparison_id,
            call_id=call_id,
            created_at=record.created_at,
            selected_language=selected,
            primary_transcript=primary_transcript,
            browser_caption=record.browser_caption,
            mime_type=mime_type,
            status="done",
            models=models,
            notes=notes,
        )
        payload = finished.to_dict()
        with _LOCK:
            for idx, item in enumerate(_RECENT):
                if item.get("comparison_id") == comparison_id:
                    _RECENT[idx] = payload
                    break
            else:
                _RECENT.appendleft(payload)
        _append_log(payload)
        log.info(
            "zh asr comparison done call_id=%s selected=%s models=%s notes=%s",
            call_id,
            selected,
            [m.key for m in models],
            notes,
        )

    threading.Thread(target=_run, daemon=True, name=f"zh-asr-compare-{comparison_id[:8]}").start()
    return comparison_id
