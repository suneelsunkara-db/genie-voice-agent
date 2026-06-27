"""Gold manifest loading for ASR model training and evaluation.

The manifest is JSONL so it works locally, in Databricks jobs, and as a simple
export from a Delta table. Each line describes one utterance-level audio clip and
the human-approved reference transcript/entities used for scoring.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class ExpectedEntities:
    invoice_ids: list[str] = field(default_factory=list)
    amounts: list[str] = field(default_factory=list)
    dates: list[str] = field(default_factory=list)
    billing_actions: list[str] = field(default_factory=list)
    confirmations: list[str] = field(default_factory=list)
    refusals: list[str] = field(default_factory=list)
    account_terms: list[str] = field(default_factory=list)

    @classmethod
    def from_raw(cls, raw: dict[str, Any] | None) -> "ExpectedEntities":
        raw = raw or {}
        return cls(
            invoice_ids=_string_list(raw.get("invoice_ids")),
            amounts=_string_list(raw.get("amounts")),
            dates=_string_list(raw.get("dates")),
            billing_actions=_string_list(raw.get("billing_actions")),
            confirmations=_string_list(raw.get("confirmations")),
            refusals=_string_list(raw.get("refusals")),
            account_terms=_string_list(raw.get("account_terms")),
        )

    def groups(self) -> dict[str, list[str]]:
        return {
            "invoice_ids": self.invoice_ids,
            "amounts": self.amounts,
            "dates": self.dates,
            "billing_actions": self.billing_actions,
            "confirmations": self.confirmations,
            "refusals": self.refusals,
            "account_terms": self.account_terms,
        }


@dataclass(frozen=True)
class ASRGoldClip:
    clip_id: str
    audio_path: str
    reference_transcript: str
    call_id: str | None = None
    speaker: str | int | None = None
    audio_format: str | None = None
    sample_rate_hz: int | None = None
    duration_seconds: float | None = None
    domain: str | None = None
    scenario: str | None = None
    split: str | None = None
    dataset_version: str | None = None
    expected_entities: ExpectedEntities = field(default_factory=ExpectedEntities)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_raw(cls, raw: dict[str, Any]) -> "ASRGoldClip":
        required = ("clip_id", "audio_path", "reference_transcript")
        missing = [key for key in required if not raw.get(key)]
        if missing:
            raise ValueError(f"Manifest row missing required fields: {missing}")

        known = {
            "clip_id",
            "audio_path",
            "reference_transcript",
            "call_id",
            "speaker",
            "audio_format",
            "sample_rate_hz",
            "duration_seconds",
            "domain",
            "scenario",
            "split",
            "dataset_version",
            "expected_entities",
        }
        metadata = {key: value for key, value in raw.items() if key not in known}
        return cls(
            clip_id=str(raw["clip_id"]),
            audio_path=str(raw["audio_path"]),
            reference_transcript=str(raw["reference_transcript"]),
            call_id=_optional_str(raw.get("call_id")),
            speaker=raw.get("speaker"),
            audio_format=_optional_str(raw.get("audio_format")),
            sample_rate_hz=_optional_int(raw.get("sample_rate_hz")),
            duration_seconds=_optional_float(raw.get("duration_seconds")),
            domain=_optional_str(raw.get("domain")),
            scenario=_optional_str(raw.get("scenario")),
            split=_optional_str(raw.get("split")),
            dataset_version=_optional_str(raw.get("dataset_version")),
            expected_entities=ExpectedEntities.from_raw(raw.get("expected_entities")),
            metadata=metadata,
        )


def load_manifest(path: str | Path, *, splits: Iterable[str] | None = None) -> list[ASRGoldClip]:
    """Load a JSONL ASR manifest.

    Blank lines and comment lines starting with `#` are ignored. If `splits` is
    provided, only rows with a matching `split` value are returned.
    """
    wanted = {str(split) for split in splits} if splits else None
    clips: list[ASRGoldClip] = []
    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            try:
                raw = json.loads(text)
                clip = ASRGoldClip.from_raw(raw)
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"Invalid manifest row {manifest_path}:{line_no}: {exc}") from exc
            if wanted and clip.split not in wanted:
                continue
            clips.append(clip)
    return clips


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)]


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)
