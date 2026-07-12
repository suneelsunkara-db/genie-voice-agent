from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from genie_voice.asr_eval.manifest import ASRGoldClip, ExpectedEntities, load_manifest


@dataclass(frozen=True)
class EvalManifest:
    path: str
    clips: tuple[ASRGoldClip, ...]

    def __len__(self) -> int:
        return len(self.clips)


def load_eval_manifest(path: str | Path, *, splits: Iterable[str] | None = None) -> EvalManifest:
    clips = tuple(load_manifest(path, splits=splits))
    return EvalManifest(path=str(path), clips=clips)


def write_manifest_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    if rows:
        text += "\n"
    target.write_text(text, encoding="utf-8")


def empty_entities() -> dict[str, list[str]]:
    return {
        "invoice_ids": [],
        "amounts": [],
        "billing_actions": [],
        "confirmations": [],
        "refusals": [],
        "account_terms": [],
    }


def entities_from_row(raw: dict[str, Any] | None) -> ExpectedEntities:
    return ExpectedEntities.from_raw(raw or empty_entities())
