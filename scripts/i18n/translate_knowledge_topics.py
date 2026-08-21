#!/usr/bin/env python3
"""Generate the Knowledge Agent's localized question catalog offline.

The page's category headings, questions, previews, topics, and source labels are
product content, not governed workspace facts. They are translated once and
committed so changing language is instant and does not add an LLM call to a live
voice turn.

The canonical English question remains a separate identity/audit field. The
caller-language question is the primary display contract and is also suitable for
a future click-to-ask action; Genie One accepts multilingual questions.

Usage:
    .venv/bin/python scripts/i18n/translate_knowledge_topics.py
    .venv/bin/python scripts/i18n/translate_knowledge_topics.py hi-IN th-TH
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))
sys.path.insert(0, str(REPO_ROOT))

from genie_voice.i18n import LANGUAGE_SPECS  # noqa: E402
from realtime_api.config import RealtimeSettings, databricks_profile  # noqa: E402
from realtime_api.knowledge_tools import KNOWLEDGE_CATEGORIES, WORKSPACE_PROMPTS  # noqa: E402
from realtime_api.services import _SdkDeployClient  # noqa: E402

OUT_PATH = REPO_ROOT / "realtime_api" / "phrases" / "knowledge_topics.json"
SKIP_CODES = {"en-US", "zh-CN-sensevoice", "zh-CN-paraformer"}
FIELDS = ("topic", "question", "preview", "source")
CHUNK_SIZE = 35

_SYSTEM = (
    "You are a professional product localizer. Translate the VALUES of this JSON "
    "catalog from English into {name} ({tag}) for a Databricks voice application's "
    "Knowledge Agent page. The values are category headings, suggested questions, "
    "short descriptions of what asking will do, and compact source labels. "
    "Return ONLY a valid JSON object with EXACTLY the same keys. Translate only "
    "values. Keep Genie One, Genie Agents, Databricks, Lakebase, Unity Catalog, "
    "SQL, API, AI/BI, metric view, dashboard, and product names verbatim where "
    "that is the established product term. Do not add facts, promises, examples, "
    "or capabilities. Preserve punctuation including · and question marks. Use "
    "natural, professional language for a business user, not a word-for-word "
    "translation. Keep questions easy to say aloud and descriptions concise. "
    "Do not add notes, comments, markdown, or code fences."
)


def _source() -> dict[str, object]:
    return {
        "categories": list(KNOWLEDGE_CATEGORIES),
        "topics": {
            str(prompt["id"]): {field: str(prompt[field]) for field in FIELDS}
            for prompt in WORKSPACE_PROMPTS
        },
    }


def _source_hash(source: dict[str, object]) -> str:
    encoded = json.dumps(source, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _flatten(source: dict[str, object]) -> dict[str, str]:
    values: dict[str, str] = {}
    for index, category in enumerate(source["categories"]):
        values[f"category::{index}"] = str(category)
    for topic_id, topic in source["topics"].items():
        for field in FIELDS:
            values[f"topic::{topic_id}::{field}"] = str(topic[field])
    return values


def _build(flat: dict[str, str], source: dict[str, object], source_hash: str) -> dict[str, object]:
    categories = {
        str(category): flat[f"category::{index}"]
        for index, category in enumerate(source["categories"])
    }
    topics = {
        str(topic_id): {
            field: flat[f"topic::{topic_id}::{field}"]
            for field in FIELDS
        }
        for topic_id in source["topics"]
    }
    return {"_sourceHash": source_hash, "categories": categories, "topics": topics}


def _chunks(items: list[tuple[str, str]]):
    for index in range(0, len(items), CHUNK_SIZE):
        yield dict(items[index : index + CHUNK_SIZE])


def _strip_fences(text: str) -> str:
    text = text.strip()
    if not text.startswith("```"):
        return text
    text = text.split("\n", 1)[1] if "\n" in text else text
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    if text.lstrip().startswith("json"):
        text = text.lstrip()[4:]
    return text.strip()


def _translate(client, endpoint: str, name: str, tag: str, values: dict[str, str]) -> dict[str, str]:
    translated: dict[str, str] = {}
    for chunk in _chunks(list(values.items())):
        response = client.predict(
            endpoint=endpoint,
            inputs={
                "messages": [
                    {"role": "system", "content": _SYSTEM.format(name=name, tag=tag)},
                    {"role": "user", "content": json.dumps(chunk, ensure_ascii=False)},
                ],
                "max_tokens": 5000,
            },
        )
        choices = response.get("choices") or []
        content = _strip_fences(
            (choices[0].get("message") or {}).get("content") if choices else ""
        )
        parsed = json.loads(content)
        if not isinstance(parsed, dict) or set(parsed) != set(chunk):
            raise ValueError("model did not return exactly the requested catalog keys")
        translated.update({key: str(value).strip() for key, value in parsed.items()})
    return translated


def main() -> int:
    source = _source()
    source_hash = _source_hash(source)
    english_flat = _flatten(source)
    requested = [arg for arg in sys.argv[1:] if not arg.startswith("-")]
    english_only = "--english-only" in sys.argv
    targets = (
        []
        if english_only
        else requested or [tag for tag in LANGUAGE_SPECS if tag not in SKIP_CODES]
    )

    existing: dict[str, object] = {}
    if OUT_PATH.exists():
        try:
            existing = json.loads(OUT_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
    existing["en-US"] = _build(english_flat, source, source_hash)

    settings = RealtimeSettings.resolve()
    client = _SdkDeployClient(databricks_profile(), predict_timeout_s=180.0)
    endpoint = next(
        (arg.split("=", 1)[1] for arg in sys.argv[1:] if arg.startswith("--endpoint=")),
        settings.i18n_endpoint or settings.llm_endpoint,
    )
    print(
        f"LLM endpoint: {endpoint} | languages: {len(targets)} | "
        f"display strings: {len(english_flat)}"
    )

    failures = 0
    for tag in targets:
        spec = LANGUAGE_SPECS.get(tag)
        name = spec.english_name if spec else tag
        try:
            translated = _translate(client, endpoint, name, tag, english_flat)
            existing[tag] = _build(translated, source, source_hash)
            print(f"  ok {tag} ({name})")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  x {tag} ({name}): {exc}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps({key: existing[key] for key in sorted(existing)}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(existing)} language catalogs -> {OUT_PATH}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
