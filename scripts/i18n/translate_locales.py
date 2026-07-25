#!/usr/bin/env python3
"""Offline UI-copy localizer for the Genie Voice frontend.

Voice-first UI chrome for every supported language, generated ONCE offline so the
app pays zero translation cost at runtime (the browser just loads a static JSON
bundle). Reads the English message catalog emitted by the frontend extractor
(``frontend/src/locales/en.json``) and, for every config-supported language that
isn't hand-authored, uses the Databricks-served multilingual LLM to translate the
*values* — preserving keys, ``{placeholder}`` tokens, brand names, and canonical
IDs/currency. Results are written to ``frontend/src/locales/<tag>.json`` with a
``_sourceHash`` so staleness is detectable.

Usage (needs Databricks auth, same profile as the realtime API):
    .venv/bin/python scripts/i18n/translate_locales.py                # all langs
    .venv/bin/python scripts/i18n/translate_locales.py fr-FR vi-VN    # subset
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
from realtime_api.services import _SdkDeployClient  # noqa: E402

LOCALES_DIR = REPO_ROOT / "frontend" / "src" / "locales"
EN_PATH = LOCALES_DIR / "en.json"

# Hand-authored locales (kept for top quality) + the zh ASR-comparison variants
# (same audio language) are never machine-translated.
SKIP_CODES = {"en-US", "th-TH", "id-ID", "zh-CN", "zh-CN-sensevoice", "zh-CN-paraformer"}

# Tokens the model must copy verbatim (brands, schema, currency, IDs).
_PROTECTED = (
    "Genie, Databricks, Deepgram, Qwen3, SenseVoice, Paraformer, VoxCPM, STT, TTS, "
    "ASR, Lakebase, Customer 360, USD, the '$' sign, and any INV-/CUST-/CALL- IDs"
)

# Keys are translated in chunks to keep each response well within output limits.
_CHUNK = 40


def _system_prompt(language_name: str, tag: str) -> str:
    return (
        f"You are a professional software localizer. Translate the VALUES of a UI "
        f"string catalog from English into {language_name} ({tag}). "
        "Return ONLY a valid JSON object with EXACTLY the same keys as the input; "
        "translate only the values. "
        f"Keep every {{placeholder}} token EXACTLY as-is (same name, keep the braces). "
        f"Do NOT translate these tokens — copy them verbatim: {_PROTECTED}. "
        "Preserve arrows (→), ellipses (…), and punctuation. Keep translations short "
        "and natural for UI labels/buttons. Do not add notes, comments, or code fences."
    )


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        # drop a leading 'json' language hint if present
        if text.lstrip().startswith("json"):
            text = text.lstrip()[4:]
    return text.strip()


def _repair_escapes(text: str) -> str:
    """Escape stray backslashes so a near-valid JSON string parses.

    JSON only allows \\" \\\\ \\/ \\b \\f \\n \\r \\t \\uXXXX. Any other backslash is
    illegal; the model sometimes emits one before punctuation. Double it so the
    literal backslash (or the intended character) survives parsing.
    """
    import re

    return re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", text)


def _chunks(items: list[tuple[str, str]], size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _translate_chunk(client, endpoint: str, language_name: str, tag: str, chunk: dict[str, str]) -> dict[str, str]:
    messages = [
        {"role": "system", "content": _system_prompt(language_name, tag)},
        {"role": "user", "content": json.dumps(chunk, ensure_ascii=False, indent=0)},
    ]
    resp = client.predict(
        endpoint=endpoint,
        inputs={"messages": messages, "max_tokens": 4000, "temperature": 0.0},
    )
    choices = resp.get("choices") or []
    content = _strip_fences((choices[0].get("message") or {}).get("content") if choices else "")
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        # Some languages occasionally trip the model into emitting a stray backslash
        # (e.g. before an apostrophe) that isn't a valid JSON escape. Escape any
        # backslash not followed by a legal JSON escape char, then re-parse.
        parsed = json.loads(_repair_escapes(content))
    if not isinstance(parsed, dict):
        raise ValueError("model did not return a JSON object")
    return parsed


def main() -> int:
    if not EN_PATH.exists():
        print(f"Missing {EN_PATH}. Run the frontend extractor first (see extract_catalog.ts).")
        return 1

    english = json.loads(EN_PATH.read_text(encoding="utf-8"))
    source_hash = hashlib.sha256(EN_PATH.read_bytes()).hexdigest()[:16]
    items = list(english.items())

    requested = [a for a in sys.argv[1:] if not a.startswith("-")]
    targets = requested or [code for code in LANGUAGE_SPECS if code not in SKIP_CODES]

    settings = RealtimeSettings.resolve()
    client = _SdkDeployClient(databricks_profile())
    endpoint = settings.llm_endpoint
    print(f"LLM endpoint: {endpoint} | languages: {len(targets)} | keys: {len(items)}")

    for tag in targets:
        spec = LANGUAGE_SPECS.get(tag)
        name = spec.english_name if spec else tag
        translated: dict[str, str] = {}
        try:
            for chunk in _chunks(items, _CHUNK):
                translated.update(_translate_chunk(client, endpoint, name, tag, dict(chunk)))
        except Exception as exc:  # noqa: BLE001
            print(f"  ✗ {tag} ({name}): {exc}")
            continue
        # Keep only known keys; fill any the model dropped from English so the
        # bundle is always complete (buildCopy also backfills, this is belt+braces).
        out = {key: str(translated.get(key, english[key])) for key in english}
        out["_sourceHash"] = source_hash
        (LOCALES_DIR / f"{tag}.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"  ✓ {tag} ({name}) -> {len(out) - 1} keys")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
