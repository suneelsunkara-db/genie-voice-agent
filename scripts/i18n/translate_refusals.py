#!/usr/bin/env python3
"""Offline localizer for the SPOKEN refusal catalog.

The refusal line is what the caller hears when a tool failed, a permission check
denied them, or the governed read timed out. It has to be reliable exactly when
the rest of the turn is not, so it is translated once, offline, and committed —
the same contract as the frontend message catalog, for the same reason.

Writes ``realtime_api/phrases/refusals.json`` as ``{tag: {error_code: text}}``.
English is the source and is written verbatim, never through the model.

Usage (needs Databricks auth, same profile as the realtime API):
    .venv/bin/python scripts/i18n/translate_refusals.py              # all langs
    .venv/bin/python scripts/i18n/translate_refusals.py fr-FR hi-IN  # subset
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))
sys.path.insert(0, str(REPO_ROOT))

from genie_voice.i18n import LANGUAGE_SPECS  # noqa: E402
from realtime_api.config import RealtimeSettings, databricks_profile  # noqa: E402
from realtime_api.runtime.refuse import _REFUSE_EN  # noqa: E402
from realtime_api.services import _SdkDeployClient  # noqa: E402

OUT_PATH = REPO_ROOT / "realtime_api" / "phrases" / "refusals.json"

# English is the source of truth and is emitted as authored.
SKIP_CODES = {"en-US"}

_PROTECTED = "Databricks, Genie, and the product names they appear in"

_SYSTEM = (
    "You are a professional software localizer working on a REALTIME VOICE "
    "assistant. Translate the VALUES of this catalog from English into "
    "{name} ({tag}). Each value is spoken aloud to a caller at the moment the "
    "assistant cannot answer: it either found no data it can cite, lacks "
    "permission, timed out, was interrupted, cannot help with the request, or "
    "needs the caller to clarify. "
    "Return ONLY a valid JSON object with EXACTLY the same keys as the input; "
    "translate only the values. "
    f"Do NOT translate these tokens — copy them verbatim: {_PROTECTED}. "
    "Write for the EAR, not the eye: natural spoken register, no markdown, no "
    "bullets, no abbreviations a text-to-speech voice would mispronounce. Use the "
    "level of politeness a bank or telecom call centre would use with a customer "
    "in this language, including the appropriate formality of address. "
    "Never promise an outcome, never blame the caller, and never claim the "
    "assistant did something it did not do. Do not add notes or code fences."
)


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


def main() -> int:
    english = {code.value: text for code, text in _REFUSE_EN.items()}
    requested = [arg for arg in sys.argv[1:] if not arg.startswith("-")]
    targets = requested or [tag for tag in LANGUAGE_SPECS if tag not in SKIP_CODES]

    settings = RealtimeSettings.resolve()
    client = _SdkDeployClient(databricks_profile(), predict_timeout_s=180.0)
    # Offline, so not bound to the low-latency call-path model. Register and
    # politeness are exactly where a small fast model produces the wrong thing.
    endpoint = next(
        (a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--endpoint=")),
        settings.i18n_endpoint or settings.llm_endpoint,
    )
    print(f"LLM endpoint: {endpoint} | languages: {len(targets)} | codes: {len(english)}")

    existing: dict[str, dict[str, str]] = {}
    if OUT_PATH.exists():
        try:
            existing = json.loads(OUT_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}

    out: dict[str, dict[str, str]] = dict(existing)
    out["en-US"] = english

    for tag in targets:
        spec = LANGUAGE_SPECS.get(tag)
        name = spec.english_name if spec else tag
        try:
            response = client.predict(
                endpoint=endpoint,
                inputs={
                    "messages": [
                        {"role": "system", "content": _SYSTEM.format(name=name, tag=tag)},
                        {
                            "role": "user",
                            "content": json.dumps(english, ensure_ascii=False, indent=0),
                        },
                    ],
                    "max_tokens": 1200,
                },
            )
            choices = response.get("choices") or []
            content = _strip_fences(
                (choices[0].get("message") or {}).get("content") if choices else ""
            )
            parsed = json.loads(content)
            if not isinstance(parsed, dict):
                raise ValueError("model did not return a JSON object")
        except Exception as exc:  # noqa: BLE001
            print(f"  x {tag} ({name}): {exc}")
            continue
        # A dropped key would silently speak English; fill it so the bundle is
        # always complete and the gap is visible in review instead.
        out[tag] = {key: str(parsed.get(key) or english[key]).strip() for key in english}
        missing = [key for key in english if not parsed.get(key)]
        note = f" (untranslated: {', '.join(missing)})" if missing else ""
        print(f"  ok {tag} ({name}){note}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps({key: out[key] for key in sorted(out)}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(out)} languages -> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
