#!/usr/bin/env python3
"""Offline localizer for deterministic runtime speech."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from string import Formatter

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))
sys.path.insert(0, str(REPO_ROOT))

from genie_voice.i18n import LANGUAGE_SPECS  # noqa: E402
from genie_voice.databricks.ai_gateway import inference_context  # noqa: E402
from realtime_api.config import RealtimeSettings, databricks_profile  # noqa: E402
from realtime_api.runtime.phrases import ENGLISH  # noqa: E402
from realtime_api.services import _SdkDeployClient  # noqa: E402

OUT_PATH = REPO_ROOT / "realtime_api" / "phrases" / "runtime.json"

SYSTEM = (
    "Translate the VALUES of this JSON catalog from English into {name} ({tag}) "
    "for a real-time voice assistant. Return only a valid JSON object with exactly "
    "the same keys. Preserve every {{placeholder}} character-for-character. "
    "Databricks, Genie, Telco, Financial Services, Knowledge Agent, Statement "
    "Insights, and Rewards Optimizer are product names; copy them verbatim. Write "
    "natural, concise spoken language suitable for text-to-speech. Do not add "
    "claims, promises, notes, markdown, or code fences."
)


def _fields(value: str) -> set[str]:
    return {
        str(field)
        for _, field, _, _ in Formatter().parse(value)
        if field is not None
    }


def _strip_fences(value: str) -> str:
    value = value.strip()
    if value.startswith("```"):
        value = value.split("\n", 1)[1] if "\n" in value else value
        value = value.rsplit("```", 1)[0]
        if value.lstrip().startswith("json"):
            value = value.lstrip()[4:]
    return value.strip()


def main() -> int:
    requested = [arg for arg in sys.argv[1:] if not arg.startswith("--")]
    targets = requested or [tag for tag in LANGUAGE_SPECS if tag != "en-US"]
    settings = RealtimeSettings.resolve()
    client = _SdkDeployClient(databricks_profile(), predict_timeout_s=180.0)
    endpoint = next(
        (arg.split("=", 1)[1] for arg in sys.argv[1:] if arg.startswith("--endpoint=")),
        settings.i18n_endpoint or settings.llm_endpoint,
    )
    try:
        existing = json.loads(OUT_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        existing = {}
    output: dict[str, dict[str, str]] = dict(existing)
    output["en-US"] = dict(ENGLISH)

    for tag in targets:
        spec = LANGUAGE_SPECS.get(tag)
        name = spec.english_name if spec else tag
        current = output.get(tag) if isinstance(output.get(tag), dict) else {}
        source = {
            key: value
            for key, value in ENGLISH.items()
            if not str(current.get(key) or "").strip()
            or _fields(str(current.get(key) or "")) != _fields(value)
        }
        if not source:
            print(f"ok {tag} ({name}) [already complete]")
            continue
        try:
            with inference_context(
                {
                    "traffic_class": "i18n_offline",
                    "surface": "offline_i18n",
                    "profile": "none",
                    "capability": "translate_runtime_phrases",
                    "model_role": "i18n",
                }
            ):
                response = client.predict(
                    endpoint=endpoint,
                    inputs={
                        "messages": [
                            {"role": "system", "content": SYSTEM.format(name=name, tag=tag)},
                            {
                                "role": "user",
                                "content": json.dumps(source, ensure_ascii=False),
                            },
                        ],
                        "max_tokens": 2200,
                    },
                )
            choices = response.get("choices") or []
            content = _strip_fences(
                str((choices[0].get("message") or {}).get("content") or "")
                if choices
                else ""
            )
            translated = json.loads(content)
            if set(translated) != set(source):
                raise ValueError("translated catalog keys do not match English")
            invalid = [
                key
                for key, value in source.items()
                if _fields(str(translated[key])) != _fields(value)
            ]
            if invalid:
                raise ValueError(f"placeholder mismatch: {invalid}")
            output[tag] = {
                **{key: str(value) for key, value in current.items()},
                **{key: str(translated[key]).strip() for key in source},
            }
            print(f"ok {tag} ({name})")
        except Exception as exc:  # noqa: BLE001
            print(f"x {tag} ({name}): {exc}")

    OUT_PATH.write_text(
        json.dumps(
            {key: output[key] for key in sorted(output)},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    missing = sorted(set(LANGUAGE_SPECS) - set(output))
    if missing:
        print("missing languages: " + ", ".join(missing))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
