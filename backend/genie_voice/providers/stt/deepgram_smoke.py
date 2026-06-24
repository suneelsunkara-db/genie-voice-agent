"""Credit-safe Deepgram validation helper.

Usage:
  python -m genie_voice.providers.stt.deepgram_smoke
  python -m genie_voice.providers.stt.deepgram_smoke --listen-once

Notes:
  - Reads DEEPGRAM_API_KEY from settings/env (.env preferred).
  - Default mode only calls /v1/projects (auth check, near-zero cost).
  - --listen-once performs exactly ONE prerecorded transcription request.
"""

from __future__ import annotations

import argparse
import json
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from genie_voice.config import get_settings


def _deepgram_key() -> str:
    key = get_settings().secrets.deepgram_api_key.strip()
    if not key:
        raise RuntimeError(
            "DEEPGRAM_API_KEY is empty. Put it in .env (not config/.env.example)."
        )
    return key


def _get_json(url: str, key: str) -> tuple[int, dict]:
    req = Request(url, method="GET")
    req.add_header("Authorization", f"Token {key}")
    req.add_header("Accept", "application/json")
    with urlopen(req, timeout=20) as resp:
        status = resp.getcode()
        body = json.loads(resp.read().decode("utf-8"))
        return status, body


def _post_json(url: str, payload: dict, key: str) -> tuple[int, dict]:
    data = json.dumps(payload).encode("utf-8")
    req = Request(url, data=data, method="POST")
    req.add_header("Authorization", f"Token {key}")
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/json")
    with urlopen(req, timeout=45) as resp:
        status = resp.getcode()
        body = json.loads(resp.read().decode("utf-8"))
        return status, body


def _projects_check(key: str) -> None:
    status, body = _get_json("https://api.deepgram.com/v1/projects", key)
    projects = body.get("projects") or body.get("results") or []
    first_name = projects[0].get("name", "unknown") if projects else "none"
    print(f"[ok] projects endpoint status={status}")
    print(f"[ok] projects found={len(projects)} first={first_name}")


def _listen_once(key: str, audio_url: str, model: str) -> None:
    status, body = _post_json(
        f"https://api.deepgram.com/v1/listen?model={model}&smart_format=true",
        {"url": audio_url},
        key,
    )
    ch = (body.get("results", {}).get("channels") or [{}])[0]
    alt = (ch.get("alternatives") or [{}])[0]
    transcript = (alt.get("transcript") or "").strip()
    conf = alt.get("confidence")
    print(f"[ok] listen endpoint status={status}")
    print(f"[ok] transcript_present={'yes' if transcript else 'no'} confidence={conf}")
    if transcript:
        print(f"[preview] {transcript[:180]}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Credit-safe Deepgram auth/STT smoke tests."
    )
    parser.add_argument(
        "--listen-once",
        action="store_true",
        help="Run exactly one prerecorded STT request after projects check.",
    )
    parser.add_argument(
        "--audio-url",
        default="https://dpgr.am/spacewalk.wav",
        help="Hosted audio URL for prerecorded STT test.",
    )
    parser.add_argument(
        "--model",
        default="nova-3",
        help="Deepgram model for prerecorded STT test.",
    )
    args = parser.parse_args()

    try:
        key = _deepgram_key()
        _projects_check(key)
        if args.listen_once:
            _listen_once(key, args.audio_url, args.model)
    except (HTTPError, URLError, TimeoutError, RuntimeError, ValueError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

