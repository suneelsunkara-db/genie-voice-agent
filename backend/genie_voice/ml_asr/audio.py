from __future__ import annotations

import base64
import mimetypes
import os
import subprocess
import tempfile
from pathlib import Path

from genie_voice.ml_asr.runtime import is_volume_mode, volume_path


def read_audio_bytes(path: str) -> bytes:
    if path.startswith("data:"):
        _, _, encoded = path.partition(",")
        return base64.b64decode(encoded)
    if path.startswith("/Volumes/") or path.startswith("dbfs:/Volumes/"):
        normalized = volume_path(path)
        if is_volume_mode():
            return Path(normalized).read_bytes()
        return _read_volume_audio(path)
    if path.startswith("file://"):
        path = path.removeprefix("file://")
    return Path(path).read_bytes()


def mime_type_for(path: str, audio_format: str | None = None) -> str:
    if audio_format and "/" in audio_format:
        return audio_format
    return mimetypes.guess_type(path)[0] or "audio/wav"


def speaker_number(speaker: object) -> int:
    text = str(speaker or "1").strip()
    return int(text) if text.isdigit() else 1


def _read_volume_audio(path: str) -> bytes:
    uri = path if path.startswith("dbfs:") else f"dbfs:{path}"
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        cmd = ["databricks", "fs", "cp", uri, str(tmp_path), "--overwrite"]
        from genie_voice.ml_asr.runtime import databricks_profile

        profile = databricks_profile()
        if profile:
            cmd = ["databricks", "--profile", profile, "fs", "cp", uri, str(tmp_path), "--overwrite"]
        subprocess.run(cmd, check=True, text=True, capture_output=True, timeout=120, env=os.environ.copy())
        return tmp_path.read_bytes()
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Databricks CLI is required to read /Volumes audio from a local process."
        ) from exc
    finally:
        tmp_path.unlink(missing_ok=True)
