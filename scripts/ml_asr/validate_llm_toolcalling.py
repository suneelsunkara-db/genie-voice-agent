"""Validate the LLM middle stage (temperature + tool calling) live.

Exercises the exact ``DatabricksServing.respond`` code path used by the API,
but with a Databricks-SDK-backed client (no local mlflow needed). Confirms the
configured endpoint accepts ``temperature`` + ``tools`` and that the tool-calling
loop (get_current_time) round-trips to a spoken answer.

Usage:
    python scripts/ml_asr/validate_llm_toolcalling.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_REPO / "backend") not in sys.path:
    sys.path.insert(0, str(_REPO / "backend"))

from _realtime_config import databricks, realtime_voice  # noqa: E402
from realtime_api.services import DatabricksServing, _SdkDeployClient  # noqa: E402


def main() -> None:
    rv = realtime_voice()
    llm = str(rv.get("llm_endpoint") or "")
    defaults = rv.get("llm_defaults") or {}

    serving = DatabricksServing(
        client=_SdkDeployClient(databricks().get("profile") or None),
        stt_endpoint="",
        llm_endpoint=llm,
        tts_endpoint="",
        llm_temperature=float(defaults.get("temperature", 0.4)),
        llm_max_tokens=int(defaults.get("max_tokens", 512)),
        llm_tools_enabled=bool(defaults.get("tools_enabled", True)),
        llm_max_tool_iterations=int(defaults.get("max_tool_iterations", 3)),
    )

    probes = [
        ("What time is it right now in Bangkok?", "en-US"),
        ("ตอนนี้ที่กรุงเทพกี่โมงแล้ว", "th-TH"),
        ("Give me a friendly one-line greeting.", "en-US"),
    ]
    print(f"endpoint={llm} temperature={serving.llm_temperature} tools={serving.llm_tools_enabled}\n")
    for question, language in probes:
        start = time.perf_counter()
        try:
            answer = serving.respond(question, language=language)
            ms = (time.perf_counter() - start) * 1000
            print(f"[{language}] {ms:6.0f} ms  Q: {question}")
            print(f"          A: {answer!r}\n")
        except Exception as exc:  # noqa: BLE001
            print(f"[{language}] ERROR: {exc}\n")


if __name__ == "__main__":
    main()
