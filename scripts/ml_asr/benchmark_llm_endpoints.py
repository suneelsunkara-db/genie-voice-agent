"""Benchmark candidate Databricks LLM endpoints for the voice middle stage.

Measures real per-turn latency (median + p95) across languages for each
candidate, using the same ChatCompletions shape the API uses. Confirms the
``temperature`` parameter is accepted. One warmup call per endpoint is discarded
to avoid cold-routing skew.

Usage:
    python scripts/ml_asr/benchmark_llm_endpoints.py --reps 5
"""
from __future__ import annotations

import argparse
import statistics
import time
from typing import Any

from _realtime_config import databricks

_CANDIDATES = [
    "databricks-qwen3-next-80b-a3b-instruct",
    "databricks-gemini-3-1-flash-lite",
    "databricks-gemini-3-5-flash",
    "databricks-claude-haiku-4-5",
    "databricks-claude-sonnet-5",  # baseline (rejects temperature)
]

_PROMPTS = {
    "en-US": "What time do you open on Saturday?",
    "th-TH": "ร้านเปิดกี่โมงในวันเสาร์",
    "id-ID": "Jam berapa kalian buka pada hari Sabtu?",
    "zh-CN": "你们周六几点开门？",
}


def _system(language: str) -> str:
    return (
        "You are a concise, friendly voice assistant. Reply in one to three short spoken "
        "sentences with no markdown, no lists, and no emoji. "
        f"Always respond in the user's language ({language})."
    )


def _call(client: Any, endpoint: str, language: str, *, temperature: bool) -> tuple[float, str]:
    body: dict[str, Any] = {
        "messages": [
            {"role": "system", "content": _system(language)},
            {"role": "user", "content": _PROMPTS[language]},
        ],
        "max_tokens": 256,
    }
    if temperature:
        body["temperature"] = 0.4
    start = time.perf_counter()
    r = client.api_client.do("POST", f"/serving-endpoints/{endpoint}/invocations", body=body)
    elapsed = (time.perf_counter() - start) * 1000
    ch = (r.get("choices") or [{}])[0].get("message", {}).get("content")
    return elapsed, (ch if isinstance(ch, str) else "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--languages", default="en-US,th-TH,id-ID,zh-CN")
    args = parser.parse_args()
    languages = [x.strip() for x in args.languages.split(",") if x.strip()]

    from databricks.sdk import WorkspaceClient

    client = WorkspaceClient(profile=databricks().get("profile") or None)

    print(f"reps={args.reps} langs={languages}\n")
    header = f"{'endpoint':44} {'temp':6} {'median':>8} {'p95':>8} {'min':>7} {'max':>7}"
    print(header)
    print("-" * len(header))

    for ep in _CANDIDATES:
        # Probe temperature support once; fall back to no-temperature if rejected.
        use_temp = True
        try:
            _call(client, ep, languages[0], temperature=True)
        except Exception as exc:  # noqa: BLE001
            use_temp = not ("temperature" in str(exc).lower())
            if not use_temp:
                try:
                    _call(client, ep, languages[0], temperature=False)  # warmup no-temp
                except Exception as exc2:  # noqa: BLE001
                    print(f"{ep:44} ERROR {str(exc2)[:60]}")
                    continue

        all_lat: list[float] = []
        per_lang: dict[str, float] = {}
        sample = ""
        failed = None
        for language in languages:
            lat: list[float] = []
            for _ in range(args.reps):
                try:
                    ms, text = _call(client, ep, language, temperature=use_temp)
                    lat.append(ms)
                    if language == "th-TH" and text:
                        sample = text[:46]
                except Exception as exc:  # noqa: BLE001
                    failed = str(exc)[:60]
                    break
            if lat:
                per_lang[language] = statistics.median(lat)
                all_lat.extend(lat)
            if failed:
                break

        if failed or not all_lat:
            print(f"{ep:44} FAILED {failed or 'no data'}")
            continue

        srt = sorted(all_lat)
        p95 = srt[min(len(srt) - 1, int(round(0.95 * (len(srt) - 1))))]
        temp_label = "OK" if use_temp else "REJECT"
        print(
            f"{ep:44} {temp_label:6} {statistics.median(all_lat):7.0f}m {p95:7.0f}m "
            f"{min(all_lat):6.0f}m {max(all_lat):6.0f}m"
        )
        lang_str = "  ".join(f"{k.split('-')[0]}={v:.0f}" for k, v in per_lang.items())
        print(f"{'':44} per-lang median(ms): {lang_str}")
        if sample:
            print(f"{'':44} th sample: {sample!r}")
        print()


if __name__ == "__main__":
    main()
