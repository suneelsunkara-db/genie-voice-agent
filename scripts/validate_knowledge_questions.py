"""Ask Genie One every question the Knowledge page publishes and record the result.

Run from the repo root with a valid OBO token in ``GENIE_OBO_LOCAL_TOKEN``:

    PYTHONPATH=backend:. .venv/bin/python scripts/validate_knowledge_questions.py

Writes one JSON file per question under ``.run/knowledge_validation/`` and prints a
verdict table. This exists so the published question set is evidence-backed rather
than assumed: a question stays on the page only if Genie One actually answers it.
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import sys
import time

OUT_DIR = pathlib.Path(".run/knowledge_validation")

# Phrases that mean "Genie One declined" rather than "Genie One answered".
_DECLINE = re.compile(
    r"\b(i can(?:'|no)t|cannot|unable to|don'?t have (?:access|the ability)|"
    r"not able to|no information|outside (?:my|the) scope|i do not have)\b",
    re.I,
)


def verdict(payload: dict) -> tuple[str, str]:
    """Classify one Genie One response as answered / declined / failed."""
    if payload.get("error") or payload.get("denied"):
        return "FAILED", str(payload.get("error") or "denied")[:90]
    status = str(payload.get("status") or "").lower()
    answer = str(payload.get("final_answer") or payload.get("answer") or "").strip()
    if status in {"failed", "cancelled"} or payload.get("timeout"):
        return "FAILED", f"status={status or 'timeout'}"
    if not answer:
        return "EMPTY", f"status={status}, no answer text"
    # A decline usually opens with the refusal, so only the head is inspected —
    # a long answer that merely mentions limits later is still an answer.
    if _DECLINE.search(answer[:400]):
        return "DECLINED", answer[:110].replace("\n", " ")
    return "ANSWERED", f"{len(answer)} chars"


def main() -> int:
    from genie_voice.config import get_settings
    from realtime_api.knowledge_tools import WORKSPACE_PROMPTS
    from realtime_api.runtime.genie_one import run_workspace_query
    from realtime_api.runtime.identity import SessionPrincipal

    token = (os.environ.get("GENIE_OBO_LOCAL_TOKEN") or "").strip()
    if not token:
        print("GENIE_OBO_LOCAL_TOKEN is not set; cannot validate.", file=sys.stderr)
        return 2

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    principal = SessionPrincipal(username="validation", access_token=token)
    host = get_settings().databricks_host
    results: list[dict] = []

    for index, prompt in enumerate(WORKSPACE_PROMPTS, start=1):
        question = prompt["question"]
        started = time.perf_counter()
        raw = run_workspace_query(
            question,
            principal=principal,
            host=host,
            session_id="validation",
            turn_id=index,
            timeout_s=240,
        )
        elapsed = round(time.perf_counter() - started, 1)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {"error": "non-JSON tool result", "raw": raw[:500]}

        state, detail = verdict(payload)
        (OUT_DIR / f"{prompt['id']}.json").write_text(
            json.dumps(payload, indent=2, default=str)
        )
        results.append(
            {
                "id": prompt["id"],
                "category": prompt["category"],
                "question": question,
                "verdict": state,
                "detail": detail,
                "seconds": elapsed,
            }
        )
        print(f"[{index:2d}/{len(WORKSPACE_PROMPTS)}] {state:9s} {elapsed:6.1f}s  {prompt['id']}  {detail}", flush=True)

    (OUT_DIR / "summary.json").write_text(json.dumps(results, indent=2))
    print("\n=== VERDICT SUMMARY ===")
    for state in ("ANSWERED", "DECLINED", "EMPTY", "FAILED"):
        group = [r for r in results if r["verdict"] == state]
        print(f"\n{state} ({len(group)})")
        for row in group:
            print(f"  - {row['id']}: {row['question'][:80]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
