"""Analyze ASR entity misses across baseline and LoRA result files."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ENTITY_GROUPS = (
    "invoice_ids",
    "amounts",
    "dates",
    "billing_actions",
    "confirmations",
    "refusals",
    "account_terms",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare entity misses across ASR result JSONL files.")
    parser.add_argument("--base-whisper", required=True)
    parser.add_argument("--lora-whisper", required=True)
    parser.add_argument("--deepgram", required=True)
    parser.add_argument("--json-output", required=True)
    parser.add_argument("--markdown-output", required=True)
    parser.add_argument("--max-examples", type=int, default=20)
    args = parser.parse_args()

    runs = {
        "base_whisper": load_results(Path(args.base_whisper)),
        "lora_whisper": load_results(Path(args.lora_whisper)),
        "deepgram": load_results(Path(args.deepgram)),
    }
    validate_clip_sets(runs)

    analysis = {
        "inputs": {
            "base_whisper": args.base_whisper,
            "lora_whisper": args.lora_whisper,
            "deepgram": args.deepgram,
        },
        "summary": summarize_runs(runs),
        "transitions": compare_base_to_lora(runs["base_whisper"], runs["lora_whisper"]),
        "lora_open_errors": collect_open_errors(
            runs,
            target_run="lora_whisper",
            max_examples=args.max_examples,
        ),
    }

    json_output = Path(args.json_output)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(analysis, indent=2), encoding="utf-8")

    markdown_output = Path(args.markdown_output)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.write_text(render_markdown(analysis), encoding="utf-8")
    print(json.dumps(analysis["summary"], indent=2))


def load_results(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            clip_id = row.get("clip_id")
            if not clip_id:
                raise ValueError(f"{path}:{line_no} missing clip_id")
            rows[str(clip_id)] = row
    return rows


def validate_clip_sets(runs: dict[str, dict[str, dict[str, Any]]]) -> None:
    clip_sets = {name: set(rows) for name, rows in runs.items()}
    expected = next(iter(clip_sets.values()))
    for name, clips in clip_sets.items():
        if clips != expected:
            missing = sorted(expected - clips)[:10]
            extra = sorted(clips - expected)[:10]
            raise ValueError(f"{name} clip set mismatch; missing={missing}, extra={extra}")


def summarize_runs(runs: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    return {name: summarize_run(rows) for name, rows in runs.items()}


def summarize_run(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    entity_totals = defaultdict(lambda: {"expected": 0, "matched": 0, "missed": 0})
    missed_values = defaultdict(Counter)
    clips_with_misses = defaultdict(set)
    for clip_id, row in rows.items():
        scores = (row.get("score") or {}).get("entity_scores") or {}
        for group in ENTITY_GROUPS:
            score = scores.get(group) or {}
            expected = int(score.get("expected") or 0)
            matched = int(score.get("matched") or 0)
            missing = [str(value) for value in score.get("missing") or []]
            entity_totals[group]["expected"] += expected
            entity_totals[group]["matched"] += matched
            entity_totals[group]["missed"] += len(missing)
            if missing:
                clips_with_misses[group].add(clip_id)
            for value in missing:
                missed_values[group][value] += 1

    return {
        "clips": len(rows),
        "avg_wer": avg((row.get("score") or {}).get("wer") for row in rows.values()),
        "avg_cer": avg((row.get("score") or {}).get("cer") for row in rows.values()),
        "avg_entity_accuracy": avg(
            (row.get("score") or {}).get("entity_accuracy") for row in rows.values()
        ),
        "entity_groups": {
            group: {
                **counts,
                "accuracy": None
                if counts["expected"] == 0
                else counts["matched"] / counts["expected"],
                "clips_with_misses": len(clips_with_misses[group]),
                "top_missed_values": missed_values[group].most_common(10),
            }
            for group, counts in entity_totals.items()
        },
    }


def compare_base_to_lora(
    base_rows: dict[str, dict[str, Any]],
    lora_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    transitions: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for group in ENTITY_GROUPS:
        recovered = []
        regressed = []
        unchanged_misses = []
        for clip_id in sorted(base_rows):
            base_missing = missing_values(base_rows[clip_id], group)
            lora_missing = missing_values(lora_rows[clip_id], group)
            if base_missing and not lora_missing:
                recovered.append(example_for(clip_id, group, base_rows[clip_id], lora_rows[clip_id]))
            elif not base_missing and lora_missing:
                regressed.append(example_for(clip_id, group, base_rows[clip_id], lora_rows[clip_id]))
            elif base_missing and lora_missing:
                unchanged_misses.append(example_for(clip_id, group, base_rows[clip_id], lora_rows[clip_id]))
        transitions[group] = {
            "recovered_count": len(recovered),
            "regressed_count": len(regressed),
            "unchanged_miss_count": len(unchanged_misses),
            "recovered_examples": recovered[:10],
            "regressed_examples": regressed[:10],
            "unchanged_miss_examples": unchanged_misses[:10],
        }
    return transitions


def collect_open_errors(
    runs: dict[str, dict[str, dict[str, Any]]],
    *,
    target_run: str,
    max_examples: int,
) -> dict[str, list[dict[str, Any]]]:
    target_rows = runs[target_run]
    errors: dict[str, list[dict[str, Any]]] = {}
    for group in ENTITY_GROUPS:
        group_errors = []
        for clip_id, target_row in sorted(target_rows.items()):
            missing = missing_values(target_row, group)
            if not missing:
                continue
            item = {
                "clip_id": clip_id,
                "group": group,
                "missing": missing,
                "reference": target_row.get("reference_transcript"),
                "hypotheses": {
                    name: rows[clip_id].get("transcript") for name, rows in runs.items()
                },
                "missed_by": [
                    name for name, rows in runs.items() if missing_values(rows[clip_id], group)
                ],
            }
            group_errors.append(item)
        errors[group] = group_errors[:max_examples]
    return errors


def missing_values(row: dict[str, Any], group: str) -> list[str]:
    score = ((row.get("score") or {}).get("entity_scores") or {}).get(group) or {}
    return [str(value) for value in score.get("missing") or []]


def example_for(
    clip_id: str,
    group: str,
    base_row: dict[str, Any],
    lora_row: dict[str, Any],
) -> dict[str, Any]:
    return {
        "clip_id": clip_id,
        "group": group,
        "base_missing": missing_values(base_row, group),
        "lora_missing": missing_values(lora_row, group),
        "reference": lora_row.get("reference_transcript"),
        "base_transcript": base_row.get("transcript"),
        "lora_transcript": lora_row.get("transcript"),
    }


def render_markdown(analysis: dict[str, Any]) -> str:
    lines = [
        "# ASR Entity Error Analysis",
        "",
        "## Summary",
        "",
        "| Run | WER | CER | Entity Accuracy |",
        "|---|---:|---:|---:|",
    ]
    for name, summary in analysis["summary"].items():
        lines.append(
            f"| {name} | {fmt(summary['avg_wer'])} | {fmt(summary['avg_cer'])} | "
            f"{fmt(summary['avg_entity_accuracy'])} |"
        )

    lines.extend(["", "## Entity Misses", ""])
    for name, summary in analysis["summary"].items():
        lines.extend([f"### {name}", "", "| Group | Expected | Missed | Accuracy | Clips With Misses |", "|---|---:|---:|---:|---:|"])
        for group, counts in summary["entity_groups"].items():
            lines.append(
                f"| {group} | {counts['expected']} | {counts['missed']} | "
                f"{fmt(counts['accuracy'])} | {counts['clips_with_misses']} |"
            )
        lines.append("")

    lines.extend(["## Base Whisper To LoRA Transitions", ""])
    lines.append("| Group | Recovered | Regressed | Still Missed |")
    lines.append("|---|---:|---:|---:|")
    for group, transition in analysis["transitions"].items():
        lines.append(
            f"| {group} | {transition['recovered_count']} | "
            f"{transition['regressed_count']} | {transition['unchanged_miss_count']} |"
        )

    lines.extend(["", "## LoRA Open Error Examples", ""])
    for group in (
        "invoice_ids",
        "dates",
        "amounts",
        "billing_actions",
        "refusals",
        "account_terms",
    ):
        examples = analysis["lora_open_errors"].get(group) or []
        lines.extend([f"### {group}", ""])
        if not examples:
            lines.extend(["No open LoRA misses.", ""])
            continue
        for item in examples[:10]:
            lines.extend(
                [
                    f"- `{item['clip_id']}` missing `{', '.join(item['missing'])}`; missed by: {', '.join(item['missed_by'])}",
                    f"  - Reference: {item['reference']}",
                    f"  - Base Whisper: {item['hypotheses'].get('base_whisper')}",
                    f"  - LoRA Whisper: {item['hypotheses'].get('lora_whisper')}",
                    f"  - Deepgram: {item['hypotheses'].get('deepgram')}",
                ]
            )
        lines.append("")
    return "\n".join(lines)


def avg(values: Any) -> float | None:
    present = [float(value) for value in values if value is not None]
    return None if not present else sum(present) / len(present)


def fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


if __name__ == "__main__":
    main()
