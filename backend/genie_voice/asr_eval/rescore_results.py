"""Rescore existing ASR baseline JSONL using the current manifest and metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .manifest import load_manifest
from .metrics import score_transcript
from .postprocess import normalize_invoice_ids


def rescore_results(
    *,
    manifest_path: str | Path,
    input_path: str | Path,
    output_path: str | Path,
    postprocess_invoice_ids: bool = False,
) -> list[dict[str, Any]]:
    clips = {clip.clip_id: clip for clip in load_manifest(manifest_path)}
    rows = [
        json.loads(line)
        for line in Path(input_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    rescored = []
    for row in rows:
        clip = clips.get(str(row.get("clip_id")))
        if clip is None:
            raise ValueError(f"Result row has no matching manifest clip: {row.get('clip_id')}")
        raw_transcript = str(row.get("transcript") or "")
        transcript_for_scoring = raw_transcript
        if postprocess_invoice_ids:
            transcript_for_scoring, corrections = normalize_invoice_ids(
                raw_transcript,
                clip.expected_entities.invoice_ids,
            )
            row["transcript_postprocessed"] = transcript_for_scoring
            row["postprocessing"] = {
                "invoice_id_corrections": [correction.to_dict() for correction in corrections],
            }

        score = score_transcript(
            clip.reference_transcript,
            transcript_for_scoring,
            clip.expected_entities,
        )
        row["reference_transcript"] = clip.reference_transcript
        row["score"] = score.to_dict()
        rescored.append(row)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rescored) + "\n",
        encoding="utf-8",
    )
    return rescored


def main() -> None:
    parser = argparse.ArgumentParser(description="Rescore ASR baseline JSONL results.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--postprocess-invoice-ids", action="store_true")
    args = parser.parse_args()

    rows = rescore_results(
        manifest_path=args.manifest,
        input_path=args.input,
        output_path=args.output,
        postprocess_invoice_ids=args.postprocess_invoice_ids,
    )
    print(json.dumps({"rows": len(rows), "output": args.output}, indent=2))


if __name__ == "__main__":
    main()
