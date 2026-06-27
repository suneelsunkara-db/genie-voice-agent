"""Bootstrap an interim realistic synthetic ASR holdout set.

This is not a substitute for real recorded calls. It exists to exercise the
holdout workflow with unseen, contact-center-like clips until real audio arrives.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
from pathlib import Path
from typing import Any

from genie_voice.config import get_settings


ROOT = Path(__file__).resolve().parents[2]
LOCAL_HOLDOUT = ROOT / ".run/asr_model_training/holdout"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic holdout WAVs and manifest.")
    parser.add_argument("--clips", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260627)
    parser.add_argument("--profile", default="fe-vm-vdm-classic-rcn6ip")
    parser.add_argument("--skip-upload", action="store_true")
    args = parser.parse_args()

    if shutil.which("say") is None or shutil.which("afconvert") is None:
        raise SystemExit("This bootstrap requires macOS 'say' and 'afconvert'.")

    random.seed(args.seed)
    settings = get_settings()
    volume_root = (
        f"/Volumes/{settings.databricks.catalog}/"
        f"{settings.databricks.schema_name}/"
        f"{settings.volume.streaming_name}/asr_model_training"
    )
    holdout_root = f"{volume_root}/datasets/holdout"
    remote_audio_root = f"{holdout_root}/audio"
    remote_manifest = f"{holdout_root}/manifests/asr_real_audio_holdout_v1.jsonl"

    local_audio = LOCAL_HOLDOUT / "audio"
    local_manifest = LOCAL_HOLDOUT / "manifests/asr_real_audio_holdout_v1.jsonl"
    local_audio.mkdir(parents=True, exist_ok=True)
    local_manifest.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx, scenario in enumerate(build_scenarios(args.clips), start=1):
        clip_id = f"HOLDOUT-SYNTH-{idx:03d}"
        wav = local_audio / f"{clip_id}.wav"
        synthesize(scenario["text"], wav)
        rows.append(
            {
                "clip_id": clip_id,
                "audio_path": f"{remote_audio_root}/{wav.name}",
                "reference_transcript": scenario["text"],
                "call_id": f"HOLDOUT-SYNTH-CALL-{idx:03d}",
                "speaker": scenario["speaker"],
                "audio_format": "wav",
                "sample_rate_hz": 16000,
                "domain": "billing_contact_center",
                "scenario": scenario["scenario"],
                "split": "holdout",
                "dataset_version": "realistic_synthetic_holdout_v1",
                "expected_entities": scenario["entities"],
                "metadata": {
                    "holdout_kind": "realistic_synthetic",
                    "not_for_training": True,
                    "requires_real_audio_replacement_before_production_gate": True,
                },
            }
        )

    local_manifest.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )

    if not args.skip_upload:
        upload(local_audio, local_manifest, remote_audio_root, remote_manifest, args.profile)

    print(
        json.dumps(
            {
                "clips": len(rows),
                "local_manifest": str(local_manifest),
                "remote_manifest": remote_manifest,
                "remote_audio_root": remote_audio_root,
                "holdout_kind": "realistic_synthetic",
            },
            indent=2,
        )
    )


def synthesize(text: str, wav: Path) -> None:
    aiff = wav.with_suffix(".aiff")
    subprocess.run(["say", "-o", str(aiff), text], check=True)
    subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", str(aiff), str(wav)], check=True)
    aiff.unlink(missing_ok=True)


def upload(local_audio: Path, local_manifest: Path, remote_audio_root: str, remote_manifest: str, profile: str) -> None:
    dbx = ["databricks", "--profile", profile]
    subprocess.run(dbx + ["fs", "mkdirs", f"dbfs:{remote_audio_root}"], check=True)
    subprocess.run(dbx + ["fs", "mkdirs", f"dbfs:{Path(remote_manifest).parent}"], check=True)
    for wav in sorted(local_audio.glob("HOLDOUT-SYNTH-*.wav")):
        subprocess.run(
            dbx + ["fs", "cp", str(wav), f"dbfs:{remote_audio_root}/{wav.name}", "--overwrite"],
            check=True,
        )
    subprocess.run(dbx + ["fs", "cp", str(local_manifest), f"dbfs:{remote_manifest}", "--overwrite"], check=True)


def build_scenarios(count: int) -> list[dict[str, Any]]:
    first_names = ["Ava", "Noah", "Priya", "Luis", "Mina", "Owen", "Nia", "Ethan"]
    amounts = ["$49.00", "$79.25", "$112.40", "$188.05", "$249.99", "$310.10"]
    dates = ["April 21", "May 23", "June 30", "July 8", "August 14", "September 3"]
    scenarios: list[dict[str, Any]] = []

    templates = [
        (
            "invoice_dispute",
            "Hi, this is {name}. I am calling about invoice {invoice}; the {amount} charge looks wrong.",
            lambda invoice, amount, date: {
                "invoice_ids": [invoice],
                "amounts": [amount],
                "dates": [],
                "billing_actions": [],
                "confirmations": [],
                "refusals": [],
                "account_terms": ["invoice"],
            },
        ),
        (
            "autopay_failed",
            "My autopay failed and invoice {invoice} is now overdue, but I can pay {amount} on {date}.",
            lambda invoice, amount, date: {
                "invoice_ids": [invoice],
                "amounts": [amount],
                "dates": [date],
                "billing_actions": [],
                "confirmations": [],
                "refusals": [],
                "account_terms": ["autopay", "invoice", "payment"],
            },
        ),
        (
            "waiver_request",
            "Please waive the late fee on invoice {invoice}; I paid the balance on {date}.",
            lambda invoice, amount, date: {
                "invoice_ids": [invoice],
                "amounts": [],
                "dates": [date],
                "billing_actions": ["waive"],
                "confirmations": [],
                "refusals": [],
                "account_terms": ["invoice", "payment"],
            },
        ),
        (
            "payment_plan_confirmation",
            "Yes, that works. Split invoice {invoice} into two payments of {amount}.",
            lambda invoice, amount, date: {
                "invoice_ids": [invoice],
                "amounts": [amount],
                "dates": [],
                "billing_actions": ["payment plan"],
                "confirmations": ["yes", "that works"],
                "refusals": [],
                "account_terms": ["invoice", "payment"],
            },
        ),
        (
            "refusal",
            "No, do not charge that card for invoice {invoice}; I want to use another payment method.",
            lambda invoice, amount, date: {
                "invoice_ids": [invoice],
                "amounts": [],
                "dates": [],
                "billing_actions": [],
                "confirmations": [],
                "refusals": ["no", "do not"],
                "account_terms": ["card", "invoice", "payment"],
            },
        ),
    ]

    for idx in range(count):
        scenario, template, entity_fn = templates[idx % len(templates)]
        invoice = f"INV-{91000 + idx * 7:05d}"
        amount = amounts[idx % len(amounts)]
        date = dates[idx % len(dates)]
        text = template.format(
            name=first_names[idx % len(first_names)],
            invoice=invoice,
            amount=amount,
            date=date,
        )
        scenarios.append(
            {
                "scenario": scenario,
                "speaker": "customer",
                "text": text,
                "entities": entity_fn(invoice, amount, date),
            }
        )
    return scenarios


if __name__ == "__main__":
    main()
