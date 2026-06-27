"""Self-contained Whisper LoRA fine-tuning runner for Databricks GPU clusters."""
from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Whisper with LoRA on ASR manifest clips.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-model", default="openai/whisper-small.en")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--validation-split", default="validation")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_rows = load_manifest(args.manifest, split=args.train_split, limit=args.max_train_samples)
    eval_rows = load_manifest(args.manifest, split=args.validation_split, limit=args.max_eval_samples)
    if not train_rows:
        raise SystemExit(f"No train rows found for split={args.train_split}")
    if not eval_rows:
        raise SystemExit(f"No validation rows found for split={args.validation_split}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "manifest": args.manifest,
        "output_dir": str(output_dir),
        "base_model": args.base_model,
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "seed": args.seed,
        "dry_run": args.dry_run,
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    if args.dry_run:
        write_preview(output_dir, train_rows, eval_rows)
        print(json.dumps({"status": "dry_run_ok", **run_config}, indent=2))
        return

    from datasets import Audio, Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )

    processor = WhisperProcessor.from_pretrained(args.base_model)
    model = WhisperForConditionalGeneration.from_pretrained(args.base_model)
    model.config.use_cache = False
    model.generation_config.language = None
    model.generation_config.task = None

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = build_dataset(train_rows).cast_column("audio", Audio(sampling_rate=16000))
    eval_dataset = build_dataset(eval_rows).cast_column("audio", Audio(sampling_rate=16000))
    train_dataset = train_dataset.map(lambda item: prepare_item(item, processor), remove_columns=train_dataset.column_names)
    eval_dataset = eval_dataset.map(lambda item: prepare_item(item, processor), remove_columns=eval_dataset.column_names)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir / "trainer"),
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        report_to=[],
        remove_unused_columns=False,
        label_names=["labels"],
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=WhisperDataCollator(processor),
        processing_class=processor,
    )
    train_result = trainer.train()
    metrics = train_result.metrics
    trainer.save_model(str(output_dir / "adapter"))
    processor.save_pretrained(str(output_dir / "processor"))
    (output_dir / "train_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({"status": "train_ok", "metrics": metrics, **run_config}, indent=2))


def load_manifest(path: str, *, split: str, limit: int | None) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            row = json.loads(line)
            if row.get("split") != split:
                continue
            if (
                split == "train"
                and row.get("replace_with_real_call_audio_before_training")
                and row.get("dataset_version") != "billing_scenarios_augmented_v1"
            ):
                # Synthetic originals are useful for baselines; train first on augmented/domain rows.
                continue
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def build_dataset(rows: list[dict[str, Any]]) -> Any:
    from datasets import Dataset

    return Dataset.from_list(
        [
            {
                "audio": row["audio_path"],
                "sentence": row["reference_transcript"],
                "clip_id": row["clip_id"],
            }
            for row in rows
        ]
    )


def prepare_item(item: dict[str, Any], processor: Any) -> dict[str, Any]:
    audio = item["audio"]
    inputs = processor.feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_tensors="pt",
    )
    labels = processor.tokenizer(item["sentence"]).input_ids
    return {
        "input_features": inputs.input_features[0],
        "labels": labels,
    }


@dataclass
class WhisperDataCollator:
    processor: Any

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch


def write_preview(output_dir: Path, train_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]]) -> None:
    preview = {
        "train_clip_ids": [row["clip_id"] for row in train_rows[:10]],
        "eval_clip_ids": [row["clip_id"] for row in eval_rows[:10]],
    }
    (output_dir / "dry_run_preview.json").write_text(json.dumps(preview, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
