from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

from genie_voice.ml_asr.config import config_summary, load_config


def summarize(*, config_path: str | None = None, volume_mode: bool | None = None) -> Path:
    from genie_voice.ml_asr.runtime import is_volume_mode

    config = load_config(config_path=config_path)
    volume_mode = is_volume_mode() if volume_mode is None else volume_mode
    index = {
        "catalog": config_summary(config),
        "datasets": {},
    }

    for dataset_id, dataset in config.datasets.items():
        language_summaries: dict[str, dict] = {}
        for language in dataset.languages:
            model_summaries: dict[str, dict] = {}
            for model_spec in config.models_for_language(language):
                summary_path = config.result_dir(language, dataset_id, model_spec.model_id, volume_mode=volume_mode) / "summary.json"
                results_path = config.result_dir(language, dataset_id, model_spec.model_id, volume_mode=volume_mode) / "results.jsonl"
                if summary_path.is_file():
                    model_summaries[model_spec.model_id] = json.loads(summary_path.read_text(encoding="utf-8"))
                elif results_path.is_file():
                    rows = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines() if line.strip()]
                    model_summaries[model_spec.model_id] = _summarize_rows(
                        rows, model_spec.model_id, model_spec.label, dataset.eval_tier
                    )
                else:
                    model_summaries[model_spec.model_id] = {"status": "missing", "results_path": str(results_path)}
            language_summaries[language] = {
                "models": model_summaries,
                "ranking": _rank_models(model_summaries, dataset.eval_tier),
            }
        index["datasets"][dataset_id] = {
            "eval_tier": dataset.eval_tier,
            "languages": language_summaries,
        }

    out = Path(config.remote_index_path if volume_mode else Path(config.local_results_dir) / "index.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def _summarize_rows(rows: list[dict], model_id: str, model_label: str, eval_tier: str) -> dict:
    ok_rows = [row for row in rows if not row.get("error")]
    summary = {
        "model_id": model_id,
        "model_label": model_label,
        "eval_tier": eval_tier,
        "clip_count": len(rows),
        "success_count": len(ok_rows),
        "error_count": len(rows) - len(ok_rows),
        "avg_wer": _avg([row["score"]["wer"] for row in ok_rows]),
        "avg_cer": _avg([row["score"]["cer"] for row in ok_rows]),
        "avg_entity_accuracy": _avg(
            [row["score"]["entity_accuracy"] for row in ok_rows if row["score"]["entity_accuracy"] is not None]
        ),
        "p95_latency_ms": _p95([row["latency_ms"] for row in ok_rows]),
        "avg_latency_ms": _avg([row["latency_ms"] for row in ok_rows]),
    }
    if eval_tier == "business":
        unsafe = [bool((row.get("readiness") or {}).get("unsafe_for_resolution")) for row in ok_rows]
        summary["unsafe_for_resolution_rate"] = _avg([1.0 if value else 0.0 for value in unsafe])
        summary["avg_critical_entity_accuracy"] = _avg(
            [
                (row.get("readiness") or {}).get("critical_entity_accuracy")
                for row in ok_rows
                if (row.get("readiness") or {}).get("critical_entity_accuracy") is not None
            ]
        )
        from genie_voice.ml_asr.benchmark_export import _entity_groups_from_rows

        summary["entity_groups"] = _entity_groups_from_rows(ok_rows)
    return summary


def _avg(values: list[float | int]) -> float | None:
    if not values:
        return None
    return round(mean(values), 4)


def _p95(values: list[int]) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, int(round(0.95 * (len(ordered) - 1))))
    return ordered[index]


def _rank_models(model_summaries: dict[str, dict], eval_tier: str) -> list[dict]:
    candidates = []
    for model_id, summary in model_summaries.items():
        if summary.get("status") == "missing":
            continue
        item = {
            "model_id": model_id,
            "model_label": summary.get("model_label", model_id),
            "avg_wer": summary.get("avg_wer"),
            "avg_cer": summary.get("avg_cer"),
            "p95_latency_ms": summary.get("p95_latency_ms"),
        }
        if eval_tier == "business":
            item["avg_critical_entity_accuracy"] = summary.get("avg_critical_entity_accuracy")
            item["unsafe_for_resolution_rate"] = summary.get("unsafe_for_resolution_rate")
        candidates.append(item)

    if eval_tier == "business":
        candidates.sort(
            key=lambda item: (
                -(item.get("avg_critical_entity_accuracy") or 0.0),
                item.get("unsafe_for_resolution_rate") if item.get("unsafe_for_resolution_rate") is not None else 1.0,
                item.get("avg_wer") if item.get("avg_wer") is not None else float("inf"),
            )
        )
    else:
        candidates.sort(
            key=lambda item: (
                item.get("avg_wer") if item.get("avg_wer") is not None else float("inf"),
                item.get("p95_latency_ms") if item.get("p95_latency_ms") is not None else float("inf"),
            )
        )
    return candidates
