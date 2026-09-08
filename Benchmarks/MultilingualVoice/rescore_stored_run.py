#!/usr/bin/env python3
"""Re-score a persisted multilingual benchmark run without model inference."""
from __future__ import annotations

import argparse
import json
from typing import Any

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementParameterListItem

from evaluators import (
    evaluate_asr,
    evaluate_tts_roundtrip,
    primary_metric,
    primary_metric_name,
)


def _rows(result: Any) -> list[dict[str, Any]]:
    columns = [column.name for column in result.manifest.schema.columns]
    return [dict(zip(columns, row)) for row in (result.result.data_array or [])]


def _execute(
    client: WorkspaceClient,
    warehouse_id: str,
    statement: str,
    parameters: list[StatementParameterListItem],
) -> Any:
    result = client.statement_execution.execute_statement(
        warehouse_id=warehouse_id,
        statement=statement,
        parameters=parameters,
        wait_timeout="50s",
    )
    state = getattr(getattr(result, "status", None), "state", None)
    if str(state) not in {"StatementState.SUCCEEDED", "SUCCEEDED"}:
        raise RuntimeError(f"SQL statement failed: {getattr(result, 'status', None)}")
    return result


def _params(**values: tuple[str, Any]) -> list[StatementParameterListItem]:
    return [
        StatementParameterListItem(name=name, type=kind, value=str(value))
        for name, (kind, value) in values.items()
    ]


def rescore(
    *,
    run_id: str,
    profile: str,
    warehouse_id: str,
    table_prefix: str,
) -> None:
    client = WorkspaceClient(profile=profile)
    runs_table = f"{table_prefix}.benchmark_runs"
    samples_table = f"{table_prefix}.benchmark_samples"
    run_rows = _rows(
        _execute(
            client,
            warehouse_id,
            (
                f"SELECT language, evaluator, scores FROM {runs_table} "  # noqa: S608
                "WHERE run_id = :run_id AND dataset = 'fleurs' ORDER BY language"
            ),
            _params(run_id=("STRING", run_id)),
        )
    )
    if not run_rows:
        raise RuntimeError(f"No FLEURS rows found for run {run_id}")

    for run in run_rows:
        language = str(run["language"])
        evaluator = str(run["evaluator"])
        samples = _rows(
            _execute(
                client,
                warehouse_id,
                (
                    "SELECT reference, transcript, response, tts_audio_bytes, tts_roundtrip "
                    f"FROM {samples_table} "  # noqa: S608
                    "WHERE run_id = :run_id AND dataset = 'fleurs' AND language = :language "
                    "ORDER BY sample_index"
                ),
                _params(run_id=("STRING", run_id), language=("STRING", language)),
            )
        )
        for sample in samples:
            raw_roundtrip = sample.get("tts_roundtrip")
            if isinstance(raw_roundtrip, str):
                sample["tts_roundtrip"] = json.loads(raw_roundtrip or "{}")
            sample["tts_audio_bytes"] = int(sample.get("tts_audio_bytes") or 0)

        previous = json.loads(run.get("scores") or "{}")
        scores = evaluate_asr(samples)
        scores.update(evaluate_tts_roundtrip(samples))
        scores["primary_metric"] = primary_metric_name(evaluator, language)
        scores["legacy_scores"] = previous.get("legacy_scores", previous)
        primary = primary_metric(evaluator, scores, language=language)
        if primary is None:
            raise RuntimeError(f"No primary score for {language}")

        _execute(
            client,
            warehouse_id,
            (
                f"UPDATE {runs_table} SET primary_score = :primary_score, scores = :scores "  # noqa: S608
                "WHERE run_id = :run_id AND dataset = 'fleurs' AND language = :language"
            ),
            _params(
                primary_score=("DOUBLE", primary),
                scores=("STRING", json.dumps(scores, ensure_ascii=False)),
                run_id=("STRING", run_id),
                language=("STRING", language),
            ),
        )
        print(
            f"{language}: {scores['primary_metric']}={primary:.4f} "
            f"ci95={scores.get(scores['primary_metric'] + '_ci95')}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--warehouse-id", required=True)
    parser.add_argument(
        "--table-prefix",
        default="partner_demo_catalog.genie_voice_contact_center",
    )
    args = parser.parse_args()
    rescore(
        run_id=args.run_id,
        profile=args.profile,
        warehouse_id=args.warehouse_id,
        table_prefix=args.table_prefix,
    )


if __name__ == "__main__":
    main()
