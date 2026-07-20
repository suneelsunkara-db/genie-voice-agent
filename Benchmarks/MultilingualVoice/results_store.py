"""Persist benchmark results to UC Delta tables via the in-job SparkSession.

Two tables (both USING DELTA, managed by Unity Catalog):

  - benchmark_runs    - one row per (run_id, dataset, language) sweep unit
  - benchmark_samples - one row per sample within a run

Writes are idempotent: a run is written only once per (run_id, dataset, language)
via MERGE. Resume queries the table for completed (dataset, language) pairs
in the current run_id, so a crash leaves a clean boundary and re-runs skip
completed units.
"""
from __future__ import annotations

import json
from typing import Any

from paths import delta_catalog, delta_schema


def _spark():
    from pyspark.sql import SparkSession

    return SparkSession.builder.getOrCreate()


def _runs_table():
    return delta_catalog() + "." + delta_schema() + ".benchmark_runs"


def _samples_table():
    return delta_catalog() + "." + delta_schema() + ".benchmark_samples"


def _ensure_tables():
    spark = _spark()
    spark.sql("CREATE DATABASE IF NOT EXISTS " + delta_catalog() + "." + delta_schema())
    spark.sql(
        "CREATE TABLE IF NOT EXISTS " + _runs_table() + " ("
        "run_id STRING, dataset STRING, language STRING, evaluator STRING, "
        "samples INT, errors INT, primary_score DOUBLE, "
        "scores STRING, latency_ms STRING, issues_count INT, issues_by_kind STRING, "
        "wall_seconds DOUBLE, timestamp STRING, status STRING"
        ") USING DELTA"
    )
    spark.sql(
        "CREATE TABLE IF NOT EXISTS " + _samples_table() + " ("
        "run_id STRING, dataset STRING, language STRING, sample_index INT, "
        "reference STRING, transcript STRING, response STRING, "
        "detected_language STRING, tts_audio_bytes INT, tts_roundtrip STRING, "
        "error STRING, stt_ms INT, llm_ms INT, tts_first_ms INT, "
        "client_ttfa_ms INT, total_ms INT, timestamp STRING"
        ") USING DELTA"
    )


def completed_pairs(run_id):
    """Return set of 'dataset@language' keys already completed for this run_id."""
    _ensure_tables()
    spark = _spark()
    rows = spark.sql(
        "SELECT dataset, language FROM " + _runs_table()
        + " WHERE run_id = '" + run_id + "' AND status = 'complete'"
    ).collect()
    return {r.dataset + "@" + r.language for r in rows}


def write_run(run_id, run, *, sample_rows=None, status="complete"):
    """Upsert one run + its sample rows into the Delta tables (idempotent)."""
    _ensure_tables()
    spark = _spark()

    scores = json.dumps(run.get("scores") or {}, ensure_ascii=False)
    latency = json.dumps(run.get("latency_ms") or {}, ensure_ascii=False)
    issues = run.get("issues") or {}
    issues_by_kind = json.dumps(issues.get("by_kind") or {}, ensure_ascii=False)

    run_row = spark.createDataFrame([(
        run_id, run["dataset"], run["language"], run["evaluator"],
        int(run.get("samples", 0)), int(run.get("errors", 0)),
        float(run.get("primary_score", 0.0)),
        scores, latency, int(issues.get("count", 0)), issues_by_kind,
        float(run.get("wall_seconds", 0.0)),
        run.get("timestamp"),
        status,
    )], schema=(
        "run_id string, dataset string, language string, evaluator string, "
        "samples int, errors int, primary_score double, scores string, "
        "latency_ms string, issues_count int, issues_by_kind string, "
        "wall_seconds double, timestamp string, status string"
    ))
    run_row.createOrReplaceTempView("_mlv_run_row")
    spark.sql(
        "MERGE INTO " + _runs_table() + " t "
        "USING _mlv_run_row s "
        "ON t.run_id = s.run_id AND t.dataset = s.dataset AND t.language = s.language "
        "WHEN MATCHED THEN UPDATE SET * "
        "WHEN NOT MATCHED THEN INSERT *"
    )

    if sample_rows:
        for i, sr in enumerate(sample_rows):
            sr["run_id"] = run_id
            sr["timestamp"] = run.get("timestamp")
            sr["sample_index"] = i
        rows_data = [(
            sr["run_id"], sr["dataset"], sr["language"], int(sr["sample_index"]),
            sr.get("reference") or "", sr.get("transcript") or "", sr.get("response") or "",
            sr.get("detected_language") or "", int(sr.get("tts_audio_bytes", 0)),
            json.dumps(sr.get("tts_roundtrip") or {}, ensure_ascii=False),
            sr.get("error") or "",
            sr.get("stt_ms"), sr.get("llm_ms"), sr.get("tts_first_ms"),
            sr.get("client_ttfa_ms"), sr.get("total_ms"),
            sr.get("timestamp"),
        ) for sr in sample_rows]
        schema = (
            "run_id string, dataset string, language string, sample_index int, "
            "reference string, transcript string, response string, "
            "detected_language string, tts_audio_bytes int, tts_roundtrip string, "
            "error string, stt_ms int, llm_ms int, tts_first_ms int, "
            "client_ttfa_ms int, total_ms int, timestamp string"
        )
        spark.createDataFrame(rows_data, schema=schema).createOrReplaceTempView("_mlv_sample_rows")
        spark.sql(
            "DELETE FROM " + _samples_table()
            + " WHERE run_id = '" + run_id + "' AND dataset = '" + run["dataset"]
            + "' AND language = '" + run["language"] + "'"
        )
        spark.sql("INSERT INTO " + _samples_table() + " SELECT * FROM _mlv_sample_rows")
