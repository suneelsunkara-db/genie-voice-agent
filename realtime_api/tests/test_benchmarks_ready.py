"""Unit tests for the benchmark readiness gate (select_ready_runs).

A benchmark run is only promoted to the default UI view once a single run_id has
swept the full language breadth AND carries accuracy + TTFT on every unit that
produced results. A partial/in-flight run must not replace the last full run.
"""
from __future__ import annotations

from realtime_api.benchmarks import select_ready_runs


def _run(run_id, language, *, ts, ttft=True, dataset="fleurs", samples=20, errors=0, score=0.12):
    latency = {"stt_ms": {"p50": 400}}
    if ttft:
        latency["client_ttfa_ms"] = {"p50": 1200, "p95": 1800}
    return {
        "run_id": run_id,
        "dataset": dataset,
        "language": language,
        "samples": samples,
        "errors": errors,
        "primary_score": score,
        "latency_ms": latency,
        "timestamp": ts,
    }


def test_full_run_beats_older_full_run() -> None:
    rows = [
        _run("r1", "en", ts="2026-01-01T00:00:00Z"),
        _run("r1", "hi", ts="2026-01-01T00:00:00Z"),
        _run("r2", "en", ts="2026-02-01T00:00:00Z"),
        _run("r2", "hi", ts="2026-02-01T00:00:00Z"),
    ]
    selected, meta = select_ready_runs(rows)
    assert meta["fleurs"]["ready"] is True
    assert meta["fleurs"]["run_id"] == "r2"
    assert {r["run_id"] for r in selected} == {"r2"}


def test_partial_newer_run_does_not_replace_full_run() -> None:
    rows = [
        _run("full", "en", ts="2026-01-01T00:00:00Z"),
        _run("full", "hi", ts="2026-01-01T00:00:00Z"),
        # Newer, but only one language — a subset sweep must not be promoted.
        _run("partial", "en", ts="2026-03-01T00:00:00Z"),
    ]
    selected, meta = select_ready_runs(rows)
    assert meta["fleurs"]["ready"] is True
    assert meta["fleurs"]["run_id"] == "full"
    assert {r["run_id"] for r in selected} == {"full"}


def test_missing_ttft_blocks_readiness_and_falls_back() -> None:
    rows = [
        _run("r1", "en", ts="2026-01-01T00:00:00Z", ttft=True),
        _run("r1", "hi", ts="2026-01-01T00:00:00Z", ttft=False),  # metric not ready
    ]
    selected, meta = select_ready_runs(rows)
    assert meta["fleurs"]["ready"] is False
    # Fallback keeps the latest row per language so the page is never blank.
    assert {r["language"] for r in selected} == {"en", "hi"}


def test_fully_errored_language_still_counts_ready() -> None:
    rows = [
        _run("r1", "en", ts="2026-01-01T00:00:00Z", ttft=True),
        # Every sample errored -> no audio -> no TTFT, but the unit finished.
        _run("r1", "hi", ts="2026-01-01T00:00:00Z", ttft=False, samples=20, errors=20),
    ]
    selected, meta = select_ready_runs(rows)
    assert meta["fleurs"]["ready"] is True
    assert meta["fleurs"]["run_id"] == "r1"
    assert len(selected) == 2


def test_datasets_selected_independently() -> None:
    rows = [
        _run("gen", "en", ts="2026-01-01T00:00:00Z", dataset="fleurs"),
        _run("gen", "hi", ts="2026-01-01T00:00:00Z", dataset="fleurs"),
        _run("vend", "en", ts="2026-01-02T00:00:00Z", dataset="fleurs_deepgram_stt"),
    ]
    selected, meta = select_ready_runs(rows)
    assert set(meta) == {"fleurs", "fleurs_deepgram_stt"}
    assert meta["fleurs"]["ready"] is True
    assert meta["fleurs_deepgram_stt"]["ready"] is True
    assert {r["dataset"] for r in selected} == {"fleurs", "fleurs_deepgram_stt"}
