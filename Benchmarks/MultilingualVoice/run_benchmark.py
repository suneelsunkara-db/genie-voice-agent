"""Multilingual voice benchmark runner — one (dataset, language) pair per invocation.

In the task-per-pair architecture, the Databricks job submits one task per
(dataset, language) pair. Each task runs this script once, scores its pair, and
writes a row to the ``benchmark_runs`` Delta table (plus per-sample rows to
``benchmark_samples``). Resume = re-run the job; each task checks Delta for its
pair in the current ``run_id`` and exits early if already complete.

Run offline against fixtures with ``--fixture`` (no network, no HF download).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import quantiles

# Entry-point dir: prefer MLV_BENCHMARK_DIR env, then sys.argv[0] (set by
# spark_python_task even under exec), then __file__ for local runs, then cwd.
def _resolve_benchmark_dir() -> Path:
    raw = os.environ.get("MLV_BENCHMARK_DIR")
    if raw:
        return Path(raw)
    if sys.argv and sys.argv[0]:
        entry = Path(sys.argv[0]).resolve()
        if entry.name == "run_benchmark.py" and entry.parent.is_dir():
            return entry.parent
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd()


_BENCHMARK_DIR = _resolve_benchmark_dir()
sys.path.insert(0, str(_BENCHMARK_DIR))

import languages as langmap  # noqa: E402
from evaluators import EVALUATORS, evaluate_tts_roundtrip, primary_metric, primary_metric_name  # noqa: E402
from issue_log import IssueTracker  # noqa: E402
from realtime_client import DATASET_CAPABILITIES  # noqa: E402
from volume_logging import issues_log_path, setup_run_logging  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("mlv")

_FIXTURES = _BENCHMARK_DIR / "fixtures"
_RESULTS = _BENCHMARK_DIR / "results"

DATASET_EVALUATOR = {"fleurs": "asr", "belebele": "mcq", "ccfqa": "qa"}

# Belebele sends a full spoken passage (mean ~71 s, max ~98 s) as one turn and
# marks the end with audio.end. These overrides stop server-side VAD from
# finalizing the turn mid-passage: max_turn_seconds is an audio-duration ceiling
# above the longest passage, and vad_silence_ms is set high so inter-sentence
# pauses in the concatenated FLEURS audio don't end the turn early.
BELEBELE_MAX_TURN_SECONDS = 150
BELEBELE_VAD_SILENCE_MS = 600_000
# STT of a ~100 s passage plus LLM + TTS needs a longer per-message wait than the
# short FLEURS/CCFQA turns.
BELEBELE_TIMEOUT_S = 300.0

# FLEURS clips are single read-speech utterances (mostly <30 s). Setting these
# overrides puts the client in BATCH mode: it streams the audio as fast as the
# socket allows instead of pacing it at wall-clock speed, which is ~10-20x faster
# per turn. audio.end still marks the boundary and the high VAD ceiling keeps a
# brief pause from finalizing the turn early, so the whole clip is transcribed.
# Accuracy is unchanged (same audio + STT model) — only the artificial real-time
# delay is removed. This is what made full FLEURS sweeps take 20-40 min.
FLEURS_MAX_TURN_SECONDS = 150
FLEURS_VAD_SILENCE_MS = 30_000


def _percentiles(values: list[int]) -> dict[str, int]:
    if not values:
        return {}
    sorted_v = sorted(values)
    n = len(sorted_v)
    return {
        "p50": sorted_v[n // 2],
        "p95": sorted_v[min(n - 1, int(n * 0.95))],
        "p99": sorted_v[min(n - 1, int(n * 0.99))],
        "mean": round(sum(sorted_v) / n),
    }


def _apply_job_wiring(args: argparse.Namespace) -> None:
    """Populate in-process env from parameters + secret scope.

    Serverless spark_python_task ignores env-spec env vars, so config + non-secrets
    arrive as CLI parameters. Secrets are read at runtime via the ambient identity.
    """
    if args.benchmark_dir:
        os.environ["MLV_BENCHMARK_DIR"] = args.benchmark_dir
    if args.config:
        os.environ["GENIE_CONFIG"] = args.config
    if args.databricks_host:
        os.environ["DATABRICKS_HOST"] = args.databricks_host
    if args.sp_client_id:
        os.environ["MLV_SP_CLIENT_ID"] = args.sp_client_id
    if args.secret_scope:
        from databricks_auth import read_workspace_secret

        if args.sp_secret_key:
            os.environ["MLV_SP_CLIENT_SECRET"] = read_workspace_secret(args.secret_scope, args.sp_secret_key)
        if args.hf_secret_key:
            token = read_workspace_secret(args.secret_scope, args.hf_secret_key)
            os.environ["HF_TOKEN"] = token
            os.environ["HUGGING_FACE_HUB_TOKEN"] = token


# ---------------------------------------------------------------------------
# Client construction
# ---------------------------------------------------------------------------
def _build_token_provider(args: argparse.Namespace):
    from databricks_auth import build_token_provider
    from paths import benchmark_sp_credentials, databricks_host, databricks_profile

    static = args.auth_token or os.getenv("MLV_AUTH_TOKEN")
    client_id, client_secret = benchmark_sp_credentials()
    profile = (args.databricks_profile or "").strip() or databricks_profile()
    return build_token_provider(
        host=databricks_host(),
        client_id=client_id,
        client_secret=client_secret,
        profile=profile,
        static_token=static,
    )


def build_client(args: argparse.Namespace, lang: str, *, dataset: str):
    if args.transport == "inprocess":
        from realtime_client import InProcessRealtimeClient

        sys.path.insert(0, str(_BENCHMARK_DIR.parents[1]))
        from realtime_api.app import create_app  # type: ignore

        return InProcessRealtimeClient(create_app(), language=lang)

    from realtime_client import WebSocketRealtimeClient, detect_capabilities

    token_provider = _build_token_provider(args)
    available = detect_capabilities(args.api_host, prefix=args.api_prefix, auth_token=token_provider)
    log.info("API capabilities available: %s", {k: v for k, v in available.items() if v})
    default_capability = DATASET_CAPABILITIES.get(dataset or "", "speech-llm-toolassist-speech")
    timeout_s = BELEBELE_TIMEOUT_S if dataset == "belebele" else 180.0
    return WebSocketRealtimeClient(
        args.api_host,
        prefix=args.api_prefix,
        language=lang,
        auth_token=token_provider,
        default_capability=default_capability,
        available_capabilities=available,
        timeout_s=timeout_s,
    )


# ---------------------------------------------------------------------------
# One (dataset, language) run — generator pipeline, one PCM resident
# ---------------------------------------------------------------------------
def _log_turn_issues(
    tracker: IssueTracker, *, dataset: str, lang: str, index: int, result, phase: str = "inference",
) -> None:
    if result.error:
        tracker.record(
            dataset=dataset, language=lang, sample_index=index, phase=phase,
            kind="api_error", message=result.error,
        )
    elif phase == "inference" and not (result.transcript or "").strip():
        tracker.record(
            dataset=dataset, language=lang, sample_index=index, phase=phase,
            kind="empty_transcript", message="API returned an empty transcript",
        )


def run_one(
    args: argparse.Namespace, dataset: str, lang: str, tracker: IssueTracker,
) -> tuple[dict, list[dict]]:
    """Score one (dataset, language) pair. Returns (run_dict, sample_rows).

    Audio is read from the Volume-staged artifact written by the prepare phase,
    not from HuggingFace — the benchmark phase never touches parquet/arrow, so
    each parallel task stays tiny in memory.
    """
    from paths import benchmark_results_dir
    from staging import load_staged

    staged_dir = Path(args.out_dir) if args.out_dir else benchmark_results_dir()
    evaluator = DATASET_EVALUATOR[dataset]
    tracker.reset_run()
    client = build_client(args, lang, dataset=dataset)

    rows: list[dict] = []
    latencies: dict[str, list[int]] = {
        "stt_ms": [], "llm_ms": [], "tts_first_ms": [], "client_ttfa_ms": [], "total_ms": [],
    }
    errors = 0
    sample_count = 0
    stt_capability = DATASET_CAPABILITIES.get(dataset, "speech-llm-toolassist-speech")

    turn_overrides: dict = {}
    if dataset == "belebele":
        # Full passage as audio; question + options handed to the LLM as text
        # context; VAD held off so the long turn ends only on audio.end.
        turn_overrides = {
            "max_turn_seconds": BELEBELE_MAX_TURN_SECONDS,
            "vad_silence_ms": BELEBELE_VAD_SILENCE_MS,
        }
    elif dataset == "fleurs":
        # Batch mode (no real-time pacing) — see FLEURS_* constants above.
        turn_overrides = {
            "max_turn_seconds": FLEURS_MAX_TURN_SECONDS,
            "vad_silence_ms": FLEURS_VAD_SILENCE_MS,
        }

    for i, sample in enumerate(load_staged(dataset, lang, args.limit, out_dir=staged_dir)):
        sample_count += 1
        result = None
        for attempt in range(3):
            result = client.run_turn(
                sample["pcm"], sample_rate_hz=sample["sample_rate"],
                language=lang, capability=stt_capability,
                context=sample.get("context"), **turn_overrides,
            )
            if not result.error:
                break
            if attempt < 2 and _retryable_error(result.error):
                from realtime_client import _backoff_sleep

                log.warning("turn %d retrying after error (attempt %d): %s", i, attempt + 1, result.error)
                _backoff_sleep(attempt)
                continue
            break
        if result is None:
            raise RuntimeError(f"turn {i} produced no result after retries")
        if result.error:
            errors += 1
            log.warning("turn %d error: %s", i, result.error)
        _log_turn_issues(tracker, dataset=dataset, lang=lang, index=i, result=result)

        if dataset == "fleurs" and args.tts_roundtrip:
            reference = str(sample.get("reference") or "").strip()
            if reference:
                tts_result = client.synthesize(reference, language=lang)
                if tts_result.error:
                    errors += 1
                    tracker.record(
                        dataset=dataset, language=lang, sample_index=i, phase="roundtrip",
                        kind="tts_synthesize_error", message=tts_result.error,
                    )
                elif tts_result.tts_audio:
                    result.response_text = reference
                    result.tts_audio = tts_result.tts_audio
                    result.tts_sample_rate = tts_result.tts_sample_rate
                    result.tts_first_ms = tts_result.tts_first_ms
                    result.client_ttfa_ms = tts_result.client_ttfa_ms
                    client.roundtrip_tts(result, spoken_text=reference, language=lang)
                else:
                    tracker.record(
                        dataset=dataset, language=lang, sample_index=i, phase="roundtrip",
                        kind="no_tts_audio", message="text-to-speech returned no audio",
                    )
            rt = result.roundtrip or {}
            if rt.get("error"):
                errors += 1
                tracker.record(
                    dataset=dataset, language=lang, sample_index=i, phase="roundtrip",
                    kind="roundtrip_error", message=str(rt["error"]),
                )
            elif result.tts_audio and not (rt.get("reheard_text") or "").strip():
                tracker.record(
                    dataset=dataset, language=lang, sample_index=i, phase="roundtrip",
                    kind="roundtrip_empty_transcript", message="round-trip STT returned empty text",
                )

        for key in latencies:
            val = getattr(result, key, None)
            if isinstance(val, int):
                latencies[key].append(val)
        rows.append({
            "dataset": dataset,
            "language": lang,
            "reference": sample.get("reference"),
            "correct_choice": sample.get("correct_choice"),
            "transcript": result.transcript,
            "response": result.response_text,
            "detected_language": result.detected_language,
            "tts_audio_bytes": len(result.tts_audio),
            "tts_roundtrip": result.roundtrip,
            "error": result.error or (result.roundtrip or {}).get("error"),
            "stt_ms": result.stt_ms, "llm_ms": result.llm_ms,
            "tts_first_ms": result.tts_first_ms, "client_ttfa_ms": result.client_ttfa_ms,
            "total_ms": result.total_ms,
        })
        del sample, result

    scores = EVALUATORS[evaluator](rows)
    if dataset == "fleurs" and args.tts_roundtrip:
        scores.update(evaluate_tts_roundtrip(rows))
    scores["primary_metric"] = primary_metric_name(evaluator, lang)

    latency_summary = {k: _percentiles(v) for k, v in latencies.items() if v}
    issues = tracker.run_summary()
    run = {
        "dataset": dataset,
        "language": lang,
        "evaluator": evaluator,
        "samples": sample_count,
        "errors": errors,
        "issues": issues,
        "primary_score": primary_metric(evaluator, scores, language=lang),
        "scores": scores,
        "latency_ms": latency_summary,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return run, rows


def _retryable_error(message: str) -> bool:
    msg = message.lower()
    return any(n in msg for n in ("timeout", "connectionclosed", "keepalive", "connection reset"))


# ---------------------------------------------------------------------------
# Fixture mode (offline)
# ---------------------------------------------------------------------------
def run_fixture(args: argparse.Namespace) -> list[dict]:
    runs = []
    for path in sorted(_FIXTURES.glob("*.json")):
        fx = json.loads(path.read_text(encoding="utf-8"))
        dataset = fx["dataset"]
        evaluator = DATASET_EVALUATOR[dataset]
        rows = fx["rows"]
        fx_lang = fx.get("language", "en")
        scores = EVALUATORS[evaluator](rows)
        if dataset == "fleurs":
            scores.update(evaluate_tts_roundtrip(rows))
        scores["primary_metric"] = primary_metric_name(evaluator, fx_lang)
        runs.append({
            "dataset": dataset,
            "language": fx_lang,
            "evaluator": evaluator,
            "samples": len(rows),
            "errors": sum(1 for r in rows if r.get("error")),
            "primary_score": primary_metric(evaluator, scores, language=fx_lang),
            "scores": scores,
            "latency_ms": fx.get("latency_ms", {}),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
    return runs


def _write_fixture_summary(runs: list[dict], out_dir: Path) -> Path:
    from volume_io import write_text

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runs": runs,
        "datasets": sorted({r["dataset"] for r in runs}),
        "languages": sorted({r["language"] for r in runs}),
    }
    path = out_dir / "summary.json"
    write_text(path, json.dumps(payload, indent=2, ensure_ascii=False))
    return path


# ---------------------------------------------------------------------------
# Args + main
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multilingual voice benchmark (one dataset×language pair)")
    p.add_argument("--dataset", default="all", choices=["all", "fleurs", "belebele", "ccfqa"])
    p.add_argument("--language", default="", help="single 2-letter code; empty = all covered for the dataset")
    p.add_argument("--languages", default="", help="comma-separated codes (prepare phase; overrides --language)")
    p.add_argument(
        "--mode", default="benchmark", choices=["benchmark", "prepare"],
        help="prepare: extract samples to the Volume; benchmark: read staged data and score",
    )
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--transport", default="ws", choices=["ws", "inprocess"])
    # Empty defaults -> resolved from config (realtime_voice.benchmark) in main().
    p.add_argument("--api-host", default="")
    p.add_argument("--api-prefix", default="")
    p.add_argument("--auth-token", default=None)
    p.add_argument("--databricks-profile", default="")
    p.add_argument("--tts-roundtrip", action="store_true")
    p.add_argument(
        "--max-audio-seconds", type=float, default=120.0,
        help="Belebele passage safety ceiling (passages max ~98s; full passage is sent)",
    )
    p.add_argument("--fixture", action="store_true")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--run-label", default="")
    p.add_argument("--run-id", default="", help="sweep id shared across tasks")
    # Job wiring (serverless env-spec env vars are not propagated).
    p.add_argument("--benchmark-dir", default=None)
    p.add_argument("--config", default=None)
    p.add_argument("--databricks-host", default=None)
    p.add_argument("--sp-client-id", default=None)
    p.add_argument("--secret-scope", default=None)
    p.add_argument("--sp-secret-key", default=None)
    p.add_argument("--hf-secret-key", default=None)
    return p.parse_args(argv)


def _run_prepare(args, pairs, out_dir: Path, run_id: str) -> int:
    """Prepare phase: stream each pair from HF and stage it to the Volume.

    Runs as one task, one pair at a time, so the heavyweight parquet decode is
    bounded to a single pair's memory and never happens on the parallel
    benchmark tasks.
    """
    from staging import is_staged, stage_pair

    log_mgr = setup_run_logging(out_dir, run_label=args.run_label or f"prepare_{run_id}")
    staged_pairs = 0
    staged_samples = 0
    for dataset, lg in pairs:
        if is_staged(dataset, lg, args.limit, out_dir=out_dir):
            log.info("stage skip %s@%s (already staged, limit=%d)", dataset, lg, args.limit)
            continue
        t0 = time.perf_counter()
        n = stage_pair(dataset, lg, args.limit, args.max_audio_seconds, out_dir=out_dir)
        staged_pairs += 1
        staged_samples += n
        log.info("staged %s@%s: %d samples in %.1fs", dataset, lg, n, time.perf_counter() - t0)
        log_mgr.checkpoint_to_volume()
    log.info(
        "prepare done: %d/%d pairs newly staged, %d samples (limit=%d)",
        staged_pairs, len(pairs), staged_samples, args.limit,
    )
    log_mgr.checkpoint_to_volume()
    return 0


def main(argv: list[str] | None = None) -> int:
    from hf_datasets import ensure_hf_token

    args = parse_args(argv)
    _apply_job_wiring(args)
    ensure_hf_token()

    _hf = os.getenv("HF_TOKEN", "")
    log.info(
        "startup HF_TOKEN present=%s len=%d prefix=%r is_hf=%s",
        bool(_hf), len(_hf), _hf[:10], _hf.startswith("hf_"),
    )

    if args.fixture:
        out_dir = Path(args.out_dir or str(_RESULTS))
        log_mgr = setup_run_logging(out_dir, run_label=args.run_label or None)
        runs = run_fixture(args)
        path = _write_fixture_summary(runs, out_dir)
        log.info("wrote fixture summary -> %s (%d runs)", path, len(runs))
        log_mgr.checkpoint_to_volume()
        return 0

    from paths import benchmark_results_dir

    out_dir = Path(args.out_dir or str(benchmark_results_dir()))
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    # Resolve (dataset, language) pairs. The prepare phase passes --languages
    # (comma list) to stage many pairs; benchmark tasks pass a single --language.
    if args.languages.strip():
        requested: list[str] | None = [c.strip() for c in args.languages.split(",") if c.strip()]
    elif args.language.strip():
        requested = [args.language.strip()]
    else:
        requested = None
    datasets = ["fleurs", "belebele", "ccfqa"] if args.dataset == "all" else [args.dataset]
    pairs: list[tuple[str, str]] = []
    for dataset in datasets:
        pairs.extend((dataset, l) for l in langmap.resolve_languages(dataset, requested))
    if not pairs:
        log.error("no (dataset, language) pairs to run")
        return 1

    if args.mode == "prepare":
        return _run_prepare(args, pairs, out_dir, run_id)

    if not args.api_host:
        from paths import benchmark_api_host, benchmark_api_prefix

        args.api_host = benchmark_api_host()
        args.api_prefix = args.api_prefix or benchmark_api_prefix()

    log_mgr = setup_run_logging(out_dir, run_label=args.run_label or run_id)
    tracker = IssueTracker(issues_log_path(out_dir))

    from results_store import completed_pairs, write_run

    completed = completed_pairs(run_id) if os.getenv("DATABRICKS_RUNTIME_VERSION") else set()

    for dataset, lg in pairs:
        key = f"{dataset}@{lg}"
        if key in completed:
            log.info("skip %s (already complete for run_id=%s)", key, run_id)
            continue
        log.info("=== %s @%s (run_id=%s) ===", dataset, lg, run_id)
        t0 = time.perf_counter()
        try:
            run, sample_rows = run_one(args, dataset, lg, tracker)
        except Exception as exc:  # noqa: BLE001
            log.error("run failed for %s @%s: %s", dataset, lg, exc)
            tracker.record(
                dataset=dataset, language=lg, kind="run_failed",
                message=f"{type(exc).__name__}: {exc}", phase="load",
            )
            tracker.flush()
            log_mgr.checkpoint_to_volume()
            raise
        run["wall_seconds"] = round(time.perf_counter() - t0, 1)
        log.info("  primary=%s latency=%s issues=%d", run["primary_score"], run["latency_ms"], run["issues"]["count"])

        if os.getenv("DATABRICKS_RUNTIME_VERSION"):
            write_run(run_id, run, sample_rows=sample_rows)
        else:
            # Local mode: write summary.json for inspection.
            _write_local_summary(run, sample_rows, out_dir)

        tracker.flush()
        log_mgr.checkpoint_to_volume()

    log.info("done; results in Delta table (run_id=%s)", run_id)
    log_mgr.checkpoint_to_volume()
    return 0


def _write_local_summary(run: dict, sample_rows: list[dict], out_dir: Path) -> None:
    from volume_io import write_text

    payload = {"generated_at": datetime.now(timezone.utc).isoformat(), "runs": [run]}
    write_text(out_dir / "summary.json", json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    _code = main()
    if _code:
        raise SystemExit(_code)
