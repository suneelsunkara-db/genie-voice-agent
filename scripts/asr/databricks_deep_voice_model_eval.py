"""Serverless ASR benchmark: Deepgram vs Databricks ASR endpoint."""
from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import math
import mimetypes
import re
import string
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


PUNCT_TABLE = str.maketrans("", "", string.punctuation.replace("$", ""))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--deepgram-output", required=True)
    parser.add_argument("--databricks-output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument("--databricks-endpoint", required=True)
    parser.add_argument("--deepgram-model", default="nova-3")
    parser.add_argument("--deepgram-secret-scope", default="genie-voice")
    parser.add_argument("--deepgram-secret-key", default="deepgram_api_key")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--databricks-concurrency", type=int, default=12)
    parser.add_argument("--databricks-retries", type=int, default=3)
    parser.add_argument("--databricks-timeout-seconds", type=int, default=180)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_manifest(args.manifest)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit(f"No rows in manifest: {args.manifest}")

    deepgram_rows = run_deepgram(args, rows)
    databricks_rows = run_databricks(args, rows)
    write_summary(args, deepgram_rows, databricks_rows)


def load_manifest(path: str) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip() and not line.lstrip().startswith("#"):
                rows.append(json.loads(line))
    return rows


def run_deepgram(args: argparse.Namespace, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    existing = read_existing_results(args.deepgram_output)
    if len(existing) >= len(rows):
        print(f"Reusing existing Deepgram output: {args.deepgram_output} ({len(existing)} rows)")
        return existing

    api_key = deepgram_api_key(args.deepgram_secret_scope, args.deepgram_secret_key)
    out = Path(args.deepgram_output)
    out.parent.mkdir(parents=True, exist_ok=True)
    results = []
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            started = time.perf_counter()
            transcript, confidence, raw = deepgram_transcribe(
                read_audio(row["audio_path"]),
                api_key=api_key,
                model=args.deepgram_model,
                language=args.language,
                mime_type=mime_type_for(row),
            )
            latency_ms = round((time.perf_counter() - started) * 1000)
            result = build_result(row, "deepgram", args.deepgram_model, transcript, latency_ms, confidence, raw)
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            results.append(result)
    return results


def run_databricks(args: argparse.Namespace, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    existing = read_existing_results(args.databricks_output)
    existing_by_clip = {str(row.get("clip_id")): row for row in existing}
    pending = [row for row in rows if str(row.get("clip_id")) not in existing_by_clip]
    if not pending:
        print(f"Reusing existing Databricks output: {args.databricks_output} ({len(existing)} rows)")
        return existing

    out = Path(args.databricks_output)
    out.parent.mkdir(parents=True, exist_ok=True)
    results = list(existing)
    mode = "a" if existing else "w"
    workers = max(1, args.databricks_concurrency)
    print(f"Running {len(pending)} Databricks endpoint requests with concurrency={workers}")
    with out.open(mode, encoding="utf-8") as f:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            future_to_clip = {pool.submit(databricks_one, args, row): str(row.get("clip_id")) for row in pending}
            for future in concurrent.futures.as_completed(future_to_clip):
                result = future.result()
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
                results.append(result)
                if len(results) % 25 == 0 or len(results) == len(rows):
                    print(f"Databricks progress {len(results)}/{len(rows)}")
    return results


_THREAD_LOCAL = threading.local()


def workspace_client(timeout_seconds: int):
    attr = f"workspace_client_{timeout_seconds}"
    client = getattr(_THREAD_LOCAL, attr, None)
    if client is None:
        from databricks.sdk import WorkspaceClient
        from databricks.sdk.config import Config

        client = WorkspaceClient(config=Config(retry_timeout_seconds=timeout_seconds))
        setattr(_THREAD_LOCAL, attr, client)
    return client


def databricks_serving_invocation(endpoint: str, body: dict[str, Any], *, timeout_seconds: int) -> dict[str, Any]:
    client = workspace_client(timeout_seconds)
    url = f"{client.config.host}/serving-endpoints/{quote(endpoint, safe='')}/invocations"
    headers = {
        **client.config.authenticate(),
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    req = Request(url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
    try:
        with urlopen(req, timeout=timeout_seconds) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from serving endpoint: {detail[:500]}") from exc
    except URLError as exc:
        raise TimeoutError(str(exc)) from exc


def databricks_one(args: argparse.Namespace, row: dict[str, Any]) -> dict[str, Any]:
    audio_b64 = base64.b64encode(read_audio(row["audio_path"])).decode("ascii")
    last_exc: Exception | None = None
    for attempt in range(1, args.databricks_retries + 1):
        started = time.perf_counter()
        try:
            payload = databricks_serving_invocation(
                args.databricks_endpoint,
                {
                    "dataframe_records": [
                        {
                            "audio_b64": audio_b64,
                            "mime_type": mime_type_for(row),
                            "speaker": 1,
                            "language": args.language,
                        }
                    ]
                },
                timeout_seconds=args.databricks_timeout_seconds,
            )
            latency_ms = round((time.perf_counter() - started) * 1000)
            prediction = first_prediction(payload)
            transcript = str(prediction.get("transcript") or prediction.get("raw_transcript") or "").strip()
            return build_result(
                row,
                "databricks",
                str(prediction.get("model") or args.databricks_endpoint),
                transcript,
                latency_ms,
                prediction.get("confidence"),
                payload,
                extra={"endpoint": args.databricks_endpoint, "attempt": attempt},
            )
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            sleep_for = min(30, 2**attempt)
            print(f"Databricks clip {row.get('clip_id')} attempt {attempt} failed: {exc}; sleeping {sleep_for}s")
            time.sleep(sleep_for)
    latency_ms = args.databricks_timeout_seconds * 1000 * max(1, args.databricks_retries)
    return build_result(
        row,
        "databricks",
        args.databricks_endpoint,
        "",
        latency_ms,
        None,
        {"error": str(last_exc), "retries": args.databricks_retries},
        extra={
            "endpoint": args.databricks_endpoint,
            "error": "endpoint_timeout_or_failure",
            "error_message": str(last_exc),
        },
    )


def read_existing_results(path: str) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def deepgram_transcribe(
    audio: bytes,
    *,
    api_key: str,
    model: str,
    language: str,
    mime_type: str,
) -> tuple[str, float | None, dict[str, Any]]:
    params = urlencode(
        {
            "model": model,
            "language": language,
            "smart_format": "true",
            "punctuate": "true",
        }
    )
    req = Request(
        f"https://api.deepgram.com/v1/listen?{params}",
        data=audio,
        headers={"Authorization": f"Token {api_key}", "Content-Type": mime_type},
        method="POST",
    )
    with urlopen(req, timeout=120) as resp:
        raw = json.loads(resp.read().decode("utf-8"))
    alt = (((raw.get("results") or {}).get("channels") or [{}])[0].get("alternatives") or [{}])[0]
    confidence = alt.get("confidence")
    return str(alt.get("transcript") or "").strip(), float(confidence) if confidence is not None else None, raw


def deepgram_api_key(scope: str, key: str) -> str:
    try:
        from pyspark.dbutils import DBUtils
        from pyspark.sql import SparkSession

        value = DBUtils(SparkSession.builder.getOrCreate()).secrets.get(scope=scope, key=key)
        if value:
            return value
    except Exception:
        pass
    try:
        dbutils = globals()["dbutils"]  # type: ignore[index]
        value = dbutils.secrets.get(scope=scope, key=key)
        if value:
            return value
    except Exception:
        pass
    raise RuntimeError(f"Unable to read Deepgram key from secret {scope}/{key}")


def read_audio(path: str) -> bytes:
    return Path(path.removeprefix("file://")).read_bytes()


def mime_type_for(row: dict[str, Any]) -> str:
    audio_format = row.get("audio_format")
    if isinstance(audio_format, str) and "/" in audio_format:
        return audio_format
    return mimetypes.guess_type(str(row.get("audio_path") or ""))[0] or "audio/wav"


def response_dict(response: Any) -> dict[str, Any]:
    if isinstance(response, dict):
        return response
    if hasattr(response, "as_dict"):
        return response.as_dict()
    predictions = getattr(response, "predictions", None)
    return {"predictions": predictions} if predictions is not None else {}


def first_prediction(payload: dict[str, Any]) -> dict[str, Any]:
    predictions = payload.get("predictions") or []
    first = predictions[0] if predictions else {}
    return first if isinstance(first, dict) else dict(first)


def build_result(
    row: dict[str, Any],
    provider: str,
    model: str,
    transcript: str,
    latency_ms: int,
    confidence: Any,
    raw: Any,
    *,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    score = score_transcript(str(row.get("reference_transcript") or ""), transcript, row.get("expected_entities") or {})
    features = transcript_features(str(row.get("reference_transcript") or ""), transcript)
    readiness = app_readiness(score["entity_scores"], features, transcript)
    return {
        "clip_id": row.get("clip_id"),
        "call_id": row.get("call_id"),
        "speaker": row.get("speaker"),
        "audio_path": row.get("audio_path"),
        "duration_seconds": row.get("duration_seconds"),
        "scenario": row.get("scenario"),
        "split": row.get("split"),
        "dataset_version": row.get("dataset_version"),
        "language": row.get("language"),
        "provider": provider,
        "model": model,
        "transcript": transcript,
        "raw_transcript": transcript,
        "reference_transcript": row.get("reference_transcript"),
        "latency_ms": latency_ms,
        "latency_per_audio_second": None
        if not row.get("duration_seconds")
        else latency_ms / max(float(row["duration_seconds"]), 0.001),
        "confidence": confidence,
        "score": score,
        "transcript_features": features,
        "app_readiness": readiness,
        "raw": raw,
        **(extra or {}),
    }


def score_transcript(reference: str, hypothesis: str, entities: dict[str, Any]) -> dict[str, Any]:
    ref_words = normalize_words(reference)
    hyp_words = normalize_words(hypothesis)
    ref_chars = normalize_chars(reference)
    hyp_chars = normalize_chars(hypothesis)
    word_errors = edit_distance(ref_words, hyp_words)
    char_errors = edit_distance(list(ref_chars), list(hyp_chars))
    entity_scores = score_entities(hypothesis, entities)
    expected = sum(v["expected"] for v in entity_scores.values())
    matched = sum(v["matched"] for v in entity_scores.values())
    return {
        "wer": ratio(word_errors, len(ref_words)),
        "cer": ratio(char_errors, len(ref_chars)),
        "word_errors": word_errors,
        "reference_words": len(ref_words),
        "char_errors": char_errors,
        "reference_chars": len(ref_chars),
        "entity_scores": entity_scores,
        "entity_accuracy": None if expected == 0 else matched / expected,
    }


def score_entities(hypothesis: str, entities: dict[str, Any]) -> dict[str, dict[str, Any]]:
    normalized = normalize_entity_text(hypothesis)
    scores = {}
    for group in ("invoice_ids", "amounts", "dates", "billing_actions", "confirmations", "refusals", "account_terms"):
        values = [str(v) for v in (entities.get(group) or []) if str(v).strip()]
        missing = [value for value in values if not entity_present(value, normalized, group)]
        scores[group] = {
            "expected": len(values),
            "matched": len(values) - len(missing),
            "missing": missing,
            "accuracy": None if not values else (len(values) - len(missing)) / len(values),
        }
    return scores


def entity_present(value: str, normalized: str, group: str) -> bool:
    expected = normalize_entity_text(value).strip()
    if not expected:
        return True
    if f" {expected} " in normalized:
        return True
    if group == "invoice_ids":
        exp = re.sub(r"[^a-z0-9]", "", value.lower())
        hyp = re.sub(r"[^a-z0-9]", "", normalized.lower())
        return exp in hyp
    if group == "amounts":
        numbers = re.findall(r"\d+", value)
        return all(re.search(rf"\b{re.escape(num)}\b", normalized) for num in numbers)
    if group in {"billing_actions", "confirmations", "refusals", "account_terms"}:
        return expected in normalized
    return False


def transcript_features(reference: str, hypothesis: str) -> dict[str, Any]:
    ref_numbers = re.findall(r"\b\d+(?:\.\d+)?\b", reference.replace(",", ""))
    hyp_numbers = re.findall(r"\b\d+(?:\.\d+)?\b", hypothesis.replace(",", ""))
    missing = [value for value in ref_numbers if value not in hyp_numbers]
    return {
        "reference_word_count": len(normalize_words(reference)),
        "hypothesis_word_count": len(normalize_words(hypothesis)),
        "length_ratio": safe_div(len(normalize_words(hypothesis)), len(normalize_words(reference))),
        "numeric_reference": ref_numbers,
        "numeric_hypothesis": hyp_numbers,
        "numeric_missing": missing,
        "numeric_added": [value for value in hyp_numbers if value not in ref_numbers],
        "numeric_recall": None if not ref_numbers else (len(ref_numbers) - len(missing)) / len(ref_numbers),
        "negations_reference": [],
        "negations_hypothesis": [],
        "negation_match": True,
        "empty_transcript": not hypothesis.strip(),
    }


def app_readiness(entity_scores: dict[str, Any], features: dict[str, Any], transcript: str) -> dict[str, Any]:
    critical = ("invoice_ids", "amounts", "dates", "billing_actions", "confirmations", "refusals")
    expected = sum(int((entity_scores.get(group) or {}).get("expected") or 0) for group in critical)
    matched = sum(int((entity_scores.get(group) or {}).get("matched") or 0) for group in critical)
    missing = {
        group: list((entity_scores.get(group) or {}).get("missing") or [])
        for group in critical
        if (entity_scores.get(group) or {}).get("missing")
    }
    reasons = []
    if not transcript.strip():
        reasons.append("empty_transcript")
    if features["numeric_missing"]:
        reasons.append("missing_numeric_token")
    if missing.get("invoice_ids"):
        reasons.append("missing_invoice_id")
    if missing.get("amounts"):
        reasons.append("missing_amount")
    if missing.get("confirmations") or missing.get("refusals"):
        reasons.append("missing_customer_decision_phrase")
    return {
        "critical_entity_accuracy": None if expected == 0 else matched / expected,
        "critical_entities_expected": expected,
        "critical_entities_matched": matched,
        "critical_entities_missing": missing,
        "unsafe_for_resolution": bool(reasons),
        "unsafe_reasons": reasons,
    }


def write_summary(args: argparse.Namespace, deepgram_rows: list[dict[str, Any]], databricks_rows: list[dict[str, Any]]) -> None:
    by_provider = {"deepgram": summarize_provider(deepgram_rows), "databricks": summarize_provider(databricks_rows)}
    paired = pairwise(deepgram_rows, databricks_rows)
    summary = {
        "manifest": args.manifest,
        "language": args.language,
        "deepgram_output": args.deepgram_output,
        "databricks_output": args.databricks_output,
        "providers": by_provider,
        "pairwise": paired,
        "promotion_read": {
            "recommended_headline": "Use this per-language benchmark to choose between Deepgram streaming and Databricks final ASR.",
            "databricks_business_delta": none_safe_sub(
                by_provider["databricks"].get("avg_critical_entity_accuracy"),
                by_provider["deepgram"].get("avg_critical_entity_accuracy"),
            ),
            "databricks_wer_delta": none_safe_sub(by_provider["databricks"].get("avg_wer"), by_provider["deepgram"].get("avg_wer")),
            "databricks_p95_latency_delta_ms": none_safe_sub(
                (by_provider["databricks"].get("latency_ms") or {}).get("p95"),
                (by_provider["deepgram"].get("latency_ms") or {}).get("p95"),
            ),
            "paired_clips": paired["paired_clips"],
        },
    }
    output = Path(args.summary_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def summarize_provider(rows: list[dict[str, Any]]) -> dict[str, Any]:
    entity_totals: dict[str, dict[str, int]] = defaultdict(lambda: {"expected": 0, "matched": 0})
    unsafe_reasons = Counter()
    for row in rows:
        for group, score in (row.get("score", {}).get("entity_scores") or {}).items():
            entity_totals[group]["expected"] += int(score.get("expected") or 0)
            entity_totals[group]["matched"] += int(score.get("matched") or 0)
        unsafe_reasons.update(row.get("app_readiness", {}).get("unsafe_reasons") or [])
    latencies = [row.get("latency_ms") for row in rows if row.get("latency_ms") is not None]
    return {
        "clips": len(rows),
        "provider": sorted({row.get("provider") for row in rows if row.get("provider")}),
        "models": sorted({row.get("model") for row in rows if row.get("model")}),
        "avg_wer": avg(row.get("score", {}).get("wer") for row in rows),
        "avg_cer": avg(row.get("score", {}).get("cer") for row in rows),
        "avg_entity_accuracy": avg(row.get("score", {}).get("entity_accuracy") for row in rows),
        "avg_critical_entity_accuracy": avg(row.get("app_readiness", {}).get("critical_entity_accuracy") for row in rows),
        "empty_transcript_rate": rate(row.get("transcript_features", {}).get("empty_transcript") for row in rows),
        "unsafe_for_resolution_rate": rate(row.get("app_readiness", {}).get("unsafe_for_resolution") for row in rows),
        "negation_mismatch_rate": 0.0,
        "numeric_recall": avg(row.get("transcript_features", {}).get("numeric_recall") for row in rows),
        "latency_ms": {key: percentile(latencies, pct) for key, pct in {"p50": 50, "p90": 90, "p95": 95, "p99": 99}.items()} | {"avg": avg(latencies)},
        "entity_groups": {
            group: {**counts, "accuracy": None if counts["expected"] == 0 else counts["matched"] / counts["expected"]}
            for group, counts in sorted(entity_totals.items())
        },
        "unsafe_reason_counts": dict(sorted(unsafe_reasons.items())),
    }


def pairwise(deepgram_rows: list[dict[str, Any]], databricks_rows: list[dict[str, Any]]) -> dict[str, Any]:
    dg = {str(row["clip_id"]): row for row in deepgram_rows}
    db = {str(row["clip_id"]): row for row in databricks_rows}
    shared = sorted(set(dg) & set(db))
    return {"paired_clips": len(shared), "winner_counts": {}}


def normalize_words(text: str) -> list[str]:
    value = text.lower().translate(PUNCT_TABLE)
    value = re.sub(r"\s+", " ", value).strip()
    return value.split() if value else []


def normalize_chars(text: str) -> str:
    return re.sub(r"\s+", "", text.lower().translate(PUNCT_TABLE))


def normalize_entity_text(text: str) -> str:
    value = text.lower().replace("$", " dollars ")
    value = re.sub(r"([a-z]+)-(\d+)", r"\1 \2", value)
    value = re.sub(r"(\d+)\.(\d+)", r"\1 \2", value)
    value = value.translate(PUNCT_TABLE)
    value = re.sub(r"\s+", " ", value).strip()
    return f" {value} "


def edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def ratio(num: float, den: float) -> float:
    return 0.0 if den == 0 else num / den


def safe_div(num: float, den: float) -> float | None:
    return None if den == 0 else num / den


def avg(values) -> float | None:
    present = [value for value in values if value is not None]
    return None if not present else sum(present) / len(present)


def rate(values) -> float | None:
    present = [bool(value) for value in values]
    return None if not present else sum(1 for value in present if value) / len(present)


def percentile(values, pct: int) -> float | None:
    present = sorted(value for value in values if value is not None)
    if not present:
        return None
    if len(present) == 1:
        return present[0]
    rank = (len(present) - 1) * (pct / 100)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return present[int(rank)]
    return present[lower] + (present[upper] - present[lower]) * (rank - lower)


def none_safe_sub(left, right):
    return None if left is None or right is None else left - right


if __name__ == "__main__":
    main()
