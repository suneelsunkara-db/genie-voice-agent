"""Stream benchmark audio from HuggingFace parquets via HTTP range reads.

``load_*`` are **generators** yielding one decoded sample at a time, so at any
instant only one PCM buffer (~600 KB for 18 s @ 16 kHz) is resident. The caller
iterates, scores, and drops the reference before the next yield.

Parquets are read **remotely** through ``HfFileSystem`` (range requests), never
downloaded whole: a single 2M-Belebele language file is 5-7 GB but we only need
the first ``limit`` samples, which live in the first row group. We read only the
columns each loader needs and stop after ``limit`` rows, so peak memory is one
row-group column chunk — not the whole file. This keeps the job inside the
serverless container's memory and avoids multi-GB downloads to local disk.
"""
from __future__ import annotations

import io
import os
from functools import lru_cache
from typing import Any, Iterator

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import HfFileSystem, list_repo_files

from paths import config_dir

# Rows converted to Python at once via ``to_pylist()``. pyarrow reads a whole
# row-group column chunk into Arrow regardless, but a small batch bounds the
# (much larger) Python-side float explosion from decoded audio lists.
_PARQUET_BATCH_ROWS = 1


def _local_config_path() -> Path:
    return config_dir() / "config.local.yaml"


def _env_token(name: str) -> str | None:
    val = os.getenv(name)
    return val.strip() if val and val.strip() else None


def ensure_hf_token() -> None:
    if _env_token("HF_TOKEN") or _env_token("HUGGING_FACE_HUB_TOKEN"):
        return
    config_path = _local_config_path()
    if not config_path.exists():
        return
    import yaml

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    token = (cfg.get("secrets") or {}).get("hf_token")
    if token and str(token).strip():
        token = str(token).strip()
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token


@lru_cache(maxsize=8)
def _repo_files(repo: str) -> tuple[str, ...]:
    ensure_hf_token()
    return tuple(list_repo_files(repo, repo_type="dataset"))


# ---------------------------------------------------------------------------
# Remote parquet streaming (HTTP range reads — no whole-file download)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _hf_fs() -> HfFileSystem:
    ensure_hf_token()
    return HfFileSystem(token=_env_token("HF_TOKEN") or _env_token("HUGGING_FACE_HUB_TOKEN"))


def _hf_fs_path(repo: str, path: str) -> str:
    return f"datasets/{repo}/{path}"


# ---------------------------------------------------------------------------
# Audio decoding
# ---------------------------------------------------------------------------
def _decode_audio_field(audio: dict[str, Any]) -> tuple[np.ndarray, int]:
    """Decode an HF Audio column from parquet (bytes WAV or embedded float list)."""
    if not audio:
        raise ValueError("empty audio")
    if "array" in audio:
        return np.asarray(audio["array"], dtype=np.float32), int(audio.get("sampling_rate") or 16_000)
    if "wav" in audio:
        sr = int(audio.get("sampling_rate") or 16_000)
        return np.asarray(audio["wav"], dtype=np.float32), sr
    raw = audio.get("bytes")
    if raw:
        arr, sr = sf.read(io.BytesIO(raw))
        return np.asarray(arr, dtype=np.float32), int(sr)
    raise ValueError("unrecognized audio field")


def _float_to_pcm16(array: np.ndarray, sample_rate: int) -> tuple[bytes, int]:
    """Convert float32 audio to PCM s16le, resampling to 16 kHz if needed."""
    arr = np.clip(np.asarray(array, dtype=np.float32), -1.0, 1.0)
    pcm = (arr * 32767.0).astype("<i2").tobytes()
    accepted = {8_000, 16_000, 24_000, 48_000}
    if sample_rate not in accepted:
        from realtime_client import resample_pcm16

        pcm = resample_pcm16(pcm, sample_rate, 16_000)
        sample_rate = 16_000
    return pcm, sample_rate


# ---------------------------------------------------------------------------
# Parquet row streaming
# ---------------------------------------------------------------------------
def _parquet_rows(
    repo: str,
    path: str,
    *,
    columns: list[str] | None = None,
    max_rows: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Stream rows from one remote parquet via HTTP range reads.

    Only ``columns`` are read (projected against the file's actual schema, so a
    missing optional column is silently skipped), and iteration stops at
    ``max_rows``. Because pyarrow reads row groups lazily, stopping early means
    only the first row group(s) are ever fetched — not the whole file.
    """
    with _hf_fs().open(_hf_fs_path(repo, path), "rb") as fh:
        pf = pq.ParquetFile(fh)
        projected = None
        if columns:
            available = set(pf.schema_arrow.names)
            projected = [c for c in columns if c in available] or None
        seen = 0
        for batch in pf.iter_batches(batch_size=_PARQUET_BATCH_ROWS, columns=projected):
            for row in batch.to_pylist():
                yield row
                seen += 1
                if max_rows is not None and seen >= max_rows:
                    return


def _fleurs_parquet_path(config: str) -> str:
    files = _repo_files("google/fleurs")
    # Prefer the canonical test split, not a lexicographic first match.
    matches = [f for f in files if f.endswith(".parquet") and config in f and "/test-" in f]
    if not matches:
        matches = [f for f in files if f.endswith(".parquet") and config in f and "test" in f]
    if not matches:
        raise FileNotFoundError(f"no FLEURS test parquet for {config}")
    return sorted(matches)[0]


# ---------------------------------------------------------------------------
# Dataset loaders — generators yielding one sample at a time
# ---------------------------------------------------------------------------
def load_fleurs(lang: str, limit: int) -> Iterator[dict[str, Any]]:
    from languages import FLEURS

    config = FLEURS[lang]
    seen = 0
    cols = ["audio", "transcription", "raw_transcription"]
    for row in _parquet_rows(
        "google/fleurs", _fleurs_parquet_path(config), columns=cols, max_rows=limit
    ):
        arr, sr = _decode_audio_field(row["audio"])
        pcm, pcm_sr = _float_to_pcm16(arr, sr)
        yield {
            "pcm": pcm,
            "sample_rate": pcm_sr,
            "reference": row.get("transcription") or row.get("raw_transcription"),
        }
        seen += 1
        if seen >= limit:
            return


def load_belebele(lang: str, limit: int, max_audio_seconds: float) -> Iterator[dict[str, Any]]:
    """Spoken reading-comprehension MCQ (2M-Belebele).

    The dataset ships the passage as audio (``audio_segments``, FLEURS-matched
    sentence recordings) but the question and the four options as TEXT only
    (their ``*_audio`` columns are null). To score the voice pipeline faithfully
    we send the full passage AUDIO through STT and hand the exact question +
    options to the LLM as textual context (see ``session.start.context``) — no
    self-TTS of text we already have, which would only add re-transcription
    noise. The whole passage is captured (comprehension needs it), bounded only
    by ``max_audio_seconds`` as a safety ceiling.
    """
    from languages import BELEBELE

    code = BELEBELE[lang]
    path = f"data/lang={code}/{code}.parquet"
    cols = [
        "has_matched_audio", "audio_segments", "correct_answer_num",
        "question", "mc_answer1", "mc_answer2", "mc_answer3", "mc_answer4",
    ]
    seen = 0
    for row in _parquet_rows("facebook/2M-Belebele", path, columns=cols):
        if not row.get("has_matched_audio"):
            continue
        segments = row.get("audio_segments") or []
        chunks: list[np.ndarray] = []
        sr = 16_000
        total = 0
        # Safety ceiling only: passages max out ~98 s, so the full passage is
        # normally captured. Stop decoding past the cap so a pathological row
        # can't materialise unbounded float arrays in memory.
        cap = int(max_audio_seconds * sr)
        for group in segments:
            if not isinstance(group, list):
                continue
            for item in group:
                aud = (item or {}).get("audio")
                if not isinstance(aud, dict):
                    continue
                arr, sr = _decode_audio_field(aud)
                chunks.append(arr)
                total += len(arr)
            if total >= cap:
                break
        if not chunks:
            continue
        merged = np.concatenate(chunks)
        if len(merged) > cap:
            merged = merged[:cap]
        pcm, pcm_sr = _float_to_pcm16(merged, sr)
        yield {
            "pcm": pcm,
            "sample_rate": pcm_sr,
            "correct_choice": row.get("correct_answer_num"),
            "context": _belebele_context(row),
        }
        seen += 1
        if seen >= limit:
            return


def _belebele_context(row: dict[str, Any]) -> str:
    """Build the textual grounding for a Belebele item: question + options + task.

    The passage arrives as audio (transcribed by STT); this supplies the parts
    the dataset only provides as text. The final instruction pins the answer to a
    parseable single digit regardless of the assistant's conversational style.
    """
    question = str(row.get("question") or "").strip()
    options = [str(row.get(f"mc_answer{i}") or "").strip() for i in range(1, 5)]
    lines = [f"Question: {question}", "Options:"]
    lines += [f"{i}. {opt}" for i, opt in enumerate(options, start=1)]
    lines.append("Answer with only the number (1, 2, 3, or 4) of the correct option.")
    return "\n".join(lines)


def load_ccfqa(lang: str, limit: int) -> Iterator[dict[str, Any]]:
    from languages import CCFQA

    iso3 = CCFQA[lang]
    q_key, a_key = f"{iso3}_q", f"{iso3}_a"
    cols = ["lang", "audio", q_key, a_key, "src_q", "src_a"]
    files = sorted(
        f for f in _repo_files("yxdu/ccfqa")
        if f.endswith(".parquet") and "/test-" in f
    )
    seen = 0
    for path in files:
        for row in _parquet_rows("yxdu/ccfqa", path, columns=cols):
            if str(row.get("lang") or "") != iso3:
                continue
            audio = row.get("audio")
            if not isinstance(audio, dict):
                continue
            arr, sr = _decode_audio_field(audio)
            pcm, pcm_sr = _float_to_pcm16(arr, sr)
            yield {
                "pcm": pcm,
                "sample_rate": pcm_sr,
                "reference": row.get(a_key) or row.get("src_a"),
                "question": row.get(q_key) or row.get("src_q"),
            }
            seen += 1
            if seen >= limit:
                return


def load_samples(dataset: str, lang: str, limit: int, max_audio_seconds: float) -> Iterator[dict[str, Any]]:
    if dataset == "fleurs":
        return load_fleurs(lang, limit)
    if dataset == "ccfqa":
        return load_ccfqa(lang, limit)
    if dataset == "belebele":
        return load_belebele(lang, limit, max_audio_seconds)
    raise ValueError(f"unknown dataset {dataset}")
