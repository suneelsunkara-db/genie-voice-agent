"""Two-phase data staging: extract benchmark samples to the UC Volume once, then
serve them cheaply to the parallel benchmark tasks.

Phase 1 (prepare, one long-running task) streams each (dataset, language) pair
from HuggingFace and writes a compact per-pair artifact to the Volume. This is
the only place the heavyweight parquet decode happens, and it runs one pair at a
time, so peak memory is bounded to a single pair.

Phase 2 (benchmark, many parallel tasks) reads its pair's small staged artifact
(~12 MB of PCM for 20 samples) — no HuggingFace, no parquet, no arrow — so each
task's memory stays tiny and the tasks can run concurrently without OOM.

Artifact format: one JSONL file per (dataset, language, limit) at
``<results_dir>/_staged/limit<N>/<dataset>/<lang>.jsonl``. Each line is one
sample: metadata (reference / correct_choice / question / sample_rate) plus the
16-bit PCM audio base64-encoded. The file is written whole (atomic overwrite),
so its presence means the pair is fully staged.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Iterator

from paths import benchmark_results_dir
from volume_io import read_text, volume_exists, write_text

# Metadata keys carried through from the loaders (everything except the PCM).
# ``context`` carries Belebele's question + options grounding for the LLM turn.
_META_KEYS = ("sample_rate", "reference", "correct_choice", "question", "context")

# Staged-artifact schema version. Bump whenever the loader output or staging
# format changes so artifacts from an older format are never silently reused:
# ``is_staged`` looks under the versioned path, so old artifacts simply aren't
# found and the pair is re-staged fresh. v2: Belebele now stages the FULL passage
# audio plus the question/options ``context`` (v1 staged an 18 s passage slice
# with no context, which scored ~0%).
_STAGING_VERSION = "v2"


def staged_file(dataset: str, lang: str, limit: int, *, out_dir: Path | None = None) -> Path:
    root = (out_dir or benchmark_results_dir()) / "_staged" / _STAGING_VERSION / f"limit{limit}"
    return root / dataset / f"{lang}.jsonl"


def is_staged(dataset: str, lang: str, limit: int, *, out_dir: Path | None = None) -> bool:
    return volume_exists(staged_file(dataset, lang, limit, out_dir=out_dir))


def stage_pair(
    dataset: str,
    lang: str,
    limit: int,
    max_audio_seconds: float,
    *,
    out_dir: Path | None = None,
    overwrite: bool = False,
) -> int:
    """Extract up to ``limit`` samples for one pair and write the staged JSONL.

    Returns the number of samples staged. Idempotent: if the artifact already
    exists and ``overwrite`` is False, it is left untouched and its sample count
    is returned.
    """
    from hf_datasets import load_samples

    target = staged_file(dataset, lang, limit, out_dir=out_dir)
    if not overwrite and volume_exists(target):
        return _count_lines(target)

    lines: list[str] = []
    for sample in load_samples(dataset, lang, limit, max_audio_seconds):
        record: dict[str, Any] = {k: sample[k] for k in _META_KEYS if k in sample}
        record["pcm_b64"] = base64.b64encode(sample["pcm"]).decode("ascii")
        lines.append(json.dumps(record, ensure_ascii=False))
        del sample

    write_text(target, "\n".join(lines))
    return len(lines)


def load_staged(
    dataset: str, lang: str, limit: int, *, out_dir: Path | None = None
) -> Iterator[dict[str, Any]]:
    """Yield staged samples for one pair in the same shape as ``load_samples``."""
    target = staged_file(dataset, lang, limit, out_dir=out_dir)
    if not volume_exists(target):
        raise FileNotFoundError(
            f"no staged data for {dataset}@{lang} (limit={limit}) at {target}; "
            "run the prepare phase first"
        )
    text = read_text(target)
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        pcm = base64.b64decode(record.pop("pcm_b64"))
        record["pcm"] = pcm
        yield record


def _count_lines(path: Path) -> int:
    text = read_text(path)
    return sum(1 for line in text.splitlines() if line.strip())
