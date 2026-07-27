"""Published leaderboard baselines for the FLEURS benchmark.

These are *reference* numbers copied from papers / public leaderboards for
FLEURS ASR. They are NOT re-measured by us and are NOT apples-to-apples:
subsets, decoding, normalization, and supported-language intersections differ.
The UI must label them as published references and show the citation. Every
value carries its source + a caveat.

Shape (flattened by ``reference_rows()`` into the same row schema the Delta
results use, tagged ``source="reference"``):

  dataset -> system_id -> {
     label, source, url, note,
     aggregate: {<metric>: value},                # dataset-level mean
     per_language: { <2-letter>: {<metric>: value} }
  }

Metric conventions match our own rows: FLEURS uses error rate (wer/cer, 0..1,
lower better).
"""
from __future__ import annotations

from typing import Any

# --- FLEURS: published ASR references (error rate, lower is better) ----------
#
# Aggregates intentionally preserve each source's published evaluation slice
# (e.g. FLEURS-54, FLEURS-77, or leaderboard average). They are references, not
# exact same-subset re-runs. Per-language rows are only included where we have a
# source with language-level numbers in the same metric convention.
_WHISPER_FLEURS = {
    "en": {"wer": 0.042},
    "es": {"wer": 0.030},
    "fr": {"wer": 0.047},
    "de": {"wer": 0.045},
    "it": {"wer": 0.040},
    "pt": {"wer": 0.039},
    "ru": {"wer": 0.088},
    "tr": {"wer": 0.096},
    "sv": {"wer": 0.070},
    "da": {"wer": 0.096},
    "zh": {"cer": 0.057},
    "ja": {"cer": 0.049},
    "ko": {"cer": 0.070},
}

REFERENCES: dict[str, dict[str, dict[str, Any]]] = {
    "fleurs": {
        "whisper-large-v3": {
            "label": "Whisper large-v3",
            "source": "OpenAI Whisper (Radford et al.) / FLEURS; Open ASR Leaderboard",
            "url": "https://huggingface.co/openai/whisper-large-v3",
            "note": (
                "Published FLEURS WER (CER for zh/ja). Batch ASR only — no realtime "
                "streaming or end-to-end latency; subset may differ from our sample."
            ),
            "aggregate": {"wer": 0.074},  # Open ASR Leaderboard mean
            "per_language": _WHISPER_FLEURS,
        },
        "whisper-large-v2": {
            "label": "Whisper large-v2",
            "source": "SeamlessM4T / Nature 2024, Table 4",
            "url": "https://www.nature.com/articles/s41586-024-08359-z/tables/4",
            "note": "Published FLEURS ASR WER over the 77-language overlap used by SeamlessM4T.",
            "aggregate": {"wer": 0.417},
            "per_language": {},
        },
        "mms-1b": {
            "label": "MMS 1B",
            "source": "Meta MMS (Pratap et al., JMLR 2024), Table 3",
            "url": "https://www.jmlr.org/papers/v25/23-1318.html",
            "note": "Published FLEURS-54 ASR WER with n-gram language models.",
            "aggregate": {"wer": 0.207},
            "per_language": {},
        },
        "mms-1b-lsah": {
            "label": "MMS 1B LSAH",
            "source": "Meta MMS (Pratap et al., JMLR 2024), Table 3",
            "url": "https://www.jmlr.org/papers/v25/23-1318.html",
            "note": "Published FLEURS-54 ASR WER with language-specific adapter heads and n-gram language models.",
            "aggregate": {"wer": 0.191},
            "per_language": {},
        },
        "seamlessm4t-medium": {
            "label": "SeamlessM4T Medium",
            "source": "SeamlessM4T / Nature 2024, Table 4",
            "url": "https://www.nature.com/articles/s41586-024-08359-z/tables/4",
            "note": "Published FLEURS ASR WER over the 77-language overlap used by SeamlessM4T.",
            "aggregate": {"wer": 0.219},
            "per_language": {},
        },
        "seamlessm4t-large": {
            "label": "SeamlessM4T Large",
            "source": "SeamlessM4T / Nature 2024, Table 4",
            "url": "https://www.nature.com/articles/s41586-024-08359-z/tables/4",
            "note": "Published FLEURS ASR WER over the 77-language overlap used by SeamlessM4T.",
            "aggregate": {"wer": 0.226},
            "per_language": {},
        },
        "seamlessm4t-large-v2": {
            "label": "SeamlessM4T Large v2",
            "source": "SeamlessM4T / Nature 2024, Table 4",
            "url": "https://www.nature.com/articles/s41586-024-08359-z/tables/4",
            "note": "Published FLEURS ASR WER over the 77-language overlap used by SeamlessM4T.",
            "aggregate": {"wer": 0.185},
            "per_language": {},
        },
    },
}

# Our own stack, for labelling the measured rows in the same vocabulary.
OUR_SYSTEM_ID = "genie-voice"
OUR_SYSTEM_LABEL = "Genie Voice (Qwen3-ASR + VoxCPM TTS)"


def _primary_metric(dataset: str, metrics: dict[str, Any]) -> tuple[str | None, float | None]:
    """Pick the headline metric/value for a reference entry, matching our rows."""
    if dataset == "fleurs":
        for key in ("cer", "wer"):
            if key in metrics:
                return key, float(metrics[key])
        return None, None
    return None, None


def reference_rows() -> list[dict[str, Any]]:
    """Flatten REFERENCES into result-row dicts tagged ``source="reference"``.

    Emits one aggregate row per (dataset, system) and one per-language row for
    each language a source reports, so the UI can overlay both a headline
    comparison and per-language bars where available.
    """
    rows: list[dict[str, Any]] = []
    for dataset, systems in REFERENCES.items():
        for system_id, spec in systems.items():
            base = {
                "system": system_id,
                "system_label": spec["label"],
                "source": "reference",
                "dataset": dataset,
                "evaluator": "asr",
                "reference_source": spec.get("source"),
                "reference_url": spec.get("url"),
                "note": spec.get("note"),
            }
            agg = spec.get("aggregate") or {}
            metric, value = _primary_metric(dataset, agg)
            if value is not None:
                rows.append({
                    **base,
                    "language": None,      # dataset-level aggregate
                    "scope": "aggregate",
                    "primary_metric": metric,
                    "primary_score": value,
                    "scores": dict(agg),
                })
            for lang, metrics in (spec.get("per_language") or {}).items():
                m, v = _primary_metric(dataset, metrics)
                if v is None:
                    continue
                rows.append({
                    **base,
                    "language": lang,
                    "scope": "language",
                    "primary_metric": m,
                    "primary_score": v,
                    "scores": dict(metrics),
                })
    return rows
