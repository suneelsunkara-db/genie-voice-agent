"""Published leaderboard baselines for the benchmarks we run ourselves.

These are *reference* numbers copied from the papers / public leaderboards for
the SAME datasets our realtime API is scored on (FLEURS, 2M-Belebele, CCFQA).
They are NOT re-measured by us and are NOT apples-to-apples: subsets, decoding,
ASR/LLM cascades and judges differ. The UI must label them as published
references and show the citation. Every value carries its source + a caveat.

Shape (flattened by ``reference_rows()`` into the same row schema the Delta
results use, tagged ``source="reference"``):

  dataset -> system_id -> {
     label, source, url, note,
     aggregate: {<metric>: value},                # dataset-level mean
     per_language: { <2-letter>: {<metric>: value} }
  }

Metric conventions match our own rows: FLEURS uses error rate (wer/cer, 0..1,
lower better); Belebele/CCFQA use accuracy (0..100, higher better).
"""
from __future__ import annotations

from typing import Any

# --- FLEURS: Whisper large-v3 ASR (error rate, lower is better) --------------
# Per-language FLEURS WER (CER for non-spaced scripts) as published third-party
# evaluations of Whisper large-v3 (OpenAI Whisper paper Appendix D / FLEURS).
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
    },
    # --- 2M-Belebele: cascaded speech comprehension (accuracy, higher better)
    # ACL 2025 Findings, Table 2 (zero-shot, averaged over 39 languages at the
    # Whisper/SeamlessM4T/2M-Belebele intersection). Directly comparable to our
    # cascaded STT->LLM MCQ setup.
    "belebele": {
        "seamlessm4t-llama3-70b": {
            "label": "SeamlessM4T + Llama-3 70B",
            "source": "2M-Belebele (Costa-jussà et al., ACL 2025 Findings), Table 2, zero-shot",
            "url": "https://aclanthology.org/2025.findings-acl.569/",
            "note": "Cascaded ASR->LLM, 39-language average; English = 95.5%.",
            "aggregate": {"acc": 84.8},
            "per_language": {"en": {"acc": 95.5}},
        },
        "whisper-llama3-70b": {
            "label": "Whisper + Llama-3 70B",
            "source": "2M-Belebele (Costa-jussà et al., ACL 2025 Findings), Table 2, zero-shot",
            "url": "https://aclanthology.org/2025.findings-acl.569/",
            "note": "Cascaded ASR->LLM, 39-language average; English = 95.7%.",
            "aggregate": {"acc": 79.1},
            "per_language": {"en": {"acc": 95.7}},
        },
    },
    # --- CCFQA: spoken question answering (LLM-judged accuracy, higher better)
    # AAAI 2026, Table 6 / Table 8 (SQA average LLM-based accuracy).
    "ccfqa": {
        "gpt-4o-mini-audio": {
            "label": "GPT-4o-mini-Audio",
            "source": "CCFQA (Du et al., AAAI 2026), Table 8 (SQA avg LLM-accuracy)",
            "url": "https://arxiv.org/abs/2508.07295",
            "note": "End-to-end audio LLM; LLM-judged accuracy averaged over 8 languages.",
            "aggregate": {"acc": 40.4},
            "per_language": {},
        },
        "qwen2.5-omni-7b": {
            "label": "Qwen2.5-Omni-7B",
            "source": "CCFQA (Du et al., AAAI 2026), Table 8 (SQA avg LLM-accuracy)",
            "url": "https://arxiv.org/abs/2508.07295",
            "note": "Open-source omni model; LLM-judged accuracy averaged over 8 languages.",
            "aggregate": {"acc": 33.2},
            "per_language": {},
        },
    },
}

# Our own stack, for labelling the measured rows in the same vocabulary.
OUR_SYSTEM_ID = "genie-voice"
OUR_SYSTEM_LABEL = "Genie Voice (Qwen3-ASR + Qwen3-Next)"


def _primary_metric(dataset: str, metrics: dict[str, Any]) -> tuple[str | None, float | None]:
    """Pick the headline metric/value for a reference entry, matching our rows."""
    if dataset == "fleurs":
        for key in ("cer", "wer"):
            if key in metrics:
                return key, float(metrics[key])
        return None, None
    if "acc" in metrics:
        return "acc", float(metrics["acc"])
    return None, None


def reference_rows() -> list[dict[str, Any]]:
    """Flatten REFERENCES into result-row dicts tagged ``source="reference"``.

    Emits one aggregate row per (dataset, system) and one per-language row for
    each language a source reports, so the UI can overlay both a headline
    comparison and per-language bars where available.
    """
    rows: list[dict[str, Any]] = []
    for dataset, systems in REFERENCES.items():
        evaluator = "asr" if dataset == "fleurs" else ("mcq" if dataset == "belebele" else "qa")
        for system_id, spec in systems.items():
            base = {
                "system": system_id,
                "system_label": spec["label"],
                "source": "reference",
                "dataset": dataset,
                "evaluator": evaluator,
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
