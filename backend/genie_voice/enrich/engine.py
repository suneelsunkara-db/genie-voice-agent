"""Enrichment engine entry point.

The Databricks Foundation Model API is the SOLE engine for conversation insights
(see enrich.fm). There is NO heuristic/rules fallback: when the FM is unavailable
(offline, missing auth, or an endpoint outage) the insight is returned with the
contract keys set to null plus `"available": False`, so the agent UI degrades
gracefully (shows "—") instead of faking an answer or 500-ing.

Both surfaces use this module:
    live agent-assist -> enrich_utterance (per utterance)
    call-level gold   -> summarize_call   (per call)
The gold refresh task uses set-based inference with the `ai_query` SQL function directly.
"""
from __future__ import annotations

import logging
from typing import Any

from genie_voice.config import Settings, get_settings

log = logging.getLogger(__name__)

# Contract keys, so an "unavailable" result still matches what callers expect.
_CALL_KEYS = [
    "primary_intent", "all_intents", "sentiment_score", "sentiment_label",
    "disposition", "resolution_status", "next_best_action",
    "mentioned_invoice_id", "mentioned_amount", "summary",
]
_UTTERANCE_KEYS = [
    "primary_intent", "all_intents", "sentiment_score", "sentiment_label",
    "next_best_action", "mentioned_invoice_id", "mentioned_amount",
]


def _unavailable(keys: list[str], **extra: Any) -> dict[str, Any]:
    out: dict[str, Any] = {k: ([] if k == "all_intents" else None) for k in keys}
    out["available"] = False
    out.update(extra)
    return out


def summarize_call(
    utterances: list[dict[str, Any]], settings: Settings | None = None
) -> dict[str, Any]:
    settings = settings or get_settings()
    try:
        from genie_voice.enrich.fm import fm_summarize_call

        return {**fm_summarize_call(utterances, settings), "available": True}
    except Exception as exc:  # noqa: BLE001 - never break the caller; report unavailable
        log.warning("FM summarize_call unavailable (%s)", exc)
        return _unavailable(_CALL_KEYS)


def enrich_utterance(
    text: str,
    settings: Settings | None = None,
    *,
    speaker: int | None = None,
    issue_status: str = "open",
) -> dict[str, Any]:
    settings = settings or get_settings()
    try:
        if speaker == 1:
            from genie_voice.enrich.fm import fm_enrich_customer_utterance

            return {
                **fm_enrich_customer_utterance(text, issue_status, settings),
                "available": True,
                "speaker": speaker,
            }
        from genie_voice.enrich.fm import fm_enrich_utterance as fm_enrich

        return {**fm_enrich(text, settings, speaker=speaker), "available": True}
    except Exception as exc:  # noqa: BLE001
        log.warning("FM enrich_utterance unavailable (%s)", exc)
        keys = _UTTERANCE_KEYS if speaker != 1 else [
            *_UTTERANCE_KEYS,
            "customer_signal",
            "payment_plan_requested",
            "waiver_requested",
        ]
        return _unavailable(keys, speaker=speaker)
