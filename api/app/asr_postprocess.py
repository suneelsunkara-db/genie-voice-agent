"""Runtime ASR transcript post-processing helpers."""
from __future__ import annotations

from typing import Any

from genie_voice.asr_eval.postprocess import normalize_invoice_ids

from .deps import serving


def postprocess_transcript_for_call(
    call_id: str,
    transcript: str,
    settings: Any,
) -> tuple[str, dict[str, Any]]:
    """Apply configured ASR post-processing using active call account context."""
    options = settings.providers.stt.active_options()
    if not options.get("postprocess_invoice_ids"):
        return transcript, {"enabled": False, "invoice_id_corrections": []}

    candidate_invoice_ids = _candidate_invoice_ids(call_id)
    processed, corrections = normalize_invoice_ids(transcript, candidate_invoice_ids)
    return processed, {
        "enabled": True,
        "invoice_id_corrections": [correction.to_dict() for correction in corrections],
    }


def _candidate_invoice_ids(call_id: str) -> list[str]:
    try:
        facts = serving().get_call_account_facts(call_id)
    except Exception:  # noqa: BLE001 - ASR should not fail because account context is absent.
        return []

    invoices = facts.get("invoices") or []
    ids: list[str] = []
    for invoice in invoices:
        invoice_id = str((invoice or {}).get("invoice_id") or "").strip()
        if invoice_id:
            ids.append(invoice_id)
    return ids
