"""Foundation Model-backed enrichment (Databricks Foundation Model API).

Produces the conversation-insight contract for both surfaces:
  - live agent-assist  -> `fm_enrich_utterance` (per utterance, low latency) via
    the Databricks SDK `serving_endpoints.query` (recommended live path).
  - call-level rollup   -> `fm_summarize_call` (per call) via the same endpoint.

The gold refresh task runs set-based inference with the `ai_query` SQL
function + structured outputs. This module is the request/response client used by
the API/offline live runner.

Endpoint + params come from `settings.enrichment`. Output is validated/coerced to
the contract; callers wrap these in try/except and return an unavailable insight
instead of faking a heuristic answer.
"""
from __future__ import annotations

import json
import re
from typing import Any

from genie_voice.config import Settings, get_settings

# The conversation-derived contract. These enums steer the model and are the
# single source of the allowed intent / next-best-action vocabulary.
INTENTS = [
    "billing_dispute", "late_fee", "payment_arrangement", "refund",
    "autopay_issue", "plan_inquiry", "cancellation_risk", "billing_inquiry",
]
NEXT_BEST_ACTIONS = [
    "escalate_retention_offer", "offer_fee_waiver", "process_refund",
    "set_up_payment_plan", "offer_plan_upgrade", "continue",
]
SENTIMENT_LABELS = ["negative", "neutral", "positive"]
DISPOSITIONS = ["resolved", "follow_up", "escalated"]
RESOLUTIONS = ["resolved", "open"]

# Full call-level record (what gold_call_insights stores, minus call_id/customer_id).
CALL_PROPERTIES: dict[str, Any] = {
    "primary_intent": {"type": ["string", "null"], "enum": INTENTS + [None]},
    "all_intents": {"type": "array", "items": {"type": "string", "enum": INTENTS}},
    "sentiment_score": {"type": "number"},
    "sentiment_label": {"type": "string", "enum": SENTIMENT_LABELS},
    "disposition": {"type": "string", "enum": DISPOSITIONS},
    "resolution_status": {"type": "string", "enum": RESOLUTIONS},
    "next_best_action": {"type": "string", "enum": NEXT_BEST_ACTIONS},
    "mentioned_invoice_id": {"type": ["string", "null"]},
    "mentioned_amount": {"type": ["number", "null"]},
    "summary": {"type": "string"},
}
# Per-utterance (live) record: same vocabulary, no call-level disposition/summary.
UTTERANCE_KEYS = [
    "primary_intent", "all_intents", "sentiment_score", "sentiment_label",
    "next_best_action", "mentioned_invoice_id", "mentioned_amount",
]
CUSTOMER_SIGNALS = ["request_help", "confirm_proceed", "decline", "escalate", "neutral"]
CUSTOMER_UTTERANCE_KEYS = [
    *UTTERANCE_KEYS,
    "customer_signal",
    "payment_plan_requested",
    "waiver_requested",
]

SYSTEM_PROMPT = (
    "You are a contact-center conversation analytics engine. You read a transcript "
    "between a support AGENT and a CUSTOMER and extract structured insights for "
    "billing/account operations. Sentiment must reflect the CUSTOMER only, in "
    "[-1, 1]. Detect intents and entities (an invoice id looks like INV-90231; a "
    "dollar amount like $312.00). Respond with ONLY a single compact JSON object "
    "that matches the requested fields - no prose, no code fences."
)
_SYSTEM = SYSTEM_PROMPT

# Reused by BOTH the live serving call and the batch ai_query prompt so the two
# surfaces extract identically.
CALL_INSTRUCTION = (
    "Return JSON with EXACTLY these keys: "
    f"primary_intent (one of {INTENTS} or null), all_intents (subset of those), "
    f"sentiment_score (number -1..1), sentiment_label (one of {SENTIMENT_LABELS}), "
    f"disposition (one of {DISPOSITIONS}), resolution_status (one of {RESOLUTIONS}), "
    f"next_best_action (one of {NEXT_BEST_ACTIONS}), mentioned_invoice_id (string or null), "
    "mentioned_amount (number or null), summary (one concise sentence)."
)
UTTERANCE_INSTRUCTION = (
    "Return JSON for THIS SINGLE utterance with EXACTLY these keys: "
    f"primary_intent (one of {INTENTS} or null), all_intents (subset), "
    f"sentiment_score (number -1..1), sentiment_label (one of {SENTIMENT_LABELS}), "
    f"next_best_action (one of {NEXT_BEST_ACTIONS}), mentioned_invoice_id (string or null), "
    "mentioned_amount (number or null)."
)
CUSTOMER_UTTERANCE_INSTRUCTION = (
    "Return JSON for THIS customer utterance with EXACTLY these keys: "
    f"primary_intent (one of {INTENTS} or null), all_intents (subset), "
    f"sentiment_score (number -1..1), sentiment_label (one of {SENTIMENT_LABELS}), "
    f"next_best_action (one of {NEXT_BEST_ACTIONS}), mentioned_invoice_id (string or null), "
    "mentioned_amount (number or null), "
    f"customer_signal (one of {CUSTOMER_SIGNALS}), "
    "payment_plan_requested (boolean), waiver_requested (boolean). "
    "confirm_proceed means the customer agrees to proceed with a previously offered action."
)
RESOLUTION_SIGNAL_INSTRUCTION = (
    "Return JSON for THIS customer utterance with EXACTLY these keys: "
    "customer_signal (one of request_help, confirm_proceed, decline, escalate, neutral), "
    "payment_plan_requested (boolean), waiver_requested (boolean). "
    "confirm_proceed = customer agrees to proceed with a previously offered action."
)


def call_json_schema() -> dict[str, Any]:
    """OpenAI/ai_query-style json_schema for the call-level record."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "call_insights",
            "schema": {
                "type": "object",
                "properties": CALL_PROPERTIES,
                "required": list(CALL_PROPERTIES.keys()),
            },
            "strict": True,
        },
    }


# --------------------------------------------------------------------------- #
def _chat(settings: Settings, user: str, *, expect: str) -> dict[str, Any]:
    """One chat round-trip to the serving endpoint; returns parsed JSON.

    `expect` describes the fields to return, appended to the prompt so the model
    emits exactly the contract keys.
    """
    from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

    from genie_voice.databricks.client import get_workspace_client

    client = get_workspace_client(settings)
    kwargs: dict[str, Any] = {
        "name": settings.enrichment.model_endpoint,
        "messages": [
            ChatMessage(role=ChatMessageRole.SYSTEM, content=_SYSTEM),
            ChatMessage(role=ChatMessageRole.USER, content=f"{expect}\n\nTRANSCRIPT:\n{user}"),
        ],
        "max_tokens": settings.enrichment.max_tokens,
    }
    # Some reasoning models (e.g. Claude Opus 4.x) REJECT `temperature` with a
    # BadRequest. Only send it when configured, and transparently retry without
    # it if the endpoint rejects it - so the live FM path doesn't silently fall
    # back to the heuristic on an unsupported-parameter error.
    if settings.enrichment.temperature is not None:
        kwargs["temperature"] = settings.enrichment.temperature
    try:
        resp = client.serving_endpoints.query(**kwargs)
    except Exception as exc:  # noqa: BLE001
        if "temperature" in kwargs and "temperature" in str(exc).lower():
            kwargs.pop("temperature")
            resp = client.serving_endpoints.query(**kwargs)
        else:
            raise
    content = resp.choices[0].message.content
    return _parse_json(content)


def _parse_json(content: str) -> dict[str, Any]:
    try:
        return json.loads(content)
    except (json.JSONDecodeError, TypeError):
        # Salvage the first {...} block if the model wrapped it in prose/fences.
        match = re.search(r"\{.*\}", content or "", re.DOTALL)
        if not match:
            raise ValueError(f"model returned non-JSON: {content!r}")
        return json.loads(match.group(0))


def _coerce(data: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    """Keep only contract keys, normalize types so downstream casts are safe."""
    out: dict[str, Any] = {k: data.get(k) for k in keys}
    if "all_intents" in out:
        ai = out.get("all_intents") or []
        out["all_intents"] = sorted({str(i) for i in ai if i in INTENTS})
    for num in ("sentiment_score", "mentioned_amount"):
        if num in out and out[num] is not None:
            try:
                out[num] = round(float(out[num]), 3 if num == "sentiment_score" else 2)
            except (TypeError, ValueError):
                out[num] = None
    if "sentiment_score" in out and out.get("sentiment_score") is not None:
        out["sentiment_score"] = max(-1.0, min(1.0, out["sentiment_score"]))
    return out


def _coerce_customer_signal(data: dict[str, Any]) -> dict[str, Any]:
    out = _coerce(data, CUSTOMER_UTTERANCE_KEYS)
    signal = str(out.get("customer_signal") or "neutral")
    if signal not in CUSTOMER_SIGNALS:
        signal = "neutral"
    out["customer_signal"] = signal
    out["payment_plan_requested"] = bool(out.get("payment_plan_requested"))
    out["waiver_requested"] = bool(out.get("waiver_requested"))
    return out


# --------------------------------------------------------------------------- #
def fm_summarize_call(utterances: list[dict[str, Any]], settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or get_settings()
    transcript = "\n".join(
        f"{(u.get('speaker_role') or u.get('speaker') or 'speaker')}: {u.get('text', '')}"
        for u in utterances
    )
    data = _chat(settings, transcript, expect=CALL_INSTRUCTION)
    return _coerce(data, list(CALL_PROPERTIES.keys()))


def fm_enrich_utterance(
    text: str, settings: Settings | None = None, *, speaker: int | None = None
) -> dict[str, Any]:
    settings = settings or get_settings()
    data = _chat(settings, text, expect=UTTERANCE_INSTRUCTION)
    return {**_coerce(data, UTTERANCE_KEYS), "speaker": speaker}


def fm_enrich_customer_utterance(
    text: str,
    issue_status: str,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Single FM call: utterance enrichment + resolution transition signals."""
    settings = settings or get_settings()
    prompt = (
        f"{CUSTOMER_UTTERANCE_INSTRUCTION}\n"
        f"CURRENT_ISSUE_STATUS: {issue_status}\n"
        f"CUSTOMER_UTTERANCE:\n{text}"
    )
    data = _chat(settings, prompt, expect=CUSTOMER_UTTERANCE_INSTRUCTION)
    return _coerce_customer_signal(data)


def fm_customer_resolution_signal(
    text: str,
    issue_status: str,
    settings: Settings | None = None,
) -> dict[str, Any]:
    settings = settings or get_settings()
    prompt = (
        f"{RESOLUTION_SIGNAL_INSTRUCTION}\n"
        f"CURRENT_ISSUE_STATUS: {issue_status}\n"
        f"CUSTOMER_UTTERANCE:\n{text}"
    )
    data = _chat(settings, prompt, expect=RESOLUTION_SIGNAL_INSTRUCTION)
    signal = str(data.get("customer_signal") or "neutral")
    allowed = {"request_help", "confirm_proceed", "decline", "escalate", "neutral"}
    if signal not in allowed:
        signal = "neutral"
    return {
        "customer_signal": signal,
        "payment_plan_requested": bool(data.get("payment_plan_requested")),
        "waiver_requested": bool(data.get("waiver_requested")),
    }
