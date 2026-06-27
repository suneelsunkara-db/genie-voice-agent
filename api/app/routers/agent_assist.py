"""Agent-assist endpoints - live call state served from Lakebase.

GET /assist READS pre-computed nudges from the low-latency serving store (the
enricher writes them per utterance) - the UI polls this cheaply without ever
triggering inference. POST /assist is the on-demand path: enrich ONE live
utterance now with the Foundation Model and merge the nudge into call state, so a
streaming transcript can push true real-time insights.
"""
from __future__ import annotations

import base64
import copy
import json
import logging
import time
from urllib.request import Request, urlopen

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from genie_voice.assist.billing import prepare_billing_adjustment
from genie_voice.assist.genie_facts import (
    fetch_validated_account_metrics,
    genie_account_insight,
)
from genie_voice.assist.resolution import (
    evaluate_resolution,
    finalize_resolution_after_billing,
    resolution_event_for_transition,
)
from genie_voice.assist.validation import validate_reply_against_metrics
from genie_voice.config import get_settings
from genie_voice.databricks.client import get_workspace_client
from genie_voice.serve.lakebase import (
    _apply_billing_adjustments,
    _apply_resolution_status_overlay,
)

from ..asr_postprocess import postprocess_transcript_for_call
from ..deps import genie, serving

router = APIRouter(prefix="/calls", tags=["agent-assist"])
log = logging.getLogger(__name__)


class UtteranceIn(BaseModel):
    text: str
    speaker: int | None = None


class MicAudioIn(BaseModel):
    audio_b64: str
    mime_type: str = "audio/webm"
    speaker: int = 1



def _transcribe_with_deepgram(audio_bytes: bytes, mime_type: str, settings) -> str:
    key = settings.secrets.deepgram_api_key.strip()
    if not key:
        raise HTTPException(status_code=400, detail="DEEPGRAM_API_KEY is not configured")
    req = Request(
        "https://api.deepgram.com/v1/listen?model=nova-3&smart_format=true&punctuate=true",
        data=audio_bytes,
        method="POST",
    )
    req.add_header("Authorization", f"Token {key}")
    req.add_header("Content-Type", mime_type or "audio/webm")
    req.add_header("Accept", "application/json")
    try:
        with urlopen(req, timeout=45) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Deepgram call failed: {exc}") from exc

    channels = body.get("results", {}).get("channels") or []
    alternatives = channels[0].get("alternatives") if channels else []
    transcript = (alternatives[0].get("transcript") if alternatives else "") or ""
    transcript = transcript.strip()
    if not transcript:
        raise HTTPException(status_code=422, detail="No transcript returned from Deepgram")
    return transcript


def _prediction_response_dict(response) -> dict:
    if isinstance(response, dict):
        return response
    if hasattr(response, "as_dict"):
        return response.as_dict()
    predictions = getattr(response, "predictions", None)
    if predictions is not None:
        return {"predictions": predictions}
    return {}


def _transcribe_with_databricks_model(body: MicAudioIn, settings) -> str:
    options = settings.providers.stt.active_options()
    endpoint = str(options.get("endpoint") or "").strip()
    if not endpoint:
        raise HTTPException(status_code=400, detail="Databricks STT endpoint is not configured")

    client = get_workspace_client(settings)
    try:
        response = client.serving_endpoints.query(
            name=endpoint,
            dataframe_records=[
                {
                    "audio_b64": body.audio_b64,
                    "mime_type": body.mime_type or "audio/webm",
                    "speaker": body.speaker,
                }
            ],
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Databricks STT call failed: {exc}") from exc

    payload = _prediction_response_dict(response)
    predictions = payload.get("predictions") or []
    first = predictions[0] if predictions else {}
    transcript = str(first.get("raw_transcript") or first.get("transcript") or "").strip()
    if not transcript:
        raise HTTPException(status_code=422, detail="No transcript returned from Databricks STT")
    return transcript


def _transcribe_mic_audio(body: MicAudioIn, audio_bytes: bytes, settings) -> str:
    provider = settings.providers.stt.active
    if provider == "deepgram":
        return _transcribe_with_deepgram(audio_bytes, body.mime_type, settings)
    if provider == "databricks":
        return _transcribe_with_databricks_model(body, settings)
    raise HTTPException(status_code=400, detail=f"Unsupported STT provider for mic transcription: {provider}")



_BAD_AGENT_REPLY_MARKERS = (
    "customer_id =",
    "overdue_invoice_count",
    "declined_payment_count",
    "`customers`",
    "`invoices`",
    "please specify what information",
    "outside my scope",
    "i do not generate customer-facing",
    "i can provide data",
    "i cannot provide",
    "no resolvable genie space",
    "authenticate to databricks",
    "unrelated to the database",
    "database schema",
    "provided tables",
    "only able to answer questions about the data",
    "data-related question",
)


def _looks_like_bad_agent_reply(text: str) -> bool:
    t = text.lower()
    if any(m in t for m in _BAD_AGENT_REPLY_MARKERS):
        return True
    if "`" in text and "=" in text:
        return True
    return False


def _pipeline_step(key: str, label: str, started: float, *, detail: str = "") -> dict[str, object]:
    elapsed = round((time.perf_counter() - started) * 1000)
    return {
        "key": key,
        "label": label,
        "status": "done",
        "elapsed_ms": elapsed,
        **({"detail": detail} if detail else {}),
    }


def _account_context_snippet(
    account: dict[str, object] | None,
    metrics_validation: dict[str, object] | None = None,
) -> str:
    if not account or not account.get("found"):
        return "No account facts available."
    cust = account.get("customer") or {}
    summary = account.get("summary") or {}
    customer_id = account.get("customer_id") or cust.get("customer_id")
    lines = [
        f"customer_id: {customer_id}",
        f"name: {cust.get('full_name')}",
        f"status: {summary.get('status')}",
        f"overdue_invoice_count: {summary.get('overdue_invoice_count')}",
        f"overdue_amount_usd: {summary.get('overdue_amount')}",
        f"recent_declined_payments: {summary.get('recent_declined_payments')}",
        f"autopay_enabled: {summary.get('autopay_enabled')}",
        f"issue_status: {summary.get('issue_status')}",
    ]
    if summary.get("resolution_note"):
        lines.append(f"resolution_note: {summary.get('resolution_note')}")
    if metrics_validation:
        lines.append(f"genie_facts_validated: {metrics_validation.get('genie_validated')}")
        mismatches = metrics_validation.get("mismatches") or []
        if mismatches:
            lines.append(f"genie_lakebase_mismatches: {', '.join(str(m) for m in mismatches)}")
    for inv in account.get("invoices") or []:
        if str(inv.get("status")) in ("overdue", "resolved"):
            lines.append(
                "primary_invoice: "
                f"{inv.get('invoice_id')} status={inv.get('status')} "
                f"amount=${inv.get('amount')} late_fee=${inv.get('late_fee')}"
            )
            break
    return "\n".join(str(x) for x in lines)


def _compose_agent_reply(
    call_id: str,
    customer_id: str | None,
    customer_message: str,
    resolution: dict,
    account: dict[str, object] | None,
    genie_insight: str | None = None,
) -> tuple[str | None, dict[str, object]]:
    """Phrase a customer-facing reply with the FM, grounded in validated Lakebase
    facts and (when available) a pre-fetched Genie account insight.

    `genie_insight` is a short NL summary fetched OFF the critical path and cached
    in call state. When present, the reply legitimately opens with "Based on Genie
    insights"; when absent, it uses a neutral account-grounded opener so the claim
    is never false."""
    status = str(resolution.get("status") or "open")
    issue_closed = status == "closed"
    opener = "Based on Genie insights, " if genie_insight else "Based on your account, "

    metrics_result = fetch_validated_account_metrics(
        genie(),
        account,
        customer_id,
        skip_genie_query=issue_closed or bool(account and account.get("found")),
    )
    metrics = metrics_result.authoritative
    genie_skipped = issue_closed or metrics_result.genie_error in (
        "genie_metrics_skipped_post_close",
        "lakebase_authoritative",
    )
    validation_meta: dict[str, object] = {
        "genie_validated": metrics_result.genie_validated,
        "mismatches": metrics_result.mismatches,
        "genie_error": metrics_result.genie_error,
        "genie_skipped": genie_skipped,
        "authoritative_metrics": {
            "overdue_invoice_count": metrics.overdue_invoice_count,
            "overdue_amount": metrics.overdue_amount,
            "recent_declined_payments": metrics.recent_declined_payments,
        },
        "output_validated": False,
        "output_issues": [],
        "reply_available": False,
    }

    context = _account_context_snippet(account, validation_meta)
    cust_label = customer_id or "this customer"
    metrics_block = "\n".join(metrics.as_context_lines())
    genie_block = (
        f"GENIE ACCOUNT INSIGHT (pre-fetched, authoritative narrative):\n{genie_insight}\n\n"
        if genie_insight
        else ""
    )
    validation_meta["genie_insight_used"] = bool(genie_insight)

    if issue_closed:
        genie_question = f"""You are a customer-facing billing support agent on call {call_id}.
The customer ({cust_label}) just confirmed: "{customer_message}"

{genie_block}VALIDATED ACCOUNT FACTS (Lakebase authoritative; Genie cross-checked):
{metrics_block}

{context}

The issue is CLOSED. Payment plan and/or waiver have been applied per resolution_note.
Write a warm 3-4 sentence reply that:
1) thanks them for confirming,
2) states what was done (waiver and/or payment plan),
3) confirms the issue is closed and when they'll see the update,
4) offers brief help if needed.

Use plain spoken English. Start with: "{opener}..."
Do NOT mention SQL, field names, backticks, or ask them to proceed again.
Do NOT cite overdue balances that contradict the validated facts above."""
    else:
        genie_question = f"""You are a customer-facing billing support agent on call {call_id}.
Customer ({cust_label}) said: "{customer_message}"

{genie_block}VALIDATED ACCOUNT FACTS for THIS customer only (use ONLY these numbers):
{metrics_block}

{context}

Write a 3-4 sentence reply:
1) empathy,
2) what the account shows in plain language (use the validated facts exactly),
3) one concrete action you can take now,
4) a confirmation question.

Start with: "{opener}..."
Do NOT use SQL, table/column names, backticks, or portfolio-wide aggregates."""

    try:
        t0 = time.perf_counter()
        from genie_voice.enrich.fm import fm_compose_agent_reply
        from genie_voice.config import get_settings

        raw = fm_compose_agent_reply(genie_question, settings=get_settings()).strip()
        log.info(
            "assist fm prose call_id=%s elapsed_ms=%.0f closed=%s",
            call_id,
            (time.perf_counter() - t0) * 1000,
            issue_closed,
        )
        if raw and not _looks_like_bad_agent_reply(raw):
            ok, issues = validate_reply_against_metrics(
                raw,
                metrics,
                issue_closed=issue_closed,
            )
            validation_meta["output_validated"] = ok
            validation_meta["output_issues"] = issues
            if ok:
                validation_meta["reply_available"] = True
                return raw, validation_meta
            validation_meta["genie_error"] = "reply_failed_metric_validation"
    except Exception as exc:  # noqa: BLE001
        validation_meta["genie_error"] = str(exc)

    return None, validation_meta


@router.get("")
def list_calls() -> dict:
    return {"calls": serving().list_call_states()}


@router.get("/{call_id}/assist")
def get_assist(call_id: str) -> dict:
    state = serving().get_call_state(call_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"No state for call {call_id}")
    return state


@router.post("/{call_id}/assist")
def post_assist(call_id: str, body: UtteranceIn) -> dict:
    """Enrich a single live utterance on demand and persist it as the call's
    current nudge. Uses the Foundation Model engine; if the FM is unavailable the
    nudge is returned with `available=False` (no heuristic fallback) so the agent
    UI degrades gracefully instead of faking an answer."""
    from genie_voice.enrich.engine import enrich_utterance

    s = get_settings()
    t_start = time.perf_counter()
    pipeline_steps: list[dict[str, object]] = []
    existing = serving().get_call_state(call_id) or {}
    inner = existing.get("state") or {}
    previous_resolution = dict(inner.get("resolution") or {})
    current_status = str(previous_resolution.get("status") or "open")

    t_fm = time.perf_counter()
    nudge = enrich_utterance(
        body.text, s, speaker=body.speaker, issue_status=current_status
    )
    fm_detail = "unavailable" if not nudge.get("available", True) else str(nudge.get("primary_intent") or "enriched")
    pipeline_steps.append(_pipeline_step("fm_enrich", "Foundation model enrichment", t_fm, detail=fm_detail))
    log.info("assist fm call_id=%s elapsed_ms=%.0f", call_id, (time.perf_counter() - t_fm) * 1000)

    inner["live"] = nudge
    utterances = list(inner.get("utterances") or [])
    utterances.append({"text": body.text, "speaker": body.speaker})
    inner["utterances"] = utterances

    customer_id = existing.get("customer_id")
    t_resolution = time.perf_counter()
    account_source = serving().load_account_facts_source(customer_id) if customer_id else None
    resolution = evaluate_resolution(
        inner, body.text, body.speaker, account_source, nudge, s
    )
    pipeline_steps.append(
        _pipeline_step(
            "resolution",
            "Resolution evaluation",
            t_resolution,
            detail=str(resolution.get("status") or "open"),
        )
    )

    pending_adjustment: dict[str, object] | None = None
    actions = dict(resolution.get("actions") or {})
    if actions.get("pending_close") and customer_id and account_source:
        prepared = prepare_billing_adjustment(call_id, customer_id, resolution, account_source)
        if prepared.get("ok"):
            pending_adjustment = prepared["adjustment"]

    agent_reply: str | None = None
    agent_validation: dict[str, object] | None = None
    if body.speaker == 1:
        reply_account = serving().get_call_account_facts(call_id)
        reply_resolution = resolution
        if pending_adjustment and account_source:
            reply_resolution = finalize_resolution_after_billing(
                copy.deepcopy(resolution),
                {"applied": True, "adjustment": pending_adjustment},
            )
            reply_account = _apply_resolution_status_overlay(
                _apply_billing_adjustments(copy.deepcopy(account_source), [pending_adjustment]),
                reply_resolution,
            )
        cached_insight = (inner.get("genie_insight") or {}).get("text")
        t_genie = time.perf_counter()
        agent_reply, agent_validation = _compose_agent_reply(
            call_id,
            customer_id,
            body.text,
            reply_resolution,
            reply_account,
            genie_insight=cached_insight,
        )
        genie_detail = "reply ready" if agent_reply else str((agent_validation or {}).get("genie_error") or "unavailable")
        if agent_validation:
            if agent_validation.get("reply_available"):
                genie_detail = "reply validated"
            elif agent_validation.get("mismatches"):
                genie_detail = f"mismatch: {', '.join(str(m) for m in agent_validation.get('mismatches') or [])}"
        pipeline_steps.append(
            _pipeline_step("genie_reply", "Genie agent reply", t_genie, detail=genie_detail)
        )
        log.info(
            "assist agent_reply call_id=%s elapsed_ms=%.0f reply=%s",
            call_id,
            (time.perf_counter() - t_genie) * 1000,
            bool(agent_reply),
        )
        if agent_reply:
            utterances.append({"text": agent_reply, "speaker": 0})
            inner["utterances"] = utterances

    billing_result: dict[str, object] | None = None
    if actions.get("pending_close") and customer_id and account_source:
        t_billing = time.perf_counter()
        if not pending_adjustment:
            billing_result = {"applied": False, "reason": "no_overdue_invoice"}
        elif body.speaker == 1 and not agent_reply:
            billing_result = {"applied": False, "reason": "agent_reply_unavailable"}
        else:
            billing_result = serving().apply_billing_resolution(
                call_id,
                customer_id,
                resolution,
                account_source,
                adjustment=pending_adjustment,
            )
        resolution = finalize_resolution_after_billing(resolution, billing_result)
        billing_detail = (
            "applied"
            if billing_result and billing_result.get("applied")
            else str(billing_result.get("reason") if billing_result else "skipped")
        )
        pipeline_steps.append(
            _pipeline_step("billing", "Billing adjustment", t_billing, detail=billing_detail)
        )

    inner["resolution"] = resolution
    t_persist = time.perf_counter()
    serving().upsert_call_state(call_id, existing.get("customer_id"), inner)

    transition = resolution_event_for_transition(previous_resolution, resolution)
    if transition:
        serving().append_resolution_event(
            call_id=call_id,
            event_type=transition["event_type"],
            issue_status=transition["issue_status"],
            note=transition.get("note"),
            actions=transition.get("actions") or {},
        )

    pipeline_steps.append(_pipeline_step("persist", "Lakebase state update", t_persist))

    log.info(
        "assist complete call_id=%s total_ms=%.0f status=%s",
        call_id,
        (time.perf_counter() - t_start) * 1000,
        resolution.get("status"),
    )

    return {
        "call_id": call_id,
        "model": s.enrichment.model_endpoint,
        "live": nudge,
        "resolution": resolution,
        "agent_reply": agent_reply,
        "agent_validation": agent_validation,
        "billing": billing_result,
        "close_block_reason": (resolution.get("actions") or {}).get("close_block_reason"),
        "pipeline_steps": pipeline_steps,
        "total_elapsed_ms": round((time.perf_counter() - t_start) * 1000),
    }


@router.post("/{call_id}/mic-transcribe")
def post_mic_transcribe(call_id: str, body: MicAudioIn) -> dict:
    """Transcribe browser mic audio with Deepgram and route into the same assist flow."""
    s = get_settings()
    try:
        audio_bytes = base64.b64decode(body.audio_b64)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid audio payload: {exc}") from exc

    raw_text = _transcribe_mic_audio(body, audio_bytes, s)
    text, postprocessing = postprocess_transcript_for_call(call_id, raw_text, s)

    # Reuse existing assist behavior so the app workflow remains unchanged.
    nudge = post_assist(call_id, UtteranceIn(text=text, speaker=body.speaker))
    return {
        **nudge,
        "transcript": text,
        "raw_transcript": raw_text,
        "asr_provider": s.providers.stt.active,
        "asr_postprocessing": postprocessing,
    }


@router.post("/{call_id}/genie-insight")
def post_genie_insight(call_id: str) -> dict:
    """Fetch a Genie NL account insight and cache it in call state.

    Call this OFF the live reply path (e.g. when the agent opens the call) so the
    per-utterance reply can ground on Genie without paying Genie's latency inline.
    """
    from datetime import UTC, datetime

    state = serving().get_call_state(call_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"No state for call {call_id}")
    customer_id = state.get("customer_id")
    inner = state.get("state") or {}
    text = genie_account_insight(genie(), customer_id)
    inner["genie_insight"] = (
        {"text": text, "fetched_at": datetime.now(UTC).isoformat()} if text else None
    )
    serving().upsert_call_state(call_id, customer_id, inner)
    return {"call_id": call_id, "genie_insight": inner.get("genie_insight")}


@router.get("/{call_id}/account")
def get_call_account(call_id: str) -> dict:
    """Account facts for the customer on this call (live transcript + account
    state side by side, the way an agent sees it)."""
    state = serving().get_call_state(call_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"No state for call {call_id}")
    return serving().get_call_account_facts(call_id)


@router.get("/{call_id}/resolution-events")
def get_resolution_events(call_id: str) -> dict:
    state = serving().get_call_state(call_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"No state for call {call_id}")
    return {"call_id": call_id, "events": serving().list_resolution_events(call_id)}


@router.get("/{call_id}/alignment")
def get_call_alignment(call_id: str) -> dict:
    """Check Lakebase resolution + billing vs account facts for a call."""
    from genie_voice.assist.alignment import alignment_report

    state = serving().get_call_state(call_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"No state for call {call_id}")
    customer_id = state.get("customer_id")
    resolution = (state.get("state") or {}).get("resolution") or {}
    facts = serving().get_call_account_facts(call_id)
    events = serving().list_resolution_events(call_id)
    adjustments = serving().list_billing_adjustments(
        customer_id or "", call_id=call_id, active_only=True
    )
    return alignment_report(
        call_id=call_id,
        customer_id=customer_id,
        resolution=resolution,
        lakebase_adjustments=adjustments,
        account_summary=facts.get("summary") or {},
        resolution_events=events,
    )


@router.post("/{call_id}/reset-demo-session")
def reset_demo_session(call_id: str) -> dict:
    state = serving().get_call_state(call_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"No state for call {call_id}")
    return serving().reset_demo_session(call_id)
