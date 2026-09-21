"""Audio in → STT → LLM (+ tools) → TTS."""
from __future__ import annotations

import asyncio
import json
import logging
import time
from functools import lru_cache
from typing import Any, AsyncIterator

from genie_voice.databricks.ai_gateway import GatewayPolicyDenied

from ..capabilities import SPEECH_LLM_TOOLASSIST_SPEECH
from ..guardrails import report
from ..guardrails.boundaries import (
    SensitiveInputDenied,
    admit_speech_output,
    admit_transcript,
)
from ..languages import base_code, english_name
from ..profiles import get_profile
from ..runtime.phrases import phrase as fixed_phrase
from ..runtime.refuse import ErrorCode, refuse_speech
from ..session import VoiceSession
from ..tracing import TurnTrace, submit_trace
from . import ServingBundle
from ._shared import language_mismatch, resolve_language, stream_tts, transcribe
from genie_voice.databricks.ai_gateway import (
    pop_inference_context,
    push_inference_context,
)

logger = logging.getLogger("realtime_voice")


def _public_agent_event(envelope: dict[str, Any]) -> dict[str, Any]:
    """Project an internal runtime event onto the browser-safe wire contract.

    Agent Mode deliberately works in canonical English. Its raw report belongs to
    the server-side rendering pipeline and must not reach the browser through the
    generic action result before localization. The original event is retained by
    ``_consume`` for evidence composition; only its public projection is redacted.
    """
    if envelope.get("kind") != "action.completed":
        return envelope
    payload = envelope.get("payload")
    if not isinstance(payload, dict) or payload.get("name") != "start_deep_dive":
        return envelope
    result = payload.get("result")
    if not isinstance(result, dict):
        return envelope

    public_result = dict(result)
    for internal_field in ("report", "canonical_question", "reasoning"):
        public_result.pop(internal_field, None)
    return {
        **envelope,
        "payload": {
            **payload,
            "result": public_result,
        },
    }


@lru_cache(maxsize=1)
def _llm_turn_timeout_s() -> float:
    """Config-sourced LLM per-turn budget (realtime_voice.timeouts.llm_turn_s).

    Covers cold-start + multi-tool-iteration loops; generous enough for the tool
    rounds but tight enough that a hung endpoint doesn't strand the caller. Cached
    (config is static per process); falls back to the dataclass default on error.
    """
    try:
        from ..config import RealtimeSettings

        return float(RealtimeSettings.resolve().llm_turn_timeout_s)
    except Exception:  # noqa: BLE001
        return 50.0


@lru_cache(maxsize=1)
def _agent_work_timeout_s() -> float:
    """Long-work budget for Agent Mode / deep investigations (not the 50s LLM cap)."""
    try:
        from ..config import RealtimeSettings

        return float(RealtimeSettings.resolve().deep_dive_read_timeout_s)
    except Exception:  # noqa: BLE001
        return 420.0


# A whole-turn budget must be strictly larger than the tool budget it supervises.
# The tool's own deadline returns a typed timeout carrying the conversation id; the
# LLM then has to render that into words and TTS has to speak them. When the two
# budgets are equal the outer one fires first, cancels the turn, and throws the
# graceful path away — the caller hears a hard failure instead of a sentence.
_TURN_BUDGET_MARGIN_S = 30.0


@lru_cache(maxsize=1)
def _navigation_min_confidence() -> float:
    """Config-sourced confidence gate for semantic navigation."""
    from ..config import RealtimeSettings

    return float(RealtimeSettings.resolve().navigation_min_confidence)


def _timeout_for_profile(profile_name: str | None) -> float:
    """Card / agent-mode packs use the long Agent Mode budget; others use LLM turn."""
    if profile_name == "card":
        return _agent_work_timeout_s()
    return _llm_turn_timeout_s()


_LONG_WORK_ADAPTERS = frozenset({"genie_one_mcp", "agent_mode"})


def _tool_work_timeout_s(adapter: str | None, profile_name: str | None) -> float:
    """Budget handed to the long-running tool on this route.

    Genie One and Agent Mode are long governed reads. Measured Genie One answers on
    the workspace lane land between two and three and a half minutes, so any cap
    below that converts a correct answer into a timeout without saving the caller a
    single second. Both share the long-work budget.
    """
    if adapter in _LONG_WORK_ADAPTERS:
        return _agent_work_timeout_s()
    return _timeout_for_profile(profile_name)


def _timeout_for_route(adapter: str | None, profile_name: str | None) -> float:
    """Whole-turn budget for the selected ADAPTER, not the pack's name.

    On long-work routes the tool enforces its own deadline, so the turn gets that
    budget plus a margin: the turn must outlive the tool for the tool's graceful
    timeout to reach the caller. Elsewhere the turn budget IS the only deadline and
    needs no margin. Deriving this from the route means a new pack that reaches the
    workspace lane inherits the right budget without editing this function.
    """
    if adapter in _LONG_WORK_ADAPTERS:
        return _agent_work_timeout_s() + _TURN_BUDGET_MARGIN_S
    return _timeout_for_profile(profile_name)

# Voice-first spoken moments (filler acknowledgment + "switch language" prompt)
# are generated by the multilingual model in the *selected* language and cached
# per language — no hardcoded per-language phrase tables and no English fallback,
# so they render correctly for every supported language. Filler is warmed in the
# background the first time a language is used, so no turn blocks on generating it.
_FILLER_CACHE: dict[str, str] = {}
_FILLER_WARMING: set[str] = set()
_PROGRESS_CACHE: dict[tuple[str, str], str] = {}
_SWITCH_CACHE: dict[str, str] = {}

_WORKSPACE_FIRST_PROMPT = (
    "You are a concise voice assistant. The deterministic router selected the "
    "governed workspace lane. Call workspace_query exactly once with the user's "
    "complete question. Do not answer from memory."
)
# Genie One resolves references against the conversation IT holds, so the one thing
# a continuing turn must not do is "help" by rewriting the user's words.
_WORKSPACE_CONTINUE_PROMPT = (
    "You are a concise voice assistant. This call has an open conversation with the "
    "governed workspace assistant, and that assistant holds what was said there — "
    "including any question it asked. Call workspace_query exactly once, passing the "
    "user's words through UNCHANGED: it resolves them against its own context, so "
    "restating, expanding, or guessing at the question behind them destroys the "
    "reference. Do not answer from memory."
)
_CLARIFY_PROMPT = (
    "You are a concise voice assistant. The navigation policy could not safely "
    "choose one evidence source. Ask exactly one short clarifying question in "
    "{language}. Do not answer the original question and do not mention routing."
)
_CONFIRM_PROMPT = (
    "You are a concise billing voice assistant. An exact billing offer is already "
    "present in the conversation, but the caller has not explicitly confirmed it. "
    "Restate that same action briefly and ask one yes-or-no confirmation question "
    "in {language}. Never ask for an account number or invent a different action, "
    "invoice, or amount."
)
_REFUSE_PROMPT = (
    "You are a concise voice assistant. The navigation policy determined that no "
    "authorized capability can answer this request reliably. Say so briefly in "
    "{language} and invite the caller to ask a supported question. "
    "Do not invent an answer."
)
_SELF_KNOW_PROMPT = (
    "You are the Genie Assisted Voice application. Briefly explain your own "
    "capabilities or help the caller navigate the app in {language}. Do not claim "
    "facts about the caller's workspace."
)


def _tools_and_prompt(
    profile: Any,
    *,
    route_adapter: str,
    workspace_selected: bool,
    conversation_joinable: bool,
    briefing: str = "",
) -> tuple[list[dict[str, Any]], str]:
    """Tool set and system prompt for this turn.

    A workspace turn gets the workspace tool and nothing else. Withdrawing the pack's
    tools is the point: a pack prompt legitimately says "search my corpus for ANY
    question about this domain", so leaving both in reach turns an answered question
    into a coin toss the caller experiences as the assistant forgetting what it asked.
    """
    from ..runtime.genie_one import WORKSPACE_QUERY_SPEC

    if route_adapter == "confirm":
        return [], _CONFIRM_PROMPT
    if route_adapter == "clarify":
        return [], _CLARIFY_PROMPT
    if route_adapter == "refuse":
        return [], _REFUSE_PROMPT
    if route_adapter == "self_know":
        return [], _SELF_KNOW_PROMPT
    if not workspace_selected:
        return profile.tools_spec(), profile.system_prompt
    if not conversation_joinable:
        return [WORKSPACE_QUERY_SPEC], _WORKSPACE_FIRST_PROMPT
    note = f"\n\n{briefing}" if briefing else ""
    return [WORKSPACE_QUERY_SPEC], f"{_WORKSPACE_CONTINUE_PROMPT}{note}"


def _navigation_intents(decision: Any) -> list[Any]:
    """Typed navigation outcomes that execute an app-selection signal directly."""
    from ..profiles import ResolvedIntent
    from ..runtime import capability_descriptor

    selection = capability_descriptor(decision.capability_id).selection
    if selection is None:
        return []
    return [
        ResolvedIntent(
            name=selection.tool_name,
            arguments=dict(selection.arguments),
            confirm_phrase=selection.confirm_phrase,
        )
    ]


def _tools_for_capability(
    tools: list[dict[str, Any]],
    capability_id: Any,
) -> list[dict[str, Any]]:
    """Withdraw tools not owned by the policy-validated capability."""
    from ..runtime import capability_descriptor

    descriptor = capability_descriptor(capability_id)
    if descriptor.selection is not None:
        return []
    if not descriptor.tools:
        return tools
    return [
        spec
        for spec in tools
        if str((spec.get("function") or {}).get("name") or "") in descriptor.tools
    ]


def _prompt_for_capability(profile: str, capability_id: Any, fallback: str) -> str:
    """Capability-local instructions matching the tools policy left in reach."""
    from ..runtime import capability_descriptor

    prompt = capability_descriptor(capability_id).prompt
    return prompt or fallback


def _forced_tool_choice(tools: list[dict[str, Any]]) -> dict[str, Any] | str:
    names = [
        str((spec.get("function") or {}).get("name") or "")
        for spec in tools
        if (spec.get("function") or {}).get("name")
    ]
    if len(names) == 1:
        return {"type": "function", "function": {"name": names[0]}}
    return "auto"


def _spoken_answer(
    *,
    response_text: str,
    rendered_summary: str,
    runtime_error_code: str | None,
    refuse_text: str | None,
    language: str,
) -> str:
    """Choose natural customer-facing text; structured claims are citations only."""
    from ..runtime.refuse import ErrorCode, refuse_speech

    if runtime_error_code == "timeout":
        return refuse_speech(ErrorCode.TIMEOUT, language=language)
    if rendered_summary:
        return rendered_summary
    if response_text.strip():
        return response_text
    return refuse_text or refuse_speech(ErrorCode.NO_EVIDENCE, language=language)


def _finalize_billing_offer(
    session: VoiceSession,
    *,
    capability_id: Any,
    tts_chunks: int,
) -> None:
    """Open confirmation only after the exact offer produced audible output."""
    from ..runtime import CapabilityId

    if capability_id != CapabilityId.BILLING_ACTION_PREPARE:
        return
    candidate = session.profile_state.pop("pending_billing_offer", None)
    if tts_chunks > 0 and isinstance(candidate, dict):
        session.profile_state["pending_confirm_mutate"] = candidate


_PROGRESS_KEYS = ("progress_1", "progress_2", "progress_3")

# Narration of a REAL upstream step ("running the query", "resolving the model
# name") beats a generic hold phrase, so when Genie reports what it is doing we say
# that instead. Cached per (language, step) because the same handful of steps
# recurs on every workspace question, making repeats free after the first turn.
_STEP_CACHE: dict[tuple[str, str], str] = {}
_MAX_STEP_CACHE = 256
# A hold phrase that takes longer than this to generate is not worth making the
# answer it describes wait for it.
_STEP_PHRASE_BY_LABEL = {
    "Understanding your question": "progress.understanding",
    "Finding the right data": "progress.finding_data",
    "Running the analysis": "progress.running_analysis",
    "Checking the results": "progress.checking_results",
    "Preparing your answer": "progress.preparing_answer",
}


async def _step_phrase(bundle: ServingBundle, language: str, label: str) -> str:
    """Localized narration for one upstream progress step ("" when unavailable)."""
    base = base_code(language)
    cached = _STEP_CACHE.get((base, label))
    if cached is not None:
        return cached
    phrase_key = _STEP_PHRASE_BY_LABEL.get(label)
    if phrase_key is None:
        return ""
    text = fixed_phrase(phrase_key, language=language)
    if text and len(_STEP_CACHE) < _MAX_STEP_CACHE:
        _STEP_CACHE[(base, label)] = text
    return text


async def warm_filler(bundle: ServingBundle, language: str) -> None:
    """Public entrypoint to pre-load the in-language filler (see _warm_filler).

    Called at session start so the acknowledgment is cached and ready before the
    first (often cold, tool-heavy) turn finishes — otherwise turn 1 races LLM
    generation against the grace window and the caller hears dead air.
    """
    await _warm_filler(bundle, language)


async def _warm_filler(bundle: ServingBundle, language: str) -> None:
    """Load and cache bounded in-language engagement phrases."""
    # No concrete language yet ("auto") → nothing to warm; the per-turn warm fires
    # once STT resolves the actual language. Guarding here keeps a stray "auto"
    # from caching an English clip under a bogus key.
    if not language or language == "auto":
        return
    base = base_code(language)
    progress_ready = all((base, key) in _PROGRESS_CACHE for key in _PROGRESS_KEYS)
    if (base in _FILLER_CACHE and progress_ready) or base in _FILLER_WARMING:
        return
    _FILLER_WARMING.add(base)
    if base not in _FILLER_CACHE:
        _FILLER_CACHE[base] = fixed_phrase("filler.ack", language=language)
    for key in _PROGRESS_KEYS:
        if (base, key) not in _PROGRESS_CACHE:
            _PROGRESS_CACHE[(base, key)] = fixed_phrase(
                f"filler.{key}", language=language
            )
    _FILLER_WARMING.discard(base)


# Spoken confirmations for deterministically-resolved intents are reviewed copy,
# loaded in the caller's language and cached per (phrase key, base-language).
_CONFIRM_CACHE: dict[tuple[str, str], str] = {}


async def _confirm_phrase(bundle: ServingBundle, phrase_key: str, language: str) -> str:
    """Short spoken confirmation for a pre-routed intent, in the caller's language."""
    key = (phrase_key, base_code(language))
    cached = _CONFIRM_CACHE.get(key)
    if cached is not None:
        return cached
    text = fixed_phrase(phrase_key, language=language)
    if text:
        _CONFIRM_CACHE[key] = text
    return text


async def _switch_prompt(bundle: ServingBundle, expected: str, detected: str) -> str:
    """Spoken 'wrong language' prompt, rendered in the agent's selected language."""
    key = f"{base_code(expected)}>{base_code(detected)}"
    cached = _SWITCH_CACHE.get(key)
    if cached is not None:
        return cached
    text = fixed_phrase(
        "language.switch",
        language=expected,
        detected_language=english_name(detected),
    )
    if text:
        _SWITCH_CACHE[key] = text
    return text


def turn_provenance(config) -> tuple[str, str, str]:
    """Return the validated profile, ingress surface and traffic class for a turn."""
    profile_name = config.profile or "concierge"
    surface = getattr(config, "surface", None) or {
        "concierge": "home",
        "billing": "telco",
        "card": "card",
        "knowledge": "knowledge",
    }.get(profile_name, "realtime")
    traffic_class = {
        "benchmark": "benchmark",
        "realtime-test": "diagnostic",
        "diagnostic": "diagnostic",
        "mcp": "mcp",
    }.get(surface, "conversation")
    return profile_name, surface, traffic_class


async def process_turn(
    bundle: ServingBundle,
    session: VoiceSession,
    turn_id: int,
    audio: bytes,
    *,
    context: str | None = None,
) -> AsyncIterator[dict]:
    # One trace per turn: STT → language gate → LLM (+tool) iterations → TTS. The
    # trace is submitted to a background writer in the finally block, so it is
    # persisted even on early-return paths (empty/mismatch/superseded) and errors,
    # and NEVER blocks the turn (submit is a non-blocking enqueue).
    profile_name, surface, traffic_class = turn_provenance(session.config)
    trace = TurnTrace(
        session_id=session.session_id or "",
        turn_id=turn_id,
        capability=SPEECH_LLM_TOOLASSIST_SPEECH,
        call_id=session.config.call_id,
        customer_id=session.config.customer_id,
        profile=profile_name,
        surface=surface,
        traffic_class=traffic_class,
    )
    trace.language = session.config.language
    provenance_tokens = push_inference_context(
        {
            "traffic_class": traffic_class,
            "surface": surface,
            "profile": profile_name,
            "trace_id": trace.trace_id,
            "session_id": trace.session_id,
            "turn_id": trace.turn_id,
            "capability": trace.capability,
        },
        recorder=trace.record_model_call,
    )
    localization_task: asyncio.Task | None = None
    try:
        stt_span = trace.span(
            "stt", "STT",
            input={
                "audio_bytes": len(audio),
                "sample_rate_hz": session.config.sample_rate_hz,
                "requested_language": session.config.language,
                "expected_language": session.config.expected_language,
            },
        )
        pretranscribed = None
        if getattr(session, "active_turn", None) is not None:
            pretranscribed = session.active_turn.meta.pop("pretranscribed", None)
        if (
            isinstance(pretranscribed, tuple)
            and len(pretranscribed) == 3
        ):
            transcript, detected, stt_ms = pretranscribed
            trace.set_metric("stt_reused_for_barge_classification", True)
        else:
            transcript, detected, stt_ms = await transcribe(bundle, session, audio)
        stt_resource = str(getattr(bundle.stt, "stt_endpoint", "stt"))
        transcript = admit_transcript(
            transcript,
            detected_language=detected,
            pinned_language=session.config.language,
            resource=stt_resource,
            ledger=trace.guards,
        )
        stt_span.set_output({"transcript": transcript, "detected_language": detected}).set_attribute(
            "stt_ms", stt_ms
        ).end()
        trace.input_transcript = transcript
        trace.detected_language = detected
        # Log EVERY STT result (including empties) so silent-drop turns are visible:
        # distinguishes "STT heard nothing" (silence/echo) from "STT heard the wrong
        # language" (gate) from a real transcript.
        logger.info(
            "stt turn %d: %dms detected=%s len=%d text=%r",
            turn_id, stt_ms, detected, len(transcript), transcript[:120],
        )
        if turn_id != session.turn_id:
            trace.status = "superseded"
            report(
                trace.guards, "stale_turn", "fired",
                stage="turn", surface="internal",
                reason=f"turn {turn_id} superseded by {session.turn_id}",
            )
            return
        if not transcript.strip():
            trace.status = "empty_transcript"
            report(
                trace.guards, "empty_transcript", "fired",
                stage="turn", surface="internal", reason="blank transcript; turn dropped",
            )
            return
        # Language gate: if the caller isn't speaking the selected language, don't
        # surface the off-language transcript or run the assistant. Warn the UI
        # (visual banner) and *speak* a localized prompt asking the agent to switch
        # the picker — the spoken guidance lives here because this is the only route
        # with a TTS stage. The turn is otherwise dropped (no history, no LLM/reply).
        mismatch = language_mismatch(session, detected)
        if not session.config.expected_language:
            report(
                trace.guards, "language_gate", "not_evaluated",
                seam="decision", stage="routing",
                phase="post_stt", resource="transcript_boundary",
                reason="session set no expected_language",
            )
        else:
            report(
                trace.guards, "language_gate", "fired" if mismatch else "passed",
                seam="decision", stage="routing",
                phase="post_stt", resource="transcript_boundary",
                reason=(
                    f"expected={mismatch['expected']} detected={mismatch['detected']}; "
                    "turn dropped, switch prompt spoken"
                    if mismatch
                    else f"detected={detected or 'unknown'} matches selection"
                ),
            )
        if mismatch:
            trace.status = "language_mismatch"
            with trace.span("language.gate", "GUARD", input={"detected": detected}) as gate:
                gate.set_output(mismatch)
            yield {"type": "language.mismatch", "turn_id": turn_id, **mismatch}
            try:
                prompt = await _switch_prompt(bundle, mismatch["expected"], mismatch["detected"])
            except Exception:  # noqa: BLE001
                logger.warning("switch-language phrase failed for turn %d", turn_id, exc_info=True)
                prompt = ""
            if prompt:
                report(
                    trace.guards,
                    "deterministic_speech",
                    "passed",
                    stage="product_copy",
                    owner="application",
                    phase="before_synthesis",
                    resource="language.switch",
                    reason="reviewed language-mismatch prompt",
                )
                try:
                    async for event in stream_tts(
                        bundle, session, turn_id, prompt, mismatch["expected"],
                        emit_text=False, trace=trace,
                    ):
                        if turn_id != session.turn_id:
                            break
                        yield event
                except Exception:  # noqa: BLE001
                    logger.warning("switch-language TTS failed for turn %d", turn_id, exc_info=True)
            session.set_cooldown(1.5)
            return

        language = resolve_language(session, detected)
        trace.language = language
        # Warm the in-language filler in the background so a later slow turn in this
        # language can play it without blocking on generation.
        asyncio.ensure_future(_warm_filler(bundle, language))
        yield {
            "type": "transcript.final",
            "turn_id": turn_id,
            "text": transcript,
            "language": language,
            "stt_ms": stt_ms,
        }

        session.history.append({"role": "user", "content": transcript})
        if getattr(session, "active_turn", None) is not None:
            session.active_turn.meta["utterance"] = transcript

        # Every session runs through a named profile (see profiles.py). Unset
        # profile falls back to the voice concierge — never an implicit billing pack.
        profile = get_profile(session.config.profile or "concierge")
        tool_ctx = profile.make_context(session, language)

        # GoalFrame is the live routing decision—not a test-only helper and not an
        # LLM choosing among API names.
        from ..runtime.goal_frame import RouteDecision

        has_obo = bool(getattr(getattr(session, "principal", None), "has_token", False))
        conversation = session.workspace_conversation
        conversation_joinable = conversation.is_open and has_obo
        navigation_decision = None
        navigation_intents: list[Any] = []
        route: RouteDecision | None = None

        # Every profile uses the same silent, typed navigation procedure. The model
        # proposes one language-neutral capability; policy validates profile access,
        # confidence, OBO, effect and conversation ownership before tools are exposed.
        from ..runtime import (
            CapabilityId,
            NavigationDecision,
            NavigationReason,
            capability_descriptor,
            classifier_capabilities,
            route_for_navigation,
            run_profile_navigation,
        )

        classifier_catalog = classifier_capabilities(profile.name)
        classifier_was_used = not (
            profile.name == "knowledge" and conversation.is_open
        )
        semantic_span = (
            trace.span(
                "navigation.semantic",
                "LLM",
                input={
                    "profile": profile.name,
                    "language": language,
                    "capabilities": [item.id.value for item in classifier_catalog],
                },
            )
            if classifier_was_used
            else None
        )
        navigation_context = next(
            (
                str(message.get("content") or "")[:1200]
                for message in reversed(session.history[:-1])
                if message.get("role") == "assistant" and message.get("content")
            ),
            "",
        )
        # A cancelled prior preparation may leave an uncommitted candidate, but
        # it is never an open offer and must not cross into a new turn.
        session.profile_state.pop("pending_billing_offer", None)
        offer_open = bool(session.profile_state.get("pending_confirm_mutate"))
        try:
            navigation_decision, route = await run_profile_navigation(
                transcript,
                profile=profile.name,
                language=language,
                classifier=bundle.llm,
                has_obo=has_obo,
                conversation_open=conversation.is_open,
                min_confidence=_navigation_min_confidence(),
                context=navigation_context,
                offer_open=offer_open,
            )
            if semantic_span is not None:
                semantic_span.set_output(
                    navigation_decision.model_dump(mode="json")
                ).end()
            if classifier_was_used:
                report(
                    trace.guards,
                    "navigation.semantic",
                    "delegated",
                    seam="decision",
                    stage="routing",
                    owner="qwen",
                    reason=navigation_decision.reason.value,
                )
        except Exception as exc:  # noqa: BLE001
            if semantic_span is not None:
                semantic_span.set_status("error").set_attribute(
                    "error", repr(exc)
                ).end()
            logger.warning(
                "%s navigation classifier failed for turn %d",
                profile.name,
                turn_id,
                exc_info=True,
            )
            report(
                trace.guards,
                "navigation.semantic",
                "error",
                seam="decision",
                stage="routing",
                owner="qwen",
                reason=type(exc).__name__,
            )
            navigation_decision = NavigationDecision(
                capability_id=CapabilityId.CLARIFY,
                confidence=0.0,
                ambiguous=True,
                reason=NavigationReason.AMBIGUOUS,
            )

        if route is None:
            route = route_for_navigation(
                navigation_decision,
                utterance=transcript,
                profile=profile.name,
            )
        navigation_intents = _navigation_intents(navigation_decision)
        if navigation_decision.reason == NavigationReason.CONFIRMATION_REQUIRED:
            # Keep the exact preparation snapshot. Replacing it with a boolean
            # would sever confirmation from the action/customer/invoice it covers.
            session.profile_state.setdefault("pending_confirm_mutate", True)
        elif navigation_decision.capability_id not in {
            CapabilityId.CLARIFY,
            CapabilityId.REFUSE,
            CapabilityId.BILLING_ACTION,
        }:
            session.profile_state.pop("pending_confirm_mutate", None)
        policy_outcome = (
            "fired"
            if navigation_decision.capability_id
            in {CapabilityId.CLARIFY, CapabilityId.REFUSE}
            else "passed"
        )
        report(
            trace.guards,
            "navigation.policy",
            policy_outcome,
            seam="decision",
            stage="routing",
            owner="us",
            reason=(
                f"{navigation_decision.capability_id.value}:"
                f"{navigation_decision.reason.value}"
            )
        )
        trace.set_metric(
            "navigation_capability",
            navigation_decision.capability_id.value,
        )
        trace.set_metric(
            "navigation_confidence",
            navigation_decision.confidence,
        )
        trace.set_metric(
            "navigation_reason",
            navigation_decision.reason.value,
        )
        assert route is not None
        trace.set_metric("router_path", route.path.value)
        trace.set_metric("router_adapter", route.adapter)
        if getattr(session, "active_turn", None) is not None:
            session.active_turn.meta["goal_frame"] = route.frame
            session.active_turn.meta["route_adapter"] = route.adapter
        workspace_selected = route.adapter == "genie_one_mcp"
        if conversation_joinable:
            trace.set_metric("workspace_conversation_turns", conversation.turns)
        turn_budget_s = _timeout_for_route(route.adapter, session.config.profile)
        trace.set_metric("turn_budget_s", turn_budget_s)

        # ── Deterministic intent pre-router (framework seam) ──────────────
        # Navigation/selection is a DETERMINISTIC action; don't gate it solely
        # on the conversational LLM choosing to emit the tool call. If the
        # profile resolves a confident intent from this transcript, run it via
        # the same tool_runner (so the usual tool.called event fires), speak a
        # short in-language confirmation, and SKIP the LLM turn. Ambiguous / no
        # match returns [] and falls through to the LLM below, which can clarify
        # and still call the tool itself. Fully generic: the engine never names
        # a domain — only the profile knows what its intents are.
        resolved = navigation_intents
        if resolved:
            with trace.span("intent.router", "GUARD", input={"transcript": transcript}) as span:
                span.set_output({"intents": [r.name for r in resolved]})
            confirm_text = ""
            confirm_phrase_key = ""
            executed_intent = False
            for r in resolved:
                try:
                    result_json = profile.tool_runner(r.name, r.arguments, tool_ctx)
                except Exception:  # noqa: BLE001
                    logger.warning("intent-router tool %s failed for turn %d", r.name, turn_id, exc_info=True)
                    continue
                executed_intent = True
                try:
                    result_obj = json.loads(result_json)
                except (TypeError, ValueError):
                    result_obj = result_json
                yield {
                    "type": "tool.called",
                    "turn_id": turn_id,
                    "name": r.name,
                    "arguments": r.arguments,
                    "result": result_obj,
                }
                if r.confirm_phrase and not confirm_text:
                    confirm_phrase_key = r.confirm_phrase
                    confirm_text = await _confirm_phrase(
                        bundle, r.confirm_phrase, language
                    )
            if executed_intent:
                if turn_id != session.turn_id:
                    trace.status = "superseded"
                    return
                if confirm_text:
                    report(
                        trace.guards,
                        "deterministic_speech",
                        "passed",
                        stage="product_copy",
                        owner="application",
                        phase="before_synthesis",
                        resource=confirm_phrase_key,
                        reason="reviewed navigation confirmation",
                    )
                    session.history.append({"role": "assistant", "content": confirm_text})
                    trace.output_text = confirm_text
                    yield {"type": "response.text", "turn_id": turn_id, "text": confirm_text}
                    tts_span = trace.span("tts", "TTS", input={"text": confirm_text, "language": language})
                    async for event in stream_tts(
                        bundle, session, turn_id, confirm_text, language, trace=trace
                    ):
                        if turn_id != session.turn_id:
                            break
                        yield event
                    tts_span.set_attribute("tts_first_ms", trace.tts_first_ms).end()
                if profile.after_turn is not None:
                    profile.after_turn(tool_ctx, session)
                session.set_cooldown(1.5)
                return

        runtime_loop = asyncio.get_running_loop()
        runtime_event_queue: asyncio.Queue | None = None

        def _guarded_tool_runner(name: str, arguments: dict, ctx) -> str:
            from ..runtime.goal_frame import enforce_effect
            from ..tool_registry import tool_effect

            effect = tool_effect(name, profile=profile.name)
            blocked = enforce_effect(
                effect,
                user_confirmed=bool(
                    navigation_decision.capability_id == CapabilityId.BILLING_ACTION
                    and navigation_decision.confirmed
                    and session.profile_state.get("pending_confirm_mutate")
                ),
            )
            if blocked is not None:
                trace.increment_metric("effect_blocked")
                return json.dumps(
                    {
                        "error": blocked.message,
                        "error_evidence": blocked.as_dict(),
                        "effect_class": effect,
                    }
                )
            trace.set_metric("effect_class", effect)
            if name == "workspace_query":
                from genie_voice.config import get_settings

                from ..runtime.genie_one import run_workspace_query

                question = str(arguments.get("question") or transcript)

                def _on_workspace_progress(payload: dict[str, Any]) -> None:
                    # The MCP client runs in the tool thread. Marshal progress back
                    # onto the turn's event loop; never touch asyncio.Queue directly
                    # from that thread.
                    if runtime_event_queue is not None:
                        runtime_loop.call_soon_threadsafe(
                            runtime_event_queue.put_nowait,
                            ("workspace.progress", dict(payload)),
                        )

                workspace_started = time.perf_counter()
                workspace_timeout_s = _tool_work_timeout_s(
                    route.adapter, session.config.profile
                )
                logger.info(
                    "workspace_query started turn=%d timeout_s=%.1f conversation_open=%s",
                    turn_id,
                    workspace_timeout_s,
                    bool(conversation.handle),
                )
                raw = run_workspace_query(
                    question,
                    principal=getattr(session, "principal", None),
                    host=get_settings().databricks_host,
                    session_id=session.session_id or "unknown",
                    turn_id=turn_id,
                    timeout_s=workspace_timeout_s,
                    conversation_id=conversation.handle,
                    on_progress=_on_workspace_progress,
                )
                workspace_ms = round((time.perf_counter() - workspace_started) * 1000)
                trace.set_metric("workspace_query_ms", workspace_ms)
                try:
                    returned = json.loads(raw)
                except json.JSONDecodeError:
                    returned = {}
                logger.info(
                    "workspace_query finished turn=%d elapsed_ms=%d status=%s timeout=%s",
                    turn_id,
                    workspace_ms,
                    returned.get("status") if isinstance(returned, dict) else None,
                    returned.get("timeout") if isinstance(returned, dict) else None,
                )
                if isinstance(returned, dict):
                    conversation.bind(returned.get("conversation_id"))
                    conversation.record(
                        question=question,
                        answer=str(returned.get("final_answer") or ""),
                    )
                return raw
            return profile.tool_runner(name, arguments, ctx)

        tools_for_route, system_prompt_for_route = _tools_and_prompt(
            profile,
            route_adapter=route.adapter,
            workspace_selected=workspace_selected,
            conversation_joinable=conversation_joinable,
            briefing=conversation.briefing(),
        )
        tools_for_route = _tools_for_capability(
            tools_for_route,
            navigation_decision.capability_id,
        )
        system_prompt_for_route = _prompt_for_capability(
            profile.name,
            navigation_decision.capability_id,
            system_prompt_for_route,
        )
        capability = capability_descriptor(navigation_decision.capability_id)
        tool_choice = (
            _forced_tool_choice(tools_for_route) if capability.requires_tool else "auto"
        )

        respond_extra: dict = {
            "system_prompt": system_prompt_for_route,
            "tools_override": tools_for_route,
            "tool_runner": _guarded_tool_runner,
            "tool_choice": tool_choice,
        }
        t = time.perf_counter()
        respond_fn = getattr(bundle.llm, "respond_with_tools", None)
        tool_invocations: list[dict] = []
        runtime_error_code: str | None = None

        # Snapshot the history the LLM will actually see this turn (text-only —
        # tool calls are NOT persisted across turns).
        history_for_turn = [dict(m) for m in session.history[:-1]]
        trace.span("history", "GUARD", input={"messages": history_for_turn}).set_output(
            {"message_count": len(history_for_turn)}
        ).end()

        response_text = ""
        # Wire sequence for this turn's ordered runtime events. Hoisted out of the
        # runtime branch so the committed-speech event below extends the same
        # sequence whichever LLM path ran.
        last_event_seq = int(session.event_seq_by_turn.get(turn_id, 0))
        if respond_fn:
            from ..runtime import (
                AgentContext,
                AgentEventKind,
                AgentGoal,
                LiveToolRespondAdapter,
            )

            active = getattr(session, "active_turn", None)
            cancellation = getattr(active, "cancel", None)
            if cancellation is None:
                from ..runtime import CancellationToken

                cancellation = CancellationToken()
            runtime = LiveToolRespondAdapter(respond_fn)
            event_queue: asyncio.Queue = asyncio.Queue()
            runtime_event_queue = event_queue
            seq_offset = last_event_seq

            async def _produce_runtime_events() -> None:
                try:
                    async for agent_event in runtime.run(
                        AgentGoal(
                            utterance=transcript,
                            language=language,
                            meta={
                                "route_path": route.path.value,
                                "adapter": "genie_one_mcp" if workspace_selected else route.adapter,
                                "goal_frame": (
                                    {
                                        "scope": route.frame.scope,
                                        "depth": route.frame.depth,
                                        "effect": route.frame.effect,
                                        "pack_id": route.frame.pack_id,
                                    }
                                    if route.frame
                                    else None
                                ),
                            },
                        ),
                        AgentContext(
                            turn_id=turn_id,
                            history=session.history[:-1],
                            tool_ctx=tool_ctx,
                            context=context,
                            system_prompt=respond_extra["system_prompt"],
                            tools_override=respond_extra["tools_override"],
                            tool_runner=respond_extra["tool_runner"],
                            tool_choice=respond_extra["tool_choice"],
                            trace=trace,
                        ),
                        cancellation,
                    ):
                        await event_queue.put(agent_event)
                finally:
                    await event_queue.put(None)

            runtime_task = asyncio.create_task(_produce_runtime_events())
            from ..runtime import SpeechKind, SpeechRequest
            from ..runtime.engagement import EngagementKind, LongWorkEngagement
            from ..runtime.genie_one import (
                TERMINAL_STATUSES,
                current_progress_step,
                normalize_progress_steps,
            )

            # ``turn.accepted`` arrives immediately, but it is a protocol event—not
            # something the caller can hear. The old filler gate mistook it for a
            # response and then allowed 60–90 seconds of silence. This clock runs
            # independently of runtime events and stops the moment work completes.
            engagement = LongWorkEngagement()
            engagement_started = time.monotonic()
            engagement_seq_shift = 0
            spoken_step = ""

            async def _speak_engagement(
                phrase: str,
                *,
                stage: str,
                elapsed_s: float,
                speech_kind,
                steps: list[dict[str, str]] | None = None,
            ):
                """Emit the on-screen progress moment and speak those exact words.

                The page renders the same event that drives the audio, so what is
                displayed and what is heard can never drift apart.
                """
                nonlocal engagement_seq_shift, last_event_seq
                scheduler = getattr(session, "speech_scheduler", None)
                if scheduler is not None and not scheduler.accept(
                    SpeechRequest(kind=speech_kind, text=phrase)
                ):
                    return
                engagement_seq_shift += 1
                last_event_seq += 1
                session.event_seq_by_turn[turn_id] = last_event_seq
                payload: dict[str, Any] = {
                    "stage": stage,
                    "text": phrase,
                    "elapsed_ms": round(elapsed_s * 1000),
                }
                if steps:
                    payload["steps"] = steps
                yield {
                    "type": "turn.event",
                    "turn_id": turn_id,
                    "seq": last_event_seq,
                    "kind": "speech.progress",
                    "payload": payload,
                }
                span = trace.span(
                    f"tts.{stage}",
                    "TTS",
                    input={"text": phrase, "language": language},
                )
                try:
                    async for event in stream_tts(
                        bundle,
                        session,
                        turn_id,
                        phrase,
                        language,
                        mark_final=False,
                        emit_text=False,
                        trace=trace,
                        primary=False,
                    ):
                        if turn_id != session.turn_id:
                            break
                        yield event
                except Exception:  # noqa: BLE001
                    logger.warning(
                        "engagement TTS %s failed for turn %d",
                        stage,
                        turn_id,
                        exc_info=True,
                    )
                    span.set_status("error")
                span.end()

            async def _consume(agent_event) -> None:
                nonlocal response_text
                if agent_event.kind == AgentEventKind.ACTION_COMPLETED:
                    tool_invocations.append(
                        {
                            "name": agent_event.payload.get("name"),
                            "arguments": agent_event.payload.get("arguments"),
                            "result": agent_event.payload.get("result"),
                        }
                    )
                elif agent_event.kind == AgentEventKind.ANSWER_FINAL:
                    response_text = str(agent_event.payload.get("text") or "")

            try:
                async with asyncio.timeout(turn_budget_s):
                    current = None
                    while True:
                        if not runtime_task.done():
                            elapsed_s = time.monotonic() - engagement_started
                            for stage in engagement.pop_due(elapsed_s):
                                base = base_code(language)
                                phrase = (
                                    _FILLER_CACHE.get(base)
                                    if stage.kind == EngagementKind.ACK
                                    else _PROGRESS_CACHE.get((base, stage.key))
                                )
                                if not phrase:
                                    continue
                                async for event in _speak_engagement(
                                    phrase,
                                    stage=stage.key,
                                    elapsed_s=elapsed_s,
                                    speech_kind=(
                                        SpeechKind.ACK
                                        if stage.kind == EngagementKind.ACK
                                        else SpeechKind.PROGRESS
                                    ),
                                ):
                                    yield event

                        if current is None:
                            if runtime_task.done() and event_queue.empty():
                                break
                            next_due = engagement.next_due_s
                            wait_s = (
                                None
                                if next_due is None or runtime_task.done()
                                else max(
                                    0.0,
                                    next_due
                                    - (time.monotonic() - engagement_started),
                                )
                            )
                            try:
                                if wait_s is None:
                                    current = await event_queue.get()
                                else:
                                    current = await asyncio.wait_for(
                                        event_queue.get(), timeout=wait_s
                                    )
                            except asyncio.TimeoutError:
                                continue
                            if current is None:
                                break
                        if (
                            isinstance(current, tuple)
                            and len(current) == 2
                            and current[0] == "workspace.progress"
                        ):
                            progress_payload = current[1]
                            steps = normalize_progress_steps(
                                progress_payload.get("progress_steps")
                            )
                            progress_elapsed_s = time.monotonic() - engagement_started
                            engagement_seq_shift += 1
                            last_event_seq += 1
                            session.event_seq_by_turn[turn_id] = last_event_seq
                            yield {
                                "type": "turn.event",
                                "turn_id": turn_id,
                                "seq": last_event_seq,
                                "kind": "tool.progress",
                                "payload": {
                                    "name": "workspace_query",
                                    "status": progress_payload.get("status"),
                                    "steps": steps,
                                    "elapsed_ms": round(progress_elapsed_s * 1000),
                                },
                            }
                            # Saying what the workspace is actually doing beats a
                            # generic hold phrase, so a new upstream step takes the
                            # microphone and pushes the heartbeat out behind it. Once
                            # the read is done there is nothing left to narrate and
                            # narrating anyway would only delay the answer itself.
                            step_label = current_progress_step(steps)
                            still_working = (
                                str(progress_payload.get("status") or "").lower()
                                not in TERMINAL_STATUSES
                            )
                            if still_working and step_label and step_label != spoken_step:
                                spoken_step = step_label
                                step_text = await _step_phrase(
                                    bundle, language, step_label
                                )
                                if step_text:
                                    engagement.defer(progress_elapsed_s)
                                    async for event in _speak_engagement(
                                        step_text,
                                        stage="tool_step",
                                        elapsed_s=progress_elapsed_s,
                                        speech_kind=SpeechKind.PROGRESS,
                                        steps=steps,
                                    ):
                                        yield event
                            current = None
                            continue
                        await _consume(current)
                        # Ordered AgentRuntime events are now a real wire contract,
                        # not a test-only library.
                        wire_event = _public_agent_event(current.envelope())
                        wire_event["seq"] = (
                            seq_offset + int(current.seq) + engagement_seq_shift
                        )
                        last_event_seq = int(wire_event["seq"])
                        session.event_seq_by_turn[turn_id] = last_event_seq
                        yield wire_event
                        current = None
            except TimeoutError:
                cancellation.cancel()
                runtime_error_code = "timeout"
                logger.error("AgentRuntime timed out for turn %d", turn_id)
                yield {
                    "type": "turn.event",
                    "turn_id": turn_id,
                    "seq": last_event_seq + 1,
                    "kind": "turn.failed",
                    "payload": {"code": "timeout", "retryable": True},
                }
                session.event_seq_by_turn[turn_id] = last_event_seq + 1
            finally:
                if not runtime_task.done():
                    runtime_task.cancel()
                await asyncio.gather(runtime_task, return_exceptions=True)
        else:
            response_text = await asyncio.wait_for(
                asyncio.to_thread(
                    bundle.llm.respond,
                    transcript,
                    language=language,
                    context=context,
                    tool_ctx=tool_ctx,
                ),
                timeout=turn_budget_s,
            )

        llm_ms = round((time.perf_counter() - t) * 1000)
        trace.output_text = response_text
        logger.info("LLM responded in %dms (turn %d, tools=%d)", llm_ms, turn_id, len(tool_invocations))
        if turn_id != session.turn_id:
            trace.status = "superseded"
            return

        session.history.append({"role": "assistant", "content": response_text})
        # Keep history bounded to last 10 exchanges (20 messages)
        if len(session.history) > 20:
            session.history = session.history[-20:]

        for invocation in tool_invocations:
            yield {
                "type": "tool.called",
                "turn_id": turn_id,
                "name": invocation["name"],
                "arguments": invocation.get("arguments"),
                "result": invocation.get("result"),
            }

        # Structured evidence remains available for tables and traces.
        # Customer-facing speech comes from natural prose returned by Genie/Agent
        # Mode, or from the conversational response when the tool returns rows only.
        refuse_text: str | None = None
        # A long written answer may need the same dual rendering as the FSI deep
        # dive: short translated voice summary + full translated panel report.
        # ``render_answer`` is what gets summarized for the voice; ``render_report``
        # is the written answer the panel paints. They differ when the governed
        # result is a table: the summary is spoken, but the panel renders the typed
        # rows as a table/chart rather than a second copy of them as markdown.
        render_answer = ""
        render_report = ""
        render_question = transcript
        render_source = ""
        render_source_language = language
        from ..runtime import evidence_from_tool_result
        from ..runtime.answer_rendering import upstream_answer_render

        for invocation in tool_invocations:
            name = str(invocation.get("name") or "")
            raw = invocation.get("result")
            if isinstance(raw, str):
                try:
                    raw = json.loads(raw)
                except json.JSONDecodeError:
                    raw = {"answer": raw}
            ev = evidence_from_tool_result(
                name,
                raw if isinstance(raw, dict) else {"answer": raw},
            )
            answer_source, panel_report = upstream_answer_render(ev)
            if answer_source:
                render_answer = answer_source
                render_report = panel_report
                render_source = ev.source
                render_source_language = str(
                    ev.meta.get("source_language") or language
                )
                args = invocation.get("arguments")
                if isinstance(args, dict) and args.get("question"):
                    render_question = str(args["question"])
            if ev.error is not None and refuse_text is None:
                refuse_text = refuse_speech(ev.error.code, language=language)

        rendered_summary = ""
        if render_answer:
            from ..runtime.answer_rendering import summarize_for_voice

            try:
                rendered_summary = await asyncio.to_thread(
                    summarize_for_voice,
                    render_question,
                    render_answer,
                    language,
                    source_language=render_source_language,
                    trace=trace,
                )
            except Exception:  # noqa: BLE001
                logger.warning("long-answer summary failed for turn %d", turn_id, exc_info=True)

        speech_text = _spoken_answer(
            response_text=response_text,
            rendered_summary=rendered_summary,
            runtime_error_code=runtime_error_code,
            refuse_text=refuse_text,
            language=language,
        )
        if navigation_decision.capability_id == CapabilityId.BILLING_ACTION:
            session.profile_state.pop("pending_confirm_mutate", None)

        localization_queue: asyncio.Queue[str | None] | None = None
        from ..runtime.answer_rendering import localize_answer_stream, same_language

        # Full report rendering is independent of the optional spoken summary.
        # Previously an empty summary suppressed answer.render.* entirely, leaving
        # the UI's raw English action result visible on every non-English call.
        localization_pending = bool(render_report) and not same_language(
            render_source_language, language
        )
        if render_report or rendered_summary:
            # Tables are rendered separately from typed evidence. This event carries
            # natural prose only, in either its source language or the call language.
            last_event_seq += 1
            session.event_seq_by_turn[turn_id] = last_event_seq
            yield {
                "type": "turn.event",
                "turn_id": turn_id,
                "seq": last_event_seq,
                "kind": "answer.render.started",
                "payload": {
                    "question": render_question,
                    "summary": rendered_summary,
                    # Never flash English during a non-English call. The panel
                    # opens with the translated summary while deltas fill the report.
                    "report": "" if localization_pending else render_report,
                    "report_language": (
                        language if localization_pending else render_source_language
                    ),
                    "localization_pending": localization_pending,
                    "source": render_source,
                },
            }

            if localization_pending:
                localization_queue = asyncio.Queue()
                loop = asyncio.get_running_loop()

                def _localize() -> None:
                    try:
                        for delta in localize_answer_stream(
                            render_report,
                            language,
                            source_language=render_source_language,
                            trace=trace,
                        ):
                            loop.call_soon_threadsafe(localization_queue.put_nowait, delta)
                    finally:
                        loop.call_soon_threadsafe(localization_queue.put_nowait, None)

                localization_task = asyncio.create_task(asyncio.to_thread(_localize))

        # Transport sanitation removes accidental tool-call markup. It does not
        # inspect, rewrite, reject, or replace Genie/Agent Mode answer content.
        speech_admission = admit_speech_output(
            speech_text,
            resource=str(getattr(bundle.tts, "tts_endpoint", "tts")),
            ledger=trace.guards,
        )
        speech_text = speech_admission.text
        yield {
            "type": "response.text",
            "turn_id": turn_id,
            "text": speech_text,
            "llm_ms": llm_ms,
        }

        # The exact words TTS is about to speak. A client that renders `response.text`
        # now sees the same admitted text; publish the committed event as the typed
        # evidence/citation contract used by richer clients.
        # Emitted before synthesis starts, so the text lands as the voice begins.
        last_event_seq += 1
        session.event_seq_by_turn[turn_id] = last_event_seq
        yield {
            "type": "turn.event",
            "turn_id": turn_id,
            "seq": last_event_seq,
            "kind": "speech.committed",
            "payload": {
                "text": speech_text,
                "basis": "evidence" if tool_invocations else "conversation",
            },
        }

        # Let the profile persist any small cross-turn state (e.g. a selected use
        # case) onto session.profile_state. Generic: the engine doesn't know or
        # care what a given profile stores.
        if profile.after_turn is not None:
            profile.after_turn(tool_ctx, session)

        # The span's duration_ms is the FULL synthesis+stream (blocking) time.
        # tts_first_ms is this synthesis' own time-to-first-audio; the turn-level
        # figure the caller actually felt is trace.ttft_ms, which may belong to a
        # filler that played earlier.
        tts_span = trace.span(
            "tts",
            "TTS",
            input={
                "text": speech_text,
                "language": language,
                "basis": "evidence" if tool_invocations else "conversation",
            },
        )
        tts_chunks = 0
        tts_first_ms: int | None = None
        async for event in stream_tts(
            bundle, session, turn_id, speech_admission, language, trace=trace
        ):
            # Paint any full-answer translation produced since the last audio
            # chunk. Translation and TTS started together, so these ordered events
            # reach the panel while the short summary is being spoken.
            if localization_queue is not None:
                while not localization_queue.empty():
                    delta = localization_queue.get_nowait()
                    if delta is None:
                        localization_pending = False
                        last_event_seq += 1
                        session.event_seq_by_turn[turn_id] = last_event_seq
                        yield {
                            "type": "turn.event",
                            "turn_id": turn_id,
                            "seq": last_event_seq,
                            "kind": "answer.render.completed",
                            "payload": {"report_language": language},
                        }
                        break
                    last_event_seq += 1
                    session.event_seq_by_turn[turn_id] = last_event_seq
                    yield {
                        "type": "turn.event",
                        "turn_id": turn_id,
                        "seq": last_event_seq,
                        "kind": "answer.render.delta",
                        "payload": {"delta": delta, "report_language": language},
                    }
            if event.get("type") == "response.audio":
                tts_chunks += 1
                if tts_first_ms is None and event.get("tts_first_ms") is not None:
                    tts_first_ms = event.get("tts_first_ms")
                    tts_span.set_attribute("tts_first_ms", tts_first_ms)
            yield event
        _finalize_billing_offer(
            session,
            capability_id=navigation_decision.capability_id,
            tts_chunks=tts_chunks,
        )
        if localization_task is not None:
            await localization_task
        if localization_queue is not None:
            while not localization_queue.empty():
                delta = localization_queue.get_nowait()
                last_event_seq += 1
                session.event_seq_by_turn[turn_id] = last_event_seq
                if delta is None:
                    localization_pending = False
                    yield {
                        "type": "turn.event",
                        "turn_id": turn_id,
                        "seq": last_event_seq,
                        "kind": "answer.render.completed",
                        "payload": {"report_language": language},
                    }
                else:
                    yield {
                        "type": "turn.event",
                        "turn_id": turn_id,
                        "seq": last_event_seq,
                        "kind": "answer.render.delta",
                        "payload": {"delta": delta, "report_language": language},
                    }
        tts_span.set_output({"chunks": tts_chunks, "tts_first_ms": tts_first_ms}).end()

        yield {
            "type": "turn.final",
            "turn_id": turn_id,
        }

        scheduler = getattr(session, "speech_scheduler", None)
        if scheduler is not None:
            trace.set_metric("speech_budget_skips", list(getattr(scheduler, "skipped", [])))

        # After TTS completes, suppress turn finalization for 1.5s to prevent
        # speaker→mic echo from immediately triggering a false follow-up turn.
        session.set_cooldown(1.5)
    except asyncio.CancelledError:
        trace.status = "cancelled"
        raise
    except (GatewayPolicyDenied, SensitiveInputDenied) as exc:
        # Policy DENY is a governed turn outcome, not an infrastructure failure.
        # An input denial retracts the rejected user message from server history;
        # the browser receives the same disposition in a typed event.
        guard_id = (
            "gateway.service_policy"
            if isinstance(exc, GatewayPolicyDenied)
            else "sensitive_input_tiering"
        )
        if not any(entry.guard_id == guard_id for entry in trace.guards.entries):
            report(
                trace.guards,
                guard_id,
                "fired",
                stage="input_transcript" if exc.is_input_denial else "routing",
                owner=(
                    "gateway"
                    if isinstance(exc, GatewayPolicyDenied)
                    else "application"
                ),
                phase=exc.phase or "on_call_or_result",
                resource=exc.resource,
                reason=(
                    f"{exc.policy_name}: policy denied "
                    f"{exc.phase or 'request_or_response'}"
                ),
            )
        input_removed = False
        if (
            exc.is_input_denial
            and session.history
            and session.history[-1].get("role") == "user"
            and session.history[-1].get("content") == trace.input_transcript
        ):
            session.history.pop()
            input_removed = True
        if getattr(session, "active_turn", None) is not None and exc.is_input_denial:
            session.active_turn.meta.pop("utterance", None)

        trace.status = "blocked"
        trace.error = None
        language = trace.language or session.config.language or "en-US"
        refusal = (
            fixed_phrase("sensitive.blocked", language=language)
            if isinstance(exc, SensitiveInputDenied)
            else refuse_speech(ErrorCode.UNSUPPORTED, language=language)
        )
        trace.output_text = refusal
        seq = int(session.event_seq_by_turn.get(turn_id, 0)) + 1
        session.event_seq_by_turn[turn_id] = seq
        yield {
            "type": "guardrail.denied",
            "turn_id": turn_id,
            "seq": seq,
            "policy_id": exc.policy_name,
            "phase": exc.phase or "unknown",
            "resource": exc.resource,
            "input_removed": input_removed,
            "message": refusal,
        }
        yield {"type": "response.text", "turn_id": turn_id, "text": refusal}
        admission = admit_speech_output(
            refusal,
            resource=str(getattr(bundle.tts, "tts_endpoint", "tts")),
            ledger=trace.guards,
        )
        async for event in stream_tts(
            bundle,
            session,
            turn_id,
            admission,
            language,
            trace=trace,
        ):
            yield event
        yield {
            "type": "turn.final",
            "turn_id": turn_id,
            "status": "blocked",
        }
        session.set_cooldown(1.5)
        return
    except Exception as exc:  # noqa: BLE001
        trace.status = "error"
        trace.error = repr(exc)
        raise
    finally:
        # Barge-in/cancellation must not leave the conversion endpoint streaming in
        # an orphan task after this turn has been superseded.
        if localization_task is not None and not localization_task.done():
            localization_task.cancel()
            await asyncio.gather(localization_task, return_exceptions=True)
        pop_inference_context(provenance_tokens)
        submit_trace(trace)
