"""AgentRuntime protocol + adapter over blocking ``respond_with_tools``."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Protocol

from .cancellation import CancellationToken
from .events import AgentEvent, AgentEventKind, EventSequencer


@dataclass
class AgentGoal:
    """Mutable goal for the active turn (amend path may update fields)."""

    utterance: str
    language: str = "en"
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentContext:
    turn_id: int
    history: list[dict[str, str]] = field(default_factory=list)
    tool_ctx: Any = None
    context: str | None = None
    system_prompt: str | None = None
    tools_override: list[dict[str, Any]] | None = None
    tool_runner: Any = None
    tool_choice: Any = "auto"
    trace: Any = None
    extra: dict[str, Any] = field(default_factory=dict)


class AgentRuntime(Protocol):
    async def run(
        self,
        goal: AgentGoal,
        context: AgentContext,
        cancellation: CancellationToken,
    ) -> AsyncIterator[AgentEvent]: ...


RespondFn = Callable[..., tuple[str, list[dict[str, Any]]]]


class RespondWithToolsAdapter:
    """Wraps ``ServingClient.respond_with_tools`` as a live AgentEvent stream.

    Tool invocations emit ``action.started`` / ``action.completed`` *before*
    ``answer.final``, so UI/progress clients see work before final text.
    """

    def __init__(self, respond_fn: RespondFn) -> None:
        self._respond = respond_fn

    async def run(
        self,
        goal: AgentGoal,
        context: AgentContext,
        cancellation: CancellationToken,
    ) -> AsyncIterator[AgentEvent]:
        import asyncio

        seq = EventSequencer(context.turn_id)
        yield seq.emit(
            AgentEventKind.TURN_ACCEPTED,
            {"utterance": goal.utterance, **dict(goal.meta)},
        )

        cancellation.raise_if_cancelled()

        def _run() -> tuple[str, list[dict[str, Any]]]:
            return self._respond(
                goal.utterance,
                language=goal.language,
                context=context.context,
                tool_ctx=context.tool_ctx,
                history=context.history,
                trace=context.trace,
                system_prompt=context.system_prompt,
                tools_override=context.tools_override,
                tool_runner=context.tool_runner,
                tool_choice=context.tool_choice,
            )

        text, invocations = await asyncio.to_thread(_run)
        cancellation.raise_if_cancelled()

        for inv in invocations:
            name = str(inv.get("name") or "")
            yield seq.emit(
                AgentEventKind.ACTION_STARTED,
                {"name": name, "arguments": inv.get("arguments")},
            )
            yield seq.emit(
                AgentEventKind.ACTION_COMPLETED,
                {"name": name, "result": inv.get("result")},
            )

        yield seq.emit(
            AgentEventKind.ANSWER_FINAL,
            {"text": text, "display_only": bool(invocations)},
        )
        yield seq.emit(AgentEventKind.TURN_COMPLETED, {"text": text})


class LiveToolRespondAdapter:
    """Like RespondWithToolsAdapter but emits tool events during the tool loop.

    Requires a ``respond_fn`` that accepts ``on_tool`` callback
    ``(phase, name, arguments, result) -> None``. When the serving client lacks
    that hook, use :class:`RespondWithToolsAdapter` instead.
    """

    def __init__(self, respond_fn: RespondFn) -> None:
        self._respond = respond_fn

    async def run(
        self,
        goal: AgentGoal,
        context: AgentContext,
        cancellation: CancellationToken,
    ) -> AsyncIterator[AgentEvent]:
        import asyncio

        seq = EventSequencer(context.turn_id)
        pending: list[AgentEvent] = []

        yield seq.emit(
            AgentEventKind.TURN_ACCEPTED,
            {"utterance": goal.utterance, **dict(goal.meta)},
        )

        def on_tool(
            phase: str,
            name: str,
            arguments: dict[str, Any] | None = None,
            result: Any = None,
        ) -> None:
            if phase == "started":
                pending.append(
                    seq.emit(
                        AgentEventKind.ACTION_STARTED,
                        {"name": name, "arguments": arguments or {}},
                    )
                )
            elif phase == "completed":
                pending.append(
                    seq.emit(
                        AgentEventKind.ACTION_COMPLETED,
                        {"name": name, "arguments": arguments or {}, "result": result},
                    )
                )
                # Evidence is emitted as soon as the tool finishes, before the
                # model's final prose. The UI may display prose, but the speech
                # lane consumes only this structured object.
                from .pack_schema import evidence_from_tool_result

                evidence = evidence_from_tool_result(name, result)
                pending.append(
                    seq.emit(
                        AgentEventKind.EVIDENCE_AVAILABLE,
                        {"name": name, "evidence": evidence.as_dict()},
                    )
                )

        def _run() -> tuple[str, list[dict[str, Any]]]:
            kwargs: dict[str, Any] = {
                "language": goal.language,
                "context": context.context,
                "tool_ctx": context.tool_ctx,
                "history": context.history,
                "trace": context.trace,
                "system_prompt": context.system_prompt,
                "tools_override": context.tools_override,
                "tool_runner": context.tool_runner,
                "on_tool": on_tool,
                "tool_choice": context.tool_choice,
            }
            return self._respond(goal.utterance, **kwargs)

        task = asyncio.create_task(asyncio.to_thread(_run))
        while not task.done():
            cancellation.raise_if_cancelled()
            while pending:
                yield pending.pop(0)
            await asyncio.sleep(0.02)

        text, _invocations = task.result()
        while pending:
            yield pending.pop(0)

        cancellation.raise_if_cancelled()
        yield seq.emit(
            AgentEventKind.ANSWER_FINAL,
            {"text": text, "display_only": bool(_invocations)},
        )
        yield seq.emit(AgentEventKind.TURN_COMPLETED, {"text": text})
