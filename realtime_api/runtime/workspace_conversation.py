"""Where a governed workspace conversation lives across turns.

Genie One is a stateful chat: it resolves "Unity Catalog", "the second one" and
"why is EMEA so low?" against what it just said, recalls the caller's earlier
conversations on its own, and (in Beta) retains facts as durable memory. See
https://docs.databricks.com/aws/en/genie-one/chat#add-to-genie-ones-memory.

So this runtime deliberately does NOT interpret follow-ups. It holds one thing —
the handle to the conversation Genie One is keeping — and passes it back upstream
on every workspace turn. No wording is inspected, nothing is re-phrased, and no
context is reconstructed locally, because every one of those would be this process
guessing at state another system owns authoritatively.

THE REPLACEMENT POINT: when upstream memory makes the handle unnecessary, set
``upstream_memory=True`` (config ``realtime_voice.genie_one_upstream_memory``).
``handle`` then returns None, every question is asked standalone, and Genie One
supplies the continuity. Nothing else in the runtime changes: no router rules, no
prompts, no client events — this object is the only place that knows.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WorkspaceConversation:
    """The open Genie One conversation for one call, and nothing more."""

    # Set from what Genie One returns; never invented here.
    conversation_id: str | None = None
    # Turns this call has sent into the conversation (observability only).
    turns: int = 0
    # The last question asked upstream and the summary the caller heard back. Kept
    # so a turn can tell the model what conversation it is joining, NOT to rebuild
    # the context Genie One already holds.
    last_question: str = ""
    last_answer: str = ""
    # True once upstream memory removes the need to carry a handle at all.
    upstream_memory: bool = False

    @property
    def is_open(self) -> bool:
        """Whether a later turn has a conversation it can join."""
        return bool(self.conversation_id) and not self.upstream_memory

    @property
    def handle(self) -> str | None:
        """The conversation id to continue, or None to ask standalone."""
        return None if self.upstream_memory else (self.conversation_id or None)

    def bind(self, conversation_id: str | None) -> None:
        """Adopt the conversation Genie One reported for this turn."""
        if conversation_id:
            self.conversation_id = str(conversation_id)
            self.turns += 1

    def record(self, *, question: str, answer: str) -> None:
        self.last_question = (question or "").strip()
        self.last_answer = (answer or "").strip()

    def close(self) -> None:
        """Forget the handle. The conversation itself lives on in Databricks."""
        self.conversation_id = None
        self.turns = 0
        self.last_question = ""
        self.last_answer = ""

    def briefing(self, *, max_chars: int = 600) -> str:
        """One line telling a turn which conversation it is joining, or "".

        Deliberately thin: enough for the model to recognise that a reply belongs
        upstream, never a local reconstruction of the conversation.
        """
        if not self.is_open or not self.last_question:
            return ""
        answer = self.last_answer[:max_chars]
        lines = [
            "An open governed workspace conversation is being continued.",
            f"Last asked upstream: {self.last_question}",
        ]
        if answer:
            lines.append(f"What the caller heard back: {answer}")
        return "\n".join(lines)
