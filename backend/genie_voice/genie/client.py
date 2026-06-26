"""Genie Conversation API client (Phase 2 analytics).

Thin wrapper over the Databricks SDK Genie API. The space is resolved BY NAME
(`databricks.genie_space_name`) - never a hardcoded id. Async two-step flow:
start a conversation, poll to completion, return text + generated SQL + rows.
"""
from __future__ import annotations

from typing import Any

from genie_voice.config import Settings, get_settings
from genie_voice.genie.space import find_space_ids


class GenieClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._space_id: str | None = None

    @staticmethod
    def _looks_like_stale_space_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        return (
            "resource_does_not_exist" in msg
            or "not found" in msg
            or "trashed" in msg
            or "space" in msg and "exist" in msg
        )

    def _resolve_space_id(self, force_refresh: bool = False) -> str | None:
        if self._space_id and not force_refresh:
            return self._space_id
        try:
            from genie_voice.databricks.client import get_workspace_client

            client = get_workspace_client(self.settings)
            matches = find_space_ids(client, self.settings.databricks.genie_space_name)
            if len(matches) > 1:
                raise RuntimeError(
                    "Multiple Genie spaces share the configured name; run "
                    "`python -m genie_voice.genie.space` to reconcile duplicates: "
                    + ", ".join(matches)
                )
            self._space_id = matches[0] if matches else None
        except Exception:  # noqa: BLE001 - no workspace / not authenticated
            self._space_id = None
        return self._space_id

    @staticmethod
    def _extract_followups(msg: Any) -> list[str]:
        """Genie's suggested follow-up questions (SDK shape varies by version)."""
        out: list[str] = []
        for source in (
            getattr(msg, "suggested_questions", None),
            getattr(msg, "suggestions", None),
        ):
            for item in source or []:
                content = (
                    getattr(item, "question", None)
                    or getattr(item, "content", None)
                    or (item if isinstance(item, str) else None)
                )
                if content:
                    out.append(str(content))
        return out

    def _query_result_rows(
        self, client: Any, space_id: str, conversation_id: str, message_id: str, attachment_id: str | None
    ) -> tuple[list | None, list[str]]:
        """Read rows via the attachment-scoped query-result endpoint.

        Falls back to the deprecated message-level call only if the installed SDK
        lacks the attachment-scoped method.
        """
        try:
            if attachment_id and hasattr(client.genie, "get_message_attachment_query_result"):
                result = client.genie.get_message_attachment_query_result(
                    space_id, conversation_id, message_id, attachment_id
                )
            else:
                result = client.genie.get_message_query_result(
                    space_id, conversation_id, message_id
                )
            sr = getattr(result, "statement_response", None) if result else None
            inner = getattr(sr, "result", None) if sr else None
            rows = getattr(inner, "data_array", None) if inner else None
            manifest = getattr(sr, "manifest", None) if sr else None
            schema = getattr(manifest, "schema", None) if manifest else None
            columns = [
                str(c.name)
                for c in (getattr(schema, "columns", None) or [])
                if getattr(c, "name", None)
            ]
            return rows, columns
        except Exception:  # noqa: BLE001
            return None, []

    def _extract_message(self, client: Any, space_id: str, msg: Any, question: str) -> dict[str, Any]:
        """Shape a Genie message into NL-first output (description preferred over SQL)."""
        text, query, description, attachment_id = None, None, None, None
        for att in (msg.attachments or []):
            if getattr(att, "text", None):
                text = att.text.content
            if getattr(att, "query", None):
                query = getattr(att.query, "query", None)
                description = getattr(att.query, "description", None)
                attachment_id = getattr(att, "attachment_id", None)

        rows, columns = (None, [])
        if msg.conversation_id and msg.id:
            rows, columns = self._query_result_rows(
                client, space_id, msg.conversation_id, msg.id, attachment_id
            )

        return {
            "question": question,
            "answer": text,
            "description": description,
            "sql": query,  # debug only; UI should prefer `description`
            "rows": rows,
            "columns": columns,
            "suggested_followups": self._extract_followups(msg),
            "conversation_id": msg.conversation_id,
            "message_id": msg.id,
        }

    def ask(self, question: str, conversation_id: str | None = None) -> dict[str, Any]:
        """Ask Genie a question.

        When `conversation_id` is provided, the question is sent as a follow-up in
        that existing conversation (Genie retains context); otherwise a new
        conversation is started. Scope a conversation to a single session.
        """
        space_id = self._resolve_space_id(force_refresh=False)
        if not space_id:
            raise RuntimeError(
                "No resolvable Genie space. Authenticate to Databricks and run "
                "`python -m genie_voice.genie.space`."
            )

        try:
            from genie_voice.databricks.client import get_workspace_client

            client = get_workspace_client(self.settings)
            if conversation_id:
                msg = client.genie.create_message_and_wait(space_id, conversation_id, question)
            else:
                msg = client.genie.start_conversation_and_wait(space_id, question)
            return self._extract_message(client, space_id, msg, question)
        except Exception as exc:  # noqa: BLE001
            # Core resilience: deployments recreate spaces by name, so a cached id
            # can become stale. Re-resolve by name once and retry (new conversation
            # only - a follow-up id is meaningless against a refreshed space).
            if self._looks_like_stale_space_error(exc) and not conversation_id:
                retry_space = self._resolve_space_id(force_refresh=True)
                if retry_space and retry_space != space_id:
                    try:
                        from genie_voice.databricks.client import get_workspace_client

                        client = get_workspace_client(self.settings)
                        msg = client.genie.start_conversation_and_wait(retry_space, question)
                        return self._extract_message(client, retry_space, msg, question)
                    except Exception as retry_exc:  # noqa: BLE001
                        raise RuntimeError(f"Genie query failed after space refresh: {retry_exc}") from retry_exc
            raise RuntimeError(f"Genie query failed: {exc}") from exc
