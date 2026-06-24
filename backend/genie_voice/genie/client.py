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

    def ask(self, question: str) -> dict[str, Any]:
        space_id = self._resolve_space_id(force_refresh=False)
        if not space_id:
            raise RuntimeError(
                "No resolvable Genie space. Authenticate to Databricks and run "
                "`python -m genie_voice.genie.space`."
            )

        try:
            from genie_voice.databricks.client import get_workspace_client

            client = get_workspace_client(self.settings)
            msg = client.genie.start_conversation_and_wait(space_id, question)

            text, query, rows = None, None, None
            columns: list[str] = []
            for att in (msg.attachments or []):
                if getattr(att, "text", None):
                    text = att.text.content
                if getattr(att, "query", None):
                    query = att.query.query
            if msg.conversation_id and msg.id:
                try:
                    result = client.genie.get_message_query_result(
                        space_id, msg.conversation_id, msg.id
                    )
                    rows = result.statement_response.result.data_array if result else None
                    manifest = getattr(result.statement_response, "manifest", None) if result else None
                    schema = getattr(manifest, "schema", None) if manifest else None
                    columns = [
                        str(c.name)
                        for c in (getattr(schema, "columns", None) or [])
                        if getattr(c, "name", None)
                    ]
                except Exception:  # noqa: BLE001
                    rows = None
                    columns = []

            return {"question": question, "answer": text, "sql": query, "rows": rows, "columns": columns}
        except Exception as exc:  # noqa: BLE001
            # Core resilience: deployments recreate spaces by name, so a cached id
            # can become stale. Re-resolve by name once and retry.
            if self._looks_like_stale_space_error(exc):
                retry_space = self._resolve_space_id(force_refresh=True)
                if retry_space and retry_space != space_id:
                    try:
                        from genie_voice.databricks.client import get_workspace_client

                        client = get_workspace_client(self.settings)
                        msg = client.genie.start_conversation_and_wait(retry_space, question)
                        text, query, rows = None, None, None
                        columns = []
                        for att in (msg.attachments or []):
                            if getattr(att, "text", None):
                                text = att.text.content
                            if getattr(att, "query", None):
                                query = att.query.query
                        if msg.conversation_id and msg.id:
                            try:
                                result = client.genie.get_message_query_result(
                                    retry_space, msg.conversation_id, msg.id
                                )
                                rows = result.statement_response.result.data_array if result else None
                                manifest = getattr(result.statement_response, "manifest", None) if result else None
                                schema = getattr(manifest, "schema", None) if manifest else None
                                columns = [
                                    str(c.name)
                                    for c in (getattr(schema, "columns", None) or [])
                                    if getattr(c, "name", None)
                                ]
                            except Exception:  # noqa: BLE001
                                rows = None
                                columns = []
                        return {
                            "question": question,
                            "answer": text,
                            "sql": query,
                            "rows": rows,
                            "columns": columns,
                        }
                    except Exception as retry_exc:  # noqa: BLE001
                        raise RuntimeError(f"Genie query failed after space refresh: {retry_exc}") from retry_exc
            raise RuntimeError(f"Genie query failed: {exc}") from exc
