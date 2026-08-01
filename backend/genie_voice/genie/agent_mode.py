"""Genie Agent-mode API client (the async, deep "why" lane).

Agent mode runs multi-step reasoning: it plans, runs SQL, iterates on results, and
returns a cited report with supporting tables. Results stream as Server-Sent
Events from:

    POST /api/2.0/genie/agents/{agent_id}/responses

where ``agent_id`` is the 32-char Genie Agent (space) id. This is a Beta API: a
workspace admin must enable "Agent Mode APIs for Genie Agents" under Previews.

This module is the reusable client for Phase 2's async lane and doubles as a
live validator: ``python -m genie_voice.genie.agent_mode`` sends the two anchor
questions to the card Genie Agent and diffs the report against GROUND_TRUTH.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from genie_voice.config import Settings, get_settings

logger = logging.getLogger("genie_voice.agent_mode")


@dataclass
class AgentModeResult:
    status: str = "unknown"
    conversation_id: str | None = None
    report_text: str = ""
    tables: list[dict[str, Any]] = field(default_factory=list)  # {sql, columns, preview_rows, markdown}
    reasoning: list[str] = field(default_factory=list)
    sql_calls: list[str] = field(default_factory=list)
    error: dict[str, Any] | None = None
    raw_output: list[dict[str, Any]] = field(default_factory=list)

    def haystack(self) -> str:
        """All returned text + table rows, comma-stripped, for numeric containment."""
        blob = json.dumps(
            {"report": self.report_text, "tables": self.tables, "sql": self.sql_calls},
            default=str,
        )
        return re.sub(r"(?<=\d),(?=\d)", "", blob)


class GenieAgentModeClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def _host(self) -> str:
        host = self.settings.databricks_host.rstrip("/")
        if not host:
            raise RuntimeError("databricks host is not configured.")
        return host

    def _auth_headers(self) -> dict[str, str]:
        from genie_voice.databricks.client import get_workspace_client

        client = get_workspace_client(self.settings)
        # Config.authenticate() yields the ready-to-use Authorization header for
        # whatever auth the SDK resolved (U2M OAuth locally, SP on the app).
        headers = dict(client.config.authenticate() or {})
        headers["Content-Type"] = "application/json"
        headers["Accept"] = "text/event-stream"
        return headers

    def resolve_agent_id(self, name: str | None = None) -> str | None:
        """The Agent id IS the Genie space id (resolved by configured name)."""
        from genie_voice.databricks.client import get_workspace_client
        from genie_voice.genie.space import find_space_ids

        name = name or self.settings.card_issuer.genie_space_name
        client = get_workspace_client(self.settings)
        matches = find_space_ids(client, name)
        return matches[0] if matches else None

    def ask(
        self,
        question: str,
        *,
        agent_id: str | None = None,
        space_name: str | None = None,
        conversation_id: str | None = None,
        enable_viz: bool = False,
        read_timeout_s: float = 420.0,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> AgentModeResult:
        """Send one question to Agent mode and consume the SSE stream to the end.

        The target Agent (Genie space) is resolved by NAME: pass ``space_name`` to
        pick a specific domain's space (e.g. the card space) so the caller isn't
        pinned to any default; ``agent_id`` (a resolved id) still wins if given.

        The question is asked in ENGLISH and the report comes back in English by
        design. Asking the agent to write its narrative in the caller's language
        used to work for some languages and silently killed the run for others:
        with "write your report in Hindi" appended, the agent planned, ran all its
        SQL, emitted "now I'll prepare the final response in Hindi" — and then
        closed the stream with no terminal event, three runs out of three, while
        English, Spanish and Thai completed. English is also the fastest path
        (~29s vs 43-54s). Localizing the report is therefore the caller's job, not
        the agent's: see ``realtime_api.deep_dive``.

        ``on_event`` (optional) is called with a small normalized progress dict as
        each SSE item finalizes, so a caller can surface live reasoning while the
        agent works: {"kind": "started"|"reasoning"|"sql"|"report"|"error", ...}.
        Never blocks on it and swallows callback errors (surfacing is best-effort).
        """
        import requests

        def _emit(ev: dict[str, Any]) -> None:
            if on_event is None:
                return
            try:
                on_event(ev)
            except Exception:  # noqa: BLE001
                pass

        agent_id = agent_id or self.resolve_agent_id(space_name)
        if not agent_id:
            raise RuntimeError("No Genie Agent id; run genie.space_card to create the Agent.")

        url = f"{self._host()}/api/2.0/genie/agents/{agent_id}/responses"
        body: dict[str, Any] = {
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": question}],
                }
            ],
            "enable_viz": enable_viz,
        }
        if conversation_id:
            body["conversation_id"] = conversation_id

        result = AgentModeResult()
        with requests.post(
            url, headers=self._auth_headers(), json=body, stream=True,
            timeout=(30.0, read_timeout_s),
        ) as resp:
            if resp.status_code >= 400:
                # HTTP error before the stream starts (e.g. FEATURE_DISABLED in Beta).
                try:
                    payload = resp.json()
                except Exception:  # noqa: BLE001
                    payload = {"error": {"message": resp.text[:500]}}
                result.status = "failed"
                result.error = payload.get("error", payload)
                result.error = {**result.error, "http_status": resp.status_code}
                _emit({"kind": "error", "error": result.error, "status": result.status})
                return result

            seen: list[str] = []
            for event in _iter_sse(resp):
                etype = event.get("type")
                seen.append(str(etype))
                if etype == "response.created":
                    result.conversation_id = (event.get("response") or {}).get("conversation_id")
                    _emit({"kind": "started", "conversation_id": result.conversation_id})
                elif etype == "response.output_item.done":
                    # Each reasoning step / SQL call finalizes here — stream it out.
                    prog = _item_progress(event.get("item") or {})
                    if prog:
                        _emit(prog)
                elif etype in ("response.completed", "response.failed"):
                    response = event.get("response") or {}
                    result.status = response.get("status", etype.split(".")[-1])
                    result.conversation_id = response.get("conversation_id") or result.conversation_id
                    result.error = response.get("error")
                    result.raw_output = response.get("output") or []
                    _parse_output(result, result.raw_output)
                    if result.error:
                        _emit({"kind": "error", "error": result.error, "status": result.status})
                    else:
                        _emit({
                            "kind": "report",
                            "status": result.status,
                            "report": result.report_text,
                            "tables": result.tables,
                            "sql": result.sql_calls,
                            "reasoning": result.reasoning,
                        })
                    break
            else:
                # The stream ended with no response.completed / response.failed.
                # Previously this returned silently: the caller emitted a bare
                # `done`, the UI said "ended without a result", and nothing
                # anywhere recorded WHY — so the same failure was undiagnosable
                # every time. Say what happened, and name the events we did see.
                result.status = "incomplete"
                result.error = {
                    "message": (
                        "Genie Agent Mode closed the stream without a terminal "
                        "response event"
                    ),
                    "events_seen": seen[-8:],
                }
                logger.warning(
                    "agent-mode stream ended with no terminal event; saw %s", seen or "nothing"
                )
                _emit({"kind": "error", "error": result.error, "status": result.status})
        return result


def _iter_sse(resp) -> Any:
    """Yield decoded SSE ``data`` JSON objects (buffers multi-line data blocks)."""
    data_lines: list[str] = []
    for raw in resp.iter_lines(decode_unicode=True):
        if raw is None:
            continue
        line = raw.rstrip("\r")
        if line == "":  # event boundary
            if data_lines:
                payload = "\n".join(data_lines)
                data_lines = []
                try:
                    yield json.loads(payload)
                except json.JSONDecodeError:
                    pass
            continue
        if line.startswith(":"):  # comment/heartbeat
            continue
        if line.startswith("data:"):
            data_lines.append(line[len("data:"):].lstrip())
        # 'event:' / 'id:' lines are ignored; type is carried inside the data JSON.
    if data_lines:  # flush trailing block with no terminating blank line
        try:
            yield json.loads("\n".join(data_lines))
        except json.JSONDecodeError:
            pass


def _item_progress(item: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize a finalized SSE output item into a small live-progress dict.

    Recognizes the two step kinds worth showing while the agent works: its
    reasoning text and the SQL it runs. Returns None for items with nothing to
    surface (e.g. the final assistant message, which arrives via the report).
    """
    itype = item.get("type")
    if itype == "reasoning":
        texts = [c.get("text") for c in (item.get("content") or []) if c.get("text")]
        if texts:
            return {"kind": "reasoning", "text": " ".join(t for t in texts if t)}
    elif itype == "function_call":
        args = item.get("arguments")
        if isinstance(args, str):
            try:
                sql = json.loads(args).get("sql")
            except json.JSONDecodeError:
                sql = None
            if sql:
                return {"kind": "sql", "sql": sql}
    return None


def _parse_output(result: AgentModeResult, output: list[dict[str, Any]]) -> None:
    """Extract report text, tables, reasoning and SQL from the final output items."""
    report_parts: list[str] = []
    for item in output:
        itype = item.get("type")
        if itype == "reasoning":
            for chunk in item.get("content") or []:
                if chunk.get("text"):
                    result.reasoning.append(chunk["text"])
        elif itype == "function_call":
            args = item.get("arguments")
            if isinstance(args, str):
                try:
                    sql = json.loads(args).get("sql")
                    if sql:
                        result.sql_calls.append(sql)
                except json.JSONDecodeError:
                    pass
        elif itype == "message" and item.get("role") == "assistant":
            for chunk in item.get("content") or []:
                if chunk.get("type") != "output_text":
                    continue
                meta = chunk.get("metadata") or {}
                if meta.get("preview_rows") is not None:
                    result.tables.append({
                        "sql": meta.get("sql"),
                        "columns": [c.get("name") for c in (meta.get("columns") or [])],
                        "preview_rows": meta.get("preview_rows"),
                        "markdown": chunk.get("text"),
                    })
                    if meta.get("sql"):
                        result.sql_calls.append(meta["sql"])
                else:
                    report_parts.append(chunk.get("text") or "")
    # Drop [N](url) citation links down to [N] for readable plain text.
    text = "\n".join(p for p in report_parts if p)
    result.report_text = re.sub(r"\[(\d+)\]\((https?://[^)]+)\)", r"[\1]", text)


# --------------------------------------------------------------------------- #
# Live validation of the two anchor questions against the card Agent
# --------------------------------------------------------------------------- #
def _anchor_checks() -> list[dict[str, Any]]:
    from genie_voice.datagen.card.generators_card import GROUND_TRUTH

    si = GROUND_TRUTH["statement_insights"]; d = si["drivers"]
    ro = GROUND_TRUTH["rewards_optimizer"]; lk = ro["leakage"]
    return [
        {
            "label": "statement_insights (Agent mode)",
            "question": (
                "For cardholder CH-0001, why did the statement balance increase this cycle "
                "(cycle 2025-12) compared with their prior cycles? Investigate the transactions "
                "and break the increase down by driver with the dollar amount for each."
            ),
            "expected": [int(d["flight"]), int(d["foreign_trip_spend"]), int(d["annual_fee"]),
                         int(d["foreign_tx_fee"]), int(d["interest"])],
        },
        {
            "label": "rewards_optimizer (Agent mode)",
            "question": (
                "For cardholder CH-0002, where are they losing rewards points this cycle "
                "(cycle 2025-12) and why? Quantify the points missed by reason and give the total gap."
            ),
            "expected": [lk["dining_wrong_card"], lk["inactive_grocery_bonus"],
                         lk["reversed_points"], ro["points_gap"]],
        },
    ]


def main() -> int:
    settings = get_settings()
    client = GenieAgentModeClient(settings)
    agent_id = client.resolve_agent_id()
    if not agent_id:
        print(f"No Genie Agent named '{settings.card_issuer.genie_space_name}'. Run space_card first.")
        return 1
    print(f"Agent-mode validation against agent {agent_id} ('{settings.card_issuer.genie_space_name}')\n")

    passed = 0
    checks = _anchor_checks()
    for chk in checks:
        try:
            res = client.ask(chk["question"])
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] {chk['label']}: {exc}\n")
            continue
        if res.status != "completed":
            err = res.error or {}
            print(f"[FAIL] {chk['label']}: status={res.status} error={err}")
            if str(err.get("http_status")) == "404" or "FEATURE_DISABLED" in str(err):
                print("       -> Enable 'Agent Mode APIs for Genie Agents' in the workspace Previews page (Beta).")
            print()
            continue
        hay = res.haystack()
        missing = [n for n in chk["expected"] if not re.search(rf"(?<!\d){n}(?!\d)", hay)]
        ok = not missing
        passed += 1 if ok else 0
        print(f"[{'ok  ' if ok else 'FAIL'}] {chk['label']}")
        print(f"    Q: {chk['question']}")
        print(f"    report: {res.report_text[:600].strip()}")
        if res.sql_calls:
            print(f"    queries run: {len(res.sql_calls)}")
        if missing:
            print(f"    MISSING expected numbers: {missing}")
        print()

    print(f"{'PASS' if passed == len(checks) else 'PARTIAL/FAIL'}: {passed}/{len(checks)} agent-mode checks matched ground truth")
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
