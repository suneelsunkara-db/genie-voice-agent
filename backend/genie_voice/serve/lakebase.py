"""Lakebase (Autoscaling) serving layer.

Lakebase = serverless Postgres on Databricks. We use it as the low-latency store
the Agent Assist UI reads:
  - `{lakebase.schema}.call_state`             : live enrichment per call
  - `{lakebase.schema}.live_call_utterances`   : live transcript turns for CDF
  - `{lakebase.schema}.call_facts`             : operational call metadata for CDF

Auth model (matches the app's U2M identity): instead of storing a Postgres
password, we MINT a short-lived Postgres OAuth token via the Lakebase
Autoscaling (Projects) API (`/api/2.0/postgres/credentials`, scoped to the
project's read-write compute endpoint) and connect as the running user. The
endpoint host is discovered from `/api/2.0/postgres/`.

When `lakebase.enabled` is false (e.g. offline deploy) it falls back to an
in-process store so the end-to-end local flow still works.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from datetime import UTC, datetime
from typing import Any

from genie_voice.config import Settings, get_settings
from genie_voice.databricks.warehouse_sql import warehouse_configured

_MEM: dict[str, dict[str, Any]] = {}
_MEM_EVENTS: dict[str, list[dict[str, Any]]] = {}
_MEM_ADJUSTMENTS: dict[str, list[dict[str, Any]]] = {}
_LOCK = threading.Lock()


def _pg_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _account_facts(
    customer_id: str,
    customer: dict[str, Any] | None,
    invoices: list[dict[str, Any]],
    payments: list[dict[str, Any]],
) -> dict[str, Any]:
    """Shape account facts + a small agent-assist summary the UI can show."""
    open_invoices = [i for i in invoices if str(i.get("status")) in ("open", "overdue", "disputed")]
    overdue = [i for i in invoices if str(i.get("status")) == "overdue"]
    declined = [p for p in payments if str(p.get("status")) == "declined"]
    return {
        "customer_id": customer_id,
        "found": customer is not None,
        "customer": customer,
        "invoices": invoices,
        "payments": payments,
        "summary": {
            "open_invoice_count": len(open_invoices),
            "overdue_invoice_count": len(overdue),
            "overdue_amount": round(sum(_to_float(i.get("amount")) for i in overdue), 2),
            "autopay_enabled": bool(customer.get("autopay_enabled")) if customer else None,
            "status": customer.get("status") if customer else None,
            "recent_declined_payments": len(declined),
            "issue_status": "open",
            "resolution_note": None,
            "resolved_at": None,
        },
    }


def _recompute_summary(facts: dict[str, Any]) -> None:
    invoices = list(facts.get("invoices") or [])
    payments = list(facts.get("payments") or [])
    customer = facts.get("customer") or {}
    open_invoices = [i for i in invoices if str(i.get("status")) in ("open", "overdue", "disputed")]
    overdue = [i for i in invoices if str(i.get("status")) == "overdue"]
    declined = [p for p in payments if str(p.get("status")) == "declined"]
    summary = facts.get("summary") or {}
    summary.update(
        {
            "open_invoice_count": len(open_invoices),
            "overdue_invoice_count": len(overdue),
            "overdue_amount": round(sum(_to_float(i.get("amount")) for i in overdue), 2),
            "autopay_enabled": bool(customer.get("autopay_enabled")) if customer else None,
            "status": customer.get("status") if customer else None,
            "recent_declined_payments": len(declined),
        }
    )
    facts["summary"] = summary


def _apply_resolution_status_overlay(
    facts: dict[str, Any], resolution: dict[str, Any] | None
) -> dict[str, Any]:
    """Attach live issue status metadata without simulating invoice mutations."""
    if not resolution:
        return facts
    summary = facts.get("summary") or {}
    summary["issue_status"] = str(resolution.get("status") or "open")
    summary["resolution_note"] = resolution.get("note")
    summary["resolved_at"] = resolution.get("resolved_at")
    facts["summary"] = summary
    return facts


def _apply_billing_adjustments(facts: dict[str, Any], adjustments: list[dict[str, Any]]) -> dict[str, Any]:
    if not adjustments:
        return facts
    by_invoice = {str(a.get("invoice_id")): a for a in adjustments if a.get("invoice_id")}
    invoices = []
    for inv in facts.get("invoices") or []:
        inv = dict(inv)
        adj = by_invoice.get(str(inv.get("invoice_id")))
        if not adj:
            invoices.append(inv)
            continue
        inv["amount"] = f"{float(adj.get('amount_after', inv.get('amount'))):.2f}"
        inv["late_fee"] = f"{float(adj.get('late_fee_after', inv.get('late_fee'))):.2f}"
        inv["status"] = adj.get("status_after", inv.get("status"))
        inv["resolution_status"] = "closed"
        inv["resolution_updated_at"] = adj.get("applied_at")
        invoices.append(inv)
    facts["invoices"] = invoices
    _recompute_summary(facts)
    return facts


def _apply_resolution_overlay(facts: dict[str, Any], resolution: dict[str, Any] | None) -> dict[str, Any]:
    """Legacy overlay hook — resolution metadata only; billing uses persisted adjustments."""
    return _apply_resolution_status_overlay(facts, resolution)


class LakebaseServing:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.enabled = self.settings.lakebase.enabled
        self._cred: dict[str, Any] | None = None  # cached {host,user,token,exp}

    # ---- credential resolution -------------------------------------------- #
    def _credentials(self) -> dict[str, Any]:
        """Resolve (host, port, dbname, user, password) by discovering the instance
        and minting a short-lived Postgres OAuth token via the SDK."""
        lb = self.settings.lakebase
        # Reuse a still-valid minted token (tokens last ~1h; refresh at 50 min).
        if self._cred and self._cred["exp"] > time.time():
            return self._cred["value"]

        from genie_voice.databricks.client import current_user, get_workspace_client

        client = get_workspace_client(self.settings)
        endpoint, host = self._resolve_endpoint(client, lb.instance)
        # Lakebase Autoscaling (Projects API): mint a short-lived Postgres OAuth
        # token scoped to the read-write compute endpoint. Use the REST endpoint
        # directly because serverless jobs may bundle an older SDK without
        # WorkspaceClient.postgres.
        cred = client.api_client.do(
            "POST",
            "/api/2.0/postgres/credentials",
            body={"endpoint": endpoint},
            headers={"Accept": "application/json", "Content-Type": "application/json"},
        )
        user = self.settings.databricks.run_as or current_user(client)
        value = {
            "host": host,
            "port": lb.port,
            "dbname": lb.database,
            "user": user,
            "password": cred["token"],
        }
        self._cred = {"value": value, "exp": time.time() + 50 * 60}
        return value

    @staticmethod
    def _resolve_endpoint(client, instance: str) -> tuple[str, str]:
        """Resolve a Lakebase *project* (by id or display name) to its default
        branch's read-write compute endpoint. Returns (endpoint_resource_name,
        host) via the `/api/2.0/postgres/` (Lakebase Autoscaling) API."""
        ac = client.api_client
        projects = ac.do("GET", "/api/2.0/postgres/projects").get("projects", []) or []
        proj = next(
            (
                p for p in projects
                if instance in (p.get("project_id"), (p.get("status") or {}).get("display_name"))
                or p.get("project_id") == instance.replace("_", "-")
            ),
            None,
        )
        if not proj:
            raise RuntimeError(
                f"Lakebase project '{instance}' not found via /api/2.0/postgres/projects"
            )
        pid = proj["project_id"]
        branches = ac.do(
            "GET", f"/api/2.0/postgres/projects/{pid}/branches"
        ).get("branches", []) or []
        branch = next((b for b in branches if (b.get("status") or {}).get("default")), None) \
            or (branches[0] if branches else None)
        if not branch:
            raise RuntimeError(f"No branches for Lakebase project '{pid}'")
        bid = branch["branch_id"]
        eps = ac.do(
            "GET", f"/api/2.0/postgres/projects/{pid}/branches/{bid}/endpoints"
        ).get("endpoints", []) or []
        ep = next(
            (e for e in eps
             if (e.get("status") or {}).get("endpoint_type") == "ENDPOINT_TYPE_READ_WRITE"),
            None,
        ) or (eps[0] if eps else None)
        if not ep:
            raise RuntimeError(f"No compute endpoints for Lakebase branch '{bid}'")
        host = (((ep.get("status") or {}).get("hosts")) or {}).get("host")
        if not host:
            raise RuntimeError(f"Endpoint for '{pid}' has no host yet (still starting?)")
        return ep["name"], host

    def _conn(self):
        import psycopg

        c = self._credentials()
        return psycopg.connect(
            host=c["host"], port=c["port"], dbname=c["dbname"],
            user=c["user"], password=c["password"], sslmode="require", autocommit=True,
        )

    def _table(self, table: str) -> str:
        return f"{_pg_ident(self.settings.lakebase.schema_name)}.{_pg_ident(table)}"

    def _ensure_serving_schema(self, cur) -> None:
        cur.execute(f"CREATE SCHEMA IF NOT EXISTS {_pg_ident(self.settings.lakebase.schema_name)}")

    def ensure_schema(self) -> None:
        if not self.enabled:
            return
        state_tbl = self._table(self.settings.lakebase.serving_table)
        utterances_tbl = self._table(self.settings.lakebase.live_utterances_table)
        facts_tbl = self._table("call_facts")
        events_tbl = self._table("resolution_events")
        adjustments_tbl = self._table("billing_adjustments")
        with self._conn() as conn, conn.cursor() as cur:
            self._ensure_serving_schema(cur)
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {state_tbl} (
                    call_id     TEXT PRIMARY KEY,
                    customer_id TEXT,
                    state       JSONB,
                    updated_at  TIMESTAMPTZ DEFAULT now()
                )
                """
            )
            cur.execute(f"ALTER TABLE {state_tbl} REPLICA IDENTITY FULL")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {utterances_tbl} (
                    utterance_id TEXT PRIMARY KEY,
                    call_id      TEXT NOT NULL,
                    turn_index   INTEGER NOT NULL,
                    channel      INTEGER,
                    speaker_role TEXT,
                    start_sec    DOUBLE PRECISION,
                    end_sec      DOUBLE PRECISION,
                    text         TEXT,
                    confidence   DOUBLE PRECISION,
                    updated_at   TIMESTAMPTZ DEFAULT now(),
                    UNIQUE (call_id, turn_index)
                )
                """
            )
            cur.execute(f"ALTER TABLE {utterances_tbl} REPLICA IDENTITY FULL")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {facts_tbl} (
                    call_id         TEXT PRIMARY KEY,
                    customer_id     TEXT NOT NULL,
                    agent_id        TEXT,
                    call_ts         TIMESTAMPTZ,
                    duration_sec    INTEGER,
                    csat            INTEGER,
                    audio_path      TEXT,
                    transcript_path TEXT,
                    updated_at      TIMESTAMPTZ DEFAULT now()
                )
                """
            )
            cur.execute(f"ALTER TABLE {facts_tbl} REPLICA IDENTITY FULL")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {events_tbl} (
                    event_id     TEXT PRIMARY KEY,
                    call_id      TEXT NOT NULL,
                    event_type   TEXT NOT NULL,
                    issue_status TEXT,
                    note         TEXT,
                    actions      JSONB,
                    created_at   TIMESTAMPTZ DEFAULT now()
                )
                """
            )
            cur.execute(f"ALTER TABLE {events_tbl} REPLICA IDENTITY FULL")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {adjustments_tbl} (
                    adjustment_id TEXT PRIMARY KEY,
                    call_id       TEXT NOT NULL,
                    customer_id   TEXT NOT NULL,
                    invoice_id    TEXT NOT NULL,
                    waiver_applied BOOLEAN NOT NULL DEFAULT false,
                    payment_plan_applied BOOLEAN NOT NULL DEFAULT false,
                    amount_before  NUMERIC(10,2) NOT NULL,
                    late_fee_before NUMERIC(10,2) NOT NULL,
                    status_before  TEXT NOT NULL,
                    amount_after   NUMERIC(10,2) NOT NULL,
                    late_fee_after NUMERIC(10,2) NOT NULL,
                    status_after   TEXT NOT NULL,
                    applied_at     TIMESTAMPTZ DEFAULT now(),
                    reverted_at    TIMESTAMPTZ
                )
                """
            )
            cur.execute(f"ALTER TABLE {adjustments_tbl} REPLICA IDENTITY FULL")

    # ---- live call state -------------------------------------------------- #
    def upsert_call_state(self, call_id: str, customer_id: str | None, state: dict) -> None:
        if not self.enabled:
            with _LOCK:
                _MEM[call_id] = {"call_id": call_id, "customer_id": customer_id, "state": state}
            return
        tbl = self._table(self.settings.lakebase.serving_table)
        with self._conn() as conn, conn.cursor() as cur:
            self._ensure_serving_schema(cur)
            cur.execute(
                f"""
                INSERT INTO {tbl} (call_id, customer_id, state, updated_at)
                VALUES (%s, %s, %s, now())
                ON CONFLICT (call_id)
                DO UPDATE SET customer_id = EXCLUDED.customer_id,
                              state = EXCLUDED.state,
                              updated_at = now()
                """,
                (call_id, customer_id, json.dumps(state)),
            )
            self._replace_live_utterances(cur, call_id, state.get("utterances") or [])

    def get_call_state(self, call_id: str) -> dict[str, Any] | None:
        if not self.enabled:
            with _LOCK:
                return _MEM.get(call_id)
        tbl = self._table(self.settings.lakebase.serving_table)
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(
                f"SELECT call_id, customer_id, state FROM {tbl} WHERE call_id = %s",
                (call_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            return {"call_id": row[0], "customer_id": row[1], "state": row[2]}

    # ---- account facts (UC reference tables) ------------------------------ #
    def load_account_facts_source(self, customer_id: str) -> dict[str, Any]:
        """UC/local account facts without persisted billing adjustments applied."""
        return self._load_account_facts_source(customer_id)

    def _load_account_facts_source(self, customer_id: str) -> dict[str, Any]:
        if not self.enabled:
            return self._account_facts_local(customer_id)
        try:
            return self._account_facts_uc(customer_id)
        except Exception:
            return self._account_facts_local(customer_id)

    def get_account_facts(self, customer_id: str) -> dict[str, Any]:
        """Serve account facts from UC/local source merged with persisted billing adjustments."""
        facts = self._load_account_facts_source(customer_id)
        return _apply_billing_adjustments(facts, self.list_billing_adjustments(customer_id))

    def get_call_account_facts(self, call_id: str) -> dict[str, Any]:
        state = self.get_call_state(call_id)
        if not state:
            return {"customer_id": None, "found": False, "summary": {"issue_status": "open"}}
        customer_id = state.get("customer_id")
        if not customer_id:
            return {"customer_id": None, "found": False, "summary": {"issue_status": "open"}}
        facts = self.get_account_facts(customer_id)
        resolution = ((state.get("state") or {}).get("resolution") or {})
        return _apply_resolution_status_overlay(facts, resolution)

    def _serving_table(self, table: str) -> str:
        """Postgres identifier for a Lakebase-native serving table."""
        return self._table(table)

    def _replace_live_utterances(self, cur, call_id: str, utterances: list[dict[str, Any]]) -> None:
        tbl = self._table(self.settings.lakebase.live_utterances_table)
        cur.execute(f"DELETE FROM {tbl} WHERE call_id = %s", (call_id,))
        for idx, item in enumerate(utterances):
            speaker = item.get("speaker") or item.get("speaker_role")
            turn_index = int(item.get("turn_index", idx))
            channel = item.get("channel", speaker if isinstance(speaker, int) else None)
            utterance_id = item.get("utterance_id") or f"{call_id}-{turn_index}"
            cur.execute(
                f"""
                INSERT INTO {tbl}
                  (utterance_id, call_id, turn_index, channel, speaker_role,
                   start_sec, end_sec, text, confidence, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, now())
                ON CONFLICT (utterance_id)
                DO UPDATE SET call_id = EXCLUDED.call_id,
                              turn_index = EXCLUDED.turn_index,
                              channel = EXCLUDED.channel,
                              speaker_role = EXCLUDED.speaker_role,
                              start_sec = EXCLUDED.start_sec,
                              end_sec = EXCLUDED.end_sec,
                              text = EXCLUDED.text,
                              confidence = EXCLUDED.confidence,
                              updated_at = now()
                """,
                (
                    utterance_id,
                    call_id,
                    turn_index,
                    int(channel) if channel is not None else None,
                    str(speaker) if speaker is not None else None,
                    item.get("start_sec"),
                    item.get("end_sec"),
                    item.get("text"),
                    item.get("confidence"),
                ),
            )

    def upsert_call_fact(self, fact: dict[str, Any]) -> None:
        if not self.enabled:
            return
        tbl = self._table("call_facts")
        with self._conn() as conn, conn.cursor() as cur:
            self._ensure_serving_schema(cur)
            cur.execute(
                f"""
                INSERT INTO {tbl}
                  (call_id, customer_id, agent_id, call_ts, duration_sec, csat,
                   audio_path, transcript_path, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, now())
                ON CONFLICT (call_id)
                DO UPDATE SET customer_id = EXCLUDED.customer_id,
                              agent_id = EXCLUDED.agent_id,
                              call_ts = EXCLUDED.call_ts,
                              duration_sec = EXCLUDED.duration_sec,
                              csat = EXCLUDED.csat,
                              audio_path = EXCLUDED.audio_path,
                              transcript_path = EXCLUDED.transcript_path,
                              updated_at = now()
                """,
                (
                    fact.get("call_id"),
                    fact.get("customer_id"),
                    fact.get("agent_id"),
                    fact.get("call_ts"),
                    fact.get("duration_sec"),
                    fact.get("csat"),
                    fact.get("audio_path"),
                    fact.get("transcript_path"),
                ),
            )

    def replace_live_utterances(self, call_id: str, utterances: list[dict[str, Any]]) -> None:
        if not self.enabled:
            return
        with self._conn() as conn, conn.cursor() as cur:
            self._ensure_serving_schema(cur)
            self._replace_live_utterances(cur, call_id, utterances)

    def append_resolution_event(
        self,
        call_id: str,
        event_type: str,
        issue_status: str | None,
        note: str | None,
        actions: dict[str, Any] | None,
    ) -> bool:
        """Append a timeline row; skip exact duplicates of the latest entry."""
        latest = self.list_resolution_events(call_id, limit=1)
        if latest:
            prev = latest[0]
            if (
                prev.get("event_type") == event_type
                and str(prev.get("issue_status") or "") == str(issue_status or "")
                and str(prev.get("note") or "") == str(note or "")
            ):
                return False

        entry = {
            "event_id": f"{call_id}-{uuid.uuid4().hex[:12]}",
            "call_id": call_id,
            "event_type": event_type,
            "issue_status": issue_status,
            "note": note,
            "actions": actions or {},
            "created_at": datetime.now(UTC).isoformat(),
        }
        if not self.enabled:
            with _LOCK:
                _MEM_EVENTS.setdefault(call_id, []).append(entry)
            return True
        tbl = self._table("resolution_events")
        with self._conn() as conn, conn.cursor() as cur:
            self._ensure_serving_schema(cur)
            cur.execute(
                f"""
                INSERT INTO {tbl}
                  (event_id, call_id, event_type, issue_status, note, actions, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, now())
                """,
                (
                    entry["event_id"],
                    call_id,
                    event_type,
                    issue_status,
                    note,
                    json.dumps(actions or {}),
                ),
            )
        return True

    def clear_resolution_events(self, call_id: str) -> int:
        """Delete all resolution timeline rows for a call."""
        if not self.enabled:
            with _LOCK:
                removed = len(_MEM_EVENTS.get(call_id) or [])
                _MEM_EVENTS[call_id] = []
                return removed
        tbl = self._table("resolution_events")
        with self._conn() as conn, conn.cursor() as cur:
            self._ensure_serving_schema(cur)
            cur.execute(f"DELETE FROM {tbl} WHERE call_id = %s", (call_id,))
            return int(cur.rowcount or 0)

    def list_resolution_events(self, call_id: str, limit: int = 20) -> list[dict[str, Any]]:
        if not self.enabled:
            with _LOCK:
                events = list(_MEM_EVENTS.get(call_id) or [])
                return list(reversed(events[-limit:]))
        tbl = self._table("resolution_events")
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT event_id, call_id, event_type, issue_status, note, actions, created_at
                FROM {tbl}
                WHERE call_id = %s
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (call_id, limit),
            )
            rows = cur.fetchall()
            return [
                {
                    "event_id": r[0],
                    "call_id": r[1],
                    "event_type": r[2],
                    "issue_status": r[3],
                    "note": r[4],
                    "actions": r[5] or {},
                    "created_at": r[6].isoformat() if hasattr(r[6], "isoformat") else r[6],
                }
                for r in rows
            ]

    def list_billing_adjustments(
        self,
        customer_id: str,
        *,
        call_id: str | None = None,
        active_only: bool = True,
    ) -> list[dict[str, Any]]:
        if not customer_id:
            return []
        if not self.enabled:
            with _LOCK:
                rows = list(_MEM_ADJUSTMENTS.get(customer_id, []))
            if call_id:
                rows = [r for r in rows if r.get("call_id") == call_id]
            if active_only:
                rows = [r for r in rows if not r.get("reverted_at")]
            return rows
        tbl = self._table("billing_adjustments")
        with self._conn() as conn, conn.cursor() as cur:
            self._ensure_serving_schema(cur)
            clauses = ["customer_id = %s"]
            params: list[Any] = [customer_id]
            if call_id:
                clauses.append("call_id = %s")
                params.append(call_id)
            if active_only:
                clauses.append("reverted_at IS NULL")
            where = " AND ".join(clauses)
            cur.execute(
                f"""
                SELECT adjustment_id, call_id, customer_id, invoice_id,
                       waiver_applied, payment_plan_applied,
                       amount_before, late_fee_before, status_before,
                       amount_after, late_fee_after, status_after,
                       applied_at, reverted_at
                FROM {tbl}
                WHERE {where}
                ORDER BY applied_at DESC
                """,
                tuple(params),
            )
            rows = cur.fetchall()
            return [
                {
                    "adjustment_id": r[0],
                    "call_id": r[1],
                    "customer_id": r[2],
                    "invoice_id": r[3],
                    "waiver_applied": r[4],
                    "payment_plan_applied": r[5],
                    "amount_before": float(r[6]),
                    "late_fee_before": float(r[7]),
                    "status_before": r[8],
                    "amount_after": float(r[9]),
                    "late_fee_after": float(r[10]),
                    "status_after": r[11],
                    "applied_at": r[12].isoformat() if hasattr(r[12], "isoformat") else r[12],
                    "reverted_at": r[13].isoformat() if r[13] and hasattr(r[13], "isoformat") else r[13],
                }
                for r in rows
            ]

    def apply_billing_resolution(
        self,
        call_id: str,
        customer_id: str,
        resolution: dict[str, Any],
        account: dict[str, Any],
        *,
        adjustment: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        from genie_voice.assist.billing import prepare_billing_adjustment

        if adjustment is None:
            prepared = prepare_billing_adjustment(call_id, customer_id, resolution, account)
            if not prepared.get("ok"):
                return {"applied": False, "reason": prepared.get("reason")}
            adjustment = prepared["adjustment"]
        return self._persist_billing_adjustment(adjustment)

    def _persist_billing_adjustment(self, adjustment: dict[str, Any]) -> dict[str, Any]:
        adjustment_id = adjustment["adjustment_id"]
        call_id = adjustment["call_id"]
        customer_id = adjustment["customer_id"]

        uc_result: dict[str, Any] = {"ok": False, "skipped": True}
        if warehouse_configured(self.settings):
            try:
                from genie_voice.databricks import warehouse_sql

                uc_result = warehouse_sql.apply_billing_resolution_uc(self.settings, adjustment)
                uc_result["skipped"] = False
            except Exception as exc:  # noqa: BLE001
                return {"applied": False, "reason": f"uc_write_failed: {exc}"}
        elif self.enabled:
            return {
                "applied": False,
                "reason": "sql_warehouse_required_for_uc_billing_writes",
            }

        if not self.enabled:
            with _LOCK:
                rows = _MEM_ADJUSTMENTS.setdefault(customer_id, [])
                rows[:] = [r for r in rows if r.get("adjustment_id") != adjustment_id]
                rows.append(adjustment)
            return {"applied": True, "adjustment": adjustment, "uc": uc_result}

        tbl = self._table("billing_adjustments")
        with self._conn() as conn, conn.cursor() as cur:
            self._ensure_serving_schema(cur)
            cur.execute(
                f"""
                INSERT INTO {tbl}
                  (adjustment_id, call_id, customer_id, invoice_id,
                   waiver_applied, payment_plan_applied,
                   amount_before, late_fee_before, status_before,
                   amount_after, late_fee_after, status_after, applied_at)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,now())
                ON CONFLICT (adjustment_id) DO UPDATE SET
                  amount_after = EXCLUDED.amount_after,
                  late_fee_after = EXCLUDED.late_fee_after,
                  status_after = EXCLUDED.status_after,
                  reverted_at = NULL,
                  applied_at = now()
                """,
                (
                    adjustment_id,
                    call_id,
                    customer_id,
                    adjustment["invoice_id"],
                    adjustment["waiver_applied"],
                    adjustment["payment_plan_applied"],
                    adjustment["amount_before"],
                    adjustment["late_fee_before"],
                    adjustment["status_before"],
                    adjustment["amount_after"],
                    adjustment["late_fee_after"],
                    adjustment["status_after"],
                ),
            )
        return {"applied": True, "adjustment": adjustment, "uc": uc_result}

    def revert_billing_adjustments(self, call_id: str) -> dict[str, Any]:
        reverted: list[str] = []
        if not self.enabled:
            with _LOCK:
                for customer_id, rows in list(_MEM_ADJUSTMENTS.items()):
                    kept: list[dict[str, Any]] = []
                    for row in rows:
                        if row.get("call_id") == call_id and not row.get("reverted_at"):
                            if warehouse_configured(self.settings):
                                try:
                                    from genie_voice.databricks import warehouse_sql

                                    warehouse_sql.revert_billing_resolution_uc(self.settings, row)
                                except Exception:  # noqa: BLE001
                                    pass
                            reverted.append(str(row.get("adjustment_id")))
                        else:
                            kept.append(row)
                    _MEM_ADJUSTMENTS[customer_id] = kept
            return {"call_id": call_id, "reverted": reverted}

        tbl = self._table("billing_adjustments")
        with self._conn() as conn, conn.cursor() as cur:
            self._ensure_serving_schema(cur)
            cur.execute(
                f"""
                SELECT adjustment_id, call_id, customer_id, invoice_id,
                       amount_before, late_fee_before, status_before
                FROM {tbl}
                WHERE call_id = %s AND reverted_at IS NULL
                """,
                (call_id,),
            )
            rows = cur.fetchall()
            for row in rows:
                payload = {
                    "adjustment_id": row[0],
                    "call_id": row[1],
                    "customer_id": row[2],
                    "invoice_id": row[3],
                    "amount_before": float(row[4]),
                    "late_fee_before": float(row[5]),
                    "status_before": row[6],
                }
                if warehouse_configured(self.settings):
                    from genie_voice.databricks import warehouse_sql

                    warehouse_sql.revert_billing_resolution_uc(self.settings, payload)
                cur.execute(
                    f"UPDATE {tbl} SET reverted_at = now() WHERE adjustment_id = %s",
                    (row[0],),
                )
                reverted.append(row[0])
        return {"call_id": call_id, "reverted": reverted}

    def reset_demo_session(self, call_id: str) -> dict[str, Any]:
        """Reset per-call runtime artifacts so the scenario can be replayed."""
        billing_reset = self.revert_billing_adjustments(call_id)
        events_cleared = self.clear_resolution_events(call_id)
        state = self.get_call_state(call_id)
        if not state:
            return {"call_id": call_id, "reset": False, "reason": "call_not_found"}

        inner = dict(state.get("state") or {})
        inner.pop("live", None)
        inner.pop("resolution", None)
        inner["utterances"] = []
        self.upsert_call_state(call_id, state.get("customer_id"), inner)

        if not self.enabled:
            return {
                "call_id": call_id,
                "reset": True,
                "billing": billing_reset,
                "resolution_events_cleared": events_cleared,
            }

        utterances_tbl = self._table(self.settings.lakebase.live_utterances_table)
        with self._conn() as conn, conn.cursor() as cur:
            self._ensure_serving_schema(cur)
            cur.execute(f"DELETE FROM {utterances_tbl} WHERE call_id = %s", (call_id,))
        return {
            "call_id": call_id,
            "reset": True,
            "billing": billing_reset,
            "resolution_events_cleared": events_cleared,
        }

    @staticmethod
    def _query(cur, sql: str, params: tuple) -> list[dict[str, Any]]:
        try:
            cur.execute(sql, params)
        except Exception:  # noqa: BLE001 - table may not be synced yet
            return []
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]

    def _account_facts_local(self, customer_id: str) -> dict[str, Any]:
        import json
        import os

        base = os.environ.get("GENIE_LOCAL_VOLUME_DIR")
        tables = os.path.normpath(os.path.join(base, "..", "tables")) if base else None

        def load(name: str) -> list[dict[str, Any]]:
            if not tables:
                return []
            path = os.path.join(tables, f"{name}.json")
            if not os.path.exists(path):
                return []
            with open(path) as fh:
                return json.load(fh)

        customer = next((c for c in load("customers") if c.get("customer_id") == customer_id), None)
        invoices = [i for i in load("invoices") if i.get("customer_id") == customer_id]
        payments = [p for p in load("payments") if p.get("customer_id") == customer_id]
        return _account_facts(customer_id, customer, invoices, payments)

    def _account_facts_uc(self, customer_id: str) -> dict[str, Any]:
        from genie_voice.databricks.client import get_workspace_client

        wh = self.settings.databricks.sql_warehouse_id
        if not wh:
            raise RuntimeError("databricks.sql_warehouse_id is required for account facts.")
        client = get_workspace_client(self.settings)

        def query(sql: str) -> list[dict[str, Any]]:
            res = client.statement_execution.execute_statement(
                warehouse_id=wh,
                statement=sql,
                wait_timeout="30s",
            )
            manifest = getattr(res, "manifest", None)
            cols = [
                c.name for c in (getattr(getattr(manifest, "schema", None), "columns", None) or [])
            ]
            rows = (res.result.data_array if res.result else None) or []
            return [dict(zip(cols, row)) for row in rows]

        safe_customer = customer_id.replace("'", "''")
        customer = (
            query(f"SELECT * FROM {self.settings.fqtn('customers')} WHERE customer_id = '{safe_customer}'")
            or [None]
        )[0]
        invoices = query(
            f"SELECT * FROM {self.settings.fqtn('invoices')} "
            f"WHERE customer_id = '{safe_customer}' ORDER BY due_date DESC"
        )
        payments = query(
            f"SELECT * FROM {self.settings.fqtn('payments')} "
            f"WHERE customer_id = '{safe_customer}' ORDER BY payment_date DESC LIMIT 10"
        )
        return _account_facts(customer_id, customer, invoices, payments)

    def list_call_states(self, limit: int = 50) -> list[dict[str, Any]]:
        if not self.enabled:
            with _LOCK:
                return list(_MEM.values())[:limit]
        tbl = self._table(self.settings.lakebase.serving_table)
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(
                f"SELECT call_id, customer_id, state FROM {tbl} "
                f"ORDER BY updated_at DESC LIMIT %s",
                (limit,),
            )
            return [{"call_id": r[0], "customer_id": r[1], "state": r[2]} for r in cur.fetchall()]
