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
import os
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
_ISSUES_CACHE: dict[str, Any] = {"ts": 0.0, "value": None}
_ISSUES_TTL_S = 60.0


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


def _customer_has_account_issue(facts: dict[str, Any]) -> bool:
    """True when a customer has billing/account risk like the CUST-4028 demo profile."""
    if not facts.get("found"):
        return False
    summary = facts.get("summary") or {}
    customer = facts.get("customer") or {}
    if str(customer.get("status")) == "at_risk":
        return True
    if int(summary.get("overdue_invoice_count") or 0) > 0:
        return True
    if int(summary.get("recent_declined_payments") or 0) > 0:
        return True
    if not summary.get("autopay_enabled") and int(summary.get("open_invoice_count") or 0) > 0:
        return True
    return any(str(inv.get("status")) == "disputed" for inv in facts.get("invoices") or [])


def _issue_rationale(facts: dict[str, Any]) -> str:
    summary = facts.get("summary") or {}
    customer = facts.get("customer") or {}
    parts: list[str] = []
    if str(customer.get("status")) == "at_risk":
        parts.append("at-risk account")
    if int(summary.get("overdue_invoice_count") or 0) > 0:
        parts.append("overdue invoice exposure")
    if not summary.get("autopay_enabled"):
        parts.append("autopay off")
    if int(summary.get("recent_declined_payments") or 0) > 0:
        parts.append("declined payments")
    if any(str(inv.get("status")) == "disputed" for inv in facts.get("invoices") or []):
        parts.append("billing dispute")
    if int(summary.get("disputed_count") or 0) > 0:
        parts.append("billing dispute")
    return ", ".join(dict.fromkeys(parts)) if parts else "account issue"


def _row_to_issue_rationale(row: dict[str, Any]) -> str:
    return _issue_rationale(
        {
            "customer": {"status": row.get("customer_status")},
            "summary": {
                "overdue_invoice_count": row.get("overdue_invoice_count"),
                "autopay_enabled": row.get("autopay_enabled"),
                "recent_declined_payments": row.get("recent_declined_payments"),
                "disputed_count": row.get("disputed_count"),
            },
            "invoices": [{"status": "disputed"}] if int(row.get("disputed_count") or 0) > 0 else [],
        }
    )


def _apply_resolution_overlay(facts: dict[str, Any], resolution: dict[str, Any] | None) -> dict[str, Any]:
    """Legacy overlay hook — resolution metadata only; billing uses persisted adjustments."""
    return _apply_resolution_status_overlay(facts, resolution)


class LakebaseServing:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        # Offline deploy (no Databricks) is signalled by GENIE_LOCAL_VOLUME_DIR;
        # in that mode there is no Lakebase to reach, so fall back to in-memory.
        # This replaces the old GENIE_LAKEBASE__ENABLED=false env override.
        offline = bool(os.environ.get("GENIE_LOCAL_VOLUME_DIR"))
        self.enabled = self.settings.lakebase.enabled and not offline
        self._cred: dict[str, Any] | None = None  # cached {host,user,token,exp}
        self._pool = None  # psycopg_pool.ConnectionPool, built lazily
        self._pool_token: str | None = None  # token the live pool was built with

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

    def _get_pool(self):
        """Process-wide psycopg connection pool, rebuilt only when the short-lived
        Postgres OAuth token rotates (~hourly). Returns ``None`` when ``psycopg_pool``
        is not installed, in which case callers fall back to a direct connection.

        Pooling removes the TLS + auth handshake from every query, which is the
        dominant per-call cost on the Lakebase serving path. Without it each of
        the many UI fetches (call states, accounts, account facts, resolution
        events) opened a brand-new connection, so they resolved at noticeably
        different times under load.
        """
        try:
            from psycopg_pool import ConnectionPool  # noqa: F401
        except Exception:  # noqa: BLE001 - dependency optional; degrade gracefully
            return None
        cred = self._credentials()  # cheap: cached token, refreshed ~every 50 min
        if self._pool is None or self._pool_token != cred["password"]:
            self._build_pool(cred)
        return self._pool

    def _build_pool(self, cred: dict[str, Any]) -> None:
        from psycopg_pool import ConnectionPool

        if self._pool is not None:
            try:
                self._pool.close()
            except Exception:  # noqa: BLE001
                pass
        self._pool = ConnectionPool(
            kwargs=dict(
                host=cred["host"], port=cred["port"], dbname=cred["dbname"],
                user=cred["user"], password=cred["password"],
                sslmode="require", autocommit=True,
            ),
            min_size=1,
            max_size=8,
            max_idle=120.0,
            max_lifetime=3600.0,
            timeout=15.0,
            name="lakebase",
            open=True,
            # No per-checkout health ping: ``check=ConnectionPool.check_connection``
            # adds a full server round-trip to EVERY checkout, which is pure latency
            # on the voice tool path (account lookups / billing writes run several
            # checkouts per turn). Staleness is bounded instead by ``max_idle`` (idle
            # connections are recycled well under Postgres' server-side idle timeout)
            # and ``max_lifetime``; the pool is also rebuilt whenever the OAuth token
            # rotates. A rare dropped connection surfaces as a query error rather than
            # costing every healthy call an extra RTT.
        )
        self._pool_token = cred["password"]

    def _direct_conn(self):
        import psycopg

        c = self._credentials()
        return psycopg.connect(
            host=c["host"], port=c["port"], dbname=c["dbname"],
            user=c["user"], password=c["password"], sslmode="require", autocommit=True,
        )

    def _conn(self):
        """Check out a connection as a context manager.

        Uses the pool when ``psycopg_pool`` is available (connection is returned to
        the pool, not closed, on exit); otherwise opens a direct connection. Either
        way existing callers keep using ``with self._conn() as conn, conn.cursor()
        as cur:`` unchanged.
        """
        pool = self._get_pool()
        if pool is not None:
            return pool.connection()
        return self._direct_conn()

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
            self._ensure_reference_tables(cur)
            self._ensure_traces_table(cur)

    def _ensure_reference_tables(self, cur) -> None:
        """Create the customers/invoices/payments serving cache (idempotent).

        These are a low-latency SERVING CACHE of the UC Delta source of truth,
        snapshot-loaded by snapshot_reference_tables(). They are read on the
        lookup_account hot path so it never touches the SQL warehouse.
        Deliberately left at the DEFAULT replica identity (NOT full) so Lakebase
        CDF SKIPS them — reference data flows Delta -> Lakebase, never the
        reverse. Split out from ensure_schema so a reference refresh never has to
        touch (and re-own) unrelated tables like voice_traces.
        """
        customers_tbl = self._table("customers")
        invoices_tbl = self._table("invoices")
        payments_tbl = self._table("payments")
        self._ensure_serving_schema(cur)
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {customers_tbl} (
                customer_id     TEXT PRIMARY KEY,
                full_name       TEXT,
                segment         TEXT,
                region          TEXT,
                plan            TEXT,
                monthly_charge  NUMERIC(10,2),
                tenure_months   INTEGER,
                status          TEXT,
                autopay_enabled BOOLEAN,
                email           TEXT,
                signup_date     DATE
            )
            """
        )
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {invoices_tbl} (
                invoice_id  TEXT PRIMARY KEY,
                customer_id TEXT NOT NULL,
                period      TEXT,
                issue_date  DATE,
                due_date    DATE,
                amount      NUMERIC(10,2),
                late_fee    NUMERIC(10,2),
                status      TEXT,
                paid_date   DATE
            )
            """
        )
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {payments_tbl} (
                payment_id   TEXT PRIMARY KEY,
                invoice_id   TEXT NOT NULL,
                customer_id  TEXT NOT NULL,
                amount       NUMERIC(10,2),
                payment_date DATE,
                method       TEXT,
                status       TEXT
            )
            """
        )
        cur.execute(
            f"CREATE INDEX IF NOT EXISTS invoices_customer_idx ON {invoices_tbl} (customer_id)"
        )
        cur.execute(
            f"CREATE INDEX IF NOT EXISTS payments_customer_idx ON {payments_tbl} (customer_id)"
        )

    # ---- voice traces (observability) ------------------------------------- #
    def _ensure_traces_table(self, cur) -> None:
        """Idempotently ensure the voice_traces table + indexes exist.

        Called from ensure_schema AND lazily from every trace read/write so a
        cold table (startup race, fresh Lakebase branch) can never surface as a
        500 or a dropped trace.

        IMPORTANT: the serving role may have INSERT/SELECT on ``voice_traces`` but
        NOT own it (a common Lakebase/Postgres setup where the table is created by
        a migration role). In that case ``CREATE INDEX IF NOT EXISTS`` raises
        ``InsufficientPrivilege`` — and, worse, poisons the surrounding
        transaction so the caller's INSERT/SELECT then fails too. So we first
        probe for the table with ``to_regclass`` (transaction-safe, never raises)
        and skip ALL DDL when it already exists. The result is cached per process
        so we probe at most once.
        """
        if getattr(self, "_traces_table_ready", False):
            return
        traces_tbl = self._table("voice_traces")
        # Fast path: table already exists → no DDL, no ownership needed.
        cur.execute("SELECT to_regclass(%s)", (traces_tbl,))
        row = cur.fetchone()
        if row and row[0] is not None:
            self._traces_table_ready = True
            return
        # Cold table: create it (we necessarily own what we just created).
        self._ensure_serving_schema(cur)
        # Voice observability: one row per turn holding the full span tree
        # (STT → LLM iterations + tool calls → TTS). `trace` is the complete
        # JSON doc; the promoted columns exist for cheap listing/filtering.
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {traces_tbl} (
                trace_id                    TEXT PRIMARY KEY,
                session_id                  TEXT,
                turn_id                     INTEGER,
                call_id                     TEXT,
                customer_id                 TEXT,
                capability                  TEXT,
                language                    TEXT,
                detected_language           TEXT,
                status                      TEXT,
                input_transcript            TEXT,
                output_text                 TEXT,
                tool_names                  JSONB,
                apply_billing_action_called BOOLEAN,
                lookup_account_count        INTEGER,
                llm_iterations              INTEGER,
                total_ms                    DOUBLE PRECISION,
                trace                       JSONB,
                created_at                  TIMESTAMPTZ DEFAULT now()
            )
            """
        )
        cur.execute(
            f"CREATE INDEX IF NOT EXISTS voice_traces_session_idx "
            f"ON {traces_tbl} (session_id, turn_id)"
        )
        cur.execute(
            f"CREATE INDEX IF NOT EXISTS voice_traces_created_idx "
            f"ON {traces_tbl} (created_at DESC)"
        )
        self._traces_table_ready = True
    def _trace_file(self) -> str:
        """Durable local store for traces when Lakebase is disabled (dev/offline).

        Traces are appended as JSON lines to a file on disk so they survive process
        restarts — never in-memory only.
        """
        base = os.environ.get("GENIE_TRACE_DIR")
        if not base:
            import tempfile

            base = os.path.join(tempfile.gettempdir(), "genie_voice_traces")
        os.makedirs(base, exist_ok=True)
        return os.path.join(base, "voice_traces.jsonl")

    def _read_trace_file(self) -> list[dict[str, Any]]:
        path = self._trace_file()
        if not os.path.exists(path):
            return []
        rows: list[dict[str, Any]] = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return rows

    def _mirror_trace_to_uc(self, trace: dict[str, Any]) -> None:
        """Best-effort append to the governed UC Delta table (retained copy)."""
        if not warehouse_configured(self.settings):
            return
        try:
            from genie_voice.databricks import warehouse_sql

            warehouse_sql.insert_voice_trace_uc(self.settings, trace)
        except Exception:  # noqa: BLE001 - Lakebase is the source of truth for the UI
            pass

    def insert_voice_trace(self, trace: dict[str, Any]) -> None:
        """Persist one turn trace. Called from the background trace-writer thread."""
        trace_id = str(trace.get("trace_id") or uuid.uuid4().hex)
        if not self.enabled:
            # Durable local fallback: append to a JSONL file on disk (not memory).
            try:
                with open(self._trace_file(), "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(trace, ensure_ascii=False) + "\n")
            except OSError:
                pass
            return
        tbl = self._table("voice_traces")
        with self._conn() as conn, conn.cursor() as cur:
            self._ensure_traces_table(cur)
            cur.execute(
                f"""
                INSERT INTO {tbl}
                  (trace_id, session_id, turn_id, call_id, customer_id, capability,
                   language, detected_language, status, input_transcript, output_text,
                   tool_names, apply_billing_action_called, lookup_account_count,
                   llm_iterations, total_ms, trace, created_at)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,now())
                ON CONFLICT (trace_id) DO NOTHING
                """,
                (
                    trace_id,
                    trace.get("session_id"),
                    trace.get("turn_id"),
                    trace.get("call_id"),
                    trace.get("customer_id"),
                    trace.get("capability"),
                    trace.get("language"),
                    trace.get("detected_language"),
                    trace.get("status"),
                    trace.get("input_transcript"),
                    trace.get("output_text"),
                    json.dumps(trace.get("tool_names") or []),
                    bool(trace.get("apply_billing_action_called")),
                    int(trace.get("lookup_account_count") or 0),
                    int(trace.get("llm_iterations") or 0),
                    trace.get("total_ms"),
                    json.dumps(trace),
                ),
            )
        # Mirror to the governed UC Delta table for retained, SQL-queryable eval
        # history. Runs on the writer thread, so it never adds turn latency.
        self._mirror_trace_to_uc(trace)

    def list_voice_traces(
        self,
        *,
        limit: int = 100,
        session_id: str | None = None,
        call_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Compact trace rows (no span bodies) for the observability list view."""
        if not self.enabled:
            rows = list(reversed(self._read_trace_file()))
            if session_id:
                rows = [r for r in rows if r.get("session_id") == session_id]
            if call_id:
                rows = [r for r in rows if r.get("call_id") == call_id]
            return [self._trace_summary(r) for r in rows[:limit]]
        tbl = self._table("voice_traces")
        clauses: list[str] = []
        params: list[Any] = []
        if session_id:
            clauses.append("session_id = %s")
            params.append(session_id)
        if call_id:
            clauses.append("call_id = %s")
            params.append(call_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        with self._conn() as conn, conn.cursor() as cur:
            self._ensure_traces_table(cur)
            cur.execute(
                f"""
                SELECT trace_id, session_id, turn_id, call_id, customer_id, capability,
                       language, detected_language, status, input_transcript, output_text,
                       tool_names, apply_billing_action_called, lookup_account_count,
                       llm_iterations, total_ms, created_at
                FROM {tbl}
                {where}
                ORDER BY created_at DESC
                LIMIT %s
                """,
                tuple(params),
            )
            cols = [d[0] for d in cur.description]
            out: list[dict[str, Any]] = []
            for row in cur.fetchall():
                item = dict(zip(cols, row))
                created = item.get("created_at")
                if hasattr(created, "isoformat"):
                    item["created_at"] = created.isoformat()
                out.append(item)
            return out

    def get_voice_trace(self, trace_id: str) -> dict[str, Any] | None:
        """Full trace document (with the span tree) for the detail view."""
        if not self.enabled:
            for r in reversed(self._read_trace_file()):
                if r.get("trace_id") == trace_id:
                    return r
            return None
        tbl = self._table("voice_traces")
        with self._conn() as conn, conn.cursor() as cur:
            self._ensure_traces_table(cur)
            cur.execute(f"SELECT trace FROM {tbl} WHERE trace_id = %s", (trace_id,))
            row = cur.fetchone()
            return row[0] if row else None

    @staticmethod
    def _trace_summary(trace: dict[str, Any]) -> dict[str, Any]:
        keys = (
            "trace_id", "session_id", "turn_id", "call_id", "customer_id", "capability",
            "language", "detected_language", "status", "input_transcript", "output_text",
            "tool_names", "apply_billing_action_called", "lookup_account_count",
            "llm_iterations", "total_ms", "started_at",
        )
        return {k: trace.get(k) for k in keys}

    # ---- live call state -------------------------------------------------- #
    def upsert_call_state(self, call_id: str, customer_id: str | None, state: dict) -> None:
        if not self.enabled:
            with _LOCK:
                _MEM[call_id] = {"call_id": call_id, "customer_id": customer_id, "state": state}
            return
        tbl = self._table(self.settings.lakebase.serving_table)
        # Schema + tables are created once at startup via ensure_schema(); the
        # per-write CREATE SCHEMA IF NOT EXISTS was a redundant round-trip (and
        # never created the table anyway), so it's dropped from the hot path.
        with self._conn() as conn, conn.cursor() as cur:
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
            # The in-progress live conversation lives ONLY in call_state (the
            # state JSON above); it is intentionally NOT written into
            # live_call_utterances. That table is the immutable, analytics-facing
            # transcript owned solely by the seed/ingest path. Keeping the live
            # overlay out of it means a live or demo session (and its reset) can
            # never mutate or wipe a call's durable transcript, so the
            # "every call has >=1 utterance" invariant holds by construction.

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
        # Reference facts are served exclusively from the Lakebase serving cache
        # (sub-ms Postgres, never the SQL warehouse on the hot path); the cache is
        # populated by snapshot_reference_tables() at deploy/refresh time. Local
        # mode (Lakebase disabled) reads the datagen JSON. No warehouse fallback:
        # if Lakebase is the serving store, it is the ONLY serving read path.
        if not self.enabled:
            return self._account_facts_local(customer_id)
        return self._account_facts_lakebase(customer_id)

    def get_account_facts(self, customer_id: str) -> dict[str, Any]:
        """Serve account facts from UC/local source merged with persisted billing adjustments.

        On the Lakebase path this reads facts (customers/invoices/payments) AND
        billing adjustments on a SINGLE pooled connection — one checkout per
        lookup instead of two, cutting a full connect/round-trip off the voice
        tool hot path.
        """
        if not self.enabled:
            facts = self._account_facts_local(customer_id)
            return _apply_billing_adjustments(facts, self.list_billing_adjustments(customer_id))
        with self._conn() as conn, conn.cursor() as cur:
            facts = self._account_facts_lakebase_cur(cur, customer_id)
            adjustments = self._list_billing_adjustments_cur(cur, customer_id)
        return _apply_billing_adjustments(facts, adjustments)

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

    def _upsert_live_utterances(self, cur, call_id: str, utterances: list[dict[str, Any]]) -> None:
        """Append-only, idempotent upsert of transcript turns keyed by utterance_id.

        Utterances are immutable events, so this NEVER deletes: a call's turns can
        never be wiped by a write path (which is what previously let the live/demo
        reset flow empty a seed call's transcript and break the DQ invariant). The
        DO UPDATE is change-aware — it only writes when a column actually differs —
        so re-running an identical ingest produces ZERO CDF change-rows (no churn,
        no delete+reinsert window that could lose data mid-run).
        """
        tbl = self._table(self.settings.lakebase.live_utterances_table)
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
                WHERE {tbl}.call_id IS DISTINCT FROM EXCLUDED.call_id
                   OR {tbl}.turn_index IS DISTINCT FROM EXCLUDED.turn_index
                   OR {tbl}.channel IS DISTINCT FROM EXCLUDED.channel
                   OR {tbl}.speaker_role IS DISTINCT FROM EXCLUDED.speaker_role
                   OR {tbl}.start_sec IS DISTINCT FROM EXCLUDED.start_sec
                   OR {tbl}.end_sec IS DISTINCT FROM EXCLUDED.end_sec
                   OR {tbl}.text IS DISTINCT FROM EXCLUDED.text
                   OR {tbl}.confidence IS DISTINCT FROM EXCLUDED.confidence
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

    def upsert_live_utterances(self, call_id: str, utterances: list[dict[str, Any]]) -> None:
        """Append-only ingest of a call's transcript turns (see _upsert_live_utterances).

        Only the seed/ingest path writes durable transcripts here; the live voice
        path keeps its in-progress turns in call_state (state JSON), so a live or
        demo session can never mutate or delete this analytics-facing table.
        """
        if not self.enabled:
            return
        with self._conn() as conn, conn.cursor() as cur:
            self._ensure_serving_schema(cur)
            self._upsert_live_utterances(cur, call_id, utterances)

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
        # Bootstrap ensure_schema() owns DDL; no per-write schema round-trip.
        with self._conn() as conn, conn.cursor() as cur:
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
        with self._conn() as conn, conn.cursor() as cur:
            return self._list_billing_adjustments_cur(
                cur, customer_id, call_id=call_id, active_only=active_only
            )

    def _list_billing_adjustments_cur(
        self,
        cur,
        customer_id: str,
        *,
        call_id: str | None = None,
        active_only: bool = True,
    ) -> list[dict[str, Any]]:
        """Run the billing-adjustments SELECT on an existing cursor.

        Factored out so get_account_facts can share one connection for facts +
        adjustments. Bootstrap ensure_schema() owns DDL; no per-read schema DDL.
        """
        tbl = self._table("billing_adjustments")
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

        # Write ONLY to Lakebase (fast OLTP, low-ms). The governed UC copy is
        # produced off the hot path by Lakebase Change Data Feed
        # (billing_adjustments -> lb_billing_adjustments_history, ~15s), so no
        # synchronous SQL-warehouse statement ever sits on the voice turn. The
        # waiver/plan is expressed purely as this adjustment row and overlaid on
        # reads (_apply_billing_adjustments); the canonical invoices table is
        # never mutated in place, which keeps Delta the single source of truth
        # and avoids a Postgres->Delta write-back loop on reference data.
        if not self.enabled:
            with _LOCK:
                rows = _MEM_ADJUSTMENTS.setdefault(customer_id, [])
                rows[:] = [r for r in rows if r.get("adjustment_id") != adjustment_id]
                rows.append(adjustment)
            return {"applied": True, "adjustment": adjustment}

        tbl = self._table("billing_adjustments")
        # Bootstrap ensure_schema() owns DDL; no per-write schema round-trip.
        with self._conn() as conn, conn.cursor() as cur:
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
        return {"applied": True, "adjustment": adjustment}

    def revert_billing_adjustments(self, call_id: str) -> dict[str, Any]:
        # Revert is also Lakebase-only: marking reverted_at streams to UC via CDF
        # (an update_postimage row), and reads honor reverted_at IS NULL. No
        # synchronous SQL-warehouse write on this path either.
        reverted: list[str] = []
        if not self.enabled:
            with _LOCK:
                for customer_id, rows in list(_MEM_ADJUSTMENTS.items()):
                    kept: list[dict[str, Any]] = []
                    for row in rows:
                        if row.get("call_id") == call_id and not row.get("reverted_at"):
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
                UPDATE {tbl} SET reverted_at = now()
                WHERE call_id = %s AND reverted_at IS NULL
                RETURNING adjustment_id
                """,
                (call_id,),
            )
            reverted = [row[0] for row in cur.fetchall()]
        return {"call_id": call_id, "reverted": reverted}

    def reset_demo_session(self, call_id: str) -> dict[str, Any]:
        """Reset per-call runtime artifacts so the scenario can be replayed.

        Reset clears only the ephemeral live overlay (the call_state scratchpad),
        reverts the customer's live billing adjustments, and clears this call's
        resolution events. It deliberately does NOT touch live_call_utterances:
        that table holds the immutable seed transcript that analytics/Genie/DQ
        rely on, and previously deleting it here is exactly what left seed calls
        with zero utterances. The durable transcript stays intact across replays.
        """
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

    def _account_facts_lakebase(self, customer_id: str) -> dict[str, Any]:
        """Read account facts from the Lakebase serving cache (the sole serving
        read path; never the SQL warehouse).

        Reads execute directly and let errors propagate — a missing/broken
        serving cache must fail loudly, not silently degrade. A customer that is
        genuinely absent returns found=False (correct semantics, not a fallback).
        """
        with self._conn() as conn, conn.cursor() as cur:
            return self._account_facts_lakebase_cur(cur, customer_id)

    def _account_facts_lakebase_cur(self, cur, customer_id: str) -> dict[str, Any]:
        """Run the three account-facts SELECTs on an existing cursor.

        Factored out so get_account_facts can read facts AND billing adjustments
        on a single pooled connection (one checkout per lookup, not two).
        """
        def rows(sql: str) -> list[dict[str, Any]]:
            cur.execute(sql, (customer_id,))
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]

        customers_tbl = self._table("customers")
        invoices_tbl = self._table("invoices")
        payments_tbl = self._table("payments")
        customer_rows = rows(f"SELECT * FROM {customers_tbl} WHERE customer_id = %s")
        customer = customer_rows[0] if customer_rows else None
        invoices = rows(f"SELECT * FROM {invoices_tbl} WHERE customer_id = %s ORDER BY due_date DESC")
        payments = rows(
            f"SELECT * FROM {payments_tbl} WHERE customer_id = %s "
            f"ORDER BY payment_date DESC LIMIT 10"
        )
        return _account_facts(customer_id, customer, invoices, payments)

    # Reference table -> (primary key, [(column, postgres_cast_type), ...]). The
    # SQL warehouse returns every value as a string, so each column is bound as
    # text and CAST to its Postgres type in the INSERT (CAST(NULL AS t) is valid).
    _REFERENCE_TABLE_SPECS: dict[str, tuple[str, list[tuple[str, str]]]] = {
        "customers": ("customer_id", [
            ("customer_id", "TEXT"), ("full_name", "TEXT"), ("segment", "TEXT"),
            ("region", "TEXT"), ("plan", "TEXT"), ("monthly_charge", "NUMERIC(10,2)"),
            ("tenure_months", "INTEGER"), ("status", "TEXT"),
            ("autopay_enabled", "BOOLEAN"), ("email", "TEXT"), ("signup_date", "DATE"),
        ]),
        "invoices": ("invoice_id", [
            ("invoice_id", "TEXT"), ("customer_id", "TEXT"), ("period", "TEXT"),
            ("issue_date", "DATE"), ("due_date", "DATE"), ("amount", "NUMERIC(10,2)"),
            ("late_fee", "NUMERIC(10,2)"), ("status", "TEXT"), ("paid_date", "DATE"),
        ]),
        "payments": ("payment_id", [
            ("payment_id", "TEXT"), ("invoice_id", "TEXT"), ("customer_id", "TEXT"),
            ("amount", "NUMERIC(10,2)"), ("payment_date", "DATE"),
            ("method", "TEXT"), ("status", "TEXT"),
        ]),
    }

    def snapshot_reference_tables(self) -> dict[str, int]:
        """Snapshot reverse-ETL: copy UC Delta reference tables into Lakebase.

        Reads customers/invoices/payments from the UC source of truth via the SQL
        warehouse (a one-off setup/refresh cost, NOT on the hot path) and upserts
        them into the Lakebase serving cache read by lookup_account. Idempotent
        (INSERT ... ON CONFLICT DO UPDATE keyed on PK), so re-running only touches
        changed rows and reference data is slowly-changing. Run from
        infra/lakebase/setup_lakebase.py and the refresh schedule.
        """
        if not self.enabled:
            return {}
        if not warehouse_configured(self.settings):
            raise RuntimeError("snapshot_reference_tables requires databricks.sql_warehouse_id")
        from genie_voice.databricks.client import get_workspace_client

        with self._conn() as conn, conn.cursor() as cur:
            self._ensure_reference_tables(cur)
        client = get_workspace_client(self.settings)
        wh = self.settings.databricks.sql_warehouse_id
        counts: dict[str, int] = {}
        for table, (pk, colspecs) in self._REFERENCE_TABLE_SPECS.items():
            cols = [c for c, _ in colspecs]
            res = client.statement_execution.execute_statement(
                warehouse_id=wh,
                statement=f"SELECT {', '.join(cols)} FROM {self.settings.fqtn(table)}",
                wait_timeout="50s",
            )
            rows = (res.result.data_array if res.result else None) or []
            tbl = self._table(table)
            placeholders = ", ".join(f"CAST(%s AS {t})" for _, t in colspecs)
            updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in cols if c != pk)
            with self._conn() as conn, conn.cursor() as cur:
                for row in rows:
                    values = [(v if v not in ("", None) else None) for v in row]
                    cur.execute(
                        f"INSERT INTO {tbl} ({', '.join(cols)}) VALUES ({placeholders}) "
                        f"ON CONFLICT ({pk}) DO UPDATE SET {updates}",
                        tuple(values),
                    )
            counts[table] = len(rows)
        return counts

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

    def _reference_tables_dir(self) -> str | None:
        import os

        base = os.environ.get("GENIE_LOCAL_VOLUME_DIR")
        if not base:
            return None
        return os.path.normpath(os.path.join(base, "..", "tables"))

    def _load_reference_table(self, name: str) -> list[dict[str, Any]]:
        import os

        tables = self._reference_tables_dir()
        if not tables:
            return []
        path = os.path.join(tables, f"{name}.json")
        if not os.path.exists(path):
            return []
        with open(path) as fh:
            rows = json.load(fh)
        return rows if isinstance(rows, list) else []

    def _warehouse_query(self, sql: str) -> list[dict[str, Any]]:
        from genie_voice.databricks.client import get_workspace_client

        wh = self.settings.databricks.sql_warehouse_id
        if not wh:
            raise RuntimeError("databricks.sql_warehouse_id is required.")
        client = get_workspace_client(self.settings)
        res = client.statement_execution.execute_statement(
            warehouse_id=wh,
            statement=sql,
            wait_timeout="45s",
        )
        manifest = getattr(res, "manifest", None)
        cols = [
            c.name for c in (getattr(getattr(manifest, "schema", None), "columns", None) or [])
        ]
        rows = (res.result.data_array if res.result else None) or []
        return [dict(zip(cols, row)) for row in rows]

    def _customers_with_issues_rows(self) -> list[dict[str, Any]]:
        """One batch pass over reference tables — no per-customer warehouse round-trips."""
        if self._reference_tables_dir():
            return self._customers_with_issues_local()
        if warehouse_configured(self.settings):
            return self._customers_with_issues_uc()
        raise RuntimeError("no reference data source configured")

    def _customers_with_issues_local(self) -> list[dict[str, Any]]:
        customers = self._load_reference_table("customers")
        invoices = self._load_reference_table("invoices")
        payments = self._load_reference_table("payments")
        inv_by: dict[str, list[dict[str, Any]]] = {}
        pay_by: dict[str, list[dict[str, Any]]] = {}
        for inv in invoices:
            cid = str(inv.get("customer_id") or "")
            if cid:
                inv_by.setdefault(cid, []).append(inv)
        for pay in payments:
            cid = str(pay.get("customer_id") or "")
            if cid:
                pay_by.setdefault(cid, []).append(pay)

        out: list[dict[str, Any]] = []
        for customer in customers:
            cid = str(customer.get("customer_id") or "")
            if not cid:
                continue
            cust_invoices = inv_by.get(cid, [])
            cust_payments = pay_by.get(cid, [])
            facts = _account_facts(cid, customer, cust_invoices, cust_payments)
            if not _customer_has_account_issue(facts):
                continue
            summary = facts.get("summary") or {}
            disputed = sum(1 for i in cust_invoices if str(i.get("status")) == "disputed")
            out.append(
                {
                    "customer_id": cid,
                    "full_name": customer.get("full_name"),
                    "customer_status": customer.get("status"),
                    "autopay_enabled": summary.get("autopay_enabled"),
                    "overdue_invoice_count": summary.get("overdue_invoice_count", 0),
                    "overdue_amount": summary.get("overdue_amount", 0),
                    "recent_declined_payments": summary.get("recent_declined_payments", 0),
                    "disputed_count": disputed,
                }
            )
        return out

    def _customers_with_issues_uc(self) -> list[dict[str, Any]]:
        customers = self.settings.fqtn("customers")
        invoices = self.settings.fqtn("invoices")
        payments = self.settings.fqtn("payments")
        sql = f"""
        WITH inv AS (
          SELECT customer_id,
                 SUM(CASE WHEN status IN ('open', 'overdue', 'disputed') THEN 1 ELSE 0 END) AS open_invoice_count,
                 SUM(CASE WHEN status = 'overdue' THEN 1 ELSE 0 END) AS overdue_invoice_count,
                 SUM(CASE WHEN status = 'overdue' THEN CAST(amount AS DOUBLE) ELSE 0 END) AS overdue_amount,
                 SUM(CASE WHEN status = 'disputed' THEN 1 ELSE 0 END) AS disputed_count
          FROM {invoices}
          GROUP BY customer_id
        ),
        pays AS (
          SELECT customer_id, COUNT(*) AS recent_declined_payments
          FROM {payments}
          WHERE status = 'declined'
          GROUP BY customer_id
        )
        SELECT c.customer_id,
               c.full_name,
               c.status AS customer_status,
               c.autopay_enabled,
               COALESCE(inv.overdue_invoice_count, 0) AS overdue_invoice_count,
               COALESCE(inv.overdue_amount, 0) AS overdue_amount,
               COALESCE(pays.recent_declined_payments, 0) AS recent_declined_payments,
               COALESCE(inv.disputed_count, 0) AS disputed_count
        FROM {customers} c
        LEFT JOIN inv ON inv.customer_id = c.customer_id
        LEFT JOIN pays ON pays.customer_id = c.customer_id
        WHERE c.status = 'at_risk'
           OR COALESCE(inv.overdue_invoice_count, 0) > 0
           OR COALESCE(pays.recent_declined_payments, 0) > 0
           OR COALESCE(inv.disputed_count, 0) > 0
           OR (NOT COALESCE(c.autopay_enabled, false) AND COALESCE(inv.open_invoice_count, 0) > 0)
        ORDER BY overdue_invoice_count DESC, overdue_amount DESC, c.customer_id
        """
        return self._warehouse_query(sql)

    def _attach_call_context(
        self,
        row: dict[str, Any],
        call_by_customer: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        cid = str(row["customer_id"])
        call = call_by_customer.get(cid)
        signals: dict[str, Any] = {}
        if call:
            state = call.get("state") or {}
            gold = state.get("gold") or {}
            live = state.get("live") or {}
            signals = {
                "primary_intent": gold.get("primary_intent") or live.get("primary_intent"),
                "sentiment_label": gold.get("sentiment_label") or live.get("sentiment_label"),
                "next_best_action": gold.get("next_best_action") or live.get("next_best_action"),
            }
        resolution = ((call or {}).get("state") or {}).get("resolution") or {}
        return {
            "customer_id": cid,
            "full_name": row.get("full_name"),
            "customer_status": row.get("customer_status"),
            "call_id": call.get("call_id") if call else None,
            "issue_status": str(resolution.get("status") or "open"),
            "overdue_invoice_count": row.get("overdue_invoice_count", 0),
            "overdue_amount": row.get("overdue_amount", 0),
            "autopay_enabled": row.get("autopay_enabled"),
            "recent_declined_payments": row.get("recent_declined_payments", 0),
            "rationale": _row_to_issue_rationale(row),
            **signals,
        }

    def list_customers_with_issues(self, limit: int = 50) -> list[dict[str, Any]]:
        """Customers with billing/account risk and their active assist call when present."""
        now = time.monotonic()
        issue_rows: list[dict[str, Any]]
        cached_rows = _ISSUES_CACHE.get("value")
        if cached_rows is None or now - float(_ISSUES_CACHE["ts"]) >= _ISSUES_TTL_S:
            try:
                issue_rows = self._customers_with_issues_rows()
            except Exception:
                issue_rows = []
            _ISSUES_CACHE["value"] = issue_rows
            _ISSUES_CACHE["ts"] = now
        else:
            issue_rows = list(cached_rows)

        calls = self.list_call_states(limit=200)
        call_by_customer: dict[str, dict[str, Any]] = {}
        for row in calls:
            cid = row.get("customer_id")
            if cid and cid not in call_by_customer:
                call_by_customer[str(cid)] = row

        if not issue_rows:
            seen: set[str] = set()
            for row in calls:
                cid = row.get("customer_id")
                if not cid or str(cid) in seen:
                    continue
                seen.add(str(cid))
                try:
                    facts = self.get_account_facts(str(cid))
                except Exception:
                    continue
                if not _customer_has_account_issue(facts):
                    continue
                summary = facts.get("summary") or {}
                customer = facts.get("customer") or {}
                issue_rows.append(
                    {
                        "customer_id": str(cid),
                        "full_name": customer.get("full_name"),
                        "customer_status": customer.get("status"),
                        "autopay_enabled": summary.get("autopay_enabled"),
                        "overdue_invoice_count": summary.get("overdue_invoice_count", 0),
                        "overdue_amount": summary.get("overdue_amount", 0),
                        "recent_declined_payments": summary.get("recent_declined_payments", 0),
                        "disputed_count": sum(
                            1 for i in facts.get("invoices") or [] if str(i.get("status")) == "disputed"
                        ),
                    }
                )

        out = [self._attach_call_context(row, call_by_customer) for row in issue_rows]

        def _priority(item: dict[str, Any]) -> tuple:
            open_issue = 0 if str(item.get("issue_status")) == "closed" else 1
            return (
                open_issue,
                int(item.get("overdue_invoice_count") or 0),
                float(item.get("overdue_amount") or 0),
                1 if item.get("customer_status") == "at_risk" else 0,
            )

        out.sort(key=_priority, reverse=True)
        return out[:limit]
