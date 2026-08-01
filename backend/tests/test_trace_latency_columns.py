"""Tests for the voice_traces promoted-column migration and TTFT persistence.

``ttft_ms`` (time to first audio), the TTS timing columns and ``guard_roster``
(the per-turn guardrail ledger) were all added after the table first shipped, and
``voice_traces`` is owned by whichever principal created it —
in practice the deployed app's service principal, while a developer's own role
holds DML only. So the migration has to satisfy three things at once:

  1. the owner adds the columns exactly once,
  2. a non-owner's denied ALTER costs nothing more than the columns staying absent,
  3. trace writes keep working either way, since observability must never break a
     call.

Exercised with a fake connection/cursor — no database required.
"""
from __future__ import annotations

import json

import pytest

from genie_voice.serve.lakebase import (
    _TRACE_LATENCY_COLUMNS,
    _TRACE_MIGRATED_COLUMNS,
    LakebaseServing,
)

_V1_COLUMNS = [
    "trace_id", "session_id", "turn_id", "call_id", "customer_id", "capability",
    "language", "detected_language", "status", "input_transcript", "output_text",
    "tool_names", "apply_billing_action_called", "lookup_account_count",
    "llm_iterations", "total_ms", "trace", "created_at",
]


class FakeCursor:
    def __init__(self, columns: list[str], *, can_alter: bool):
        self._columns = list(columns)
        self._can_alter = can_alter
        self.executed: list[tuple[str, object]] = []
        self._last: list[tuple] = []

    def execute(self, sql, params=None):
        upper = sql.strip().upper()
        if upper.startswith("ALTER TABLE"):
            if not self._can_alter:
                raise RuntimeError("must be owner of table voice_traces")
            column = sql.split("ADD COLUMN IF NOT EXISTS")[1].split()[0]
            self._columns.append(column)
        self.executed.append((sql, params))
        if "INFORMATION_SCHEMA.COLUMNS" in upper:
            self._last = [(c,) for c in self._columns]
        elif "TO_REGCLASS" in upper:
            self._last = [("serving.voice_traces",)]
        else:
            self._last = []

    # psycopg exposes result column names here; the list path zips them onto rows.
    description = ()

    def fetchall(self):
        return self._last

    def fetchone(self):
        return self._last[0] if self._last else None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeConn:
    def __init__(self, cursor: FakeCursor):
        self._cursor = cursor

    def cursor(self):
        return self._cursor

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeServing:
    """Stand-in exposing only what the migration/insert paths touch."""

    def __init__(self, columns=None, *, can_alter: bool):
        self.enabled = True
        self.cursor = FakeCursor(columns if columns is not None else _V1_COLUMNS, can_alter=can_alter)
        self._traces_table_ready = True
        self.settings = _Settings()
        self.mirrored = 0

    def _conn(self):
        return FakeConn(self.cursor)

    def _table(self, name: str) -> str:
        return f'"serving"."{name}"'

    def _ensure_traces_table(self, cur):
        return None

    def _mirror_trace_to_uc(self, trace):
        self.mirrored += 1

    # Real implementations under test.
    _trace_columns = LakebaseServing._trace_columns
    _trace_write_columns = LakebaseServing._trace_write_columns
    migrate_voice_traces = LakebaseServing.migrate_voice_traces
    insert_voice_trace = LakebaseServing.insert_voice_trace
    list_voice_traces = LakebaseServing.list_voice_traces


class _Lakebase:
    schema_name = "serving"


class _Settings:
    lakebase = _Lakebase()


def _insert_sql(serving: FakeServing) -> str:
    for sql, _ in serving.cursor.executed:
        if sql.strip().upper().startswith("INSERT INTO"):
            return sql
    raise AssertionError("no INSERT was executed")


def _insert_params(serving: FakeServing) -> tuple:
    for sql, params in serving.cursor.executed:
        if sql.strip().upper().startswith("INSERT INTO"):
            return params
    raise AssertionError("no INSERT was executed")


def test_owner_adds_every_promoted_column_once():
    s = FakeServing(can_alter=True)
    added = s.migrate_voice_traces()
    assert added == list(_TRACE_MIGRATED_COLUMNS)

    alters = [sql for sql, _ in s.cursor.executed if sql.strip().upper().startswith("ALTER")]
    assert len(alters) == len(_TRACE_MIGRATED_COLUMNS)

    # Cached: a second pass neither re-probes nor re-alters.
    before = len(s.cursor.executed)
    assert s._trace_columns() >= set(_TRACE_MIGRATED_COLUMNS)
    assert len(s.cursor.executed) == before


def test_each_column_is_added_with_its_own_type():
    # The migration started out latency-only and hardcoded DOUBLE PRECISION;
    # guard_roster is JSONB, so a shared type would produce an unusable column.
    s = FakeServing(can_alter=True)
    s.migrate_voice_traces()
    alters = {
        sql.split("IF NOT EXISTS")[1].strip(): sql
        for sql, _ in s.cursor.executed
        if sql.strip().upper().startswith("ALTER")
    }
    assert alters["guard_roster JSONB"]
    assert alters["ttft_ms DOUBLE PRECISION"]


def test_already_migrated_table_issues_no_ddl():
    s = FakeServing(_V1_COLUMNS + list(_TRACE_MIGRATED_COLUMNS), can_alter=True)
    assert s.migrate_voice_traces() == []
    assert not [sql for sql, _ in s.cursor.executed if sql.strip().upper().startswith("ALTER")]


def test_non_owner_migration_is_a_no_op_not_an_error():
    s = FakeServing(can_alter=False)
    assert s.migrate_voice_traces() == []
    assert s._trace_columns().isdisjoint(_TRACE_MIGRATED_COLUMNS)


def test_guard_roster_is_written_as_json_once_the_column_exists():
    s = FakeServing(_V1_COLUMNS + list(_TRACE_MIGRATED_COLUMNS), can_alter=True)
    roster = [{"guard_id": "language_gate", "outcome": "passed"}]
    s.insert_voice_trace({"trace_id": "t1", "total_ms": 1.0, "guard_roster": roster})

    columns = _insert_sql(s).split("(", 1)[1].split(")", 1)[0].split(", ")
    params = _insert_params(s)
    assert json.loads(params[columns.index("guard_roster")]) == roster


def test_list_reads_the_roster_from_json_when_the_column_is_absent():
    # Only the table's owner can add the column, so a developer's role reads a
    # table without it. The roster is in the trace document either way, and the
    # Guardrails view must not look empty just because the ALTER was denied.
    s = FakeServing(can_alter=False)
    s.list_voice_traces(limit=5)
    select = next(
        sql for sql, _ in s.cursor.executed if sql.strip().upper().startswith("SELECT TRACE_ID")
    )
    assert "trace -> 'guard_roster' AS guard_roster" in select


def test_list_prefers_the_promoted_column_when_it_exists():
    s = FakeServing(_V1_COLUMNS + list(_TRACE_MIGRATED_COLUMNS), can_alter=True)
    s.list_voice_traces(limit=5)
    select = next(
        sql for sql, _ in s.cursor.executed if sql.strip().upper().startswith("SELECT TRACE_ID")
    )
    assert "trace ->" not in select
    assert "guard_roster" in select


def test_guard_roster_absent_column_does_not_break_the_write():
    s = FakeServing(can_alter=False)
    s.insert_voice_trace({"trace_id": "t1", "total_ms": 1.0, "guard_roster": [{"guard_id": "x"}]})
    sql = _insert_sql(s)
    assert "guard_roster" not in sql
    # The roster still rides along inside the trace document.
    params = _insert_params(s)
    blob = json.loads([p for p in params if isinstance(p, str) and p.startswith("{")][-1])
    assert blob["guard_roster"] == [{"guard_id": "x"}]


def test_ttft_is_written_once_the_columns_exist():
    s = FakeServing(_V1_COLUMNS + list(_TRACE_LATENCY_COLUMNS), can_alter=True)
    s.insert_voice_trace(
        {
            "trace_id": "t1",
            "ttft_ms": 1234.5,
            "tts_first_ms": 900.0,
            "server_ttfb_ms": 700.0,
            "server_gen_ms": 1800.0,
            "total_ms": 6000.0,
        }
    )
    sql = _insert_sql(s)
    for column in _TRACE_LATENCY_COLUMNS:
        assert column in sql
    params = _insert_params(s)
    assert 1234.5 in params
    # Placeholder count must match the column list, or psycopg would raise.
    assert sql.count("%s") == len(params)


def test_write_still_succeeds_when_columns_are_absent():
    """A non-owner keeps tracing: latency drops out of the row, not the write."""
    s = FakeServing(can_alter=False)
    s.insert_voice_trace({"trace_id": "t1", "ttft_ms": 1234.5, "total_ms": 6000.0})

    sql = _insert_sql(s)
    for column in _TRACE_LATENCY_COLUMNS:
        assert column not in sql
    params = _insert_params(s)
    assert sql.count("%s") == len(params)
    assert 1234.5 not in params
    # ...and the full trace JSON still carries it, so nothing is actually lost.
    blob = json.loads([p for p in params if isinstance(p, str) and p.startswith("{")][-1])
    assert blob["ttft_ms"] == 1234.5


@pytest.mark.parametrize("value", [None, 0.0])
def test_absent_or_zero_ttft_round_trips_verbatim(value):
    s = FakeServing(_V1_COLUMNS + list(_TRACE_LATENCY_COLUMNS), can_alter=True)
    s.insert_voice_trace({"trace_id": "t1", "ttft_ms": value, "total_ms": 1.0})
    columns = _insert_sql(s).split("(", 1)[1].split(")", 1)[0].split(", ")
    params = _insert_params(s)
    assert params[columns.index("ttft_ms")] == value
