"""Regression test for the voice_traces ownership fix.

The serving role may have INSERT/SELECT on ``voice_traces`` but NOT own it. The
old code ran ``CREATE INDEX IF NOT EXISTS`` on every read/write, which raises
``InsufficientPrivilege`` and aborted the surrounding INSERT/SELECT — every voice
trace (billing AND card) failed and ``/traces`` 500'd.

The fix: probe with ``to_regclass`` (transaction-safe, never raises) and skip ALL
DDL when the table already exists, cached per process. These tests exercise that
branch logic with a fake cursor — no database required.
"""
from __future__ import annotations

from genie_voice.serve.lakebase import LakebaseServing


class FakeCursor:
    """Records executed SQL; returns a configurable to_regclass probe result."""

    def __init__(self, regclass_result):
        self._regclass_result = regclass_result
        self.executed: list[str] = []

    def execute(self, sql, params=None):
        self.executed.append(sql)

    def fetchone(self):
        return (self._regclass_result,)

    def joined(self) -> str:
        return " ".join(self.executed).upper()


class FakeSelf:
    """Minimal stand-in providing just what _ensure_traces_table touches."""

    def __init__(self):
        self.serving_schema_calls = 0

    def _table(self, name: str) -> str:
        return f'"serving"."{name}"'

    def _ensure_serving_schema(self, cur):
        self.serving_schema_calls += 1


def test_existing_table_skips_all_ddl():
    s = FakeSelf()
    cur = FakeCursor(regclass_result="serving.voice_traces")  # table exists
    LakebaseServing._ensure_traces_table(s, cur)

    sql = cur.joined()
    assert "TO_REGCLASS" in sql            # it probed
    assert "CREATE TABLE" not in sql       # ...and did NOT attempt owner-only DDL
    assert "CREATE INDEX" not in sql
    assert s.serving_schema_calls == 0
    assert s._traces_table_ready is True   # cached for next time


def test_missing_table_creates_table_and_indexes():
    s = FakeSelf()
    cur = FakeCursor(regclass_result=None)  # cold / fresh branch
    LakebaseServing._ensure_traces_table(s, cur)

    sql = cur.joined()
    assert "CREATE TABLE" in sql
    assert sql.count("CREATE INDEX") >= 2
    assert s.serving_schema_calls == 1
    assert s._traces_table_ready is True


def test_cached_ready_is_a_noop():
    s = FakeSelf()
    s._traces_table_ready = True
    cur = FakeCursor(regclass_result="anything")
    LakebaseServing._ensure_traces_table(s, cur)
    assert cur.executed == []  # not even the probe runs once cached
