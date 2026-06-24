"""Dynamic Genie space management.

We never hardcode a Genie space id. Instead the space is reconciled BY NAME:
  - trash any existing spaces whose title == databricks.genie_space_name
  - create a fresh space from a generated `serialized_space` payload that points
    at our Unity Catalog tables, with entity matching on categorical columns,
    example SQL, instructions, and benchmark questions.

Because our datagen tables declare informational PK/FK constraints, we omit
explicit join specs - Genie infers joins from the constraints.

CLI:  python -m genie_voice.genie.space        # DQ gate, then recreate by name
"""
from __future__ import annotations

import json
import secrets
from typing import Any

from genie_voice.config import Settings, get_settings

# Categorical columns to enable entity/format matching on (lifts NL accuracy).
_ENTITY_COLUMNS: dict[str, list[str]] = {
    "customers": ["segment", "region", "plan", "status"],
    "invoices": ["status"],
    "payments": ["status", "method"],
    "billing_adjustments": ["status_before", "status_after"],
    "gold_call_insights": ["primary_intent", "disposition", "sentiment_label", "resolution_status"],
    "agents": ["team"],
}

# Tables to expose in the space (logical names -> resolved to fqtn).
# Best practice (Databricks): keep the space tightly focused - few tables, only
# what's needed to answer the target questions. We expose the analytics-friendly
# tables and deliberately EXCLUDE raw per-utterance history rows - they're
# verbose, not analytics-shaped, and would dilute Genie's accuracy.
_SPACE_TABLES = [
    "gold_call_insights", "lb_call_facts_history", "customers", "invoices",
    "payments", "agents", "billing_adjustments",
]

# Best practice: prefer example SQL over text instructions; keep text concise,
# specific, and non-conflicting (joins, semantics, units, clarification rule).
_TEXT_INSTRUCTIONS = [
    "Domain: contact-center billing/account support. gold_call_insights = one row of NLP insights per call (intent, sentiment, disposition, next_best_action, summary); lb_call_facts_history is Lakebase CDF history for operational call metadata (agent_id, call_ts, duration_sec, csat).\n",
    "Use current call metadata by selecting the latest non-delete lb_call_facts_history row per call_id: row_number() over (partition by call_id order by _sort_by desc)=1 and _pg_change_type <> 'delete'.\n",
    "Joins: gold_call_insights.call_id = lb_call_facts_history.call_id after applying the latest-row filter; gold_call_insights.customer_id = customers.customer_id; gold_call_insights.mentioned_invoice_id = invoices.invoice_id; lb_call_facts_history.agent_id = agents.agent_id; invoices.customer_id = customers.customer_id; payments.invoice_id = invoices.invoice_id; billing_adjustments.customer_id = customers.customer_id; billing_adjustments.invoice_id = invoices.invoice_id; billing_adjustments.call_id links to call history operationally.\n",
    "Semantics: handle time = current lb_call_facts_history.duration_sec (seconds); CSAT = current lb_call_facts_history.csat (1-5). A call is resolved when gold_call_insights.resolution_status = 'resolved'. invoices.status='overdue' = unpaid past due_date; 'disputed' = billing dispute. billing_adjustments rows are written by live agent-assist (waiver/payment plan) and UPDATE the linked invoice; use reverted_at IS NULL for active adjustments. A customer is at cancellation risk when customers.status='at_risk' OR array_contains(gold_call_insights.all_intents,'cancellation_risk').\n",
    "Units: all dollar columns (invoices.amount, invoices.late_fee, gold_call_insights.mentioned_amount, customers.monthly_charge) are USD; round money to 2 decimals.\n",
    "Clarification: when a question about call volume or trends omits a time range, ask the user to specify the period (e.g. 'Which month or date range?') before answering.\n",
    "Instructions you must follow when providing summaries: cite the table/column names used; round money to 2 decimals and percentages to 1 decimal.\n",
]


def _example_sqls(fq) -> list[dict[str, Any]]:
    call_facts_history = fq("lb_call_facts_history")
    current_call_facts = (
        "(\n"
        "  SELECT *\n"
        "  FROM (\n"
        "    SELECT *, row_number() OVER (PARTITION BY call_id ORDER BY _sort_by DESC) AS _rn\n"
        f"    FROM {call_facts_history}\n"
        "    WHERE _pg_change_type IN ('insert', 'update_postimage', 'delete')\n"
        "  )\n"
        "  WHERE _rn = 1 AND _pg_change_type <> 'delete'\n"
        ")"
    )
    return [
        {
            "question": ["How many calls do we have by primary intent?"],
            "sql": [
                "SELECT primary_intent, count(*) AS calls\n",
                f"FROM {fq('gold_call_insights')}\n",
                "GROUP BY primary_intent\n",
                "ORDER BY calls DESC",
            ],
        },
        {
            "question": ["Which customers called about a billing dispute and have an overdue invoice?"],
            "sql": [
                "SELECT DISTINCT c.customer_id, c.full_name\n",
                f"FROM {fq('gold_call_insights')} g\n",
                f"JOIN {fq('customers')} c ON g.customer_id = c.customer_id\n",
                f"JOIN {fq('invoices')} i ON i.customer_id = c.customer_id\n",
                "WHERE array_contains(g.all_intents, 'billing_dispute')\n",
                "  AND i.status = 'overdue'",
            ],
        },
        {
            "question": ["What is the average handle time by agent team?"],
            "sql": [
                "SELECT a.team, avg(f.duration_sec) AS avg_handle_time_sec\n",
                f"FROM {current_call_facts} f\n",
                f"JOIN {fq('agents')} a ON f.agent_id = a.agent_id\n",
                "GROUP BY a.team\n",
                "ORDER BY avg_handle_time_sec DESC",
            ],
        },
        {
            "question": ["What is the average CSAT by primary intent?"],
            "sql": [
                "SELECT g.primary_intent, round(avg(f.csat), 2) AS avg_csat, count(*) AS calls\n",
                f"FROM {fq('gold_call_insights')} g\n",
                f"JOIN {current_call_facts} f ON g.call_id = f.call_id\n",
                "GROUP BY g.primary_intent\n",
                "ORDER BY avg_csat ASC",
            ],
        },
        {
            "question": ["Which customers are at risk of cancelling?"],
            "sql": [
                "SELECT DISTINCT c.customer_id, c.full_name, c.plan, c.tenure_months\n",
                f"FROM {fq('customers')} c\n",
                f"LEFT JOIN {fq('gold_call_insights')} g ON g.customer_id = c.customer_id\n",
                "WHERE c.status = 'at_risk'\n",
                "   OR array_contains(g.all_intents, 'cancellation_risk')",
            ],
        },
        {
            "question": ["What share of late_fee calls ended with resolution_status = resolved?"],
            "sql": [
                "SELECT round(100.0 * sum(CASE WHEN resolution_status = 'resolved' THEN 1 ELSE 0 END) / count(*), 1) AS pct_resolved\n",
                f"FROM {fq('gold_call_insights')}\n",
                "WHERE array_contains(all_intents, 'late_fee')",
            ],
        },
        {
            "question": ["Total disputed invoice amount by region this quarter."],
            "sql": [
                "SELECT c.region, round(sum(i.amount), 2) AS disputed_amount\n",
                f"FROM {fq('invoices')} i\n",
                f"JOIN {fq('customers')} c ON i.customer_id = c.customer_id\n",
                "WHERE i.status = 'disputed'\n",
                "  AND i.issue_date >= date_trunc('QUARTER', current_date())\n",
                "GROUP BY c.region\n",
                "ORDER BY disputed_amount DESC",
            ],
        },
        {
            "question": ["What is the average customer sentiment by plan?"],
            "sql": [
                "SELECT c.plan, round(avg(g.sentiment_score), 3) AS avg_sentiment\n",
                f"FROM {fq('gold_call_insights')} g\n",
                f"JOIN {fq('customers')} c ON g.customer_id = c.customer_id\n",
                "GROUP BY c.plan\n",
                "ORDER BY avg_sentiment ASC",
            ],
        },
        {
            "question": ["For autopay_issue calls, how many customers had a declined payment?"],
            "sql": [
                "SELECT count(DISTINCT g.customer_id) AS customers_with_declined_payment\n",
                f"FROM {fq('gold_call_insights')} g\n",
                f"JOIN {fq('payments')} p ON g.customer_id = p.customer_id\n",
                "WHERE array_contains(g.all_intents, 'autopay_issue')\n",
                "  AND p.status = 'declined'",
            ],
        },
        {
            "question": ["Which agents on the retention team have the highest CSAT?"],
            "sql": [
                "SELECT a.agent_id, a.full_name, round(avg(f.csat), 2) AS avg_csat, count(*) AS calls\n",
                f"FROM {current_call_facts} f\n",
                f"JOIN {fq('agents')} a ON f.agent_id = a.agent_id\n",
                "WHERE a.team = 'retention'\n",
                "GROUP BY a.agent_id, a.full_name\n",
                "ORDER BY avg_csat DESC",
            ],
        },
        {
            "question": ["Sum of overdue invoice amounts for customers in the enterprise segment."],
            "sql": [
                "SELECT round(sum(i.amount), 2) AS overdue_amount\n",
                f"FROM {fq('invoices')} i\n",
                f"JOIN {fq('customers')} c ON i.customer_id = c.customer_id\n",
                "WHERE i.status = 'overdue'\n",
                "  AND c.segment = 'enterprise'",
            ],
        },
    ]


def build_serialized_space(settings: Settings) -> str:
    fq = settings.fqtn
    from genie_voice.datagen.schema import MODEL, SAMPLE_QUESTIONS

    tables = sorted(
        [
            {
                "identifier": fq(name),
                "description": [
                    (
                        "Lakebase CDF history for operational call_facts. Use the latest "
                        "non-delete row per call_id ordered by _sort_by for current call "
                        "metadata such as agent_id, call_ts, duration_sec, csat."
                    )
                    if name == "lb_call_facts_history"
                    else MODEL[name].comment
                ],
                "column_configs": sorted(
                    [
                        {
                            "column_name": col,
                            "enable_entity_matching": True,
                            "enable_format_assistance": True,
                        }
                        for col in _ENTITY_COLUMNS.get(name, [])
                    ],
                    key=lambda x: x["column_name"],
                ),
            }
            for name in _SPACE_TABLES
        ],
        key=lambda x: x["identifier"],
    )

    q_ids = sorted(secrets.token_hex(16) for _ in SAMPLE_QUESTIONS)
    sample_questions = sorted(
        [{"id": q_ids[i], "question": [SAMPLE_QUESTIONS[i]]} for i in range(len(SAMPLE_QUESTIONS))],
        key=lambda x: x["id"],
    )

    examples = _example_sqls(fq)
    ex_ids = sorted(secrets.token_hex(16) for _ in examples)
    example_question_sqls = sorted(
        [
            {"id": ex_ids[i], "question": examples[i]["question"], "sql": examples[i]["sql"]}
            for i in range(len(examples))
        ],
        key=lambda x: x["id"],
    )

    benchmarks = sorted(
        [
            {
                "id": secrets.token_hex(16),
                "question": [ex["question"][0]],
                "answer": [{"format": "SQL", "content": ex["sql"]}],
            }
            for ex in examples
        ],
        key=lambda x: x["id"],
    )

    config = {
        "version": 2,
        "config": {"sample_questions": sample_questions},
        "data_sources": {"tables": tables},
        "instructions": {
            "text_instructions": [{"id": secrets.token_hex(16), "content": _TEXT_INSTRUCTIONS}],
            "example_question_sqls": example_question_sqls,
        },
        "benchmarks": {"questions": benchmarks},
    }
    return json.dumps(config)


def find_space_id(client, name: str) -> str | None:
    """Resolve a Genie space id by exact title.

    `client.genie.list_spaces()` returns a `GenieListSpacesResponse` (spaces live
    under `.spaces`, paginated via `next_page_token`) - it is NOT directly
    iterable, so we read `.spaces` and follow pagination.
    """
    try:
        token: str | None = None
        while True:
            resp = client.genie.list_spaces(page_token=token) if token else client.genie.list_spaces()
            for space in (getattr(resp, "spaces", None) or []):
                if getattr(space, "title", None) == name:
                    return space.space_id
            token = getattr(resp, "next_page_token", None)
            if not token:
                return None
    except Exception:  # noqa: BLE001
        return None


def find_space_ids(client, name: str) -> list[str]:
    """Resolve all Genie space ids whose title exactly matches `name`."""
    out: list[str] = []
    try:
        token: str | None = None
        while True:
            resp = client.genie.list_spaces(page_token=token) if token else client.genie.list_spaces()
            for space in (getattr(resp, "spaces", None) or []):
                if getattr(space, "title", None) == name:
                    sid = getattr(space, "space_id", None)
                    if sid:
                        out.append(sid)
            token = getattr(resp, "next_page_token", None)
            if not token:
                return out
    except Exception:  # noqa: BLE001
        return out


def ensure_space(settings: Settings | None = None, *, require_quality: bool = True) -> str | None:
    """Recreate the space by configured name after the quality gate passes."""
    settings = settings or get_settings()
    if require_quality:
        from genie_voice.databricks.data_quality import run_data_quality

        run_data_quality(settings)
    name = settings.databricks.genie_space_name
    wh = settings.databricks.sql_warehouse_id
    if not wh:
        raise RuntimeError("genie: sql_warehouse_id is required to create a Genie space.")

    from genie_voice.databricks.client import current_user, get_workspace_client

    client = get_workspace_client(settings)
    serialized = build_serialized_space(settings)
    parent = f"/Users/{settings.databricks.run_as or current_user(client)}"

    # Recreate by name on every deployment so the space never keeps stale
    # serialized curation and reruns do not create duplicate active spaces.
    trash_failures: list[str] = []
    for existing in find_space_ids(client, name):
        try:
            client.genie.trash_space(existing)
            print(f"genie: trashed existing space '{name}' ({existing})")
        except Exception as exc:  # noqa: BLE001
            try:
                client.api_client.do("DELETE", f"/api/2.0/genie/spaces/{existing}")
                print(f"genie: trashed existing space '{name}' ({existing})")
            except Exception as exc2:  # noqa: BLE001
                print(f"genie: could not trash existing space '{name}' ({existing}): {exc2 or exc}")
                trash_failures.append(existing)
    if trash_failures:
        raise RuntimeError(
            "Refusing to create a new Genie space because existing matching spaces "
            "could not be trashed: " + ", ".join(trash_failures)
        )

    try:
        space = client.genie.create_space(
            warehouse_id=wh, serialized_space=serialized,
            title=name, parent_path=parent,
        )
        sid = getattr(space, "space_id", None)
        if not sid:
            raise RuntimeError("Genie create_space returned no space_id")
        print(f"genie: created space '{name}' ({sid})")
        return sid
    except Exception as exc:  # noqa: BLE001
        # Fallback to raw REST if the installed SDK lacks create_space.
        try:
            resp = client.api_client.do(
                "POST", "/api/2.0/genie/spaces",
                body={"serialized_space": serialized, "warehouse_id": wh,
                      "title": name, "parent_path": parent},
            )
            sid = resp.get("space_id") if isinstance(resp, dict) else None
            if not sid:
                raise RuntimeError("Genie create_space REST response returned no space_id")
            print(f"genie: created space '{name}' ({sid})")
            return sid
        except Exception as exc2:  # noqa: BLE001
            raise RuntimeError(f"genie: could not create space '{name}': {exc2 or exc}") from exc2


def main() -> None:
    sid = ensure_space()
    if sid:
        s = get_settings()
        host = s.databricks_host.rstrip("/")
        print(f"genie space ready: {host}/genie/rooms/{sid}")


if __name__ == "__main__":
    main()
