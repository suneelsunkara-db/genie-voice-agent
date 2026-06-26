"""Fetch customer metrics from Genie for cross-validation."""
from __future__ import annotations

from typing import Any

from genie_voice.assist.validation import (
    AccountMetrics,
    ValidationResult,
    cross_validate_metrics,
    genie_customer_metrics_question,
    metrics_from_account,
    parse_genie_metrics,
)


def fetch_validated_account_metrics(
    genie_client: Any,
    account: dict[str, Any] | None,
    customer_id: str | None,
    *,
    skip_genie_query: bool = False,
) -> ValidationResult:
    """Query Genie for customer metrics and cross-check against Lakebase facts."""
    lakebase = metrics_from_account(account)
    if not customer_id:
        return cross_validate_metrics(lakebase, None, genie_error="no customer_id")

    if skip_genie_query:
        if lakebase is None:
            return ValidationResult(
                authoritative=AccountMetrics(0, 0.0, 0),
                genie_metrics=None,
                genie_validated=False,
                genie_error="post_close_metrics_unavailable",
            )
        return ValidationResult(
            authoritative=lakebase,
            genie_metrics=None,
            genie_validated=False,
            genie_error="genie_metrics_skipped_post_close",
        )

    if lakebase is not None and account and account.get("found"):
        return ValidationResult(
            authoritative=lakebase,
            genie_metrics=None,
            genie_validated=False,
            genie_error="lakebase_authoritative",
        )

    genie_metrics: AccountMetrics | None = None
    genie_error: str | None = None
    try:
        result = genie_client.ask(genie_customer_metrics_question(customer_id))
        genie_metrics = parse_genie_metrics(result)
        if genie_metrics is None:
            genie_error = "Genie returned no parseable customer metrics"
    except Exception as exc:  # noqa: BLE001
        genie_error = str(exc)

    return cross_validate_metrics(lakebase, genie_metrics, genie_error=genie_error)


def genie_account_insight(genie_client: Any, customer_id: str | None) -> str | None:
    """Fetch a short natural-language account insight from Genie (no SQL).

    Intended to run OFF the live reply critical path (e.g. when a call is opened)
    and be cached in call state. The live FM reply then grounds on this cached
    text, so "Based on Genie insights" is truthful without putting Genie's
    multi-second latency in the per-utterance path.
    """
    if not customer_id:
        return None
    try:
        result = genie_client.ask(
            f"For customer_id = '{customer_id}' only (never aggregate other "
            "customers), state in two short sentences: the number of overdue "
            "invoices and the total overdue amount in USD; the number of declined "
            "payments in the last 90 days; and the current account status. "
            "Answer in plain English with the actual numbers. Do NOT ask any "
            "clarifying questions and do NOT include SQL or column names."
        )
    except Exception:  # noqa: BLE001 - insight is best-effort, never fatal
        return None
    # Prefer `answer` (the NL response with real numbers) over `description`
    # (which only paraphrases the question). Both are SQL-free.
    text = (result.get("answer") or result.get("description") or "").strip()
    if not text:
        return None
    # Genie sometimes returns a clarifying question instead of answering (its space
    # has a time-range clarification rule). A clarification has no data rows and is
    # phrased as a question - treat it as "no insight" so the live reply falls back
    # honestly instead of parroting Genie's question or claiming a false fact.
    rows = result.get("rows") or []
    if not rows and text.rstrip().endswith("?"):
        return None
    return text
