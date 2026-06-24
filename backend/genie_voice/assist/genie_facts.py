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
