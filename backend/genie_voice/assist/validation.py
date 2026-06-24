"""Cross-validate Genie account reads against Lakebase and guard agent replies."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


_AMOUNT_RE = re.compile(r"\$[\d,]+(?:\.\d{1,2})?")
_COUNT_BEFORE_OVERDUE_RE = re.compile(
    r"\b(\d+)\s+overdue\s+invoices?\b",
    re.IGNORECASE,
)
_FIELD_INT_RE = re.compile(
    r"(?:overdue_invoice_count|declined_payment(?:_count)?|recent_declined_payments)"
    r"\s*[=:]\s*(\d+)",
    re.IGNORECASE,
)
_FIELD_AMOUNT_RE = re.compile(
    r"overdue_amount(?:_usd)?\s*[=:]\s*\$?([\d,]+(?:\.\d{1,2})?)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class AccountMetrics:
    overdue_invoice_count: int
    overdue_amount: float
    recent_declined_payments: int
    status: str | None = None

    def as_context_lines(self) -> list[str]:
        lines = [
            f"overdue_invoice_count: {self.overdue_invoice_count}",
            f"overdue_amount_usd: {self.overdue_amount:.2f}",
            f"recent_declined_payments: {self.recent_declined_payments}",
        ]
        if self.status:
            lines.append(f"customer_status: {self.status}")
        return lines


@dataclass
class ValidationResult:
    authoritative: AccountMetrics
    genie_metrics: AccountMetrics | None = None
    genie_validated: bool = False
    mismatches: list[str] = field(default_factory=list)
    genie_error: str | None = None


def _to_float(value: Any) -> float:
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return 0.0


def _to_int(value: Any) -> int:
    try:
        return int(float(str(value).replace(",", "")))
    except (TypeError, ValueError):
        return 0


def metrics_from_account(account: dict[str, Any] | None) -> AccountMetrics | None:
    if not account or not account.get("found"):
        return None
    summary = account.get("summary") or {}
    customer = account.get("customer") or {}
    return AccountMetrics(
        overdue_invoice_count=_to_int(summary.get("overdue_invoice_count")),
        overdue_amount=round(_to_float(summary.get("overdue_amount")), 2),
        recent_declined_payments=_to_int(summary.get("recent_declined_payments")),
        status=str(summary.get("status") or customer.get("status") or "") or None,
    )


def _normalize_column(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _metrics_from_row_dict(row: dict[str, Any]) -> AccountMetrics:
    normalized = {_normalize_column(str(k)): v for k, v in row.items()}

    def pick(*keys: str, default: Any = None) -> Any:
        for key in keys:
            norm = _normalize_column(key)
            if norm in normalized and normalized[norm] is not None:
                return normalized[norm]
        return default

    return AccountMetrics(
        overdue_invoice_count=_to_int(
            pick("overdue_invoice_count", "overdue_invoices", "num_overdue_invoices", default=0)
        ),
        overdue_amount=round(
            _to_float(pick("overdue_amount_usd", "overdue_amount", "total_overdue_amount", default=0)),
            2,
        ),
        recent_declined_payments=_to_int(
            pick(
                "recent_declined_payments",
                "declined_payment_count",
                "declined_payments",
                default=0,
            )
        ),
        status=str(pick("customer_status", "status") or "") or None,
    )


def parse_genie_metrics(genie_response: dict[str, Any]) -> AccountMetrics | None:
    """Parse Genie SQL rows using column metadata, with answer-text fallback."""
    columns = [str(c) for c in (genie_response.get("columns") or []) if c]
    rows = genie_response.get("rows") or []
    if columns and rows:
        first = rows[0]
        if isinstance(first, (list, tuple)):
            row_dict = {
                columns[i]: first[i] for i in range(min(len(columns), len(first)))
            }
            return _metrics_from_row_dict(row_dict)
        if isinstance(first, dict):
            return _metrics_from_row_dict(first)

    overdue_count: int | None = None
    overdue_amount: float | None = None
    declined: int | None = None
    status: str | None = None

    rows = genie_response.get("rows") or []
    if rows:
        flat: list[Any] = []
        for row in rows:
            if isinstance(row, (list, tuple)):
                flat.extend(row)
            elif isinstance(row, dict):
                flat.extend(row.values())
        nums = [_to_float(v) for v in flat if str(v).replace(".", "", 1).replace("-", "", 1).isdigit()]
        ints = [_to_int(v) for v in flat if str(v).strip().lstrip("-").isdigit()]
        if ints:
            overdue_count = ints[0]
        if len(ints) > 1:
            declined = ints[-1] if len(ints) >= 3 else ints[1]
        if nums:
            overdue_amount = round(max(nums), 2)

    answer = str(genie_response.get("answer") or "")
    if answer:
        for match in _FIELD_INT_RE.finditer(answer):
            key = match.group(0).lower()
            val = _to_int(match.group(1))
            if "declined" in key:
                declined = val
            else:
                overdue_count = val
        amount_match = _FIELD_AMOUNT_RE.search(answer)
        if amount_match:
            overdue_amount = round(_to_float(amount_match.group(1)), 2)
        count_match = _COUNT_BEFORE_OVERDUE_RE.search(answer)
        if count_match:
            overdue_count = _to_int(count_match.group(1))
        for amt in _AMOUNT_RE.findall(answer):
            candidate = round(_to_float(amt.lstrip("$")), 2)
            if candidate > 0:
                overdue_amount = candidate
                break
        status_match = re.search(
            r"(?:customer_)?status\s*[=:]\s*['\"]?([a-z_]+)['\"]?",
            answer,
            re.IGNORECASE,
        )
        if status_match:
            status = status_match.group(1).lower()

    if overdue_count is None and overdue_amount is None and declined is None:
        return None
    return AccountMetrics(
        overdue_invoice_count=overdue_count if overdue_count is not None else 0,
        overdue_amount=overdue_amount if overdue_amount is not None else 0.0,
        recent_declined_payments=declined if declined is not None else 0,
        status=status,
    )


def cross_validate_metrics(
    lakebase: AccountMetrics | None,
    genie: AccountMetrics | None,
    *,
    genie_error: str | None = None,
) -> ValidationResult:
    """Lakebase is authoritative when Genie disagrees or is unavailable."""
    if lakebase is None:
        return ValidationResult(
            authoritative=genie or AccountMetrics(0, 0.0, 0),
            genie_metrics=genie,
            genie_validated=False,
            genie_error=genie_error or "Lakebase account facts unavailable",
        )

    if genie is None:
        return ValidationResult(
            authoritative=lakebase,
            genie_metrics=None,
            genie_validated=False,
            genie_error=genie_error or "Genie metrics unavailable",
        )

    mismatches: list[str] = []
    if genie.overdue_invoice_count != lakebase.overdue_invoice_count:
        mismatches.append(
            f"overdue_invoice_count genie={genie.overdue_invoice_count} "
            f"lakebase={lakebase.overdue_invoice_count}"
        )
    if abs(genie.overdue_amount - lakebase.overdue_amount) > 0.02:
        mismatches.append(
            f"overdue_amount genie={genie.overdue_amount:.2f} "
            f"lakebase={lakebase.overdue_amount:.2f}"
        )
    if genie.recent_declined_payments != lakebase.recent_declined_payments:
        mismatches.append(
            f"recent_declined_payments genie={genie.recent_declined_payments} "
            f"lakebase={lakebase.recent_declined_payments}"
        )

    return ValidationResult(
        authoritative=lakebase,
        genie_metrics=genie,
        genie_validated=len(mismatches) == 0,
        mismatches=mismatches,
    )


def validate_close_eligible(
    actions: dict[str, Any],
    account: dict[str, Any] | None,
) -> tuple[bool, str | None]:
    """Account-aware guard before closing an issue on customer confirmation."""
    if not actions.get("payment_plan_requested") and not actions.get("waiver_requested"):
        return False, "close blocked: no payment plan or waiver was requested"

    if not account or not account.get("found"):
        return False, "close blocked: account facts unavailable"

    summary = account.get("summary") or {}
    overdue_count = _to_int(summary.get("overdue_invoice_count"))
    overdue_amount = _to_float(summary.get("overdue_amount"))

    if actions.get("payment_plan_requested") and overdue_count == 0 and overdue_amount <= 0:
        return False, "close blocked: no overdue balance for payment arrangement"

    if actions.get("waiver_requested"):
        overdue_inv = next(
            (i for i in (account.get("invoices") or []) if str(i.get("status")) == "overdue"),
            None,
        )
        if overdue_inv is None:
            return False, "close blocked: no overdue invoice to waive fee on"
        if _to_float(overdue_inv.get("late_fee")) <= 0:
            return False, "close blocked: overdue invoice has no late fee to waive"

    return True, None


def validate_reply_against_metrics(
    reply: str,
    metrics: AccountMetrics,
    *,
    issue_closed: bool = False,
) -> tuple[bool, list[str]]:
    """Reject agent replies that cite numbers contradicting authoritative metrics."""
    issues: list[str] = []
    text = reply.strip()
    if not text:
        return False, ["empty reply"]

    for count in {_to_int(m.group(1)) for m in _COUNT_BEFORE_OVERDUE_RE.finditer(text)}:
        if issue_closed and count > 0:
            issues.append(f"closed issue but reply cites {count} overdue invoices")
        elif not issue_closed and count != metrics.overdue_invoice_count:
            issues.append(
                f"reply cites {count} overdue invoices; expected {metrics.overdue_invoice_count}"
            )

    for field_count in {_to_int(m.group(1)) for m in _FIELD_INT_RE.finditer(text)}:
        if "declined" in text.lower() and field_count != metrics.recent_declined_payments:
            issues.append(
                f"reply cites {field_count} declined payments; "
                f"expected {metrics.recent_declined_payments}"
            )

    amounts = [round(_to_float(m.lstrip("$")), 2) for m in _AMOUNT_RE.findall(text)]
    if amounts and not issue_closed and metrics.overdue_amount > 0:
        if not any(abs(a - metrics.overdue_amount) <= 0.02 for a in amounts):
            issues.append(
                f"reply amounts {amounts} do not match overdue_amount {metrics.overdue_amount:.2f}"
            )
    if amounts and issue_closed and metrics.overdue_amount <= 0:
        large = [a for a in amounts if a >= 50]
        if large:
            issues.append(f"closed issue but reply cites large balance {large}")

    return len(issues) == 0, issues


def genie_customer_metrics_question(customer_id: str) -> str:
    return f"""For customer_id = '{customer_id}' ONLY (never aggregate other customers), return ONE row with:
- overdue_invoice_count (integer)
- overdue_amount_usd (decimal, sum of overdue invoice amounts)
- recent_declined_payments (integer, last 90 days)
- customer_status (text)

Use governed invoice, payment, and customer tables. Filter strictly to this customer_id.
Return only these four fields for this customer."""
