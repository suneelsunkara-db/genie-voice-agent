"""Account-facts endpoints backed by governed UC reference tables."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..deps import serving

router = APIRouter(prefix="/accounts", tags=["accounts"])


@router.get("/with-issues")
def list_customers_with_issues() -> dict:
    """Customers with billing/account risk (overdue, at-risk, declined pay, etc.)."""
    rows = serving().list_customers_with_issues()
    return {"customers": rows, "count": len(rows)}


@router.get("/{customer_id}")
def get_account(customer_id: str) -> dict:
    facts = serving().get_account_facts(customer_id)
    if not facts.get("found"):
        raise HTTPException(status_code=404, detail=f"No account facts for {customer_id}")
    return facts
