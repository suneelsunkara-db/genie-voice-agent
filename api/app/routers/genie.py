"""Genie analytics endpoint (Phase 2)."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..deps import genie

router = APIRouter(prefix="/genie", tags=["genie"])


class AskRequest(BaseModel):
    question: str


@router.post("/ask")
def ask(req: AskRequest) -> dict:
    try:
        return genie().ask(req.question)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=str(exc)) from exc
