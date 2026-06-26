"""Genie analytics endpoint (Phase 2)."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..deps import genie

router = APIRouter(prefix="/genie", tags=["genie"])


class AskRequest(BaseModel):
    question: str
    # Pass the conversation_id returned by a previous /ask to send a follow-up in
    # the same thread (Genie keeps context). Omit to start a new conversation.
    conversation_id: str | None = None


@router.post("/ask")
def ask(req: AskRequest) -> dict:
    try:
        return genie().ask(req.question, conversation_id=req.conversation_id)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=str(exc)) from exc
