"""Simple question/answer endpoint for iOS Shortcuts ("Hey Siri, Ask Jarvis").

``POST /api/ask`` with ``{"question": "..."}`` runs the server's agent (the
read-only personal agent by default) and returns the answer plus a version
cleaned up for speech. It sits under ``/api`` so the API key is required.
"""

from __future__ import annotations

import asyncio
import re

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api", tags=["assistant"])

_MAX_QUESTION = 2000
_CITATION = re.compile(r"\s*\[(?:m\d+|e[0-9a-f]{6,10})\]")
_URL = re.compile(r"https?://\S+")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=_MAX_QUESTION)


def spoken_text(text: str) -> str:
    """Answer text suitable for a voice: no citations, markdown or links."""
    text = re.sub(r"(?is)<think>.*?</think>", "", text or "")
    text = _CITATION.sub("", text)
    text = _URL.sub("", text)
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[*_`]{1,3}([^*_`]+)[*_`]{1,3}", r"\1", text)
    text = re.sub(r"\s*\n\s*", ". ", text.strip())
    text = re.sub(r"\.\s*\.", ".", text)
    return " ".join(text.split())


@router.post("/ask")
async def ask(body: AskRequest, request: Request) -> dict:
    agent = getattr(request.app.state, "agent", None)
    if agent is None:
        raise HTTPException(status_code=503, detail="No assistant agent is configured.")
    question = body.question.strip()
    try:
        result = await asyncio.to_thread(agent.run, question)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"The assistant failed: {exc}")
    answer = (getattr(result, "content", "") or "").strip()
    return {"answer": answer, "spoken": spoken_text(answer)}
