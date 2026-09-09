from fastapi import APIRouter, HTTPException
import httpx

from app.schemas.models import ChatRequest, ChatResponse
from app.services.chatbot import chat


router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
)


@router.post(
    "",
    response_model=ChatResponse,
)
def chat_endpoint(request: ChatRequest):
    """
    Ask a question about the knowledge base.
    """

    try:
        answer = chat(request.query)

        return ChatResponse(
            answer=answer,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Chat failed: {exc}",
        )