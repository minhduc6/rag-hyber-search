from typing import List, Any
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session, joinedload
from app.db.session import get_db
from app.models.user import User
from app.models.chat import Chat, Message
from app.models.knowledge import KnowledgeBase
from app.schemas.chat import AgentRequest
from app.schemas.chat import (
    ChatCreate,
    ChatResponse,
    ChatUpdate,
    MessageCreate,
    MessageResponse
)
from app.services.chat_service import generate_response
from fastapi.responses import JSONResponse
import os

router = APIRouter()
CHAT_ID = int(os.environ.get("CHAT_ID", "1"))  # Set a default/fallback chat_id for testing

@router.post("/agentRequest")
async def test_agent_message(
    *,
    db: Session = Depends(get_db),
    request_data: AgentRequest
):
    chat_id = CHAT_ID
    chat = (
        db.query(Chat)
        .options(joinedload(Chat.knowledge_bases))
        .filter(Chat.id == chat_id)
        .first()
    )
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    last_message = {
        "role": "user",
        "content": request_data.message
    }
    if last_message["role"] != "user":
        raise HTTPException(status_code=400, detail="Last message must be from user")

    # Tạo list full messages gồm history + last message
    history = []
    if request_data.history:
        history = [{"role": h.role, "content": h.content} for h in request_data.history]

    full_messages = history + [last_message]

    knowledge_base_ids = [kb.id for kb in chat.knowledge_bases]
    response_content = ""
    async for chunk in generate_response(
            query=request_data.message,
            messages={"messages": full_messages},
            knowledge_base_ids=knowledge_base_ids,
            chat_id=chat_id,
            db=db
        ):
        response_content += chunk

    return JSONResponse(content={"reply": response_content})



