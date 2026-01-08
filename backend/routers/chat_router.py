from fastapi import APIRouter
from pydantic import BaseModel
from backend.chat.chat_engine import ChatEngine


router = APIRouter()
chatbot = ChatEngine()

class ChatRequest(BaseModel):
    message: str
    metrics: dict | None = None

@router.post("/chat/")
async def chat_api(req: ChatRequest):
    reply = chatbot.chat(req.message, req.metrics)
    return {"reply": reply}
