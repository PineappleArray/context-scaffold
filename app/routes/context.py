from fastapi import APIRouter
from app.models.schemas import ContextBuildRequest, ContextWindow
from app.core.context_builder import ContextBuilder
from app.db.redis_client import RedisClient
import json

router = APIRouter()

context_builder: ContextBuilder = None
redis: RedisClient = None

@router.post("/api/v1/context/build")
async def build_context(request: ContextBuildRequest) -> ContextWindow:
    raw_messages = await redis.get_messages(request.session_id, n=50)
    recent_messages = [json.loads(m) for m in raw_messages]

    return context_builder.build_context( 
        retrieved_topics=[],
        recent_messages=recent_messages,
        max_tokens=request.max_tokens
    )