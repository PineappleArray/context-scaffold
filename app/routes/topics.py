from fastapi import APIRouter
from app.models.schemas import ExtractRequest, ExtractResponse
from app.core.topic_extractor import TopicExtractor

router = APIRouter()

extractor: TopicExtractor = None

@router.post("/api/v1/topics/extract")
async def extract_topics(request: ExtractRequest) -> ExtractResponse:
    topics = extractor.extract_topics(request.message, top_n=5)
    return ExtractResponse(topics=topics)