# app/routes/memory.py
from fastapi import APIRouter
from app.models.schemas import (
    RetrieveRequest, RetrieveResponse,
    StoreRequest, StoreResponse,
    ReinforceRequest, ReinforceResponse
)
from app.db.pgclient import PgClient
from app.core.activation import Activation

router = APIRouter()

pg: PgClient = None

@router.post("/api/v1/memory/retrieve")
async def retrieve(request: RetrieveRequest) -> RetrieveResponse:
    rows = await pg.query_similar(
        request.query_embedding,
        request.max_topics,
        request.active_users
    )

    retrieved = []
    pruned = 0
    for row in rows:
        result = Activation(
            topic_id=row["topic_id"],
            timestamps=row["timestamps"],
            current_time=request.current_time,
            query_embedding=request.query_embedding,
            topic_embedding=list(row["embedding"]),
            retrieval_threshold=request.retrieval_threshold
        )
        if result.above_threshold:
            retrieved.append(result)
        else:
            pruned += 1

    return RetrieveResponse(
        retrieved=retrieved,
        pruned_below_threshold=pruned,
        total_candidates_scored=len(rows)
    )

@router.post("/api/v1/memory/store")
async def store(request: StoreRequest) -> StoreResponse:
    topic_id = f"usr_{request.user_id}_{request.session_id}_{request.timestamp}"
    await pg.add_topic(
        topic_id=topic_id,
        user_id=request.user_id,
        session_id=request.session_id,
        content=request.content,
        embedding=request.embedding,
        entity_type=request.entity_type,
        confidence=request.confidence,
        timestamps=[request.timestamp],
        access_count=0
    )
    return StoreResponse(
        topic_id=topic_id,
        stored=True,
        links_created=len(request.links),
        user_topic_count=0
    )

@router.patch("/api/v1/memory/reinforce")
async def reinforce(request: ReinforceRequest) -> ReinforceResponse:
    await pg.reinforce_topic(request.topic_id, request.access_timestamp)
    row = await pg.get_topic(request.topic_id)
    return ReinforceResponse(
        topic_id=request.topic_id,
        access_count=row["access_count"],
        timestamps=row["timestamps"],
        new_activation=0.0
    )