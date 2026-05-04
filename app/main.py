from fastapi import FastAPI
from app.routes import chat, params, memory, topics, context
from app.core.production_rules import ProductionRules
from app.core.topic_extractor import TopicExtractor
from app.core.context_builder import ContextBuilder
from app.inference.model import Model
from app.db.pgclient import PgClient
from app.db.redis_client import RedisClient
import os

app = FastAPI(title="Context-Scaffold")

@app.on_event("startup")
async def startup():
    pg = PgClient()
    await pg.setup()

    redis_client = RedisClient()
    await redis_client.setup()

    extractor = TopicExtractor()
    ctx_builder = ContextBuilder(pg, extractor)

    # inject into routes
    chat.production_rules = ProductionRules(
        pg_client=pg,
        redis_client=redis_client,
        extractor=extractor,
        context_builder=ctx_builder,
    )
    memory.pg = pg
    topics.extractor = extractor
    context.context_builder = ctx_builder
    context.redis = redis_client

app.include_router(chat.router)
app.include_router(params.router)
app.include_router(memory.router)
app.include_router(topics.router)
app.include_router(context.router)

@app.get("/health")
async def health():
    return {"status": "ok"}