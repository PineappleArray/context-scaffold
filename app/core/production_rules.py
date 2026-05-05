from app.core.topic_extractor import TopicExtractor
from app.core.activation import spreading_activation, base_level_learning, add_noise, total_activation
from app.core.context_builder import ContextBuilder
from app.db.pgclient import PgClient
from app.db.redis_client import RedisClient
from app.inference.model import Model
from app.models.schemas import (
    ChatResponse, RetrievedTopicSummary, ExtractedTopicSummary
)
from sentence_transformers import SentenceTransformer
import time
import json
import uuid


class ProductionRules:
    def __init__(self, pg_client, redis_client, extractor, context_builder, llm_client):
        self.pg = pg_client
        self.redis = redis_client
        self.extractor = extractor
        self.context_builder = context_builder
        self.llm = llm_client
        self.embed_model = extractor.embed_model

    async def process_input(self, message, user_id, session_id) -> ChatResponse:

        if not self.should_retrieve(message):
            response = await self.llm.generate_raw([
                {"role": "user", "content": message}
            ])
            return ChatResponse(
                session_id=session_id,
                response=response
            )

        # 2. extract topics from message
        topics = self.extractor.extract_topics(message)
        new_topics = []

        for topic in topics:
            if not self.should_store(topic):
                continue

            topic_id = f"topic:{user_id}_{uuid.uuid4().hex[:8]}"
            embedding = self.generate_embedding(topic.content)
            await self.pg.upsert_topic(topic_id, user_id, session_id, topic, embedding)

            new_topics.append(ExtractedTopicSummary(
                topic_id=topic_id,
                content=topic.content,
                entity_type=topic.entity_type
            ))

        # 3. generate query embedding from the full message
        query_embedding = self.generate_embedding(message)

        # 4. get active users from redis
        buffer = await self.redis.get_buffer(session_id)
        if buffer and "active_users" in buffer:
            active_users = json.loads(buffer["active_users"])
        else:
            active_users = [user_id]

        # 5. query similar topics from postgres
        rows = await self.pg.query_similar(query_embedding, n_results=20, user_ids=active_users)

        # 6. score each with ACT-R activation
        scored = []
        pruned = 0
        for row in rows:
            result = total_activation(
                topic_id=row["topic_id"],
                timestamps=list(row["timestamps"]),
                current_time=time.time(),
                query_embedding=query_embedding,
                topic_embedding=list(row["embedding"]),
                decay=0.5,
                noise_sigma=0.25,
            retrieval_threshold=-1.0
            )
        if result.above_threshold:
            scored.append({
                "topic_id": result.topic_id,
                "content": row["content"],
                "user_id": row["user_id"],
                "total_activation": result.total_activation,
                "breakdown": result.breakdown
            })
        else:
            pruned += 1

        # 7. reinforce topics that passed threshold
        for s in scored:
            await self.pg.reinforce_topic(s.topic_id, time.time())
            await self.pg.update_topic_activation(s.topic_id, s.total_activation)

        # 8. update redis buffer
        await self.redis.set_buffer(session_id, "speech_input_buffer", message)
        await self.redis.set_buffer(
            session_id, "retrieval_buffer",
            json.dumps([{"topic_id": s.topic_id, "activation": s.total_activation} for s in scored])
        )

        # 9. store message in redis history
        msg_json = json.dumps({
            "user_id": user_id,
            "content": message,
            "timestamp": time.time()
        })
        await self.redis.add_messages(session_id, msg_json)
        await self.redis.refresh_session(session_id)

        # 10. get recent messages for context
        raw_messages = await self.redis.get_messages(session_id, n=50)
        recent_messages = [json.loads(m) for m in raw_messages]

        # 11. build context window
        context = self.context_builder.build_context(
            retrieved_topics=scored,
            recent_messages=recent_messages,
            max_tokens=2048
        )

        # 12. generate response
        response = await self.llm.generate(context)

        # 13. store assistant response in redis history
        response_json = json.dumps({
            "user_id": "assistant",
            "content": response,
            "timestamp": time.time()
        })
        await self.redis.add_message(session_id, response_json)
        await self.redis.set_buffer(session_id, "speech_output_buffer", response)

        # 14. return structured response
        return ChatResponse(
            session_id=session_id,
            response=response,
            retrieved_topics=[
                RetrievedTopicSummary(
                    topic_id=s["topic_id"],
                    content=s["content"],
                    activation=s["total_activation"],
                    source_user=s["user_id"]
                ) for s in scored
            ],
            new_topics_extracted=new_topics,
            context_window_tokens_used=context.total_tokens,
            topics_pruned=pruned
        )

    def should_store(self, topic, confidence_threshold=0.3):
        return topic.confidence >= confidence_threshold

    def should_retrieve(self, message):
        return len(message.split()) > 3

    def generate_embedding(self, text):
        return self.embed_model.encode(text).tolist()