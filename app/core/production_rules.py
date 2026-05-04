from app.core.topic_extractor import TopicExtractor
from app.core.activation import Activation
from app.core.context_builder import ContextBuilder
from app.db.pgclient import PgClient
from app.db.redis_client import RedisClient
from sentence_transformers import SentenceTransformer
import time
import json

class ProductionRules:
    def __init__(self, pg_client, redis_client, extractor, context_builder):
        self.pg_client = pg_client
        self.redis = redis_client
        self.extractor = extractor
        self.context_builder = context_builder
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    async def process_input(self, message, user_id, session_id):
        topics = self.extractor.extract_topics(message) 
        for topic in topics:
            print(topic) 
            topic_id = f"topic:{user_id}_{uuid.uuid4().hex[:8]}"
            await self.pg_client.upsert_topic(topic, topic_id, user_id, session_id)

        embedding = self.generate_embedding(message)

        active_users = await self.redis.get_active_users()

        rows = await self.pg.query_similar(embedding, n_results=20, user_ids=active_users)

        # 2. score each with ACT-R activation
        scored = []
        for row in rows:
            result = Activation.total_activation(
                topic_id=row["topic_id"],
                timestamps=row["timestamps"],
                current_time=time.time(),
                query_embedding=embedding,
                topic_embedding=list(row["embedding"]),
                decay=0.5,
                noise_sigma=0.25,
                retrieval_threshold=-1.0
            )
        if result.above_threshold:
            scored.append(result)

        # 3. reinforce the ones that passed
        for s in scored:
            await self.pg.reinforce_topic(s.topic_id, time.time())

        messages = await self.redis.get_messages(session_id, n=50)
        recent_messages = [json.loads(m) for m in messages]

        # 4. pass to context builder
        context = self.context_builder.build_context(
            retrieved_topics=scored,
            recent_messages=recent_messages,
            max_tokens=4096
        )

        tot_activation = Activation.total_activation()
        new_activations = [act for act in tot_activation if act.above_threshold]

        self.redis.reinforce_topics([act.topic_id for act in new_activations])
        context_window = ContextBuilder.build_context_window(new_activations)

        

    def should_store(self, topic, confidence_threshold=0.3):
        return topic.confidence >= confidence_threshold

    # Very primitive relevance check - can be improved with better heuristics or a relevance model
    def should_retrieve(self, message):
        return len(message.split()) > 3

    async def generate_embedding(self, text):
        return self.embed_model.encode(text).tolist()