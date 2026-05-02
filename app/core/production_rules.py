from app.core.topic_extractor import TopicExtractor
from app.core.activation import total_activation
from app.core.context_builder import ContextBuilder
from app.db.pgclient import PgClient
from app.db.redis_client import RedisClient
from sentence_transformers import SentenceTransformer

class ProductionRules:
    def __init__(self, pg_client, redis_client, extractor, context_builder):
        self.pg = pg_client
        self.redis = redis_client
        self.extractor = extractor
        self.context_builder = context_builder
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    async def process_input(self, message, user_id, session_id):
        topics = self.extractor.extract_topics(message) 
        #placeholder for postgres store logic - would need to check if topic already exists, update activation, etc.
        embedding = self.generate_embedding(message)
        #placeholder for pgvector query logic - would query with the message embedding, get similar topics, and then score them with total_activation
        tot_activation = total_activation()
        # 1. extract topics from message
        # 2. store new topics in postgres
        # 3. get embedding for the message
        # 4. query similar topics from postgres
        # 5. score each with total_activation
        # 6. reinforce topics that passed threshold
        # 7. update redis buffer state
        # 8. build context window
        # 9. return context for inference

    def should_store(self, topic, confidence_threshold=0.3):
        return topic.confidence >= confidence_threshold

    # Very primitive relevance check - can be improved with better heuristics or a relevance model
    def should_retrieve(self, message):
        return len(message.split()) > 3

    async def generate_embedding(self, text):
        return self.embed_model.encode(text).tolist()