import asyncpg
from pgvector.asyncpg import register_vector
from app.models.schemas import ExtractedTopic
import time


class PgClient:

    def __init__(self, database_url="postgresql://scaffold_user:scaffold_pass@localhost:5432/context_scaffold"):
        self.database_url = database_url
        self.pool = None

    async def setup(self):
        self.pool = await asyncpg.create_pool(self.database_url)
        async with self.pool.acquire() as conn:
            await register_vector(conn)
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

    async def close(self):
        if self.pool:
            await self.pool.close()

    # ─── Topics ───────────────────────────────────────────

    async def upsert_topic(self, topic_id, user_id, session_id, topic: ExtractedTopic, embedding):
        async with self.pool.acquire() as conn:
            await register_vector(conn)
            await conn.execute("""
                INSERT INTO topics (topic_id, user_id, session_id, content,
                    embedding, entity_type, confidence, timestamps,
                    access_count, decay_param, last_activation, created_at)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)
                ON CONFLICT (topic_id) DO UPDATE SET
                    embedding = $5, confidence = $7,
                    timestamps = $8, access_count = $9
            """, topic_id, user_id, session_id,
                topic.content, embedding, topic.entity_type.value,
                topic.confidence, [time.time()],
                0, 0.5, 0.0, time.time())

    async def get_topic(self, topic_id):
        async with self.pool.acquire() as conn:
            return await conn.fetchrow(
                "SELECT * FROM topics WHERE topic_id = $1", topic_id)

    async def get_user_topics(self, user_id):
        async with self.pool.acquire() as conn:
            return await conn.fetch(
                "SELECT * FROM topics WHERE user_id = $1", user_id)

    async def query_similar(self, query_embedding, n_results, user_ids):
        async with self.pool.acquire() as conn:
            await register_vector(conn)
            return await conn.fetch("""
                SELECT topic_id, user_id, content, embedding,
                       timestamps, access_count, entity_type, confidence,
                       1 - (embedding <=> $1) AS similarity
                FROM topics
                WHERE user_id = ANY($2)
                ORDER BY embedding <=> $1
                LIMIT $3
            """, query_embedding, user_ids, n_results)

    async def reinforce_topic(self, topic_id, timestamp):
        async with self.pool.acquire() as conn:
            await conn.execute("""
                UPDATE topics
                SET timestamps = array_append(timestamps, $2),
                    access_count = access_count + 1
                WHERE topic_id = $1
            """, topic_id, timestamp)

    async def update_topic_activation(self, topic_id, activation):
        async with self.pool.acquire() as conn:
            await conn.execute("""
                UPDATE topics SET last_activation = $2 WHERE topic_id = $1
            """, topic_id, activation)

    async def delete_topic(self, topic_id):
        async with self.pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM topics WHERE topic_id = $1", topic_id)
            return "DELETE 1" in result

    # ─── User Profiles ───────────────────────────────────

    async def upsert_user_profile(self, user_id, session_id=None):
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO user_profiles (user_id, active_session)
                VALUES ($1, $2)
                ON CONFLICT (user_id) DO UPDATE SET active_session = $2
            """, user_id, session_id)

    async def get_user_profile(self, user_id):
        async with self.pool.acquire() as conn:
            return await conn.fetchrow(
                "SELECT * FROM user_profiles WHERE user_id = $1", user_id)

    async def update_user_profile(self, user_id, retrieval_threshold=None,
                                   noise_param=None, speech_in_weight=None):
        async with self.pool.acquire() as conn:
            profile = await self.get_user_profile(user_id)
            if not profile:
                return None
            await conn.execute("""
                UPDATE user_profiles
                SET retrieval_threshold = $2,
                    noise_param = $3,
                    speech_in_weight = $4
                WHERE user_id = $1
            """, user_id,
                retrieval_threshold if retrieval_threshold is not None else profile["retrieval_threshold"],
                noise_param if noise_param is not None else profile["noise_param"],
                speech_in_weight if speech_in_weight is not None else profile["speech_in_weight"])

    # ─── Sessions ─────────────────────────────────────────

    async def create_session(self, session_id, active_user_ids):
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO sessions (session_id, active_users, created_at, updated_at)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (session_id) DO UPDATE SET
                    active_users = $2, updated_at = $4
            """, session_id, active_user_ids, time.time(), time.time())

    async def get_session(self, session_id):
        async with self.pool.acquire() as conn:
            return await conn.fetchrow(
                "SELECT * FROM sessions WHERE session_id = $1", session_id)

    async def update_session(self, session_id, active_user_ids):
        async with self.pool.acquire() as conn:
            await conn.execute("""
                UPDATE sessions
                SET active_users = $2, updated_at = $3
                WHERE session_id = $1
            """, session_id, active_user_ids, time.time())

    async def delete_session(self, session_id):
        async with self.pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM sessions WHERE session_id = $1", session_id)