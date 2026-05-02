import asyncpg
from pgvector.asyncpg import register_vector
import time


class PgClient:
    def __init__(self, database_url):
        self.database_url = database_url
        self.conn = None

    async def setup(self):
        self.conn = await asyncpg.connect(self.database_url)
        await register_vector(self.conn)

    async def init_user(self, user_id, session_id):
        await self.conn.execute("""
            INSERT INTO user_profiles (user_id, active_session)
            VALUES ($1, $2)
            ON CONFLICT (user_id) DO UPDATE SET active_session = $2
        """, user_id, session_id)

    async def add_session(self, session_id, active_users, updated_at):
        await self.conn.execute("""
            INSERT INTO sessions (session_id, active_users, created_at, updated_at)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (session_id) DO UPDATE SET updated_at = $4
        """, session_id, active_users, updated_at, updated_at)

    async def add_topic(self, topic_id, user_id, session_id, content,
                        embedding, entity_type, confidence, timestamps, access_count):
        await self.conn.execute("""
            INSERT INTO topics (topic_id, user_id, session_id, content,
                embedding, entity_type, confidence, timestamps,
                access_count, created_at)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)
            ON CONFLICT (topic_id) DO UPDATE SET
                embedding = $5, confidence = $7,
                timestamps = $8, access_count = $9
        """, topic_id, user_id, session_id, content, embedding,
            entity_type, confidence, timestamps, access_count, time.time())

    async def delete_topic(self, topic_id):
        result = await self.conn.execute(
            "DELETE FROM topics WHERE topic_id = $1", topic_id)
        return "DELETE 1" in result

    async def query_similar(self, query_embedding, n_results, user_ids):
        return await self.conn.fetch("""
            SELECT topic_id, user_id, content, embedding,
                   timestamps, access_count, entity_type, confidence,
                   1 - (embedding <=> $1) AS similarity
            FROM topics
            WHERE user_id = ANY($2)
            ORDER BY embedding <=> $1
            LIMIT $3
        """, query_embedding, user_ids, n_results)

    async def get_topic(self, topic_id):
        return await self.conn.fetchrow(
            "SELECT * FROM topics WHERE topic_id = $1", topic_id)

    async def reinforce_topic(self, topic_id, timestamp):
        await self.conn.execute("""
            UPDATE topics
            SET timestamps = array_append(timestamps, $2),
                access_count = access_count + 1
            WHERE topic_id = $1
        """, topic_id, timestamp)

    async def close(self):
        if self.conn:
            await self.conn.close()