import asyncpg
from pgvector.asyncpg import register_vector

class PgClient:
    async def init(self):
        self.pool = await asyncpg.create_pool(self.database_url)
        async with self.pool.acquire() as conn:
            await register_vector(conn)
            with open("app/db/schema.sql") as f:
                await conn.execute(f.read())

    async def init(self):
        self.pool = await asyncpg.create_pool(self.database_url)
        async with self.pool.acquire() as conn:
            await register_vector(conn)
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

    async def upsert_topic(self, topic):
        async with self.pool.acquire() as conn:
            await register_vector(conn)
            await conn.execute("""
                INSERT INTO topics (topic_id, user_id, session_id, content,
                    embedding, entity_type, confidence, timestamps,
                    access_count, created_at)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)
                ON CONFLICT (topic_id) DO UPDATE SET
                    embedding = $5, confidence = $7,
                    timestamps = $8, access_count = $9
            """, topic.topic_id, topic.user_id, topic.session_id,
                topic.content, topic.embedding, topic.metadata.entity_type,
                topic.metadata.confidence, topic.timestamps,
                topic.access_count, topic.created_at)

    async def query_similar(self, query_embedding, n_results, user_ids):
        async with self.pool.acquire() as conn:
            await register_vector(conn)
            rows = await conn.fetch("""
                SELECT topic_id, user_id, content, embedding,
                       timestamps, access_count, entity_type, confidence,
                       1 - (embedding <=> $1) AS similarity
                FROM topics
                WHERE user_id = ANY($2)
                ORDER BY embedding <=> $1
                LIMIT $3
            """, query_embedding, user_ids, n_results)
            return rows

    async def delete_topic(self, topic_id):
        async with self.pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM topics WHERE topic_id = $1", topic_id
            )
            return "DELETE 1" in result

    async def get_topic(self, topic_id):
        async with self.pool.acquire() as conn:
            return await conn.fetchrow(
                "SELECT * FROM topics WHERE topic_id = $1", topic_id
            )

    async def reinforce_topic(self, topic_id, timestamp):
        async with self.pool.acquire() as conn:
            await conn.execute("""
                UPDATE topics
                SET timestamps = array_append(timestamps, $2),
                    access_count = access_count + 1
                WHERE topic_id = $1
            """, topic_id, timestamp)