"""
Tests for pg_client.py
Run: pytest tests/test_pg.py -v
Requires: docker compose up postgres
"""

import pytest
import asyncpg
from pgvector.asyncpg import register_vector

DATABASE_URL = "postgresql://scaffold_user:scaffold_pass@localhost:5432/context_scaffold"


async def get_conn():
    conn = await asyncpg.connect(DATABASE_URL)
    await register_vector(conn)
    await conn.execute("DELETE FROM topics")
    await conn.execute("DELETE FROM user_profiles")
    await conn.execute("DELETE FROM sessions")
    return conn


# ─── Connection ───────────────────────────────────────────

async def test_connection():
    conn = await get_conn()
    try:
        result = await conn.fetchval("SELECT 1")
        assert result == 1
    finally:
        await conn.close()


async def test_pgvector_extension():
    conn = await get_conn()
    try:
        result = await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')"
        )
        assert result is True
    finally:
        await conn.close()


# ─── Topics CRUD ──────────────────────────────────────────

async def test_insert_topic():
    conn = await get_conn()
    try:
        embedding = [0.1] * 384
        await conn.execute("""
            INSERT INTO topics (topic_id, user_id, session_id, content, embedding,
                entity_type, confidence, timestamps, access_count, created_at)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)
        """, "test_001", "jane", "sess_abc", "moving to Denver",
            embedding, "life_event", 0.92, [1000.0, 2000.0], 0, 1714400000.0)

        row = await conn.fetchrow("SELECT * FROM topics WHERE topic_id = $1", "test_001")
        assert row is not None
        assert row["user_id"] == "jane"
        assert row["content"] == "moving to Denver"
    finally:
        await conn.close()


async def test_delete_topic():
    conn = await get_conn()
    try:
        embedding = [0.1] * 384
        await conn.execute("""
            INSERT INTO topics (topic_id, user_id, session_id, content, embedding,
                entity_type, confidence, timestamps, access_count, created_at)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)
        """, "test_del", "jane", "sess_abc", "delete me",
            embedding, "other", 0.5, [1000.0], 0, 1714400000.0)

        result = await conn.execute("DELETE FROM topics WHERE topic_id = $1", "test_del")
        assert "DELETE 1" in result

        row = await conn.fetchrow("SELECT * FROM topics WHERE topic_id = $1", "test_del")
        assert row is None
    finally:
        await conn.close()


# ─── Vector Similarity Search ────────────────────────────

async def test_query_similar():
    conn = await get_conn()
    try:
        emb_denver = [0.9] + [0.1] * 383
        emb_hiking = [0.1] + [0.9] * 383

        await conn.execute("""
            INSERT INTO topics (topic_id, user_id, session_id, content, embedding,
                entity_type, confidence, timestamps, access_count, created_at)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)
        """, "sim_001", "jane", "sess_abc", "moving to Denver",
            emb_denver, "life_event", 0.9, [1000.0], 0, 1714400000.0)

        await conn.execute("""
            INSERT INTO topics (topic_id, user_id, session_id, content, embedding,
                entity_type, confidence, timestamps, access_count, created_at)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)
        """, "sim_002", "jane", "sess_abc", "likes hiking",
            emb_hiking, "preference", 0.8, [1000.0], 0, 1714400000.0)

        query_emb = [0.85] + [0.15] * 383
        rows = await conn.fetch("""
            SELECT topic_id, content, 1 - (embedding <=> $1) AS similarity
            FROM topics
            WHERE user_id = ANY($2)
            ORDER BY embedding <=> $1
            LIMIT $3
        """, query_emb, ["jane"], 5)

        assert len(rows) == 2
        assert rows[0]["topic_id"] == "sim_001"
        assert rows[0]["similarity"] > rows[1]["similarity"]
    finally:
        await conn.close()


async def test_query_filters_by_user():
    conn = await get_conn()
    try:
        embedding = [0.5] * 384

        await conn.execute("""
            INSERT INTO topics (topic_id, user_id, session_id, content, embedding,
                entity_type, confidence, timestamps, access_count, created_at)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)
        """, "usr_jane_001", "jane", "sess_abc", "topic from jane",
            embedding, "other", 0.8, [1000.0], 0, 1714400000.0)

        await conn.execute("""
            INSERT INTO topics (topic_id, user_id, session_id, content, embedding,
                entity_type, confidence, timestamps, access_count, created_at)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)
        """, "usr_bob_001", "bob", "sess_abc", "topic from bob",
            embedding, "other", 0.8, [1000.0], 0, 1714400000.0)

        rows = await conn.fetch("""
            SELECT topic_id FROM topics
            WHERE user_id = ANY($1)
            ORDER BY embedding <=> $2
            LIMIT 10
        """, ["jane"], embedding)

        assert len(rows) == 1
        assert rows[0]["topic_id"] == "usr_jane_001"

        rows = await conn.fetch("""
            SELECT topic_id FROM topics
            WHERE user_id = ANY($1)
            ORDER BY embedding <=> $2
            LIMIT 10
        """, ["jane", "bob"], embedding)

        assert len(rows) == 2
    finally:
        await conn.close()


# ─── Timestamps / Reinforcement ──────────────────────────

async def test_reinforce_topic():
    conn = await get_conn()
    try:
        embedding = [0.1] * 384
        await conn.execute("""
            INSERT INTO topics (topic_id, user_id, session_id, content, embedding,
                entity_type, confidence, timestamps, access_count, created_at)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)
        """, "reinf_001", "jane", "sess_abc", "test reinforce",
            embedding, "other", 0.5, [1000.0], 0, 1714400000.0)

        await conn.execute("""
            UPDATE topics
            SET timestamps = array_append(timestamps, $2),
                access_count = access_count + 1
            WHERE topic_id = $1
        """, "reinf_001", 2000.0)

        row = await conn.fetchrow("SELECT * FROM topics WHERE topic_id = $1", "reinf_001")
        assert len(row["timestamps"]) == 2
        assert row["timestamps"][-1] == 2000.0
        assert row["access_count"] == 1
    finally:
        await conn.close()


# ─── User Profiles ───────────────────────────────────────

async def test_user_profile_crud():
    conn = await get_conn()
    try:
        await conn.execute("""
            INSERT INTO user_profiles (user_id, active_session, retrieval_threshold,
                noise_param, speech_in_weight)
            VALUES ($1,$2,$3,$4,$5)
        """, "jane", "sess_abc", -1.0, 0.25, 0.6)

        row = await conn.fetchrow("SELECT * FROM user_profiles WHERE user_id = $1", "jane")
        assert row is not None
        assert row["retrieval_threshold"] == -1.0

        await conn.execute("""
            UPDATE user_profiles SET retrieval_threshold = $2 WHERE user_id = $1
        """, "jane", -0.5)

        row = await conn.fetchrow("SELECT * FROM user_profiles WHERE user_id = $1", "jane")
        assert row["retrieval_threshold"] == -0.5
    finally:
        await conn.close()