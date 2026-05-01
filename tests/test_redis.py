"""
Tests for redis_client.py
Run: pytest tests/test_redis.py -v
Requires: docker compose up redis
"""

import json
import redis.asyncio as aioredis


REDIS_URL = "redis://localhost:6379"


async def get_redis():
    client = aioredis.from_url(REDIS_URL, decode_responses=True)
    await client.flushdb()
    return client


# ─── Connection ───────────────────────────────────────────

async def test_connection():
    client = await get_redis()
    try:
        result = await client.ping()
        assert result is True
    finally:
        await client.aclose()


# ─── Session Buffer ──────────────────────────────────────

async def test_create_session_buffer():
    client = await get_redis()
    try:
        session_id = "sess_test_001"
        active_users = ["jane", "bob"]

        await client.hset(f"session:{session_id}", mapping={
            "speech_input_buffer": "",
            "speech_output_buffer": "",
            "retrieval_buffer": "[]",
            "active_users": json.dumps(active_users)
        })

        result = await client.hgetall(f"session:{session_id}")
        assert result["speech_input_buffer"] == ""
        assert json.loads(result["active_users"]) == ["jane", "bob"]
    finally:
        await client.aclose()


async def test_update_buffer_field():
    client = await get_redis()
    try:
        session_id = "sess_test_002"
        await client.hset(f"session:{session_id}", mapping={
            "speech_input_buffer": "",
            "speech_output_buffer": "",
            "retrieval_buffer": "[]",
            "active_users": "[]"
        })

        await client.hset(
            f"session:{session_id}",
            "speech_input_buffer",
            "Hey when is Jane moving?"
        )

        result = await client.hget(f"session:{session_id}", "speech_input_buffer")
        assert result == "Hey when is Jane moving?"
    finally:
        await client.aclose()


async def test_retrieval_buffer_json():
    client = await get_redis()
    try:
        session_id = "sess_test_003"
        retrieval_data = [
            {"topic_id": "topic_001", "activation": 2.31},
            {"topic_id": "topic_002", "activation": 1.47}
        ]

        await client.hset(
            f"session:{session_id}",
            "retrieval_buffer",
            json.dumps(retrieval_data)
        )

        raw = await client.hget(f"session:{session_id}", "retrieval_buffer")
        parsed = json.loads(raw)
        assert len(parsed) == 2
        assert parsed[0]["topic_id"] == "topic_001"
        assert parsed[0]["activation"] == 2.31
    finally:
        await client.aclose()


# ─── Messages List ───────────────────────────────────────

async def test_add_messages():
    client = await get_redis()
    try:
        session_id = "sess_test_004"
        key = f"messages:{session_id}"

        msg1 = json.dumps({"user_id": "jane", "content": "I found an apartment", "timestamp": 1000})
        msg2 = json.dumps({"user_id": "bob", "content": "Where?", "timestamp": 1001})

        await client.rpush(key, msg1)
        await client.rpush(key, msg2)

        messages = await client.lrange(key, 0, -1)
        assert len(messages) == 2
        assert json.loads(messages[0])["user_id"] == "jane"
        assert json.loads(messages[1])["user_id"] == "bob"
    finally:
        await client.aclose()


async def test_get_last_n_messages():
    client = await get_redis()
    try:
        session_id = "sess_test_005"
        key = f"messages:{session_id}"

        for i in range(20):
            msg = json.dumps({"user_id": "jane", "content": f"message {i}", "timestamp": 1000 + i})
            await client.rpush(key, msg)

        messages = await client.lrange(key, -5, -1)
        assert len(messages) == 5
        assert json.loads(messages[0])["content"] == "message 15"
        assert json.loads(messages[4])["content"] == "message 19"
    finally:
        await client.aclose()


async def test_trim_messages():
    client = await get_redis()
    try:
        session_id = "sess_test_006"
        key = f"messages:{session_id}"

        for i in range(100):
            msg = json.dumps({"user_id": "jane", "content": f"msg {i}", "timestamp": 1000 + i})
            await client.rpush(key, msg)

        await client.ltrim(key, -50, -1)

        length = await client.llen(key)
        assert length == 50

        first = json.loads((await client.lrange(key, 0, 0))[0])
        assert first["content"] == "msg 50"
    finally:
        await client.aclose()


# ─── User Session Mapping ────────────────────────────────

async def test_user_session_mapping():
    client = await get_redis()
    try:
        await client.set("user_session:jane", "sess_abc123")
        await client.set("user_session:bob", "sess_abc123")

        jane_session = await client.get("user_session:jane")
        bob_session = await client.get("user_session:bob")

        assert jane_session == "sess_abc123"
        assert bob_session == "sess_abc123"
    finally:
        await client.aclose()


async def test_delete_user_session():
    client = await get_redis()
    try:
        await client.set("user_session:jane", "sess_abc123")
        await client.delete("user_session:jane")

        result = await client.get("user_session:jane")
        assert result is None
    finally:
        await client.aclose()


# ─── TTL / Expiration ────────────────────────────────────

async def test_session_expiry():
    client = await get_redis()
    try:
        session_id = "sess_test_ttl"
        await client.hset(f"session:{session_id}", mapping={
            "speech_input_buffer": "test",
            "active_users": "[]"
        })

        await client.expire(f"session:{session_id}", 2)

        ttl = await client.ttl(f"session:{session_id}")
        assert ttl > 0
        assert ttl <= 2
    finally:
        await client.aclose()


async def test_refresh_session_ttl():
    client = await get_redis()
    try:
        session_id = "sess_test_refresh"
        key = f"session:{session_id}"

        await client.hset(key, mapping={"speech_input_buffer": "test"})
        await client.expire(key, 10)

        await client.expire(key, 3600)

        ttl = await client.ttl(key)
        assert ttl > 10
    finally:
        await client.aclose()


# ─── Full Session Lifecycle ──────────────────────────────

async def test_full_session_lifecycle():
    client = await get_redis()
    try:
        session_id = "sess_lifecycle"

        # 1. Create session
        await client.hset(f"session:{session_id}", mapping={
            "speech_input_buffer": "",
            "speech_output_buffer": "",
            "retrieval_buffer": "[]",
            "active_users": json.dumps(["jane", "bob"])
        })
        await client.set("user_session:jane", session_id)
        await client.set("user_session:bob", session_id)

        # 2. Simulate a message
        msg = json.dumps({"user_id": "jane", "content": "Hello!", "timestamp": 1000})
        await client.rpush(f"messages:{session_id}", msg)
        await client.hset(f"session:{session_id}", "speech_input_buffer", "Hello!")

        # 3. Simulate retrieval results
        retrieval = [{"topic_id": "t001", "activation": 2.31}]
        await client.hset(
            f"session:{session_id}", "retrieval_buffer", json.dumps(retrieval)
        )

        # 4. Verify state
        buffer = await client.hgetall(f"session:{session_id}")
        assert buffer["speech_input_buffer"] == "Hello!"
        assert json.loads(buffer["retrieval_buffer"])[0]["activation"] == 2.31

        messages = await client.lrange(f"messages:{session_id}", 0, -1)
        assert len(messages) == 1

        # 5. Cleanup session
        await client.delete(f"session:{session_id}")
        await client.delete(f"messages:{session_id}")
        await client.delete("user_session:jane")
        await client.delete("user_session:bob")

        assert await client.exists(f"session:{session_id}") == 0
        assert await client.get("user_session:jane") is None
    finally:
        await client.aclose()