import redis.asyncio as redis
import uuid
import json

class RedisClient:

    async def setup(self):
        try:
            self.client = redis.Redis(host='localhost', port=6379, decode_responses=True)
            await self.client.ping()
            print("Connected to Redis successfully!")
        except redis.ConnectionError as e:
            print(f"Failed to connect to Redis: {e}")
            raise e
    
    async def add_user(self, user_id, ttl=3600):
        await self.client.sadd("users", user_id)
        await self.client.expire(f"{user_id}:messages", ttl)

    async def refresh_session(self, session_id, ttl=3600):
        await self.client.expire(f"session:{session_id}", ttl)
        await self.client.expire(f"messages:{session_id}", ttl)

    async def add_messages(self, session_id, message_json):
        key = f"messages:{session_id}"
        await self.client.rpush(key, message_json)
        await self.client.ltrim(key, -50, -1)
    
    async def get_messages(self, session_id, n=10):
        await self.refresh_session(session_id)
        key = f"messages:{session_id}"
        return await self.client.lrange(key, -n, -1)
    
    async def create_session(self, active_users, ttl=3600):
        session_id = str(uuid.uuid4())
        await self.client.hset(f"session:{session_id}", mapping={
            "speech_input_buffer": "",
            "speech_output_buffer": "",
            "retrieval_buffer": "[]",
            "active_users": json.dumps(active_users)
        })
        await self.client.expire(f"session:{session_id}", ttl)
        return session_id
    
    async def set_buffer(self, session_id, field, value):
        await self.client.hset(f"session:{session_id}", field, value)

    async def get_buffer(self, session_id):
        return await self.client.hgetall(f"session:{session_id}")
    
    async def get_active_users(self):
        users = await self.client.smembers("users")
        return list(users)
    



