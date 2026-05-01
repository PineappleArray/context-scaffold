import redis.asyncio as redis

class redis:

    async def setup(self):
        try:
            self.client = redis.Redis(host='localhost', port=6379, decode_responses=True)
            await self.client.ping()
            print("Connected to Redis successfully!")
        except redis.ConnectionError as e:
            print(f"Failed to connect to Redis: {e}")
            raise e
    
    async def add_user(self, user_id):
        await self.client.sadd("users", user_id)

    async def add_messages(self, user_id, messages):
        key = f"{user_id}:messages"
        await self.client.hset(key, *messages)
        return True
    
    async def get_messages(self, user_id, n=10):
        key = f"{user_id}:messages"
        return await self.client.hget(key, *range(n))


