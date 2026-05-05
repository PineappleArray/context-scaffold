import numpy as np
from openai import AsyncOpenAI
from transformers import AutoTokenizer
from app.core.activation import total_activation
class Model:
    def __init__(self, base_url="http://localhost:8000/8000"):
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key="not-needed"
        )
        self.model = "google/gemma-4-E4B-it"

    async def generate(self, context_window):
        messages = [
            {"role": "system", "content": context_window.system_prompt}
        ]

        # add retrieved topic context
        for block in context_window.context_blocks:
            messages.append({
                "role": "system",
                "content": f"[Context] {block.content}"
            })

        # add recent messages
        for msg in context_window.recent_messages:
            messages.append({
                "role": "user",
                "content": f"[{msg.user_id}]: {msg.content}"
            })

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=512,
            temperature=0.7
        )

        return response.choices[0].message.content
        
    