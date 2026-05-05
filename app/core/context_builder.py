from transformers import AutoTokenizer
from app.db.pgclient import PgClient
from app.core.activation import total_activation
from app.models.schemas import ContextWindow, ContextBlock


class ContextBuilder:
    def __init__(self, pg_client: PgClient, activation: Activation):
        self.pg_client = pg_client
        self.activation = activation

    def build_context(self, retrieved_topics, recent_messages, max_tokens=4096):
        used = 0
        context_blocks = []

        # system prompt
        system = "You are a interacting with a user. Have a conversation with them. Use the provided context to inform your responses. If the context is not relevant, you can ignore it."
        used += self.count_tokens(system)

        # gets topics by their activation strength
        for topic in sorted(retrieved_topics, key=lambda t: t.activation, reverse=True):
            tokens = self.count_tokens(topic.content)
            if used + tokens > max_tokens:
                break
            context_blocks.append(ContextBlock(content=topic.content, sources=[topic.topic_id], token_count=tokens))
            used += tokens

        # keeps the most recent messages until we hit the token limit
        included_messages = []
        for msg in reversed(recent_messages):
            tokens = self.count_tokens(msg.content)
            if used + tokens > max_tokens:
                break
            included_messages.insert(0, msg)
            used += tokens

        return ContextWindow(
            system_prompt=system,
            context_blocks=context_blocks,
            recent_messages=included_messages,
            total_tokens=used,
            budget_remaining=max_tokens - used
        )
    
    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))
    
    def __init__(self, pg_client: PgClient, activation: Activation):
        self.pg_client = pg_client
        self.activation = activation
        self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-E4B-it")