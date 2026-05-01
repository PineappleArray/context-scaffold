from db.pgclient import PgClient
from activation import Activation
from transformers import AutoTokenizer

class ContextBuilder:
    def __init__(self, pg_client: PgClient, activation: Activation):
        self.pg_client = pg_client
        self.activation = activation

    async def build_context(self, query_embedding: list[float], user_ids: list[str], current_time: float, decay: float = 0.5, noise_sigma: float = 0.25, retrieval_threshold: float = -1.0) -> list[dict]:
        rows = await self.pg_client.query_similar(query_embedding, n_results=100, user_ids=user_ids)
        context = []
        for row in rows:
            activation_result = self.activation.total_activation(
                topic_id=row["topic_id"],
                timestamps=row["timestamps"],
                current_time=current_time,
                query_embedding=query_embedding,
                topic_embedding=row["embedding"],
                decay=decay,
                noise_sigma=noise_sigma,
                retrieval_threshold=retrieval_threshold,
            )
            if activation_result.above_threshold:
                context.append({
                    "topic_id": row["topic_id"],
                    "content": row["content"],
                    "entity_type": row["entity_type"],
                    "confidence": row["confidence"],
                    "activation": activation_result.total_activation,
                    "activation_breakdown": activation_result.breakdown.dict(),
                })
        context.sort(key=lambda x: x["activation"], reverse=True)
        return context