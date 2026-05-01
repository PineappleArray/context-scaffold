import numpy as np
from app.models.schemas import (
    ActivationRequest,
    ActivationResult,
    ActivationBreakdown,
    ContextChunk,
    ControlParams,
)

class Activation:

    # B(m) = ln(Σ t_k^(-d))
    def base_level_learning(self, control_params: ControlParams) -> float:
        elapsed_times = control_params.current_time - np.array(control_params.timestamps)
        elapsed_times = elapsed_times[elapsed_times > 0]  # Filter out non-positive elapsed times
        if len(elapsed_times) == 0:
            return float("-inf")
        else:
            return np.log(np.sum(elapsed_times ** (-control_params.decay)))
        
    # SA = Σ W_j * cos_sim(source_j, topic)
    def spreading_activation(self, query_embedding: list[float],topic_embedding: list[float],context_chunks: list[ContextChunk] | None = None,) -> float:
        def cosine_similarity(vec_a, vec_b):
            arr_a = np.array(vec_a)
            arr_b = np.array(vec_b)

            if np.linalg.norm(arr_a) == 0 or np.linalg.norm(arr_b) == 0:
                return 0.0
            else:
                return np.dot(arr_a, arr_b) / (np.linalg.norm(arr_a) * np.linalg.norm(arr_b))
        spreading_activation = cosine_similarity(query_embedding, topic_embedding)

        if context_chunks:
            for chunk in context_chunks:
                spreading_activation += chunk.weight * cosine_similarity(chunk.embedding, topic_embedding)
        return spreading_activation

    def add_noise(self, sigma: float = 0.25) -> float:
        return np.random.normal(0, sigma)

    def total_activation(self, topic_id: str, timestamps: list[float], current_time: float, query_embedding: list[float], topic_embedding: list[float], decay: float = 0.5, noise_sigma: float = 0.25, retrieval_threshold: float = -1.0, context_chunks: list[ContextChunk] | None = None,
    ) -> ActivationResult:
        b = self.base_level_learning(timestamps, current_time, decay)
        spreading_activation = self.spreading_activation(query_embedding, topic_embedding, context_chunks)
        noise = self.add_noise(noise_sigma)
        total = b + spreading_activation + noise
        return ActivationResult(
            topic_id=topic_id,
            total_activation=total,
            breakdown=ActivationBreakdown(
                base_level=b,
                spreading=spreading_activation,
                noise=noise,
            ),
            above_threshold=total >= retrieval_threshold,
        )