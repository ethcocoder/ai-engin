import torch
from typing import List

class ReasoningEngine:
    """
    Phase 2: Reasoning Engine Layer.
    Allows for vector mathematics indicating cognitive behavior such as concept blending, 
    extrapolation (prediction), and imagination.
    """
    def __init__(self, latent_dim: int = 128):
        self.latent_dim = latent_dim

    def blend(self, vector_a: torch.Tensor, vector_b: torch.Tensor, weight_a: float = 0.5) -> torch.Tensor:
        """
        Concept arithmetic: blends two latent concepts together.
        Example: vector(House) + vector(Boat) -> vector(HouseBoat concept)
        """
        return (vector_a * weight_a) + (vector_b * (1.0 - weight_a))

    def imagine(self, base_vector: torch.Tensor, noise_scale: float = 0.1) -> torch.Tensor:
        """
        Generates a 'new idea' by adding structured noise for latent exploration.
        This provides variability from a known deterministic state.
        """
        noise = torch.randn_like(base_vector) * noise_scale
        return base_vector + noise

    def extrapolate(self, vector_sequence: List[torch.Tensor], step_size: float = 1.0) -> torch.Tensor:
        """
        Temporal prediction: given a sequence of state vectors (t-2, t-1, t), 
        predict the next state (t+1) using linear extrapolation.
        """
        if len(vector_sequence) < 2:
            raise ValueError("Need at least 2 vectors to extrapolate.")
        
        # Basic derivative between last two states
        diff = vector_sequence[-1] - vector_sequence[-2]
        return vector_sequence[-1] + (diff * step_size)
