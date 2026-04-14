import torch
from typing import List, Dict, Any, Optional

class LatentMemory:
    """
    Phase 1: Core Memory Layer for storing and retrieving abstract latent representations.
    Instead of using external DBs, this acts as an in-memory vector storage engine.
    """
    def __init__(self, latent_dim: int = 128):
        self.latent_dim = latent_dim
        self.memory_bank: List[Dict[str, Any]] = []

    def store(self, item_id: str, vector: torch.Tensor, metadata: dict = None):
        """Stores a latent vector in the memory bank."""
        flat_vector = vector.view(-1).cpu() 
        if flat_vector.shape[-1] != self.latent_dim:
            raise ValueError(f"Expected 1D latent_dim of {self.latent_dim}, got {flat_vector.shape[-1]}")
        
        self.memory_bank.append({
            'id': item_id,
            'vector': flat_vector,
            'metadata': metadata or {}
        })

    def search(self, query_vector: torch.Tensor, top_k: int = 5) -> List[Dict[str, Any]]:
        """Finds the most similar vectors using cosine similarity."""
        if not self.memory_bank:
            return []

        query_flat = query_vector.view(-1).cpu()
        
        # Stack all vectors in memory
        memory_matrix = torch.stack([item['vector'] for item in self.memory_bank])
        
        # Compute cosine similarity
        similarities = torch.nn.functional.cosine_similarity(query_flat.unsqueeze(0), memory_matrix)
        
        # Get top-k indices
        top_k = min(top_k, len(self.memory_bank))
        scores, indices = torch.topk(similarities, top_k)
        
        results = []
        for i in range(top_k):
            idx = indices[i].item()
            results.append({
                'id': self.memory_bank[idx]['id'],
                'score': scores[i].item(),
                'metadata': self.memory_bank[idx]['metadata']
            })
            
        return results

    def get(self, item_id: str) -> Optional[torch.Tensor]:
        """Retrieves a specific latent vector by ID."""
        for item in self.memory_bank:
            if item['id'] == item_id:
                return item['vector']
        return None
