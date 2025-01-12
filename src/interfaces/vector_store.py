from abc import ABC, abstractmethod
from typing import List, Dict, Any


class VectorStore(ABC):
    @abstractmethod
    async def store_embeddings(self, documents: List[str], embeddings: List[List[float]],
                               metadata: List[Dict[str, Any]] = None) -> None:
        """Store document embeddings with optional metadata"""
        pass

    @abstractmethod
    async def search(self, query_embedding: List[float], limit: int = 3) -> List[Dict[str, Any]]:
        """Search for similar documents using query embedding"""
        pass
