from abc import ABC, abstractmethod
from typing import List

from app.interfaces.schemas import Document


class VectorStore(ABC):
    @abstractmethod
    async def store_embeddings(self, documents: List[Document], embeddings: List[List[float]]) -> None:
        """Store document embeddings"""
        pass

    @abstractmethod
    async def search(self, query_embedding: List[float], limit: int = 3) -> List[Document]:
        """Search for similar documents using query embedding"""
        pass
