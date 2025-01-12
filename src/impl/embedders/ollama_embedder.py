from typing import List

import httpx
from lancedb.embeddings import EmbeddingFunctionRegistry

from interfaces.embedder import Embedder


class AsyncOllamaEmbedder(Embedder):
    def __init__(self, model_name: str = "mxbai-embed-large", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.client = httpx.AsyncClient()

        # Initialize LanceDB registry and embedder
        registry = EmbeddingFunctionRegistry.get_instance()
        self._lance_embedder = registry.get("ollama").create(name=model_name)

    @property
    def SourceField(self):
        return self._lance_embedder.SourceField

    @property
    def VectorField(self):
        return self._lance_embedder.VectorField

    def ndims(self):
        return self._lance_embedder.ndims()

    async def _get_embedding(self, text: str) -> List[float]:
        response = await self.client.post(
            f"{self.base_url}/api/embeddings",
            json={
                "model": self.model_name,
                "prompt": text
            }
        )
        return response.json()['embedding']

    async def embed_documents(self, documents: List[str]) -> List[List[float]]:
        embeddings = []
        for doc in documents:
            embedding = await self._get_embedding(doc)
            embeddings.append(embedding)
        return embeddings

    async def embed_query(self, query: str) -> List[float]:
        return await self._get_embedding(query)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
