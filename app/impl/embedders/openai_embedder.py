from typing import List

import openai
from lancedb.embeddings import EmbeddingFunctionRegistry

from app.interfaces.embedder import Embedder
from app.interfaces.schemas import Document


class AsyncOpenAIEmbedder(Embedder):
    def __init__(
            self,
            api_key: str,
            model_name: str = "text-embedding-3-small",
            organization: str = None
    ):
        self.model_name = model_name
        self.client = openai.AsyncOpenAI(
            api_key=api_key,
            organization=organization
        )

        # Initialize LanceDB registry and embedder
        registry = EmbeddingFunctionRegistry.get_instance()
        self._lance_embedder = registry.get("openai").create(
            api_key=api_key,
            model=model_name
        )

    @property
    def SourceField(self):
        return self._lance_embedder.SourceField

    @property
    def VectorField(self):
        return self._lance_embedder.VectorField

    def ndims(self):
        return self._lance_embedder.ndims()

    async def _get_embedding(self, text: str) -> List[float]:
        response = await self.client.embeddings.create(
            model=self.model_name,
            input=text
        )
        return response.data[0].embedding

    async def embed_documents(self, documents: List[Document]) -> List[List[float]]:
        embeddings = []
        for doc in documents:
            print(doc)
            embedding = await self._get_embedding(doc.content)
            embeddings.append(embedding)
        return embeddings

    async def embed_query(self, query: str) -> List[float]:
        return await self._get_embedding(query)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass  # OpenAI client doesn't require explicit cleanup
