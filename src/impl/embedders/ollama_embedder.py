import asyncio

import aiohttp
import numpy as np
from typing import List, Dict, Any, Optional

from src.interfaces.embedder import Embedder


class OllamaEmbedder(Embedder):
    """
    Async embedder implementation using Ollama's API for generating embeddings.

    This implementation uses the Ollama API to generate embeddings for documents
    and queries. It supports batch processing and configurable models.
    """

    DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
    DEFAULT_BASE_URL = "http://localhost:11434"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Ollama embedder with configuration.

        Args:
            config: Dictionary containing configuration parameters:
                - base_url: Ollama API base URL (default: http://localhost:11434)
                - model: Model name to use for embeddings (default: nomic-embed-text)
                - timeout: Request timeout in seconds (default: 30)
                - batch_size: Maximum batch size for document embedding (default: 32)
        """
        super().__init__(config)
        self._base_url = self.config.get('base_url', self.DEFAULT_BASE_URL)
        self._model = self.config.get('model', self.DEFAULT_EMBEDDING_MODEL)
        self._timeout = aiohttp.ClientTimeout(total=self.config.get('timeout', 30))
        self._batch_size = self.config.get('batch_size', 32)
        self._session: Optional[aiohttp.ClientSession] = None
        self._embedding_dim: Optional[int] = None

    async def _ensure_session(self) -> None:
        """Ensure aiohttp session is created."""
        if self._session is None:
            self._session = aiohttp.ClientSession(timeout=self._timeout)

    async def _get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for a single text using Ollama API.

        Args:
            text: Text to embed

        Returns:
            numpy.ndarray: Embedding vector
        """
        await self._ensure_session()

        payload = {
            "model": self._model,
            "prompt": text,
            "options": {
                "embedding": True
            }
        }

        async with self._session.post(
                f"{self._base_url}/api/generate",
                json=payload
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(
                    f"Ollama API error (status {response.status}): {error_text}"
                )

            result = await response.json()
            if "embedding" not in result:
                raise RuntimeError("No embedding in Ollama API response")

            embedding = np.array(result["embedding"])

            # Set embedding dimension if not yet set
            if self._embedding_dim is None:
                self._embedding_dim = embedding.shape[0]

            return embedding

    async def embed_documents(self, documents: List[str]) -> np.ndarray:
        """
        Convert a list of documents into their vector representations.

        Args:
            documents: List of document texts to embed

        Returns:
            numpy.ndarray: Array of document embeddings
        """
        if not documents:
            raise ValueError("Documents list is empty")

        embeddings = []

        # Process in batches
        for i in range(0, len(documents), self._batch_size):
            batch = documents[i:i + self._batch_size]
            batch_embeddings = await asyncio.gather(
                *[self._get_embedding(doc) for doc in batch]
            )
            embeddings.extend(batch_embeddings)

        return np.array(embeddings)

    async def embed_query(self, query: str) -> np.ndarray:
        """
        Convert a query text into its vector representation.

        Args:
            query: Query text to embed

        Returns:
            numpy.ndarray: Query embedding vector
        """
        if not query.strip():
            raise ValueError("Query is empty")

        return await self._get_embedding(query)

    @property
    def embedding_dimension(self) -> int:
        """
        Get the dimension of the embedding space.

        Returns:
            int: Dimension of the embedding vectors

        Raises:
            RuntimeError: If embedding dimension is not yet determined
        """
        if self._embedding_dim is None:
            raise RuntimeError(
                "Embedding dimension not yet determined. "
                "Make at least one embedding request first."
            )
        return self._embedding_dim

    async def validate_config(self) -> bool:
        """
        Validate the configuration by testing connection to Ollama API.

        Returns:
            bool: True if configuration is valid

        Raises:
            ValueError: If configuration is invalid
        """
        try:
            # Test connection with a simple embedding request
            await self._ensure_session()
            await self._get_embedding("test")
            return True
        except Exception as e:
            raise ValueError(f"Invalid configuration: {str(e)}")

    async def shutdown(self) -> None:
        """
        Close the aiohttp session.
        """
        if self._session:
            await self._session.close()
            self._session = None