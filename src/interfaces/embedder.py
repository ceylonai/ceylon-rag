from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np


class Embedder(ABC):
    """
    Abstract base class for asynchronous document and query embedding implementations.

    This interface defines the contract for embedding documents and queries
    into vector representations that can be used by vector stores for similarity search.
    The interface provides asynchronous methods for embedding operations to support
    high-throughput and non-blocking implementations.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the embedder with optional configuration.

        Args:
            config: Dictionary containing embedder-specific configuration parameters
        """
        self.config = config or {}

    @abstractmethod
    async def embed_documents(self, documents: List[str]) -> np.ndarray:
        """
        Asynchronously convert a list of document texts into their vector representations.

        Args:
            documents: List of document texts to embed

        Returns:
            numpy.ndarray: Array of document embeddings with shape (n_documents, embedding_dim)
                         where embedding_dim is the dimension of the embedding space

        Raises:
            ValueError: If documents list is empty or contains invalid content
            RuntimeError: If embedding generation fails
        """
        pass

    @abstractmethod
    async def embed_query(self, query: str) -> np.ndarray:
        """
        Asynchronously convert a query text into its vector representation.

        Args:
            query: Query text to embed

        Returns:
            numpy.ndarray: Query embedding vector with shape (embedding_dim,)
                         where embedding_dim is the dimension of the embedding space

        Raises:
            ValueError: If query is empty or contains invalid content
            RuntimeError: If embedding generation fails
        """
        pass

    async def embed_queries(self, queries: List[str]) -> np.ndarray:
        """
        Asynchronously embed multiple queries. Default implementation processes
        queries sequentially, but implementations can override for batch processing.

        Args:
            queries: List of query texts to embed

        Returns:
            numpy.ndarray: Array of query embeddings with shape (n_queries, embedding_dim)
        """
        embeddings = []
        for query in queries:
            embedding = await self.embed_query(query)
            embeddings.append(embedding)
        return np.array(embeddings)

    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        """
        Get the dimension of the embedding space.

        Returns:
            int: Dimension of the embedding vectors produced by this embedder
        """
        pass

    @property
    def supports_batch_encoding(self) -> bool:
        """
        Whether the embedder supports efficient batch encoding of documents.

        Returns:
            bool: True if batch encoding is supported, False otherwise
        """
        return True

    async def validate_config(self) -> bool:
        """
        Asynchronously validate the current embedder configuration.
        Useful for configurations that require external resource validation.

        Returns:
            bool: True if configuration is valid, False otherwise

        Raises:
            ValueError: If configuration is invalid with description of issues
        """
        return True

    def get_config(self) -> Dict[str, Any]:
        """
        Get the current configuration of the embedder.

        Returns:
            Dict[str, Any]: Current configuration parameters
        """
        return self.config.copy()

    async def shutdown(self) -> None:
        """
        Perform any necessary cleanup operations when shutting down the embedder.
        Override this method if your embedder implementation needs to release
        resources or close connections.

        Returns:
            None
        """
        pass