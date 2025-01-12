import asyncio
from typing import Dict, Any

import numpy as np
import pytest
from aioresponses import aioresponses

from src.impl.embedders.ollama_embedder import OllamaEmbedder

# Test data
MOCK_EMBEDDING = [0.1, 0.2, 0.3, 0.4]
TEST_DOCS = ["This is a test document", "Another test document"]
TEST_QUERY = "test query"

@pytest.fixture
def mock_response() -> Dict[str, Any]:
    """Fixture for mock Ollama API response"""
    return {
        "model": "nomic-embed-text",
        "embedding": MOCK_EMBEDDING
    }

@pytest.fixture
def embedder():
    """Fixture for OllamaEmbedder instance"""
    config = {
        "model": "nomic-embed-text",
        "batch_size": 2,
        "timeout": 5
    }
    return OllamaEmbedder(config)

@pytest.fixture
def mock_aiohttp():
    """Fixture for mocking aiohttp requests"""
    with aioresponses() as m:
        yield m

class TestOllamaEmbedder:
    """Test suite for OllamaEmbedder"""

    @pytest.mark.asyncio
    async def test_initialization(self, embedder):
        """Test embedder initialization"""
        assert embedder._model == "nomic-embed-text"
        assert embedder._batch_size == 2
        assert embedder._base_url == "http://localhost:11434"
        assert embedder._session is None

    @pytest.mark.asyncio
    async def test_embed_query(self, embedder, mock_aiohttp, mock_response):
        """Test query embedding"""
        # Mock the API response
        mock_aiohttp.post(
            f"{embedder._base_url}/api/generate",
            payload=mock_response
        )

        # Get embedding
        embedding = await embedder.embed_query(TEST_QUERY)

        # Verify the result
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (len(MOCK_EMBEDDING),)
        np.testing.assert_array_almost_equal(
            embedding, np.array(MOCK_EMBEDDING)
        )

    @pytest.mark.asyncio
    async def test_embed_documents(self, embedder, mock_aiohttp, mock_response):
        """Test document embedding"""
        # Mock the API response for each document
        for _ in TEST_DOCS:
            mock_aiohttp.post(
                f"{embedder._base_url}/api/generate",
                payload=mock_response
            )

        # Get embeddings
        embeddings = await embedder.embed_documents(TEST_DOCS)

        # Verify the results
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (len(TEST_DOCS), len(MOCK_EMBEDDING))
        for embedding in embeddings:
            np.testing.assert_array_almost_equal(
                embedding, np.array(MOCK_EMBEDDING)
            )

    @pytest.mark.asyncio
    async def test_empty_documents(self, embedder):
        """Test handling of empty document list"""
        with pytest.raises(ValueError, match="Documents list is empty"):
            await embedder.embed_documents([])

    @pytest.mark.asyncio
    async def test_empty_query(self, embedder):
        """Test handling of empty query"""
        with pytest.raises(ValueError, match="Query is empty"):
            await embedder.embed_query("   ")

    @pytest.mark.asyncio
    async def test_api_error(self, embedder, mock_aiohttp):
        """Test handling of API errors"""
        # Mock an API error response
        mock_aiohttp.post(
            f"{embedder._base_url}/api/generate",
            status=500,
            body="Internal Server Error"
        )

        with pytest.raises(RuntimeError, match="Ollama API error"):
            await embedder.embed_query(TEST_QUERY)

    @pytest.mark.asyncio
    async def test_invalid_response(self, embedder, mock_aiohttp):
        """Test handling of invalid API response"""
        # Mock response without embedding field
        mock_aiohttp.post(
            f"{embedder._base_url}/api/generate",
            payload={"model": "nomic-embed-text"}  # Missing embedding field
        )

        with pytest.raises(RuntimeError, match="No embedding in Ollama API response"):
            await embedder.embed_query(TEST_QUERY)

    @pytest.mark.asyncio
    async def test_embedding_dimension(self, embedder, mock_aiohttp, mock_response):
        """Test embedding dimension property"""
        # Before any embeddings
        with pytest.raises(RuntimeError, match="Embedding dimension not yet determined"):
            _ = embedder.embedding_dimension

        # Mock API response
        mock_aiohttp.post(
            f"{embedder._base_url}/api/generate",
            payload=mock_response
        )

        # After getting an embedding
        await embedder.embed_query(TEST_QUERY)
        assert embedder.embedding_dimension == len(MOCK_EMBEDDING)

    @pytest.mark.asyncio
    async def test_validate_config(self, embedder, mock_aiohttp, mock_response):
        """Test configuration validation"""
        # Mock successful API response
        mock_aiohttp.post(
            f"{embedder._base_url}/api/generate",
            payload=mock_response
        )

        assert await embedder.validate_config() is True

        # Test with API error
        embedder = OllamaEmbedder({"base_url": "http://invalid-url"})
        with pytest.raises(ValueError, match="Invalid configuration"):
            await embedder.validate_config()

    @pytest.mark.asyncio
    async def test_shutdown(self, embedder, mock_aiohttp, mock_response):
        """Test proper cleanup on shutdown"""
        # Create a session by making a request
        mock_aiohttp.post(
            f"{embedder._base_url}/api/generate",
            payload=mock_response
        )
        await embedder.embed_query(TEST_QUERY)

        # Verify session exists
        assert embedder._session is not None

        # Shutdown
        await embedder.shutdown()

        # Verify session is closed
        assert embedder._session is None

    @pytest.mark.asyncio
    async def test_batch_processing(self, embedder, mock_aiohttp, mock_response):
        """Test batch processing of documents"""
        # Create a larger document list
        many_docs = TEST_DOCS * 3  # 6 documents total

        # Mock API responses for all documents
        for _ in many_docs:
            mock_aiohttp.post(
                f"{embedder._base_url}/api/generate",
                payload=mock_response
            )

        # Get embeddings
        embeddings = await embedder.embed_documents(many_docs)

        # Verify results
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (len(many_docs), len(MOCK_EMBEDDING))

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, embedder, mock_aiohttp, mock_response):
        """Test handling of concurrent embedding requests"""
        # Mock API responses
        for _ in range(5):  # Multiple concurrent requests
            mock_aiohttp.post(
                f"{embedder._base_url}/api/generate",
                payload=mock_response
            )

        # Make concurrent requests
        tasks = [
            embedder.embed_query(f"query_{i}")
            for i in range(5)
        ]
        results = await asyncio.gather(*tasks)

        # Verify results
        assert len(results) == 5
        for embedding in results:
            assert isinstance(embedding, np.ndarray)
            np.testing.assert_array_almost_equal(
                embedding, np.array(MOCK_EMBEDDING)
            )