from typing import Dict, Any

from app.interfaces.embedder import Embedder
from app.interfaces.llm import LLM
from app.interfaces.vector_store import VectorStore


class AsyncComponentFactory:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @staticmethod
    async def create_llm(type: str, **kwargs) -> LLM:
        if type == "ollama":
            from app.impl.llms.ollama import AsyncOllamaLLM
            return AsyncOllamaLLM(**kwargs)
        raise ValueError(f"Unknown LLM type: {type}")

    @staticmethod
    async def create_embedder(type: str, **kwargs) -> Embedder:
        if type == "ollama":
            from app.impl.embedders.ollama_embedder import AsyncOllamaEmbedder
            return AsyncOllamaEmbedder(**kwargs)
        raise ValueError(f"Unknown embedder type: {type}")

    @staticmethod
    async def create_vector_store(type: str, embedder: Embedder, **kwargs) -> VectorStore:
        if type == "lancedb":
            from app.impl.vector_stores.lancedb_store import AsyncLanceDBStore
            return AsyncLanceDBStore(embedder=embedder, **kwargs)
        raise ValueError(f"Unknown vector store type: {type}")
