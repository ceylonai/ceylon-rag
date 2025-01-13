import asyncio
from typing import Dict, Any

import pandas as pd

from factory.component_factory import AsyncComponentFactory


class BBCNewsRAG:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.factory = AsyncComponentFactory(config)
        self.llm = None
        self.embedder = None
        self.vector_store = None

    async def initialize(self):
        """Initialize all RAG components"""
        self.llm = await self.factory.create_llm(**self.config["llm"])
        self.embedder = await self.factory.create_embedder(**self.config["embedder"])
        self.vector_store = await self.factory.create_vector_store(
            embedder=self.embedder,
            **self.config["vector_store"]
        )

    async def ingest_data(self, df: pd.DataFrame):
        """Ingest the BBC news data from pandas DataFrame into the vector store"""
        documents = df['text'].tolist()
        embeddings = await self.embedder.embed_documents(documents)

        metadata = df.apply(
            lambda row: {
                'title': row['title'],
                'url': row['url'],
                'index': row['index']
            },
            axis=1
        ).tolist()

        await self.vector_store.store_embeddings(
            documents=documents,
            embeddings=embeddings,
            metadata=metadata
        )

    async def query(self, question: str, system_prompt: str = None) -> str:
        """Query the RAG system"""
        query_embedding = await self.embedder.embed_query(question)
        results = await self.vector_store.search(query_embedding)

        response = await self.llm.generate(
            prompt=f"Question: {question}\nContext: {results}",
            system_prompt=system_prompt or "Answer the question based on the provided context."
        )
        return response

    async def close(self):
        """Clean up resources"""
        if self.llm:
            await self.llm.__aexit__(None, None, None)
        if self.embedder:
            await self.embedder.__aexit__(None, None, None)


async def main():
    # Configuration
    config = {
        "llm": {
            "type": "ollama",
            "model_name": "llama3.2"
        },
        "embedder": {
            "type": "ollama",
            "model_name": "nomic-embed-text"
        },
        "vector_store": {
            "type": "lancedb",
            "db_path": "./data/lancedb",
            "table_name": "documents"
        }
    }

    # Create RAG instance
    rag = BBCNewsRAG(config)
    await rag.initialize()

    try:
        # Read data using pandas
        df = pd.read_csv('data/data.txt')

        # Ingest data
        await rag.ingest_data(df)

        # Example query
        question = "Do you have any news regarding GPU Production Company?"
        response = await rag.query(
            question,
            system_prompt="You are a helpful assistant that provides accurate information about the Bedford Debenhams plans based on the news articles."
        )
        print(f"Question: {question}")
        print(f"Answer: {response}")

    finally:
        # Clean up
        await rag.close()


if __name__ == "__main__":
    asyncio.run(main())
