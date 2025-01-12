import asyncio

from factory.component_factory import AsyncComponentFactory


async def main():
    # Configuration example
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
            "db_path": "./lancedb",
            "table_name": "documents"
        }
    }

    # Create components
    factory = AsyncComponentFactory(config)
    async with (
        await factory.create_llm(**config["llm"]) as llm,
        await factory.create_embedder(**config["embedder"]) as embedder
    ):
        vector_store = await factory.create_vector_store(embedder=embedder, **config["vector_store"])

        # Example usage
        query = "What is RAG?"
        query_embedding = await embedder.embed_query(query)
        results = await vector_store.search(query_embedding)
        response = await llm.generate(
            prompt=f"Question: {query}\nContext: {results}",
            system_prompt="Answer the question based on the provided context."
        )
        print(response)


if __name__ == "__main__":
    asyncio.run(main())
