import asyncio
from typing import Dict, Any, List, Tuple
import pandas as pd
from app.factory.component_factory import AsyncComponentFactory


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

    async def extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords from the input text"""
        prompt = f"""
        Extract key search terms from this text. Focus on:
        - Main topics
        - Named entities (people, companies, locations)
        - Important technical terms
        - Temporal indicators

        Text: {text}

        Return only the keywords, separated by commas.
        """

        response = await self.llm.generate(prompt=prompt)
        keywords = [k.strip() for k in response.split(',')]
        return keywords

    async def generate_query_variations(self, keywords: List[str]) -> List[str]:
        """Generate different variations of the search query"""
        prompt = f"""
        Generate 3 different search queries using these keywords: {', '.join(keywords)}

        Consider:
        1. Synonyms and related terms
        2. Different combinations of keywords
        3. Industry-specific terminology

        Return each query on a new line.
        """

        response = await self.llm.generate(prompt=prompt)
        variations = [q.strip() for q in response.split('\n') if q.strip()]
        return variations

    async def expand_context(self, question: str) -> str:
        """Expand the question with additional context"""
        prompt = f"""
        Enhance this question with relevant context:

        Original: {question}

        Consider:
        1. Industry background
        2. Recent events
        3. Related companies/technologies
        4. Market implications

        Provide an expanded version that includes this context.
        """

        expanded = await self.llm.generate(prompt=prompt)
        return expanded

    async def rerank_results(self, results: List[Dict], question: str) -> List[Dict]:
        """Rerank search results based on relevance to the question"""
        if not results:
            return results

        prompt = f"""
        Rate each result's relevance to the question on a scale of 0-10.
        Your response must be ONLY numbers separated by commas, nothing else.
        Example format: 8,5,7,3,9

        Question: {question}

        Results to rate:
        {[result.get('text', '')[:200] + '...' for result in results]}

        Ratings (ONLY numbers separated by commas):
        """

        try:
            scores_text = await self.llm.generate(prompt=prompt)
            # Clean up the response and extract numbers
            scores_text = scores_text.replace('\n', '').strip()
            scores = []

            # Extract all numbers from the text
            import re
            number_matches = re.finditer(r'\d+(?:\.\d+)?', scores_text)
            scores = [float(match.group()) for match in number_matches]

            # If we didn't get enough scores, pad with zeros
            while len(scores) < len(results):
                scores.append(0.0)

            # If we got too many scores, truncate
            scores = scores[:len(results)]

            # Combine results with scores and sort
            ranked_results = list(zip(results, scores))
            ranked_results.sort(key=lambda x: x[1], reverse=True)

            return [r[0] for r in ranked_results]

        except Exception as e:
            print(f"Error during reranking: {e}")
            # If reranking fails, return original order
            return results

    async def query(self, question: str, system_prompt: str = None) -> Tuple[str, Dict]:
        """Enhanced query processing with keyword extraction and query expansion"""

        # Extract keywords
        keywords = await self.extract_keywords(question)

        # Generate query variations
        query_variations = await self.generate_query_variations(keywords)

        # Expand the original question with context
        expanded_question = await self.expand_context(question)

        # Get embeddings for all query variations
        all_results = []
        for query in [expanded_question] + query_variations:
            query_embedding = await self.embedder.embed_query(query)
            results = await self.vector_store.search(query_embedding)
            all_results.extend(results)

        # Remove duplicates while preserving order
        seen = set()
        unique_results = []
        for result in all_results:
            result_id = result.get('metadata', {}).get('index')
            if result_id not in seen:
                seen.add(result_id)
                unique_results.append(result)

        # Rerank results
        reranked_results = await self.rerank_results(unique_results, question)

        # Generate the final response
        response = await self.llm.generate(
            prompt=f"""
            Question: {question}
            Expanded Question: {expanded_question}
            Keywords: {', '.join(keywords)}
            Context: {reranked_results[:5]}  # Use top 5 results
            """,
            system_prompt=system_prompt or """
            Provide a comprehensive answer that:
            1. Directly addresses the original question
            2. Incorporates relevant context
            3. Highlights key information from sources
            4. Maintains factual accuracy
            5. Notes any important caveats or limitations
            """
        )

        # Return both the response and search metadata
        search_metadata = {
            'keywords': keywords,
            'expanded_question': expanded_question,
            'query_variations': query_variations,
            'total_results': len(reranked_results),
            'top_sources': [r.get('metadata', {}).get('url') for r in reranked_results[:3]]
        }

        return response, search_metadata

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

    async def close(self):
        """Clean up resources"""
        if self.llm:
            await self.llm.__aexit__(None, None, None)
        if self.embedder:
            await self.embedder.__aexit__(None, None, None)


async def main():
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

    rag = BBCNewsRAG(config)
    await rag.initialize()

    try:
        df = pd.read_csv('data/data.txt')

        await rag.ingest_data(df)

        question = "Do you have any news regarding Health with sources?"
        response, metadata = await rag.query(
            question,
            system_prompt="You are a helpful assistant that provides accurate information about news articles."
        )

        print(f"Question: {question}")
        print(f"Keywords: {', '.join(metadata['keywords'])}")
        print(f"Expanded Question: {metadata['expanded_question']}")
        print(f"Query Variations: {metadata['query_variations']}")
        print(f"Top Sources: {metadata['top_sources']}")
        print(f"Answer: {response}")

    finally:
        await rag.close()


if __name__ == "__main__":
    asyncio.run(main())