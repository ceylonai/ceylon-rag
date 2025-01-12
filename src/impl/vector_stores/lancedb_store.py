import asyncio
from typing import List, Dict, Any
import lancedb
from lancedb.pydantic import LanceModel, Vector

from interfaces.vector_store import VectorStore


def create_lance_schema(embedder):
    class LanceDBSchema(LanceModel):
        text: str = embedder.SourceField()
        vector: Vector(embedder.ndims()) = embedder.VectorField()
        index: int
        title: str
        url: str

    return LanceDBSchema


class AsyncLanceDBStore(VectorStore):
    def __init__(self, embedder, db_path: str = "./lancedb", table_name: str = "documents"):
        self.db = lancedb.connect(db_path)
        self.table_name = table_name
        self.table = None
        self._lock = asyncio.Lock()
        self.schema = create_lance_schema(embedder)

    async def store_embeddings(self, documents: List[str], embeddings: List[List[float]],
                               metadata: List[Dict[str, Any]] = None) -> None:
        if metadata is None:
            metadata = [{"title": "", "url": "", "index": i} for i in range(len(documents))]

        table_name = self.db.table_names()
        if self.table_name in table_name:
            self.table = self.db.open_table(self.table_name)

        data = []
        for doc, emb, meta in zip(documents, embeddings, metadata):
            data.append({
                "text": doc,
                "vector": emb,
                "index": meta.get("index", 0),
                "title": meta.get("title", ""),
                "url": meta.get("url", "")
            })

        async with self._lock:
            if self.table is None:
                self.table = await asyncio.to_thread(
                    self.db.create_table,
                    self.table_name,
                    data=data,
                    schema=self.schema.to_arrow_schema()
                )
            else:
                await asyncio.to_thread(self.table.add, data)

    async def search(self, query_embedding: List[float], limit: int = 3) -> List[Dict[str, Any]]:
        if self.table is None:
            raise ValueError("No documents have been stored yet")

        async with self._lock:
            results = await asyncio.to_thread(
                lambda: self.table.search(query_embedding)
                .limit(limit)
                .to_pydantic(self.schema)
            )

        return [
            {
                "text": r.text,
                "title": r.title,
                "url": r.url,
                "index": r.index
            }
            for r in results
        ]
