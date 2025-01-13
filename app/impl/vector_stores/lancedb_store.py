import asyncio
from datetime import datetime
from typing import List
from uuid import UUID

import lancedb
from lancedb.pydantic import LanceModel, Vector

from app.interfaces.schemas import Document
from app.interfaces.vector_store import VectorStore


def create_lance_schema(embedder):
    class LanceDBSchema(LanceModel):
        text: str = embedder.SourceField()
        vector: Vector(embedder.ndims()) = embedder.VectorField()
        doc_id: str
        metadata_title: str  # Flattened metadata fields
        metadata_url: str
        metadata_index: int  # Required field, non-null
        created_at: str

    return LanceDBSchema


class AsyncLanceDBStore(VectorStore):
    def __init__(self, embedder, db_path: str = "./lancedb", table_name: str = "documents"):
        self.db = lancedb.connect(db_path)
        self.table_name = table_name
        self.table = None
        self._lock = asyncio.Lock()
        self.schema = create_lance_schema(embedder)

    async def store_embeddings(self, documents: List[Document], embeddings: List[List[float]]) -> None:
        if self.table_name in self.db.table_names():
            self.table = self.db.open_table(self.table_name)

        data = []
        for doc, emb in zip(documents, embeddings):
            # Ensure index exists in metadata
            if 'index' not in doc.metadata:
                raise ValueError("Document metadata must contain 'index' field")

            data.append({
                "text": doc.content,
                "vector": emb,
                "doc_id": str(doc.doc_id),
                "metadata_title": doc.metadata.get('title', ''),
                "metadata_url": doc.metadata.get('url', ''),
                "metadata_index": doc.metadata['index'],  # Required field
                "created_at": doc.created_at.isoformat()
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

    async def search(self, query_embedding: List[float], limit: int = 3) -> List[Document]:
        if self.table is None:
            raise ValueError("No documents have been stored yet")

        async with self._lock:
            results = await asyncio.to_thread(
                lambda: self.table.search(query_embedding)
                .limit(limit)
                .to_pydantic(self.schema)
            )

        return [
            Document(
                content=r.text,
                metadata={
                    'title': r.metadata_title,
                    'url': r.metadata_url,
                    'index': r.metadata_index
                },
                doc_id=UUID(r.doc_id),
                created_at=datetime.fromisoformat(r.created_at)
            )
            for r in results
        ]
