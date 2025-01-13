import asyncio
import fnmatch
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

from app.factory.component_factory import AsyncComponentFactory
from app.impl.loaders.image_loader import ImageLoader, ImageLoaderConfig
from app.impl.loaders.pdf_loader import PDFLoader, PDFLoaderConfig
from app.impl.loaders.text_loader import TextLoader, TextLoaderConfig
from app.interfaces.schemas import Document, QueryResult
from examples.utils.file_path_to_index import path_to_int64


class FolderDocument(Document):
    """Extended Document class with folder path information"""

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_document(cls, doc: Document, relative_path: str, file_type: str) -> 'FolderDocument':
        """Create FolderDocument from base Document"""
        return cls(
            content=doc.content,
            metadata={
                **doc.metadata,
                "url": relative_path,
                "file_type": file_type,
                "index": path_to_int64(relative_path)
            },
            doc_id=doc.doc_id,
            created_at=doc.created_at
        )


class IgnorePattern:
    """Handles ignore patterns similar to .gitignore"""

    def __init__(self, patterns: List[str] = None):
        self.patterns = patterns or []
        self._compile_patterns()

    def _compile_patterns(self):
        """Convert glob patterns to regex patterns"""
        self.regex_patterns = []
        for pattern in self.patterns:
            pattern = pattern.strip()
            if not pattern or pattern.startswith('#'):
                continue

            is_include = pattern.startswith('!')
            if is_include:
                pattern = pattern[1:]

            regex = fnmatch.translate(pattern)
            if not pattern.startswith('/'):
                regex = '.*' + regex

            self.regex_patterns.append((re.compile(regex), is_include))

    def should_ignore(self, path: str) -> bool:
        path = str(path).replace('\\', '/')
        should_ignore = False

        for regex, is_include in self.regex_patterns:
            if regex.match(path):
                should_ignore = not is_include

        return should_ignore

    @classmethod
    def from_file(cls, ignore_file: Union[str, Path]) -> 'IgnorePattern':
        ignore_file = Path(ignore_file)
        if not ignore_file.exists():
            return cls([])

        with open(ignore_file, 'r') as f:
            patterns = f.read().splitlines()
        return cls(patterns)


class FolderRAG:
    """RAG system for processing folders and their contents"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.factory = AsyncComponentFactory(config)
        self.llm = None
        self.embedder = None
        self.vector_store = None

        self.image_loader = ImageLoader()
        self.pdf_loader = PDFLoader()
        self.text_loader = TextLoader()

        self.ignore_pattern = IgnorePattern(config.get('ignore_patterns', []))
        self.ignored_extensions = set(config.get('ignored_extensions', {
            'git', 'pyc', 'pyo', 'pyd', 'DS_Store',
            'idea', 'vscode', 'cache', 'log', 'tmp'
        }))

        self.logger = logging.getLogger(__name__)

    async def initialize(self):
        """Initialize RAG components and document loaders"""
        self.llm = await self.factory.create_llm(**self.config["llm"])
        self.embedder = await self.factory.create_embedder(**self.config["embedder"])
        self.vector_store = await self.factory.create_vector_store(
            embedder=self.embedder,
            **self.config["vector_store"]
        )

        # Initialize loaders with their configs
        loader_config = {
            "chunk_size": self.config.get("chunk_size", 1000),
            "chunk_overlap": self.config.get("chunk_overlap", 200),
        }

        self.image_loader.initialize(ImageLoaderConfig(
            **loader_config,
            ocr_lang="eng",
            name="image_loader",
            type="document_loader",
            config={}
        ))

        self.pdf_loader.initialize(PDFLoaderConfig(
            **loader_config,
            name="pdf_loader",
            type="document_loader",
            config={}
        ))

        self.text_loader.initialize(TextLoaderConfig(
            **loader_config,
            name="text_loader",
            type="document_loader",
            config={}
        ))

        if ignore_file := self.config.get('ignore_file'):
            self.ignore_pattern = IgnorePattern.from_file(ignore_file)

    def should_process_file(self, file_path: Path, root_path: Path) -> bool:
        try:
            relative_path = str(file_path.relative_to(root_path))
        except ValueError:
            return False

        if file_path.suffix.lower()[1:] in self.ignored_extensions:
            return False

        if self.ignore_pattern.should_ignore(relative_path):
            return False

        return True

    async def process_folder(self, folder_path: Union[str, Path], recursive: bool = True) -> List[FolderDocument]:
        folder_path = Path(folder_path)
        if not folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        documents: List[FolderDocument] = []
        pattern = "**/*" if recursive else "*"

        for file_path in folder_path.glob(pattern):
            if file_path.is_file() and self.should_process_file(file_path, folder_path):
                try:
                    relative_path = str(file_path.relative_to(folder_path))
                    file_docs = await self._process_file(file_path)

                    file_type = self._get_file_type(file_path)
                    for doc in file_docs:
                        folder_doc = FolderDocument.from_document(
                            doc=doc,
                            relative_path=relative_path,
                            file_type=file_type,
                        )
                        documents.append(folder_doc)

                except Exception as e:
                    self.logger.error(f"Error processing file {file_path}: {str(e)}")

        return documents

    def _get_file_type(self, file_path: Path) -> str:
        ext = file_path.suffix.lower()[1:]

        if self.pdf_loader.supports_format(ext):
            return "pdf"
        elif self.image_loader.supports_format(ext):
            return "image"
        elif self.text_loader.supports_format(ext):
            return "text"
        else:
            return "unknown"

    async def _process_file(self, file_path: Path) -> List[Document]:
        file_type = self._get_file_type(file_path)

        try:
            if file_type == "pdf":
                return await self.pdf_loader.load(file_path)
            elif file_type == "image":
                return await self.image_loader.load(file_path)
            elif file_type == "text":
                return await self.text_loader.load(file_path)
            else:
                self.logger.warning(f"Unsupported file type: {file_path}")
                return []
        except Exception as e:
            self.logger.error(f"Error loading file {file_path}: {str(e)}")
            return []

    async def index_documents(self, documents: List[FolderDocument]):
        """Index the processed documents in the vector store"""
        if not documents:
            return

        embeddings = await self.embedder.embed_documents(documents)
        await self.vector_store.store_embeddings(documents=documents, embeddings=embeddings)

    async def search(self,
                     query: str,
                     filter_criteria: Optional[Dict[str, Any]] = None,
                     top_k: int = 5) -> QueryResult:
        """Search indexed documents with optional filtering"""
        query_embedding = await self.embedder.embed_query(query)
        results = await self.vector_store.search(query_embedding, limit=top_k)

        return QueryResult(
            response="",  # Can be populated with LLM response if needed
            source_documents=results,
            metadata={
                "query": query,
                "filter_criteria": filter_criteria,
                "total_results": len(results),
            },
            created_at=datetime.utcnow()
        )

    async def close(self):
        """Clean up resources"""
        if self.llm:
            await self.llm.__aexit__(None, None, None)
        if self.embedder:
            await self.embedder.__aexit__(None, None, None)


async def main():
    # Example configuration
    config = {
        "llm": {
            "type": "ollama",
            "model_name": "llama2"
        },
        "embedder": {
            "type": "ollama",
            "model_name": "nomic-embed-text"
        },
        "vector_store": {
            "type": "lancedb",
            "db_path": "./data/lancedb",
            "table_name": "folder_documents"
        },
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "ignore_patterns": [
            '*.git/*',
            'venv/*',
            '*.pyc',
            '/temp/*',
            '!*.pdf',
            'tests/fixtures/*',
            '**/node_modules/*'
        ]
    }

    rag = FolderRAG(config)
    await rag.initialize()

    try:
        # Process and index a folder
        documents = await rag.process_folder("./data/projects/p1")
        print(f"Processed {len(documents)} documents")

        await rag.index_documents(documents)
        print("Documents indexed successfully")

        # Search example
        query_result = await rag.search("important information")

        print(f"\nFound {len(query_result.source_documents)} relevant documents:")
        for doc in query_result.source_documents:
            print(doc.metadata)
            print(f"\nPath: {doc.metadata.get('url')}")
            print(f"Type: {doc.metadata.get('file_type')}")
            print(f"Content preview: {doc.content[:200]}...")

    finally:
        await rag.close()


if __name__ == "__main__":
    asyncio.run(main())
