from pathlib import Path
from tkinter.font import names
from typing import Dict, Any, List, Optional, Union, Set
import asyncio
from dataclasses import dataclass
import logging
import fnmatch
import re

from app.impl.loaders.image_loader import ImageLoader, ImageLoaderConfig
from app.impl.loaders.pdf_loader import PDFLoader, PDFLoaderConfig
from app.impl.loaders.text_loader import TextLoader, TextLoaderConfig
from app.interfaces.schemas import Document
from app.factory.component_factory import AsyncComponentFactory


class FolderDocument(Document):
    """Extended Document class with folder path information"""
    relative_path: str  # Relative path from root folder
    file_type: str  # Type of document (pdf, image, text)


class IgnorePattern:
    """Handles ignore patterns similar to .gitignore"""

    def __init__(self, patterns: List[str] = None):
        self.patterns = patterns or []
        self._compile_patterns()

    def _compile_patterns(self):
        """Convert glob patterns to regex patterns"""
        self.regex_patterns = []
        for pattern in self.patterns:
            # Remove leading and trailing whitespace
            pattern = pattern.strip()

            # Skip empty lines and comments
            if not pattern or pattern.startswith('#'):
                continue

            # Handle negation (inclusion) patterns
            is_include = pattern.startswith('!')
            if is_include:
                pattern = pattern[1:]

            # Convert glob pattern to regex
            regex = fnmatch.translate(pattern)

            # Make the regex match full paths
            if not pattern.startswith('/'):
                regex = '.*' + regex

            self.regex_patterns.append((re.compile(regex), is_include))

    def should_ignore(self, path: str) -> bool:
        """
        Check if a path should be ignored
        Returns True if path should be ignored, False otherwise
        """
        path = str(path).replace('\\', '/')
        should_ignore = False

        for regex, is_include in self.regex_patterns:
            if regex.match(path):
                should_ignore = not is_include

        return should_ignore

    @classmethod
    def from_file(cls, ignore_file: Union[str, Path]) -> 'IgnorePattern':
        """Create IgnorePattern from a file"""
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

        # Initialize loaders
        self.image_loader = ImageLoader()
        self.pdf_loader = PDFLoader()
        self.text_loader = TextLoader()

        # Set up ignore patterns
        self.ignore_pattern = IgnorePattern(config.get('ignore_patterns', []))

        # Default ignored file types
        self.ignored_extensions = set(config.get('ignored_extensions', {
            'git', 'pyc', 'pyo', 'pyd', 'DS_Store',
            'idea', 'vscode', 'cache', 'log', 'tmp'
        }))

        # Set up logging
        self.logger = logging.getLogger(__name__)

    async def initialize(self):
        """Initialize RAG components and document loaders"""
        # Initialize core RAG components
        self.llm = await self.factory.create_llm(**self.config["llm"])
        self.embedder = await self.factory.create_embedder(**self.config["embedder"])
        self.vector_store = await self.factory.create_vector_store(
            embedder=self.embedder,
            **self.config["vector_store"]
        )

        # Initialize loaders with their configs
        self.image_loader.initialize(ImageLoaderConfig(
            chunk_size=self.config.get("chunk_size", 1000),
            chunk_overlap=self.config.get("chunk_overlap", 200),
            ocr_lang="eng",
            name="image_loader",
            type="document_loader",
            config={}
        ))

        self.pdf_loader.initialize(PDFLoaderConfig(
            chunk_size=self.config.get("chunk_size", 1000),
            chunk_overlap=self.config.get("chunk_overlap", 200),
            name="pdf_loader",
            type="document_loader",
            config={}
        ))

        self.text_loader.initialize(TextLoaderConfig(
            chunk_size=self.config.get("chunk_size", 1000),
            chunk_overlap=self.config.get("chunk_overlap", 200),
            name="text_loader",
            type="document_loader",
            config={}
        ))

        # Load ignore patterns from file if specified
        if ignore_file := self.config.get('ignore_file'):
            self.ignore_pattern = IgnorePattern.from_file(ignore_file)

    def should_process_file(self, file_path: Path, root_path: Path) -> bool:
        """
        Determine if a file should be processed based on ignore patterns and extensions
        """
        # Get relative path from root
        try:
            relative_path = str(file_path.relative_to(root_path))
        except ValueError:
            return False

        # Check extension
        if file_path.suffix.lower()[1:] in self.ignored_extensions:
            return False

        # Check ignore patterns
        if self.ignore_pattern.should_ignore(relative_path):
            return False

        return True

    async def process_folder(self, folder_path: Union[str, Path], recursive: bool = True) -> List[FolderDocument]:
        """Process all files in a folder and its subfolders"""
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

                    # Convert Document to FolderDocument with path information
                    for doc in file_docs:
                        folder_doc = FolderDocument(
                            content=doc.content,
                            metadata={
                                **doc.metadata,
                                "relative_path": relative_path,
                                "root_folder": str(folder_path)
                            },
                            relative_path=relative_path,
                            file_type=self._get_file_type(file_path)
                        )
                        documents.append(folder_doc)

                except Exception as e:
                    self.logger.error(f"Error processing file {file_path}: {str(e)}")
                    continue

        return documents

    def _get_file_type(self, file_path: Path) -> str:
        """Determine the type of file based on extension"""
        ext = file_path.suffix.lower()[1:]  # Remove the dot

        if self.pdf_loader.supports_format(ext):
            return "pdf"
        elif self.image_loader.supports_format(ext):
            return "image"
        elif self.text_loader.supports_format(ext):
            return "text"
        else:
            return "unknown"

    async def _process_file(self, file_path: Path) -> List[Document]:
        """Process a single file using the appropriate loader"""
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

        # Prepare documents for embedding
        texts = [doc.content for doc in documents]
        embeddings = await self.embedder.embed_documents(texts)

        # Store embeddings with metadata
        metadata = [
            {
                **doc.metadata,
                "relative_path": doc.relative_path,
                "file_type": doc.file_type
            }
            for doc in documents
        ]

        await self.vector_store.store_embeddings(
            documents=[doc.content for doc in documents],
            embeddings=embeddings,
            metadata=metadata
        )

    async def search(self, query: str,
                     filter_criteria: Optional[Dict[str, Any]] = None,
                     top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search indexed documents with optional filtering
        """
        query_embedding = await self.embedder.embed_query(query)
        results = await self.vector_store.search(
            query_embedding
        )
        return results

    async def close(self):
        """Clean up resources"""
        if self.llm:
            await self.llm.__aexit__(None, None, None)
        if self.embedder:
            await self.embedder.__aexit__(None, None, None)


# Example usage
async def main():
    # Example ignore patterns
    ignore_patterns = [
        '*.git/*',  # Ignore git directory
        'venv/*',  # Ignore virtual environment
        '*.pyc',  # Ignore Python compiled files
        '/temp/*',  # Ignore temp directory in root
        '!*.pdf',  # Don't ignore PDF files (override)
        'tests/fixtures/*',  # Ignore test fixtures
        '**/node_modules/*'  # Ignore node_modules in any directory
    ]

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
        "ignore_patterns": ignore_patterns,
        # Optional: specify path to ignore file
        "ignore_file": ".ragignore",
        # Additional file extensions to ignore
        "ignored_extensions": {
            "git", "pyc", "pyo", "pyd", "DS_Store",
            "idea", "vscode", "cache", "log", "tmp"
        }
    }

    rag = FolderRAG(config)
    await rag.initialize()

    try:
        # Process and index a folder
        documents = await rag.process_folder("./data/projects/p1")
        print(f"Indexed {len(documents)} documents")
        await rag.index_documents(documents)

        # Search with filters
        results = await rag.search(
            "important information"
        )
        print(results)
        for result in results:
            print(f"Path: {result['metadata']['relative_path']}")
            print(f"Content: {result['text'][:200]}...")
            print("---")

    finally:
        await rag.close()


if __name__ == "__main__":
    asyncio.run(main())
