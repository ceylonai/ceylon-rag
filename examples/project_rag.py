import asyncio
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

from app.factory.component_factory import AsyncComponentFactory
from app.impl.loaders.text_loader import TextLoader, TextLoaderConfig
from app.interfaces.schemas import Document, QueryResult
from examples.utils.file_path_to_index import path_to_int64


class CodeDocument(Document):
    """Extended Document class with code-specific metadata"""

    @classmethod
    def from_document(cls, doc: Document, file_path: str, language: str,
                      code_type: str = "source") -> 'CodeDocument':
        """Create CodeDocument from base Document with code-specific metadata"""
        return cls(
            content=doc.content,
            metadata={
                **doc.metadata,
                "file_path": file_path,
                "url": file_path,
                "language": language,
                "code_type": code_type,
                "file_name": Path(file_path).name,
                "extension": Path(file_path).suffix[1:],
                "index": path_to_int64(file_path)
            },
            doc_id=doc.doc_id,
            created_at=doc.created_at
        )


class CodeAnalysisRAG:
    """RAG system specialized for source code analysis"""

    LANGUAGE_EXTENSIONS = {
        'py': 'python',
        'js': 'javascript',
        'ts': 'typescript',
        'java': 'java',
        'cpp': 'c++',
        'c': 'c',
        'go': 'go',
        'rs': 'rust',
        'rb': 'ruby',
        'php': 'php',
        'cs': 'csharp',
        'scala': 'scala',
        'swift': 'swift',
        'kt': 'kotlin',
        'r': 'r',
        'sh': 'shell',
        'sql': 'sql',
        'html': 'html',
        'css': 'css',
        'md': 'markdown',
    }

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.factory = AsyncComponentFactory(config)
        self.llm = None
        self.embedder = None
        self.vector_store = None
        self.text_loader = TextLoader()

        # Initialize exclusion patterns
        self.excluded_dirs = set(config.get('excluded_dirs', [
            'venv', 'node_modules', '.git', '__pycache__', 'build', 'dist',
            '.idea', '.vscode', 'coverage', '.next', '.cache'
        ]))

        self.excluded_files = set(config.get('excluded_files', [
            '.env', '.gitignore', '.dockerignore', 'package-lock.json',
            'yarn.lock', '.DS_Store', 'Thumbs.db'
        ]))

        self.excluded_extensions = set(config.get('excluded_extensions', [
            '.pyc', '.pyo', '.pyd', '.so', '.dll', '.dylib',
            '.log', '.tmp', '.temp', '.swp', '.swo',
            '.class', '.exe', '.bin', '.obj', '.pkl'
        ]))

        # Load ignore patterns from file if specified
        self.ignore_patterns = []
        if ignore_file := config.get('ignore_file'):
            self.ignore_patterns = self._load_ignore_patterns(ignore_file)

    async def initialize(self):
        """Initialize RAG components"""
        self.llm = await self.factory.create_llm(**self.config["llm"])
        self.embedder = await self.factory.create_embedder(**self.config["embedder"])
        self.vector_store = await self.factory.create_vector_store(
            embedder=self.embedder,
            **self.config["vector_store"]
        )

        # Initialize text loader with code-specific chunking
        self.text_loader.initialize(TextLoaderConfig(
            chunk_size=self.config.get("chunk_size", 1000),
            chunk_overlap=self.config.get("chunk_overlap", 200),
            name="code_loader",
            type="document_loader",
            config={}
        ))

    def _get_language(self, file_path: Path) -> Optional[str]:
        """Determine programming language from file extension"""
        ext = file_path.suffix.lower()[1:]
        return self.LANGUAGE_EXTENSIONS.get(ext)

    def _load_ignore_patterns(self, ignore_file: Union[str, Path]) -> List[str]:
        """Load ignore patterns from a file (similar to .gitignore)"""
        ignore_file = Path(ignore_file)
        if not ignore_file.exists():
            return []

        with open(ignore_file, 'r') as f:
            patterns = [line.strip() for line in f.readlines()
                        if line.strip() and not line.startswith('#')]
        return patterns

    def _matches_ignore_pattern(self, file_path: str) -> bool:
        """Check if file path matches any ignore pattern"""
        import fnmatch
        normalized_path = str(file_path).replace('\\', '/')

        for pattern in self.ignore_patterns:
            if pattern.startswith('!'):  # Negation pattern
                if fnmatch.fnmatch(normalized_path, pattern[1:]):
                    return False
            elif fnmatch.fnmatch(normalized_path, pattern):
                return True

        return False

    def _should_process_file(self, file_path: Path) -> bool:
        """Determine if a file should be processed based on configured exclusions"""
        # Convert to string and normalize path separators
        path_str = str(file_path).replace('\\', '/')

        # Check directory exclusions
        if any(excluded_dir in path_str.split('/') for excluded_dir in self.excluded_dirs):
            return False

        # Check file name exclusions
        if file_path.name in self.excluded_files:
            return False

        # Check extension exclusions
        if file_path.suffix.lower() in self.excluded_extensions:
            return False

        # Check ignore patterns
        if self._matches_ignore_pattern(path_str):
            return False

        # Check if it's a supported language
        ext = file_path.suffix.lower()[1:]
        return ext in self.LANGUAGE_EXTENSIONS

    async def process_codebase(self,
                               root_path: Union[str, Path],
                               recursive: bool = True) -> List[CodeDocument]:
        """Process and index all code files in the given directory"""
        root_path = Path(root_path)
        if not root_path.exists():
            raise FileNotFoundError(f"Directory not found: {root_path}")

        documents = []
        pattern = "**/*" if recursive else "*"

        for file_path in root_path.glob(pattern):
            if file_path.is_file() and self._should_process_file(file_path):
                try:
                    relative_path = str(file_path.relative_to(root_path))
                    language = self._get_language(file_path)

                    # Load and chunk the code file
                    file_docs = await self.text_loader.load(file_path)

                    # Create CodeDocuments with appropriate metadata
                    for doc in file_docs:
                        code_doc = CodeDocument.from_document(
                            doc=doc,
                            file_path=relative_path,
                            language=language
                        )
                        documents.append(code_doc)

                except Exception as e:
                    print(f"Error processing file {file_path}: {str(e)}")

        return documents

    async def index_code(self, documents: List[CodeDocument]):
        """Index the processed code documents"""
        if not documents:
            return

        embeddings = await self.embedder.embed_documents(documents)
        await self.vector_store.store_embeddings(documents=documents, embeddings=embeddings)

    async def analyze_code(self,
                           question: str,
                           filter_criteria: Optional[Dict[str, Any]] = None,
                           top_k: int = 5) -> QueryResult:
        """Search and analyze code with specific prompting for code understanding"""

        # Create code-specific prompt
        enhanced_question = f"""
        Analyze this code-related question, focusing on:
        - Code structure and patterns
        - Implementation details
        - Best practices and potential improvements
        - Related components and dependencies

        Question: {question}
        """

        # Get embeddings and search
        query_embedding = await self.embedder.embed_query(enhanced_question)
        results = await self.vector_store.search(
            query_embedding,
            limit=top_k
        )

        for doc in results:
            print(f"File: {doc.metadata}")
            print("\n")
        # Generate code-focused response
        context = "\n\n".join([
            f"File: {doc.metadata['url']}\n"
            f"Content:\n{doc.content}"
            for doc in results
        ])

        print(context)

        response = await self.llm.generate(
            prompt=f"""
            Question: {question}

            Available code context:
            {context}

            Provide a detailed analysis that:
            1. Directly answers the question
            2. References specific code examples
            3. Explains the implementation approach
            4. Suggests any relevant improvements
            5. Notes potential issues or considerations
            """,
            system_prompt="""
            You are an expert code analyst. Provide clear, technical explanations
            with specific references to the code. Focus on practical insights
            and maintainable solutions.
            """
        )

        return QueryResult(
            response=response,
            source_documents=results,
            metadata={
                "query": question,
                "filter_criteria": filter_criteria,
                "total_results": len(results),
                # "languages": list(set(doc.metadata["language"] for doc in results))
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
    from dotenv import load_dotenv

    load_dotenv()  # take environment variables from .env.

    # Example configuration
    config = {
        # "llm": {
        #     "type": "ollama",
        #     "model_name": "phi3.5:latest"  # Using a code-specific model
        # },
        "llm": {
            "type": "openai",
            "model_name": "gpt-4o",  # Using a code-specific model
            "api_key": os.getenv("OPENAI_API_KEY")
        },
        # "embedder": {
        #     "type": "ollama",
        #     "model_name": "nomic-embed-text"
        # },
        "embedder": {
            "type": "openai",
            "model_name": "text-embedding-3-small",  # Using a code-specific model
            "api_key": os.getenv("OPENAI_API_KEY")
        },
        "vector_store": {
            "type": "lancedb",
            "db_path": "./data/lancedb",
            "table_name": "code_documents"
        },
        "chunk_size": 1000,
        "chunk_overlap": 200,

        # File exclusion configurations
        "excluded_dirs": [
            "venv",
            "node_modules",
            ".git",
            "__pycache__",
            "build",
            "dist",
            "tests/fixtures",
            ".pytest_cache",
            "data"
        ],

        "excluded_files": [
            "setup.py",
            "requirements.txt",
            "package.json",
            ".env",
            ".env.local",
            "README.md"
        ],

        "excluded_extensions": [
            ".pyc",
            ".pyo",
            ".pyd",
            ".log",
            ".csv",
            ".json"
        ],

        # Optional: Path to a .gitignore-style file for additional patterns
        "ignore_file": ".coderagignore"
    }

    rag = CodeAnalysisRAG(config)
    await rag.initialize()

    try:
        # Process and index a codebase
        documents = await rag.process_codebase("L:\\projects\\CeylonAI\\agent-rag-app")
        print(f"Processed {len(documents)} code segments")

        await rag.index_code(documents)
        print("Code indexed successfully")

        # Example analysis
        query_result = await rag.analyze_code(
            "Generate README File content with code structure and examples. add a simple getting started code too",
        )

        print("\nAnalysis Results:")
        print(query_result.response)
        # print("\nRelevant Files:")
        # for doc in query_result.source_documents:
        #     print(f"- {doc.metadata['file_path']} ({doc.metadata['language']})")

    finally:
        await rag.close()


if __name__ == "__main__":
    asyncio.run(main())