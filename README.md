# Ceylon AI RAG Framework

A powerful, modular, and extensible Retrieval-Augmented Generation (RAG) framework built with Python, supporting multiple LLM providers, embedders, and document types.

## 🌟 Features

- **Multiple Document Types**: Support for various document formats including:
  - Text files (with extensive format support)
  - PDF documents
  - Images (with OCR capabilities)
  - Source code files

- **Flexible Architecture**:
  - Modular component design
  - Pluggable LLM providers (OpenAI, Ollama)
  - Extensible embedding providers
  - Vector store integration (LanceDB)

- **Advanced RAG Capabilities**:
  - Intelligent document chunking
  - Context-aware searching
  - Query expansion and reranking
  - Metadata enrichment
  - Source attribution

- **Specialized RAG Implementations**:
  - `FolderRAG`: Process and analyze entire directory structures
  - `CodeAnalysisRAG`: Specialized for source code understanding
  - `SimpleRAG`: Basic RAG implementation for text data
  - Support for custom RAG implementations

## 🚀 Getting Started

### Installation

```bash
# Install via pip
pip install ceylon-rag

# Or install from source
git clone https://github.com/ceylonai/ceylon-rag.git
cd ceylon-rag
pip install -e .
```

### Basic Usage

Here's a simple example using the framework:

```python
import asyncio
from dotenv import load_dotenv
from ceylon_rag import SimpleRAG

async def main():
    # Load environment variables
    load_dotenv()

    # Configure the RAG system
    config = {
        "llm": {
            "type": "openai",
            "model_name": "gpt-4",
            "api_key": os.getenv("OPENAI_API_KEY")
        },
        "embedder": {
            "type": "openai",
            "model_name": "text-embedding-3-small",
            "api_key": os.getenv("OPENAI_API_KEY")
        },
        "vector_store": {
            "type": "lancedb",
            "db_path": "./data/lancedb",
            "table_name": "documents"
        }
    }

    # Initialize RAG
    rag = SimpleRAG(config)
    await rag.initialize()

    try:
        # Process your documents
        documents = await rag.process_documents("path/to/documents")
        
        # Query the system
        result = await rag.query("What are the main topics in these documents?")
        print(result.response)
        
    finally:
        await rag.close()

if __name__ == "__main__":
    asyncio.run(main())
```

## 🏗️ Architecture

### Core Components

1. **Document Loaders**
   - `TextLoader`: Handles text-based files
   - `PDFLoader`: Processes PDF documents
   - `ImageLoader`: Handles images with OCR
   - Extensible base class for custom loaders

2. **Embedders**
   - OpenAI embeddings support
   - Ollama embeddings support
   - Modular design for adding new providers

3. **LLM Providers**
   - OpenAI integration
   - Ollama integration
   - Async interface for all providers

4. **Vector Store**
   - LanceDB integration
   - Efficient vector similarity search
   - Metadata storage and retrieval

### Document Processing

The framework provides sophisticated document processing capabilities:

```python
# Example: Processing a code repository
async def analyze_codebase():
    config = {
        "llm": {
            "type": "openai",
            "model_name": "gpt-4"
        },
        "embedder": {
            "type": "openai",
            "model_name": "text-embedding-3-small"
        },
        "vector_store": {
            "type": "lancedb",
            "db_path": "./data/lancedb",
            "table_name": "code_documents"
        },
        "chunk_size": 1000,
        "chunk_overlap": 200
    }

    rag = CodeAnalysisRAG(config)
    await rag.initialize()
    
    documents = await rag.process_codebase("./src")
    await rag.index_code(documents)
    
    result = await rag.analyze_code(
        "Explain the main architecture of this codebase"
    )
    print(result.response)
```

## 🔧 Advanced Configuration

### File Exclusions

Configure file exclusions using patterns:

```python
config = {
    # ... other config options ...
    "excluded_dirs": [
        "venv",
        "node_modules",
        ".git",
        "__pycache__"
    ],
    "excluded_files": [
        ".env",
        "package-lock.json"
    ],
    "excluded_extensions": [
        ".pyc",
        ".pyo",
        ".pyd"
    ],
    "ignore_file": ".ragignore"  # Similar to .gitignore
}
```

### Chunking Configuration

Customize document chunking:

```python
config = {
    # ... other config options ...
    "chunk_size": 1000,  # Characters per chunk
    "chunk_overlap": 200,  # Overlap between chunks
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

[MIT License](LICENSE)

## 🙏 Acknowledgments

- OpenAI for GPT and embedding models
- Ollama for local LLM support
- LanceDB team for vector storage
- All contributors and users of the framework

---

## 📚 API Documentation

For detailed API documentation, please visit our [API Documentation](docs/api.md) page.

## 🔗 Links

- [GitHub Repository](https://github.com/yourusername/ceylon-rag)
- [Issue Tracker](https://github.com/yourusername/ceylon-rag/issues)
- [Documentation](https://ceylon-rag.readthedocs.io/)