from typing import List, Dict, Type, Optional
import asyncio
from pathlib import Path

from app.interfaces.document_loader import DocumentLoader, LoaderConfig
from app.interfaces.schemas import Document
from app.impl.loaders.image_loader import ImageLoader, ImageLoaderConfig
from app.impl.loaders.pdf_loader import PDFLoader, PDFLoaderConfig
from app.impl.loaders.text_loader import TextLoader, TextLoaderConfig


class RAGDataLoader:
    """Combined data loader with RAG capabilities"""

    def __init__(self):
        self.loaders: Dict[str, DocumentLoader] = {}
        self.initialize_loaders()

    def initialize_loaders(self):
        """Initialize all supported document loaders with default configurations"""
        # Initialize Image Loader
        image_loader = ImageLoader()
        image_loader.initialize(ImageLoaderConfig(
            chunk_size=1000,
            chunk_overlap=200,
            ocr_lang='eng'
        ))

        # Initialize PDF Loader
        pdf_loader = PDFLoader()
        pdf_loader.initialize(PDFLoaderConfig(
            chunk_size=1000,
            chunk_overlap=200
        ))

        # Initialize Text Loader
        text_loader = TextLoader()
        text_loader.initialize(TextLoaderConfig(
            chunk_size=1000,
            chunk_overlap=200
        ))

        # Map file extensions to loaders
        self.loaders = {
            # Image formats
            'png': image_loader,
            'jpg': image_loader,
            'jpeg': image_loader,
            'tiff': image_loader,
            'bmp': image_loader,

            # PDF format
            'pdf': pdf_loader,

            # Text formats
            'txt': text_loader,
            'md': text_loader,
            'json': text_loader,
            'yaml': text_loader,
            'yml': text_loader,
            'xml': text_loader,
            'html': text_loader,
            'htm': text_loader,
            'csv': text_loader,
            # Add more text formats as needed
        }

    async def load_directory(self, directory_path: str) -> List[Document]:
        """Load all supported documents from a directory"""
        directory = Path(directory_path)
        if not directory.exists() or not directory.is_dir():
            raise ValueError(f"Invalid directory path: {directory_path}")

        documents: List[Document] = []
        tasks = []

        # Process all files in directory
        for file_path in directory.glob('**/*'):  # Recursively get all files
            if file_path.is_file():
                extension = file_path.suffix[1:].lower()
                if extension in self.loaders:
                    # Create task for loading document
                    loader = self.loaders[extension]
                    tasks.append(self.load_file(file_path, loader))

        # Execute all loading tasks concurrently
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, list):  # Successful load
                    documents.extend(result)
                elif isinstance(result, Exception):  # Failed load
                    print(f"Error loading document: {str(result)}")

        return documents

    async def load_file(self, file_path: Path, loader: DocumentLoader) -> List[Document]:
        """Load a single file using appropriate loader"""
        try:
            return await loader.load(str(file_path))
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
            raise

    def update_loader_config(self,
                             format: str,
                             config_updates: dict) -> bool:
        """Update configuration for a specific loader"""
        if format not in self.loaders:
            return False

        loader = self.loaders[format]
        current_config = loader.config

        # Update configuration with new values
        for key, value in config_updates.items():
            if hasattr(current_config, key):
                setattr(current_config, key, value)

        # Reinitialize loader with updated config
        try:
            loader.initialize(current_config)
            return True
        except ValueError:
            return False


# Example usage:
async def main():
    # Initialize RAG data loader
    rag_loader = RAGDataLoader()

    # Optional: Update specific loader configurations
    rag_loader.update_loader_config('pdf', {
        'chunk_size': 500,
        'chunk_overlap': 100
    })

    # Load documents from directory
    try:
        documents = await rag_loader.load_directory('path/to/your/data/folder')
        print(f"Loaded {len(documents)} document chunks")

        # Process documents as needed
        for doc in documents:
            print(f"Document source: {doc.metadata.get('source')}")
            print(f"Content length: {len(doc.content)}")
            print("---")

    except Exception as e:
        print(f"Error loading documents: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())
