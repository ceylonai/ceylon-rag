from pathlib import Path
from typing import List, Optional, Union
from dataclasses import dataclass


@dataclass
class Document:
    """Document class to store text content and metadata"""
    content: str
    metadata: dict


@dataclass
class LoaderConfig:
    """Configuration for the universal text loader"""
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    encoding_fallbacks: List[str] = ('utf-8', 'latin-1', 'cp1252', 'ascii')


class UniversalTextLoader:
    """A flexible text loader that can handle any file as text"""

    def __init__(self, config: Optional[LoaderConfig] = None):
        """Initialize the loader with optional configuration"""
        self.config = config or LoaderConfig()

    def load(self, source: Union[str, Path]) -> List[Document]:
        """Load text from any file regardless of extension"""
        source_path = Path(source)
        if not source_path.exists():
            raise FileNotFoundError(f"File not found: {source}")

        # Try different encodings from the config
        text = None
        last_error = None

        for encoding in self.config.encoding_fallbacks:
            try:
                with open(source_path, 'r', encoding=encoding) as file:
                    text = file.read()
                    # Remove BOM if present
                    if text.startswith('\ufeff'):
                        text = text[1:]
                break
            except UnicodeDecodeError as e:
                last_error = e
                continue

        if text is None:
            # If all encodings fail, try binary mode with replace error handler
            try:
                with open(source_path, 'rb') as file:
                    text = file.read().decode('utf-8', errors='replace')
            except Exception as e:
                raise RuntimeError(f"Failed to read file with all encodings: {str(last_error)}") from e

        if not text.strip():
            return []

        return self._process_text(text, str(source_path))

    def _process_text(self, text: str, source: str) -> List[Document]:
        """Process the text into one or more documents based on chunking configuration"""
        if not self.config.chunk_size:
            return [Document(
                content=text.strip(),
                metadata={
                    "source": source,
                    "size": len(text)
                }
            )]

        return self._chunk_text(text, source)

    def _chunk_text(self, text: str, source: str) -> List[Document]:
        """Split text into chunks with proper handling of sentence boundaries"""
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            # Calculate chunk end position
            end = min(start + self.config.chunk_size, text_length)

            # Try to find a sentence boundary near the chunk end
            if end < text_length:
                # Look back for sentence endings
                for i in range(end, max(start, end - 100), -1):
                    if text[i - 1] in '.!?' and (i == text_length or text[i].isspace()):
                        end = i
                        break

            # Extract and clean the chunk
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(Document(
                    content=chunk_text,
                    metadata={
                        "source": source,
                        "chunk_start": start,
                        "chunk_end": end
                    }
                ))

            # Move to next chunk position
            if self.config.chunk_overlap and end < text_length:
                start = end - self.config.chunk_overlap
            else:
                start = end

        return chunks


# Example usage:
if __name__ == "__main__":
    # Create a loader with custom configuration
    config = LoaderConfig(
        chunk_size=1000,  # Split into 1000-character chunks
        chunk_overlap=100,  # 100 character overlap between chunks
        encoding_fallbacks=['utf-8', 'latin-1', 'ascii']
    )

    loader = UniversalTextLoader(config)

    # Load and process a file
    try:
        documents = loader.load("path/to/any/file.txt")
        for doc in documents:
            print(f"Chunk size: {len(doc.content)}")
            print(f"Metadata: {doc.metadata}")
    except Exception as e:
        print(f"Error loading file: {e}")