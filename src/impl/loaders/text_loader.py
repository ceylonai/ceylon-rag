from typing import List, Union
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_not_exception_type

from src.interfaces.document_loader import DocumentLoader, LoaderConfig
from src.interfaces.schemas import Document


class TextLoaderConfig(LoaderConfig):
    """Configuration for text document loader"""

    def __init__(self, **data):
        if 'supported_formats' not in data:
            # Common text file extensions
            data['supported_formats'] = [
                'txt', 'text', 'log', 'md', 'markdown',
                'rst', 'csv', 'json', 'yaml', 'yml',
                'xml', 'html', 'htm', 'css', 'js',
                'py', 'java', 'cpp', 'c', 'h', 'hpp',
                'sql', 'sh', 'bat', 'ini', 'cfg',
                'conf', 'properties', 'env'
            ]
        super().__init__(**data)


class TextLoader(DocumentLoader):
    """Text document loader implementation"""

    def __init__(self):
        self.config: TextLoaderConfig = None

    def initialize(self, config: TextLoaderConfig) -> None:
        """Initialize the text loader with configuration"""
        if not self.validate_config(config):
            raise ValueError("Invalid text loader configuration")
        self.config = config

    def validate_config(self, config: TextLoaderConfig) -> bool:
        """Validate the loader configuration"""
        if not isinstance(config, TextLoaderConfig):
            return False
        if not config.supported_formats:
            return False
        if config.chunk_size and config.chunk_size <= 0:
            return False
        if config.chunk_overlap and (config.chunk_overlap < 0 or
                                     config.chunk_overlap >= config.chunk_size):
            return False
        return True

    def supports_format(self, format: str) -> bool:
        """Check if the format is supported"""
        return format.lower() in self.config.supported_formats

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_not_exception_type((RuntimeError, FileNotFoundError, ValueError))
    )
    async def load(self, source: Union[str, Path]) -> List[Document]:
        """Load and process a text document"""
        if not self.config:
            raise RuntimeError("Loader not initialized")

        # Convert source to Path object
        source_path = Path(source)
        if not source_path.exists():
            raise FileNotFoundError(f"File not found: {source}")

        if not self.supports_format(source_path.suffix[1:]):
            raise ValueError(f"Unsupported format: {source_path.suffix}")

        documents: List[Document] = []

        try:
            # Read the text file with UTF-8 encoding
            with open(source_path, 'r', encoding='utf-8') as file:
                raw_text = file.read()

            if not raw_text.strip():
                return []

            # Create documents based on chunking configuration
            if self.config.chunk_size:
                documents.extend(self._chunk_text(raw_text, str(source_path)))
            else:
                # Create a single document if no chunking is specified
                documents.append(
                    Document(
                        content=raw_text.strip(),
                        metadata={
                            "source": str(source_path),
                            "type": "text",
                            "size": len(raw_text)
                        }
                    )
                )

        except UnicodeDecodeError:
            # Fallback to binary reading if UTF-8 fails
            with open(source_path, 'rb') as file:
                raw_text = file.read().decode('utf-8', errors='replace')
                if not raw_text.strip():
                    return []

                if self.config.chunk_size:
                    documents.extend(self._chunk_text(raw_text, str(source_path)))
                else:
                    documents.append(
                        Document(
                            content=raw_text.strip(),
                            metadata={
                                "source": str(source_path),
                                "type": "text",
                                "size": len(raw_text),
                                "encoding": "fallback"
                            }
                        )
                    )

        return documents

    def _chunk_text(self, text: str, source: str) -> List[Document]:
        """Split text into chunks with specified size and overlap, ensuring no chunk exceeds max size

        Args:
            text (str): Input text to be chunked
            source (str): Source identifier for the text

        Returns:
            List[Document]: List of Document objects containing text chunks and metadata
        """
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            # Calculate initial chunk end position
            end = min(start + self.config.chunk_size, text_length)

            # If we're not at text end, try to find a sentence boundary
            if end < text_length:
                # Search range is from end backwards, but never beyond start
                # or more than chunk_size distance
                search_end = end
                search_start = max(start, end - self.config.chunk_size)

                # Look for sentence endings (.!?) followed by space or newline
                sentence_end = None
                for i in range(search_end, search_start - 1, -1):
                    if text[i] in '.!?' and (i + 1 == text_length or text[i + 1].isspace()):
                        sentence_end = i + 1
                        break

                # Only use sentence boundary if found within search range
                if sentence_end is not None:
                    end = sentence_end

            # Ensure chunk doesn't exceed max size even if no sentence boundary found
            if end - start > self.config.chunk_size:
                end = start + self.config.chunk_size

            # Create document from chunk
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    Document(
                        content=chunk_text,
                        metadata={
                            "source": source,
                            "type": "text",
                            "chunk_start": start,
                            "chunk_end": end
                        }
                    )
                )

            # Move to next chunk position considering overlap
            if self.config.chunk_overlap:
                start = end - self.config.chunk_overlap
            else:
                start = end

        return chunks
