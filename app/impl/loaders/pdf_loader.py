from typing import List, Union
from pathlib import Path

from pypdf import PdfReader
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_not_exception_type

from app.interfaces.document_loader import DocumentLoader, LoaderConfig
from app.interfaces.schemas import Document


class PDFLoaderConfig(LoaderConfig):
    """Configuration for PDF document loader"""

    def __init__(self, **data):
        if 'supported_formats' not in data:
            data['supported_formats'] = ['pdf']
        super().__init__(**data)


class PDFLoader(DocumentLoader):
    """PDF document loader implementation"""

    def __init__(self):
        self.config: PDFLoaderConfig = None

    def initialize(self, config: PDFLoaderConfig) -> None:
        """Initialize the PDF loader with configuration"""
        if not self.validate_config(config):
            raise ValueError("Invalid PDF loader configuration")
        self.config = config

    def validate_config(self, config: PDFLoaderConfig) -> bool:
        """Validate the loader configuration"""
        if not isinstance(config, PDFLoaderConfig):
            return False
        if 'pdf' not in config.supported_formats:
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
        # Add ValueError to non-retried exceptions
    )
    async def load(self, source: Union[str, Path]) -> List[Document]:
        """Load and process a PDF document"""
        if not self.config:
            raise RuntimeError("Loader not initialized")

        # Convert source to Path object
        source_path = Path(source)
        if not source_path.exists():
            raise FileNotFoundError(f"File not found: {source}")

        if not self.supports_format(source_path.suffix[1:]):
            raise ValueError(f"Unsupported format: {source_path.suffix}")

        documents: List[Document] = []

        with open(source_path, 'rb') as file:
            # Create PDF reader object
            reader = PdfReader(file)

            # Extract text from each page
            raw_text = ""
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    raw_text += text + "\n\n"

            if not raw_text.strip():
                return []

            # Create documents based on chunking configuration
            if self.config.chunk_size:
                documents.extend(self._chunk_text(raw_text))
            else:
                # Create a single document if no chunking is specified
                documents.append(
                    Document(
                        content=raw_text.strip(),
                        metadata={
                            "source": str(source_path),
                            "type": "pdf",
                            "pages": len(reader.pages)
                        }
                    )
                )



        return documents

    def _chunk_text(self, text: str) -> List[Document]:
        """Split text into chunks with specified size and overlap"""
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            # Calculate chunk end position
            end = start + self.config.chunk_size

            # Adjust chunk boundary to nearest sentence end if possible
            if end < text_length:
                # Look for sentence endings (.!?) followed by space or newline
                for i in range(min(end + 100, text_length), max(start, end - 100), -1):
                    if i < text_length and text[i] in '.!?' and (i + 1 == text_length or text[i + 1].isspace()):
                        end = i + 1
                        break

            # Create document from chunk
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    Document(
                        content=chunk_text,
                        metadata={
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
