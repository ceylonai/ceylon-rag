# app/impl/loaders/image_loader.py
from typing import List, Union
from pathlib import Path
from PIL import Image, ExifTags
from PIL import Image
import pytesseract
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_not_exception_type

from app.interfaces.document_loader import DocumentLoader, LoaderConfig
from app.interfaces.schemas import Document


class ImageLoaderConfig(LoaderConfig):
    """Configuration for Image document loader"""

    ocr_lang: str
    ocr_config: str

    def __init__(self, **data):
        if 'supported_formats' not in data:
            # Common image formats
            data['supported_formats'] = ['png', 'jpg', 'jpeg', 'tiff', 'bmp']
        if 'ocr_lang' not in data:
            data['ocr_lang'] = 'eng'  # Default OCR language
        if 'ocr_config' not in data:
            data['ocr_config'] = ''  # Additional Tesseract configuration
        super().__init__(**data)


class ImageLoader(DocumentLoader):
    """Image document loader implementation with OCR capabilities"""

    def __init__(self):
        self.config: ImageLoaderConfig = None

    def initialize(self, config: ImageLoaderConfig) -> None:
        """Initialize the image loader with configuration"""
        if not self.validate_config(config):
            raise ValueError("Invalid image loader configuration")
        self.config = config

    def validate_config(self, config: ImageLoaderConfig) -> bool:
        """Validate the loader configuration"""
        if not isinstance(config, ImageLoaderConfig):
            return False

        # Validate supported formats
        if not any(format in config.supported_formats
                   for format in ['png', 'jpg', 'jpeg', 'tiff', 'bmp']):
            return False

        # Validate chunking parameters if specified
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
        """Load and process an image document"""
        if not self.config:
            raise RuntimeError("Loader not initialized")

        # Convert source to Path object
        source_path = Path(source)
        if not source_path.exists():
            raise FileNotFoundError(f"File not found: {source}")

        if not self.supports_format(source_path.suffix[1:].lower()):
            raise ValueError(f"Unsupported format: {source_path.suffix}")

        with Image.open(source_path) as img:
            # Extract image metadata
            metadata = self._extract_metadata(img, source_path)

            # Perform OCR
            text = pytesseract.image_to_string(
                img,
                lang=self.config.ocr_lang,
                config=self.config.ocr_config
            )

            if not text.strip():
                # No text found in image
                return [Document(
                    content="",
                    metadata={
                        **metadata,
                        "ocr_status": "no_text_found"
                    }
                )]

            # Create documents based on chunking configuration
            if self.config.chunk_size:
                return self._chunk_text(text, metadata)
            else:
                # Return single document if no chunking specified
                return [Document(
                    content=text.strip(),
                    metadata=metadata
                )]

    def _extract_metadata(self, img: Image.Image, source_path: Path) -> dict:
        """Extract metadata from image"""
        metadata = {
            "source": str(source_path),
            "type": "image",
            "format": img.format,
            "mode": img.mode,
            "size": img.size,
        }

        # Extract EXIF data if available
        if hasattr(img, '_getexif') and img._getexif():
            exif = img._getexif()
            metadata["exif"] = {
                ExifTags.TAGS[k]: str(v)
                for k, v in exif.items()
                if k in ExifTags.TAGS
            }

        return metadata

    def _chunk_text(self, text: str, metadata: dict) -> List[Document]:
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
                for i in range(min(end + 100, text_length),
                               max(start, end - 100), -1):
                    if i < text_length and text[i] in '.!?' and \
                            (i + 1 == text_length or text[i + 1].isspace()):
                        end = i + 1
                        break

            # Create document from chunk
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunk_metadata = {
                    **metadata,
                    "chunk_start": start,
                    "chunk_end": end
                }
                chunks.append(Document(
                    content=chunk_text,
                    metadata=chunk_metadata
                ))

            # Move to next chunk position considering overlap
            if self.config.chunk_overlap:
                start = end - self.config.chunk_overlap
            else:
                start = end

        return chunks


# Example usage:
"""
# Initialize loader
loader = ImageLoader()
config = ImageLoaderConfig(
    chunk_size=1000,
    chunk_overlap=200,
    ocr_lang='eng',
    ocr_config='--psm 1'  # Automatic page segmentation with OSD
)
loader.initialize(config)

# Process image
documents = await loader.load('path/to/image.png')
"""
