from abc import abstractmethod
from pathlib import Path
from typing import List, Optional, Union

from pydantic import Field

from .base import Component
from .schemas import ComponentConfig, Document


class LoaderConfig(ComponentConfig):
    """Configuration for document loaders"""
    supported_formats: List[str] = Field(..., description="Supported file formats")
    chunk_size: Optional[int] = Field(None, description="Document chunk size")
    chunk_overlap: Optional[int] = Field(None, description="Chunk overlap size")


class DocumentLoader(Component[LoaderConfig]):
    """Interface for document loaders"""

    @abstractmethod
    async def load(self, source: Union[str, Path]) -> List[Document]:
        """Load documents from a source"""
        pass

    @abstractmethod
    def supports_format(self, format: str) -> bool:
        """Check if the loader supports a given format"""
        pass
