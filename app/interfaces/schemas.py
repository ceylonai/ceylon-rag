from datetime import datetime
from typing import Any, Dict, List
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator, field_validator


class Document(BaseModel):
    """Base document class to standardize document handling"""
    content: str = Field(..., description="The document content")
    metadata: Dict[str, Any] = Field(..., description="Document metadata")
    doc_id: UUID = Field(default_factory=uuid4, description="Unique document identifier")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Document creation timestamp")

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            UUID: str,
            datetime: lambda v: v.isoformat()
        }


class EmbeddingVector(BaseModel):
    """Model for embedding vectors"""
    vector: List[float] = Field(..., description="Embedding vector")
    dimension: int = Field(..., description="Vector dimension")

    @field_validator('vector')
    def validate_dimension(cls, v, values):
        if 'dimension' in values and len(v) != values['dimension']:
            raise ValueError(f"Vector dimension {len(v)} does not match specified dimension {values['dimension']}")
        return v


class QueryResult(BaseModel):
    """Standardized query result format"""
    response: str = Field(..., description="Generated response")
    source_documents: List[Document] = Field(..., description="Source documents used for response generation")
    metadata: Dict[str, Any] = Field(..., description="Query result metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Result creation timestamp")


class ComponentConfig(BaseModel):
    """Base configuration for components"""
    name: str = Field(..., description="Component name")
    type: str = Field(..., description="Component type")
    config: Dict[str, Any] = Field(..., description="Component configuration")
