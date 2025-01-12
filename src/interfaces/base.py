from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from .schemas import Document, QueryResult, ComponentConfig

T = TypeVar('T', bound=ComponentConfig)


class Component(Generic[T], ABC):
    """Base component interface"""

    @abstractmethod
    def initialize(self, config: T) -> None:
        """Initialize the component with configuration"""
        pass

    @abstractmethod
    def validate_config(self, config: T) -> bool:
        """Validate component configuration"""
        pass