from abc import ABC, abstractmethod
from typing import List, AsyncGenerator, Optional


class LLM(ABC):
    @abstractmethod
    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate a response for the given prompt"""
        pass

    @abstractmethod
    async def streaming_generate(self, prompt: str, system_prompt: Optional[str] = None) -> AsyncGenerator[str, None]:
        """Generate a streaming response for the given prompt"""
        pass
