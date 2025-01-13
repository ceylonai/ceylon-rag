import httpx
from typing import AsyncGenerator, Optional, Dict, Any

from interfaces.llm import LLM


class AsyncOllamaLLM(LLM):
    def __init__(self, model_name: str = "llama3.1", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.client = httpx.AsyncClient()

    async def _post_request(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        response = await self.client.post(
            f"{self.base_url}/{endpoint}",
            json=payload
        )
        return response.json()

    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = await self._post_request("api/chat", {
            "model": self.model_name,
            "messages": messages,
            "stream": False
        })
        return response['message']['content']

    async def streaming_generate(self, prompt: str, system_prompt: Optional[str] = None) -> AsyncGenerator[str, None]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        async with self.client.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model_name,
                    "messages": messages,
                    "stream": True
                }
        ) as response:
            async for line in response.aiter_lines():
                if line.strip():
                    chunk = httpx.json.loads(line)
                    yield chunk['message']['content']

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
