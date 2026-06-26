from __future__ import annotations

from typing import Any, Dict, List, Optional
import json
import os
import requests


class LLMClient:
    """
    Generic LLM client.

    Supports two common local API styles:

    1. OpenAI-compatible API:
       POST /v1/chat/completions

    2. Ollama native API:
       POST /api/chat

    You can switch using provider:
    - provider="openai_compatible"
    - provider="ollama"
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        model: str = "qwen2.5-coder:7b",
        provider: str = "openai_compatible",
        temperature: float = 0.1,
        max_tokens: int = 2048,
        timeout: int = 120,
        api_key: Optional[str] = None,
    ):
        self.base_url = base_url.rstrip("/")
        if self.base_url.endswith("/v1"):
            self.base_url = self.base_url[:-3]
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or ""

    def chat(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[Any] = None,
    ) -> str:
        if self.provider == "ollama":
            return self._chat_ollama(messages)

        return self._chat_openai_compatible(messages)

    def _chat_openai_compatible(self, messages: List[Dict[str, str]]) -> str:
        url = f"{self.base_url}/v1/chat/completions"

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )

        response.raise_for_status()
        data = response.json()

        try:
            return data["choices"][0]["message"]["content"]
        except Exception:
            return json.dumps(data, indent=2, ensure_ascii=False)

    def _chat_ollama(self, messages: List[Dict[str, str]]) -> str:
        url = f"{self.base_url}/api/chat"

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }

        response = requests.post(
            url,
            json=payload,
            timeout=self.timeout,
        )

        response.raise_for_status()
        data = response.json()

        try:
            return data["message"]["content"]
        except Exception:
            return json.dumps(data, indent=2, ensure_ascii=False)


def create_local_llm_client(
    provider: str = "openai_compatible",
    base_url: Optional[str] = None,
    model: str = "qwen2.5-coder:7b",
) -> LLMClient:
    """

    If you are using your ai-inference-engine and it exposes OpenAI-compatible API:
        provider="openai_compatible"
        base_url="http://localhost:8000"

    If you are using Ollama directly:
        provider="ollama"
        base_url="http://localhost:11434"
    """

    if base_url is None:
        if provider == "ollama":
            base_url = "http://localhost:11434"
        else:
            base_url = "http://localhost:8000"

    return LLMClient(
        base_url=base_url,
        model=model,
        provider=provider,
        temperature=0.1,
        max_tokens=512,
        timeout=60,
    )