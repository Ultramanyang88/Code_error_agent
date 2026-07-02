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
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Any:
        if self.provider == "ollama":
            return self._chat_ollama(messages, tools)

        return self._chat_openai_compatible(messages, tools)

    def _chat_openai_compatible(self, messages: List[Dict[str, str]], tools=None) -> str:
        url = f"{self.base_url}/v1/chat/completions"

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

        response = requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )

        if not response.ok:
            raise requests.exceptions.HTTPError(
                f"{response.status_code} error from {url}: {response.text[:1000]}",
                response=response,
            )
        data = response.json()

        try:
            messages = data["choices"][0]["message"]
        except Exception:
            return json.dumps(data, indent=2, ensure_ascii=False)

        if tools:
            return {"content": messages.get("content"), "tool_calls": messages.get("tool_calls")}
        return messages.get("content", "")

    def _chat_ollama(self, messages: List[Dict[str, str]], tools=None) -> str:
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
        
        if tools:
            payload["tools"] = tools

        response = requests.post(
            url,
            json=payload,
            timeout=self.timeout,
        )

        response.raise_for_status()
        data = response.json()
        messages = data.get("message", {})
        if tools:
            return {"content": messages.get("content"), "tool_calls": messages.get("tool_calls")}
        try:
            return messages["content"]
        except Exception:
            return json.dumps(data, indent=2, ensure_ascii=False)

    def chat_stream(self, messages: List[Dict[str, str]]):
        if self.provider == "ollama":
            yield from self._chat_stream_ollama(messages)
        else:
            yield from self._chat_stream_openai_compatible(messages)

    def _chat_stream_openai_compatible(self, messages):
        url = f"{self.base_url}/v1/chat/completions"
        payload = {"model": self.model, "messages": messages,
                   "temperature": self.temperature, "max_tokens": self.max_tokens, "stream": True}
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        with requests.post(url, json=payload, headers=headers, timeout=self.timeout, stream=True) as resp:
            if not resp.ok:
                raise requests.exceptions.HTTPError(
                    f"{resp.status_code} error from {url}: {resp.text[:1000]}",
                    response=resp,
                )
            for line in resp.iter_lines():
                if not line or not line.startswith(b"data: "):
                    continue
                chunk = line[len(b"data: "):]
                if chunk.strip() == b"[DONE]":
                    break
                try:
                    delta = json.loads(chunk)["choices"][0]["delta"].get("content")
                except Exception:
                    continue
                if delta:
                    yield delta

    def _chat_stream_ollama(self, messages):
        url = f"{self.base_url}/api/chat"
        payload = {"model": self.model, "messages": messages, "stream": True,
                   "options": {"temperature": self.temperature, "num_predict": self.max_tokens}}
        with requests.post(url, json=payload, timeout=self.timeout, stream=True) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                delta = data.get("message", {}).get("content")
                if delta:
                    yield delta
                if data.get("done"):
                    break

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