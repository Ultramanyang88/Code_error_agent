from __future__ import annotations

from typing import Any, Dict, List, Optional
import json
import os
import requests


class LLMClient:
    """
    Generic LLM client. Supports two API styles via `provider`:
    "openai_compatible" (POST /v1/chat/completions) and "ollama" (POST /api/chat).
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
        # LITELLM_MASTER_KEY takes priority: when base_url points at a
        # litellm gateway (see create_local_llm_client()), that's the
        # credential the gateway itself checks -- it's a single shared
        # secret for every model behind the gateway, including local
        # Ollama ones that need no OpenAI key at all. OPENAI_API_KEY stays
        # supported for calling api.openai.com directly, without a gateway
        # in front (e.g. the web UI's explicit "OpenAI" provider choice).
        self.api_key = (
            api_key
            or os.environ.get("LITELLM_MASTER_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or ""
        )

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
    model: Optional[str] = None,
) -> LLMClient:
    """
    base_url/model resolve from $LLM_BASE_URL/$LLM_MODEL when not given
    explicitly, so a litellm gateway (or any other OpenAI-compatible
    endpoint) can be configured once in .env and every caller -- CLI flags
    left at their defaults, the web UI's non-explicit paths -- picks it up
    automatically instead of needing --base-url/--model passed every time.
    Falls back to the pre-gateway defaults (a bare local inference server on
    :8000, or Ollama directly on :11434) when neither the argument nor the
    env var is set, so this still works with zero configuration.

    Auth: LLMClient reads $LITELLM_MASTER_KEY first, then $OPENAI_API_KEY
    (see LLMClient.__init__) -- nothing to pass here either, as long as
    .env has the right one set for whichever endpoint base_url points at.

    Examples:
        # litellm gateway proxying Ollama + OpenAI behind one endpoint --
        # set once in .env: LLM_BASE_URL=http://localhost:4000/v1,
        # LLM_MODEL=local-coder, LITELLM_MASTER_KEY=...
        create_local_llm_client()

        # Direct to Ollama, no gateway
        create_local_llm_client(provider="ollama", base_url="http://localhost:11434")
    """
    if base_url is None:
        base_url = os.environ.get("LLM_BASE_URL")
    if base_url is None:
        base_url = "http://localhost:11434" if provider == "ollama" else "http://localhost:8000"

    if model is None:
        model = os.environ.get("LLM_MODEL") or "qwen2.5-coder:7b"

    return LLMClient(
        base_url=base_url,
        model=model,
        provider=provider,
        temperature=0.1,
        max_tokens=512,
        timeout=60,
    )