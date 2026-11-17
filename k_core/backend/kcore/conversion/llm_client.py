"""本地 LLM 用戶端：使用 OpenAI 相容介面（LM Studio、Ollama、vLLM 等皆提供）。

LM Studio 預設位址為 http://localhost:1234/v1：
- GET  {base_url}/models               列出可用模型
- POST {base_url}/chat/completions     對話；response_format 可指定 JSON Schema 強制結構化輸出
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Protocol

import httpx


class LLMError(RuntimeError):
    pass


class LLMClient(Protocol):
    def list_models(self) -> list[str]: ...

    def complete(self, system: str, user: str, json_schema: dict[str, Any] | None) -> str: ...


class OpenAICompatibleClient:
    def __init__(self, base_url: str, model: str, api_key: str | None = None, temperature: float = 0.0,
                 max_tokens: int | None = None, timeout: float = 600.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def list_models(self) -> list[str]:
        try:
            response = httpx.get(f"{self.base_url}/models", headers=self.headers, timeout=15)
            response.raise_for_status()
        except httpx.HTTPError as error:
            raise LLMError(f"cannot reach LLM endpoint {self.base_url}: {error}") from error
        return [item.get("id", "") for item in response.json().get("data", [])]

    def complete(self, system: str, user: str, json_schema: dict[str, Any] | None) -> str:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": self.temperature,
            "stream": False,
        }
        if self.max_tokens:
            body["max_tokens"] = self.max_tokens
        if json_schema is not None:
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": "kcore_conversion", "schema": json_schema}}
        try:
            response = httpx.post(f"{self.base_url}/chat/completions", json=body, headers=self.headers,
                                  timeout=self.timeout)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"] or ""
        except httpx.HTTPStatusError as error:
            raise LLMError(f"LLM endpoint returned {error.response.status_code}: {error.response.text[:500]}") from error
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as error:
            raise LLMError(f"LLM request failed: {error}") from error


def client_from_profile(profile: Any) -> OpenAICompatibleClient:
    api_key = os.environ.get(profile.api_key_env) if profile.api_key_env else None
    return OpenAICompatibleClient(profile.base_url, profile.model, api_key, profile.temperature,
                                  profile.max_tokens, profile.timeout_seconds)


_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json(text: str) -> Any:
    """從模型回應取出 JSON：移除推理模型的 <think> 區塊與 Markdown 程式碼框。"""
    cleaned = _THINK.sub("", text).strip()
    fenced = _FENCE.search(cleaned)
    if fenced:
        cleaned = fenced.group(1).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("response does not contain a JSON object") from None
        return json.loads(cleaned[start:end + 1])
