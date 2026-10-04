"""Minimal OpenAI-compatible HTTP client using only the standard library.

Configure with environment variables:

* ``OPENAI_API_KEY``   - API key (required to enable remote calls)
* ``OPENAI_BASE_URL``  - defaults to ``https://api.openai.com/v1``; point at Ollama, LM Studio, vLLM, Groq...
* ``RAGLENS_CHAT_MODEL`` / ``RAGLENS_EMBED_MODEL`` - model names
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field


class LLMError(RuntimeError):
    pass


@dataclass
class OpenAICompatible:
    api_key: str | None = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    base_url: str = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    chat_model: str = field(default_factory=lambda: os.getenv("RAGLENS_CHAT_MODEL", "gpt-4o-mini"))
    embed_model: str = field(default_factory=lambda: os.getenv("RAGLENS_EMBED_MODEL", "text-embedding-3-small"))
    timeout: float = 60.0

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _post(self, path: str, payload: dict) -> dict:
        if not self.api_key:
            raise LLMError("OPENAI_API_KEY is not set")
        req = urllib.request.Request(
            self.base_url.rstrip("/") + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:  # pragma: no cover - network path
            detail = exc.read().decode(errors="replace")[:300]
            raise LLMError(f"HTTP {exc.code} from {path}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:  # pragma: no cover - network path
            raise LLMError(f"Request to {path} failed: {exc}") from exc

    def chat(self, messages: list[dict], temperature: float = 0.1) -> str:
        data = self._post("/chat/completions", {"model": self.chat_model, "messages": messages,
                                                "temperature": temperature})
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, AttributeError) as exc:
            raise LLMError(f"Unexpected chat response: {str(data)[:200]}") from exc

    def embed(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            data = self._post("/embeddings", {"model": self.embed_model, "input": texts[i : i + batch_size]})
            rows = sorted(data["data"], key=lambda r: r["index"])
            vectors.extend(r["embedding"] for r in rows)
        return vectors
