"""Ollama client. The only network FORGE ever talks to is localhost."""

from __future__ import annotations

import json
from typing import Callable, Dict, List, Optional

import requests


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    """Minimal chat client for a local Ollama server."""

    def __init__(self, host: str = "http://localhost:11434", model: str = "qwen2.5:3b",
                 temperature: float = 0.2, num_ctx: int = 8192, timeout: int = 600):
        self.host = host.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.timeout = timeout

    def check_ready(self) -> None:
        """Fail fast with a helpful message if Ollama or the model is missing."""
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=10)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise OllamaError(
                "Cannot reach Ollama at "
                f"{self.host}. Start it with `ollama serve` "
                "(install from https://ollama.com if needed)."
            ) from exc
        models = [m.get("name", "") for m in resp.json().get("models", [])]
        if not any(m == self.model or m.startswith(self.model.split(":")[0]) for m in models):
            raise OllamaError(
                f"Model {self.model!r} is not pulled yet. Run: ollama pull {self.model}"
            )

    def _payload(self, messages: List[Dict[str, str]], stream: bool) -> dict:
        return {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
            },
        }

    def chat(self, messages: List[Dict[str, str]]) -> str:
        """Send a chat conversation and return the assistant's reply text."""
        try:
            resp = requests.post(
                f"{self.host}/api/chat", json=self._payload(messages, False),
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise OllamaError(f"Ollama request failed: {exc}") from exc
        data = resp.json()
        return data.get("message", {}).get("content", "").strip()

    def chat_stream(self, messages: List[Dict[str, str]],
                    on_token: Optional[Callable[[str], None]] = None) -> str:
        """Stream a reply token by token. on_token fires per chunk as it arrives."""
        parts: List[str] = []
        try:
            with requests.post(
                f"{self.host}/api/chat", json=self._payload(messages, True),
                timeout=self.timeout, stream=True,
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    chunk = data.get("message", {}).get("content", "")
                    if chunk:
                        parts.append(chunk)
                        if on_token:
                            on_token(chunk)
                    if data.get("done"):
                        break
        except requests.RequestException as exc:
            raise OllamaError(f"Ollama request failed: {exc}") from exc
        return "".join(parts).strip()
