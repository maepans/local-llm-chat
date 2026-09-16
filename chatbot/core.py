"""Core chat session logic for the local LLM chatbot.

Talks directly to Ollama's REST API instead of a framework like LangChain,
so the parts that actually matter are visible and easy to read:
- how a conversation is represented as a list of role/content messages
- how the context window is kept bounded as a chat grows
- how a streaming chat response is parsed token by token
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterator

import requests

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2"
DEFAULT_SYSTEM_PROMPT = "You are a helpful, concise assistant."


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant"
    content: str

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


class ChatSession:
    """Holds conversation state and talks to a local Ollama model.

    Only the most recent `max_turns` user/assistant exchanges are sent to
    the model, so the context window stays bounded no matter how long the
    conversation runs.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        host: str = DEFAULT_HOST,
        max_turns: int = 12,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.max_turns = max_turns
        self.system_prompt = system_prompt
        self.history: list[Message] = []

    def add_user_message(self, content: str) -> None:
        self.history.append(Message("user", content))

    def _trimmed_history(self) -> list[Message]:
        max_messages = self.max_turns * 2
        if len(self.history) <= max_messages:
            return self.history
        return self.history[-max_messages:]

    def _payload_messages(self) -> list[dict]:
        messages = [Message("system", self.system_prompt), *self._trimmed_history()]
        return [m.to_dict() for m in messages]

    def stream_reply(self) -> Iterator[str]:
        """Stream the assistant's reply token by token and append it to history.

        Ollama's /api/chat endpoint returns newline-delimited JSON: one object
        per chunk, each holding a piece of the message, with `done: true` on
        the final chunk.
        """
        response = requests.post(
            f"{self.host}/api/chat",
            json={
                "model": self.model,
                "messages": self._payload_messages(),
                "stream": True,
            },
            stream=True,
            timeout=120,
        )
        response.raise_for_status()

        full_reply: list[str] = []
        for line in response.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            token = chunk.get("message", {}).get("content", "")
            if token:
                full_reply.append(token)
                yield token
            if chunk.get("done"):
                break

        self.history.append(Message("assistant", "".join(full_reply)))

    def reset(self) -> None:
        self.history.clear()


def list_models(host: str = DEFAULT_HOST) -> list[str]:
    """Return the names of models currently pulled in the local Ollama install."""
    response = requests.get(f"{host.rstrip('/')}/api/tags", timeout=10)
    response.raise_for_status()
    return [m["name"] for m in response.json().get("models", [])]
