"""The four types every provider speaks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant"]


@dataclass(slots=True)
class ChatMessage:
    """One turn. `meta` is yours: tag a message as a thought, a voice, an edit."""

    role: Role
    content: str
    meta: dict = field(default_factory=dict)

    def copy(self) -> "ChatMessage":
        return ChatMessage(self.role, self.content, dict(self.meta))

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content, "meta": dict(self.meta)}


def system(content: str, **meta: Any) -> ChatMessage:
    return ChatMessage("system", content, meta)


def user(content: str, **meta: Any) -> ChatMessage:
    return ChatMessage("user", content, meta)


def assistant(content: str, **meta: Any) -> ChatMessage:
    return ChatMessage("assistant", content, meta)


@dataclass(slots=True)
class Usage:
    """Token counts. Providers that do not report them leave zeros."""

    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(slots=True)
class Completion:
    """What every provider returns, whatever it is underneath."""

    text: str
    model: str
    usage: Usage = field(default_factory=Usage)
    latency_s: float = 0.0
    finish_reason: str = ""
    raw: Any = None

    def as_message(self, **meta: Any) -> ChatMessage:
        return ChatMessage("assistant", self.text, meta)


@dataclass(slots=True)
class GenOptions:
    """Sampling knobs that mean the same thing everywhere.

    Anything a single provider supports and the others do not goes in `extra`,
    which is passed through untouched.
    """

    temperature: float = 0.7
    max_tokens: int = 1024
    stop: list[str] = field(default_factory=list)
    seed: int | None = None
    extra: dict = field(default_factory=dict)
