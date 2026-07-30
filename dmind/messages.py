"""What flows between modules.

A module never calls another module. It emits a `Task`, the bus routes it, and
the scheduler delivers it on the next tick. `Task.payload` is deliberately
untyped: it can be a string, a slice of context, a vector, or anything else you
invent. The dataclasses here are conveniences, not requirements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from .llm.types import ChatMessage

# A payload is anything. These aliases exist to make signatures readable.
Payload = Any


@dataclass(slots=True)
class Text:
    """Plain text. The most common payload."""

    text: str

    def __str__(self) -> str:
        return self.text


@dataclass(slots=True)
class Context:
    """A conversation, whole or partial.

    Use this when one module hands another module something to read or rewrite.
    `note` says which slice this is, so the receiver does not have to guess.
    """

    messages: list[ChatMessage]
    note: str = ""

    def copy(self) -> "Context":
        return Context([m.copy() for m in self.messages], self.note)


@dataclass(slots=True)
class Vector:
    """An arbitrary numeric payload: an activation, an embedding, a probe score.

    `values` is any sequence of floats, so a list works and numpy works. The
    runtime never inspects it.
    """

    values: Sequence[float]
    name: str = ""
    meta: dict = field(default_factory=dict)


@dataclass(slots=True)
class Task:
    """One unit of work sitting in one module's queue.

    `kind` is the routing key. A task of kind `"user_prompt"` is handled by the
    receiving module's `on_user_prompt` method.
    """

    id: str
    kind: str
    payload: Payload
    src: str
    dst: str
    t_created: int
    t_deliver: int
    cause: str | None = None

    def describe(self) -> str:
        """One line, for logs and the console sink."""
        return f"{self.src} -> {self.dst} [{self.kind}] {_preview(self.payload)}"


def _preview(payload: Payload, width: int = 60) -> str:
    """A short, safe rendering of any payload. Never raises."""
    try:
        if isinstance(payload, Text):
            body = payload.text
        elif isinstance(payload, Context):
            body = f"<{len(payload.messages)} messages: {payload.note or 'context'}>"
        elif isinstance(payload, Vector):
            body = f"<vector {payload.name or '?'} dim={len(payload.values)}>"
        elif isinstance(payload, str):
            body = payload
        else:
            body = repr(payload)
    except Exception:
        return "<unrenderable payload>"
    body = " ".join(body.split())
    return body if len(body) <= width else body[: width - 1] + "…"
