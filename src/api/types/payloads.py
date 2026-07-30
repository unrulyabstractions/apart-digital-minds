"""What a task carries.

A payload is anything. The runtime never inspects it. These three are
conveniences for the common cases, and `preview` renders any of them for a log.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from .messages import ChatMessage

#: A payload is anything at all, including a tensor.
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

    `values` is any sequence of floats, so a list works and numpy works.
    """

    values: Sequence[float]
    name: str = ""
    meta: dict = field(default_factory=dict)


def preview(payload: Payload, width: int = 60) -> str:
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
