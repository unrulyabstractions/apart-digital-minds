"""The vocabulary that crosses every boundary.

These are data, not behaviour. They carry no policy and import nothing from
`src.dminds`, so an implementation can be swapped without changing what the
parts say to each other.

Read this file first. Every interface in `src/api` is written in these terms.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Sequence

# ---------------------------------------------------------------- messages

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


# ---------------------------------------------------------------- model io


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

    Anything one provider supports and the others do not goes in `extra`, which
    is passed through untouched.
    """

    temperature: float = 0.7
    max_tokens: int = 1024
    stop: list[str] = field(default_factory=list)
    seed: int | None = None
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------- payloads

#: A payload is anything. The runtime never inspects it.
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


# ---------------------------------------------------------------- tasks


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
        return f"{self.src} -> {self.dst} [{self.kind}] {preview(self.payload)}"


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


# ---------------------------------------------------------------- routing


@dataclass(frozen=True, slots=True)
class Route:
    """One entry in a routing table."""

    src: str
    kind: str
    dst: str
    as_kind: str

    def describe(self) -> str:
        renamed = f" as {self.as_kind}" if self.as_kind != self.kind else ""
        return f"{self.src} --{self.kind}--> {self.dst}{renamed}"


# ---------------------------------------------------------------- memory


@dataclass(slots=True)
class Episode:
    """One remembered thing."""

    text: str
    t: int = 0
    source: str = ""
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"text": self.text, "t": self.t, "source": self.source, "meta": self.meta}


# ---------------------------------------------------------------- tracing


@dataclass(slots=True)
class Event:
    """One line in the trace.

    `tick` is logical and is the only field logic may read. `wall` and
    `elapsed_s` are for you.
    """

    seq: int
    tick: int
    wall: str
    elapsed_s: float
    module: str
    kind: str
    data: dict = field(default_factory=dict)
    duration_s: float | None = None
    task_id: str | None = None
    cause: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    def line(self) -> str:
        """One readable line."""
        head = (
            f"t={self.tick:<3} {self.elapsed_s:7.3f}s "
            f"{self.module:<14} {self.kind:<14}"
        )
        detail = self.data.get("summary") or _compact(self.data)
        if self.duration_s is not None:
            detail = f"{detail}  ({self.duration_s * 1000:.0f}ms)"
        return f"{head} {detail}"


def _compact(data: dict, width: int = 90) -> str:
    if not data:
        return ""
    parts = []
    for key, value in data.items():
        text = value if isinstance(value, str) else repr(value)
        text = " ".join(str(text).split())
        if len(text) > 40:
            text = text[:39] + "…"
        parts.append(f"{key}={text}")
    joined = " ".join(parts)
    return joined if len(joined) <= width else joined[: width - 1] + "…"
