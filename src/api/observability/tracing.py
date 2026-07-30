"""Who writes events, and what a module writes to."""

from __future__ import annotations

from typing import Any, ContextManager, Protocol

from ..types import Event
from .sinks import Sink


class Logger(Protocol):
    """What one module writes to. Every event it emits carries its name.

    Obtained from `Tracer.bind`, and handed to handlers as `ctx.log`.
    """

    @property
    def tick(self) -> int:
        """The tick currently being run."""

    def event(self, kind: str, **data: Any) -> Event:
        """Record a structured event of your own kind."""

    def note(self, summary: str, **data: Any) -> Event:
        """Record a free-form line."""

    def span(self, kind: str, **data: Any) -> ContextManager[dict]:
        """Time a block. Mutate the yielded dict to attach results.

            with ctx.log.span("retrieval") as span:
                span["hits"] = 3
        """

    def child(self, suffix: str) -> "Logger":
        """A logger for a subagent, named `parent/child`."""


class Tracer(Protocol):
    """Owns the sequence counter and fans events out to sinks."""

    run_id: str
    tick: int

    def emit(
        self,
        module: str,
        kind: str,
        data: dict | None = None,
        duration_s: float | None = None,
        task_id: str | None = None,
        cause: str | None = None,
    ) -> Event:
        """Record one event and hand it to every sink."""

    def bind(self, module: str) -> Logger:
        """A logger stamped with one module's name."""

    def add_sink(self, sink: Sink) -> "Tracer":
        """Attach a destination."""

    def close(self) -> None:
        """Close every sink."""
