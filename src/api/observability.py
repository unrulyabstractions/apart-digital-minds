"""Interfaces for logging and tracing.

Instrumentation is not optional in this runtime, so these contracts are what
every module and every model call is guaranteed to speak.

Implemented in `src.dminds.trace` by `RunTracer`, `ModuleLog`, and four sinks.
"""

from __future__ import annotations

from typing import Any, ContextManager, Protocol, runtime_checkable

from .types import Event

# Event kinds the runtime emits. Yours can be any string.
TICK_START = "tick.start"
TICK_END = "tick.end"
TASK_EMIT = "task.emit"
TASK_DELIVER = "task.deliver"
HANDLE_START = "handle.start"
HANDLE_END = "handle.end"
HANDLE_ERROR = "handle.error"
LLM_REQUEST = "llm.request"
LLM_RESPONSE = "llm.response"
LLM_ERROR = "llm.error"
MEMORY_WRITE = "memory.write"
NOTE = "note"

EVENT_KINDS = (
    TICK_START,
    TICK_END,
    TASK_EMIT,
    TASK_DELIVER,
    HANDLE_START,
    HANDLE_END,
    HANDLE_ERROR,
    LLM_REQUEST,
    LLM_RESPONSE,
    LLM_ERROR,
    MEMORY_WRITE,
    NOTE,
)


@runtime_checkable
class Sink(Protocol):
    """Somewhere events go. Implement these two methods and attach it.

    Shipped implementations write JSONL, split per module, print to a stream,
    or keep events in a list.
    """

    def write(self, event: Event) -> None: ...

    def close(self) -> None: ...


class Logger(Protocol):
    """What one module writes to. Every event it emits is tagged with its name.

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
        """Time a block. Mutate the yielded dict to attach results."""

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
